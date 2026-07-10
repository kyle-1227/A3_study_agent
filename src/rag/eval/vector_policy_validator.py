"""Temporary vector retrieval validation for splitter policies."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import shutil
from typing import Any
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.rag.eval.chunk_metrics import metadata_text, sanitized_preview
from src.rag.eval.chunk_optimizer import (
    BASELINE_CANDIDATE_NAME,
    ChunkOptimizerConfig,
    ChunkPolicyCandidate,
    generate_candidates,
)
from src.rag.ids import normalize_for_hash, sha1_text
from src.rag.indexer import (
    COLLECTION_NAME,
    _add_documents_resilient,
    _content_id,
    _get_embedding,
    _index_batch_size,
    _index_max_retries,
    _l2_to_relevance,
    _resolve_persist_dir,
)
from src.rag.loader import load_documents

RETRIEVAL_BACKEND = "temporary_chroma_vector"
DEFAULT_OUTPUT_DIR = Path("reports")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_INDEX_ROOT = Path("reports/retrieval_vector_eval/indexes")
DEFAULT_POLICY_REPORT = Path("reports/splitter_policy_candidates.json")
DEFAULT_TOP_K = (1, 3, 5, 10)
DEFAULT_MAX_POLICIES = 4
ERROR_MESSAGE_CHARS = 160
TOO_SHORT_NOISE_CHARS = 80
ADVISORY_WARNING = (
    "Vector retrieval policy validation is advisory only; do not modify "
    "production index without rollout planning."
)
CONSERVATIVE_SOURCE_QUERY_TYPES = {"source_title"}
POLICY_NAME_PATTERN = re.compile(r"^(recursive|structure)_size(\d+)_overlap(\d+)$")


@dataclass(frozen=True)
class RetrievalPolicyValidationConfig:
    """Configuration for temporary vector retrieval policy validation."""

    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    index_root: Path = DEFAULT_INDEX_ROOT
    policy_report: Path | None = DEFAULT_POLICY_REPORT
    max_policies: int = DEFAULT_MAX_POLICIES
    max_queries: int | None = None
    top_k: tuple[int, ...] = DEFAULT_TOP_K
    subjects: tuple[str, ...] = ()
    trace_enabled: bool = False
    trace_output: Path | None = None
    reuse_index: bool = False
    force_rebuild_index: bool = False
    run_id: str | None = None
    project_root: Path | None = None
    too_short_chars: int = TOO_SHORT_NOISE_CHARS
    fail_fast: bool = True


@dataclass(frozen=True)
class RetrievalEvalCase:
    """A policy-independent evidence retrieval query."""

    query_id: str
    subject: str
    query: str
    query_type: str
    difficulty: str
    gold_evidence_id: str
    gold_source_relpath: str
    gold_section_id: str | None
    gold_section_path: str | None
    gold_anchor_hash: str | None
    gold_anchor_type: str | None
    baseline_gold_chunk_id: str
    gold_doc_id: str | None
    anchor_text: str = field(default="", repr=False, compare=False)

    def to_report(self) -> dict[str, Any]:
        """Return the JSON-safe representation without anchor text."""

        return {
            "query_id": self.query_id,
            "subject": self.subject,
            "query": self.query,
            "query_type": self.query_type,
            "difficulty": self.difficulty,
            "gold_evidence_id": self.gold_evidence_id,
            "gold_source_relpath": self.gold_source_relpath,
            "gold_section_id": self.gold_section_id,
            "gold_section_path": self.gold_section_path,
            "gold_anchor_hash": self.gold_anchor_hash,
            "gold_anchor_type": self.gold_anchor_type,
            "baseline_gold_chunk_id": self.baseline_gold_chunk_id,
            "gold_doc_id": self.gold_doc_id,
        }


@dataclass(frozen=True)
class DatasetBundle:
    """Generated eval cases plus in-memory-only anchors."""

    cases: list[RetrievalEvalCase]
    anchor_text_by_query_id: dict[str, str]
    warnings: list[str]


class RetrievalTraceWriter:
    """Trace writer for retrieval-vector validation events."""

    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: Path,
        trace_output: Path | None,
        run_id: str,
        project_root: Path,
    ) -> None:
        self.enabled = enabled
        self.project_root = project_root
        path = (
            trace_output
            if trace_output is not None
            else output_dir / f"retrieval_vector_policy_trace_{run_id}.jsonl"
        )
        self.path = path if path.is_absolute() else project_root / path
        self._handle = None
        self.write_failed = False

    @property
    def path_label(self) -> str | None:
        if not self.enabled:
            return None
        return _path_label(self.path, self.project_root)

    def clear(self) -> None:
        if not self.enabled:
            return
        self.close()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, event: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        record = {"event": event, "timestamp_utc": _utc_now(), **payload}
        try:
            if self._handle is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._handle = self.path.open("a", encoding="utf-8")
            self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._handle.flush()
        except OSError:
            self.write_failed = True
            self.enabled = False
            self.close()

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.close()
        finally:
            self._handle = None


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _project_root(path: Path | None = None) -> Path:
    return path.resolve() if path is not None else Path(__file__).resolve().parents[3]


def _resolve_project_path(path: Path, project_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _path_label(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.name


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _hash_payload(payload: Any) -> str:
    return sha1_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _sanitize_error(exc: Exception) -> str:
    text = str(exc)
    text = re.sub(r"[A-Za-z]:\\[^;,\n\r]+", "<local_path>", text)
    text = re.sub(
        r"(?i)\b(authorization|cookie|api[_-]?key|secret|bearer)\b\s*[:=]\s*\S+",
        r"\1=<redacted>",
        text,
    )
    return sanitized_preview(text, max_chars=ERROR_MESSAGE_CHARS)


def _common_trace_payload(
    *,
    run_id: str,
    sampled: bool,
    max_queries: int | None,
    policy_count: int,
    query_count: int,
    baseline_policy: str,
    global_best_policy: str | None,
    global_action: str | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "retrieval_backend": RETRIEVAL_BACKEND,
        "sampled": sampled,
        "max_queries": max_queries,
        "policy_count": policy_count,
        "query_count": query_count,
        "baseline_policy": baseline_policy,
        "global_best_policy": global_best_policy,
        "global_action": global_action,
    }


def _candidate_trace_payload(candidate: ChunkPolicyCandidate) -> dict[str, Any]:
    return {
        "policy_id": candidate.name,
        "candidate_id": candidate.name,
        "policy_name": candidate.name,
        "splitter_mode": candidate.splitter_mode,
        "chunk_size": candidate.chunk_size,
        "chunk_overlap": candidate.chunk_overlap,
    }


def _trace_error_payload(exc: Exception) -> dict[str, str]:
    return {
        "error_type": type(exc).__name__,
        "error_message": _sanitize_error(exc),
    }


def _safe_policy_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"Unsafe policy name: {name!r}")
    return name


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_index_root(index_root: Path, *, project_root: Path) -> Path:
    """Resolve and validate that the temporary index root is not production Chroma."""

    resolved = _resolve_project_path(index_root, project_root)
    chroma_dir = Path(_resolve_persist_dir()).resolve()
    if resolved == chroma_dir or _is_relative_to(resolved, chroma_dir):
        raise ValueError("index_root must not be chroma_store or inside chroma_store")
    return resolved


def _safe_remove_policy_index(policy_dir: Path, index_root: Path) -> None:
    resolved = policy_dir.resolve()
    root = index_root.resolve()
    if resolved == root or not _is_relative_to(resolved, root):
        raise ValueError("Refusing to remove path outside temporary index root")
    if resolved.exists():
        shutil.rmtree(resolved)


def _policy_index_dir(
    candidate: ChunkPolicyCandidate, index_root: Path, project_root: Path
) -> Path:
    root = validate_index_root(index_root, project_root=project_root)
    return root / _safe_policy_name(candidate.name)


def _policy_from_name(name: str, too_short_chars: int) -> ChunkPolicyCandidate | None:
    match = POLICY_NAME_PATTERN.match(name)
    if match is None:
        return None
    mode, chunk_size, chunk_overlap = match.groups()
    return ChunkPolicyCandidate(
        name=name,
        splitter_mode=mode,
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        too_short_chars=too_short_chars,
    )


def _candidate_from_report_item(
    item: dict[str, Any], *, too_short_chars: int
) -> tuple[ChunkPolicyCandidate | None, str | None, float]:
    name_value = item.get("policy_name") or item.get("name")
    if not isinstance(name_value, str) or not name_value:
        return None, "policy entry missing policy_name", 0.0

    mode = item.get("splitter_mode")
    chunk_size = item.get("chunk_size")
    chunk_overlap = item.get("chunk_overlap")
    if (
        isinstance(mode, str)
        and isinstance(chunk_size, int)
        and not isinstance(chunk_size, bool)
        and isinstance(chunk_overlap, int)
        and not isinstance(chunk_overlap, bool)
    ):
        candidate = ChunkPolicyCandidate(
            name=name_value,
            splitter_mode=mode,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            too_short_chars=too_short_chars,
        )
    else:
        candidate = _policy_from_name(name_value, too_short_chars)
        if candidate is None:
            return None, f"unable to parse policy {name_value}", 0.0

    score_value = item.get("score")
    score = float(score_value) if isinstance(score_value, int | float) else 0.0
    return candidate, None, score


def _baseline_candidate(too_short_chars: int) -> ChunkPolicyCandidate:
    candidate = _policy_from_name(BASELINE_CANDIDATE_NAME, too_short_chars)
    if candidate is None:
        raise ValueError("BASELINE_CANDIDATE_NAME is not parseable")
    return candidate


def select_policy_candidates(
    *,
    policy_report: Path | None,
    max_policies: int,
    data_dir: Path,
    project_root: Path,
    too_short_chars: int = TOO_SHORT_NOISE_CHARS,
) -> tuple[list[ChunkPolicyCandidate], list[str]]:
    """Select policies dynamically from Phase 4G reports or generated candidates."""

    if max_policies < 1:
        raise ValueError("max_policies must be >= 1")

    warnings: list[str] = []
    baseline = _baseline_candidate(too_short_chars)
    selected: dict[str, ChunkPolicyCandidate] = {baseline.name: baseline}
    scored: list[tuple[float, str, ChunkPolicyCandidate]] = []

    report_path = (
        _resolve_project_path(policy_report, project_root)
        if policy_report is not None
        else None
    )
    if report_path is not None and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        entries = report.get("global_ranking")
        if not isinstance(entries, list):
            entries = report.get("candidates")
        if not isinstance(entries, list):
            entries = []
            warnings.append("policy report has no usable global_ranking")
        for item in entries:
            if not isinstance(item, dict):
                continue
            status = item.get("status") or item.get("global_status")
            candidate, warning, score = _candidate_from_report_item(
                item, too_short_chars=too_short_chars
            )
            if warning is not None:
                warnings.append(warning)
                continue
            if candidate is None:
                continue
            if candidate.name == BASELINE_CANDIDATE_NAME:
                selected[candidate.name] = candidate
                continue
            if status == "pass":
                scored.append((score, candidate.name, candidate))
    else:
        generated = generate_candidates(
            ChunkOptimizerConfig(
                data_dir=data_dir,
                max_candidates=max_policies,
                too_short_chars=too_short_chars,
                project_root=project_root,
            )
        )
        return generated[:max_policies], warnings

    for _, _, candidate in sorted(scored, key=lambda item: (-item[0], item[1])):
        if len(selected) >= max_policies:
            break
        selected[candidate.name] = candidate
    return list(selected.values())[:max_policies], warnings


def _discover_subject_dirs(
    data_dir: Path, subjects: tuple[str, ...]
) -> tuple[list[tuple[str, Path]], list[dict[str, str]]]:
    skipped: list[dict[str, str]] = []
    if not data_dir.is_dir():
        return [], [{"subject": "", "reason": f"data directory not found: {data_dir}"}]

    if subjects:
        pairs: list[tuple[str, Path]] = []
        for subject in subjects:
            directory = data_dir / subject
            if not directory.is_dir():
                skipped.append(
                    {"subject": subject, "reason": "subject directory missing"}
                )
                continue
            pairs.append((subject, directory))
        return pairs, skipped

    pairs = []
    for directory in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        subject = directory.name
        if subject == "_needs_ocr":
            skipped.append(
                {"subject": subject, "reason": "quarantined OCR-needed directory"}
            )
            continue
        pairs.append((subject, directory))
    return pairs, skipped


def _load_documents_for_policy(
    *,
    candidate: ChunkPolicyCandidate,
    data_dir: Path,
    subjects: tuple[str, ...],
    project_root: Path,
    config: RetrievalPolicyValidationConfig,
    trace: RetrievalTraceWriter,
    trace_base: dict[str, Any],
) -> tuple[list[Document], list[dict[str, str]]]:
    resolved_data_dir = _resolve_project_path(data_dir, project_root)
    subject_dirs, skipped = _discover_subject_dirs(resolved_data_dir, subjects)
    for item in skipped:
        if item.get("reason") == "quarantined OCR-needed directory":
            continue
        exc = FileNotFoundError(item.get("reason") or "subject load failed")
        trace.write(
            "subject_load_failed",
            {
                **trace_base,
                **_candidate_trace_payload(candidate),
                "subject": item.get("subject", ""),
                **_trace_error_payload(exc),
            },
        )
        item["error_type"] = type(exc).__name__
        item["error_message"] = _sanitize_error(exc)
        if config.fail_fast:
            raise exc

    documents: list[Document] = []
    for subject, directory in subject_dirs:
        trace.write(
            "subject_load_started",
            {
                **trace_base,
                **_candidate_trace_payload(candidate),
                "subject": subject,
            },
        )
        try:
            loaded = load_documents(
                directory,
                subject=subject,
                doc_type="course_material",
                splitter_mode=candidate.splitter_mode,
                chunk_size=candidate.chunk_size,
                chunk_overlap=candidate.chunk_overlap,
            )
            documents.extend(loaded)
            trace.write(
                "subject_load_finished",
                {
                    **trace_base,
                    **_candidate_trace_payload(candidate),
                    "subject": subject,
                    "chunk_count": len(loaded),
                },
            )
        except Exception as exc:
            trace.write(
                "subject_load_failed",
                {
                    **trace_base,
                    **_candidate_trace_payload(candidate),
                    "subject": subject,
                    **_trace_error_payload(exc),
                },
            )
            if config.fail_fast:
                raise
            skipped.append(
                {
                    "subject": subject,
                    "reason": "subject load failed",
                    "error_type": type(exc).__name__,
                    "error_message": _sanitize_error(exc),
                }
            )
    return documents, skipped


def _load_error_count(skipped: list[dict[str, str]]) -> int:
    return sum(1 for item in skipped if item.get("error_type"))


def _source_relpath(doc: Document) -> str:
    return metadata_text(doc.metadata, "source_relpath") or metadata_text(
        doc.metadata, "source_file"
    )


def _section_path(doc: Document) -> str:
    return metadata_text(doc.metadata, "section_path")


def _query_text(value: str, *, max_chars: int = 120) -> str:
    return " ".join(value.split())[:max_chars]


def _source_title(source_relpath: str) -> str:
    stem = Path(source_relpath).stem
    return _query_text(re.sub(r"[_\-]+", " ", stem), max_chars=120)


def _definition_anchor(text: str) -> str:
    compact = " ".join(text.split())
    patterns = (r"\bmeans\b", r"\bis\b", r"\bdefined as\b", r"是", r"指")
    for pattern in patterns:
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if match is None:
            continue
        start = max(0, match.start() - 60)
        end = min(len(compact), match.end() + 60)
        phrase = compact[start:end].strip(" ,.;:，。；：")
        if 8 <= len(phrase) <= 140:
            return phrase
    return ""


def _keyword_anchor(text: str) -> str:
    tokens = re.findall(r"[\w\u4e00-\u9fff]{2,}", text)
    unique: list[str] = []
    for token in tokens:
        if token.casefold() not in {item.casefold() for item in unique}:
            unique.append(token)
        if len(unique) >= 6:
            break
    return " ".join(unique)


def _content_anchor(text: str, *, max_chars: int = 120) -> str:
    compact = " ".join(text.split())
    return compact[:max_chars] if len(compact) >= 8 else ""


def _case_from_document(
    doc: Document, *, baseline_gold_chunk_id: str | None = None
) -> RetrievalEvalCase | None:
    subject = metadata_text(doc.metadata, "subject")
    source_relpath = _source_relpath(doc)
    chunk_id = metadata_text(doc.metadata, "chunk_id")
    if not subject or not source_relpath or not chunk_id:
        return None

    section_id = metadata_text(doc.metadata, "section_id") or None
    section_path = _section_path(doc) or None
    section_title = metadata_text(doc.metadata, "section_title")
    anchor_text = ""
    anchor_type: str | None = None

    if section_title:
        query_type = "section_title"
        query = _query_text(section_title)
        anchor_text = _content_anchor(doc.page_content) or section_title
        anchor_type = "section_content_anchor"
        difficulty = "easy"
    elif section_path:
        query_type = "section_path"
        query = _query_text(section_path)
        anchor_text = _content_anchor(doc.page_content) or section_path
        anchor_type = "section_content_anchor"
        difficulty = "medium"
    else:
        definition = _definition_anchor(doc.page_content)
        if definition:
            query_type = "definition_like"
            anchor_text = definition
            anchor_type = "definition_like"
            query = _query_text(f"{_source_title(source_relpath)} definition")
            difficulty = "hard"
        else:
            keyword = _keyword_anchor(doc.page_content)
            if keyword:
                query_type = "keyword_anchor"
                anchor_text = keyword
                anchor_type = "keyword_anchor"
                query = _query_text(f"{_source_title(source_relpath)} key evidence")
                difficulty = "medium"
            else:
                query_type = "source_title"
                query = _source_title(source_relpath)
                difficulty = "easy"

    if not query:
        return None

    anchor_hash = sha1_text(normalize_for_hash(anchor_text)) if anchor_text else None
    evidence_key = "|".join(
        [
            source_relpath,
            section_id or "",
            section_path or "",
            anchor_hash or "",
            query_type,
        ]
    )
    evidence_id = f"evidence_{sha1_text(evidence_key)[:16]}"
    query_id = f"query_{sha1_text(f'{evidence_id}|{query_type}|{query}')[:16]}"
    return RetrievalEvalCase(
        query_id=query_id,
        subject=subject,
        query=query,
        query_type=query_type,
        difficulty=difficulty,
        gold_evidence_id=evidence_id,
        gold_source_relpath=source_relpath,
        gold_section_id=section_id,
        gold_section_path=section_path,
        gold_anchor_hash=anchor_hash,
        gold_anchor_type=anchor_type,
        baseline_gold_chunk_id=chunk_id
        if baseline_gold_chunk_id is None
        else baseline_gold_chunk_id,
        gold_doc_id=metadata_text(doc.metadata, "doc_id") or None,
        anchor_text=anchor_text,
    )


def _sample_cases(
    cases: list[RetrievalEvalCase], max_queries: int | None
) -> list[RetrievalEvalCase]:
    ordered = sorted(
        cases,
        key=lambda case: (
            case.subject,
            case.gold_source_relpath,
            case.query_type,
            case.query_id,
        ),
    )
    if max_queries is None:
        return ordered
    if max_queries < 1:
        raise ValueError("max_queries must be >= 1")
    grouped: dict[str, list[RetrievalEvalCase]] = defaultdict(list)
    for case in ordered:
        grouped[case.subject].append(case)
    output: list[RetrievalEvalCase] = []
    subjects = sorted(grouped)
    index = 0
    while len(output) < max_queries:
        added = False
        for subject in subjects:
            if index < len(grouped[subject]):
                output.append(grouped[subject][index])
                added = True
                if len(output) >= max_queries:
                    break
        if not added:
            break
        index += 1
    return output


def generate_retrieval_eval_dataset(
    documents: list[Document],
    *,
    max_queries: int | None = None,
    supplemental_documents: list[Document] | None = None,
) -> DatasetBundle:
    """Generate policy-independent evidence cases without serializing anchors."""

    by_evidence: dict[str, RetrievalEvalCase] = {}
    warnings: list[str] = []
    for doc in documents:
        case = _case_from_document(doc)
        if case is None:
            continue
        by_evidence.setdefault(case.gold_evidence_id, case)
    for doc in supplemental_documents or []:
        case = _case_from_document(doc, baseline_gold_chunk_id="")
        if case is None:
            continue
        by_evidence.setdefault(case.gold_evidence_id, case)
    cases = _sample_cases(list(by_evidence.values()), max_queries)
    if not cases:
        warnings.append("no retrieval eval cases generated")
    return DatasetBundle(
        cases=cases,
        anchor_text_by_query_id={
            case.query_id: case.anchor_text for case in cases if case.anchor_text
        },
        warnings=warnings,
    )


def _dataset_report(
    *,
    bundle: DatasetBundle,
    generated_at_utc: str,
    config: RetrievalPolicyValidationConfig,
    project_root: Path,
) -> dict[str, Any]:
    return {
        "generated_at_utc": generated_at_utc,
        "retrieval_backend": RETRIEVAL_BACKEND,
        "sampled": config.max_queries is not None,
        "max_queries": config.max_queries,
        "query_count": len(bundle.cases),
        "data_dir": _path_label(
            _resolve_project_path(config.data_dir, project_root), project_root
        ),
        "cases": [case.to_report() for case in bundle.cases],
        "warnings": bundle.warnings,
    }


def _duplicate_chunk_ids(documents: list[Document]) -> set[str]:
    counts = Counter(metadata_text(doc.metadata, "chunk_id") for doc in documents)
    return {chunk_id for chunk_id, count in counts.items() if chunk_id and count > 1}


def _is_noise_doc(
    doc: Document,
    *,
    case: RetrievalEvalCase,
    duplicate_chunk_ids: set[str],
    too_short_chars: int,
) -> bool:
    chunk_id = metadata_text(doc.metadata, "chunk_id")
    source_relpath = _source_relpath(doc)
    if not doc.page_content.strip():
        return True
    if not chunk_id or not source_relpath:
        return True
    if chunk_id in duplicate_chunk_ids:
        return True
    if len(doc.page_content) < too_short_chars:
        return True
    return bool(case.gold_source_relpath and source_relpath != case.gold_source_relpath)


def _anchor_hit(case: RetrievalEvalCase, doc: Document, anchor_text: str) -> bool:
    if not case.gold_anchor_hash or not anchor_text:
        return False
    if sha1_text(normalize_for_hash(anchor_text)) != case.gold_anchor_hash:
        return False
    return normalize_for_hash(anchor_text) in normalize_for_hash(doc.page_content)


def evidence_hit(case: RetrievalEvalCase, doc: Document, anchor_text: str = "") -> bool:
    """Return policy-independent evidence hit, never relying on chunk_id."""

    if (
        case.gold_section_id
        and metadata_text(doc.metadata, "section_id") == case.gold_section_id
    ):
        return True
    if case.gold_section_path and _section_path(doc) == case.gold_section_path:
        return True
    if _anchor_hit(case, doc, anchor_text):
        return True
    return (
        case.query_type in CONSERVATIVE_SOURCE_QUERY_TYPES
        and _source_relpath(doc) == case.gold_source_relpath
    )


def _section_hit(case: RetrievalEvalCase, doc: Document) -> bool:
    if case.gold_section_id:
        return metadata_text(doc.metadata, "section_id") == case.gold_section_id
    if case.gold_section_path:
        return _section_path(doc) == case.gold_section_path
    return False


def _retrieved_payload(
    *,
    rank: int,
    doc: Document,
    score: float,
    case: RetrievalEvalCase,
    is_noise: bool,
    anchor_text: str,
) -> dict[str, Any]:
    is_gold_section = _section_hit(case, doc)
    is_gold_source = _source_relpath(doc) == case.gold_source_relpath
    return {
        "rank": rank,
        "chunk_id": metadata_text(doc.metadata, "chunk_id"),
        "section_id": metadata_text(doc.metadata, "section_id"),
        "section_path": _section_path(doc),
        "source_relpath": _source_relpath(doc),
        "score": round(float(score), 4),
        "is_gold_chunk": metadata_text(doc.metadata, "chunk_id")
        == case.baseline_gold_chunk_id,
        "is_gold_evidence": evidence_hit(case, doc, anchor_text),
        "is_gold_section": is_gold_section,
        "is_gold_source": is_gold_source,
        "is_noise": is_noise,
    }


def _search_vectorstore(
    vectorstore: Any, case: RetrievalEvalCase, *, max_k: int
) -> list[tuple[Document, float]]:
    return vectorstore.similarity_search_with_score(
        case.query,
        k=max_k,
        filter={"subject": {"$eq": case.subject}},
    )


def _evaluate_policy_queries(
    *,
    policy_name: str,
    vectorstore: Any,
    cases: list[RetrievalEvalCase],
    anchor_text_by_query_id: dict[str, str],
    duplicate_chunk_ids: set[str],
    top_k: tuple[int, ...],
    too_short_chars: int,
    trace: RetrievalTraceWriter,
    trace_base: dict[str, Any],
    fail_fast: bool,
) -> tuple[list[dict[str, Any]], float]:
    max_k = max(top_k)
    records: list[dict[str, Any]] = []
    successful_queries = 0
    for case in cases:
        trace.write(
            "retrieval_query_started",
            {
                **trace_base,
                "query_id": case.query_id,
                "subject": case.subject,
                "query_type": case.query_type,
                "policy_name": policy_name,
                "source_relpath": case.gold_source_relpath,
            },
        )
        trace.write(
            "vector_query_started",
            {
                **trace_base,
                "query_id": case.query_id,
                "subject": case.subject,
                "query_type": case.query_type,
                "policy_name": policy_name,
                "source_relpath": case.gold_source_relpath,
            },
        )
        try:
            results = _search_vectorstore(vectorstore, case, max_k=max_k)
            successful_queries += 1
            query_error: dict[str, str] = {}
        except TypeError as exc:
            error_payload = {
                **trace_base,
                "query_id": case.query_id,
                "subject": case.subject,
                "query_type": case.query_type,
                "policy_name": policy_name,
                "source_relpath": case.gold_source_relpath,
                **_trace_error_payload(exc),
            }
            trace.write("vector_filter_unsupported", error_payload)
            trace.write("vector_query_failed", error_payload)
            if fail_fast:
                raise
            results = []
            query_error = {
                "error_type": type(exc).__name__,
                "error_message": _sanitize_error(exc),
            }
        except Exception as exc:
            error_payload = {
                **trace_base,
                "query_id": case.query_id,
                "subject": case.subject,
                "query_type": case.query_type,
                "policy_name": policy_name,
                "source_relpath": case.gold_source_relpath,
                **_trace_error_payload(exc),
            }
            trace.write("vector_query_failed", error_payload)
            if fail_fast:
                raise
            results = []
            query_error = {
                "error_type": type(exc).__name__,
                "error_message": _sanitize_error(exc),
            }
        anchor_text = anchor_text_by_query_id.get(case.query_id, "")
        retrieved: list[dict[str, Any]] = []
        evidence_rank: int | None = None
        section_rank: int | None = None
        source_rank: int | None = None
        baseline_chunk_rank: int | None = None
        for rank, (doc, score) in enumerate(results[:max_k], start=1):
            is_noise = _is_noise_doc(
                doc,
                case=case,
                duplicate_chunk_ids=duplicate_chunk_ids,
                too_short_chars=too_short_chars,
            )
            item = _retrieved_payload(
                rank=rank,
                doc=doc,
                score=score,
                case=case,
                is_noise=is_noise,
                anchor_text=anchor_text,
            )
            retrieved.append(item)
            if evidence_rank is None and item["is_gold_evidence"]:
                evidence_rank = rank
            if section_rank is None and item["is_gold_section"]:
                section_rank = rank
            if source_rank is None and item["is_gold_source"]:
                source_rank = rank
            if baseline_chunk_rank is None and item["is_gold_chunk"]:
                baseline_chunk_rank = rank
        record = {
            "query_id": case.query_id,
            "policy_name": policy_name,
            "subject": case.subject,
            "query_type": case.query_type,
            "source_relpath": case.gold_source_relpath,
            "top_k": max_k,
            "gold_evidence_id": case.gold_evidence_id,
            "gold_source_relpath": case.gold_source_relpath,
            "gold_section_id": case.gold_section_id,
            "gold_section_path": case.gold_section_path,
            "gold_anchor_hash": case.gold_anchor_hash,
            "hit_evidence": evidence_rank is not None,
            "hit_section": section_rank is not None,
            "hit_source": source_rank is not None,
            "evidence_rank": evidence_rank,
            "section_rank": section_rank,
            "source_rank": source_rank,
            "baseline_chunk_rank": baseline_chunk_rank,
            "retrieved": retrieved,
            **query_error,
        }
        records.append(record)
        trace.write(
            "retrieval_query_finished",
            {
                **trace_base,
                "query_id": case.query_id,
                "subject": case.subject,
                "query_type": case.query_type,
                "policy_name": policy_name,
                "source_relpath": case.gold_source_relpath,
                "top_k": max_k,
                "hit_evidence": evidence_rank is not None,
                "hit_section": section_rank is not None,
                "hit_source": source_rank is not None,
                "rank": evidence_rank,
            },
        )
        trace.write(
            "vector_query_finished",
            {
                **trace_base,
                "query_id": case.query_id,
                "subject": case.subject,
                "query_type": case.query_type,
                "policy_name": policy_name,
                "source_relpath": case.gold_source_relpath,
                "top_k": max_k,
                "hit_evidence": evidence_rank is not None,
                "hit_section": section_rank is not None,
                "hit_source": source_rank is not None,
                "rank": evidence_rank,
            },
        )
    success_rate = round(successful_queries / len(cases), 4) if cases else 0.0
    return records, success_rate


def _rank_recall(records: list[dict[str, Any]], rank_key: str, k: int) -> float:
    if not records:
        return 0.0
    hits = sum(
        1
        for record in records
        if isinstance(record.get(rank_key), int) and int(record[rank_key]) <= k
    )
    return round(hits / len(records), 4)


def _section_recall(records: list[dict[str, Any]], k: int) -> float | None:
    eligible = [
        record
        for record in records
        if record.get("gold_section_id") or record.get("gold_section_path")
    ]
    if not eligible:
        return None
    return _rank_recall(eligible, "section_rank", k)


def _noise_at(records: list[dict[str, Any]], k: int) -> float:
    noise = 0
    total = 0
    for record in records:
        for item in record.get("retrieved", [])[:k]:
            total += 1
            if item.get("is_noise"):
                noise += 1
    return round(noise / total, 4) if total else 0.0


def _mrr(records: list[dict[str, Any]], rank_key: str) -> float:
    if not records:
        return 0.0
    return round(
        sum(
            1 / int(record[rank_key])
            for record in records
            if isinstance(record.get(rank_key), int) and int(record[rank_key]) > 0
        )
        / len(records),
        4,
    )


def compute_retrieval_metrics(
    *,
    records: list[dict[str, Any]],
    top_k: tuple[int, ...],
    chunk_count: int,
    source_count: int,
    baseline_chunk_count: int,
    index_build_status: str,
    embedding_success_rate: float,
    load_error_count: int = 0,
    index_error_count: int = 0,
) -> dict[str, Any]:
    """Compute evidence-first retrieval metrics."""

    failed_queries = [
        {
            "query_id": str(record.get("query_id") or ""),
            "subject": str(record.get("subject") or ""),
            "query_type": str(record.get("query_type") or ""),
            "source_relpath": str(record.get("source_relpath") or ""),
            "error_type": str(record.get("error_type") or ""),
            "error_message": str(record.get("error_message") or ""),
        }
        for record in records
        if record.get("error_type")
    ]
    query_error_count = len(failed_queries)
    query_success_count = len(records) - query_error_count
    query_success_rate = (
        round(query_success_count / len(records), 4) if records else 0.0
    )
    metrics: dict[str, Any] = {
        "query_count": len(records),
        "query_error_count": query_error_count,
        "query_success_count": query_success_count,
        "query_success_rate": query_success_rate,
        "failed_query_ids": [item["query_id"] for item in failed_queries],
        "failed_queries": failed_queries,
        "load_error_count": load_error_count,
        "index_error_count": index_error_count,
        "chunk_count": chunk_count,
        "source_count": source_count,
        "chunk_count_ratio": round(chunk_count / baseline_chunk_count, 4)
        if baseline_chunk_count
        else None,
        "index_build_status": index_build_status,
        "embedding_success_rate": embedding_success_rate,
        "evidence_mrr": _mrr(records, "evidence_rank"),
        "mrr": _mrr(records, "evidence_rank"),
    }
    for k in top_k:
        evidence_recall = _rank_recall(records, "evidence_rank", k)
        metrics[f"evidence_recall_at_{k}"] = evidence_recall
        metrics[f"recall_at_{k}"] = evidence_recall
        metrics[f"source_recall_at_{k}"] = _rank_recall(records, "source_rank", k)
        metrics[f"section_recall_at_{k}"] = _section_recall(records, k)
        metrics[f"noise_at_{k}"] = _noise_at(records, k)
        metrics[f"baseline_chunk_recall_at_{k}"] = _rank_recall(
            records, "baseline_chunk_rank", k
        )
    return metrics


def _group_records_by_subject(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("subject") or "unknown")].append(record)
    return grouped


def _source_count(documents: list[Document]) -> int:
    return len({_source_relpath(doc) for doc in documents if _source_relpath(doc)})


def _policy_config_fingerprint(candidate: ChunkPolicyCandidate) -> str:
    return _hash_payload(
        {
            "policy_name": candidate.name,
            "splitter_mode": candidate.splitter_mode,
            "chunk_size": candidate.chunk_size,
            "chunk_overlap": candidate.chunk_overlap,
            "too_short_chars": candidate.too_short_chars,
        }
    )


def _embedding_fingerprint(embedding: Any) -> str:
    return _hash_payload(
        {
            "embedding_class": type(embedding).__name__,
            "model": getattr(embedding, "model", ""),
            "base_url": getattr(embedding, "base_url", ""),
            "document_input_type": getattr(embedding, "document_input_type", ""),
            "query_input_type": getattr(embedding, "query_input_type", ""),
            "timeout": getattr(embedding, "timeout", ""),
            "api_key_configured": bool(getattr(embedding, "api_key", None)),
        }
    )


def _source_manifest_fingerprint(documents: list[Document]) -> str:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for doc in documents:
        metadata = doc.metadata
        key = (
            metadata_text(metadata, "subject") or "unknown",
            _source_relpath(doc) or "unknown",
        )
        row = grouped.setdefault(
            key,
            {
                "subject": key[0],
                "source_relpath": key[1],
                "source_file_sha1": metadata_text(metadata, "source_file_sha1"),
                "source_file_size": metadata.get("source_file_size", 0),
                "chunk_count": 0,
            },
        )
        row["chunk_count"] += 1
    return _hash_payload(
        sorted(
            grouped.values(), key=lambda item: (item["subject"], item["source_relpath"])
        )
    )


def _expected_index_fingerprints(
    *, candidate: ChunkPolicyCandidate, documents: list[Document], embedding: Any
) -> dict[str, Any]:
    return {
        "policy_config_fingerprint": _policy_config_fingerprint(candidate),
        "embedding_fingerprint": _embedding_fingerprint(embedding),
        "source_manifest_fingerprint": _source_manifest_fingerprint(documents),
        "chunk_count": len(documents),
        "source_count": _source_count(documents),
    }


def _manifest_mismatch_reason(
    manifest: dict[str, Any], expected: dict[str, Any]
) -> str | None:
    if manifest.get("build_status") != "success":
        return f"build_status is {manifest.get('build_status')!r}"
    for key in (
        "policy_config_fingerprint",
        "embedding_fingerprint",
        "source_manifest_fingerprint",
        "chunk_count",
        "source_count",
    ):
        if manifest.get(key) != expected.get(key):
            return f"{key} mismatch"
    return None


def _subject_chunk_counts(documents: list[Document]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for doc in documents:
        counts[metadata_text(doc.metadata, "subject") or "unknown"] += 1
    return dict(counts)


def _subject_source_counts(documents: list[Document]) -> dict[str, int]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for doc in documents:
        grouped[metadata_text(doc.metadata, "subject") or "unknown"].add(
            _source_relpath(doc)
        )
    return {subject: len(sources) for subject, sources in grouped.items()}


def _metric_delta(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    keys = (
        "evidence_recall_at_5",
        "evidence_mrr",
        "mrr",
        "section_recall_at_5",
        "source_recall_at_5",
        "noise_at_5",
    )
    output: dict[str, Any] = {}
    for key in keys:
        candidate_value = candidate.get(key)
        baseline_value = baseline.get(key)
        delta_key = f"{key}_delta"
        if isinstance(candidate_value, int | float) and isinstance(
            baseline_value, int | float
        ):
            output[delta_key] = round(float(candidate_value) - float(baseline_value), 4)
        else:
            output[delta_key] = None
    return output


def _has_subject_regression(subject_entries: dict[str, Any], policy_name: str) -> bool:
    for subject_payload in subject_entries.values():
        entry = subject_payload["policies"].get(policy_name)
        if entry is None:
            continue
        delta = entry.get("metrics_delta_vs_baseline", {})
        if (
            delta.get("evidence_recall_at_5_delta") is not None
            and delta["evidence_recall_at_5_delta"] <= -0.05
        ):
            return True
        if (
            delta.get("evidence_mrr_delta") is not None
            and delta["evidence_mrr_delta"] <= -0.03
        ):
            return True
        if (
            delta.get("source_recall_at_5_delta") is not None
            and delta["source_recall_at_5_delta"] < 0
        ):
            return True
        if (
            delta.get("noise_at_5_delta") is not None
            and delta["noise_at_5_delta"] > 0.05
        ):
            return True
    return False


def _candidate_passes_global_gate(
    metrics: dict[str, Any],
    delta: dict[str, Any],
    *,
    section_recall_required: bool,
    has_subject_regression: bool,
) -> bool:
    if metrics.get("index_build_status") != "success":
        return False
    if metrics.get("embedding_success_rate") != 1.0:
        return False
    if int(metrics.get("query_count") or 0) < 50:
        return False
    if (delta.get("evidence_recall_at_5_delta") or 0.0) < 0.05:
        return False
    if (delta.get("evidence_mrr_delta") or 0.0) < 0:
        return False
    section_delta = delta.get("section_recall_at_5_delta")
    if section_recall_required and section_delta is not None and section_delta < 0.05:
        return False
    if (delta.get("source_recall_at_5_delta") or 0.0) < 0:
        return False
    if (delta.get("noise_at_5_delta") or 0.0) > 0.02:
        return False
    chunk_ratio = metrics.get("chunk_count_ratio")
    if chunk_ratio is None or float(chunk_ratio) > 1.20:
        return False
    return not has_subject_regression


def _candidate_passes_subject_gate(
    metrics: dict[str, Any], delta: dict[str, Any]
) -> bool:
    if int(metrics.get("query_count") or 0) < 20:
        return False
    if (delta.get("evidence_recall_at_5_delta") or 0.0) < 0.05:
        return False
    if (delta.get("evidence_mrr_delta") or 0.0) < 0:
        return False
    if (delta.get("source_recall_at_5_delta") or 0.0) < 0:
        return False
    if (delta.get("noise_at_5_delta") or 0.0) > 0.02:
        return False
    chunk_ratio = metrics.get("chunk_count_ratio")
    return chunk_ratio is not None and float(chunk_ratio) <= 1.20


def _confidence(metrics: dict[str, Any], delta: dict[str, Any]) -> str:
    if (
        int(metrics.get("query_count") or 0) >= 50
        and (delta.get("evidence_recall_at_5_delta") or 0.0) >= 0.08
        and (delta.get("evidence_mrr_delta") or 0.0) >= 0.05
        and (delta.get("noise_at_5_delta") or 0.0) <= 0
        and metrics.get("chunk_count_ratio") is not None
        and float(metrics["chunk_count_ratio"]) <= 1.10
        and (delta.get("source_recall_at_5_delta") or 0.0) >= 0
    ):
        return "high"
    if _candidate_passes_subject_gate(metrics, delta):
        return "medium"
    return "low"


def _rank_policy(item: dict[str, Any]) -> tuple[float, float, float, float, str]:
    metrics = item["global_metrics"]
    return (
        -float(metrics.get("evidence_recall_at_5") or 0.0),
        -float(metrics.get("evidence_mrr") or 0.0),
        float(metrics.get("noise_at_5") or 0.0),
        float(metrics.get("chunk_count_ratio") or 999.0),
        str(item["policy_name"]),
    )


def _metrics_are_clean(metrics: dict[str, Any]) -> bool:
    return (
        metrics.get("index_build_status") == "success"
        and int(metrics.get("query_error_count") or 0) == 0
        and int(metrics.get("load_error_count") or 0) == 0
        and int(metrics.get("index_error_count") or 0) == 0
        and float(metrics.get("query_success_rate") or 0.0) == 1.0
    )


def _policy_entry_status(metrics: dict[str, Any]) -> str:
    if metrics.get("index_build_status") != "success":
        return "failed"
    return "success" if _metrics_are_clean(metrics) else "partial_failed"


def _candidate_is_recommendable(entry: dict[str, Any]) -> bool:
    return entry.get("status") == "success" and _metrics_are_clean(
        entry.get("global_metrics", {})
    )


def _build_recommendation(
    *,
    policy_entries: list[dict[str, Any]],
    subject_report: dict[str, Any],
    baseline_policy: str,
) -> dict[str, Any]:
    baseline_entry = next(
        (entry for entry in policy_entries if entry["policy_name"] == baseline_policy),
        None,
    )
    if baseline_entry is None or not _candidate_is_recommendable(baseline_entry):
        return {
            "global_best_policy": baseline_policy,
            "global_action": "validation_failed",
            "global_confidence": "low",
            "global_reason": "baseline policy failed retrieval validation",
            "subject_policy_map": {},
        }

    successful = [
        entry for entry in policy_entries if _candidate_is_recommendable(entry)
    ]
    if not successful or int(baseline_entry["global_metrics"]["query_count"]) == 0:
        return {
            "global_best_policy": baseline_policy,
            "global_action": "validation_failed",
            "global_confidence": "low",
            "global_reason": "retrieval eval dataset is empty",
            "subject_policy_map": {},
        }

    best = sorted(successful, key=_rank_policy)[0]
    section_recall_required = (
        baseline_entry["global_metrics"].get("section_recall_at_5") is not None
    )
    has_regression = _has_subject_regression(subject_report, best["policy_name"])
    if best["policy_name"] == baseline_policy:
        action = "keep_current_default"
        reason = "baseline remains the best retrieval policy"
    elif _candidate_passes_global_gate(
        best["global_metrics"],
        best["metrics_delta_vs_baseline"],
        section_recall_required=section_recall_required,
        has_subject_regression=has_regression,
    ):
        action = "consider_candidate"
        reason = "candidate improves evidence retrieval without gate regressions"
    else:
        action = "needs_manual_review"
        reason = "best candidate does not satisfy retrieval validation gate"

    subject_policy_map: dict[str, Any] = {}
    for subject, payload in sorted(subject_report.items()):
        baseline_subject = payload["policies"].get(baseline_policy)
        subject_candidates = [
            {"policy_name": name, **entry}
            for name, entry in payload["policies"].items()
            if entry.get("status") == "success" and _metrics_are_clean(entry["metrics"])
        ]
        best_subject = (
            sorted(
                subject_candidates,
                key=lambda item: _rank_policy(
                    {
                        "policy_name": item["policy_name"],
                        "global_metrics": item["metrics"],
                    }
                ),
            )[0]
            if subject_candidates
            else None
        )
        if baseline_subject is None or baseline_subject.get("status") != "success":
            subject_action = "needs_manual_review"
            recommended_policy = None
            confidence = "low"
            subject_reason = "subject missing from baseline report"
            delta = {}
        elif best_subject is None:
            subject_action = "validation_failed"
            recommended_policy = None
            confidence = "low"
            subject_reason = "no valid subject policy metrics"
            delta = {}
        elif best_subject["policy_name"] == baseline_policy:
            subject_action = "keep_current_default"
            recommended_policy = None
            confidence = "low"
            subject_reason = "baseline remains best subject policy"
            delta = best_subject["metrics_delta_vs_baseline"]
        elif _candidate_passes_subject_gate(
            best_subject["metrics"], best_subject["metrics_delta_vs_baseline"]
        ):
            subject_action = "consider_candidate"
            recommended_policy = best_subject["policy_name"]
            confidence = _confidence(
                best_subject["metrics"], best_subject["metrics_delta_vs_baseline"]
            )
            subject_reason = "candidate improves subject evidence retrieval"
            delta = best_subject["metrics_delta_vs_baseline"]
        else:
            subject_action = "needs_manual_review"
            recommended_policy = None
            confidence = "low"
            subject_reason = "subject candidate does not satisfy retrieval gate"
            delta = best_subject["metrics_delta_vs_baseline"]
        subject_policy_map[subject] = {
            "recommended_policy": recommended_policy,
            "action": subject_action,
            "confidence": confidence,
            "reason": subject_reason,
            "metrics_delta_vs_baseline": delta,
            "do_not_auto_apply": True,
        }

    return {
        "global_best_policy": best["policy_name"],
        "global_action": action,
        "global_confidence": _confidence(
            best["global_metrics"], best["metrics_delta_vs_baseline"]
        ),
        "global_reason": reason,
        "subject_policy_map": subject_policy_map,
    }


def _report_paths(
    output_dir: Path, project_root: Path, max_queries: int | None
) -> dict[str, Path]:
    suffix = f"_max_queries{max_queries}" if max_queries is not None else ""
    resolved_output = _resolve_project_path(output_dir, project_root)
    return {
        "dataset": resolved_output / f"retrieval_vector_eval_dataset{suffix}.json",
        "candidates": resolved_output
        / f"retrieval_vector_policy_candidates{suffix}.json",
        "subject": resolved_output / f"retrieval_vector_subject_report{suffix}.json",
        "recommendation": resolved_output
        / f"retrieval_vector_recommendation{suffix}.json",
    }


def _index_manifest_payload(
    *,
    candidate: ChunkPolicyCandidate,
    documents: list[Document],
    persist_directory: Path,
    project_root: Path,
    build_status: str,
    embedding_provider: str,
    fingerprints: dict[str, Any],
    error_type: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "policy_name": candidate.name,
        "splitter_mode": candidate.splitter_mode,
        "chunk_size": candidate.chunk_size,
        "chunk_overlap": candidate.chunk_overlap,
        "chunk_count": len(documents),
        "source_count": _source_count(documents),
        "policy_config_fingerprint": fingerprints["policy_config_fingerprint"],
        "embedding_fingerprint": fingerprints["embedding_fingerprint"],
        "source_manifest_fingerprint": fingerprints["source_manifest_fingerprint"],
        "embedding_provider": embedding_provider,
        "persist_directory": _path_label(persist_directory, project_root),
        "created_at_utc": _utc_now(),
        "build_status": build_status,
        "error_type": error_type,
        "error_message": error_message,
    }


def _load_temp_index(policy_dir: Path):
    embedding = _get_embedding()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=str(policy_dir),
        relevance_score_fn=_l2_to_relevance,
    )


def _build_or_load_policy_index(
    *,
    candidate: ChunkPolicyCandidate,
    documents: list[Document],
    config: RetrievalPolicyValidationConfig,
    project_root: Path,
) -> tuple[Any, dict[str, Any], Path]:
    policy_dir = _policy_index_dir(candidate, config.index_root, project_root)
    manifest_path = policy_dir / "index_manifest.json"
    embedding = _get_embedding()
    label = getattr(embedding, "model", None)
    embedding_provider = (
        label if isinstance(label, str) and label else type(embedding).__name__
    )
    expected = _expected_index_fingerprints(
        candidate=candidate, documents=documents, embedding=embedding
    )
    if config.force_rebuild_index:
        _safe_remove_policy_index(
            policy_dir,
            validate_index_root(config.index_root, project_root=project_root),
        )
    if manifest_path.exists() and not config.force_rebuild_index:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatch = _manifest_mismatch_reason(manifest, expected)
        if mismatch is None:
            return _load_temp_index(policy_dir), manifest, manifest_path
        if config.reuse_index:
            invalid_manifest = _index_manifest_payload(
                candidate=candidate,
                documents=documents,
                persist_directory=policy_dir,
                project_root=project_root,
                build_status="invalid_index",
                embedding_provider=embedding_provider,
                fingerprints=expected,
                error_type="invalid_index",
                error_message=mismatch,
            )
            _write_json(invalid_manifest, manifest_path)
            raise RuntimeError(f"invalid_index: {mismatch}")
        _safe_remove_policy_index(
            policy_dir,
            validate_index_root(config.index_root, project_root=project_root),
        )
    elif config.reuse_index:
        policy_dir.mkdir(parents=True, exist_ok=True)
        invalid_manifest = _index_manifest_payload(
            candidate=candidate,
            documents=documents,
            persist_directory=policy_dir,
            project_root=project_root,
            build_status="invalid_index",
            embedding_provider=embedding_provider,
            fingerprints=expected,
            error_type="invalid_index",
            error_message=f"temporary index is missing for policy {candidate.name}",
        )
        _write_json(invalid_manifest, manifest_path)
        raise RuntimeError(
            f"invalid_index: temporary index is missing for policy {candidate.name}"
        )

    if policy_dir.exists() and not manifest_path.exists():
        _safe_remove_policy_index(
            policy_dir,
            validate_index_root(config.index_root, project_root=project_root),
        )

    try:
        policy_dir.mkdir(parents=True, exist_ok=True)
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding,
            persist_directory=str(policy_dir),
            relevance_score_fn=_l2_to_relevance,
        )
        _add_documents_resilient(
            vectorstore,
            documents,
            [_content_id(doc) for doc in documents],
            batch_size=_index_batch_size(),
            max_retries=_index_max_retries(),
        )
        manifest = _index_manifest_payload(
            candidate=candidate,
            documents=documents,
            persist_directory=policy_dir,
            project_root=project_root,
            build_status="success",
            embedding_provider=embedding_provider,
            fingerprints=expected,
        )
        _write_json(manifest, manifest_path)
        return vectorstore, manifest, manifest_path
    except Exception as exc:
        manifest = _index_manifest_payload(
            candidate=candidate,
            documents=documents,
            persist_directory=policy_dir,
            project_root=project_root,
            build_status="failed",
            embedding_provider=embedding_provider,
            fingerprints=expected,
            error_type=type(exc).__name__,
            error_message=_sanitize_error(exc),
        )
        _write_json(manifest, manifest_path)
        raise


def _empty_retrieval_metrics(
    *,
    top_k: tuple[int, ...],
    index_build_status: str,
    load_error_count: int = 0,
    index_error_count: int | None = None,
) -> dict[str, Any]:
    if index_error_count is None:
        resolved_index_error_count = 1 if index_build_status == "failed" else 0
    else:
        resolved_index_error_count = index_error_count
    metrics: dict[str, Any] = {
        "query_count": 0,
        "query_error_count": 0,
        "query_success_count": 0,
        "query_success_rate": 0.0,
        "failed_query_ids": [],
        "failed_queries": [],
        "load_error_count": load_error_count,
        "index_error_count": resolved_index_error_count,
        "chunk_count": 0,
        "source_count": 0,
        "chunk_count_ratio": None,
        "index_build_status": index_build_status,
        "embedding_success_rate": 0.0,
        "evidence_mrr": 0.0,
        "mrr": 0.0,
    }
    for k in top_k:
        metrics[f"evidence_recall_at_{k}"] = 0.0
        metrics[f"recall_at_{k}"] = 0.0
        metrics[f"source_recall_at_{k}"] = 0.0
        metrics[f"section_recall_at_{k}"] = None
        metrics[f"noise_at_{k}"] = 0.0
        metrics[f"baseline_chunk_recall_at_{k}"] = 0.0
    return metrics


def _failed_policy_entry(
    *,
    candidate: ChunkPolicyCandidate,
    manifest_path: Path | None,
    error_type: str,
    error_message: str,
    project_root: Path,
    top_k: tuple[int, ...] = DEFAULT_TOP_K,
    load_error_count: int = 0,
    skipped_subjects: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    metrics = _empty_retrieval_metrics(
        top_k=top_k,
        index_build_status="failed",
        load_error_count=load_error_count,
    )
    skipped_subjects = skipped_subjects or []
    return {
        "policy_name": candidate.name,
        "splitter_mode": candidate.splitter_mode,
        "chunk_size": candidate.chunk_size,
        "chunk_overlap": candidate.chunk_overlap,
        "index_manifest_path": _path_label(manifest_path, project_root)
        if manifest_path
        else None,
        "index_build_status": "failed",
        "status": "failed",
        "failure_reason": error_message,
        "load_error_count": load_error_count,
        "index_error_count": 1,
        "query_error_count": 0,
        "query_success_rate": 0.0,
        "skipped_subject_count": len(skipped_subjects),
        "skipped_subjects": skipped_subjects,
        "global_metrics": metrics,
        "metrics_delta_vs_baseline": {},
        "retrieval_results": [],
        "error_type": error_type,
        "error_message": error_message,
    }


def _build_subject_report(
    *,
    per_policy_subject_metrics: dict[str, dict[str, dict[str, Any]]],
    baseline_policy: str,
    top_k: tuple[int, ...],
) -> dict[str, Any]:
    subjects: set[str] = set()
    for metrics_by_subject in per_policy_subject_metrics.values():
        subjects.update(metrics_by_subject)
    policy_names = sorted(per_policy_subject_metrics)
    output: dict[str, Any] = {}
    for subject in sorted(subjects):
        baseline_metrics = per_policy_subject_metrics.get(baseline_policy, {}).get(
            subject
        )
        policies: dict[str, Any] = {}
        ranking: list[dict[str, Any]] = []
        for policy_name in policy_names:
            metrics_by_subject = per_policy_subject_metrics[policy_name]
            metrics = metrics_by_subject.get(subject)
            if metrics is None:
                reason = (
                    "subject missing from baseline report"
                    if policy_name == baseline_policy
                    else "subject missing from candidate report"
                )
                status = (
                    "partial_failed" if policy_name == baseline_policy else "failed"
                )
                policies[policy_name] = {
                    "metrics": _empty_retrieval_metrics(
                        top_k=top_k, index_build_status="failed"
                    ),
                    "metrics_delta_vs_baseline": {},
                    "status": status,
                    "reason": reason,
                }
                continue
            delta = (
                _metric_delta(metrics, baseline_metrics)
                if baseline_metrics is not None
                else {}
            )
            policies[policy_name] = {
                "metrics": metrics,
                "metrics_delta_vs_baseline": delta,
                "status": _policy_entry_status(metrics),
                "reason": "subject metrics available",
            }
            ranking.append(
                {
                    "policy_name": policy_name,
                    "metrics": metrics,
                    "metrics_delta_vs_baseline": delta,
                }
            )
        output[subject] = {
            "baseline_policy": baseline_policy,
            "policies": policies,
            "ranking": sorted(
                ranking,
                key=lambda item: _rank_policy(
                    {
                        "policy_name": item["policy_name"],
                        "global_metrics": item["metrics"],
                    }
                ),
            ),
        }
    return output


def validate_retrieval_policies(
    config: RetrievalPolicyValidationConfig,
) -> dict[str, Any]:
    """Validate splitter policies using temporary vector retrieval."""

    if config.max_policies < 1:
        raise ValueError("max_policies must be >= 1")
    if config.max_queries is not None and config.max_queries < 1:
        raise ValueError("max_queries must be >= 1")
    if any(value < 1 for value in config.top_k):
        raise ValueError("top_k values must be positive integers")
    if config.reuse_index and config.force_rebuild_index:
        raise ValueError("reuse_index and force_rebuild_index cannot both be set")

    project_root = _project_root(config.project_root)
    validate_index_root(config.index_root, project_root=project_root)
    run_id = config.run_id or uuid4().hex[:12]
    top_k = tuple(sorted(set(config.top_k)))
    paths = _report_paths(config.output_dir, project_root, config.max_queries)
    trace = RetrievalTraceWriter(
        enabled=config.trace_enabled,
        output_dir=_resolve_project_path(config.output_dir, project_root),
        trace_output=config.trace_output,
        run_id=run_id,
        project_root=project_root,
    )
    if trace.enabled:
        trace.clear()

    generated_at_utc = _utc_now()
    policies: list[ChunkPolicyCandidate] = []
    global_best_policy: str | None = None
    global_action: str | None = None
    try:
        policies, policy_warnings = select_policy_candidates(
            policy_report=config.policy_report,
            max_policies=config.max_policies,
            data_dir=config.data_dir,
            project_root=project_root,
            too_short_chars=config.too_short_chars,
        )
        trace_base = _common_trace_payload(
            run_id=run_id,
            sampled=config.max_queries is not None,
            max_queries=config.max_queries,
            policy_count=len(policies),
            query_count=0,
            baseline_policy=BASELINE_CANDIDATE_NAME,
            global_best_policy=None,
            global_action=None,
        )
        trace.write("retrieval_vector_validation_started", trace_base)
        trace.write("retrieval_policy_validation_started", trace_base)
        for candidate in policies:
            trace.write(
                "retrieval_policy_selected",
                {
                    **trace_base,
                    "policy_name": candidate.name,
                    "splitter_mode": candidate.splitter_mode,
                    "chunk_size": candidate.chunk_size,
                    "chunk_overlap": candidate.chunk_overlap,
                },
            )

        baseline_candidate = next(
            candidate
            for candidate in policies
            if candidate.name == BASELINE_CANDIDATE_NAME
        )
        baseline_documents, baseline_skipped = _load_documents_for_policy(
            candidate=baseline_candidate,
            data_dir=config.data_dir,
            subjects=config.subjects,
            project_root=project_root,
            config=config,
            trace=trace,
            trace_base=trace_base,
        )
        documents_by_policy: dict[str, list[Document]] = {
            BASELINE_CANDIDATE_NAME: baseline_documents
        }
        skipped_by_policy: dict[str, list[dict[str, str]]] = {
            BASELINE_CANDIDATE_NAME: baseline_skipped
        }
        all_skipped = list(baseline_skipped)
        for candidate in policies:
            if candidate.name == BASELINE_CANDIDATE_NAME:
                continue
            docs, skipped = _load_documents_for_policy(
                candidate=candidate,
                data_dir=config.data_dir,
                subjects=config.subjects,
                project_root=project_root,
                config=config,
                trace=trace,
                trace_base=trace_base,
            )
            documents_by_policy[candidate.name] = docs
            skipped_by_policy[candidate.name] = skipped
            all_skipped.extend(skipped)
        supplemental_documents = [
            doc
            for policy_name, docs in documents_by_policy.items()
            if policy_name != BASELINE_CANDIDATE_NAME
            for doc in docs
        ]
        trace.write("retrieval_eval_dataset_started", trace_base)
        dataset = generate_retrieval_eval_dataset(
            baseline_documents,
            max_queries=config.max_queries,
            supplemental_documents=supplemental_documents,
        )
        dataset_report = _dataset_report(
            bundle=dataset,
            generated_at_utc=generated_at_utc,
            config=config,
            project_root=project_root,
        )
        dataset_report["skipped"] = all_skipped
        _write_json(dataset_report, paths["dataset"])
        query_count = len(dataset.cases)
        trace_base = _common_trace_payload(
            run_id=run_id,
            sampled=config.max_queries is not None,
            max_queries=config.max_queries,
            policy_count=len(policies),
            query_count=query_count,
            baseline_policy=BASELINE_CANDIDATE_NAME,
            global_best_policy=None,
            global_action=None,
        )
        trace.write("retrieval_eval_dataset_finished", trace_base)

        if query_count == 0:
            candidates_report = {
                "generated_at_utc": generated_at_utc,
                "retrieval_backend": RETRIEVAL_BACKEND,
                "baseline_policy": BASELINE_CANDIDATE_NAME,
                "policy_count": len(policies),
                "query_count": 0,
                "top_k": list(top_k),
                "policies": [],
                "warnings": [*policy_warnings, *dataset.warnings],
            }
            subject_report = {
                "generated_at_utc": generated_at_utc,
                "retrieval_backend": RETRIEVAL_BACKEND,
                "baseline_policy": BASELINE_CANDIDATE_NAME,
                "subjects": {},
            }
            recommendation_report = {
                "generated_at_utc": generated_at_utc,
                "retrieval_backend": RETRIEVAL_BACKEND,
                "baseline_policy": BASELINE_CANDIDATE_NAME,
                "global_best_policy": BASELINE_CANDIDATE_NAME,
                "global_action": "validation_failed",
                "global_confidence": "low",
                "global_reason": "retrieval eval dataset is empty",
                "subject_policy_map": {},
                "do_not_auto_apply": True,
                "warnings": [ADVISORY_WARNING],
            }
            _write_json(candidates_report, paths["candidates"])
            _write_json(subject_report, paths["subject"])
            _write_json(recommendation_report, paths["recommendation"])
            path_payload = {
                "dataset_report_path": _path_label(paths["dataset"], project_root),
                "candidates_report_path": _path_label(
                    paths["candidates"], project_root
                ),
                "subject_report_path": _path_label(paths["subject"], project_root),
                "recommendation_report_path": _path_label(
                    paths["recommendation"], project_root
                ),
            }
            final_trace_base = _common_trace_payload(
                run_id=run_id,
                sampled=config.max_queries is not None,
                max_queries=config.max_queries,
                policy_count=len(policies),
                query_count=0,
                baseline_policy=BASELINE_CANDIDATE_NAME,
                global_best_policy=BASELINE_CANDIDATE_NAME,
                global_action="validation_failed",
            )
            trace.write(
                "policy_recommendation_skipped_due_to_failures", final_trace_base
            )
            trace.write("policy_recommendation_built", final_trace_base)
            trace.write(
                "retrieval_recommendation_written",
                {**final_trace_base, **path_payload},
            )
            trace.write(
                "retrieval_vector_validation_finished",
                {**final_trace_base, **path_payload},
            )
            trace.write(
                "retrieval_policy_validation_finished",
                {**final_trace_base, **path_payload},
            )
            trace.close()
            return {
                "dataset_report": dataset_report,
                "candidates_report": candidates_report,
                "subject_report": subject_report,
                "recommendation_report": recommendation_report,
                **path_payload,
                "trace_path": trace.path_label,
            }

        policy_entries: list[dict[str, Any]] = []
        per_policy_subject_metrics: dict[str, dict[str, dict[str, Any]]] = {}
        baseline_chunk_count = len(baseline_documents)
        baseline_subject_chunk_counts = _subject_chunk_counts(baseline_documents)

        for candidate in policies:
            candidate_load_errors = _load_error_count(
                skipped_by_policy.get(candidate.name, [])
            )
            candidate_skipped = skipped_by_policy.get(candidate.name, [])
            trace.write(
                "policy_candidate_started",
                {
                    **trace_base,
                    **_candidate_trace_payload(candidate),
                },
            )
            trace.write(
                "retrieval_policy_index_started",
                {
                    **trace_base,
                    **_candidate_trace_payload(candidate),
                },
            )
            trace.write(
                "temporary_vector_index_started",
                {
                    **trace_base,
                    **_candidate_trace_payload(candidate),
                    "load_error_count": candidate_load_errors,
                },
            )
            try:
                docs = documents_by_policy[candidate.name]
                vectorstore, manifest, manifest_path = _build_or_load_policy_index(
                    candidate=candidate,
                    documents=docs,
                    config=config,
                    project_root=project_root,
                )
                trace.write(
                    "retrieval_policy_index_finished",
                    {
                        **trace_base,
                        **_candidate_trace_payload(candidate),
                        "index_manifest_path": _path_label(manifest_path, project_root),
                        "chunk_count": len(docs),
                        "source_count": _source_count(docs),
                    },
                )
                trace.write(
                    "temporary_vector_index_finished",
                    {
                        **trace_base,
                        **_candidate_trace_payload(candidate),
                        "index_manifest_path": _path_label(manifest_path, project_root),
                        "chunk_count": len(docs),
                        "source_count": _source_count(docs),
                    },
                )
            except Exception as exc:
                manifest_path = (
                    _policy_index_dir(candidate, config.index_root, project_root)
                    / "index_manifest.json"
                )
                error_payload = {
                    **trace_base,
                    **_candidate_trace_payload(candidate),
                    **_trace_error_payload(exc),
                }
                trace.write(
                    "retrieval_policy_index_failed",
                    error_payload,
                )
                trace.write("temporary_vector_index_failed", error_payload)
                trace.write("policy_candidate_failed", error_payload)
                if config.fail_fast:
                    raise
                entry = _failed_policy_entry(
                    candidate=candidate,
                    manifest_path=manifest_path if manifest_path.exists() else None,
                    error_type=type(exc).__name__,
                    error_message=_sanitize_error(exc),
                    project_root=project_root,
                    top_k=top_k,
                    load_error_count=candidate_load_errors,
                    skipped_subjects=candidate_skipped,
                )
                policy_entries.append(entry)
                per_policy_subject_metrics[candidate.name] = {}
                continue

            duplicate_ids = _duplicate_chunk_ids(docs)
            try:
                query_records, success_rate = _evaluate_policy_queries(
                    policy_name=candidate.name,
                    vectorstore=vectorstore,
                    cases=dataset.cases,
                    anchor_text_by_query_id=dataset.anchor_text_by_query_id,
                    duplicate_chunk_ids=duplicate_ids,
                    top_k=top_k,
                    too_short_chars=config.too_short_chars,
                    trace=trace,
                    trace_base=trace_base,
                    fail_fast=config.fail_fast,
                )
            except Exception as exc:
                error_payload = {
                    **trace_base,
                    **_candidate_trace_payload(candidate),
                    **_trace_error_payload(exc),
                }
                trace.write("policy_candidate_failed", error_payload)
                if config.fail_fast:
                    raise
                query_records = [
                    {
                        "query_id": case.query_id,
                        "policy_name": candidate.name,
                        "subject": case.subject,
                        "query_type": case.query_type,
                        "source_relpath": case.gold_source_relpath,
                        "top_k": max(top_k),
                        "gold_evidence_id": case.gold_evidence_id,
                        "gold_source_relpath": case.gold_source_relpath,
                        "gold_section_id": case.gold_section_id,
                        "gold_section_path": case.gold_section_path,
                        "gold_anchor_hash": case.gold_anchor_hash,
                        "hit_evidence": False,
                        "hit_section": False,
                        "hit_source": False,
                        "evidence_rank": None,
                        "section_rank": None,
                        "source_rank": None,
                        "baseline_chunk_rank": None,
                        "retrieved": [],
                        "error_type": type(exc).__name__,
                        "error_message": _sanitize_error(exc),
                    }
                    for case in dataset.cases
                ]
                success_rate = 0.0
            metrics = compute_retrieval_metrics(
                records=query_records,
                top_k=top_k,
                chunk_count=len(docs),
                source_count=_source_count(docs),
                baseline_chunk_count=baseline_chunk_count,
                index_build_status=str(manifest.get("build_status", "success")),
                embedding_success_rate=success_rate,
                load_error_count=candidate_load_errors,
                index_error_count=0,
            )
            subject_records = _group_records_by_subject(query_records)
            subject_chunk_counts = _subject_chunk_counts(docs)
            subject_source_counts = _subject_source_counts(docs)
            subject_metrics: dict[str, dict[str, Any]] = {}
            for subject, records in sorted(subject_records.items()):
                subject_metrics[subject] = compute_retrieval_metrics(
                    records=records,
                    top_k=top_k,
                    chunk_count=subject_chunk_counts.get(subject, 0),
                    source_count=subject_source_counts.get(subject, 0),
                    baseline_chunk_count=baseline_subject_chunk_counts.get(subject, 0),
                    index_build_status=str(manifest.get("build_status", "success")),
                    embedding_success_rate=success_rate,
                    load_error_count=candidate_load_errors,
                    index_error_count=0,
                )
            per_policy_subject_metrics[candidate.name] = subject_metrics
            status = _policy_entry_status(metrics)
            failure_reason = (
                ""
                if status == "success"
                else "candidate has load, index, or query errors"
            )
            entry = {
                "policy_name": candidate.name,
                "splitter_mode": candidate.splitter_mode,
                "chunk_size": candidate.chunk_size,
                "chunk_overlap": candidate.chunk_overlap,
                "index_manifest_path": _path_label(manifest_path, project_root),
                "index_build_status": str(manifest.get("build_status", "success")),
                "status": status,
                "failure_reason": failure_reason,
                "load_error_count": candidate_load_errors,
                "index_error_count": 0,
                "query_error_count": metrics["query_error_count"],
                "query_success_rate": metrics["query_success_rate"],
                "skipped_subject_count": len(candidate_skipped),
                "skipped_subjects": candidate_skipped,
                "global_metrics": metrics,
                "metrics_delta_vs_baseline": {},
                "retrieval_results": query_records,
                "error_type": "" if status == "success" else "ValidationFailure",
                "error_message": failure_reason,
            }
            policy_entries.append(entry)
            trace.write(
                "retrieval_policy_scored",
                {
                    **trace_base,
                    "policy_name": candidate.name,
                    "evidence_recall_at_5": metrics.get("evidence_recall_at_5"),
                    "evidence_mrr": metrics.get("evidence_mrr"),
                    "source_recall_at_5": metrics.get("source_recall_at_5"),
                    "noise_at_5": metrics.get("noise_at_5"),
                },
            )
            if status != "success":
                trace.write(
                    "policy_candidate_failed",
                    {
                        **trace_base,
                        **_candidate_trace_payload(candidate),
                        "error_type": "ValidationFailure",
                        "error_message": failure_reason,
                        "status": status,
                        "query_error_count": metrics["query_error_count"],
                        "load_error_count": candidate_load_errors,
                        "index_error_count": 0,
                    },
                )
            trace.write(
                "policy_candidate_finished",
                {
                    **trace_base,
                    **_candidate_trace_payload(candidate),
                    "status": status,
                    "query_success_rate": metrics["query_success_rate"],
                },
            )

        baseline_entry = next(
            (
                entry
                for entry in policy_entries
                if entry["policy_name"] == BASELINE_CANDIDATE_NAME
            ),
            None,
        )
        baseline_metrics = (
            baseline_entry["global_metrics"] if baseline_entry is not None else {}
        )
        for entry in policy_entries:
            entry["metrics_delta_vs_baseline"] = (
                _metric_delta(entry["global_metrics"], baseline_metrics)
                if baseline_metrics
                else {}
            )

        subject_entries = _build_subject_report(
            per_policy_subject_metrics=per_policy_subject_metrics,
            baseline_policy=BASELINE_CANDIDATE_NAME,
            top_k=top_k,
        )
        for subject, payload in subject_entries.items():
            for policy_name, item in payload["policies"].items():
                trace.write(
                    "retrieval_subject_scored",
                    {
                        **trace_base,
                        "subject": subject,
                        "policy_name": policy_name,
                        "evidence_recall_at_5": item["metrics"].get(
                            "evidence_recall_at_5"
                        ),
                        "evidence_mrr": item["metrics"].get("evidence_mrr"),
                        "source_recall_at_5": item["metrics"].get("source_recall_at_5"),
                        "noise_at_5": item["metrics"].get("noise_at_5"),
                    },
                )

        recommendation_core = _build_recommendation(
            policy_entries=policy_entries,
            subject_report=subject_entries,
            baseline_policy=BASELINE_CANDIDATE_NAME,
        )
        global_best_policy = recommendation_core["global_best_policy"]
        global_action = recommendation_core["global_action"]
        recommendation_trace = {
            **trace_base,
            "global_best_policy": global_best_policy,
            "global_action": global_action,
        }
        if any(not _candidate_is_recommendable(entry) for entry in policy_entries):
            trace.write(
                "policy_recommendation_skipped_due_to_failures",
                recommendation_trace,
            )
        trace.write("policy_recommendation_built", recommendation_trace)
        candidates_report = {
            "generated_at_utc": generated_at_utc,
            "retrieval_backend": RETRIEVAL_BACKEND,
            "baseline_policy": BASELINE_CANDIDATE_NAME,
            "policy_count": len(policies),
            "query_count": query_count,
            "top_k": list(top_k),
            "policies": sorted(policy_entries, key=_rank_policy),
            "warnings": policy_warnings,
        }
        subject_report = {
            "generated_at_utc": generated_at_utc,
            "retrieval_backend": RETRIEVAL_BACKEND,
            "baseline_policy": BASELINE_CANDIDATE_NAME,
            "subject_count": len(subject_entries),
            "subjects": subject_entries,
        }
        recommendation_report = {
            "generated_at_utc": generated_at_utc,
            "retrieval_backend": RETRIEVAL_BACKEND,
            "baseline_policy": BASELINE_CANDIDATE_NAME,
            **recommendation_core,
            "do_not_auto_apply": True,
            "warnings": [ADVISORY_WARNING],
        }
        _write_json(candidates_report, paths["candidates"])
        _write_json(subject_report, paths["subject"])
        _write_json(recommendation_report, paths["recommendation"])

        final_trace_base = _common_trace_payload(
            run_id=run_id,
            sampled=config.max_queries is not None,
            max_queries=config.max_queries,
            policy_count=len(policies),
            query_count=query_count,
            baseline_policy=BASELINE_CANDIDATE_NAME,
            global_best_policy=global_best_policy,
            global_action=global_action,
        )
        path_payload = {
            "dataset_report_path": _path_label(paths["dataset"], project_root),
            "candidates_report_path": _path_label(paths["candidates"], project_root),
            "subject_report_path": _path_label(paths["subject"], project_root),
            "recommendation_report_path": _path_label(
                paths["recommendation"], project_root
            ),
        }
        trace.write(
            "retrieval_recommendation_written",
            {**final_trace_base, **path_payload},
        )
        trace.write(
            "retrieval_vector_validation_finished",
            {**final_trace_base, **path_payload},
        )
        trace.write(
            "retrieval_policy_validation_finished",
            {**final_trace_base, **path_payload},
        )
        trace.close()
        return {
            "dataset_report": dataset_report,
            "candidates_report": candidates_report,
            "subject_report": subject_report,
            "recommendation_report": recommendation_report,
            **path_payload,
            "trace_path": trace.path_label,
        }
    except Exception as exc:
        trace.write(
            "retrieval_vector_validation_failed",
            {
                **_common_trace_payload(
                    run_id=run_id,
                    sampled=config.max_queries is not None,
                    max_queries=config.max_queries,
                    policy_count=len(policies),
                    query_count=0,
                    baseline_policy=BASELINE_CANDIDATE_NAME,
                    global_best_policy=global_best_policy,
                    global_action=global_action or "validation_failed",
                ),
                "error_type": type(exc).__name__,
                "error_message": _sanitize_error(exc),
            },
        )
        trace.write(
            "retrieval_policy_validation_failed",
            {
                **_common_trace_payload(
                    run_id=run_id,
                    sampled=config.max_queries is not None,
                    max_queries=config.max_queries,
                    policy_count=len(policies),
                    query_count=0,
                    baseline_policy=BASELINE_CANDIDATE_NAME,
                    global_best_policy=global_best_policy,
                    global_action=global_action or "validation_failed",
                ),
                "error_type": type(exc).__name__,
                "error_message": _sanitize_error(exc),
            },
        )
        trace.close()
        raise
