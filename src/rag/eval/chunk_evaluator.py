"""Read-only chunking evaluation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document

from src.rag.chunking.splitter_factory import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    VALID_SPLITTER_MODES,
    chunk_policy_version_for_mode,
)
from src.rag.eval.chunk_metrics import (
    ChunkMetricsConfig,
    duplicate_flags,
    evaluate_documents,
    metadata_int,
    metadata_text,
    sanitized_preview,
)
from src.rag.loader import load_documents

COURSE_DOC_TYPE = "course_material"
DEFAULT_OUTPUT_DIR = Path("reports")
TRACE_PREVIEW_CHARS = 120
CHUNK_COUNT_REVIEW_RATIO = 1.2
SHORT_RATIO_REVIEW_DELTA = 0.02
DUPLICATE_RATIO_REVIEW_DELTA = 0.01


@dataclass(frozen=True)
class ChunkEvaluationConfig:
    """Configuration for a single chunking evaluation run."""

    mode: str
    data_dir: Path
    output_dir: Path = DEFAULT_OUTPUT_DIR
    output_path: Path | None = None
    subjects: tuple[str, ...] = ()
    too_short_chars: int = 80
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    sample_limit: int | None = None
    trace_enabled: bool = True
    trace_output: Path | None = None
    run_id: str | None = None
    project_root: Path | None = None


class TraceWriter:
    """Small JSONL trace writer that only records bounded diagnostic payloads."""

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
        self.path = (
            trace_output
            if trace_output is not None
            else output_dir / f"chunk_eval_trace_{run_id}.jsonl"
        )
        self.path = self.path if self.path.is_absolute() else project_root / self.path

    @property
    def path_label(self) -> str | None:
        if not self.enabled:
            return None
        return _path_label(self.path, self.project_root)

    def write(self, event: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "event": event,
            "timestamp_utc": _utc_now(),
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def clear(self) -> None:
        """Start a fresh explicit trace file."""

        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _project_root(path: Path | None = None) -> Path:
    return path.resolve() if path is not None else Path(__file__).resolve().parents[3]


def _path_label(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.name


def _trace_relpath(value: str) -> str:
    if not value:
        return ""
    path = Path(value)
    return path.name if path.is_absolute() else value.replace("\\", "/")


def _error_message(exc: Exception) -> str:
    return sanitized_preview(str(exc), max_chars=160)


def _write_run_failed(
    trace: TraceWriter,
    *,
    run_id: str,
    exc: Exception,
    mode: str | None = None,
    baseline_mode: str | None = None,
    candidate_mode: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "run_id": run_id,
        "error_type": type(exc).__name__,
        "error_message": _error_message(exc),
    }
    if mode is not None:
        payload["mode"] = mode
    if baseline_mode is not None:
        payload["baseline_mode"] = baseline_mode
    if candidate_mode is not None:
        payload["candidate_mode"] = candidate_mode
    trace.write("run_failed", payload)


def _output_path(config: ChunkEvaluationConfig) -> Path:
    output_dir = config.output_dir
    if config.output_path is not None:
        path = config.output_path
    else:
        path = output_dir / f"chunk_eval_{config.mode}.json"
    project_root = _project_root(config.project_root)
    return path if path.is_absolute() else project_root / path


def _compare_output_path(output_dir: Path, project_root: Path) -> Path:
    path = output_dir / "chunk_eval_compare.json"
    return path if path.is_absolute() else project_root / path


def _validate_mode(mode: str) -> None:
    if mode not in VALID_SPLITTER_MODES:
        expected = ", ".join(VALID_SPLITTER_MODES)
        raise ValueError(f"Invalid mode {mode!r}. Expected one of: {expected}.")


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
                {
                    "subject": subject,
                    "reason": "quarantined OCR-needed directory",
                }
            )
            continue
        if not any(directory.iterdir()):
            skipped.append({"subject": subject, "reason": "empty directory"})
            continue
        pairs.append((subject, directory))
    return pairs, skipped


def _stable_sample(
    documents: list[Document], sample_limit: int | None
) -> tuple[list[Document], bool]:
    if sample_limit is None:
        return documents, False
    if sample_limit < 0:
        raise ValueError("sample_limit must be >= 0")
    ordered = sorted(
        documents,
        key=lambda doc: (
            metadata_text(doc.metadata, "subject"),
            metadata_text(doc.metadata, "source_relpath"),
            metadata_int(doc.metadata, "chunk_index"),
            metadata_text(doc.metadata, "chunk_id"),
        ),
    )
    return ordered[:sample_limit], True


def _load_documents_for_mode(
    config: ChunkEvaluationConfig,
    trace: TraceWriter,
) -> tuple[list[Document], list[dict[str, str]]]:
    project_root = _project_root(config.project_root)
    data_dir = (
        config.data_dir
        if config.data_dir.is_absolute()
        else project_root / config.data_dir
    )
    subject_dirs, skipped = _discover_subject_dirs(data_dir, config.subjects)
    documents: list[Document] = []
    for subject, directory in subject_dirs:
        try:
            docs = load_documents(
                directory,
                subject=subject,
                doc_type=COURSE_DOC_TYPE,
                splitter_mode=config.mode,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
        except Exception as exc:
            skipped.append(
                {"subject": subject, "reason": f"{type(exc).__name__}: {exc}"}
            )
            continue
        trace.write(
            "subject_loaded",
            {
                "mode": config.mode,
                "subject": subject,
                "chunk_count": len(docs),
                "source_count": len(
                    {
                        metadata_text(doc.metadata, "source_relpath")
                        or metadata_text(doc.metadata, "source_file")
                        for doc in docs
                    }
                ),
            },
        )
        documents.extend(docs)
    return documents, skipped


def _trace_chunk_events(
    documents: list[Document],
    trace: TraceWriter,
    *,
    mode: str,
    too_short_chars: int,
) -> None:
    duplicates = duplicate_flags(documents)
    for doc, is_duplicate in zip(documents, duplicates, strict=True):
        trace.write(
            "chunk_evaluated",
            {
                "mode": mode,
                "doc_id": metadata_text(doc.metadata, "doc_id"),
                "chunk_id": metadata_text(doc.metadata, "chunk_id"),
                "chunk_index": metadata_int(doc.metadata, "chunk_index"),
                "content_sha1": metadata_text(doc.metadata, "content_sha1"),
                "chunk_chars": len(doc.page_content),
                "subject": metadata_text(doc.metadata, "subject"),
                "source_relpath": _trace_relpath(
                    metadata_text(doc.metadata, "source_relpath")
                ),
                "section_id": metadata_text(doc.metadata, "section_id"),
                "section_title": sanitized_preview(
                    metadata_text(doc.metadata, "section_title"),
                    max_chars=TRACE_PREVIEW_CHARS,
                ),
                "section_chunk_index": metadata_int(
                    doc.metadata, "section_chunk_index"
                ),
                "is_short": len(doc.page_content) < too_short_chars,
                "is_empty": not doc.page_content.strip(),
                "is_duplicate": is_duplicate,
                "preview": sanitized_preview(
                    doc.page_content, max_chars=TRACE_PREVIEW_CHARS
                ),
            },
        )


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _evaluate_mode(
    config: ChunkEvaluationConfig,
    *,
    reset_trace: bool,
) -> dict[str, Any]:
    """Evaluate one splitter mode and write its JSON report."""

    project_root = _project_root(config.project_root)
    run_id = config.run_id or uuid4().hex[:12]
    output_path = _output_path(config)
    trace = TraceWriter(
        enabled=config.trace_enabled,
        output_dir=config.output_dir,
        trace_output=config.trace_output,
        run_id=run_id,
        project_root=project_root,
    )
    if reset_trace and config.trace_output is not None:
        trace.clear()

    try:
        _validate_mode(config.mode)
        trace.write(
            "run_started",
            {
                "run_id": run_id,
                "mode": config.mode,
                "sampled": config.sample_limit is not None,
                "sample_limit": config.sample_limit,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
            },
        )

        documents, skipped = _load_documents_for_mode(config, trace)
        documents, sampled = _stable_sample(documents, config.sample_limit)
        metrics_config = ChunkMetricsConfig(
            too_short_chars=config.too_short_chars,
            too_long_chars=int(config.chunk_size * 1.2),
        )
        metrics = evaluate_documents(documents, config=metrics_config)
        _trace_chunk_events(
            documents,
            trace,
            mode=config.mode,
            too_short_chars=config.too_short_chars,
        )

        payload = {
            "mode": config.mode,
            "data_dir": _path_label(
                (project_root / config.data_dir)
                if not config.data_dir.is_absolute()
                else config.data_dir,
                project_root,
            ),
            "generated_at_utc": _utc_now(),
            "sampled": sampled,
            "sample_limit": config.sample_limit,
            "trace_enabled": config.trace_enabled,
            "trace_path": trace.path_label,
            "config": {
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "too_short_chars": config.too_short_chars,
                "too_long_chars": metrics_config.too_long_chars,
                "chunk_policy_version": chunk_policy_version_for_mode(config.mode),
            },
            "skipped": skipped,
            "summary": metrics["summary"],
            "metadata": metrics["metadata"],
            "structure": metrics["structure"],
            "per_subject": metrics["per_subject"],
            "per_source": metrics["per_source"],
            "warnings": metrics["warnings"],
        }
        _write_json(payload, output_path)
        trace.write(
            "report_written",
            {
                "mode": config.mode,
                "report_path": _path_label(output_path, project_root),
            },
        )
        trace.write(
            "run_finished",
            {
                "run_id": run_id,
                "mode": config.mode,
                "total_chunks": payload["summary"]["total_chunks"],
            },
        )
        return payload
    except Exception as exc:
        _write_run_failed(trace, run_id=run_id, mode=config.mode, exc=exc)
        raise


def evaluate_mode(config: ChunkEvaluationConfig) -> dict[str, Any]:
    """Evaluate one splitter mode and write its JSON report."""

    return _evaluate_mode(config, reset_trace=True)


def _summary_delta(
    baseline_report: dict[str, Any], candidate_report: dict[str, Any]
) -> dict[str, Any]:
    baseline = baseline_report["summary"]
    candidate = candidate_report["summary"]
    baseline_metadata = baseline_report["metadata"]
    candidate_metadata = candidate_report["metadata"]
    baseline_chunks = int(baseline["total_chunks"])
    candidate_chunks = int(candidate["total_chunks"])
    return {
        "total_chunks_delta": candidate_chunks - baseline_chunks,
        "total_chunks_ratio": round(candidate_chunks / baseline_chunks, 4)
        if baseline_chunks
        else None,
        "too_short_count_delta": int(candidate["too_short_count"])
        - int(baseline["too_short_count"]),
        "too_short_ratio_delta": round(
            float(candidate["too_short_ratio"]) - float(baseline["too_short_ratio"]),
            4,
        ),
        "duplicate_chunk_count_delta": int(candidate["duplicate_chunk_count"])
        - int(baseline["duplicate_chunk_count"]),
        "duplicate_ratio_delta": round(
            float(candidate["duplicate_ratio"]) - float(baseline["duplicate_ratio"]),
            4,
        ),
        "section_metadata_coverage_delta": round(
            float(candidate_metadata["section_metadata_coverage"])
            - float(baseline_metadata["section_metadata_coverage"]),
            4,
        ),
    }


def _judgement(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    delta: dict[str, Any],
) -> dict[str, Any]:
    baseline_summary = baseline_report["summary"]
    candidate_summary = candidate_report["summary"]
    candidate_metadata = candidate_report["metadata"]
    reasons: list[str] = []
    recommendations: list[str] = []

    if int(candidate_summary["empty_chunk_count"]):
        reasons.append("candidate has empty chunks")
    if any(
        count > 0
        for count in candidate_metadata.get("missing_metadata_counts", {}).values()
    ):
        reasons.append("candidate has missing required metadata")
    if int(candidate_summary["source_count"]) < int(baseline_summary["source_count"]):
        reasons.append("candidate loses sources")
    if reasons:
        return {
            "status": "fail",
            "reasons": reasons,
            "recommendations": [
                "Do not promote candidate splitter until hard errors are fixed."
            ],
        }

    review_reasons: list[str] = []
    total_ratio = delta.get("total_chunks_ratio")
    if total_ratio is not None and total_ratio > CHUNK_COUNT_REVIEW_RATIO:
        review_reasons.append(
            f"candidate increases chunk count by {round((total_ratio - 1) * 100, 2)}%"
        )
    if delta["too_short_ratio_delta"] > SHORT_RATIO_REVIEW_DELTA:
        review_reasons.append("candidate increases short chunk ratio")
    if delta["duplicate_ratio_delta"] > DUPLICATE_RATIO_REVIEW_DELTA:
        review_reasons.append("candidate increases duplicate ratio")
    if review_reasons:
        recommendations.append(
            "Keep current splitter default until review metrics are evaluated."
        )
        recommendations.append(
            "Use chunk-level trace to inspect short or duplicate candidate chunks."
        )
        return {
            "status": "needs_review",
            "reasons": review_reasons,
            "recommendations": recommendations,
        }

    return {
        "status": "pass",
        "reasons": ["candidate has no hard errors or review threshold regressions"],
        "recommendations": [
            "Review downstream retrieval quality before changing defaults."
        ],
    }


def build_compare_report(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
) -> dict[str, Any]:
    """Build a JSON-ready comparison report from two mode reports."""

    delta = _summary_delta(baseline_report, candidate_report)
    return {
        "baseline_mode": baseline_report["mode"],
        "candidate_mode": candidate_report["mode"],
        "generated_at_utc": _utc_now(),
        "sampled": bool(
            baseline_report.get("sampled") or candidate_report.get("sampled")
        ),
        "sample_limit": baseline_report.get("sample_limit")
        if baseline_report.get("sampled")
        else candidate_report.get("sample_limit"),
        "trace_enabled": bool(
            baseline_report.get("trace_enabled")
            or candidate_report.get("trace_enabled")
        ),
        "trace_path": baseline_report.get("trace_path")
        or candidate_report.get("trace_path"),
        "summary_delta": delta,
        "judgement": _judgement(baseline_report, candidate_report, delta),
    }


def compare_modes(
    *,
    baseline_mode: str,
    candidate_mode: str,
    data_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    subjects: tuple[str, ...] = (),
    too_short_chars: int = 80,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    sample_limit: int | None = None,
    trace_enabled: bool = True,
    trace_output: Path | None = None,
    run_id: str | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate two splitter modes and write the comparison report."""

    project_root = _project_root(project_root)
    active_run_id = run_id or uuid4().hex[:12]
    trace = TraceWriter(
        enabled=trace_enabled,
        output_dir=output_dir,
        trace_output=trace_output,
        run_id=active_run_id,
        project_root=project_root,
    )
    if trace_output is not None:
        trace.clear()

    try:
        _validate_mode(baseline_mode)
        _validate_mode(candidate_mode)
        baseline_report = _evaluate_mode(
            ChunkEvaluationConfig(
                mode=baseline_mode,
                data_dir=data_dir,
                output_dir=output_dir,
                subjects=subjects,
                too_short_chars=too_short_chars,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                sample_limit=sample_limit,
                trace_enabled=trace_enabled,
                trace_output=trace_output,
                run_id=active_run_id,
                project_root=project_root,
            ),
            reset_trace=False,
        )
        candidate_report = _evaluate_mode(
            ChunkEvaluationConfig(
                mode=candidate_mode,
                data_dir=data_dir,
                output_dir=output_dir,
                subjects=subjects,
                too_short_chars=too_short_chars,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                sample_limit=sample_limit,
                trace_enabled=trace_enabled,
                trace_output=trace_output,
                run_id=active_run_id,
                project_root=project_root,
            ),
            reset_trace=False,
        )
        payload = build_compare_report(baseline_report, candidate_report)
        output_path = _compare_output_path(output_dir, project_root)
        _write_json(payload, output_path)

        trace.write(
            "comparison_written",
            {
                "baseline_mode": baseline_mode,
                "candidate_mode": candidate_mode,
                "report_path": _path_label(output_path, project_root),
            },
        )
        return payload
    except Exception as exc:
        _write_run_failed(
            trace,
            run_id=active_run_id,
            baseline_mode=baseline_mode,
            candidate_mode=candidate_mode,
            exc=exc,
        )
        raise
