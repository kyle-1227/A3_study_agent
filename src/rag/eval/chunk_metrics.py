"""Intrinsic chunking metrics for read-only splitter evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from statistics import mean, median
from typing import Any, Iterable

from langchain_core.documents import Document

from src.rag.ids import normalize_for_hash, sha1_text
from src.rag.metadata_schema import REQUIRED_EVALUATION_METADATA_FIELDS

DEFAULT_PREVIEW_CHARS = 120
DEFAULT_SAMPLE_PREVIEW_LIMIT = 20

_SECRET_PATTERNS = (
    re.compile(r"(?i)\b[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET)\s*[:=]\s*\S+"),
    re.compile(
        r"(?i)\b(DEEPSEEK_API_KEY|OPENROUTER_API_KEY|TAVILY_API_KEY)\s*[:=]\s*\S+"
    ),
    re.compile(r"(?i)\b(Authorization|Cookie|x-api-key)\s*[:=]\s*\S+"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"(?i)\bpostgres(?:ql)?://\S+"),
    re.compile(r"\bsk-[A-Za-z0-9._\-]+"),
    re.compile(r"(?i)\bsecret[_\-]?[A-Za-z0-9._\-]*\b"),
    re.compile(r"\b[A-Za-z]:\\[^\s]*"),
)


@dataclass(frozen=True)
class ChunkMetricsConfig:
    """Thresholds and field contract for evaluation metrics."""

    too_short_chars: int = 80
    too_long_chars: int = 1200
    preview_chars: int = DEFAULT_PREVIEW_CHARS
    required_metadata: tuple[str, ...] = REQUIRED_EVALUATION_METADATA_FIELDS
    sample_preview_limit: int = DEFAULT_SAMPLE_PREVIEW_LIMIT


def metadata_text(metadata: dict[str, Any], key: str) -> str:
    """Return a metadata string value, or an empty string for non-strings."""

    value = metadata.get(key)
    return value if isinstance(value, str) else ""


def metadata_int(metadata: dict[str, Any], key: str) -> int:
    """Return a metadata int value, excluding bool."""

    value = metadata.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def sanitized_preview(text: str, *, max_chars: int = DEFAULT_PREVIEW_CHARS) -> str:
    """Return a bounded preview with obvious secret/path patterns redacted."""

    preview = " ".join(text.split())[:max_chars]
    for pattern in _SECRET_PATTERNS:
        preview = pattern.sub("<redacted>", preview)
    return preview[:max_chars]


def chunk_hash(doc: Document) -> str:
    """Return the stable duplicate-detection hash for a chunk."""

    content_sha1 = metadata_text(doc.metadata, "content_sha1")
    if content_sha1:
        return content_sha1
    return sha1_text(normalize_for_hash(doc.page_content))


def duplicate_flags(documents: Iterable[Document]) -> list[bool]:
    """Mark chunks that are not the first occurrence of their content hash."""

    seen: set[str] = set()
    flags: list[bool] = []
    for doc in documents:
        key = chunk_hash(doc)
        flags.append(key in seen)
        seen.add(key)
    return flags


def _percentile(values: list[int], percentile: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = math.ceil((percentile / 100) * len(ordered))
    index = max(0, min(rank - 1, len(ordered) - 1))
    return ordered[index]


def _source_key(doc: Document) -> str:
    return (
        metadata_text(doc.metadata, "source_relpath")
        or metadata_text(doc.metadata, "source_file")
        or "unknown"
    )


def _subject_key(doc: Document) -> str:
    return metadata_text(doc.metadata, "subject") or "unknown"


def _missing_metadata_counts(
    documents: list[Document], required_metadata: tuple[str, ...]
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for doc in documents:
        for key in required_metadata:
            if doc.metadata.get(key) in ("", None):
                counts[key] += 1
    return dict(counts)


def _required_metadata_coverage(
    documents: list[Document], required_metadata: tuple[str, ...]
) -> float:
    if not documents or not required_metadata:
        return 1.0
    total_slots = len(documents) * len(required_metadata)
    missing = sum(_missing_metadata_counts(documents, required_metadata).values())
    return round((total_slots - missing) / total_slots, 4)


def _section_coverage(documents: list[Document]) -> float:
    if not documents:
        return 0.0
    count = sum(1 for doc in documents if doc.metadata.get("section_id"))
    return round(count / len(documents), 4)


def _is_title_only_chunk(doc: Document) -> bool:
    section_title = metadata_text(doc.metadata, "section_title")
    if not section_title or len(doc.page_content) >= 120:
        return False
    if section_title == "Preamble":
        return False
    text = " ".join(doc.page_content.split()).strip("#").strip().casefold()
    title = " ".join(section_title.split()).casefold()
    return bool(title) and (text == title or text.startswith(title))


def _length_stats(lengths: list[int]) -> dict[str, int | float]:
    return {
        "min_chars": min(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
        "avg_chars": round(mean(lengths), 2) if lengths else 0.0,
        "median_chars": median(lengths) if lengths else 0,
        "p10_chars": _percentile(lengths, 10),
        "p90_chars": _percentile(lengths, 90),
    }


def _sample_records(
    indexed_documents: list[tuple[int, Document]],
    duplicate_values: list[bool],
    *,
    config: ChunkMetricsConfig,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for (position, doc), is_duplicate in zip(
        indexed_documents, duplicate_values, strict=True
    ):
        if len(doc.page_content) >= config.too_short_chars:
            continue
        samples.append(
            {
                "source_relpath": _source_key(doc),
                "chunk_index": metadata_int(doc.metadata, "chunk_index")
                if "chunk_index" in doc.metadata
                else position,
                "chunk_chars": len(doc.page_content),
                "is_duplicate": is_duplicate,
                "preview": sanitized_preview(
                    doc.page_content, max_chars=config.preview_chars
                ),
            }
        )
        if len(samples) >= config.sample_preview_limit:
            break
    return samples


def _group_metrics(
    grouped_documents: dict[str, list[tuple[int, Document, bool]]],
    *,
    group_key: str,
    config: ChunkMetricsConfig,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for key, items in sorted(grouped_documents.items()):
        docs = [doc for _, doc, _ in items]
        duplicates = [is_duplicate for _, _, is_duplicate in items]
        lengths = [len(doc.page_content) for doc in docs]
        source_values = {_source_key(doc) for doc in docs}
        section_ids = {
            metadata_text(doc.metadata, "section_id")
            for doc in docs
            if metadata_text(doc.metadata, "section_id")
        }
        payload = {
            group_key: key,
            "chunk_count": len(docs),
            "source_count": len(source_values),
            **_length_stats(lengths),
            "too_short_count": sum(
                1 for value in lengths if value < config.too_short_chars
            ),
            "too_long_count": sum(
                1 for value in lengths if value > config.too_long_chars
            ),
            "empty_chunk_count": sum(1 for doc in docs if not doc.page_content.strip()),
            "duplicate_chunk_count": sum(1 for value in duplicates if value),
            "section_metadata_coverage": _section_coverage(docs),
            "unique_section_count": len(section_ids),
        }
        if group_key == "source_relpath":
            first_doc = docs[0] if docs else Document(page_content="", metadata={})
            payload.update(
                {
                    "source_file": metadata_text(first_doc.metadata, "source_file"),
                    "subject": _subject_key(first_doc),
                    "warnings": _source_warnings(payload),
                }
            )
        output.append(payload)
    return output


def _source_warnings(source_payload: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    chunk_count = int(source_payload.get("chunk_count") or 0)
    if source_payload.get("empty_chunk_count"):
        warnings.append("empty_chunks_detected")
    if source_payload.get("duplicate_chunk_count"):
        warnings.append("duplicate_chunks_detected")
    if (
        chunk_count
        and int(source_payload.get("too_short_count") or 0) / chunk_count >= 0.2
    ):
        warnings.append("high_short_chunk_ratio")
    return warnings


def evaluate_documents(
    documents: list[Document],
    *,
    config: ChunkMetricsConfig | None = None,
) -> dict[str, Any]:
    """Return JSON-ready intrinsic metrics for chunk documents."""

    active_config = config or ChunkMetricsConfig()
    indexed = list(enumerate(documents))
    duplicates = duplicate_flags(documents)
    lengths = [len(doc.page_content) for doc in documents]
    total_chunks = len(documents)
    missing_metadata = _missing_metadata_counts(
        documents, active_config.required_metadata
    )
    section_ids = [
        metadata_text(doc.metadata, "section_id")
        for doc in documents
        if metadata_text(doc.metadata, "section_id")
    ]
    section_counts = Counter(section_ids)
    chunks_with_section_title = sum(
        1 for doc in documents if metadata_text(doc.metadata, "section_title")
    )
    too_short_count = sum(
        1 for value in lengths if value < active_config.too_short_chars
    )
    duplicate_count = sum(1 for value in duplicates if value)
    empty_count = sum(1 for doc in documents if not doc.page_content.strip())

    subject_groups: dict[str, list[tuple[int, Document, bool]]] = defaultdict(list)
    source_groups: dict[str, list[tuple[int, Document, bool]]] = defaultdict(list)
    for (position, doc), is_duplicate in zip(indexed, duplicates, strict=True):
        subject_groups[_subject_key(doc)].append((position, doc, is_duplicate))
        source_groups[_source_key(doc)].append((position, doc, is_duplicate))

    warnings: list[str] = []
    if empty_count:
        warnings.append("empty_chunks_detected")
    if duplicate_count:
        warnings.append("duplicate_chunks_detected")
    if too_short_count:
        warnings.append("short_chunks_detected")
    if any(value > 0 for value in missing_metadata.values()):
        warnings.append("missing_required_metadata")

    summary = {
        "total_chunks": total_chunks,
        "source_count": len(source_groups),
        "subject_count": len(subject_groups),
        **_length_stats(lengths),
        "too_short_count": too_short_count,
        "too_short_ratio": round(too_short_count / total_chunks, 4)
        if total_chunks
        else 0.0,
        "too_long_count": sum(
            1 for value in lengths if value > active_config.too_long_chars
        ),
        "empty_chunk_count": empty_count,
        "duplicate_chunk_count": duplicate_count,
        "duplicate_ratio": round(duplicate_count / total_chunks, 4)
        if total_chunks
        else 0.0,
        "short_chunk_samples": _sample_records(
            indexed, duplicates, config=active_config
        ),
    }

    structure = {
        "chunks_with_section_id": len(section_ids),
        "chunks_with_section_title": chunks_with_section_title,
        "unique_section_count": len(section_counts),
        "avg_chunks_per_section": round(mean(section_counts.values()), 2)
        if section_counts
        else 0.0,
        "max_chunks_per_section": max(section_counts.values()) if section_counts else 0,
        "section_title_only_chunk_count": sum(
            1 for doc in documents if _is_title_only_chunk(doc)
        ),
        "preamble_chunk_count": sum(
            1
            for doc in documents
            if metadata_text(doc.metadata, "section_title") == "Preamble"
        ),
    }

    return {
        "summary": summary,
        "metadata": {
            "required_metadata_coverage": _required_metadata_coverage(
                documents, active_config.required_metadata
            ),
            "section_metadata_coverage": _section_coverage(documents),
            "missing_metadata_counts": missing_metadata,
            "required_metadata_fields": list(active_config.required_metadata),
        },
        "structure": structure,
        "per_subject": _group_metrics(
            subject_groups, group_key="subject", config=active_config
        ),
        "per_source": _group_metrics(
            source_groups, group_key="source_relpath", config=active_config
        ),
        "warnings": warnings,
        "duplicate_flags": duplicates,
    }
