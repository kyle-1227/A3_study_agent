"""Standalone chunk audit utilities.

The audit layer is intentionally read-only: it inspects LangChain Documents and
returns JSON-serializable statistics without modifying chunks or Chroma state.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
from statistics import mean
from typing import Any, Iterable

from langchain_core.documents import Document

from src.rag.chunking.validator import is_noise_chunk


DEFAULT_REQUIRED_METADATA = ("subject", "source_file", "doc_type")
DEFAULT_SAMPLE_PREVIEW_CHARS = 160
DEFAULT_SHORT_SAMPLE_LIMIT = 20
LARGE_SOURCE_BYTES = 1_000_000


@dataclass(frozen=True)
class ShortChunkSample:
    """Small bounded preview for a short chunk."""

    source_file: str
    chunk_index: int
    chunk_chars: int
    preview: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "chunk_chars": self.chunk_chars,
            "preview": self.preview,
        }


@dataclass(frozen=True)
class SuspiciousSourceFile:
    """Source-level diagnostic for suspicious chunking results."""

    source_file: str
    source_relpath: str
    source_file_size: int
    chunk_count: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "source_relpath": self.source_relpath,
            "source_file_size": self.source_file_size,
            "chunk_count": self.chunk_count,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class SourceChunkAudit:
    """Audit summary for chunks from one source file."""

    source_file: str
    chunk_count: int
    min_chars: int
    max_chars: int
    avg_chars: float
    too_short_count: int
    too_long_count: int
    empty_chunk_count: int
    duplicate_chunk_count: int
    missing_metadata_counts: dict[str, int] = field(default_factory=dict)
    source_relpath: str = ""
    source_file_size: int = 0
    warnings: tuple[str, ...] = ()
    short_chunk_samples: tuple[ShortChunkSample, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "source_relpath": self.source_relpath,
            "source_file_size": self.source_file_size,
            "chunk_count": self.chunk_count,
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
            "avg_chars": round(self.avg_chars, 2),
            "too_short_count": self.too_short_count,
            "too_long_count": self.too_long_count,
            "empty_chunk_count": self.empty_chunk_count,
            "duplicate_chunk_count": self.duplicate_chunk_count,
            "missing_metadata_counts": dict(self.missing_metadata_counts),
            "warnings": list(self.warnings),
            "short_chunk_samples": [
                item.to_dict() for item in self.short_chunk_samples
            ],
        }


@dataclass(frozen=True)
class ChunkAuditReport:
    """Top-level chunk audit report."""

    total_chunks: int
    source_count: int
    min_chars: int
    max_chars: int
    avg_chars: float
    too_short_count: int
    too_long_count: int
    empty_chunk_count: int
    duplicate_chunk_count: int
    missing_metadata_counts: dict[str, int] = field(default_factory=dict)
    per_source: tuple[SourceChunkAudit, ...] = ()
    warnings: tuple[str, ...] = ()
    short_chunk_samples: tuple[ShortChunkSample, ...] = ()
    suspicious_source_files: tuple[SuspiciousSourceFile, ...] = ()
    splitter_modes: dict[str, int] = field(default_factory=dict)
    section_metadata_coverage: dict[str, float | int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_chunks": self.total_chunks,
            "source_count": self.source_count,
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
            "avg_chars": round(self.avg_chars, 2),
            "too_short_count": self.too_short_count,
            "too_long_count": self.too_long_count,
            "empty_chunk_count": self.empty_chunk_count,
            "duplicate_chunk_count": self.duplicate_chunk_count,
            "missing_metadata_counts": dict(self.missing_metadata_counts),
            "per_source": [item.to_dict() for item in self.per_source],
            "warnings": list(self.warnings),
            "short_chunk_samples": [
                item.to_dict() for item in self.short_chunk_samples
            ],
            "suspicious_source_files": [
                item.to_dict() for item in self.suspicious_source_files
            ],
            "splitter_modes": dict(self.splitter_modes),
            "section_metadata_coverage": dict(self.section_metadata_coverage),
        }


def _content_key(text: str) -> str:
    normalized = " ".join(text.split()).casefold()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest() if normalized else ""


def _metadata_text(metadata: dict, key: str) -> str:
    value = metadata.get(key)
    return value if isinstance(value, str) else ""


def _metadata_int(metadata: dict, key: str) -> int:
    value = metadata.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _source_file(doc: Document) -> str:
    return _metadata_text(doc.metadata, "source_file") or "unknown"


def _source_relpath(documents: Iterable[Document], *, source_file: str) -> str:
    for doc in documents:
        relpath = _metadata_text(doc.metadata, "source_relpath")
        if relpath:
            return relpath
    return source_file or ""


def _source_file_size(documents: Iterable[Document]) -> int:
    return max(
        (_metadata_int(doc.metadata, "source_file_size") for doc in documents),
        default=0,
    )


def _report_chunk_index(doc: Document, *, fallback_index: int) -> int:
    value = doc.metadata.get("chunk_index")
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool)
        else fallback_index
    )


def _preview(text: str, *, max_chars: int = DEFAULT_SAMPLE_PREVIEW_CHARS) -> str:
    return " ".join(text.split())[:max_chars]


def _missing_metadata_counts(
    documents: Iterable[Document], required_metadata: tuple[str, ...]
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for doc in documents:
        for key in required_metadata:
            if not doc.metadata.get(key):
                counts[key] += 1
    return dict(counts)


def _splitter_modes(documents: Iterable[Document]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for doc in documents:
        mode = _metadata_text(doc.metadata, "splitter_mode") or "recursive"
        counts[mode] += 1
    return dict(counts)


def _section_metadata_coverage(documents: list[Document]) -> dict[str, float | int]:
    total = len(documents)
    chunks_with_section_id = sum(
        1 for doc in documents if doc.metadata.get("section_id")
    )
    chunks_with_section_title = sum(
        1 for doc in documents if doc.metadata.get("section_title")
    )
    return {
        "chunks_with_section_id": chunks_with_section_id,
        "chunks_with_section_title": chunks_with_section_title,
        "coverage_ratio": round(chunks_with_section_id / total, 4) if total else 0.0,
    }


def _duplicate_count(documents: list[Document]) -> int:
    keys = [
        _content_key(doc.page_content) for doc in documents if doc.page_content.strip()
    ]
    counts = Counter(keys)
    return sum(count - 1 for key, count in counts.items() if key and count > 1)


def _short_chunk_samples(
    indexed_documents: list[tuple[int, Document]],
    *,
    min_chars: int,
    limit: int = DEFAULT_SHORT_SAMPLE_LIMIT,
) -> tuple[ShortChunkSample, ...]:
    samples: list[ShortChunkSample] = []
    for fallback_index, doc in indexed_documents:
        if len(doc.page_content) >= min_chars:
            continue
        samples.append(
            ShortChunkSample(
                source_file=_source_file(doc),
                chunk_index=_report_chunk_index(doc, fallback_index=fallback_index),
                chunk_chars=len(doc.page_content),
                preview=_preview(doc.page_content),
            )
        )
        if len(samples) >= limit:
            break
    return tuple(samples)


def _source_warnings(
    *,
    chunk_count: int,
    source_file_size: int,
    too_short_count: int,
    empty_chunk_count: int,
    noise_count: int,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if source_file_size >= LARGE_SOURCE_BYTES and chunk_count == 1:
        warnings.append("single_chunk_large_source")
    if source_file_size >= LARGE_SOURCE_BYTES and chunk_count <= 2:
        warnings.append("very_low_chunk_count_for_large_source")
    if empty_chunk_count > 0:
        warnings.append("high_empty_chunk_count")
    if chunk_count and too_short_count / chunk_count >= 0.2:
        warnings.append("high_short_chunk_count")
    if noise_count > 0:
        warnings.append("noise_chunks_detected")
    return tuple(warnings)


def _audit_document_group(
    indexed_documents: list[tuple[int, Document]],
    *,
    source_file: str,
    min_chars: int,
    max_chars: int,
    required_metadata: tuple[str, ...],
) -> SourceChunkAudit:
    documents = [doc for _, doc in indexed_documents]
    lengths = [len(doc.page_content) for doc in documents]
    too_short_count = sum(1 for value in lengths if value < min_chars)
    empty_chunk_count = sum(1 for doc in documents if not doc.page_content.strip())
    noise_count = sum(1 for doc in documents if is_noise_chunk(doc.page_content))
    source_file_size = _source_file_size(documents)
    warnings = _source_warnings(
        chunk_count=len(documents),
        source_file_size=source_file_size,
        too_short_count=too_short_count,
        empty_chunk_count=empty_chunk_count,
        noise_count=noise_count,
    )
    return SourceChunkAudit(
        source_file=source_file,
        source_relpath=_source_relpath(documents, source_file=source_file),
        source_file_size=source_file_size,
        chunk_count=len(documents),
        min_chars=min(lengths) if lengths else 0,
        max_chars=max(lengths) if lengths else 0,
        avg_chars=mean(lengths) if lengths else 0.0,
        too_short_count=too_short_count,
        too_long_count=sum(1 for value in lengths if value > max_chars),
        empty_chunk_count=empty_chunk_count,
        duplicate_chunk_count=_duplicate_count(documents),
        missing_metadata_counts=_missing_metadata_counts(documents, required_metadata),
        warnings=warnings,
        short_chunk_samples=_short_chunk_samples(
            indexed_documents,
            min_chars=min_chars,
        ),
    )


def _suspicious_source_files(
    per_source: tuple[SourceChunkAudit, ...],
) -> tuple[SuspiciousSourceFile, ...]:
    suspicious: list[SuspiciousSourceFile] = []
    for source in per_source:
        reasons = tuple(
            reason
            for reason in source.warnings
            if reason
            in {
                "single_chunk_large_source",
                "very_low_chunk_count_for_large_source",
                "high_empty_chunk_count",
                "high_short_chunk_count",
            }
        )
        if not reasons:
            continue
        suspicious.append(
            SuspiciousSourceFile(
                source_file=source.source_file,
                source_relpath=source.source_relpath,
                source_file_size=source.source_file_size,
                chunk_count=source.chunk_count,
                reasons=reasons,
            )
        )
    return tuple(suspicious)


def audit_chunks(
    documents: list[Document],
    *,
    min_chars: int = 80,
    max_chars: int = 2000,
    required_metadata: tuple[str, ...] = DEFAULT_REQUIRED_METADATA,
) -> ChunkAuditReport:
    """Return read-only chunk quality statistics."""

    lengths = [len(doc.page_content) for doc in documents]
    indexed_documents = list(enumerate(documents))
    grouped: dict[str, list[tuple[int, Document]]] = defaultdict(list)
    for fallback_index, doc in indexed_documents:
        grouped[_source_file(doc)].append((fallback_index, doc))

    per_source = tuple(
        _audit_document_group(
            group,
            source_file=source_file,
            min_chars=min_chars,
            max_chars=max_chars,
            required_metadata=required_metadata,
        )
        for source_file, group in sorted(grouped.items())
    )
    suspicious_source_files = _suspicious_source_files(per_source)

    warnings: list[str] = []
    empty_chunk_count = sum(1 for doc in documents if not doc.page_content.strip())
    duplicate_chunk_count = _duplicate_count(documents)
    too_short_count = sum(1 for value in lengths if value < min_chars)
    too_long_count = sum(1 for value in lengths if value > max_chars)
    if empty_chunk_count:
        warnings.append("empty_chunks_detected")
    if duplicate_chunk_count:
        warnings.append("duplicate_chunks_detected")
    if too_short_count:
        warnings.append("short_chunks_detected")
    if too_long_count:
        warnings.append("long_chunks_detected")
    if suspicious_source_files:
        warnings.append("suspicious_sources_detected")

    return ChunkAuditReport(
        total_chunks=len(documents),
        source_count=len(grouped),
        min_chars=min(lengths) if lengths else 0,
        max_chars=max(lengths) if lengths else 0,
        avg_chars=mean(lengths) if lengths else 0.0,
        too_short_count=too_short_count,
        too_long_count=too_long_count,
        empty_chunk_count=empty_chunk_count,
        duplicate_chunk_count=duplicate_chunk_count,
        missing_metadata_counts=_missing_metadata_counts(documents, required_metadata),
        per_source=per_source,
        warnings=tuple(warnings),
        short_chunk_samples=_short_chunk_samples(
            indexed_documents,
            min_chars=min_chars,
        ),
        suspicious_source_files=suspicious_source_files,
        splitter_modes=_splitter_modes(documents),
        section_metadata_coverage=_section_metadata_coverage(documents),
    )
