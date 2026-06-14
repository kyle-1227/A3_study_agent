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


DEFAULT_REQUIRED_METADATA = ("subject", "source_file", "doc_type")


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "chunk_count": self.chunk_count,
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
            "avg_chars": round(self.avg_chars, 2),
            "too_short_count": self.too_short_count,
            "too_long_count": self.too_long_count,
            "empty_chunk_count": self.empty_chunk_count,
            "duplicate_chunk_count": self.duplicate_chunk_count,
            "missing_metadata_counts": dict(self.missing_metadata_counts),
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
        }


def _content_key(text: str) -> str:
    normalized = " ".join(text.split()).casefold()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest() if normalized else ""


def _missing_metadata_counts(documents: Iterable[Document], required_metadata: tuple[str, ...]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for doc in documents:
        for key in required_metadata:
            if not doc.metadata.get(key):
                counts[key] += 1
    return dict(counts)


def _duplicate_count(documents: list[Document]) -> int:
    keys = [_content_key(doc.page_content) for doc in documents if doc.page_content.strip()]
    counts = Counter(keys)
    return sum(count - 1 for key, count in counts.items() if key and count > 1)


def _audit_document_group(
    documents: list[Document],
    *,
    source_file: str,
    min_chars: int,
    max_chars: int,
    required_metadata: tuple[str, ...],
) -> SourceChunkAudit:
    lengths = [len(doc.page_content) for doc in documents]
    return SourceChunkAudit(
        source_file=source_file,
        chunk_count=len(documents),
        min_chars=min(lengths) if lengths else 0,
        max_chars=max(lengths) if lengths else 0,
        avg_chars=mean(lengths) if lengths else 0.0,
        too_short_count=sum(1 for value in lengths if value < min_chars),
        too_long_count=sum(1 for value in lengths if value > max_chars),
        empty_chunk_count=sum(1 for doc in documents if not doc.page_content.strip()),
        duplicate_chunk_count=_duplicate_count(documents),
        missing_metadata_counts=_missing_metadata_counts(documents, required_metadata),
    )


def audit_chunks(
    documents: list[Document],
    *,
    min_chars: int = 80,
    max_chars: int = 2000,
    required_metadata: tuple[str, ...] = DEFAULT_REQUIRED_METADATA,
) -> ChunkAuditReport:
    """Return read-only chunk quality statistics."""

    lengths = [len(doc.page_content) for doc in documents]
    grouped: dict[str, list[Document]] = defaultdict(list)
    for doc in documents:
        source_file = str(doc.metadata.get("source_file") or "unknown")
        grouped[source_file].append(doc)

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
    )
