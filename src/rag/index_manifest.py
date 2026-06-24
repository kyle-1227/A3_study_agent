"""Build-run manifest for RAG index inputs."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

DEFAULT_MANIFEST_NOTES = (
    "Manifest describes documents loaded in this build run. "
    "Run scripts/reset_index.py --yes before build_index.py for a clean rebuild."
)


@dataclass(frozen=True)
class SourceManifest:
    subject: str
    source_file: str
    source_relpath: str
    source_file_sha1: str
    source_file_size: int
    chunk_count: int


@dataclass(frozen=True)
class BuildManifest:
    index_version: str
    chunk_policy_version: str
    splitter_mode: str
    collection_name: str
    chroma_persist_dir: str
    embedding_model: str
    build_time_utc: str
    total_chunks: int
    source_count: int
    notes: str = DEFAULT_MANIFEST_NOTES
    sources: list[SourceManifest] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metadata_text(metadata: dict, key: str) -> str:
    value = metadata.get(key)
    return value if isinstance(value, str) else ""


def _metadata_int(metadata: dict, key: str) -> int:
    value = metadata.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _infer_splitter_mode(docs: list[Document]) -> str:
    modes = {
        _metadata_text(doc.metadata, "splitter_mode") or "recursive" for doc in docs
    }
    return modes.pop() if len(modes) == 1 else "mixed"


def _infer_chunk_policy_version(docs: list[Document]) -> str:
    versions = {
        _metadata_text(doc.metadata, "chunk_policy_version") or "recursive_v1"
        for doc in docs
    }
    return versions.pop() if len(versions) == 1 else "mixed"


def build_manifest_from_documents(
    docs: list[Document],
    *,
    collection_name: str,
    chroma_persist_dir: str,
    embedding_model: str,
    index_version: str = "a3_rag_v1",
    chunk_policy_version: str | None = None,
    splitter_mode: str | None = None,
) -> BuildManifest:
    """Build a manifest describing the documents loaded in one build run."""

    counts: Counter[str] = Counter(
        _metadata_text(doc.metadata, "doc_id")
        or _metadata_text(doc.metadata, "source_file")
        for doc in docs
    )
    representatives: dict[str, dict] = {}
    for doc in docs:
        key = _metadata_text(doc.metadata, "doc_id") or _metadata_text(
            doc.metadata, "source_file"
        )
        representatives.setdefault(key, doc.metadata)

    sources = [
        SourceManifest(
            subject=_metadata_text(metadata, "subject"),
            source_file=_metadata_text(metadata, "source_file"),
            source_relpath=_metadata_text(metadata, "source_relpath"),
            source_file_sha1=_metadata_text(metadata, "source_file_sha1"),
            source_file_size=_metadata_int(metadata, "source_file_size"),
            chunk_count=counts[key],
        )
        for key, metadata in sorted(
            representatives.items(),
            key=lambda item: (
                _metadata_text(item[1], "source_relpath"),
                _metadata_text(item[1], "source_file"),
                item[0],
            ),
        )
    ]

    return BuildManifest(
        index_version=index_version,
        chunk_policy_version=chunk_policy_version
        if chunk_policy_version is not None
        else _infer_chunk_policy_version(docs),
        splitter_mode=splitter_mode
        if splitter_mode is not None
        else _infer_splitter_mode(docs),
        collection_name=collection_name,
        chroma_persist_dir=chroma_persist_dir,
        embedding_model=embedding_model,
        build_time_utc=datetime.now(UTC).replace(microsecond=0).isoformat(),
        total_chunks=len(docs),
        source_count=len(sources),
        sources=sources,
    )


def write_build_manifest(manifest: BuildManifest, path: str | Path) -> None:
    """Write a build manifest as UTF-8 JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
