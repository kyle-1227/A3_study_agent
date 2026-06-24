"""Stable identifiers and metadata enrichment for RAG indexing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

INDEX_VERSION = "a3_rag_v1"
CHUNK_POLICY_VERSION = "recursive_v1"
STRUCTURE_CHUNK_POLICY_VERSION = "structure_v1"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCALAR_TYPES = (str, int, float, bool, type(None))


def sha1_text(value: str) -> str:
    """Return the SHA1 hex digest for UTF-8 text."""

    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def sha1_file(path: str | Path) -> str:
    """Return the SHA1 hex digest for a file's bytes."""

    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_for_hash(text: str) -> str:
    """Whitespace-normalized text for stable content hashing."""

    return " ".join(text.split())


def make_source_relpath(
    path: str | Path, *, project_root: str | Path | None = None
) -> str:
    """Return a stable source path for ID metadata."""

    source_path = Path(path).resolve()
    root = Path(project_root).resolve() if project_root is not None else _PROJECT_ROOT
    try:
        return source_path.relative_to(root).as_posix()
    except ValueError:
        return f"external/{source_path.name}"


def make_doc_id(
    *,
    subject: str,
    source_relpath: str,
    file_sha1: str,
) -> str:
    """Build a stable document ID from subject, path, and file digest."""

    return f"doc_{sha1_text(f'{subject}|{source_relpath}|{file_sha1}')}"


def make_chunk_id(
    *,
    doc_id: str,
    chunk_policy_version: str,
    chunk_index: int,
    content_sha1: str,
) -> str:
    """Build a stable chunk ID from document identity and chunk position."""

    return f"chunk_{sha1_text(f'{doc_id}|{chunk_policy_version}|{chunk_index}|{content_sha1}')}"


def _assert_scalar_metadata(metadata: dict[str, Any]) -> None:
    invalid = {
        key: type(value).__name__
        for key, value in metadata.items()
        if not isinstance(value, _SCALAR_TYPES)
    }
    if invalid:
        details = ", ".join(
            f"{key}={type_name}" for key, type_name in sorted(invalid.items())
        )
        raise TypeError(f"Document metadata must be scalar-only for Chroma: {details}")


def enrich_source_metadata(
    metadata: dict,
    *,
    source_path: str | Path,
    subject: str,
    project_root: str | Path | None = None,
) -> dict:
    """Return source-level metadata with stable document ID fields."""

    source = Path(source_path)
    source_relpath = make_source_relpath(source, project_root=project_root)
    file_sha1 = sha1_file(source)
    enriched = {
        **metadata,
        "doc_id": make_doc_id(
            subject=subject,
            source_relpath=source_relpath,
            file_sha1=file_sha1,
        ),
        "source_relpath": source_relpath,
        "source_file_sha1": file_sha1,
        "source_file_size": source.stat().st_size,
        "index_version": INDEX_VERSION,
    }
    _assert_scalar_metadata(enriched)
    return enriched


def enrich_chunk_metadata(
    doc: Document,
    *,
    doc_id: str,
    chunk_index: int,
    chunk_policy_version: str = CHUNK_POLICY_VERSION,
    index_version: str = INDEX_VERSION,
) -> Document:
    """Return a new Document with stable chunk-level metadata."""

    content_sha1 = sha1_text(normalize_for_hash(doc.page_content))
    metadata = {
        **doc.metadata,
        "chunk_id": make_chunk_id(
            doc_id=doc_id,
            chunk_policy_version=chunk_policy_version,
            chunk_index=chunk_index,
            content_sha1=content_sha1,
        ),
        "chunk_index": chunk_index,
        "chunk_policy_version": chunk_policy_version,
        "content_sha1": content_sha1,
        "chunk_chars": len(doc.page_content),
        "index_version": index_version,
    }
    _assert_scalar_metadata(metadata)
    return Document(page_content=doc.page_content, metadata=metadata)
