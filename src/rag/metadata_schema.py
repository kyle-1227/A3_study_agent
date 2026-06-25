"""Shared RAG metadata field definitions."""

from __future__ import annotations

SCALAR_METADATA_TYPES = (str, int, float, bool, type(None))

REQUIRED_STABLE_METADATA_FIELDS = (
    "doc_id",
    "chunk_id",
    "source_relpath",
    "source_file_sha1",
    "source_file_size",
    "chunk_index",
    "chunk_policy_version",
    "index_version",
    "content_sha1",
    "chunk_chars",
)

REQUIRED_EVALUATION_METADATA_FIELDS = (
    *REQUIRED_STABLE_METADATA_FIELDS,
    "source_file",
    "subject",
    "doc_type",
)
