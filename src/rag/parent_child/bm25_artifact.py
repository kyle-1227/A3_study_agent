"""Safe, deterministic per-subject BM25 corpus artifacts."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.rag.parent_child._storage_io import (
    atomic_write_bytes,
    canonical_json_bytes,
    resolve_under_root,
    sha256_bytes,
    sha256_file,
    validate_relative_path,
)


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class Bm25ArtifactError(RuntimeError):
    """Raised when a BM25 artifact violates its immutable contract."""


class Bm25CorpusRow(BaseModel):
    """One tokenized child document in a subject-local corpus."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    child_id: str = Field(min_length=1)
    tokens: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _tokens_are_non_empty(self) -> Bm25CorpusRow:
        if any(not token.strip() for token in self.tokens):
            raise ValueError("BM25 tokens must be non-empty strings")
        return self


class Bm25ArtifactManifest(BaseModel):
    """Integrity and tokenizer identity for one subject's JSONL corpus."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    artifact_relative_path: str = Field(min_length=1)
    tokenizer_name: str = Field(min_length=1)
    tokenizer_version: str = Field(min_length=1)
    dictionary_sha256: str = Field(min_length=1)
    tokenizer_fingerprint: str = Field(min_length=1)
    child_count: int = Field(ge=0)
    child_id_set_sha256: str = Field(min_length=1)
    corpus_sha256: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_integrity_fields(self) -> Bm25ArtifactManifest:
        validate_relative_path(self.artifact_relative_path)
        digest_fields = (
            self.dictionary_sha256,
            self.tokenizer_fingerprint,
            self.child_id_set_sha256,
            self.corpus_sha256,
        )
        if any(not _SHA256_PATTERN.fullmatch(value) for value in digest_fields):
            raise ValueError("BM25 manifest digests must be lowercase SHA-256 values")
        expected_fingerprint = compute_tokenizer_fingerprint(
            tokenizer_name=self.tokenizer_name,
            tokenizer_version=self.tokenizer_version,
            dictionary_sha256=self.dictionary_sha256,
        )
        if self.tokenizer_fingerprint != expected_fingerprint:
            raise ValueError("BM25 tokenizer fingerprint mismatch")
        return self


def digest_identifier_set(identifiers: Sequence[str]) -> str:
    """Digest a set of non-empty identifiers in deterministic order."""

    if any(not identifier for identifier in identifiers):
        raise ValueError("identifier sets cannot contain empty values")
    unique = sorted(set(identifiers))
    return sha256_bytes(canonical_json_bytes(unique))


def compute_tokenizer_fingerprint(
    *,
    tokenizer_name: str,
    tokenizer_version: str,
    dictionary_sha256: str,
) -> str:
    """Fingerprint every input that affects BM25 tokenization."""

    if not tokenizer_name or not tokenizer_version:
        raise ValueError("tokenizer name and version are required")
    if not _SHA256_PATTERN.fullmatch(dictionary_sha256):
        raise ValueError("dictionary_sha256 must be a lowercase SHA-256 value")
    return sha256_bytes(
        canonical_json_bytes(
            {
                "dictionary_sha256": dictionary_sha256,
                "tokenizer_name": tokenizer_name,
                "tokenizer_version": tokenizer_version,
            }
        )
    )


def write_subject_bm25_artifact(
    root: str | Path,
    artifact_relative_path: str,
    manifest_relative_path: str,
    rows: Sequence[Bm25CorpusRow],
    *,
    manifest_schema_version: str,
    expected_generation_id: str,
    expected_subject: str,
    tokenizer_name: str,
    tokenizer_version: str,
    dictionary_sha256: str,
) -> Bm25ArtifactManifest:
    """Write a JSONL corpus and its strict manifest without unsafe serialization."""

    if not manifest_schema_version:
        raise ValueError("manifest_schema_version is required")
    validate_relative_path(artifact_relative_path)
    validate_relative_path(manifest_relative_path)
    if artifact_relative_path == manifest_relative_path:
        raise ValueError("BM25 artifact and manifest paths must differ")

    sorted_rows = sorted(rows, key=lambda item: item.child_id)
    child_ids = [row.child_id for row in sorted_rows]
    if len(child_ids) != len(set(child_ids)):
        raise Bm25ArtifactError("duplicate child IDs are forbidden in BM25 corpus")
    for row in sorted_rows:
        if row.generation_id != expected_generation_id:
            raise Bm25ArtifactError("BM25 row generation mismatch")
        if row.subject != expected_subject:
            raise Bm25ArtifactError("BM25 row subject mismatch")

    corpus_bytes = b"".join(
        canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in sorted_rows
    )
    tokenizer_fingerprint = compute_tokenizer_fingerprint(
        tokenizer_name=tokenizer_name,
        tokenizer_version=tokenizer_version,
        dictionary_sha256=dictionary_sha256,
    )
    manifest = Bm25ArtifactManifest(
        schema_version=manifest_schema_version,
        generation_id=expected_generation_id,
        subject=expected_subject,
        artifact_relative_path=artifact_relative_path,
        tokenizer_name=tokenizer_name,
        tokenizer_version=tokenizer_version,
        dictionary_sha256=dictionary_sha256,
        tokenizer_fingerprint=tokenizer_fingerprint,
        child_count=len(sorted_rows),
        child_id_set_sha256=digest_identifier_set(child_ids),
        corpus_sha256=sha256_bytes(corpus_bytes),
    )

    artifact_path = atomic_write_bytes(
        root,
        artifact_relative_path,
        corpus_bytes,
        overwrite=False,
    )
    try:
        atomic_write_bytes(
            root,
            manifest_relative_path,
            canonical_json_bytes(manifest.model_dump(mode="json")),
            overwrite=False,
        )
    except BaseException:
        artifact_path.unlink()
        raise
    return manifest


def read_subject_bm25_artifact(
    root: str | Path,
    manifest_relative_path: str,
    *,
    expected_manifest_schema_version: str,
    expected_generation_id: str,
    expected_subject: str,
    expected_tokenizer_fingerprint: str,
) -> tuple[Bm25ArtifactManifest, tuple[Bm25CorpusRow, ...]]:
    """Read and fully verify one subject-local BM25 artifact."""

    manifest_path = resolve_under_root(root, manifest_relative_path, must_exist=True)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise Bm25ArtifactError("BM25 manifest must be a regular file")
    manifest = Bm25ArtifactManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.schema_version != expected_manifest_schema_version:
        raise Bm25ArtifactError("BM25 manifest schema version mismatch")
    if manifest.generation_id != expected_generation_id:
        raise Bm25ArtifactError("BM25 manifest generation mismatch")
    if manifest.subject != expected_subject:
        raise Bm25ArtifactError("BM25 manifest subject mismatch")
    if manifest.tokenizer_fingerprint != expected_tokenizer_fingerprint:
        raise Bm25ArtifactError("BM25 tokenizer fingerprint does not match runtime")

    artifact_path = resolve_under_root(
        root,
        manifest.artifact_relative_path,
        must_exist=True,
    )
    if sha256_file(artifact_path) != manifest.corpus_sha256:
        raise Bm25ArtifactError("BM25 corpus digest mismatch")

    rows: list[Bm25CorpusRow] = []
    with artifact_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.endswith(b"\n"):
                raise Bm25ArtifactError(
                    f"BM25 corpus line {line_number} lacks newline terminator"
                )
            payload = raw_line[:-1]
            if not payload:
                raise Bm25ArtifactError(f"BM25 corpus line {line_number} is empty")
            rows.append(Bm25CorpusRow.model_validate_json(payload))

    child_ids = [row.child_id for row in rows]
    if len(child_ids) != len(set(child_ids)):
        raise Bm25ArtifactError("BM25 corpus contains duplicate child IDs")
    if rows != sorted(rows, key=lambda item: item.child_id):
        raise Bm25ArtifactError("BM25 corpus rows are not canonically ordered")
    if any(row.generation_id != expected_generation_id for row in rows):
        raise Bm25ArtifactError("BM25 corpus row generation mismatch")
    if any(row.subject != expected_subject for row in rows):
        raise Bm25ArtifactError("BM25 corpus row subject mismatch")
    if len(rows) != manifest.child_count:
        raise Bm25ArtifactError("BM25 corpus child count mismatch")
    if digest_identifier_set(child_ids) != manifest.child_id_set_sha256:
        raise Bm25ArtifactError("BM25 corpus child ID-set digest mismatch")
    return manifest, tuple(rows)
