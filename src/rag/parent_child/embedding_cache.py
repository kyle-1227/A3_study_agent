"""Strict exact-content cache for resumable document embedding builds."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import re
import sqlite3
import struct
from threading import RLock
from typing import Literal, Protocol, Sequence
from uuid import uuid4

from src.rag.parent_child.project_paths import (
    resolve_project_path,
    resolve_project_root,
)


EmbeddingCacheOrigin = Literal["flat_baseline", "provider"]
EMBEDDING_CACHE_SCHEMA_VERSION = "embedding_cache_v1"
_CACHE_SCHEMA = """
CREATE TABLE cache_metadata (
    singleton INTEGER PRIMARY KEY NOT NULL CHECK (singleton = 1),
    schema_version TEXT NOT NULL,
    embedding_fingerprint TEXT NOT NULL CHECK (length(embedding_fingerprint) = 64),
    dimension INTEGER NOT NULL CHECK (dimension > 0)
) STRICT;
CREATE TABLE embeddings (
    content_sha256 TEXT PRIMARY KEY NOT NULL CHECK (length(content_sha256) = 64),
    content_chars INTEGER NOT NULL CHECK (content_chars > 0),
    vector_blob BLOB NOT NULL,
    vector_sha256 TEXT NOT NULL CHECK (length(vector_sha256) = 64),
    origin TEXT NOT NULL CHECK (origin IN ('flat_baseline', 'provider'))
) STRICT;
CREATE TABLE ambiguous_content (
    content_sha256 TEXT PRIMARY KEY NOT NULL CHECK (length(content_sha256) = 64),
    content_chars INTEGER NOT NULL CHECK (content_chars > 0),
    conflicting_row_count INTEGER NOT NULL CHECK (conflicting_row_count >= 2)
) STRICT;
"""
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class EmbeddingCacheError(RuntimeError):
    """A cache identity, row, vector, or filesystem invariant failed."""


class DocumentEmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


def _text_identity(text: str) -> tuple[str, int]:
    if not isinstance(text, str) or not text:
        raise EmbeddingCacheError("embedding cache texts must be non-empty strings")
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), len(text)


def _validate_vector(value: object, *, dimension: int) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EmbeddingCacheError("embedding cache vector must be a sequence")
    if len(value) != dimension:
        raise EmbeddingCacheError("embedding cache vector dimension mismatch")
    vector: list[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise EmbeddingCacheError(
                "embedding cache vector coordinates must be numeric"
            )
        normalized = float(coordinate)
        if not math.isfinite(normalized):
            raise EmbeddingCacheError(
                "embedding cache vector coordinates must be finite"
            )
        vector.append(normalized)
    return vector


def _encode_vector(vector: object, *, dimension: int) -> tuple[bytes, str]:
    normalized = _validate_vector(vector, dimension=dimension)
    payload = struct.pack(f"!{dimension}d", *normalized)
    return payload, hashlib.sha256(payload).hexdigest()


def _decode_vector(
    payload: object,
    *,
    expected_sha256: object,
    dimension: int,
) -> list[float]:
    if not isinstance(payload, bytes) or len(payload) != dimension * 8:
        raise EmbeddingCacheError("cached vector binary shape is invalid")
    if not isinstance(expected_sha256, str) or (
        hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise EmbeddingCacheError("cached vector digest mismatch")
    return _validate_vector(
        list(struct.unpack(f"!{dimension}d", payload)),
        dimension=dimension,
    )


class SqliteEmbeddingCache:
    """One fingerprint-bound SQLite cache with no persisted source text."""

    def __init__(
        self,
        *,
        connection: sqlite3.Connection,
        path: Path,
        schema_version: str,
        embedding_fingerprint: str,
        dimension: int,
    ) -> None:
        self._connection = connection
        self.path = path
        self.schema_version = schema_version
        self.embedding_fingerprint = embedding_fingerprint
        self.dimension = dimension
        self._lock = RLock()
        self._closed = False

    @classmethod
    def create(
        cls,
        *,
        project_root: Path,
        cache_path: Path,
        schema_version: str,
        embedding_fingerprint: str,
        dimension: int,
        busy_timeout_seconds: float,
    ) -> SqliteEmbeddingCache:
        root = resolve_project_root(project_root)
        path = resolve_project_path(root, cache_path, must_exist=False)
        if path.exists():
            raise FileExistsError(path)
        if (
            not schema_version
            or _SHA256_PATTERN.fullmatch(embedding_fingerprint) is None
        ):
            raise EmbeddingCacheError("cache schema and fingerprint are required")
        if dimension <= 0 or busy_timeout_seconds <= 0:
            raise EmbeddingCacheError("cache dimension and timeout must be positive")
        path.parent.mkdir(parents=True, exist_ok=True)
        path = resolve_project_path(root, path, must_exist=False)
        temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                temporary,
                timeout=busy_timeout_seconds,
                check_same_thread=False,
            )
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript(_CACHE_SCHEMA)
            connection.execute(
                "INSERT INTO cache_metadata VALUES (1, ?, ?, ?)",
                (schema_version, embedding_fingerprint, dimension),
            )
            connection.commit()
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise EmbeddingCacheError("new embedding cache failed integrity_check")
            connection.close()
            connection = None
            os.replace(temporary, path)
            return cls.open(
                project_root=root,
                cache_path=path,
                expected_schema_version=schema_version,
                expected_embedding_fingerprint=embedding_fingerprint,
                expected_dimension=dimension,
                busy_timeout_seconds=busy_timeout_seconds,
            )
        finally:
            if connection is not None:
                connection.close()
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def open(
        cls,
        *,
        project_root: Path,
        cache_path: Path,
        expected_schema_version: str,
        expected_embedding_fingerprint: str,
        expected_dimension: int,
        busy_timeout_seconds: float,
    ) -> SqliteEmbeddingCache:
        root = resolve_project_root(project_root)
        path = resolve_project_path(root, cache_path, must_exist=True)
        if not path.is_file() or path.is_symlink():
            raise EmbeddingCacheError("embedding cache must be a regular file")
        if (
            not expected_schema_version
            or _SHA256_PATTERN.fullmatch(expected_embedding_fingerprint) is None
        ):
            raise EmbeddingCacheError("expected cache identity is required")
        if expected_dimension <= 0 or busy_timeout_seconds <= 0:
            raise EmbeddingCacheError("cache dimension and timeout must be positive")
        connection = sqlite3.connect(
            path,
            timeout=busy_timeout_seconds,
            check_same_thread=False,
        )
        try:
            connection.execute("PRAGMA query_only=ON")
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise EmbeddingCacheError("embedding cache integrity_check failed")
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            }
            if tables != {"ambiguous_content", "cache_metadata", "embeddings"}:
                raise EmbeddingCacheError("embedding cache table inventory mismatch")
            row = connection.execute(
                "SELECT schema_version, embedding_fingerprint, dimension "
                "FROM cache_metadata WHERE singleton=1"
            ).fetchone()
            expected = (
                expected_schema_version,
                expected_embedding_fingerprint,
                expected_dimension,
            )
            if row != expected:
                raise EmbeddingCacheError("embedding cache identity mismatch")
            connection.execute("PRAGMA query_only=OFF")
            return cls(
                connection=connection,
                path=path,
                schema_version=expected_schema_version,
                embedding_fingerprint=expected_embedding_fingerprint,
                dimension=expected_dimension,
            )
        except BaseException:
            connection.close()
            raise

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

    def __enter__(self) -> SqliteEmbeddingCache:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise EmbeddingCacheError("embedding cache is closed")

    def get_many(self, texts: Sequence[str]) -> tuple[list[float] | None, ...]:
        identities = tuple(_text_identity(text) for text in texts)
        with self._lock:
            self._require_open()
            output: list[list[float] | None] = []
            for content_sha256, content_chars in identities:
                row = self._connection.execute(
                    "SELECT content_chars, vector_blob, vector_sha256 "
                    "FROM embeddings WHERE content_sha256=?",
                    (content_sha256,),
                ).fetchone()
                if row is None:
                    output.append(None)
                    continue
                if row[0] != content_chars:
                    raise EmbeddingCacheError("cached text identity length mismatch")
                output.append(
                    _decode_vector(
                        row[1],
                        expected_sha256=row[2],
                        dimension=self.dimension,
                    )
                )
            return tuple(output)

    def put_many(
        self,
        *,
        texts: Sequence[str],
        vectors: Sequence[object],
        origin: EmbeddingCacheOrigin,
    ) -> None:
        if origin not in {"flat_baseline", "provider"}:
            raise EmbeddingCacheError("embedding cache origin is invalid")
        if len(texts) != len(vectors) or not texts:
            raise EmbeddingCacheError(
                "cache insert cardinality must be nonzero and equal"
            )
        rows = []
        for text, vector in zip(texts, vectors, strict=True):
            content_sha256, content_chars = _text_identity(text)
            payload, vector_sha256 = _encode_vector(
                vector,
                dimension=self.dimension,
            )
            rows.append(
                (
                    content_sha256,
                    content_chars,
                    payload,
                    vector_sha256,
                    origin,
                )
            )
        with self._lock:
            self._require_open()
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                for row in rows:
                    if origin == "provider":
                        self._connection.execute(
                            "DELETE FROM ambiguous_content WHERE content_sha256=?",
                            (row[0],),
                        )
                    existing = self._connection.execute(
                        "SELECT content_chars, vector_blob, vector_sha256 "
                        "FROM embeddings WHERE content_sha256=?",
                        (row[0],),
                    ).fetchone()
                    if existing is not None:
                        if existing != row[1:4]:
                            raise EmbeddingCacheError(
                                "duplicate cache text has a different vector"
                            )
                        continue
                    self._connection.execute(
                        "INSERT INTO embeddings VALUES (?, ?, ?, ?, ?)",
                        row,
                    )
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    def seed_flat_many(
        self,
        *,
        texts: Sequence[str],
        vectors: Sequence[object],
    ) -> None:
        """Seed only unambiguous Flat vectors; conflicting text hashes stay misses."""

        if len(texts) != len(vectors) or not texts:
            raise EmbeddingCacheError("Flat seed cardinality must be nonzero and equal")
        rows = []
        for text, vector in zip(texts, vectors, strict=True):
            content_sha256, content_chars = _text_identity(text)
            payload, vector_sha256 = _encode_vector(
                vector,
                dimension=self.dimension,
            )
            rows.append((content_sha256, content_chars, payload, vector_sha256))
        with self._lock:
            self._require_open()
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                for row in rows:
                    ambiguous = self._connection.execute(
                        "SELECT content_chars FROM ambiguous_content "
                        "WHERE content_sha256=?",
                        (row[0],),
                    ).fetchone()
                    if ambiguous is not None:
                        if ambiguous[0] != row[1]:
                            raise EmbeddingCacheError(
                                "ambiguous Flat text identity length mismatch"
                            )
                        self._connection.execute(
                            "UPDATE ambiguous_content "
                            "SET conflicting_row_count=conflicting_row_count+1 "
                            "WHERE content_sha256=?",
                            (row[0],),
                        )
                        continue
                    existing = self._connection.execute(
                        "SELECT content_chars, vector_blob, vector_sha256 "
                        "FROM embeddings WHERE content_sha256=?",
                        (row[0],),
                    ).fetchone()
                    if existing is None:
                        self._connection.execute(
                            "INSERT INTO embeddings VALUES (?, ?, ?, ?, "
                            "'flat_baseline')",
                            row,
                        )
                        continue
                    if existing == row[1:4]:
                        continue
                    self._connection.execute(
                        "DELETE FROM embeddings WHERE content_sha256=?",
                        (row[0],),
                    )
                    self._connection.execute(
                        "INSERT INTO ambiguous_content VALUES (?, ?, 2)",
                        (row[0], row[1]),
                    )
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    def counts_by_origin(self) -> dict[str, int]:
        with self._lock:
            self._require_open()
            rows = self._connection.execute(
                "SELECT origin, COUNT(*) FROM embeddings GROUP BY origin ORDER BY origin"
            ).fetchall()
            return {str(origin): int(count) for origin, count in rows}

    def ambiguous_counts(self) -> tuple[int, int]:
        with self._lock:
            self._require_open()
            row = self._connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(conflicting_row_count), 0) "
                "FROM ambiguous_content"
            ).fetchone()
            if row is None:
                raise EmbeddingCacheError("ambiguous cache count query failed")
            return int(row[0]), int(row[1])

    def verify_integrity(self) -> None:
        """Validate every cached vector before it can influence a generation."""

        with self._lock:
            self._require_open()
            if self._connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise EmbeddingCacheError("embedding cache integrity_check failed")
            cursor = self._connection.execute(
                "SELECT content_sha256, content_chars, vector_blob, vector_sha256, "
                "origin FROM embeddings ORDER BY content_sha256"
            )
            for content_sha256, content_chars, payload, vector_sha256, origin in cursor:
                if (
                    not isinstance(content_sha256, str)
                    or _SHA256_PATTERN.fullmatch(content_sha256) is None
                    or not isinstance(content_chars, int)
                    or content_chars <= 0
                    or origin not in {"flat_baseline", "provider"}
                ):
                    raise EmbeddingCacheError("embedding cache row identity is invalid")
                _decode_vector(
                    payload,
                    expected_sha256=vector_sha256,
                    dimension=self.dimension,
                )
            ambiguous_cursor = self._connection.execute(
                "SELECT content_sha256, content_chars, conflicting_row_count "
                "FROM ambiguous_content ORDER BY content_sha256"
            )
            for content_sha256, content_chars, conflict_count in ambiguous_cursor:
                if (
                    not isinstance(content_sha256, str)
                    or _SHA256_PATTERN.fullmatch(content_sha256) is None
                    or not isinstance(content_chars, int)
                    or content_chars <= 0
                    or not isinstance(conflict_count, int)
                    or conflict_count < 2
                ):
                    raise EmbeddingCacheError(
                        "ambiguous embedding cache row is invalid"
                    )


class CachingDocumentEmbeddingProvider:
    """Use exact validated cache hits and call one configured provider for misses."""

    def __init__(
        self,
        *,
        cache: SqliteEmbeddingCache,
        upstream: DocumentEmbeddingProvider,
    ) -> None:
        self._cache = cache
        self._upstream = upstream
        self._stats_lock = RLock()
        self.cache_hit_count = 0
        self.provider_text_count = 0
        self.provider_call_count = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list) or not texts:
            raise EmbeddingCacheError("embedding cache provider requires a text list")
        cached = self._cache.get_many(texts)
        missing_by_identity: dict[str, tuple[str, list[int]]] = {}
        for index, vector in enumerate(cached):
            if vector is not None:
                continue
            identity, _chars = _text_identity(texts[index])
            existing = missing_by_identity.get(identity)
            if existing is None:
                missing_by_identity[identity] = (texts[index], [index])
                continue
            if existing[0] != texts[index]:
                raise EmbeddingCacheError("SHA-256 collision in cache miss batch")
            existing[1].append(index)
        missing_texts = [item[0] for item in missing_by_identity.values()]
        provider_vectors: list[list[float]] = []
        if missing_texts:
            raw = self._upstream.embed_documents(missing_texts)
            if not isinstance(raw, list) or len(raw) != len(missing_texts):
                raise EmbeddingCacheError(
                    "upstream embedding result cardinality differs from cache misses"
                )
            provider_vectors = [
                _validate_vector(vector, dimension=self._cache.dimension)
                for vector in raw
            ]
            self._cache.put_many(
                texts=missing_texts,
                vectors=provider_vectors,
                origin="provider",
            )
        provider_by_index: dict[int, list[float]] = {}
        for (_identity, (_text, indexes)), vector in zip(
            missing_by_identity.items(),
            provider_vectors,
            strict=True,
        ):
            for index in indexes:
                provider_by_index[index] = vector
        output = [
            provider_by_index[index] if vector is None else vector
            for index, vector in enumerate(cached)
        ]
        with self._stats_lock:
            self.cache_hit_count += sum(vector is not None for vector in cached)
            self.provider_text_count += len(missing_texts)
            self.provider_call_count += int(bool(missing_texts))
        return output


__all__ = [
    "CachingDocumentEmbeddingProvider",
    "DocumentEmbeddingProvider",
    "EMBEDDING_CACHE_SCHEMA_VERSION",
    "EmbeddingCacheError",
    "EmbeddingCacheOrigin",
    "SqliteEmbeddingCache",
]
