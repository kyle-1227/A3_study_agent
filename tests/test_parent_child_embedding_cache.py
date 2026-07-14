from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from src.rag.parent_child.embedding_cache import (
    CachingDocumentEmbeddingProvider,
    EMBEDDING_CACHE_SCHEMA_VERSION,
    EmbeddingCacheError,
    SqliteEmbeddingCache,
)
from src.rag.parent_child.project_paths import ProjectPathError


class _Embedding:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text)), 2.0, 3.0] for text in texts]


class _FailingEmbedding:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        del texts
        raise RuntimeError("provider failed")


def _create_cache(tmp_path: Path) -> SqliteEmbeddingCache:
    return SqliteEmbeddingCache.create(
        project_root=tmp_path,
        cache_path=Path("artifacts/embedding_cache.sqlite"),
        schema_version=EMBEDDING_CACHE_SCHEMA_VERSION,
        embedding_fingerprint="a" * 64,
        dimension=3,
        busy_timeout_seconds=2.0,
    )


def test_exact_cache_hits_preserve_order_and_only_request_misses(
    tmp_path: Path,
) -> None:
    with _create_cache(tmp_path) as cache:
        cache.put_many(
            texts=["alpha"],
            vectors=[[1.0, 2.0, 3.0]],
            origin="flat_baseline",
        )
        upstream = _Embedding()
        provider = CachingDocumentEmbeddingProvider(cache=cache, upstream=upstream)

        result = provider.embed_documents(["alpha", "beta", "alpha"])

        assert result == [[1.0, 2.0, 3.0], [4.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        assert upstream.calls == [["beta"]]
        assert provider.cache_hit_count == 2
        assert provider.provider_text_count == 1
        assert provider.provider_call_count == 1
        assert cache.counts_by_origin() == {"flat_baseline": 1, "provider": 1}


def test_cache_identity_mismatch_fails_open(tmp_path: Path) -> None:
    cache = _create_cache(tmp_path)
    path = cache.path
    cache.close()

    with pytest.raises(EmbeddingCacheError, match="identity mismatch"):
        SqliteEmbeddingCache.open(
            project_root=tmp_path,
            cache_path=path,
            expected_schema_version=EMBEDDING_CACHE_SCHEMA_VERSION,
            expected_embedding_fingerprint="b" * 64,
            expected_dimension=3,
            busy_timeout_seconds=2.0,
        )


def test_corrupt_cached_vector_digest_fails_read(tmp_path: Path) -> None:
    cache = _create_cache(tmp_path)
    cache.put_many(
        texts=["alpha"],
        vectors=[[1.0, 2.0, 3.0]],
        origin="provider",
    )
    path = cache.path
    cache.close()
    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE embeddings SET vector_sha256=?",
        ("0" * 64,),
    )
    connection.commit()
    connection.close()

    with SqliteEmbeddingCache.open(
        project_root=tmp_path,
        cache_path=path,
        expected_schema_version=EMBEDDING_CACHE_SCHEMA_VERSION,
        expected_embedding_fingerprint="a" * 64,
        expected_dimension=3,
        busy_timeout_seconds=2.0,
    ) as reopened:
        with pytest.raises(EmbeddingCacheError, match="digest mismatch"):
            reopened.get_many(["alpha"])


def test_provider_failure_does_not_create_cache_success(tmp_path: Path) -> None:
    with _create_cache(tmp_path) as cache:
        provider = CachingDocumentEmbeddingProvider(
            cache=cache,
            upstream=_FailingEmbedding(),
        )
        with pytest.raises(RuntimeError, match="provider failed"):
            provider.embed_documents(["missing"])
        assert cache.counts_by_origin() == {}


def test_ambiguous_flat_vector_is_miss_and_provider_resolves_once(
    tmp_path: Path,
) -> None:
    with _create_cache(tmp_path) as cache:
        cache.seed_flat_many(texts=["same"], vectors=[[1.0, 2.0, 3.0]])
        cache.seed_flat_many(texts=["same"], vectors=[[9.0, 8.0, 7.0]])
        assert cache.get_many(["same"]) == (None,)
        assert cache.ambiguous_counts() == (1, 2)
        upstream = _Embedding()
        provider = CachingDocumentEmbeddingProvider(cache=cache, upstream=upstream)

        result = provider.embed_documents(["same", "same"])

        assert result == [[4.0, 2.0, 3.0], [4.0, 2.0, 3.0]]
        assert upstream.calls == [["same"]]
        assert cache.ambiguous_counts() == (0, 0)
        assert cache.counts_by_origin() == {"provider": 1}


def test_cache_path_must_remain_inside_project_root(tmp_path: Path) -> None:
    with pytest.raises(ProjectPathError, match="inside project_root"):
        SqliteEmbeddingCache.create(
            project_root=tmp_path,
            cache_path=tmp_path.parent / "outside.sqlite",
            schema_version=EMBEDDING_CACHE_SCHEMA_VERSION,
            embedding_fingerprint="a" * 64,
            dimension=3,
            busy_timeout_seconds=2.0,
        )
