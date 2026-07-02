"""Fail-fast tests for memory embedding and retrieval."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.memory.embeddings import (
    get_embedding_provider,
    OpenAICompatibleMemoryEmbeddingProvider,
    reset_embedding_provider,
)
from src.memory.errors import MemoryEmbeddingConfigError, MemoryEmbeddingRuntimeError
from src.memory.retrieval import retrieve_top_k_memories
from src.memory.schema import EpisodicMemoryRecord
from src.memory.storage import SQLiteMemoryStore
from tests.fakes.embeddings import DeterministicFakeEmbeddingProvider


@pytest.fixture
def local_tmp_path():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        yield Path(tmpdir)


def _settings(values: dict[str, object]):
    def _get_setting(key: str, default=None):
        return values.get(key, default)

    return _get_setting


def test_dummy_provider_config_fails(monkeypatch):
    import src.memory.embeddings as embeddings

    reset_embedding_provider()
    monkeypatch.setattr(
        embeddings,
        "get_setting",
        _settings({"memory.embedding_provider": "dummy"}),
    )

    with pytest.raises(
        MemoryEmbeddingConfigError, match="Unsupported memory.embedding_provider"
    ):
        get_embedding_provider()


def test_missing_provider_config_fails(monkeypatch):
    import src.memory.embeddings as embeddings

    reset_embedding_provider()
    monkeypatch.setattr(embeddings, "get_setting", _settings({}))

    with pytest.raises(
        MemoryEmbeddingConfigError, match="memory.embedding_provider is required"
    ):
        get_embedding_provider()


def test_missing_api_key_fails_when_provider_is_constructed(monkeypatch):
    import src.memory.embeddings as embeddings

    reset_embedding_provider()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        embeddings,
        "get_setting",
        _settings(
            {
                "memory.embedding_provider": "openrouter",
                "memory.embedding.model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
                "memory.embedding.base_url": "https://openrouter.ai/api/v1",
                "memory.embedding.api_key_env": "OPENROUTER_API_KEY",
                "memory.embedding.timeout_seconds": 30,
            }
        ),
    )

    with pytest.raises(
        MemoryEmbeddingConfigError, match="OPENROUTER_API_KEY is not configured"
    ):
        get_embedding_provider()


def test_openrouter_provider_config_constructs_with_explicit_settings(monkeypatch):
    import src.memory.embeddings as embeddings

    reset_embedding_provider()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setattr(
        embeddings,
        "get_setting",
        _settings(
            {
                "memory.embedding_provider": "openrouter",
                "memory.embedding.model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
                "memory.embedding.base_url": "https://openrouter.ai/api/v1",
                "memory.embedding.api_key_env": "OPENROUTER_API_KEY",
                "memory.embedding.timeout_seconds": 30,
            }
        ),
    )

    provider = get_embedding_provider()

    assert isinstance(provider, OpenAICompatibleMemoryEmbeddingProvider)


@pytest.mark.anyio
async def test_embedding_api_exception_fails_fast():
    class FailingClient:
        async def post(self, *args, **kwargs):
            raise RuntimeError("network failed api_key=sk-secret-value")

    provider = OpenAICompatibleMemoryEmbeddingProvider(
        model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        timeout=30.0,
        api_key="test-key",
    )
    provider._client = FailingClient()

    with pytest.raises(MemoryEmbeddingRuntimeError) as exc_info:
        await provider.embed(["hello"])

    assert "network failed" in str(exc_info.value)
    assert "sk-secret-value" not in str(exc_info.value)


@pytest.mark.anyio
async def test_empty_memory_store_is_valid_empty_result(local_tmp_path: Path):
    store = SQLiteMemoryStore(db_path=local_tmp_path / "empty.db")
    provider = DeterministicFakeEmbeddingProvider()

    results = await retrieve_top_k_memories(
        user_id="missing-user",
        query="python",
        store=store,
        embedding_provider=provider,
    )

    assert results == []


@pytest.mark.anyio
async def test_db_query_failure_is_not_empty_result():
    class BrokenStore:
        async def get_all_episodic_for_user(self, user_id: str, limit: int = 200):
            raise RuntimeError("database is down")

        async def get_semantic_for_user(self, user_id: str, limit: int = 20):
            return []

    with pytest.raises(RuntimeError, match="database is down"):
        await retrieve_top_k_memories(
            user_id="u",
            query="python",
            store=BrokenStore(),
            embedding_provider=DeterministicFakeEmbeddingProvider(),
        )


@pytest.mark.anyio
async def test_fake_embedder_requires_explicit_injection(
    local_tmp_path: Path, monkeypatch
):
    import src.memory.retrieval as retrieval

    store = SQLiteMemoryStore(db_path=local_tmp_path / "memory.db")
    await store.save_episodic(
        EpisodicMemoryRecord(
            user_id="u",
            content="Python list comprehension doubles every item",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
    )
    monkeypatch.setattr(
        retrieval,
        "get_embedding_provider",
        lambda: (_ for _ in ()).throw(
            AssertionError("production factory should not be called")
        ),
    )

    results = await retrieve_top_k_memories(
        user_id="u",
        query="Python list comprehension",
        store=store,
        embedding_provider=DeterministicFakeEmbeddingProvider(),
    )

    assert results
    assert results[0].memory_type == "episodic"
