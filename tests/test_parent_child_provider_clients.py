from __future__ import annotations

import json

import httpx
import pytest

from src.config.rag_index_config import EmbeddingConfig, RerankerConfig, RetryConfig
from src.rag.parent_child.provider_clients import (
    ProviderProtocolError,
    StrictEmbeddingClient,
    StrictRerankerClient,
)
from src.rag.parent_child.retrieval import RerankCandidate


def _retry(max_attempts: int = 2) -> RetryConfig:
    return RetryConfig(
        max_attempts=max_attempts,
        initial_backoff_seconds=0.1,
        max_backoff_seconds=1.0,
        multiplier=2.0,
    )


def _embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="configured-vendor",
        protocol="openai_embeddings_v1",
        model="configured-embedding",
        base_url="https://provider.invalid/v1",
        endpoint_path="/embeddings",
        api_key_env="TEST_EMBEDDING_KEY",
        timeout_seconds=5.0,
        retry=_retry(),
        batch_size=8,
        expected_dimension=2,
        distance_metric="cosine",
        normalization_contract="unit_length_v1",
        document_input_type="document",
        query_input_type="query",
        input_type_field="input_type",
    )


def _reranker_config() -> RerankerConfig:
    return RerankerConfig(
        provider="configured-vendor",
        model="configured-reranker",
        base_url="https://provider.invalid/v1",
        endpoint_path="/rerank",
        api_key_env="TEST_RERANKER_KEY",
        timeout_seconds=5.0,
        retry=_retry(),
        batch_size=4,
        protocol="ranked_index_scores_v1",
        score_min=0.0,
        score_max=1.0,
    )


def test_embedding_client_validates_identity_dimension_and_input_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret"
        payload = json.loads(request.content)
        assert payload == {
            "model": "configured-embedding",
            "input": ["alpha"],
            "input_type": "query",
        }
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "configured-embedding",
                "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    client = StrictEmbeddingClient(
        config=_embedding_config(),
        api_key="secret",
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )
    try:
        assert client.embed_query("alpha") == [1.0, 0.0]
    finally:
        client.close()


def test_provider_retries_only_configured_endpoint_and_same_request() -> None:
    calls: list[str] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "configured-embedding",
                "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    client = StrictEmbeddingClient(
        config=_embedding_config(),
        api_key="secret",
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )
    try:
        assert client.embed_documents(["alpha"]) == [[1.0, 0.0]]
    finally:
        client.close()
    assert calls == [
        "https://provider.invalid/v1/embeddings",
        "https://provider.invalid/v1/embeddings",
    ]
    assert sleeps == [0.1]


def test_reranker_requires_complete_unique_index_set() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"results": [{"index": 0, "relevance_score": 0.9}]},
        )

    client = StrictRerankerClient(
        config=_reranker_config(),
        api_key="secret",
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )
    candidates = (
        RerankCandidate(
            schema_version="rerank_candidate_v1",
            child_id="child-a",
            content="alpha",
        ),
        RerankCandidate(
            schema_version="rerank_candidate_v1",
            child_id="child-b",
            content="beta",
        ),
    )
    try:
        with pytest.raises(ProviderProtocolError, match="indices"):
            client.rerank(query="query", candidates=candidates)
    finally:
        client.close()
