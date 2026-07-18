from __future__ import annotations

import json

import httpx
import pytest

from src.config.rag_index_config import EmbeddingConfig, RerankerConfig, RetryConfig
from src.rag.parent_child.provider_clients import (
    ProviderEmbeddingDimensionError,
    ProviderProtocolError,
    ProviderReportedError,
    RerankerContractError,
    RerankerRecoveryBudgetError,
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
        response_model="configured-embedding",
        base_url="https://provider.invalid/v1",
        endpoint_path="/embeddings",
        api_key_env="TEST_EMBEDDING_KEY",
        timeout_seconds=5.0,
        retry=_retry(),
        batch_size=8,
        max_in_flight_batches=1,
        expected_dimension=2,
        distance_metric="cosine",
        normalization_contract="unit_length_v1",
        document_input_type="document",
        query_input_type="query",
        input_type_field="input_type",
        provider_routing=None,
    )


def _openrouter_embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="openrouter",
        protocol="openrouter_embeddings_v1",
        model="configured-embedding",
        response_model="configured-openrouter-response-model",
        base_url="https://provider.invalid/v1",
        endpoint_path="/embeddings",
        api_key_env="TEST_EMBEDDING_KEY",
        timeout_seconds=5.0,
        retry=_retry(),
        batch_size=8,
        max_in_flight_batches=1,
        expected_dimension=2,
        distance_metric="cosine",
        normalization_contract="provider_output_as_is_v1",
        document_input_type="document",
        query_input_type="query",
        input_type_field=None,
        provider_routing={
            "order": ["parasail"],
            "allow_fallbacks": False,
        },
    )


def _reranker_config() -> RerankerConfig:
    return RerankerConfig(
        provider="configured-vendor",
        model="configured-reranker",
        response_model="configured-reranker",
        base_url="https://provider.invalid/v1",
        endpoint_path="/rerank",
        api_key_env="TEST_RERANKER_KEY",
        timeout_seconds=5.0,
        retry=_retry(),
        batch_size=4,
        batch_recovery={
            "schema_version": "reranker_batch_recovery_v1",
            "mode": "strict_bisect_v1",
            "max_total_requests": 9,
            "max_split_depth": 2,
            "min_batch_size": 1,
            "max_response_bytes": 4096,
        },
        protocol="ranked_index_scores_v1",
        score_min=0.0,
        score_max=1.0,
        provider_routing=None,
    )


def _openrouter_reranker_config() -> RerankerConfig:
    return RerankerConfig(
        provider="openrouter",
        model="configured-reranker-request",
        response_model="configured-reranker-response",
        base_url="https://provider.invalid/v1",
        endpoint_path="/rerank",
        api_key_env="TEST_RERANKER_KEY",
        timeout_seconds=5.0,
        retry=_retry(),
        batch_size=4,
        batch_recovery={
            "schema_version": "reranker_batch_recovery_v1",
            "mode": "strict_bisect_v1",
            "max_total_requests": 9,
            "max_split_depth": 2,
            "min_batch_size": 1,
            "max_response_bytes": 4096,
        },
        protocol="openrouter_ranked_index_scores_v1",
        score_min=0.0,
        score_max=1.0,
        provider_routing={
            "order": ["nvidia"],
            "allow_fallbacks": False,
        },
    )


def _auth_sentinel(marker: object) -> str:
    """Return a non-credential, per-test authentication protocol sentinel."""

    return f"provider-client-{id(marker)}"


def test_embedding_client_validates_identity_dimension_and_input_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {auth_sentinel}"
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

    auth_sentinel = _auth_sentinel(handler)
    client = StrictEmbeddingClient(
        config=_embedding_config(),
        api_key=auth_sentinel,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )
    try:
        assert client.embed_query("alpha") == [1.0, 0.0]
        assert client.last_http_status == 200
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
        api_key=_auth_sentinel(handler),
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


def test_provider_retries_strict_retryable_error_envelope() -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                json={
                    "error": {
                        "code": 503,
                        "message": "temporary provider failure",
                        "metadata": {},
                    }
                },
            )
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
        api_key=_auth_sentinel(handler),
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )
    try:
        assert client.embed_documents(["alpha"]) == [[1.0, 0.0]]
    finally:
        client.close()
    assert calls == 2
    assert sleeps == [0.1]


def test_provider_does_not_retry_deterministic_error_envelope() -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"error": {"code": 400, "message": "input token contract failed"}},
        )

    client = StrictEmbeddingClient(
        config=_embedding_config(),
        api_key=_auth_sentinel(handler),
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )
    try:
        with pytest.raises(ProviderReportedError) as error:
            client.embed_documents(["alpha"])
        assert error.value.code == 400
        assert error.value.retryable is False
        assert error.value.attempts_exhausted is False
    finally:
        client.close()
    assert calls == 1
    assert sleeps == []


def test_provider_reports_retryable_error_envelope_exhaustion() -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"error": {"code": 503, "message": "temporary provider failure"}},
        )

    client = StrictEmbeddingClient(
        config=_embedding_config(),
        api_key=_auth_sentinel(handler),
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )
    try:
        with pytest.raises(ProviderReportedError) as error:
            client.embed_documents(["alpha"])
        assert error.value.code == 503
        assert error.value.retryable is True
        assert error.value.attempts_exhausted is True
    finally:
        client.close()
    assert calls == 2
    assert sleeps == [0.1]


def test_embedding_dimension_mismatch_retains_actual_dimension_for_audit() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "configured-embedding",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [1.0, 0.0, 0.0]}
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    auth_sentinel = _auth_sentinel(handler)
    client = StrictEmbeddingClient(
        config=_embedding_config(),
        api_key=auth_sentinel,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )
    try:
        with pytest.raises(ProviderEmbeddingDimensionError) as error:
            client.embed_query("alpha")
        assert error.value.actual_dimension == 3
        assert error.value.expected_dimension == 2
        assert client.last_http_status == 200
    finally:
        client.close()


def test_openrouter_embedding_protocol_requires_explicit_metadata_and_no_input_type() -> (
    None
):
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload == {
            "model": "configured-embedding",
            "input": ["alpha"],
            "provider": {
                "order": ["parasail"],
                "allow_fallbacks": False,
            },
        }
        return httpx.Response(
            200,
            json={
                "id": "embedding-request-id",
                "object": "list",
                "model": "configured-openrouter-response-model",
                "provider": "configured-openrouter-provider",
                "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}],
                "usage": {
                    "prompt_tokens": 1,
                    "total_tokens": 1,
                    "cost": 0,
                    "cost_details": {
                        "upstream_inference_completions_cost": 0,
                        "upstream_inference_cost": 0,
                        "upstream_inference_prompt_cost": 0,
                    },
                    "is_byok": False,
                },
            },
        )

    client = StrictEmbeddingClient(
        config=_openrouter_embedding_config(),
        api_key=_auth_sentinel(handler),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )
    try:
        assert client.embed_query("alpha") == [1.0, 0.0]
    finally:
        client.close()


def test_openrouter_embedding_protocol_rejects_undeclared_metadata() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "embedding-request-id",
                "object": "list",
                "model": "configured-openrouter-response-model",
                "provider": "configured-openrouter-provider",
                "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}],
                "usage": {
                    "prompt_tokens": 1,
                    "total_tokens": 1,
                    "cost": 0,
                    "cost_details": {
                        "upstream_inference_completions_cost": 0,
                        "upstream_inference_cost": 0,
                        "upstream_inference_prompt_cost": 0,
                    },
                    "is_byok": False,
                },
                "undeclared_metadata": "must fail",
            },
        )

    client = StrictEmbeddingClient(
        config=_openrouter_embedding_config(),
        api_key=_auth_sentinel(handler),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )
    try:
        with pytest.raises(ProviderProtocolError, match="strict schema"):
            client.embed_query("alpha")
    finally:
        client.close()


def test_openrouter_embedding_protocol_rejects_wrong_provider_or_input_type() -> None:
    kwargs = _openrouter_embedding_config().model_dump(mode="python")
    kwargs["provider"] = "configured-vendor"
    with pytest.raises(ValueError, match="requires provider"):
        EmbeddingConfig(**kwargs)

    kwargs = _openrouter_embedding_config().model_dump(mode="python")
    kwargs["input_type_field"] = "input_type"
    with pytest.raises(ValueError, match="requires input_type_field"):
        EmbeddingConfig(**kwargs)


def test_reranker_requires_complete_unique_index_set() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "provider" not in json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.8},
                ]
            },
        )

    client = StrictRerankerClient(
        config=_reranker_config(),
        api_key=_auth_sentinel(handler),
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
        with pytest.raises(RerankerContractError) as exc_info:
            client.rerank(query="query", candidates=candidates)
        assert exc_info.value.reason_code == "duplicate_scores"
    finally:
        client.close()


def test_openrouter_reranker_protocol_requires_declared_metadata_and_identity() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "configured-reranker-request"
        assert "top_n" in payload
        assert payload["provider"] == {
            "order": ["nvidia"],
            "allow_fallbacks": False,
        }
        return httpx.Response(
            200,
            json={
                "id": "rerank-request-id",
                "model": "configured-reranker-response",
                "provider": "configured-openrouter-provider",
                "results": [
                    {
                        "index": 0,
                        "relevance_score": 0.9,
                        "document": {"text": "alpha"},
                    },
                    {
                        "index": 1,
                        "relevance_score": 0.1,
                        "document": {"text": "beta"},
                    },
                ],
                "usage": {"cost": 0, "total_tokens": 2},
            },
        )

    client = StrictRerankerClient(
        config=_openrouter_reranker_config(),
        api_key=_auth_sentinel(handler),
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
        scores = client.rerank(query="query", candidates=candidates)
    finally:
        client.close()

    assert tuple(score.child_id for score in scores) == ("child-a", "child-b")
    assert tuple(score.score for score in scores) == (0.9, 0.1)


def _rerank_candidates(count: int = 4) -> tuple[RerankCandidate, ...]:
    return tuple(
        RerankCandidate(
            schema_version="rerank_candidate_v1",
            child_id=f"child-{index}",
            content=f"body-{index}",
        )
        for index in range(count)
    )


def _complete_results(count: int) -> list[dict[str, int | float]]:
    return [
        {"index": index, "relevance_score": (count - index) / (count + 1)}
        for index in range(count)
    ]


@pytest.mark.parametrize(
    ("failure_mode", "reason_code", "expected_request_count"),
    [
        ("timeout", "timeout", 4),
        ("response_too_large", "response_too_large", 3),
        ("batch_protocol_failure", "batch_protocol_failure", 3),
        ("batch_incomplete", "batch_incomplete", 3),
    ],
)
def test_reranker_bisects_recoverable_batches_and_returns_every_real_score(
    failure_mode: str,
    reason_code: str,
    expected_request_count: int,
) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        count = len(payload["documents"])
        calls.append(count)
        if count > 2 and failure_mode == "timeout":
            raise httpx.ReadTimeout("fixture timeout", request=request)
        if count > 2 and failure_mode == "response_too_large":
            content = json.dumps({"results": _complete_results(count)}).encode()
            return httpx.Response(200, content=content + b" " * 2048)
        if count > 2 and failure_mode == "batch_protocol_failure":
            return httpx.Response(
                200,
                json={"results": "invalid-batch-envelope"},
            )
        if count > 2 and failure_mode == "batch_incomplete":
            return httpx.Response(
                200,
                json={"results": _complete_results(count - 1)},
            )
        return httpx.Response(
            200,
            json={"results": _complete_results(count)},
        )

    config_payload = _reranker_config().model_dump(mode="python")
    config_payload["batch_recovery"]["max_response_bytes"] = 1024
    config = RerankerConfig.model_validate(config_payload)
    auth_sentinel = _auth_sentinel(handler)
    client = StrictRerankerClient(
        config=config,
        api_key=auth_sentinel,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )
    try:
        scores = client.rerank(
            query="query-sensitive",
            candidates=_rerank_candidates(),
        )
        trace = client.last_recovery_trace
    finally:
        client.close()

    assert tuple(score.child_id for score in scores) == tuple(
        f"child-{index}" for index in range(4)
    )
    assert all(0.0 <= score.score <= 1.0 for score in scores)
    assert trace is not None
    assert trace.candidate_count == 4
    assert trace.scored_candidate_count == 4
    assert trace.provider_request_count == expected_request_count
    assert trace.split_count == 1
    assert trace.reason_codes == (reason_code,)
    serialized_trace = trace.model_dump_json()
    for forbidden in (
        "query-sensitive",
        "body-0",
        "child-0",
        auth_sentinel,
        "results",
    ):
        assert forbidden not in serialized_trace


@pytest.mark.parametrize(
    ("results", "reason_code"),
    [
        (
            [
                {"index": 0, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.8},
            ],
            "duplicate_scores",
        ),
        (
            [
                {"index": 0, "relevance_score": 1.5},
                {"index": 1, "relevance_score": 0.2},
            ],
            "score_invalid",
        ),
        (
            [
                {"index": 0, "relevance_score": 0.9},
                {"index": 9, "relevance_score": 0.2},
            ],
            "index_invalid",
        ),
    ],
)
def test_reranker_never_splits_ambiguous_or_illegal_scores(
    results: list[dict[str, int | float]],
    reason_code: str,
) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"results": results})

    client = StrictRerankerClient(
        config=_reranker_config(),
        api_key=_auth_sentinel(handler),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )
    try:
        with pytest.raises(RerankerContractError) as exc_info:
            client.rerank(query="query", candidates=_rerank_candidates(2))
    finally:
        client.close()

    assert exc_info.value.reason_code == reason_code
    assert calls == 1


def test_reranker_request_budget_exhaustion_is_typed_and_returns_no_partial_scores() -> (
    None
):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("fixture timeout", request=request)

    payload = _reranker_config().model_dump(mode="python")
    payload["batch_recovery"]["max_total_requests"] = 3
    config = RerankerConfig.model_validate(payload)
    client = StrictRerankerClient(
        config=config,
        api_key=_auth_sentinel(handler),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )
    try:
        with pytest.raises(RerankerRecoveryBudgetError) as exc_info:
            client.rerank(query="query", candidates=_rerank_candidates())
    finally:
        client.close()

    assert exc_info.value.reason_code == "budget_exhausted"
    assert exc_info.value.trace.provider_request_count == 3
    assert exc_info.value.trace.scored_candidate_count == 0
    assert calls == 3


def test_openrouter_reranker_rejects_candidate_echo_identity_drift() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "id": "rerank-request-id",
                "model": "configured-reranker-response",
                "provider": "configured-openrouter-provider",
                "results": [
                    {
                        "index": 0,
                        "relevance_score": 0.9,
                        "document": {"text": "different-body"},
                    },
                    {
                        "index": 1,
                        "relevance_score": 0.1,
                        "document": {"text": "body-1"},
                    },
                ],
                "usage": {"cost": 0, "total_tokens": 2},
            },
        )

    client = StrictRerankerClient(
        config=_openrouter_reranker_config(),
        api_key=_auth_sentinel(handler),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )
    try:
        with pytest.raises(RerankerContractError) as exc_info:
            client.rerank(query="query", candidates=_rerank_candidates(2))
    finally:
        client.close()

    assert exc_info.value.reason_code == "candidate_identity_mismatch"
    assert calls == 1
