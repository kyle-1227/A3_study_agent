"""Strict HTTP embedding and reranker adapters with bounded same-provider retry."""

from __future__ import annotations

from collections.abc import Callable
import math
import time
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config._rag_config import resolve_required_secret
from src.config.rag_index_config import EmbeddingConfig, RerankerConfig, RetryConfig
from src.rag.parent_child.retrieval import RerankCandidate, RerankScore


class ProviderClientError(RuntimeError):
    """Base error for strict provider transport and protocol failures."""


class ProviderTransportError(ProviderClientError):
    """The configured endpoint failed after its explicit retry policy."""


class ProviderReportedError(ProviderTransportError):
    """The configured provider returned a strict error envelope."""

    def __init__(
        self,
        *,
        code: int,
        retryable: bool,
        attempts_exhausted: bool,
    ) -> None:
        self.code = code
        self.retryable = retryable
        self.attempts_exhausted = attempts_exhausted
        super().__init__("provider returned an explicit error envelope")


class ProviderProtocolError(ProviderClientError):
    """The configured endpoint returned a response outside its strict schema."""


class ProviderEmbeddingDimensionError(ProviderProtocolError):
    """A schema-valid embedding response has the wrong configured dimension."""

    def __init__(self, *, actual_dimension: int, expected_dimension: int) -> None:
        self.actual_dimension = actual_dimension
        self.expected_dimension = expected_dimension
        super().__init__("embedding vector dimension contract mismatch")


class _StrictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class _EmbeddingUsage(_StrictResponse):
    prompt_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class _EmbeddingData(_StrictResponse):
    object: Literal["embedding"]
    index: int = Field(ge=0)
    embedding: list[float] = Field(min_length=1)


class _EmbeddingResponse(_StrictResponse):
    object: Literal["list"]
    model: str = Field(min_length=1)
    data: list[_EmbeddingData]
    usage: _EmbeddingUsage


class _OpenRouterEmbeddingCostDetails(_StrictResponse):
    """Observed OpenRouter embedding usage detail contract as of this protocol."""

    upstream_inference_completions_cost: int | float
    upstream_inference_cost: int | float
    upstream_inference_prompt_cost: int | float


class _OpenRouterEmbeddingUsage(_StrictResponse):
    """Strict OpenRouter-specific embedding usage metadata."""

    prompt_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cost: int | float
    cost_details: _OpenRouterEmbeddingCostDetails
    is_byok: bool


class _OpenRouterEmbeddingResponse(_StrictResponse):
    """Observed OpenRouter embedding response with declared metadata fields."""

    id: str = Field(min_length=1)
    object: Literal["list"]
    model: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    data: list[_EmbeddingData]
    usage: _OpenRouterEmbeddingUsage


class _RerankerResult(_StrictResponse):
    index: int = Field(ge=0)
    relevance_score: float = Field(ge=0.0, le=1.0)


class _RerankerResponse(_StrictResponse):
    results: list[_RerankerResult]


class _OpenRouterRerankerDocument(_StrictResponse):
    """OpenRouter's returned reranked document echo."""

    text: str = Field(min_length=1)


class _OpenRouterRerankerResult(_StrictResponse):
    """OpenRouter's strictly declared rerank result row."""

    index: int = Field(ge=0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    document: _OpenRouterRerankerDocument


class _OpenRouterRerankerUsage(_StrictResponse):
    """Strict OpenRouter reranker usage metadata."""

    cost: int | float
    total_tokens: int = Field(ge=0)


class _OpenRouterRerankerResponse(_StrictResponse):
    """OpenRouter rerank envelope observed through a real provider probe."""

    id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    results: list[_OpenRouterRerankerResult]
    usage: _OpenRouterRerankerUsage


_RETRYABLE_PROVIDER_ERROR_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _strict_provider_error_code(payload: object) -> int | None:
    """Recognize only OpenRouter's exact, redacted error-envelope boundary."""

    if not isinstance(payload, dict) or set(payload) != {"error"}:
        return None
    error = payload.get("error")
    if not isinstance(error, dict) or set(error) not in (
        {"code", "message"},
        {"code", "message", "metadata"},
    ):
        return None
    code = error.get("code")
    message = error.get("message")
    if type(code) is not int or not isinstance(message, str) or not message:
        return None
    if "metadata" in error and not isinstance(error.get("metadata"), dict):
        return None
    return code


class _StrictJsonHttpClient:
    def __init__(
        self,
        *,
        base_url: str,
        endpoint_path: str,
        api_key: str,
        timeout_seconds: float,
        retry: RetryConfig,
        transport: httpx.BaseTransport | None,
        sleep: Callable[[float], None],
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must be non-empty")
        self._url = base_url.rstrip("/") + endpoint_path
        self._retry = retry
        self._sleep = sleep
        self._client = httpx.Client(
            timeout=timeout_seconds,
            transport=transport,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        self._last_http_status: int | None = None

    @property
    def last_http_status(self) -> int | None:
        """Return the last received HTTP status without retaining response bodies."""

        return self._last_http_status

    def close(self) -> None:
        self._client.close()

    def post_json(self, payload: dict[str, object]) -> object:
        backoff = self._retry.initial_backoff_seconds
        last_reason = "unknown"
        for attempt in range(1, self._retry.max_attempts + 1):
            retryable = False
            try:
                response = self._client.post(self._url, json=payload)
            except httpx.TransportError as exc:
                last_reason = type(exc).__name__
                retryable = True
            else:
                self._last_http_status = response.status_code
                if 200 <= response.status_code < 300:
                    try:
                        decoded = response.json()
                    except ValueError as exc:
                        raise ProviderProtocolError(
                            "provider response is not valid JSON"
                        ) from exc
                    provider_error_code = _strict_provider_error_code(decoded)
                    if provider_error_code is None:
                        return decoded
                    last_reason = f"provider_error_{provider_error_code}"
                    retryable = provider_error_code in _RETRYABLE_PROVIDER_ERROR_CODES
                    if not retryable:
                        raise ProviderReportedError(
                            code=provider_error_code,
                            retryable=False,
                            attempts_exhausted=False,
                        )
                    if attempt == self._retry.max_attempts:
                        raise ProviderReportedError(
                            code=provider_error_code,
                            retryable=True,
                            attempts_exhausted=True,
                        )
                else:
                    last_reason = f"http_{response.status_code}"
                    retryable = response.status_code in {408, 429} or (
                        500 <= response.status_code < 600
                    )
                    if not retryable:
                        raise ProviderTransportError(
                            f"provider request failed: {last_reason}"
                        )
            if not retryable or attempt == self._retry.max_attempts:
                raise ProviderTransportError(
                    "provider request exhausted explicit retry policy: " + last_reason
                )
            self._sleep(backoff)
            backoff = min(
                backoff * self._retry.multiplier,
                self._retry.max_backoff_seconds,
            )
        raise AssertionError("unreachable retry loop")


class StrictEmbeddingClient:
    """OpenAI-compatible embedding protocol with strict response validation."""

    def __init__(
        self,
        *,
        config: EmbeddingConfig,
        api_key: str,
        transport: httpx.BaseTransport | None,
        sleep: Callable[[float], None],
    ) -> None:
        if config.protocol not in {
            "openai_embeddings_v1",
            "openrouter_embeddings_v1",
        }:
            raise ValueError("unsupported embedding protocol")
        self._config = config
        self._http = _StrictJsonHttpClient(
            base_url=config.base_url,
            endpoint_path=config.endpoint_path,
            api_key=api_key,
            timeout_seconds=config.timeout_seconds,
            retry=config.retry,
            transport=transport,
            sleep=sleep,
        )

    @classmethod
    def from_environment(
        cls,
        *,
        config: EmbeddingConfig,
        transport: httpx.BaseTransport | None,
        sleep: Callable[[float], None],
    ) -> StrictEmbeddingClient:
        return cls(
            config=config,
            api_key=resolve_required_secret(config.api_key_env),
            transport=transport,
            sleep=sleep,
        )

    @classmethod
    def production(cls, *, config: EmbeddingConfig) -> StrictEmbeddingClient:
        return cls.from_environment(config=config, transport=None, sleep=time.sleep)

    def close(self) -> None:
        self._http.close()

    @property
    def last_http_status(self) -> int | None:
        """Expose only the most recent status code for auditable probes."""

        return self._http.last_http_status

    def _embed(self, texts: list[str], *, input_type: str) -> list[list[float]]:
        if not texts or any(not isinstance(text, str) or not text for text in texts):
            raise ProviderProtocolError("embedding inputs must be non-empty strings")
        payload: dict[str, object] = {
            "model": self._config.model,
            "input": texts,
        }
        if self._config.input_type_field is not None:
            payload[self._config.input_type_field] = input_type
        if self._config.protocol == "openrouter_embeddings_v1":
            provider_routing = self._config.provider_routing
            if provider_routing is None:
                raise ProviderProtocolError(
                    "openrouter embedding request lacks provider_routing"
                )
            payload["provider"] = {
                "order": list(provider_routing.order),
                "allow_fallbacks": provider_routing.allow_fallbacks,
            }
        raw = self._http.post_json(payload)
        try:
            if self._config.protocol == "openai_embeddings_v1":
                response = _EmbeddingResponse.model_validate(raw)
            elif self._config.protocol == "openrouter_embeddings_v1":
                response = _OpenRouterEmbeddingResponse.model_validate(raw)
            else:
                raise AssertionError("EmbeddingConfig must validate protocol")
        except ValidationError as exc:
            raise ProviderProtocolError(
                "embedding response failed strict schema validation"
            ) from exc
        if response.model != self._config.response_model:
            raise ProviderProtocolError("embedding response model identity mismatch")
        if len(response.data) != len(texts):
            raise ProviderProtocolError("embedding response cardinality mismatch")
        by_index = {item.index: item for item in response.data}
        if len(by_index) != len(response.data) or set(by_index) != set(
            range(len(texts))
        ):
            raise ProviderProtocolError("embedding response indices are invalid")
        vectors: list[list[float]] = []
        for index in range(len(texts)):
            vector = by_index[index].embedding
            if len(vector) != self._config.expected_dimension:
                raise ProviderEmbeddingDimensionError(
                    actual_dimension=len(vector),
                    expected_dimension=self._config.expected_dimension,
                )
            if any(not math.isfinite(coordinate) for coordinate in vector):
                raise ProviderProtocolError("embedding vector contract mismatch")
            vectors.append(vector)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, input_type=self._config.document_input_type)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], input_type=self._config.query_input_type)[0]


class StrictRerankerClient:
    """Index-addressed reranker protocol returning complete [0,1] scores."""

    def __init__(
        self,
        *,
        config: RerankerConfig,
        api_key: str,
        transport: httpx.BaseTransport | None,
        sleep: Callable[[float], None],
    ) -> None:
        if config.protocol not in {
            "ranked_index_scores_v1",
            "openrouter_ranked_index_scores_v1",
        }:
            raise ValueError("unsupported reranker protocol")
        self._config = config
        self._http = _StrictJsonHttpClient(
            base_url=config.base_url,
            endpoint_path=config.endpoint_path,
            api_key=api_key,
            timeout_seconds=config.timeout_seconds,
            retry=config.retry,
            transport=transport,
            sleep=sleep,
        )

    @classmethod
    def from_environment(
        cls,
        *,
        config: RerankerConfig,
        transport: httpx.BaseTransport | None,
        sleep: Callable[[float], None],
    ) -> StrictRerankerClient:
        return cls(
            config=config,
            api_key=resolve_required_secret(config.api_key_env),
            transport=transport,
            sleep=sleep,
        )

    @classmethod
    def production(cls, *, config: RerankerConfig) -> StrictRerankerClient:
        return cls.from_environment(config=config, transport=None, sleep=time.sleep)

    def close(self) -> None:
        self._http.close()

    @property
    def last_http_status(self) -> int | None:
        """Expose only the most recent status code for auditable probes."""

        return self._http.last_http_status

    def rerank(
        self,
        *,
        query: str,
        candidates: tuple[RerankCandidate, ...],
    ) -> tuple[RerankScore, ...]:
        if not query or not candidates:
            raise ProviderProtocolError("reranker query and candidates are required")
        if len(candidates) > self._config.batch_size:
            raise ProviderProtocolError("reranker candidate count exceeds batch_size")
        child_ids = tuple(candidate.child_id for candidate in candidates)
        if len(child_ids) != len(set(child_ids)):
            raise ProviderProtocolError("reranker candidate child IDs must be unique")
        payload: dict[str, object] = {
            "model": self._config.model,
            "query": query,
            "documents": [candidate.content for candidate in candidates],
            "top_n": len(candidates),
        }
        if self._config.protocol == "openrouter_ranked_index_scores_v1":
            provider_routing = self._config.provider_routing
            if provider_routing is None:
                raise ProviderProtocolError(
                    "openrouter reranker request lacks provider_routing"
                )
            payload["provider"] = {
                "order": list(provider_routing.order),
                "allow_fallbacks": provider_routing.allow_fallbacks,
            }
        raw = self._http.post_json(payload)
        try:
            if self._config.protocol == "ranked_index_scores_v1":
                response = _RerankerResponse.model_validate(raw)
            elif self._config.protocol == "openrouter_ranked_index_scores_v1":
                response = _OpenRouterRerankerResponse.model_validate(raw)
                if response.model != self._config.response_model:
                    raise ProviderProtocolError(
                        "reranker response model identity mismatch"
                    )
            else:
                raise AssertionError("RerankerConfig must validate protocol")
        except ValidationError as exc:
            raise ProviderProtocolError(
                "reranker response failed strict schema validation"
            ) from exc
        indices = tuple(result.index for result in response.results)
        if len(indices) != len(set(indices)) or set(indices) != set(
            range(len(candidates))
        ):
            raise ProviderProtocolError("reranker response indices are invalid")
        by_index = {result.index: result.relevance_score for result in response.results}
        return tuple(
            RerankScore(
                schema_version="rerank_score_v1",
                child_id=candidate.child_id,
                score=by_index[index],
            )
            for index, candidate in enumerate(candidates)
        )


__all__ = [
    "ProviderClientError",
    "ProviderEmbeddingDimensionError",
    "ProviderProtocolError",
    "ProviderReportedError",
    "ProviderTransportError",
    "StrictEmbeddingClient",
    "StrictRerankerClient",
]
