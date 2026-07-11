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


class ProviderProtocolError(ProviderClientError):
    """The configured endpoint returned a response outside its strict schema."""


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


class _RerankerResult(_StrictResponse):
    index: int = Field(ge=0)
    relevance_score: float = Field(ge=0.0, le=1.0)


class _RerankerResponse(_StrictResponse):
    results: list[_RerankerResult]


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
                if 200 <= response.status_code < 300:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise ProviderProtocolError(
                            "provider response is not valid JSON"
                        ) from exc
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
        if config.protocol != "openai_embeddings_v1":
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

    def _embed(self, texts: list[str], *, input_type: str) -> list[list[float]]:
        if not texts or any(not isinstance(text, str) or not text for text in texts):
            raise ProviderProtocolError("embedding inputs must be non-empty strings")
        payload: dict[str, object] = {
            "model": self._config.model,
            "input": texts,
        }
        if self._config.input_type_field is not None:
            payload[self._config.input_type_field] = input_type
        raw = self._http.post_json(payload)
        try:
            response = _EmbeddingResponse.model_validate(raw)
        except ValidationError as exc:
            raise ProviderProtocolError(
                "embedding response failed strict schema validation"
            ) from exc
        if response.model != self._config.model:
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
            if len(vector) != self._config.expected_dimension or any(
                not math.isfinite(coordinate) for coordinate in vector
            ):
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
        if config.protocol != "ranked_index_scores_v1":
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
        raw = self._http.post_json(
            {
                "model": self._config.model,
                "query": query,
                "documents": [candidate.content for candidate in candidates],
                "top_n": len(candidates),
            }
        )
        try:
            response = _RerankerResponse.model_validate(raw)
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
    "ProviderProtocolError",
    "ProviderTransportError",
    "StrictEmbeddingClient",
    "StrictRerankerClient",
]
