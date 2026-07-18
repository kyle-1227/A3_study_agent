"""Strict HTTP embedding and reranker adapters with bounded same-provider retry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import json
import math
import time
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config._rag_config import resolve_required_secret
from src.config.rag_index_config import (
    EmbeddingConfig,
    RerankerConfig,
    RetryConfig,
)
from src.rag.parent_child.retrieval import RerankCandidate, RerankScore


RerankerReasonCode = Literal[
    "timeout",
    "response_too_large",
    "batch_protocol_failure",
    "batch_incomplete",
    "duplicate_scores",
    "index_invalid",
    "score_invalid",
    "candidate_identity_mismatch",
    "provider_identity_mismatch",
    "budget_exhausted",
]


class ProviderClientError(RuntimeError):
    """Base error for strict provider transport and protocol failures."""


class ProviderTransportError(ProviderClientError):
    """The configured endpoint failed after its explicit retry policy."""


class ProviderTimeoutError(ProviderTransportError):
    """The configured endpoint timed out after its explicit HTTP retry policy."""


class ProviderResponseTooLargeError(ProviderClientError):
    """A provider response exceeded the configured reranker byte ceiling."""


class RerankerRecoveryTrace(BaseModel):
    """Body-free counters and fingerprints for one reranker invocation."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["reranker_recovery_trace_v1"]
    candidate_count: int = Field(gt=0)
    scored_candidate_count: int = Field(ge=0)
    provider_request_count: int = Field(ge=0)
    split_count: int = Field(ge=0)
    max_split_depth_observed: int = Field(ge=0)
    candidate_identity_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    recovery_policy_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason_codes: tuple[RerankerReasonCode, ...]


class RerankerRecoveryError(ProviderClientError):
    """A bounded recovery could not prove a complete score set."""

    def __init__(
        self,
        *,
        reason_code: RerankerReasonCode,
        trace: RerankerRecoveryTrace,
    ) -> None:
        self.reason_code = reason_code
        self.trace = trace
        super().__init__(f"reranker recovery failed: {reason_code}")


class RerankerRecoveryBudgetError(RerankerRecoveryError):
    """The explicit total provider-request budget was exhausted."""


class RerankerRecoveryExhaustedError(RerankerRecoveryError):
    """The configured split depth or minimum batch size was exhausted."""


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


class RerankerContractError(ProviderProtocolError):
    """A reranker response cannot preserve score or candidate identity."""

    def __init__(self, reason_code: RerankerReasonCode) -> None:
        self.reason_code = reason_code
        super().__init__(f"reranker contract failed: {reason_code}")


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


class _RecoverableBatchError(RuntimeError):
    def __init__(self, reason_code: RerankerReasonCode) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


class _RequestBudgetExhausted(RuntimeError):
    pass


@dataclass(slots=True)
class _RerankerRecoveryState:
    candidate_ids: tuple[str, ...]
    max_total_requests: int
    policy_fingerprint: str
    provider_request_count: int = 0
    split_count: int = 0
    max_split_depth_observed: int = 0
    scored_candidate_ids: set[str] = field(default_factory=set)
    reason_codes: list[RerankerReasonCode] = field(default_factory=list)

    def consume_request(self) -> None:
        if self.provider_request_count >= self.max_total_requests:
            self.note("budget_exhausted")
            raise _RequestBudgetExhausted
        self.provider_request_count += 1

    def note(self, reason_code: RerankerReasonCode) -> None:
        if reason_code not in self.reason_codes:
            self.reason_codes.append(reason_code)

    def trace(self) -> RerankerRecoveryTrace:
        return RerankerRecoveryTrace(
            schema_version="reranker_recovery_trace_v1",
            candidate_count=len(self.candidate_ids),
            scored_candidate_count=len(self.scored_candidate_ids),
            provider_request_count=self.provider_request_count,
            split_count=self.split_count,
            max_split_depth_observed=self.max_split_depth_observed,
            candidate_identity_fingerprint=_fingerprint(self.candidate_ids),
            recovery_policy_fingerprint=self.policy_fingerprint,
            reason_codes=tuple(self.reason_codes),
        )


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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

    def post_json(
        self,
        payload: dict[str, object],
        *,
        before_attempt: Callable[[], None] | None = None,
        max_response_bytes: int | None = None,
    ) -> object:
        backoff = self._retry.initial_backoff_seconds
        last_reason = "unknown"
        for attempt in range(1, self._retry.max_attempts + 1):
            retryable = False
            try:
                if before_attempt is not None:
                    before_attempt()
                response = self._client.post(self._url, json=payload)
            except httpx.TimeoutException:
                last_reason = "timeout"
                retryable = True
            except httpx.TransportError as exc:
                last_reason = type(exc).__name__
                retryable = True
            else:
                self._last_http_status = response.status_code
                if response.status_code == 413:
                    raise ProviderResponseTooLargeError(
                        "provider rejected an oversized reranker batch"
                    )
                if 200 <= response.status_code < 300:
                    if (
                        max_response_bytes is not None
                        and len(response.content) > max_response_bytes
                    ):
                        raise ProviderResponseTooLargeError(
                            "provider response exceeded configured byte ceiling"
                        )
                    try:
                        decoded = response.json()
                    except ValueError as exc:
                        raise ProviderProtocolError(
                            "provider response is not valid JSON"
                        ) from exc
                    provider_error_code = _strict_provider_error_code(decoded)
                    if provider_error_code is None:
                        return decoded
                    if provider_error_code == 413:
                        raise ProviderResponseTooLargeError(
                            "provider reported an oversized reranker batch"
                        )
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
                if last_reason == "timeout":
                    raise ProviderTimeoutError(
                        "provider timed out after explicit retry policy"
                    )
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
        self._last_recovery_trace: RerankerRecoveryTrace | None = None
        self._recovery_policy_fingerprint = _fingerprint(
            {
                "schema_version": "reranker_recovery_identity_v1",
                "provider": config.provider,
                "model": config.model,
                "response_model": config.response_model,
                "protocol": config.protocol,
                "endpoint": f"{config.base_url.rstrip('/')}{config.endpoint_path}",
                "batch_recovery": config.batch_recovery.model_dump(mode="json"),
            }
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

    @property
    def last_recovery_trace(self) -> RerankerRecoveryTrace | None:
        """Return only body-free counters and fingerprints for the latest call."""

        return self._last_recovery_trace

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
        state = _RerankerRecoveryState(
            candidate_ids=child_ids,
            max_total_requests=self._config.batch_recovery.max_total_requests,
            policy_fingerprint=self._recovery_policy_fingerprint,
        )
        try:
            scores = self._rerank_partition(
                query=query,
                candidates=candidates,
                depth=0,
                state=state,
            )
        except _RequestBudgetExhausted as exc:
            self._last_recovery_trace = state.trace()
            raise RerankerRecoveryBudgetError(
                reason_code="budget_exhausted",
                trace=self._last_recovery_trace,
            ) from exc
        except Exception:
            self._last_recovery_trace = state.trace()
            raise
        returned_ids = tuple(score.child_id for score in scores)
        if len(returned_ids) != len(set(returned_ids)):
            state.note("duplicate_scores")
            self._last_recovery_trace = state.trace()
            raise RerankerContractError("duplicate_scores")
        if returned_ids != child_ids or state.scored_candidate_ids != set(child_ids):
            state.note("batch_incomplete")
            self._last_recovery_trace = state.trace()
            raise RerankerContractError("batch_incomplete")
        self._last_recovery_trace = state.trace()
        return scores

    def _rerank_partition(
        self,
        *,
        query: str,
        candidates: tuple[RerankCandidate, ...],
        depth: int,
        state: _RerankerRecoveryState,
    ) -> tuple[RerankScore, ...]:
        state.max_split_depth_observed = max(
            state.max_split_depth_observed,
            depth,
        )
        try:
            return self._rerank_once(
                query=query,
                candidates=candidates,
                state=state,
            )
        except ProviderTimeoutError as exc:
            reason_code: RerankerReasonCode = "timeout"
            cause: Exception = exc
        except ProviderResponseTooLargeError as exc:
            reason_code = "response_too_large"
            cause = exc
        except _RecoverableBatchError as exc:
            reason_code = exc.reason_code
            cause = exc
        state.note(reason_code)
        recovery = self._config.batch_recovery
        midpoint = len(candidates) // 2
        can_split = (
            depth < recovery.max_split_depth
            and midpoint >= recovery.min_batch_size
            and len(candidates) - midpoint >= recovery.min_batch_size
        )
        if not can_split:
            raise RerankerRecoveryExhaustedError(
                reason_code=reason_code,
                trace=state.trace(),
            ) from cause
        state.split_count += 1
        left = self._rerank_partition(
            query=query,
            candidates=candidates[:midpoint],
            depth=depth + 1,
            state=state,
        )
        right = self._rerank_partition(
            query=query,
            candidates=candidates[midpoint:],
            depth=depth + 1,
            state=state,
        )
        return left + right

    def _rerank_once(
        self,
        *,
        query: str,
        candidates: tuple[RerankCandidate, ...],
        state: _RerankerRecoveryState,
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
        raw = self._http.post_json(
            payload,
            before_attempt=state.consume_request,
            max_response_bytes=self._config.batch_recovery.max_response_bytes,
        )
        try:
            if self._config.protocol == "ranked_index_scores_v1":
                response = _RerankerResponse.model_validate(raw)
            elif self._config.protocol == "openrouter_ranked_index_scores_v1":
                response = _OpenRouterRerankerResponse.model_validate(raw)
                if response.model != self._config.response_model:
                    state.note("provider_identity_mismatch")
                    raise RerankerContractError("provider_identity_mismatch")
            else:
                raise AssertionError("RerankerConfig must validate protocol")
        except ValidationError as exc:
            locations = tuple(error["loc"] for error in exc.errors())
            if any(
                location and location[-1] == "relevance_score" for location in locations
            ):
                state.note("score_invalid")
                raise RerankerContractError("score_invalid") from exc
            if any(location and location[-1] == "index" for location in locations):
                state.note("index_invalid")
                raise RerankerContractError("index_invalid") from exc
            raise _RecoverableBatchError("batch_protocol_failure") from exc
        indices = tuple(result.index for result in response.results)
        if len(indices) != len(set(indices)):
            state.note("duplicate_scores")
            raise RerankerContractError("duplicate_scores")
        expected_indices = set(range(len(candidates)))
        if not set(indices).issubset(expected_indices):
            state.note("index_invalid")
            raise RerankerContractError("index_invalid")
        if set(indices) != expected_indices:
            raise _RecoverableBatchError("batch_incomplete")
        if isinstance(response, _OpenRouterRerankerResponse):
            for result in response.results:
                if result.document.text != candidates[result.index].content:
                    state.note("candidate_identity_mismatch")
                    raise RerankerContractError("candidate_identity_mismatch")
        by_index = {result.index: result.relevance_score for result in response.results}
        scores = tuple(
            RerankScore(
                schema_version="rerank_score_v1",
                child_id=candidate.child_id,
                score=by_index[index],
            )
            for index, candidate in enumerate(candidates)
        )
        state.scored_candidate_ids.update(score.child_id for score in scores)
        return scores


__all__ = [
    "ProviderClientError",
    "ProviderEmbeddingDimensionError",
    "ProviderProtocolError",
    "ProviderReportedError",
    "ProviderResponseTooLargeError",
    "ProviderTimeoutError",
    "ProviderTransportError",
    "RerankerContractError",
    "RerankerReasonCode",
    "RerankerRecoveryBudgetError",
    "RerankerRecoveryError",
    "RerankerRecoveryExhaustedError",
    "RerankerRecoveryTrace",
    "StrictEmbeddingClient",
    "StrictRerankerClient",
]
