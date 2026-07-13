"""Strict, redacted real-provider probes for local RAG build preflight.

The module deliberately records protocol facts only.  It never persists request
or response bodies, authorization headers, API keys, embedding values, or LLM
output text.  Callers can use :func:`run_provider_probe` directly or the CLI
wrapper in ``scripts/probe_rag_providers.py``.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import math
import os
from pathlib import Path
import time
from typing import Annotated, Literal, Protocol

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from src.config._rag_config import (
    NonBlankStr,
    RagConfigSecretError,
    StrictRagConfigModel,
    resolve_required_secret,
)
from src.config.rag_index_config import (
    BaseUrl,
    EmbeddingConfig,
    EndpointPath,
    PositiveFloat,
    RagIndexConfig,
    RerankerConfig,
    load_rag_index_config,
    resolve_rag_index_config_paths,
)
from src.rag.parent_child._storage_io import model_json_bytes
from src.rag.parent_child.project_paths import (
    atomic_write_project_bytes,
    require_project_file,
    resolve_project_path,
    resolve_project_root,
)
from src.rag.parent_child.provider_clients import (
    ProviderEmbeddingDimensionError,
    ProviderProtocolError,
    ProviderTransportError,
    StrictEmbeddingClient,
    StrictRerankerClient,
)
from src.rag.parent_child.retrieval import RerankCandidate, RerankScore


_PROBE_DOCUMENT = "Python 中的生成器可以按需产生元素。"
_PROBE_BATCH_DOCUMENT = "批量探测要求服务返回每个输入对应的向量。"
_PROBE_QUERY = "为什么生成器通常比列表节省内存？"
_RERANK_QUERY = "为什么 Python 生成器更节省内存？"
_RERANK_DOCUMENTS: tuple[str, ...] = (
    "生成器按需生成元素，不需要一次性保存全部结果。",
    "列表会在创建时把全部元素保存在内存中。",
    "HTTP 是一种应用层协议。",
)
_LLM_PROMPT = "请用一句话说明：为什么 RAG 在回答前需要先检索证据？"
_HIGH_CONSISTENCY_THRESHOLD = 0.999999
_LEGACY_EMBEDDING_PROBE_DOCUMENT_COUNT = 2

ApiKeyEnvironment = Annotated[
    NonBlankStr,
    Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$"),
]

ProbeStatus = Literal["success", "failed", "not_run", "not_requested"]
ProbeFailureType = Literal[
    "configuration",
    "missing_secret",
    "provider_transport",
    "provider_protocol",
    "llm_transport",
    "llm_protocol",
    "empty_llm_text",
    "blocked_by_prior_failure",
    "unexpected",
]


class ProviderProbeError(RuntimeError):
    """Base error for explicit provider-probe setup failures."""


class ProviderProbeValidationError(ProviderProbeError):
    """A successful transport response failed a probe invariant."""


class ChatProbeTransportError(ProviderProbeError):
    """The explicit chat endpoint did not return a success response."""


class ChatProbeProtocolError(ProviderProbeError):
    """The explicit chat endpoint returned an unsupported response schema."""


class EmptyChatCompletionError(ChatProbeProtocolError):
    """The chat response schema was valid but did not contain usable text."""


class _StrictResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class _ChatResponseMessage(_StrictResponseModel):
    role: Literal["assistant"]
    content: str | None
    reasoning_content: str | None = None
    refusal: str | None = None


class _ChatResponseChoice(_StrictResponseModel):
    index: int = Field(ge=0)
    message: _ChatResponseMessage
    finish_reason: str | None
    logprobs: None = None


class _ChatUsage(_StrictResponseModel):
    completion_tokens: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    prompt_cache_hit_tokens: int | None = Field(default=None, ge=0)
    prompt_cache_miss_tokens: int | None = Field(default=None, ge=0)


class _ChatResponse(_StrictResponseModel):
    id: str = Field(min_length=1)
    object: str = Field(min_length=1)
    created: int = Field(ge=0)
    model: str = Field(min_length=1)
    choices: list[_ChatResponseChoice] = Field(min_length=1)
    usage: _ChatUsage | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None


class _DeepSeekCompletionTokenDetails(_StrictResponseModel):
    """Strict DeepSeek completion-token metadata observed from the endpoint."""

    reasoning_tokens: int = Field(ge=0)


class _DeepSeekPromptTokenDetails(_StrictResponseModel):
    """Strict DeepSeek prompt-token metadata observed from the endpoint."""

    cached_tokens: int = Field(ge=0)


class _DeepSeekChatUsage(_StrictResponseModel):
    """DeepSeek-specific chat usage metadata with no undeclared fields."""

    completion_tokens: int = Field(ge=0)
    completion_tokens_details: _DeepSeekCompletionTokenDetails
    prompt_cache_hit_tokens: int = Field(ge=0)
    prompt_cache_miss_tokens: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    prompt_tokens_details: _DeepSeekPromptTokenDetails
    total_tokens: int = Field(ge=0)


class _DeepSeekChatResponse(_StrictResponseModel):
    """DeepSeek chat-completion contract observed by a real provider probe."""

    id: str = Field(min_length=1)
    object: str = Field(min_length=1)
    created: int = Field(ge=0)
    model: str = Field(min_length=1)
    choices: list[_ChatResponseChoice] = Field(min_length=1)
    usage: _DeepSeekChatUsage
    system_fingerprint: str | None = None


class ChatRequestMessage(StrictRagConfigModel):
    """One explicit OpenAI-compatible chat message for a probe or smoke test."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def _require_nonblank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chat message content must contain non-whitespace text")
        return value


class LlmProbeConfig(StrictRagConfigModel):
    """Fully explicit OpenAI-compatible chat configuration for local probes."""

    provider: NonBlankStr
    protocol: Literal["openai_chat_completions_v1", "deepseek_chat_completions_v1"]
    model: NonBlankStr
    base_url: BaseUrl
    endpoint_path: EndpointPath
    api_key_env: ApiKeyEnvironment
    timeout_seconds: PositiveFloat

    @model_validator(mode="after")
    def _validate_provider_specific_protocol(self) -> "LlmProbeConfig":
        if (
            self.protocol == "deepseek_chat_completions_v1"
            and self.provider != "deepseek"
        ):
            raise ValueError(
                "deepseek_chat_completions_v1 requires provider to be 'deepseek'"
            )
        return self


class ChatCompletionResult(StrictRagConfigModel):
    """Validated real chat completion retained only in the caller's memory."""

    model: NonBlankStr
    content: str = Field(min_length=1)
    http_status: int = Field(ge=200, le=299)

    @field_validator("content")
    @classmethod
    def _require_nonblank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chat completion content must contain non-whitespace text")
        return value


class EmbeddingProbeResult(StrictRagConfigModel):
    """Redacted evidence from strict embedding probe calls."""

    schema_version: Literal["embedding_provider_probe_v2"]
    status: ProbeStatus
    provider: NonBlankStr
    model: NonBlankStr
    endpoint_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    http_status: int | None = Field(ge=100, le=599)
    response_schema_valid: bool
    actual_dimension: int | None = Field(gt=0)
    document_batch_size: int = Field(ge=_LEGACY_EMBEDDING_PROBE_DOCUMENT_COUNT)
    batch_supported: bool
    input_type_supported: bool | None
    repeat_vector_similarity: float | None = Field(ge=-1.0, le=1.0)
    latency_ms: float | None = Field(ge=0.0)
    failure_type: ProbeFailureType | None

    @field_validator("repeat_vector_similarity", "latency_ms")
    @classmethod
    def _finite_optional_float(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("probe float fields must be finite")
        return value


class RerankerProbeResult(StrictRagConfigModel):
    """Redacted evidence from strict reranker probe calls."""

    schema_version: Literal["reranker_provider_probe_v1"]
    status: ProbeStatus
    provider: NonBlankStr
    model: NonBlankStr
    endpoint_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    http_status: int | None = Field(ge=100, le=599)
    response_schema_valid: bool
    returned_indices_complete_unique: bool
    score_min: float | None = Field(ge=0.0, le=1.0)
    score_max: float | None = Field(ge=0.0, le=1.0)
    relevant_documents_above_irrelevant: bool
    latency_ms: float | None = Field(ge=0.0)
    failure_type: ProbeFailureType | None

    @field_validator("score_min", "score_max", "latency_ms")
    @classmethod
    def _finite_optional_float(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("probe float fields must be finite")
        return value


class LlmProbeResult(StrictRagConfigModel):
    """Redacted evidence from an explicit real chat completion probe."""

    schema_version: Literal["llm_provider_probe_v1"]
    status: ProbeStatus
    provider: NonBlankStr | None
    model: NonBlankStr | None
    endpoint_identity_sha256: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    http_status: int | None = Field(ge=100, le=599)
    response_schema_valid: bool
    real_text_returned: bool
    output_sha256: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    latency_ms: float | None = Field(ge=0.0)
    failure_type: ProbeFailureType | None

    @field_validator("latency_ms")
    @classmethod
    def _finite_optional_float(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("probe float fields must be finite")
        return value


class ProviderProbeReport(StrictRagConfigModel):
    """Immutable, content-free report written for one provider probe run."""

    schema_version: Literal["rag_provider_probe_v2"]
    run_id: NonBlankStr
    success: bool
    failed_stage: Literal["embedding", "reranker", "llm"] | None
    embedding: EmbeddingProbeResult
    reranker: RerankerProbeResult
    llm: LlmProbeResult


class _EmbeddingClient(Protocol):
    @property
    def last_http_status(self) -> int | None: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def close(self) -> None: ...


class _RerankerClient(Protocol):
    @property
    def last_http_status(self) -> int | None: ...

    def rerank(
        self,
        *,
        query: str,
        candidates: tuple[RerankCandidate, ...],
    ) -> tuple[RerankScore, ...]: ...

    def close(self) -> None: ...


EmbeddingClientFactory = Callable[[EmbeddingConfig], _EmbeddingClient]
RerankerClientFactory = Callable[[RerankerConfig], _RerankerClient]


def _endpoint_identity(base_url: str, endpoint_path: str) -> str:
    """Hash endpoint coordinates so reports stay portable and content-free."""

    endpoint = base_url.rstrip("/") + endpoint_path
    return hashlib.sha256(endpoint.encode("utf-8")).hexdigest()


def _safe_http_status(client: object | None) -> int | None:
    status = getattr(client, "last_http_status", None)
    if isinstance(status, int) and 100 <= status <= 599:
        return status
    return None


def _latency_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000.0, 3)


def _probe_failure_type(error: BaseException, *, llm: bool) -> ProbeFailureType:
    if isinstance(error, RagConfigSecretError):
        return "missing_secret"
    if isinstance(error, (ProviderProtocolError, ProviderProbeValidationError)):
        return "provider_protocol"
    if isinstance(error, ProviderTransportError):
        return "provider_transport"
    if isinstance(error, ChatProbeProtocolError):
        return "llm_protocol"
    if isinstance(error, ChatProbeTransportError):
        return "llm_transport"
    if isinstance(error, ValueError):
        return "configuration"
    if llm:
        return "unexpected"
    return "unexpected"


def _assert_vector_contract(vector: list[float], *, expected_dimension: int) -> None:
    if len(vector) != expected_dimension:
        raise ProviderProbeValidationError("embedding dimension mismatch")
    if any(
        not isinstance(value, float) or not math.isfinite(value) for value in vector
    ):
        raise ProviderProbeValidationError(
            "embedding vector contains non-finite values"
        )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ProviderProbeValidationError("repeat vectors have different dimensions")
    if all(
        math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(left, right)
    ):
        return 1.0
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / denominator


def _new_production_embedding_client(config: EmbeddingConfig) -> StrictEmbeddingClient:
    return StrictEmbeddingClient.production(config=config)


def _new_production_reranker_client(config: RerankerConfig) -> StrictRerankerClient:
    return StrictRerankerClient.production(config=config)


def _resolve_embedding_probe_batch_size(
    *,
    config: EmbeddingConfig,
    embedding_probe_batch_size: int | None,
) -> int:
    """Validate an optional explicit probe batch without changing build config."""

    if embedding_probe_batch_size is None:
        return _LEGACY_EMBEDDING_PROBE_DOCUMENT_COUNT
    if isinstance(embedding_probe_batch_size, bool) or not isinstance(
        embedding_probe_batch_size, int
    ):
        raise ValueError("embedding probe batch size must be an integer")
    if embedding_probe_batch_size < _LEGACY_EMBEDDING_PROBE_DOCUMENT_COUNT:
        raise ValueError("embedding probe batch size must be at least two")
    if embedding_probe_batch_size > config.batch_size:
        raise ValueError(
            "embedding probe batch size exceeds configured embedding batch_size"
        )
    return embedding_probe_batch_size


def _embedding_probe_documents(*, document_batch_size: int) -> list[str]:
    """Build ephemeral unique probe inputs; callers never persist their text."""

    if document_batch_size == _LEGACY_EMBEDDING_PROBE_DOCUMENT_COUNT:
        return [_PROBE_DOCUMENT, _PROBE_BATCH_DOCUMENT]
    return [
        _PROBE_DOCUMENT,
        *(
            f"{_PROBE_BATCH_DOCUMENT} [{index}]"
            for index in range(1, document_batch_size)
        ),
    ]


def _embedding_not_run(
    config: EmbeddingConfig,
    *,
    document_batch_size: int,
    failure_type: ProbeFailureType | None,
) -> EmbeddingProbeResult:
    return EmbeddingProbeResult(
        schema_version="embedding_provider_probe_v2",
        status="not_run",
        provider=config.provider,
        model=config.model,
        endpoint_identity_sha256=_endpoint_identity(
            config.base_url, config.endpoint_path
        ),
        http_status=None,
        response_schema_valid=False,
        actual_dimension=None,
        document_batch_size=document_batch_size,
        batch_supported=False,
        input_type_supported=None,
        repeat_vector_similarity=None,
        latency_ms=None,
        failure_type=failure_type,
    )


def _reranker_not_run(
    config: RerankerConfig,
    *,
    failure_type: ProbeFailureType | None,
) -> RerankerProbeResult:
    return RerankerProbeResult(
        schema_version="reranker_provider_probe_v1",
        status="not_run",
        provider=config.provider,
        model=config.model,
        endpoint_identity_sha256=_endpoint_identity(
            config.base_url, config.endpoint_path
        ),
        http_status=None,
        response_schema_valid=False,
        returned_indices_complete_unique=False,
        score_min=None,
        score_max=None,
        relevant_documents_above_irrelevant=False,
        latency_ms=None,
        failure_type=failure_type,
    )


def _llm_not_run(
    config: LlmProbeConfig | None,
    *,
    status: ProbeStatus,
    failure_type: ProbeFailureType | None,
) -> LlmProbeResult:
    return LlmProbeResult(
        schema_version="llm_provider_probe_v1",
        status=status,
        provider=None if config is None else config.provider,
        model=None if config is None else config.model,
        endpoint_identity_sha256=(
            None
            if config is None
            else _endpoint_identity(config.base_url, config.endpoint_path)
        ),
        http_status=None,
        response_schema_valid=False,
        real_text_returned=False,
        output_sha256=None,
        latency_ms=None,
        failure_type=failure_type,
    )


def probe_embedding(
    *,
    config: EmbeddingConfig,
    embedding_probe_batch_size: int | None = None,
    client_factory: EmbeddingClientFactory = _new_production_embedding_client,
) -> EmbeddingProbeResult:
    """Make real strict embedding requests and return only redacted evidence."""

    document_batch_size = _resolve_embedding_probe_batch_size(
        config=config,
        embedding_probe_batch_size=embedding_probe_batch_size,
    )
    started_at = time.perf_counter()
    client: _EmbeddingClient | None = None
    try:
        client = client_factory(config)
        document_vectors = client.embed_documents(
            _embedding_probe_documents(document_batch_size=document_batch_size)
        )
        first_query_vector = client.embed_query(_PROBE_QUERY)
        repeated_query_vector = client.embed_query(_PROBE_QUERY)
        if len(document_vectors) != document_batch_size:
            raise ProviderProbeValidationError("embedding batch cardinality mismatch")
        for vector in (*document_vectors, first_query_vector, repeated_query_vector):
            _assert_vector_contract(
                vector, expected_dimension=config.expected_dimension
            )
        repeat_similarity = _cosine_similarity(
            first_query_vector, repeated_query_vector
        )
        if repeat_similarity < _HIGH_CONSISTENCY_THRESHOLD:
            raise ProviderProbeValidationError(
                "embedding repeat consistency is too low"
            )
        return EmbeddingProbeResult(
            schema_version="embedding_provider_probe_v2",
            status="success",
            provider=config.provider,
            model=config.model,
            endpoint_identity_sha256=_endpoint_identity(
                config.base_url, config.endpoint_path
            ),
            http_status=_safe_http_status(client),
            response_schema_valid=True,
            actual_dimension=len(first_query_vector),
            document_batch_size=document_batch_size,
            batch_supported=True,
            input_type_supported=(
                True if config.input_type_field is not None else None
            ),
            repeat_vector_similarity=repeat_similarity,
            latency_ms=_latency_ms(started_at),
            failure_type=None,
        )
    except Exception as exc:
        dimension_error = (
            exc if isinstance(exc, ProviderEmbeddingDimensionError) else None
        )
        return EmbeddingProbeResult(
            schema_version="embedding_provider_probe_v2",
            status="failed",
            provider=config.provider,
            model=config.model,
            endpoint_identity_sha256=_endpoint_identity(
                config.base_url, config.endpoint_path
            ),
            http_status=_safe_http_status(client),
            response_schema_valid=dimension_error is not None,
            actual_dimension=(
                None if dimension_error is None else dimension_error.actual_dimension
            ),
            document_batch_size=document_batch_size,
            batch_supported=dimension_error is not None,
            input_type_supported=(
                (True if dimension_error is not None else False)
                if config.input_type_field is not None
                else None
            ),
            repeat_vector_similarity=None,
            latency_ms=_latency_ms(started_at),
            failure_type=_probe_failure_type(exc, llm=False),
        )
    finally:
        if client is not None:
            client.close()


def probe_reranker(
    *,
    config: RerankerConfig,
    client_factory: RerankerClientFactory = _new_production_reranker_client,
) -> RerankerProbeResult:
    """Make one real strict reranker request and check its complete ranking."""

    started_at = time.perf_counter()
    client: _RerankerClient | None = None
    try:
        if config.batch_size < len(_RERANK_DOCUMENTS):
            raise ValueError("reranker batch_size must support three probe documents")
        client = client_factory(config)
        candidates = tuple(
            RerankCandidate(
                schema_version="rerank_candidate_v1",
                child_id=f"provider-probe-{index}",
                content=content,
            )
            for index, content in enumerate(_RERANK_DOCUMENTS)
        )
        scores = client.rerank(query=_RERANK_QUERY, candidates=candidates)
        score_by_id = {item.child_id: item.score for item in scores}
        expected_ids = {candidate.child_id for candidate in candidates}
        if len(scores) != len(expected_ids) or set(score_by_id) != expected_ids:
            raise ProviderProbeValidationError("reranker indices are incomplete")
        values = tuple(score_by_id[candidate.child_id] for candidate in candidates)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ProviderProbeValidationError("reranker scores are outside [0, 1]")
        if not (values[0] > values[2] and values[1] > values[2]):
            raise ProviderProbeValidationError("relevant reranker scores are too low")
        return RerankerProbeResult(
            schema_version="reranker_provider_probe_v1",
            status="success",
            provider=config.provider,
            model=config.model,
            endpoint_identity_sha256=_endpoint_identity(
                config.base_url, config.endpoint_path
            ),
            http_status=_safe_http_status(client),
            response_schema_valid=True,
            returned_indices_complete_unique=True,
            score_min=min(values),
            score_max=max(values),
            relevant_documents_above_irrelevant=True,
            latency_ms=_latency_ms(started_at),
            failure_type=None,
        )
    except Exception as exc:
        return RerankerProbeResult(
            schema_version="reranker_provider_probe_v1",
            status="failed",
            provider=config.provider,
            model=config.model,
            endpoint_identity_sha256=_endpoint_identity(
                config.base_url, config.endpoint_path
            ),
            http_status=_safe_http_status(client),
            response_schema_valid=False,
            returned_indices_complete_unique=False,
            score_min=None,
            score_max=None,
            relevant_documents_above_irrelevant=False,
            latency_ms=_latency_ms(started_at),
            failure_type=_probe_failure_type(exc, llm=False),
        )
    finally:
        if client is not None:
            client.close()


class StrictChatCompletionClient:
    """Small explicit OpenAI-compatible chat client without retry or fallback."""

    def __init__(
        self,
        *,
        config: LlmProbeConfig,
        api_key: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must be non-empty")
        self._config = config
        self._url = config.base_url.rstrip("/") + config.endpoint_path
        self._last_http_status: int | None = None
        self._client = httpx.Client(
            timeout=config.timeout_seconds,
            transport=transport,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    @classmethod
    def production(cls, *, config: LlmProbeConfig) -> "StrictChatCompletionClient":
        return cls(
            config=config,
            api_key=resolve_required_secret(config.api_key_env),
            transport=None,
        )

    @property
    def last_http_status(self) -> int | None:
        """Expose status only; raw response bodies are never retained."""

        return self._last_http_status

    def close(self) -> None:
        self._client.close()

    def complete(
        self,
        *,
        messages: tuple[ChatRequestMessage, ...],
    ) -> ChatCompletionResult:
        if not messages:
            raise ValueError("chat completion requires at least one message")
        payload = {
            "model": self._config.model,
            "messages": [message.model_dump(mode="json") for message in messages],
        }
        try:
            response = self._client.post(self._url, json=payload)
        except httpx.TransportError as exc:
            raise ChatProbeTransportError("chat request transport failed") from exc
        self._last_http_status = response.status_code
        if not 200 <= response.status_code < 300:
            raise ChatProbeTransportError("chat request returned non-success status")
        try:
            raw = response.json()
        except ValueError as exc:
            raise ChatProbeProtocolError("chat response is not valid JSON") from exc
        try:
            if self._config.protocol == "openai_chat_completions_v1":
                parsed = _ChatResponse.model_validate(raw)
            elif self._config.protocol == "deepseek_chat_completions_v1":
                parsed = _DeepSeekChatResponse.model_validate(raw)
            else:
                raise AssertionError("LlmProbeConfig must validate protocol")
        except ValidationError as exc:
            raise ChatProbeProtocolError("chat response failed strict schema") from exc
        if parsed.model != self._config.model:
            raise ChatProbeProtocolError("chat response model identity mismatch")
        first_choice = next(
            (choice for choice in parsed.choices if choice.index == 0),
            None,
        )
        if first_choice is None:
            raise ChatProbeProtocolError("chat response is missing choice index zero")
        content = first_choice.message.content
        if content is None or not content.strip():
            raise EmptyChatCompletionError("chat response content is empty")
        return ChatCompletionResult(
            model=parsed.model,
            content=content,
            http_status=response.status_code,
        )


def _new_production_chat_client(config: LlmProbeConfig) -> StrictChatCompletionClient:
    """Adapt the keyword-only production constructor to the probe factory API."""

    return StrictChatCompletionClient.production(config=config)


def probe_llm(
    *,
    config: LlmProbeConfig,
    client_factory: Callable[[LlmProbeConfig], StrictChatCompletionClient] = (
        _new_production_chat_client
    ),
) -> LlmProbeResult:
    """Call the explicit chat model once and retain only a text digest in reports."""

    started_at = time.perf_counter()
    client: StrictChatCompletionClient | None = None
    try:
        client = client_factory(config)
        completion = client.complete(
            messages=(ChatRequestMessage(role="user", content=_LLM_PROMPT),)
        )
        return LlmProbeResult(
            schema_version="llm_provider_probe_v1",
            status="success",
            provider=config.provider,
            model=completion.model,
            endpoint_identity_sha256=_endpoint_identity(
                config.base_url, config.endpoint_path
            ),
            http_status=completion.http_status,
            response_schema_valid=True,
            real_text_returned=True,
            output_sha256=hashlib.sha256(
                completion.content.encode("utf-8")
            ).hexdigest(),
            latency_ms=_latency_ms(started_at),
            failure_type=None,
        )
    except Exception as exc:
        return LlmProbeResult(
            schema_version="llm_provider_probe_v1",
            status="failed",
            provider=config.provider,
            model=config.model,
            endpoint_identity_sha256=_endpoint_identity(
                config.base_url, config.endpoint_path
            ),
            http_status=_safe_http_status(client),
            response_schema_valid=False,
            real_text_returned=False,
            output_sha256=None,
            latency_ms=_latency_ms(started_at),
            failure_type=(
                "empty_llm_text"
                if isinstance(exc, EmptyChatCompletionError)
                else _probe_failure_type(exc, llm=True)
            ),
        )
    finally:
        if client is not None:
            client.close()


def _map_openrouter_key_for_current_process(config: RagIndexConfig) -> None:
    """Apply only the user-approved ephemeral OpenRouter RAG-key mapping."""

    shared_key = os.environ.get("OPENROUTER_API_KEY")
    if shared_key is None or not shared_key.strip():
        return
    if (
        config.embedding.provider == "openrouter"
        and config.embedding.api_key_env == "RAG_EMBEDDING_API_KEY"
    ):
        os.environ.setdefault("RAG_EMBEDDING_API_KEY", shared_key)
    if (
        config.reranker.provider == "openrouter"
        and config.reranker.api_key_env == "RAG_RERANKER_API_KEY"
    ):
        os.environ.setdefault("RAG_RERANKER_API_KEY", shared_key)


def write_provider_probe_report(
    *,
    project_root: Path,
    output_directory: Path,
    report: ProviderProbeReport,
) -> Path:
    """Atomically write exactly one redacted provider probe report below root."""

    root = resolve_project_root(project_root)
    directory = resolve_project_path(root, output_directory, must_exist=False)
    output = directory / "provider_probe.json"
    return atomic_write_project_bytes(
        root,
        output,
        model_json_bytes(report),
        overwrite=False,
    )


def run_provider_probe(
    *,
    project_root: Path,
    index_config_path: Path,
    run_id: str,
    output_directory: Path,
    embedding_probe_batch_size: int | None = None,
    probe_llm_enabled: bool,
    llm_config: LlmProbeConfig | None,
    embedding_client_factory: EmbeddingClientFactory = _new_production_embedding_client,
    reranker_client_factory: RerankerClientFactory = _new_production_reranker_client,
    llm_client_factory: Callable[[LlmProbeConfig], StrictChatCompletionClient] = (
        _new_production_chat_client
    ),
) -> ProviderProbeReport:
    """Run ordered probes, persist a safe report, and never invoke a fallback.

    A provider failure stops later network probes.  The returned report carries
    the typed failure; the CLI maps ``success=False`` to a non-zero exit code.
    """

    root = resolve_project_root(project_root)
    config_path = require_project_file(root, index_config_path)
    config = resolve_rag_index_config_paths(
        load_rag_index_config(config_path), project_root=root
    )
    _resolve_embedding_probe_batch_size(
        config=config.embedding,
        embedding_probe_batch_size=embedding_probe_batch_size,
    )
    _map_openrouter_key_for_current_process(config)

    embedding = probe_embedding(
        config=config.embedding,
        embedding_probe_batch_size=embedding_probe_batch_size,
        client_factory=embedding_client_factory,
    )
    if embedding.status != "success":
        report = ProviderProbeReport(
            schema_version="rag_provider_probe_v2",
            run_id=run_id,
            success=False,
            failed_stage="embedding",
            embedding=embedding,
            reranker=_reranker_not_run(
                config.reranker,
                failure_type="blocked_by_prior_failure",
            ),
            llm=_llm_not_run(
                llm_config,
                status="not_run" if probe_llm_enabled else "not_requested",
                failure_type=(
                    "blocked_by_prior_failure" if probe_llm_enabled else None
                ),
            ),
        )
        write_provider_probe_report(
            project_root=root,
            output_directory=output_directory,
            report=report,
        )
        return report

    reranker = probe_reranker(
        config=config.reranker,
        client_factory=reranker_client_factory,
    )
    if reranker.status != "success":
        report = ProviderProbeReport(
            schema_version="rag_provider_probe_v2",
            run_id=run_id,
            success=False,
            failed_stage="reranker",
            embedding=embedding,
            reranker=reranker,
            llm=_llm_not_run(
                llm_config,
                status="not_run" if probe_llm_enabled else "not_requested",
                failure_type=(
                    "blocked_by_prior_failure" if probe_llm_enabled else None
                ),
            ),
        )
        write_provider_probe_report(
            project_root=root,
            output_directory=output_directory,
            report=report,
        )
        return report

    if not probe_llm_enabled:
        report = ProviderProbeReport(
            schema_version="rag_provider_probe_v2",
            run_id=run_id,
            success=True,
            failed_stage=None,
            embedding=embedding,
            reranker=reranker,
            llm=_llm_not_run(
                None,
                status="not_requested",
                failure_type=None,
            ),
        )
        write_provider_probe_report(
            project_root=root,
            output_directory=output_directory,
            report=report,
        )
        return report

    if llm_config is None:
        report = ProviderProbeReport(
            schema_version="rag_provider_probe_v2",
            run_id=run_id,
            success=False,
            failed_stage="llm",
            embedding=embedding,
            reranker=reranker,
            llm=_llm_not_run(
                None,
                status="failed",
                failure_type="configuration",
            ),
        )
        write_provider_probe_report(
            project_root=root,
            output_directory=output_directory,
            report=report,
        )
        return report

    llm = probe_llm(config=llm_config, client_factory=llm_client_factory)
    report = ProviderProbeReport(
        schema_version="rag_provider_probe_v2",
        run_id=run_id,
        success=llm.status == "success",
        failed_stage=None if llm.status == "success" else "llm",
        embedding=embedding,
        reranker=reranker,
        llm=llm,
    )
    write_provider_probe_report(
        project_root=root,
        output_directory=output_directory,
        report=report,
    )
    return report


__all__ = [
    "ChatCompletionResult",
    "ChatProbeProtocolError",
    "ChatProbeTransportError",
    "ChatRequestMessage",
    "EmbeddingProbeResult",
    "LlmProbeConfig",
    "LlmProbeResult",
    "ProviderProbeError",
    "ProviderProbeReport",
    "RerankerProbeResult",
    "StrictChatCompletionClient",
    "probe_embedding",
    "probe_llm",
    "probe_reranker",
    "run_provider_probe",
    "write_provider_probe_report",
]
