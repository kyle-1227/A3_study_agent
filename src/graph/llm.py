"""Central LLM factory and fallback invoke logic.

Provides a resilient invoke_with_fallback() that catches transient API errors
(timeouts, 502s, rate limits) and retries on a fallback model, recording the
failover event on the active OpenTelemetry span.
"""

from __future__ import annotations

import logging
import os
import asyncio
import time
import ssl
from typing import Any, Awaitable, Callable, Mapping, TypeVar

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
import httpx

from src.config import get_setting
from src.context_engineering.packing import (
    ContextImportanceError,
    ContextImportanceScores,
    emit_context_apply_policy_resolved_summary,
    parse_importance_scorer_output,
    prepare_messages_with_context_policy,
    should_emit_context_policy_summary,
)
from src.context_engineering.input_manifest import (
    llm_input_manifest_trace_payload,
    validate_llm_input_manifest,
)
from src.context_engineering.influence_runtime import (
    record_llm_input_influences,
    record_plain_output_influence,
)
from src.observability.a3_trace import emit_a3_trace
from src.observability.llm_input import (
    build_llm_input_observation,
    emit_context_usage_trace,
    raise_for_blocking_input_observation,
)
from src.context_engineering.schema import ContextConfigError, ContextUsageError

logger = logging.getLogger(__name__)
T = TypeVar("T")
_CONTEXT_POLICY_SUMMARY_EMITTED = False

DEFAULT_DEEPSEEK_PROVIDER = "deepseek_official"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"

# ---------------------------------------------------------------------------
# Recoverable errors that trigger automatic fallback
# ---------------------------------------------------------------------------

_FALLBACK_ERRORS: tuple[type[Exception], ...] = (TimeoutError, ConnectionError)

try:
    import openai

    _FALLBACK_ERRORS = (
        TimeoutError,
        ConnectionError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
        openai.RateLimitError,
    )
except ImportError:
    pass


# ---------------------------------------------------------------------------
# LLM factories
# ---------------------------------------------------------------------------


def get_node_llm(node_name: str, **overrides) -> ChatOpenAI:
    """Build a ChatOpenAI instance configured for a specific graph node.

    Reads per-node ``model``, ``base_url``, ``api_key_env``, and ``temperature``
    from ``settings.yaml``.  Falls back to ``DEEPSEEK_*`` env vars when a
    node has no explicit override in settings.
    """
    nested_prefix = f"llm.{node_name}"
    default_model = os.getenv("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL
    default_api_key_env = "DEEPSEEK_API_KEY"
    default_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    model = get_setting(
        f"{nested_prefix}.model",
        get_setting(f"{node_name}.model", default_model),
    )
    api_key_env = get_setting(
        f"{nested_prefix}.api_key_env",
        get_setting(f"{node_name}.api_key_env", default_api_key_env),
    )
    base_url = get_setting(
        f"{nested_prefix}.base_url",
        get_setting(f"{node_name}.base_url", default_base_url),
    )
    temperature = get_setting(
        f"{nested_prefix}.temperature",
        get_setting(f"{node_name}.temperature", 0.7),
    )
    max_tokens = get_setting(
        f"{nested_prefix}.max_tokens",
        get_setting(f"{node_name}.max_tokens", None),
    )
    streaming = get_setting(
        f"{nested_prefix}.streaming",
        get_setting(f"{node_name}.streaming", None),
    )

    defaults = dict(
        model=model,
        api_key=os.getenv(api_key_env),
        base_url=base_url,
        temperature=temperature,
    )
    if max_tokens is not None:
        defaults["max_tokens"] = max_tokens
    if streaming is not None:
        defaults["streaming"] = streaming
    if "openrouter.ai" in str(base_url).lower():
        headers = {}
        referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
        app_title = os.getenv("OPENROUTER_APP_TITLE", "").strip()
        if referer:
            headers["HTTP-Referer"] = referer
        if app_title:
            headers["X-Title"] = app_title
        if headers:
            defaults["default_headers"] = headers
    defaults.update(overrides)
    return ChatOpenAI(**defaults)


def get_primary_llm(**overrides) -> ChatOpenAI:
    """Build the primary chat model from DEEPSEEK_* env vars."""
    defaults = dict(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0.7,
    )
    defaults.update(overrides)
    return ChatOpenAI(**defaults)


def get_fallback_llm(**overrides) -> ChatOpenAI:
    """Build the fallback chat model from FALLBACK_* env vars.

    Defaults to the primary API config so that transient errors (502, timeout)
    get a second chance on the same endpoint.  Override ``FALLBACK_MODEL``,
    ``FALLBACK_API_KEY``, and ``FALLBACK_BASE_URL`` to point at a local Ollama
    instance or a different cloud provider.
    """
    defaults = dict(
        model=os.getenv(
            "FALLBACK_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
        ),
        api_key=os.getenv("FALLBACK_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or "not-configured",
        base_url=os.getenv(
            "FALLBACK_BASE_URL",
            os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        ),
        temperature=0.7,
    )
    defaults.update(overrides)
    return ChatOpenAI(**defaults)


# ---------------------------------------------------------------------------
# Resilient invoke
# ---------------------------------------------------------------------------


def get_llm_call_max_retries(node_name: str | None = None, default: int = 2) -> int:
    """Return the semantic retry budget for one LLM call.

    The value means "additional tries after the first attempt".
    """
    raw = None
    if node_name:
        raw = get_setting(f"llm_outputs.{node_name}.max_retries", None)
    if raw is None:
        raw = get_setting("llm_outputs.default.max_retries", default)
    try:
        value = int(raw)
    except Exception:
        value = default
    return max(0, min(3, value))


def _is_recoverable_llm_error(exc: BaseException) -> bool:
    return isinstance(exc, _FALLBACK_ERRORS) or _is_provider_transport_retryable(exc)


def _invoke_with_retries_sync(
    operation: Callable[[], T], *, max_retries: int, label: str
) -> tuple[T, int]:
    retry_count = 0
    while True:
        try:
            return operation(), retry_count
        except Exception as exc:
            if not _is_recoverable_llm_error(exc) or retry_count >= max_retries:
                raise
            retry_count += 1
            logger.warning(
                "%s retry %s/%s after %s: %s",
                label,
                retry_count,
                max_retries,
                type(exc).__name__,
                exc,
            )


async def _invoke_with_retries_async(
    operation: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    label: str,
) -> tuple[T, int]:
    retry_count = 0
    while True:
        try:
            return await operation(), retry_count
        except Exception as exc:
            if not _is_recoverable_llm_error(exc) or retry_count >= max_retries:
                raise
            retry_count += 1
            logger.warning(
                "%s retry %s/%s after %s: %s",
                label,
                retry_count,
                max_retries,
                type(exc).__name__,
                exc,
            )


def invoke_with_fallback(primary, messages, *, fallback=None, span=None):
    """Invoke *primary*; on recoverable error, failover to *fallback*.

    Args:
        primary: Primary ChatModel instance.
        messages: Message list passed to ``invoke()``.
        fallback: Optional fallback ChatModel. ``None`` → error propagates.
        span: Optional OTel span for recording fallback metadata.

    Returns:
        The LLM response from whichever model succeeded.

    Raises:
        The original error when no fallback is configured, or the fallback
        error when both models fail.
    """
    max_retries = get_llm_call_max_retries()
    try:
        response, retry_count = _invoke_with_retries_sync(
            lambda: primary.invoke(messages),
            max_retries=max_retries,
            label="Primary LLM",
        )
        if span is not None:
            span.set_attribute("llm.retry_count", retry_count)
            span.set_attribute("llm.fallback_used", False)
        return response
    except Exception as exc:
        if not _is_recoverable_llm_error(exc) or fallback is None:
            raise

        logger.warning(
            "Primary LLM failed (%s: %s), falling back",
            type(exc).__name__,
            exc,
        )

        if span is not None:
            span.set_attribute("llm.fallback_used", True)
            span.set_attribute(
                "llm.fallback_model",
                getattr(fallback, "model_name", "unknown"),
            )
            span.add_event(
                "llm.fallback_triggered",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

        response, fallback_retry_count = _invoke_with_retries_sync(
            lambda: fallback.invoke(messages),
            max_retries=max_retries,
            label="Fallback LLM",
        )
        if span is not None:
            span.set_attribute("llm.fallback_retry_count", fallback_retry_count)
        return response


async def async_invoke_with_fallback(primary, messages, *, fallback=None, span=None):
    """Async version of invoke_with_fallback; uses ainvoke() throughout.

    Args:
        primary: Primary ChatModel (or structured output chain) instance.
        messages: Message list passed to ``ainvoke()``.
        fallback: Optional fallback ChatModel. ``None`` → error propagates.
        span: Optional OTel span for recording fallback metadata.

    Returns:
        The LLM response from whichever model succeeded.

    Raises:
        The original error when no fallback is configured, or the fallback
        error when both models fail.
    """
    max_retries = get_llm_call_max_retries()
    try:
        response, retry_count = await _invoke_with_retries_async(
            lambda: primary.ainvoke(messages),
            max_retries=max_retries,
            label="Primary LLM",
        )
        if span is not None:
            span.set_attribute("llm.retry_count", retry_count)
            span.set_attribute("llm.fallback_used", False)
        return response
    except Exception as exc:
        if not _is_recoverable_llm_error(exc) or fallback is None:
            raise

        logger.warning(
            "Primary LLM failed (%s: %s), falling back",
            type(exc).__name__,
            exc,
        )

        if span is not None:
            span.set_attribute("llm.fallback_used", True)
            span.set_attribute(
                "llm.fallback_model",
                getattr(fallback, "model_name", "unknown"),
            )
            span.add_event(
                "llm.fallback_triggered",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

        response, fallback_retry_count = await _invoke_with_retries_async(
            lambda: fallback.ainvoke(messages),
            max_retries=max_retries,
            label="Fallback LLM",
        )
        if span is not None:
            span.set_attribute("llm.fallback_retry_count", fallback_retry_count)
        return response


def _message_content_chars(messages: list[Any]) -> int:
    total = 0
    for message in messages or []:
        if isinstance(message, BaseMessage):
            total += len(str(message.content or ""))
        elif isinstance(message, dict):
            total += len(str(message.get("content") or ""))
        else:
            total += len(str(message))
    return total


def _emit_llm_input_manifest_built(
    manifest: Mapping[str, Any],
    *,
    state: dict | None,
) -> None:
    emit_a3_trace(
        logger,
        "llm_input_manifest.built",
        llm_input_manifest_trace_payload(manifest),
        state=state or {},
        env_flag="LOG_A3_TRACE",
    )


def _emit_llm_input_manifest_failed(
    *,
    node_name: str,
    llm_node: str,
    state: dict | None,
    exc: BaseException,
) -> None:
    emit_a3_trace(
        logger,
        "llm_input_manifest.failed",
        {
            "node_name": node_name,
            "llm_node": llm_node,
            "reason": "manifest_build_or_validation_failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        },
        state=state or {},
        env_flag="LOG_A3_TRACE",
    )


def _provider_error_body(exc: BaseException, *, max_chars: int = 12000) -> str:
    """Return no raw provider body; provider responses may contain secrets."""
    _ = (exc, max_chars)
    return ""


def _extract_status_code(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    try:
        return int(status_code) if status_code is not None else None
    except Exception:
        return None


def _is_provider_transport_retryable(exc: BaseException) -> bool:
    if bool(getattr(exc, "retryable_provider_error", False)):
        return True
    status_code = _extract_status_code(exc)
    if status_code == 429 or (status_code is not None and 500 <= status_code <= 599):
        return True
    if isinstance(
        exc,
        (
            TimeoutError,
            ConnectionError,
            ssl.SSLError,
            httpx.TransportError,
            httpx.TimeoutException,
        ),
    ):
        return True
    retryable_type_names = {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "DeepSeekInsufficientSystemResourceError",
        "DeepSeekProviderResponseJSONError",
    }
    return type(exc).__name__ in retryable_type_names


def _provider_transport_max_retries(node_name: str | None = None) -> int:
    raw = None
    if node_name:
        raw = get_setting(f"llm_outputs.{node_name}.transport_max_retries", None)
    if raw is None:
        raw = get_setting("llm_outputs.default.transport_max_retries", None)
    if raw is None:
        raw = get_setting("provider_transport_retry.max_retries", 2)
    try:
        value = int(raw)
    except Exception:
        value = 2
    return max(1, min(3, value))


def _provider_transport_delay_seconds(attempt_index: int) -> float:
    raw = get_setting("provider_transport_retry.base_delay_seconds", 0.25)
    try:
        base = float(raw)
    except Exception:
        base = 0.25
    return max(0.0, base) * attempt_index


async def invoke_with_provider_transport_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    node_name: str,
    llm_node: str,
    provider: str,
    model: str,
    llm_input_manifest: Mapping[str, Any],
    output_mode: str = "",
    trace_stage_prefix: str = "provider_transport",
    state: dict | None = None,
) -> tuple[T, int]:
    """Retry transient provider transport failures without fallback.

    Retries only connection errors, timeouts, HTTP 429, and HTTP 5xx.
    The caller supplies the exact same operation each time, so model, prompt,
    schema, and request payload remain unchanged.
    """
    try:
        validate_llm_input_manifest(llm_input_manifest)
    except Exception as exc:
        _emit_llm_input_manifest_failed(
            node_name=node_name,
            llm_node=llm_node,
            state=state or {},
            exc=exc,
        )
        raise
    manifest_payload = llm_input_manifest_trace_payload(llm_input_manifest)
    emit_a3_trace(
        logger,
        "llm_provider.invoke_guarded",
        {
            "node_name": node_name,
            "llm_node": llm_node,
            "provider": provider,
            "model": model,
            "output_mode": output_mode,
            "manifest_id": manifest_payload.get("manifest_id", ""),
            "message_count": manifest_payload.get("message_count", 0),
            "input_estimated_tokens": manifest_payload.get(
                "input_estimated_tokens",
                0,
            ),
            "fallback_used": False,
        },
        state=state or {},
        env_flag="LOG_A3_TRACE",
    )
    max_retries = _provider_transport_max_retries(node_name)
    retry_count = 0
    while True:
        try:
            return await operation(), retry_count
        except Exception as exc:
            if not _is_provider_transport_retryable(exc) or retry_count >= max_retries:
                if retry_count > 0 and _is_provider_transport_retryable(exc):
                    final_payload = {
                        "node_name": node_name,
                        "llm_node": llm_node,
                        "provider": provider,
                        "model": model,
                        "output_mode": output_mode,
                        "retry_count": retry_count,
                        "max_retries": max_retries,
                        "fallback_used": False,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "status_code": _extract_status_code(exc),
                        "provider_error_body": _provider_error_body(exc),
                    }
                    emit_a3_trace(
                        logger,
                        "final_failure_after_retries",
                        final_payload,
                        state=state or {},
                        env_flag="LOG_A3_TRACE",
                    )
                    if trace_stage_prefix != "provider_transport":
                        emit_a3_trace(
                            logger,
                            f"{trace_stage_prefix}_final_failure",
                            final_payload,
                            state=state or {},
                            env_flag="LOG_A3_TRACE",
                        )
                raise

            retry_count += 1
            status_code = _extract_status_code(exc)
            common_payload = {
                "node_name": node_name,
                "llm_node": llm_node,
                "provider": provider,
                "model": model,
                "output_mode": output_mode,
                "retry_count": retry_count,
                "max_retries": max_retries,
                "fallback_used": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "status_code": status_code,
                "provider_error_body": _provider_error_body(exc),
            }
            emit_a3_trace(
                logger,
                "provider_transport_error",
                common_payload,
                state=state or {},
                env_flag="LOG_A3_TRACE",
            )
            emit_a3_trace(
                logger,
                "provider_transport_retry_attempt",
                {
                    **common_payload,
                    "next_attempt": retry_count + 1,
                },
                state=state or {},
                env_flag="LOG_A3_TRACE",
            )
            if trace_stage_prefix != "provider_transport":
                emit_a3_trace(
                    logger,
                    f"{trace_stage_prefix}_error",
                    common_payload,
                    state=state or {},
                    env_flag="LOG_A3_TRACE",
                )
                emit_a3_trace(
                    logger,
                    f"{trace_stage_prefix}_retry_attempt",
                    {
                        **common_payload,
                        "next_attempt": retry_count + 1,
                    },
                    state=state or {},
                    env_flag="LOG_A3_TRACE",
                )
            await asyncio.sleep(_provider_transport_delay_seconds(retry_count))


async def invoke_context_importance_scorer_raw(
    *,
    llm_node: str,
    scorer_messages: list[dict[str, str]],
    timeout_seconds: float,
    state: dict | None = None,
) -> ContextImportanceScores:
    """Invoke the raw scorer without context apply or state/memory writes.

    CE-3 production config keeps importance scoring disabled. This raw transport
    path remains isolated for explicit tests or future opt-in experiments and
    must avoid invoke_plain_llm_fail_fast(), context usage/items/packing/apply,
    and state or memory writes, while still emitting provider input manifests.
    """
    if timeout_seconds <= 0:
        raise ContextImportanceError(
            reason="context_importance_policy_invalid",
            warning="importance scorer timeout_seconds must be positive",
            error_type="ContextConfigError",
        )
    llm = get_node_llm(llm_node)
    provider = get_setting(
        f"llm.{llm_node}.provider",
        get_setting(f"{llm_node}.provider", ""),
    )
    model = get_setting(
        f"llm.{llm_node}.model",
        get_setting(f"{llm_node}.model", getattr(llm, "model_name", "")),
    )
    state_payload = state or {}
    try:
        observation = build_llm_input_observation(
            node_name="context_importance_scorer",
            llm_node=llm_node,
            provider=str(provider or ""),
            model=str(model or ""),
            messages=scorer_messages,
            state=state_payload,
            call_purpose="context_importance_scoring",
            context_apply_applied=False,
            schema_contract_first=False,
            provider_bound_messages_mutated=False,
        )
        llm_input_manifest = observation.manifest
        emit_context_usage_trace(
            logger,
            observation=observation,
            messages=scorer_messages,
            state=state_payload,
        )
        _emit_llm_input_manifest_built(
            llm_input_manifest,
            state=state_payload,
        )
        raise_for_blocking_input_observation(observation)
        record_llm_input_influences(
            node_name="context_importance_scorer",
            llm_node=llm_node,
            messages=scorer_messages,
            state=state_payload,
            manifest=llm_input_manifest,
        )
    except (ContextConfigError, ContextUsageError) as exc:
        raise ContextImportanceError(
            reason="context_importance_input_budget_failed",
            warning="importance scorer input budget validation failed",
            error_type=type(exc).__name__,
        ) from exc
    except Exception as exc:
        _emit_llm_input_manifest_failed(
            node_name="context_importance_scorer",
            llm_node=llm_node,
            state=state_payload,
            exc=exc,
        )
        raise ContextImportanceError(
            reason="context_importance_manifest_failed",
            warning="importance scorer input manifest failed",
            error_type=type(exc).__name__,
        ) from exc
    try:
        result, _transport_retry_count = await invoke_with_provider_transport_retry(
            lambda: asyncio.wait_for(
                llm.ainvoke(scorer_messages),
                timeout=timeout_seconds,
            ),
            node_name="context_importance_scorer",
            llm_node=llm_node,
            provider=str(provider or ""),
            model=str(model or ""),
            llm_input_manifest=llm_input_manifest,
            output_mode="context_importance_scores",
            state=state_payload,
        )
    except TimeoutError as exc:
        raise ContextImportanceError(
            reason="context_importance_scorer_timed_out",
            warning="importance scorer timed out",
            error_type=type(exc).__name__,
        ) from exc
    except Exception as exc:
        raise ContextImportanceError(
            reason="context_importance_llm_failed",
            warning=f"importance scorer failed: {type(exc).__name__}",
            error_type=type(exc).__name__,
        ) from exc
    raw = str(getattr(result, "content", result) or "")
    return parse_importance_scorer_output(raw)


def _emit_context_policy_summary_once(
    summary: dict[str, Any], state: dict | None
) -> None:
    """Emit CE policy summary once per process unless debug env requests more."""
    global _CONTEXT_POLICY_SUMMARY_EMITTED
    if _CONTEXT_POLICY_SUMMARY_EMITTED and not should_emit_context_policy_summary():
        return
    emit_context_apply_policy_resolved_summary(
        logger,
        summary=summary,
        state=state or {},
    )
    _CONTEXT_POLICY_SUMMARY_EMITTED = True


async def invoke_plain_llm_fail_fast(
    *,
    node_name: str,
    llm_node: str,
    messages: list[Any],
    state: dict | None = None,
    temperature: float | None = None,
    max_raw_chars: int | None = None,
) -> str:
    """Invoke a plain-text LLM call with diagnostics and no implicit fallback."""
    llm = get_node_llm(llm_node)
    model = get_setting(
        f"llm.{llm_node}.model",
        get_setting(f"{llm_node}.model", getattr(llm, "model_name", "")),
    )
    provider = get_setting(
        f"llm.{llm_node}.provider",
        get_setting(f"{llm_node}.provider", ""),
    )
    if temperature is None:
        temperature = get_setting(
            f"llm.{llm_node}.temperature",
            get_setting(f"{llm_node}.temperature", 0.7),
        )
    max_chars = int(
        max_raw_chars
        or get_setting(f"llm_outputs.{node_name}.max_raw_chars", 12000)
        or 12000
    )
    started = time.perf_counter()
    state_payload = state or {}
    prepared = prepare_messages_with_context_policy(
        logger,
        node_name=node_name,
        llm_node=llm_node,
        model=str(model or ""),
        messages=messages or [],
        state=state_payload,
    )
    if prepared.resolved_policy is not None:
        _emit_context_policy_summary_once(
            prepared.resolved_policy.summary,
            state_payload,
        )
    messages_for_llm = prepared.messages_for_llm
    base_payload = {
        "node_name": node_name,
        "llm_node": llm_node,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "message_count": len(messages_for_llm),
        "prompt_chars": _message_content_chars(messages_for_llm),
        "fallback_used": False,
        "context_apply_applied": prepared.context_apply_applied,
        "context_apply_fallback_used": False,
        "trace_call_id": prepared.trace_call_id,
        "trace_seq": prepared.next_trace_seq + 2,
    }
    try:
        context_items = (
            tuple(prepared.selection.final_items)
            if prepared.selection is not None
            else ()
        )
        observation = build_llm_input_observation(
            node_name=node_name,
            llm_node=llm_node,
            provider=str(provider or ""),
            model=str(model or ""),
            messages=messages_for_llm,
            state=state_payload,
            call_purpose="plain_llm",
            context_apply_applied=prepared.context_apply_applied,
            context_apply_status=prepared.context_apply_status,
            optional_sources_missing=prepared.optional_sources_missing,
            provider_input_budget_tokens=prepared.provider_input_budget_tokens,
            provider_input_tokens_before_context=(
                prepared.provider_input_tokens_before_context
            ),
            provider_remaining_input_tokens=(prepared.provider_remaining_input_tokens),
            effective_context_budget_tokens=(prepared.effective_context_budget_tokens),
            schema_contract_first=False,
            provider_bound_messages_mutated=prepared.context_apply_applied,
            trace_call_id=prepared.trace_call_id,
            trace_seq=prepared.next_trace_seq + 2,
            context_items=context_items,
        )
        llm_input_manifest = observation.manifest
        emit_context_usage_trace(
            logger,
            observation=observation,
            messages=messages_for_llm,
            state=state_payload,
            trace_call_id=prepared.trace_call_id,
            trace_seq=prepared.next_trace_seq + 1,
        )
        _emit_llm_input_manifest_built(
            llm_input_manifest,
            state=state_payload,
        )
        raise_for_blocking_input_observation(observation)
        record_llm_input_influences(
            node_name=node_name,
            llm_node=llm_node,
            messages=messages_for_llm,
            state=state_payload,
            manifest=llm_input_manifest,
        )
    except (ContextConfigError, ContextUsageError):
        raise
    except Exception as exc:
        _emit_llm_input_manifest_failed(
            node_name=node_name,
            llm_node=llm_node,
            state=state_payload,
            exc=exc,
        )
        raise
    max_retries = get_llm_call_max_retries(node_name)
    retry_count = 0
    total_transport_retry_count = 0

    while True:
        try:
            result, transport_retry_count = await invoke_with_provider_transport_retry(
                lambda: llm.ainvoke(messages_for_llm),
                node_name=node_name,
                llm_node=llm_node,
                provider=provider,
                model=model,
                llm_input_manifest=llm_input_manifest,
                state=state_payload,
            )
            total_transport_retry_count += transport_retry_count
            raw = str(getattr(result, "content", result) or "")
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            if not raw.strip():
                raise ValueError("plain LLM returned empty output")
            emit_a3_trace(
                logger,
                "plain_llm_output",
                {
                    **base_payload,
                    "success": True,
                    "total_elapsed_ms": elapsed_ms,
                    "raw_output_chars": len(raw),
                    "error_type": "",
                    "error_message": "",
                    "provider_transport_retry_count": total_transport_retry_count,
                    "retry_count": retry_count,
                    "max_retries": max_retries,
                },
                state=state_payload,
                env_flag="LOG_A3_TRACE",
            )
            normalized_output = raw.strip()
            record_plain_output_influence(
                node_name=node_name,
                output=normalized_output,
                state=state_payload,
            )
            return normalized_output
        except Exception as exc:
            should_retry_empty = (
                isinstance(exc, ValueError)
                and str(exc) == "plain LLM returned empty output"
                and retry_count < max_retries
            )
            if should_retry_empty:
                retry_count += 1
                emit_a3_trace(
                    logger,
                    "plain_llm_retry_attempt",
                    {
                        **base_payload,
                        "retry_count": retry_count,
                        "max_retries": max_retries,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:max_chars],
                        "provider_transport_retry_count": total_transport_retry_count,
                    },
                    state=state_payload,
                    env_flag="LOG_A3_TRACE",
                )
                continue

            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            emit_a3_trace(
                logger,
                "plain_llm_output",
                {
                    **base_payload,
                    "success": False,
                    "total_elapsed_ms": elapsed_ms,
                    "raw_output_chars": 0,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:max_chars],
                    "provider_transport_retry_count": total_transport_retry_count,
                    "retry_count": retry_count,
                    "max_retries": max_retries,
                },
                state=state_payload,
                env_flag="LOG_A3_TRACE",
            )
            raise
