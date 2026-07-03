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
from dataclasses import replace
from typing import Any, Awaitable, Callable, TypeVar

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
import httpx

from src.config import get_setting
from src.context_engineering.packing import (
    ContextApplyError,
    ContextImportanceError,
    ContextImportanceScores,
    aggregate_importance_failure,
    aggregate_importance_success,
    build_applied_messages_from_selection,
    build_importance_scorer_messages,
    emit_context_apply_selection,
    emit_context_applied,
    emit_context_apply_error,
    emit_context_apply_plan,
    emit_context_apply_policy_resolved_summary,
    emit_context_importance_scored,
    emit_context_packing_shadow,
    evaluate_context_apply_route,
    filter_context_items_by_source_policy,
    make_context_apply_skip_selection,
    parse_importance_scorer_output,
    prepare_context_apply_selection,
    resolve_context_policy,
    should_emit_context_policy_summary,
    with_context_apply_selection_warnings,
)
from src.context_engineering.providers import emit_context_items_shadow
from src.observability.context_usage import emit_context_usage_trace
from src.observability.a3_trace import emit_a3_trace

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


def _provider_error_body(exc: BaseException, *, max_chars: int = 12000) -> str:
    response = getattr(exc, "response", None)
    text = ""
    if response is not None:
        text = str(getattr(response, "text", "") or "")
        if not text:
            try:
                text = str(response.json())
            except Exception:
                text = ""
    if not text:
        body = getattr(exc, "body", None)
        if body:
            text = str(body)
    return text[:max_chars]


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
    output_mode: str = "",
    trace_stage_prefix: str = "provider_transport",
    state: dict | None = None,
) -> tuple[T, int]:
    """Retry transient provider transport failures without fallback.

    Retries only connection errors, timeouts, HTTP 429, and HTTP 5xx.
    The caller supplies the exact same operation each time, so model, prompt,
    schema, and request payload remain unchanged.
    """
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
) -> ContextImportanceScores:
    """Invoke the raw shadow scorer without context apply or trace side effects.

    Phase 3B-2A keeps importance scoring shadow-only: scorer failures are reported
    through safe context_importance_scored telemetry by the caller and do not affect
    the main LLM path, so this raw call intentionally uses direct ainvoke + timeout
    without provider transport retry. If a later non-shadow mode is introduced,
    it must use a pure transport retry wrapper while still avoiding
    invoke_plain_llm_fail_fast(), context usage/items/packing/apply, and state or
    memory writes.
    """
    if timeout_seconds <= 0:
        raise ContextImportanceError(
            reason="context_importance_policy_invalid",
            warning="importance scorer timeout_seconds must be positive",
            error_type="ContextConfigError",
        )
    llm = get_node_llm(llm_node)
    try:
        result = await asyncio.wait_for(
            llm.ainvoke(scorer_messages),
            timeout=timeout_seconds,
        )
    except TimeoutError as exc:
        raise ContextImportanceError(
            reason="context_importance_timeout",
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
    original_messages = messages or []
    state_payload = state or {}
    messages_for_llm = original_messages
    context_apply_applied = False
    context_apply_fallback_used = False
    try:
        resolved_policy = resolve_context_policy(
            node_name=node_name,
            llm_node=llm_node,
            state=state_payload,
        )
        apply_policy = resolved_policy.injection_policy
        _emit_context_policy_summary_once(resolved_policy.summary, state_payload)
    except ContextApplyError as exc:
        exc.fallback_used = True
        resolved_policy = None
        apply_policy = None
        context_apply_fallback_used = True
        emit_context_apply_error(logger, error=exc, state=state_payload)

    def _emit_original_usage_and_shadow() -> None:
        emit_context_usage_trace(
            logger,
            node_name=node_name,
            llm_node=llm_node,
            provider=str(provider or ""),
            model=str(model or ""),
            messages=messages_for_llm,
            state=state_payload,
        )
        context_items = emit_context_items_shadow(
            logger,
            node_name=node_name,
            llm_node=llm_node,
            messages=original_messages,
            state=state_payload,
        )
        emit_context_packing_shadow(
            logger,
            node_name=node_name,
            llm_node=llm_node,
            items=context_items,
            state=state_payload,
        )

    async def _emit_importance_shadow_if_allowed(
        selection, *, observe_only: bool
    ) -> None:
        if apply_policy is None:
            return
        if (
            observe_only
            and not apply_policy.importance_scoring.enabled_for_observe_only
        ):
            return
        if apply_policy.importance_scoring.disabled_reason:
            emit_context_importance_scored(
                logger,
                node_name=node_name,
                llm_node=llm_node,
                telemetry=aggregate_importance_failure(
                    items=selection.final_items,
                    policy=apply_policy.importance_scoring,
                    started_at=None,
                    reason=apply_policy.importance_scoring.disabled_reason,
                ),
                state=state_payload,
            )
            return
        if not (
            apply_policy.importance_scoring.enabled
            and apply_policy.importance_scoring.emit_shadow_telemetry
            and selection.final_items
        ):
            return
        scorer_started = time.perf_counter()
        scorer_messages = build_importance_scorer_messages(
            items=selection.final_items,
            policy=apply_policy.importance_scoring,
        )
        try:
            scores = await invoke_context_importance_scorer_raw(
                llm_node=apply_policy.importance_scoring.llm_node,
                scorer_messages=scorer_messages,
                timeout_seconds=apply_policy.importance_scoring.timeout_seconds,
            )
            telemetry = aggregate_importance_success(
                items=selection.final_items,
                scores=scores,
                policy=apply_policy.importance_scoring,
                started_at=scorer_started,
            )
        except ContextImportanceError as exc:
            telemetry = aggregate_importance_failure(
                items=selection.final_items,
                policy=apply_policy.importance_scoring,
                started_at=scorer_started,
                reason=exc.reason,
                error_type=exc.error_type,
                warning=exc.warning,
            )
        emit_context_importance_scored(
            logger,
            node_name=node_name,
            llm_node=llm_node,
            telemetry=telemetry,
            state=state_payload,
        )

    if apply_policy is None or not apply_policy.enabled:
        _emit_original_usage_and_shadow()
    elif apply_policy.mode == "disabled":
        selection = make_context_apply_skip_selection(
            skip_reason="node_policy_disabled",
            policy=apply_policy,
        )
        emit_context_apply_selection(
            logger,
            node_name=node_name,
            llm_node=llm_node,
            selection=selection,
            state=state_payload,
        )
        emit_context_usage_trace(
            logger,
            node_name=node_name,
            llm_node=llm_node,
            provider=str(provider or ""),
            model=str(model or ""),
            messages=messages_for_llm,
            state=state_payload,
        )
    else:
        assert apply_policy is not None
        assert resolved_policy is not None
        if apply_policy.mode == "observe_only":
            context_items = emit_context_items_shadow(
                logger,
                node_name=node_name,
                llm_node=llm_node,
                messages=original_messages,
                state=state_payload,
            )
            packed = emit_context_packing_shadow(
                logger,
                node_name=node_name,
                llm_node=llm_node,
                items=context_items,
                state=state_payload,
            )
            if packed is not None:
                source_filter = filter_context_items_by_source_policy(
                    packed.selected_items,
                    injectable_sources=apply_policy.injectable_sources,
                    exclude_message_source=apply_policy.exclude_message_source,
                    source_policies=resolved_policy.source_policies,
                    state=state_payload,
                )
                emit_context_apply_plan(
                    logger,
                    node_name=node_name,
                    llm_node=llm_node,
                    policy=apply_policy,
                    original_message_count=len(original_messages),
                    selected_item_count=len(packed.selected_items),
                    injectable_item_count=len(source_filter.kept_items),
                    skipped_item_count=len(source_filter.dropped_items),
                    state=state_payload,
                )
                selection = prepare_context_apply_selection(
                    packed=packed,
                    policy=apply_policy,
                    node_name=node_name,
                    llm_node=llm_node,
                    source_filter_result=source_filter,
                )
                selection = with_context_apply_selection_warnings(
                    selection,
                    ["node_policy_observe_only"],
                )
                selection = replace(
                    selection,
                    skip_reason="node_policy_observe_only",
                    rendered_context="",
                )
                emit_context_apply_selection(
                    logger,
                    node_name=node_name,
                    llm_node=llm_node,
                    selection=selection,
                    state=state_payload,
                )
                await _emit_importance_shadow_if_allowed(
                    selection,
                    observe_only=True,
                )
            else:
                emit_context_apply_plan(
                    logger,
                    node_name=node_name,
                    llm_node=llm_node,
                    policy=apply_policy,
                    original_message_count=len(original_messages),
                    selected_item_count=0,
                    injectable_item_count=0,
                    skipped_item_count=0,
                    state=state_payload,
                )
                selection = make_context_apply_skip_selection(
                    skip_reason="node_policy_observe_only",
                    warnings=["packed_context_missing"],
                    policy=apply_policy,
                )
                emit_context_apply_selection(
                    logger,
                    node_name=node_name,
                    llm_node=llm_node,
                    selection=selection,
                    state=state_payload,
                )
            emit_context_usage_trace(
                logger,
                node_name=node_name,
                llm_node=llm_node,
                provider=str(provider or ""),
                model=str(model or ""),
                messages=messages_for_llm,
                state=state_payload,
            )
        else:
            route_enabled, skip_reason, single_resource_result, route_warnings = (
                evaluate_context_apply_route(
                    policy=apply_policy,
                    node_name=node_name,
                    state=state_payload,
                )
            )
            if not route_enabled:
                selection = make_context_apply_skip_selection(
                    skip_reason=skip_reason,
                    single_resource_result=single_resource_result,
                    warnings=route_warnings,
                    policy=apply_policy,
                )
                emit_context_apply_selection(
                    logger,
                    node_name=node_name,
                    llm_node=llm_node,
                    selection=selection,
                    state=state_payload,
                )
                _emit_original_usage_and_shadow()
            else:
                context_items = emit_context_items_shadow(
                    logger,
                    node_name=node_name,
                    llm_node=llm_node,
                    messages=original_messages,
                    state=state_payload,
                )
                packed = emit_context_packing_shadow(
                    logger,
                    node_name=node_name,
                    llm_node=llm_node,
                    items=context_items,
                    state=state_payload,
                )
                try:
                    if packed is None:
                        emit_context_apply_plan(
                            logger,
                            node_name=node_name,
                            llm_node=llm_node,
                            policy=apply_policy,
                            original_message_count=len(original_messages),
                            selected_item_count=0,
                            injectable_item_count=0,
                            skipped_item_count=0,
                            state=state_payload,
                        )
                        selection = make_context_apply_skip_selection(
                            skip_reason="packed_context_missing",
                            warnings=["packed_context_missing"],
                            policy=apply_policy,
                        )
                        emit_context_apply_selection(
                            logger,
                            node_name=node_name,
                            llm_node=llm_node,
                            selection=selection,
                            state=state_payload,
                        )
                        raise ContextApplyError(
                            reason="packed_context_missing",
                            warning="context packing did not produce a PackedContext",
                            node_name=node_name,
                            llm_node=llm_node,
                            fallback_used=apply_policy.fallback_on_error,
                        )
                    source_filter = filter_context_items_by_source_policy(
                        packed.selected_items,
                        injectable_sources=apply_policy.injectable_sources,
                        exclude_message_source=apply_policy.exclude_message_source,
                        source_policies=resolved_policy.source_policies,
                        state=state_payload,
                    )
                    emit_context_apply_plan(
                        logger,
                        node_name=node_name,
                        llm_node=llm_node,
                        policy=apply_policy,
                        original_message_count=len(original_messages),
                        selected_item_count=len(packed.selected_items),
                        injectable_item_count=len(source_filter.kept_items),
                        skipped_item_count=len(source_filter.dropped_items),
                        state=state_payload,
                    )
                    selection = prepare_context_apply_selection(
                        packed=packed,
                        policy=apply_policy,
                        node_name=node_name,
                        llm_node=llm_node,
                        source_filter_result=source_filter,
                    )
                    selection = with_context_apply_selection_warnings(
                        selection,
                        route_warnings,
                    )
                    emit_context_apply_selection(
                        logger,
                        node_name=node_name,
                        llm_node=llm_node,
                        selection=selection,
                        state=state_payload,
                    )
                    await _emit_importance_shadow_if_allowed(
                        selection,
                        observe_only=False,
                    )
                    if selection.skip_reason:
                        if (
                            selection.skip_reason == "budget_fit_failed"
                            and not apply_policy.budget.fallback_if_empty_after_drop
                        ):
                            raise ContextApplyError(
                                reason="context_apply_budget_fit_failed",
                                warning="context apply could not fit any item in budget",
                                node_name=node_name,
                                llm_node=llm_node,
                                fallback_used=apply_policy.fallback_on_error,
                            )
                        messages_for_llm = original_messages
                        context_apply_fallback_used = (
                            selection.skip_reason == "budget_fit_failed"
                        )
                    else:
                        apply_result = build_applied_messages_from_selection(
                            node_name=node_name,
                            llm_node=llm_node,
                            original_messages=original_messages,
                            selection=selection,
                        )
                        emit_context_applied(
                            logger,
                            node_name=node_name,
                            llm_node=llm_node,
                            policy=apply_policy,
                            result=apply_result,
                            state=state_payload,
                        )
                        if apply_result.applied:
                            messages_for_llm = apply_result.final_messages
                        context_apply_applied = apply_result.applied
                        context_apply_fallback_used = apply_result.fallback_used
                except ContextApplyError as exc:
                    exc.fallback_used = apply_policy.fallback_on_error
                    emit_context_apply_error(logger, error=exc, state=state_payload)
                    if not apply_policy.fallback_on_error:
                        raise
                    messages_for_llm = original_messages
                    context_apply_fallback_used = True
                emit_context_usage_trace(
                    logger,
                    node_name=node_name,
                    llm_node=llm_node,
                    provider=str(provider or ""),
                    model=str(model or ""),
                    messages=messages_for_llm,
                    state=state_payload,
                )

    base_payload = {
        "node_name": node_name,
        "llm_node": llm_node,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "message_count": len(messages_for_llm),
        "prompt_chars": _message_content_chars(messages_for_llm),
        "fallback_used": False,
        "context_apply_applied": context_apply_applied,
        "context_apply_fallback_used": context_apply_fallback_used,
    }
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
                    "raw_output": raw[:max_chars],
                    "error_type": "",
                    "error_message": "",
                    "provider_error_body": "",
                    "provider_transport_retry_count": total_transport_retry_count,
                    "retry_count": retry_count,
                    "max_retries": max_retries,
                },
                state=state_payload,
                env_flag="LOG_A3_TRACE",
            )
            return raw.strip()
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
                    "raw_output": "",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:max_chars],
                    "provider_error_body": _provider_error_body(
                        exc,
                        max_chars=max_chars,
                    ),
                    "provider_transport_retry_count": total_transport_retry_count,
                    "retry_count": retry_count,
                    "max_retries": max_retries,
                },
                state=state_payload,
                env_flag="LOG_A3_TRACE",
            )
            raise
