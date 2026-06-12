"""Central LLM factory and fallback invoke logic.

Provides a resilient invoke_with_fallback() that catches transient API errors
(timeouts, 502s, rate limits) and retries on a fallback model, recording the
failover event on the active OpenTelemetry span.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from src.config import get_setting
from src.observability.a3_trace import emit_a3_trace

logger = logging.getLogger(__name__)

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
    provider = get_setting(f"{nested_prefix}.provider", get_setting(f"{node_name}.provider", "deepseek"))
    provider_name = str(provider or "").strip().lower()
    default_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
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
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
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
        model=os.getenv("FALLBACK_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
        api_key=os.getenv("FALLBACK_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "not-configured",
        base_url=os.getenv("FALLBACK_BASE_URL", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")),
        temperature=0.7,
    )
    defaults.update(overrides)
    return ChatOpenAI(**defaults)


# ---------------------------------------------------------------------------
# Resilient invoke
# ---------------------------------------------------------------------------

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
    try:
        response = primary.invoke(messages)
        if span is not None:
            span.set_attribute("llm.fallback_used", False)
        return response
    except _FALLBACK_ERRORS as exc:
        if fallback is None:
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

        return fallback.invoke(messages)


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
    try:
        response = await primary.ainvoke(messages)
        if span is not None:
            span.set_attribute("llm.fallback_used", False)
        return response
    except _FALLBACK_ERRORS as exc:
        if fallback is None:
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

        return await fallback.ainvoke(messages)


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
    model = get_setting(f"llm.{llm_node}.model", get_setting(f"{llm_node}.model", getattr(llm, "model_name", "")))
    provider = get_setting(f"llm.{llm_node}.provider", get_setting(f"{llm_node}.provider", ""))
    if temperature is None:
        temperature = get_setting(f"llm.{llm_node}.temperature", get_setting(f"{llm_node}.temperature", 0.7))
    max_chars = int(max_raw_chars or get_setting(f"llm_outputs.{node_name}.max_raw_chars", 12000) or 12000)
    started = time.perf_counter()
    base_payload = {
        "node_name": node_name,
        "llm_node": llm_node,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "message_count": len(messages or []),
        "prompt_chars": _message_content_chars(messages or []),
        "fallback_used": False,
    }
    try:
        result = await llm.ainvoke(messages)
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
            },
            state=state or {},
            env_flag="LOG_A3_TRACE",
        )
        return raw.strip()
    except Exception as exc:
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
                "provider_error_body": _provider_error_body(exc, max_chars=max_chars),
            },
            state=state or {},
            env_flag="LOG_A3_TRACE",
        )
        raise
