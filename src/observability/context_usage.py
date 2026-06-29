"""Context-window usage telemetry without prompt or schema body exposure."""

from __future__ import annotations

import logging
from typing import Any

from src.config import get_setting
from src.observability.a3_trace import emit_a3_trace


def _message_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(value)


def estimate_tokens_from_text(text: str) -> int:
    """Estimate token count from text length; result is always marked estimated."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_messages_tokens(messages: list[Any]) -> int:
    total = 0
    for message in messages or []:
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        total += estimate_tokens_from_text(_message_content(content))
    return total


def _context_budget_config() -> tuple[dict[str, Any] | None, str]:
    raw = get_setting("context_budget")
    if raw is None:
        return None, "context_budget_missing"
    if not isinstance(raw, dict):
        return None, "context_budget_invalid"
    if raw.get("enabled") is not True:
        return {}, ""
    return raw, ""


def _ratio_level(ratio: float, *, warn_ratio: float, critical_ratio: float) -> str:
    if ratio >= critical_ratio:
        return "critical"
    if ratio >= warn_ratio:
        return "warning"
    return "ok"


def build_context_usage_payload(
    *,
    node_name: str,
    llm_node: str,
    provider: str,
    model: str,
    messages: list[Any],
    output_reserved_tokens: int | None = None,
    schema_size_chars: int | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Build a context telemetry payload or an explicit warning payload."""
    config, error_code = _context_budget_config()
    if error_code:
        return "context_usage_error", {
            "node_name": node_name,
            "llm_node": llm_node,
            "provider": provider,
            "model": model,
            "reason": error_code,
            "warning": "context usage telemetry unavailable; no synthetic window data emitted",
        }
    if config is None:
        return "context_usage_error", {
            "node_name": node_name,
            "llm_node": llm_node,
            "provider": provider,
            "model": model,
            "reason": "context_budget_missing",
            "warning": "context usage telemetry unavailable; no synthetic window data emitted",
        }
    if config == {}:
        return "", None

    try:
        warn_ratio = float(config["warn_ratio"])
        critical_ratio = float(config["critical_ratio"])
    except (KeyError, TypeError, ValueError):
        return "context_usage_error", {
            "node_name": node_name,
            "llm_node": llm_node,
            "provider": provider,
            "model": model,
            "reason": "context_budget_thresholds_invalid",
            "warning": "context usage telemetry unavailable; context budget thresholds are invalid",
        }

    model_name = str(model or "").strip()
    model_limits = config.get("model_limits")
    if not model_name:
        reason = "model_missing"
        max_context_tokens = None
    elif not isinstance(model_limits, dict):
        reason = "model_limits_missing"
        max_context_tokens = None
    else:
        max_context_tokens = model_limits.get(model_name)
        reason = "" if max_context_tokens is not None else "model_window_unknown"

    try:
        max_context_tokens = int(max_context_tokens) if max_context_tokens is not None else None
    except (TypeError, ValueError):
        max_context_tokens = None
        reason = "model_window_invalid"

    if not max_context_tokens or max_context_tokens <= 0:
        return "context_usage_error", {
            "node_name": node_name,
            "llm_node": llm_node,
            "provider": provider,
            "model": model,
            "reason": reason,
            "warning": "context usage telemetry unavailable; model context window is unknown",
        }

    prompt_tokens = estimate_messages_tokens(messages)
    reserved = output_reserved_tokens if isinstance(output_reserved_tokens, int) and output_reserved_tokens > 0 else 0
    used_tokens = prompt_tokens + reserved
    remaining_tokens = max(max_context_tokens - used_tokens, 0)
    usage_ratio = round(used_tokens / max_context_tokens, 4)
    payload = {
        "node_name": node_name,
        "llm_node": llm_node,
        "provider": provider,
        "model": model_name,
        "prompt_tokens": prompt_tokens,
        "output_reserved_tokens": reserved,
        "used_tokens": used_tokens,
        "max_context_tokens": max_context_tokens,
        "usage_ratio": usage_ratio,
        "remaining_tokens": remaining_tokens,
        "estimated": True,
        "level": _ratio_level(usage_ratio, warn_ratio=warn_ratio, critical_ratio=critical_ratio),
    }
    if schema_size_chars is not None:
        payload["schema_size_chars"] = int(schema_size_chars)
    return "context_usage", payload


def emit_context_usage_trace(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str,
    provider: str,
    model: str,
    messages: list[Any],
    state: dict | None,
    output_reserved_tokens: int | None = None,
    schema_size_chars: int | None = None,
) -> None:
    """Emit context telemetry as A3_TRACE, or a warning event if unavailable."""
    stage, payload = build_context_usage_payload(
        node_name=node_name,
        llm_node=llm_node,
        provider=provider,
        model=model,
        messages=messages,
        output_reserved_tokens=output_reserved_tokens,
        schema_size_chars=schema_size_chars,
    )
    if not stage or payload is None:
        return
    emit_a3_trace(
        logger,
        stage,
        payload,
        state=state or {},
        env_flag="LOG_A3_TRACE",
    )
