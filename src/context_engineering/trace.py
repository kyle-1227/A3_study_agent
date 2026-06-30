"""Trace helpers for Context Engineering telemetry."""

from __future__ import annotations

import logging
from typing import Any

from src.context_engineering.budget import build_context_usage_payload
from src.observability.a3_trace import emit_a3_trace

_ALLOWED_USAGE_KEYS = {
    "node_name",
    "llm_node",
    "provider",
    "model",
    "input_estimated_tokens",
    "reserved_output_tokens",
    "used_tokens",
    "max_context_tokens",
    "available_tokens",
    "used_ratio",
    "warning_level",
    "estimated",
    "tokenizer_mode",
    "message_count",
    "schema_size_chars",
    "breakdown",
}

_ALLOWED_ERROR_KEYS = {
    "node_name",
    "llm_node",
    "provider",
    "model",
    "reason",
    "warning",
}


def build_context_usage_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a sanitized context usage trace payload."""
    return {key: payload[key] for key in _ALLOWED_USAGE_KEYS if key in payload}


def build_context_usage_error_event(
    *,
    node_name: str,
    llm_node: str,
    provider: str,
    model: str,
    reason: str,
    warning: str,
) -> dict[str, Any]:
    """Return a sanitized context usage error payload."""
    return {
        "node_name": node_name,
        "llm_node": llm_node,
        "provider": provider,
        "model": model,
        "reason": reason,
        "warning": warning,
    }


def emit_context_usage(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str,
    provider: str,
    model: str,
    messages: list[Any],
    state: dict | None,
    reserved_output_tokens: int | None = None,
    schema_size_chars: int | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Emit context usage or error telemetry as A3_TRACE."""
    stage, payload = build_context_usage_payload(
        node_name=node_name,
        llm_node=llm_node,
        provider=provider,
        model=model,
        messages=messages,
        reserved_output_tokens=reserved_output_tokens,
        schema_size_chars=schema_size_chars,
    )
    if not stage or payload is None:
        return stage, payload

    safe_payload = (
        build_context_usage_event(payload)
        if stage == "context_usage"
        else {key: payload[key] for key in _ALLOWED_ERROR_KEYS if key in payload}
    )
    emit_a3_trace(
        logger,
        stage,
        safe_payload,
        state=state or {},
        env_flag="LOG_A3_TRACE",
    )
    return stage, safe_payload


def emit_context_usage_error(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str,
    provider: str,
    model: str,
    reason: str,
    warning: str,
    state: dict | None,
) -> None:
    """Emit an explicit context usage error event."""
    emit_a3_trace(
        logger,
        "context_usage_error",
        build_context_usage_error_event(
            node_name=node_name,
            llm_node=llm_node,
            provider=provider,
            model=model,
            reason=reason,
            warning=warning,
        ),
        state=state or {},
        env_flag="LOG_A3_TRACE",
    )
