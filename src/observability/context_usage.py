"""Compatibility adapter only. Source of truth is src.context_engineering."""

from __future__ import annotations

import logging
from typing import Any

from src.context_engineering.budget import build_context_usage_payload as _build_payload
from src.context_engineering.tokenizer import (
    estimate_messages_tokens_mixed,
    estimate_text_tokens_mixed,
)
from src.context_engineering.trace import emit_context_usage


def estimate_tokens_from_text(text: str) -> int:
    """Compatibility wrapper for legacy callers."""
    return estimate_text_tokens_mixed(text)


def estimate_messages_tokens(messages: list[Any]) -> int:
    """Compatibility wrapper for legacy callers."""
    return estimate_messages_tokens_mixed(messages)


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
    """Build context telemetry via the Context Engineering Kernel."""
    return _build_payload(
        node_name=node_name,
        llm_node=llm_node,
        provider=provider,
        model=model,
        messages=messages,
        reserved_output_tokens=output_reserved_tokens,
        schema_size_chars=schema_size_chars,
    )


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
    trace_fields: dict[str, Any] | None = None,
) -> None:
    """Emit context telemetry via the Context Engineering Kernel."""
    emit_context_usage(
        logger,
        node_name=node_name,
        llm_node=llm_node,
        provider=provider,
        model=model,
        messages=messages,
        state=state,
        reserved_output_tokens=output_reserved_tokens,
        schema_size_chars=schema_size_chars,
        trace_fields=trace_fields,
    )
