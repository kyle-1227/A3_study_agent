"""Trace helpers for Context Engineering telemetry."""

from __future__ import annotations

import logging
from typing import Any

from src.context_engineering.budget import build_context_usage_payload
from src.context_engineering.evidence_normalizer import EvidenceNormalizationStats
from src.context_engineering.schema import ContextItem, ContextProviderError
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

_ALLOWED_BREAKDOWN_KEYS = {
    "input_estimated_tokens",
    "reserved_output_tokens",
    "schema_size_chars",
}

_ALLOWED_TOP_ITEM_KEYS = {
    "id",
    "source_type",
    "title",
    "token_estimate",
    "priority",
    "scope",
    "lifetime",
    "disclosure_level",
}


def build_context_usage_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a sanitized context usage trace payload."""
    event = {
        key: payload[key]
        for key in _ALLOWED_USAGE_KEYS
        if key in payload and key != "breakdown"
    }
    if "breakdown" in payload:
        event["breakdown"] = _safe_breakdown(payload.get("breakdown"))
    return event


def _safe_breakdown(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    safe: dict[str, int] = {}
    for key, item in value.items():
        if (
            key in _ALLOWED_BREAKDOWN_KEYS
            and isinstance(item, int)
            and not isinstance(item, bool)
        ):
            safe[key] = item
    return safe


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


def build_context_items_collected_event(
    *,
    node_name: str,
    llm_node: str,
    provider_count: int,
    items: list[ContextItem],
    trace_top_items: int,
    evidence_stats: EvidenceNormalizationStats | None = None,
) -> dict[str, Any]:
    """Return a safe context item collection summary."""
    source_counts: dict[str, int] = {}
    total_estimated_tokens = 0
    for item in items:
        source_counts[item.source_type] = source_counts.get(item.source_type, 0) + 1
        total_estimated_tokens += item.token_estimate
    event = {
        "node_name": node_name,
        "llm_node": llm_node,
        "provider_count": provider_count,
        "item_count": len(items),
        "source_counts": source_counts,
        "total_estimated_tokens": total_estimated_tokens,
        "top_items": [
            _safe_top_item(item) for item in items[: max(trace_top_items, 0)]
        ],
    }
    if evidence_stats is not None:
        event.update(evidence_stats.as_event_fields())
    return event


def build_context_provider_error_event(
    *,
    node_name: str,
    llm_node: str,
    error: ContextProviderError,
) -> dict[str, Any]:
    """Return a safe provider error summary."""
    return {
        "node_name": node_name,
        "llm_node": llm_node,
        "provider": error.provider,
        "source_type": error.source_type,
        "provider_stage": error.stage,
        "error_type": error.original_exception_type or type(error).__name__,
        "error_reason": error.sanitized_message,
    }


def _safe_top_item(item: ContextItem) -> dict[str, Any]:
    raw = item.model_dump()
    return {key: raw[key] for key in _ALLOWED_TOP_ITEM_KEYS if key in raw}


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
    trace_fields: dict[str, Any] | None = None,
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
    if trace_fields:
        safe_payload = {**trace_fields, **safe_payload}
    emit_a3_trace(
        logger,
        stage,
        safe_payload,
        state=state or {},
        env_flag="LOG_A3_TRACE",
    )
    return stage, safe_payload


def emit_context_items_collected(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str,
    provider_count: int,
    items: list[ContextItem],
    trace_top_items: int,
    state: dict | None,
    evidence_stats: EvidenceNormalizationStats | None = None,
) -> None:
    """Emit a safe context_items_collected event."""
    emit_a3_trace(
        logger,
        "context_items_collected",
        build_context_items_collected_event(
            node_name=node_name,
            llm_node=llm_node,
            provider_count=provider_count,
            items=items,
            trace_top_items=trace_top_items,
            evidence_stats=evidence_stats,
        ),
        state=state or {},
        env_flag="LOG_A3_TRACE",
    )


def emit_context_provider_error(
    logger: logging.Logger,
    *,
    error: ContextProviderError,
    node_name: str,
    llm_node: str,
    state: dict | None,
) -> None:
    """Emit a safe context_provider_error event."""
    emit_a3_trace(
        logger,
        "context_provider_error",
        build_context_provider_error_event(
            node_name=node_name,
            llm_node=llm_node,
            error=error,
        ),
        state=state or {},
        env_flag="LOG_A3_TRACE",
    )


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
