"""Safe trace helpers for Phase 3B-1 context apply."""

from __future__ import annotations

import logging
from typing import Any

from src.context_engineering.packing.apply import (
    ContextApplyError,
    ContextApplyResult,
    ContextInjectionPolicy,
)
from src.context_engineering.schema import sanitize_error_message
from src.observability.a3_trace import emit_a3_trace


def build_context_apply_plan_event(
    *,
    node_name: str,
    llm_node: str,
    policy: ContextInjectionPolicy,
    original_message_count: int,
    selected_item_count: int,
    injectable_item_count: int,
    skipped_item_count: int,
) -> dict[str, Any]:
    """Build a safe context_apply_plan event."""
    return {
        "node_name": node_name,
        "llm_node": llm_node,
        "apply_enabled": bool(policy.enabled),
        "original_message_count": original_message_count,
        "selected_item_count": selected_item_count,
        "injectable_item_count": injectable_item_count,
        "skipped_item_count": skipped_item_count,
        "injection_role": policy.role,
        "injection_position": policy.position,
    }


def build_context_applied_event(
    *,
    node_name: str,
    llm_node: str,
    policy: ContextInjectionPolicy,
    result: ContextApplyResult,
) -> dict[str, Any]:
    """Build a safe context_applied event from explicit result fields only."""
    return {
        "node_name": node_name,
        "llm_node": llm_node,
        "applied": result.applied,
        "fallback_used": result.fallback_used,
        "original_message_count": result.original_message_count,
        "final_message_count": result.final_message_count,
        "injected_items_count": result.injected_items_count,
        "skipped_items_count": result.skipped_items_count,
        "injected_context_tokens": result.injected_context_tokens,
        "injection_role": policy.role,
        "injection_position": policy.position,
        "warnings": _sanitize_warnings(result.warnings),
    }


def build_context_apply_error_event(error: ContextApplyError) -> dict[str, Any]:
    """Build a safe context_apply_error event."""
    return {
        "node_name": error.node_name,
        "llm_node": error.llm_node,
        "reason": error.reason,
        "warning": error.warning,
        "fallback_used": error.fallback_used,
        "error_type": error.original_exception_type or type(error).__name__,
    }


def emit_context_apply_plan(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str,
    policy: ContextInjectionPolicy,
    original_message_count: int,
    selected_item_count: int,
    injectable_item_count: int,
    skipped_item_count: int,
    state: dict | None,
) -> None:
    """Emit safe context_apply_plan telemetry."""
    emit_a3_trace(
        logger,
        "context_apply_plan",
        build_context_apply_plan_event(
            node_name=node_name,
            llm_node=llm_node,
            policy=policy,
            original_message_count=original_message_count,
            selected_item_count=selected_item_count,
            injectable_item_count=injectable_item_count,
            skipped_item_count=skipped_item_count,
        ),
        state=state or {},
        env_flag="LOG_A3_TRACE",
    )


def emit_context_applied(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str,
    policy: ContextInjectionPolicy,
    result: ContextApplyResult,
    state: dict | None,
) -> None:
    """Emit safe context_applied telemetry."""
    emit_a3_trace(
        logger,
        "context_applied",
        build_context_applied_event(
            node_name=node_name,
            llm_node=llm_node,
            policy=policy,
            result=result,
        ),
        state=state or {},
        env_flag="LOG_A3_TRACE",
    )


def emit_context_apply_error(
    logger: logging.Logger,
    *,
    error: ContextApplyError,
    state: dict | None,
) -> None:
    """Emit safe context_apply_error telemetry."""
    emit_a3_trace(
        logger,
        "context_apply_error",
        build_context_apply_error_event(error),
        state=state or {},
        env_flag="LOG_A3_TRACE",
    )


def _sanitize_warnings(warnings: list[str]) -> list[str]:
    return [sanitize_error_message(warning) for warning in warnings]
