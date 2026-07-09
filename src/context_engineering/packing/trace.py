"""Safe trace helpers for ContextPacker shadow mode."""

from __future__ import annotations

import logging
from typing import Any

from src.context_engineering.packing.packer import pack_context_items
from src.context_engineering.packing.policies import (
    PackingPolicy,
    get_packing_policy,
    node_enabled,
)
from src.context_engineering.packing.schema import (
    ContextPackingError,
    PackedContext,
    PackingDecision,
)
from src.context_engineering.schema import ContextItem, sanitize_error_message
from src.observability.a3_trace import emit_a3_trace

_ALLOWED_PREVIEW_KEYS = {
    "id",
    "source_type",
    "title",
    "token_estimate",
    "priority",
    "can_drop",
    "reason",
}


def build_context_packing_plan_event(
    *,
    node_name: str,
    llm_node: str,
    items: list[ContextItem],
    max_context_block_tokens: int,
    strategy: str,
) -> dict[str, Any]:
    """Build a safe pre-pack plan event."""
    return {
        "node_name": node_name,
        "llm_node": llm_node,
        "candidate_count": len(items),
        "source_counts": _source_counts(items),
        "max_context_block_tokens": max_context_block_tokens,
        "strategy": strategy,
    }


def build_context_packed_event(
    packed: PackedContext,
    *,
    trace_selected_items: int,
    trace_dropped_items: int,
) -> dict[str, Any]:
    """Build a safe packed event."""
    selected_decisions = [
        decision for decision in packed.decisions if decision.selected
    ]
    dropped_decisions = [
        decision for decision in packed.decisions if not decision.selected
    ]
    return {
        "node_name": packed.node_name,
        "llm_node": packed.llm_node,
        "strategy": packed.strategy,
        "selected_count": len(packed.selected_items),
        "dropped_count": len(packed.dropped_items),
        "selected_tokens": packed.selected_tokens,
        "dropped_tokens": packed.dropped_tokens,
        "required_tokens": packed.required_tokens,
        "optional_tokens": packed.optional_tokens,
        "remaining_tokens": packed.remaining_tokens,
        "overflow": packed.overflow,
        "selected_items_preview": [
            _decision_preview(decision)
            for decision in selected_decisions[: max(trace_selected_items, 0)]
        ],
        "dropped_items_preview": [
            _decision_preview(decision)
            for decision in dropped_decisions[: max(trace_dropped_items, 0)]
        ],
        "warnings": list(packed.warnings),
    }


def build_context_packing_error_event(error: ContextPackingError) -> dict[str, Any]:
    """Build a safe packing error event."""
    return {
        "node_name": error.node_name,
        "llm_node": error.llm_node,
        "reason": error.reason,
        "warning": error.warning,
        "selected_tokens": error.selected_tokens,
        "budget_tokens": error.budget_tokens,
        "error_type": error.original_exception_type or type(error).__name__,
    }


def build_context_packing_observed_event(
    *,
    node_name: str,
    llm_node: str,
    reason: str,
    warning: object,
    budget_tokens: int,
) -> dict[str, Any]:
    """Build a safe non-error observed/skipped packing event."""
    return {
        "node_name": str(node_name or ""),
        "llm_node": str(llm_node or ""),
        "status": "observed",
        "reason": str(reason or "context_packing_observed"),
        "warning": sanitize_error_message(warning, max_chars=300),
        "selected_tokens": 0,
        "budget_tokens": int(budget_tokens or 0),
    }


def emit_context_packing_plan(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str,
    items: list[ContextItem],
    policy: PackingPolicy,
    state: dict | None,
) -> None:
    """Emit safe context_packing_plan telemetry."""
    emit_a3_trace(
        logger,
        "context_packing_plan",
        build_context_packing_plan_event(
            node_name=node_name,
            llm_node=llm_node,
            items=items,
            max_context_block_tokens=policy.max_context_block_tokens,
            strategy=policy.strategy,
        ),
        state=state or {},
        env_flag="LOG_A3_TRACE",
    )


def emit_context_packed(
    logger: logging.Logger,
    *,
    packed: PackedContext,
    policy: PackingPolicy,
    state: dict | None,
) -> None:
    """Emit safe context_packed telemetry."""
    emit_a3_trace(
        logger,
        "context_packed",
        build_context_packed_event(
            packed,
            trace_selected_items=policy.trace_selected_items,
            trace_dropped_items=policy.trace_dropped_items,
        ),
        state=state or {},
        env_flag="LOG_A3_TRACE",
    )


def emit_context_packing_error(
    logger: logging.Logger,
    *,
    error: ContextPackingError,
    state: dict | None,
) -> None:
    """Emit safe context_packing_error telemetry."""
    emit_a3_trace(
        logger,
        "context_packing_error",
        build_context_packing_error_event(error),
        state=state or {},
        env_flag="LOG_A3_TRACE",
    )


def emit_context_packing_observed(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str,
    reason: str,
    warning: object,
    budget_tokens: int,
    state: dict | None,
) -> None:
    """Emit safe non-error context_packing_observed telemetry."""
    emit_a3_trace(
        logger,
        "context_packing_observed",
        build_context_packing_observed_event(
            node_name=node_name,
            llm_node=llm_node,
            reason=reason,
            warning=warning,
            budget_tokens=budget_tokens,
        ),
        state=state or {},
        env_flag="LOG_A3_TRACE",
    )


def emit_context_packing_shadow(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str | None,
    items: list[ContextItem],
    state: dict | None,
) -> PackedContext | None:
    """Run ContextPacker in shadow mode and emit only safe telemetry."""
    llm_node_text = str(llm_node or "")
    try:
        policy = get_packing_policy(node_name=node_name, llm_node=llm_node_text)
        if not policy.enabled or not node_enabled(policy, node_name=node_name):
            return None
        if policy.apply_to_llm:
            emit_context_packing_observed(
                logger,
                reason="apply_to_llm_unsupported",
                warning=(
                    "context_engineering.packer.apply_to_llm is ignored by "
                    "observe-only packing; active injection is controlled by node policy"
                ),
                node_name=node_name,
                llm_node=llm_node_text,
                budget_tokens=policy.max_context_block_tokens,
                state=state,
            )
            return None
        if not policy.shadow_mode:
            emit_context_packing_observed(
                logger,
                reason="shadow_mode_required",
                warning=(
                    "context_engineering.packer.shadow_mode is disabled; "
                    "observe-only shadow packing was skipped"
                ),
                node_name=node_name,
                llm_node=llm_node_text,
                budget_tokens=policy.max_context_block_tokens,
                state=state,
            )
            return None
        emit_context_packing_plan(
            logger,
            node_name=node_name,
            llm_node=llm_node_text,
            items=items,
            policy=policy,
            state=state,
        )
        packed = pack_context_items(
            node_name=node_name,
            llm_node=llm_node_text,
            items=items,
            max_context_block_tokens=policy.max_context_block_tokens,
            strategy=policy.strategy,
            enabled_sources=policy.enabled_sources,
        )
        emit_context_packed(
            logger,
            packed=packed,
            policy=policy,
            state=state,
        )
        return packed
    except ContextPackingError as exc:
        emit_context_packing_error(logger, error=exc, state=state)
        return None
    except Exception as exc:
        emit_context_packing_error(
            logger,
            error=ContextPackingError(
                reason="context_packing_error",
                warning=exc,
                node_name=node_name,
                llm_node=llm_node_text,
                original_exception_type=type(exc).__name__,
            ),
            state=state,
        )
        return None


def _decision_preview(decision: PackingDecision) -> dict[str, Any]:
    raw = {
        "id": decision.item_id,
        "source_type": decision.source_type,
        "title": sanitize_error_message(decision.title, max_chars=120),
        "token_estimate": decision.token_estimate,
        "priority": decision.priority,
        "can_drop": decision.can_drop,
        "reason": decision.reason,
    }
    return {key: raw[key] for key in _ALLOWED_PREVIEW_KEYS}


def _source_counts(items: list[ContextItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.source_type] = counts.get(item.source_type, 0) + 1
    return counts
