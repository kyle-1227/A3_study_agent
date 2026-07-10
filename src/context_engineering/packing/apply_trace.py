"""Safe trace helpers for Phase 3B-1 context apply."""

from __future__ import annotations

import logging
from typing import Any

from src.context_engineering.packing.apply import (
    ContextApplyError,
    ContextApplyResult,
    ContextApplySelection,
    ContextInjectionPolicy,
)
from src.context_engineering.packing.importance import ContextImportanceTelemetry
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
        "mode": policy.mode,
        "risk_tier": policy.risk_tier,
        "policy_source": policy.policy_source,
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
        "apply_status": result.apply_status,
        "fallback_used": result.fallback_used,
        "original_message_count": result.original_message_count,
        "final_message_count": result.final_message_count,
        "injected_items_count": result.injected_items_count,
        "skipped_items_count": result.skipped_items_count,
        "injected_context_tokens": result.injected_context_tokens,
        "budget_dropped_count": result.budget_dropped_count,
        "final_injected_count": result.final_injected_count,
        "original_estimated_tokens": result.original_estimated_tokens,
        "final_estimated_tokens": result.final_estimated_tokens,
        "token_delta": result.token_delta,
        "source_counts_after": _safe_int_dict(result.source_counts_after),
        "drop_reasons": _safe_int_dict(result.drop_reasons),
        "source_drop_reasons": _safe_int_dict(result.source_drop_reasons),
        "budget_drop_reasons": _safe_int_dict(result.budget_drop_reasons),
        "injection_role": policy.role,
        "injection_position": policy.position,
        "mode": result.mode,
        "risk_tier": result.risk_tier,
        "policy_source": result.policy_source,
        "warnings": _sanitize_warnings(result.warnings),
    }


def build_context_apply_selection_event(
    *,
    node_name: str,
    llm_node: str,
    selection: ContextApplySelection,
) -> dict[str, Any]:
    """Build a safe context_apply_selection event."""
    return {
        "node_name": node_name,
        "llm_node": llm_node,
        "skip_reason": selection.skip_reason,
        "single_resource_result": selection.single_resource_result,
        "selected_item_count": selection.selected_item_count,
        "injectable_item_count": selection.injectable_item_count,
        "skipped_item_count": selection.skipped_item_count,
        "quality_filtered_count": selection.quality_filtered_count,
        "budget_dropped_count": selection.budget_dropped_count,
        "final_injected_count": selection.final_injected_count,
        "injected_context_tokens": selection.injected_context_tokens,
        "source_counts_before": _safe_int_dict(selection.source_counts_before),
        "source_counts_after": _safe_int_dict(selection.source_counts_after),
        "source_counts_dropped": _safe_int_dict(selection.source_counts_dropped),
        "drop_reasons": _safe_int_dict(selection.drop_reasons),
        "source_drop_reasons": _safe_int_dict(selection.source_drop_reasons),
        "budget_drop_reasons": _safe_int_dict(selection.budget_drop_reasons),
        "mode": selection.mode,
        "risk_tier": selection.risk_tier,
        "policy_source": selection.policy_source,
        "warnings": _sanitize_warnings(selection.warnings),
    }


def build_context_apply_policy_resolved_summary_event(
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Build a safe context_apply_policy_resolved_summary event."""
    return {
        "enabled": bool(summary.get("enabled", False)),
        "legacy_mode_enabled": bool(summary.get("legacy_mode_enabled", False)),
        "legacy_global_enabled": bool(summary.get("legacy_global_enabled", False)),
        "node_policy_enabled": bool(summary.get("node_policy_enabled", False)),
        "node_policy_schema_configured": bool(
            summary.get("node_policy_schema_configured", False)
        ),
        "node_policy_count": _safe_int(summary.get("node_policy_count")),
        "node_group_count": _safe_int(summary.get("node_group_count")),
        "resource_type_policy_count": _safe_int(
            summary.get("resource_type_policy_count")
        ),
        "default_policy_mode": sanitize_error_message(
            summary.get("default_policy_mode", ""),
            max_chars=80,
        ),
        "default_risk_tier": _safe_int(summary.get("default_risk_tier")),
        "active_nodes": _safe_string_list(summary.get("active_nodes")),
        "observe_only_nodes": _safe_string_list(summary.get("observe_only_nodes")),
        "disabled_nodes": _safe_string_list(summary.get("disabled_nodes")),
        "source_defaults": _safe_string_list(summary.get("source_defaults")),
        "importance_scoring_enabled": bool(
            summary.get("importance_scoring_enabled", False)
        ),
        "importance_scoring_shadow_mode": bool(
            summary.get("importance_scoring_shadow_mode", False)
        ),
    }


def build_context_apply_error_event(error: ContextApplyError) -> dict[str, Any]:
    """Build a safe context_apply_error event."""
    return {
        "node_name": error.node_name,
        "llm_node": error.llm_node,
        "reason": error.reason,
        "warning": error.warning,
        "fallback_used": error.fallback_used,
        "error_scope": error.error_scope,
        "recoverable": error.recoverable,
        "required_sources_missing": [
            sanitize_error_message(source, max_chars=80)
            for source in error.required_sources_missing
        ],
        "required_sources_filtered_out": [
            sanitize_error_message(source, max_chars=80)
            for source in error.required_sources_filtered_out
        ],
        "optional_sources_missing": [
            sanitize_error_message(source, max_chars=80)
            for source in error.optional_sources_missing
        ],
        "provider_missing_reasons": {
            sanitize_error_message(source, max_chars=80): sanitize_error_message(
                reason,
                max_chars=120,
            )
            for source, reason in error.provider_missing_reasons.items()
        },
        "source_drop_reasons": _safe_int_dict(error.source_drop_reasons),
        "budget_drop_reasons": _safe_int_dict(error.budget_drop_reasons),
        "source_counts_before": _safe_int_dict(error.source_counts_before),
        "source_counts_after": _safe_int_dict(error.source_counts_after),
        "source_counts_dropped": _safe_int_dict(error.source_counts_dropped),
        "error_type": error.original_exception_type or type(error).__name__,
    }


def build_context_importance_scored_event(
    *,
    node_name: str,
    llm_node: str,
    telemetry: ContextImportanceTelemetry,
) -> dict[str, Any]:
    """Build safe aggregate context_importance_scored telemetry."""
    return {
        "node_name": node_name,
        "llm_node": llm_node,
        "source_counts": _safe_int_dict(telemetry.source_counts),
        "score_buckets": _safe_int_dict(telemetry.score_buckets),
        "reason_code_counts": _safe_int_dict(telemetry.reason_code_counts),
        "candidate_count": telemetry.candidate_count,
        "scored_count": telemetry.scored_count,
        "kept_count": telemetry.kept_count,
        "dropped_count": telemetry.dropped_count,
        "fallback_to_rule_based": telemetry.fallback_to_rule_based,
        "scoring_elapsed_ms": telemetry.scoring_elapsed_ms,
        "disabled_reason": sanitize_error_message(telemetry.disabled_reason),
        "error_reason": sanitize_error_message(telemetry.error_reason),
        "error_type": sanitize_error_message(telemetry.error_type, max_chars=120),
        "warnings": _sanitize_warnings(telemetry.warnings),
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


def emit_context_apply_policy_resolved_summary(
    logger: logging.Logger,
    *,
    summary: dict[str, Any],
    state: dict | None,
) -> None:
    """Emit safe CE apply policy summary telemetry."""
    emit_a3_trace(
        logger,
        "context_apply_policy_resolved_summary",
        build_context_apply_policy_resolved_summary_event(summary),
        state=state or {},
        env_flag="LOG_A3_TRACE",
    )


def emit_context_apply_selection(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str,
    selection: ContextApplySelection,
    state: dict | None,
) -> None:
    """Emit safe context_apply_selection telemetry."""
    emit_a3_trace(
        logger,
        "context_apply_selection",
        build_context_apply_selection_event(
            node_name=node_name,
            llm_node=llm_node,
            selection=selection,
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


def emit_context_importance_scored(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str,
    telemetry: ContextImportanceTelemetry,
    state: dict | None,
) -> None:
    """Emit safe context_importance_scored telemetry."""
    emit_a3_trace(
        logger,
        "context_importance_scored",
        build_context_importance_scored_event(
            node_name=node_name,
            llm_node=llm_node,
            telemetry=telemetry,
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


def _safe_int_dict(value: dict[str, int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, int):
            continue
        result[sanitize_error_message(key, max_chars=80)] = item
    return result


def _safe_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _safe_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = sanitize_error_message(item, max_chars=120)
        if text:
            result.append(text)
    return result
