"""CE-3 production context preparation orchestrator."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, replace
from typing import Any

from src.context_engineering.packing.apply import (
    ContextApplyError,
    ContextApplyResult,
    ContextApplySelection,
    ContextInjectionPolicy,
    build_applied_messages_from_selection,
    make_context_apply_skip_selection,
    prepare_context_apply_selection,
    with_context_apply_selection_warnings,
)
from src.context_engineering.packing.apply_trace import (
    build_context_applied_event,
    build_context_apply_error_event,
    build_context_apply_plan_event,
    build_context_apply_selection_event,
)
from src.context_engineering.packing.node_policy import (
    ResolvedContextPolicy,
    resolve_context_policy,
)
from src.context_engineering.packing.packer import pack_context_items
from src.context_engineering.packing.policies import get_packing_policy, node_enabled
from src.context_engineering.packing.schema import ContextPackingError, PackedContext
from src.context_engineering.packing.source_policy import (
    SourceFilterResult,
    filter_context_items_by_source_policy,
)
from src.context_engineering.packing.trace import (
    build_context_packed_event,
    build_context_packing_error_event,
    build_context_packing_plan_event,
)
from src.context_engineering.providers.supply import (
    ContextCollectionResult,
    ProviderSupplyPlan,
    collect_context_for_policy,
    emit_context_items_collected_for_supply,
    emit_context_provider_supply,
    emit_context_provider_supply_plan,
    emit_provider_errors,
)
from src.context_engineering.schema import ContextItem, ContextSourceType
from src.observability.a3_trace import emit_a3_trace


@dataclass(frozen=True)
class ContextPreparedMessages:
    """Prepared messages for the actual LLM call plus safe CE state."""

    messages_for_llm: list[Any]
    original_messages: list[Any]
    trace_call_id: str
    next_trace_seq: int
    context_apply_applied: bool
    context_apply_fallback_used: bool
    resolved_policy: ResolvedContextPolicy | None = None
    selection: ContextApplySelection | None = None
    apply_result: ContextApplyResult | None = None


@dataclass
class _TraceSequencer:
    request_id: str
    thread_id: str
    node_name: str
    llm_node: str
    trace_call_id: str
    trace_seq: int = 0

    def next_fields(self) -> dict[str, Any]:
        self.trace_seq += 1
        return {
            "request_id": self.request_id,
            "thread_id": self.thread_id,
            "node_name": self.node_name,
            "llm_node": self.llm_node,
            "trace_call_id": self.trace_call_id,
            "trace_seq": self.trace_seq,
        }


def prepare_messages_with_context_policy(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str,
    messages: list[Any],
    state: dict[str, Any] | None,
) -> ContextPreparedMessages:
    """Prepare actual LLM messages through the CE-3 fail-fast path."""
    state_payload = state or {}
    original_messages = list(messages or [])
    sequencer = _TraceSequencer(
        request_id=str(state_payload.get("request_id") or ""),
        thread_id=str(
            state_payload.get("thread_id") or state_payload.get("session_id") or ""
        ),
        node_name=node_name,
        llm_node=llm_node,
        trace_call_id=str(uuid.uuid4()),
    )
    try:
        resolved = resolve_context_policy(
            node_name=node_name,
            llm_node=llm_node,
            state=state_payload,
        )
        policy = resolved.injection_policy
        _emit_context_policy_resolved(
            logger,
            resolved=resolved,
            state=state_payload,
            trace_fields=sequencer.next_fields(),
        )
        if policy.mode == "disabled":
            selection = make_context_apply_skip_selection(
                skip_reason="node_policy_disabled",
                policy=policy,
            )
            _emit_context_apply_selection(
                logger,
                node_name=node_name,
                llm_node=llm_node,
                selection=selection,
                state=state_payload,
                trace_fields=sequencer.next_fields(),
            )
            return ContextPreparedMessages(
                messages_for_llm=original_messages,
                original_messages=original_messages,
                trace_call_id=sequencer.trace_call_id,
                next_trace_seq=sequencer.trace_seq,
                context_apply_applied=False,
                context_apply_fallback_used=False,
                resolved_policy=resolved,
                selection=selection,
            )

        plan, collection = _collect_for_policy(
            logger,
            policy=policy,
            node_name=node_name,
            llm_node=llm_node,
            messages=original_messages,
            state=state_payload,
            sequencer=sequencer,
        )
        if policy.mode == "active":
            _raise_if_required_sources_missing(
                plan=plan,
                collection=collection,
                node_name=node_name,
                llm_node=llm_node,
            )

        packed = _pack_items(
            logger,
            node_name=node_name,
            llm_node=llm_node,
            items=collection.items,
            state=state_payload,
            sequencer=sequencer,
            active=policy.mode == "active",
        )
        if packed is None:
            selection = make_context_apply_skip_selection(
                skip_reason="packed_context_missing",
                warnings=["packed_context_missing"],
                policy=policy,
            )
            _emit_empty_apply_plan(
                logger,
                node_name=node_name,
                llm_node=llm_node,
                policy=policy,
                original_messages=original_messages,
                state=state_payload,
                trace_fields=sequencer.next_fields(),
            )
            _emit_context_apply_selection(
                logger,
                node_name=node_name,
                llm_node=llm_node,
                selection=selection,
                state=state_payload,
                trace_fields=sequencer.next_fields(),
            )
            if policy.mode == "active":
                raise ContextApplyError(
                    reason="packed_context_missing",
                    warning="context packing did not produce a PackedContext",
                    node_name=node_name,
                    llm_node=llm_node,
                    error_scope="budget",
                    recoverable=False,
                )
            return ContextPreparedMessages(
                messages_for_llm=original_messages,
                original_messages=original_messages,
                trace_call_id=sequencer.trace_call_id,
                next_trace_seq=sequencer.trace_seq,
                context_apply_applied=False,
                context_apply_fallback_used=False,
                resolved_policy=resolved,
                selection=selection,
            )

        source_filter = filter_context_items_by_source_policy(
            packed.selected_items,
            injectable_sources=policy.injectable_sources,
            exclude_message_source=policy.exclude_message_source,
            source_policies=resolved.source_policies,
            state=state_payload,
        )
        _emit_context_source_filter(
            logger,
            node_name=node_name,
            llm_node=llm_node,
            result=source_filter,
            state=state_payload,
            trace_fields=sequencer.next_fields(),
        )
        if policy.mode == "active":
            _raise_if_required_sources_filtered_out(
                plan=plan,
                collection=collection,
                source_filter=source_filter,
                node_name=node_name,
                llm_node=llm_node,
            )
        _emit_context_apply_plan(
            logger,
            node_name=node_name,
            llm_node=llm_node,
            policy=policy,
            original_messages=original_messages,
            packed=packed,
            source_filter=source_filter,
            state=state_payload,
            trace_fields=sequencer.next_fields(),
        )
        selection = prepare_context_apply_selection(
            packed=packed,
            policy=policy,
            node_name=node_name,
            llm_node=llm_node,
            source_filter_result=source_filter,
        )
        if policy.mode == "observe_only":
            selection = with_context_apply_selection_warnings(
                selection,
                ["node_policy_observe_only"],
            )
            selection = replace(
                selection,
                skip_reason="node_policy_observe_only",
                rendered_context="",
            )
            _emit_context_apply_selection(
                logger,
                node_name=node_name,
                llm_node=llm_node,
                selection=selection,
                state=state_payload,
                trace_fields=sequencer.next_fields(),
            )
            return ContextPreparedMessages(
                messages_for_llm=original_messages,
                original_messages=original_messages,
                trace_call_id=sequencer.trace_call_id,
                next_trace_seq=sequencer.trace_seq,
                context_apply_applied=False,
                context_apply_fallback_used=False,
                resolved_policy=resolved,
                selection=selection,
            )

        _emit_context_apply_selection(
            logger,
            node_name=node_name,
            llm_node=llm_node,
            selection=selection,
            state=state_payload,
            trace_fields=sequencer.next_fields(),
        )
        if selection.skip_reason:
            raise ContextApplyError(
                reason=selection.skip_reason,
                warning=f"context apply skipped: {selection.skip_reason}",
                node_name=node_name,
                llm_node=llm_node,
                error_scope=_scope_for_skip_reason(selection.skip_reason),
                recoverable=False,
            )
        apply_result = build_applied_messages_from_selection(
            node_name=node_name,
            llm_node=llm_node,
            original_messages=original_messages,
            selection=selection,
        )
        if not apply_result.applied:
            raise ContextApplyError(
                reason="context_apply_empty",
                warning="active context apply produced no injected context",
                node_name=node_name,
                llm_node=llm_node,
                error_scope="source_filter",
                recoverable=False,
            )
        _emit_context_applied(
            logger,
            node_name=node_name,
            llm_node=llm_node,
            policy=policy,
            result=apply_result,
            state=state_payload,
            trace_fields=sequencer.next_fields(),
        )
        return ContextPreparedMessages(
            messages_for_llm=apply_result.final_messages,
            original_messages=original_messages,
            trace_call_id=sequencer.trace_call_id,
            next_trace_seq=sequencer.trace_seq,
            context_apply_applied=apply_result.applied,
            context_apply_fallback_used=False,
            resolved_policy=resolved,
            selection=selection,
            apply_result=apply_result,
        )
    except ContextApplyError as exc:
        _emit_context_apply_error(
            logger,
            error=exc,
            state=state_payload,
            trace_fields=sequencer.next_fields(),
        )
        raise


def _collect_for_policy(
    logger: logging.Logger,
    *,
    policy: ContextInjectionPolicy,
    node_name: str,
    llm_node: str,
    messages: list[Any],
    state: dict[str, Any],
    sequencer: _TraceSequencer,
) -> tuple[ProviderSupplyPlan, ContextCollectionResult]:
    requested_sources = _requested_sources(policy)
    plan, collection = collect_context_for_policy(
        node_name=node_name,
        llm_node=llm_node,
        messages=messages,
        state=state,
        requested_sources=requested_sources,
        required_sources=policy.required_sources,
        optional_sources=policy.optional_sources,
    )
    emit_context_provider_supply_plan(
        logger,
        node_name=node_name,
        llm_node=llm_node,
        plan=plan,
        state=state,
        trace_fields=sequencer.next_fields(),
    )
    emit_context_provider_supply(
        logger,
        node_name=node_name,
        llm_node=llm_node,
        result=collection,
        state=state,
        trace_fields=sequencer.next_fields(),
    )
    emit_provider_errors(
        logger,
        node_name=node_name,
        llm_node=llm_node,
        errors=collection.errors,
        state=state,
        trace_fields=sequencer.next_fields() if collection.errors else None,
    )
    emit_context_items_collected_for_supply(
        logger,
        node_name=node_name,
        llm_node=llm_node,
        result=collection,
        trace_top_items=10,
        state=state,
        trace_fields=sequencer.next_fields(),
    )
    return plan, collection


def _pack_items(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str,
    items: list[ContextItem],
    state: dict[str, Any],
    sequencer: _TraceSequencer,
    active: bool,
) -> PackedContext | None:
    try:
        policy = get_packing_policy(node_name=node_name, llm_node=llm_node)
        if not policy.enabled or not node_enabled(policy, node_name=node_name):
            return None
        emit_a3_trace(
            logger,
            "context_packing_plan",
            {
                **sequencer.next_fields(),
                **build_context_packing_plan_event(
                    node_name=node_name,
                    llm_node=llm_node,
                    items=items,
                    max_context_block_tokens=policy.max_context_block_tokens,
                    strategy=policy.strategy,
                ),
            },
            state=state,
            env_flag="LOG_A3_TRACE",
        )
        packed = pack_context_items(
            node_name=node_name,
            llm_node=llm_node,
            items=items,
            max_context_block_tokens=policy.max_context_block_tokens,
            strategy=policy.strategy,
            enabled_sources=policy.enabled_sources,
        )
        emit_a3_trace(
            logger,
            "context_packed",
            {
                **sequencer.next_fields(),
                **build_context_packed_event(
                    packed,
                    trace_selected_items=policy.trace_selected_items,
                    trace_dropped_items=policy.trace_dropped_items,
                ),
            },
            state=state,
            env_flag="LOG_A3_TRACE",
        )
        return packed
    except ContextPackingError as exc:
        emit_a3_trace(
            logger,
            "context_packing_error",
            {
                **sequencer.next_fields(),
                **build_context_packing_error_event(exc),
            },
            state=state,
            env_flag="LOG_A3_TRACE",
        )
        if active:
            raise ContextApplyError(
                reason=exc.reason,
                warning=exc.warning,
                node_name=node_name,
                llm_node=llm_node,
                original_exception_type=exc.original_exception_type
                or type(exc).__name__,
                error_scope="budget",
                recoverable=False,
            ) from exc
        return None


def _raise_if_required_sources_missing(
    *,
    plan: ProviderSupplyPlan,
    collection: ContextCollectionResult,
    node_name: str,
    llm_node: str,
) -> None:
    required = set(plan.required_sources)
    if not required:
        return
    present = {item.source_type for item in collection.items}
    missing = sorted(str(source) for source in required if source not in present)
    if not missing:
        return
    reasons = {
        source: collection.provider_missing_reasons.get(source, "provider_empty")
        for source in missing
    }
    optional_missing = sorted(
        set(collection.provider_missing_reasons).difference(set(missing))
    )
    raise ContextApplyError(
        reason="required_sources_missing",
        warning="required context sources are unavailable",
        node_name=node_name,
        llm_node=llm_node,
        original_exception_type="ContextProviderError",
        error_scope="provider",
        recoverable=False,
        required_sources_missing=tuple(missing),
        optional_sources_missing=tuple(optional_missing),
        provider_missing_reasons=reasons,
    )


def _raise_if_required_sources_filtered_out(
    *,
    plan: ProviderSupplyPlan,
    collection: ContextCollectionResult,
    source_filter: SourceFilterResult,
    node_name: str,
    llm_node: str,
) -> None:
    required = {str(source) for source in plan.required_sources}
    if not required:
        return
    collected_sources = {str(item.source_type) for item in collection.items}
    kept_sources = {str(item.source_type) for item in source_filter.kept_items}
    filtered_out = sorted(
        source
        for source in required
        if source in collected_sources and source not in kept_sources
    )
    if not filtered_out:
        return
    raise ContextApplyError(
        reason="required_sources_filtered_out",
        warning="required context sources were filtered out before injection",
        node_name=node_name,
        llm_node=llm_node,
        original_exception_type="ContextSourceFilterError",
        error_scope="source_filter",
        recoverable=False,
        required_sources_filtered_out=tuple(filtered_out),
        provider_missing_reasons=dict(collection.provider_missing_reasons),
        source_drop_reasons=dict(source_filter.source_drop_reasons),
        budget_drop_reasons=dict(source_filter.budget_drop_reasons),
        source_counts_before=dict(source_filter.source_counts_before),
        source_counts_after=dict(source_filter.source_counts_after),
        source_counts_dropped=dict(source_filter.source_counts_dropped),
    )


def _requested_sources(policy: ContextInjectionPolicy) -> tuple[ContextSourceType, ...]:
    result: list[ContextSourceType] = []
    for source in (
        *policy.required_sources,
        *policy.optional_sources,
        *policy.injectable_sources,
    ):
        if source not in result:
            result.append(source)
    return tuple(result)


def _emit_context_policy_resolved(
    logger: logging.Logger,
    *,
    resolved: ResolvedContextPolicy,
    state: dict[str, Any],
    trace_fields: dict[str, Any],
) -> None:
    emit_a3_trace(
        logger,
        "context_policy_resolved",
        {
            **trace_fields,
            "node_name": trace_fields["node_name"],
            "llm_node": trace_fields["llm_node"],
            "mode": resolved.mode,
            "risk_tier": resolved.risk_tier,
            "policy_source": resolved.policy_source,
            "required_sources": [
                str(source) for source in resolved.injection_policy.required_sources
            ],
            "optional_sources": [
                str(source) for source in resolved.injection_policy.optional_sources
            ],
            "injectable_sources": [
                str(source) for source in resolved.injection_policy.injectable_sources
            ],
            "legacy_mode_enabled": resolved.legacy_mode_enabled,
            "node_policy_enabled": resolved.node_policy_enabled,
        },
        state=state,
        env_flag="LOG_A3_TRACE",
    )


def _emit_empty_apply_plan(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str,
    policy: ContextInjectionPolicy,
    original_messages: list[Any],
    state: dict[str, Any],
    trace_fields: dict[str, Any],
) -> None:
    emit_a3_trace(
        logger,
        "context_apply_plan",
        {
            **trace_fields,
            **build_context_apply_plan_event(
                node_name=node_name,
                llm_node=llm_node,
                policy=policy,
                original_message_count=len(original_messages),
                selected_item_count=0,
                injectable_item_count=0,
                skipped_item_count=0,
            ),
        },
        state=state,
        env_flag="LOG_A3_TRACE",
    )


def _emit_context_apply_plan(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str,
    policy: ContextInjectionPolicy,
    original_messages: list[Any],
    packed: PackedContext,
    source_filter: SourceFilterResult,
    state: dict[str, Any],
    trace_fields: dict[str, Any],
) -> None:
    emit_a3_trace(
        logger,
        "context_apply_plan",
        {
            **trace_fields,
            **build_context_apply_plan_event(
                node_name=node_name,
                llm_node=llm_node,
                policy=policy,
                original_message_count=len(original_messages),
                selected_item_count=len(packed.selected_items),
                injectable_item_count=len(source_filter.kept_items),
                skipped_item_count=len(source_filter.dropped_items),
            ),
        },
        state=state,
        env_flag="LOG_A3_TRACE",
    )


def _emit_context_source_filter(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str,
    result: SourceFilterResult,
    state: dict[str, Any],
    trace_fields: dict[str, Any],
) -> None:
    emit_a3_trace(
        logger,
        "context_source_filter",
        {
            **trace_fields,
            "node_name": node_name,
            "llm_node": llm_node,
            "source_counts_before": result.source_counts_before,
            "source_counts_after": result.source_counts_after,
            "source_counts_dropped": result.source_counts_dropped,
            "drop_reasons": result.drop_reasons,
            "source_drop_reasons": result.source_drop_reasons,
            "budget_drop_reasons": result.budget_drop_reasons,
            "warnings": result.warnings,
        },
        state=state,
        env_flag="LOG_A3_TRACE",
    )


def _emit_context_apply_selection(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str,
    selection: ContextApplySelection,
    state: dict[str, Any],
    trace_fields: dict[str, Any],
) -> None:
    emit_a3_trace(
        logger,
        "context_apply_selection",
        {
            **trace_fields,
            **build_context_apply_selection_event(
                node_name=node_name,
                llm_node=llm_node,
                selection=selection,
            ),
        },
        state=state,
        env_flag="LOG_A3_TRACE",
    )


def _emit_context_applied(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str,
    policy: ContextInjectionPolicy,
    result: ContextApplyResult,
    state: dict[str, Any],
    trace_fields: dict[str, Any],
) -> None:
    emit_a3_trace(
        logger,
        "context_applied",
        {
            **trace_fields,
            **build_context_applied_event(
                node_name=node_name,
                llm_node=llm_node,
                policy=policy,
                result=result,
            ),
        },
        state=state,
        env_flag="LOG_A3_TRACE",
    )


def _emit_context_apply_error(
    logger: logging.Logger,
    *,
    error: ContextApplyError,
    state: dict[str, Any],
    trace_fields: dict[str, Any],
) -> None:
    error.fallback_used = False
    emit_a3_trace(
        logger,
        "context_apply_error",
        {**trace_fields, **build_context_apply_error_event(error)},
        state=state,
        env_flag="LOG_A3_TRACE",
    )


def _scope_for_skip_reason(skip_reason: str) -> str:
    if skip_reason in {"budget_fit_failed", "context_apply_budget_fit_failed"}:
        return "budget"
    if skip_reason in {"no_injectable_items", "quality_filtered_all"}:
        return "source_filter"
    return "policy"
