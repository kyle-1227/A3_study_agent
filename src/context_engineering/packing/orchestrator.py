"""CE-3 production context preparation orchestrator."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, replace
from typing import Any

from src.context_engineering.budget import build_context_budget
from src.context_engineering.influence import content_fingerprint
from src.context_engineering.packing.apply import (
    ContextApplyError,
    ContextApplyErrorScope,
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
from src.context_engineering.schema import (
    ContextConfigError,
    ContextItem,
    ContextSourceType,
)
from src.context_engineering.tokenizer import (
    estimate_messages_tokens_mixed,
    message_content_to_text,
)
from src.observability.a3_trace import emit_a3_trace


@dataclass(frozen=True)
class ContextPreparedMessages:
    """Prepared messages for the actual LLM call plus safe CE state."""

    messages_for_llm: list[Any]
    original_messages: list[Any]
    trace_call_id: str
    next_trace_seq: int
    context_apply_applied: bool
    resolved_policy: ResolvedContextPolicy | None = None
    selection: ContextApplySelection | None = None
    apply_result: ContextApplyResult | None = None
    context_apply_status: str = "skipped"
    optional_sources_missing: tuple[str, ...] = ()
    provider_input_budget_tokens: int = 0
    provider_input_tokens_before_context: int = 0
    provider_remaining_input_tokens: int = 0
    effective_context_budget_tokens: int = 0


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


@dataclass(frozen=True)
class _ProviderBudgetState:
    resolved: ResolvedContextPolicy
    provider_input_budget_tokens: int = 0
    provider_input_tokens_before_context: int = 0
    provider_remaining_input_tokens: int = 0
    effective_context_budget_tokens: int = 0


def prepare_messages_with_context_policy(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str,
    model: str,
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
        try:
            resolved = resolve_context_policy(
                node_name=node_name,
                llm_node=llm_node,
                state=state_payload,
            )
        except ContextConfigError as exc:
            raise ContextApplyError(
                reason=exc.reason,
                warning=exc.warning,
                node_name=node_name,
                llm_node=llm_node,
                original_exception_type=type(exc).__name__,
                error_scope="config",
                recoverable=False,
            ) from exc
        budget_state = _ProviderBudgetState(resolved=resolved)
        if resolved.mode == "active":
            budget_state = _constrain_policy_to_provider_budget(
                resolved=resolved,
                node_name=node_name,
                llm_node=llm_node,
                model=model,
                messages=original_messages,
            )
        resolved = budget_state.resolved
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
                resolved_policy=resolved,
                selection=selection,
                context_apply_status="skipped",
                provider_input_budget_tokens=(
                    budget_state.provider_input_budget_tokens
                ),
                provider_input_tokens_before_context=(
                    budget_state.provider_input_tokens_before_context
                ),
                provider_remaining_input_tokens=(
                    budget_state.provider_remaining_input_tokens
                ),
                effective_context_budget_tokens=(
                    budget_state.effective_context_budget_tokens
                ),
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
                resolved_policy=resolved,
                selection=selection,
                context_apply_status=(
                    "observed" if policy.mode == "observe_only" else "skipped"
                ),
                provider_input_budget_tokens=(
                    budget_state.provider_input_budget_tokens
                ),
                provider_input_tokens_before_context=(
                    budget_state.provider_input_tokens_before_context
                ),
                provider_remaining_input_tokens=(
                    budget_state.provider_remaining_input_tokens
                ),
                effective_context_budget_tokens=(
                    budget_state.effective_context_budget_tokens
                ),
            )

        source_filter = filter_context_items_by_source_policy(
            packed.selected_items,
            injectable_sources=policy.injectable_sources,
            exclude_message_source=policy.exclude_message_source,
            source_policies=resolved.source_policies,
            state=state_payload,
            policy_mode=resolved.runtime_policy_mode,
            target_node_name=node_name,
            existing_content_fingerprints=_message_fingerprints(original_messages),
        )
        _emit_context_source_filter(
            logger,
            node_name=node_name,
            llm_node=llm_node,
            result=source_filter,
            policy_mode=resolved.runtime_policy_mode,
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
                resolved_policy=resolved,
                selection=selection,
                context_apply_status="observed",
                provider_input_budget_tokens=(
                    budget_state.provider_input_budget_tokens
                ),
                provider_input_tokens_before_context=(
                    budget_state.provider_input_tokens_before_context
                ),
                provider_remaining_input_tokens=(
                    budget_state.provider_remaining_input_tokens
                ),
                effective_context_budget_tokens=(
                    budget_state.effective_context_budget_tokens
                ),
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
        optional_provider_errors = _optional_provider_error_sources(
            plan=plan,
            collection=collection,
        )
        apply_status = "applied"
        if optional_provider_errors:
            apply_status = "degraded_applied"
            apply_result = replace(
                apply_result,
                apply_status=apply_status,
                warnings=list(
                    dict.fromkeys(
                        [
                            *apply_result.warnings,
                            "optional_provider_error",
                            *(
                                f"optional_provider_error:{source}"
                                for source in optional_provider_errors
                            ),
                        ]
                    )
                ),
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
            resolved_policy=resolved,
            selection=selection,
            apply_result=apply_result,
            context_apply_status=apply_status,
            optional_sources_missing=optional_provider_errors,
            provider_input_budget_tokens=budget_state.provider_input_budget_tokens,
            provider_input_tokens_before_context=(
                budget_state.provider_input_tokens_before_context
            ),
            provider_remaining_input_tokens=(
                budget_state.provider_remaining_input_tokens
            ),
            effective_context_budget_tokens=(
                budget_state.effective_context_budget_tokens
            ),
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
    _emit_workspace_context_collected(
        logger,
        node_name=node_name,
        llm_node=llm_node,
        collection=collection,
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


def _constrain_policy_to_provider_budget(
    *,
    resolved: ResolvedContextPolicy,
    node_name: str,
    llm_node: str,
    model: str,
    messages: list[Any],
) -> _ProviderBudgetState:
    """Constrain CE injection to the provider's actual remaining input window."""
    try:
        budget = build_context_budget(
            node_name=node_name,
            llm_node=llm_node,
            model=model,
        )
    except ContextConfigError as exc:
        raise ContextApplyError(
            reason=exc.reason,
            warning=exc.warning,
            node_name=node_name,
            llm_node=llm_node,
            original_exception_type=type(exc).__name__,
            error_scope="config",
            recoverable=False,
        ) from exc

    input_tokens = estimate_messages_tokens_mixed(messages)
    remaining_tokens = budget.max_input_tokens - input_tokens
    if remaining_tokens <= 0:
        raise ContextApplyError(
            reason="provider_context_budget_exceeded",
            warning="provider input budget is exhausted before context injection",
            node_name=node_name,
            llm_node=llm_node,
            original_exception_type="ContextBudgetError",
            error_scope="budget",
            recoverable=False,
        )

    effective_tokens = min(
        resolved.injection_policy.max_injected_context_tokens,
        remaining_tokens,
    )
    if effective_tokens <= 0:
        raise ContextApplyError(
            reason="provider_context_budget_exceeded",
            warning="no provider input budget remains for context injection",
            node_name=node_name,
            llm_node=llm_node,
            original_exception_type="ContextBudgetError",
            error_scope="budget",
            recoverable=False,
        )

    constrained_policy = replace(
        resolved.injection_policy,
        max_injected_context_tokens=effective_tokens,
    )
    return _ProviderBudgetState(
        resolved=replace(resolved, injection_policy=constrained_policy),
        provider_input_budget_tokens=budget.max_input_tokens,
        provider_input_tokens_before_context=input_tokens,
        provider_remaining_input_tokens=remaining_tokens,
        effective_context_budget_tokens=effective_tokens,
    )


def _message_fingerprints(messages: list[Any]) -> set[str]:
    """Return deterministic content fingerprints for already-bound messages."""
    return {
        content_fingerprint(content)
        for message in messages
        if (content := message_content_to_text(message).strip())
    }


def _optional_provider_error_sources(
    *,
    plan: ProviderSupplyPlan,
    collection: ContextCollectionResult,
) -> tuple[str, ...]:
    """Return optional sources whose providers failed during collection."""
    required = {str(source) for source in plan.required_sources}
    optional = {str(source) for source in plan.optional_sources}
    return tuple(
        sorted(
            source
            for source, reason in collection.provider_missing_reasons.items()
            if source in optional
            and source not in required
            and reason == "provider_error"
        )
    )


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
            "runtime_policy_mode": resolved.runtime_policy_mode,
            "runtime_environment": resolved.runtime_environment,
            "effective_context_budget_tokens": (
                resolved.injection_policy.max_injected_context_tokens
            ),
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
    policy_mode: str,
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
            "policy_mode": policy_mode,
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


def _emit_workspace_context_collected(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str,
    collection: ContextCollectionResult,
    state: dict[str, Any],
    trace_fields: dict[str, Any],
) -> None:
    workspace_items = [item for item in collection.items if _item_from_workspace(item)]
    if not workspace_items:
        return
    source_counts: dict[str, int] = {}
    for item in workspace_items:
        source = str(item.source_type)
        source_counts[source] = source_counts.get(source, 0) + 1
    emit_a3_trace(
        logger,
        "workspace_context.collected",
        {
            **trace_fields,
            "node_name": node_name,
            "llm_node": llm_node,
            "workspace_item_count": len(workspace_items),
            "workspace_source_counts": source_counts,
        },
        state=state,
        env_flag="LOG_A3_TRACE",
    )


def _item_from_workspace(item: ContextItem) -> bool:
    metadata = item.metadata or {}
    retrieval_mode = str(metadata.get("retrieval_mode") or "")
    artifact_source = str(metadata.get("artifact_source") or "")
    return bool(
        metadata.get("workspace_id")
        or retrieval_mode.startswith("task_workspace")
        or artifact_source.startswith("task_workspace")
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
    emit_a3_trace(
        logger,
        "context_apply_error",
        {**trace_fields, **build_context_apply_error_event(error)},
        state=state,
        env_flag="LOG_A3_TRACE",
    )


def _scope_for_skip_reason(skip_reason: str) -> ContextApplyErrorScope:
    if skip_reason in {"budget_fit_failed", "context_apply_budget_fit_failed"}:
        return "budget"
    if skip_reason in {"no_injectable_items", "quality_filtered_all"}:
        return "source_filter"
    return "policy"
