"""Unified manifest and context usage observation for provider-bound inputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from src.context_engineering.budget import get_context_engineering_config
from src.context_engineering.input_accounting import build_llm_input_accounting
from src.context_engineering.input_manifest import (
    LLMInputManifest,
    build_llm_input_manifest,
)
from src.context_engineering.schema import (
    ContextConfigError,
    ContextItem,
    ContextUsageError,
)
from src.observability.a3_trace import emit_a3_trace
from src.observability.context_usage_report import (
    build_context_usage_report,
    context_usage_report_error_payload,
    legacy_context_usage_payload,
)
from src.observability.contracts import ContextUsageReport


@dataclass(frozen=True)
class LLMInputObservation:
    manifest: LLMInputManifest
    context_usage_report: ContextUsageReport | None
    legacy_context_usage: dict[str, Any] | None
    context_usage_error: dict[str, Any] | None
    blocking_error: ContextConfigError | ContextUsageError | None = None


def build_llm_input_observation(
    *,
    node_name: str,
    llm_node: str,
    provider: str,
    model: str,
    messages: list[Any],
    state: Mapping[str, Any] | None,
    call_purpose: str,
    output_mode: str = "",
    schema_name: str = "",
    schema_size_chars: int | None = None,
    context_apply_applied: bool = False,
    context_apply_status: str = "",
    optional_sources_missing: Iterable[str] = (),
    provider_input_budget_tokens: int = 0,
    provider_input_tokens_before_context: int = 0,
    provider_remaining_input_tokens: int = 0,
    effective_context_budget_tokens: int = 0,
    schema_contract_first: bool = False,
    provider_bound_messages_mutated: bool = False,
    trace_call_id: str = "",
    trace_seq: int = 0,
    context_items: Iterable[ContextItem] = (),
    reserved_output_tokens: int | None = None,
) -> LLMInputObservation:
    """Build all provider-input observability from one accounting pass."""
    safe_context_items = tuple(context_items)
    accounting = build_llm_input_accounting(messages)
    manifest = build_llm_input_manifest(
        node_name=node_name,
        llm_node=llm_node,
        provider=provider,
        model=model,
        messages=messages,
        state=state,
        call_purpose=call_purpose,
        output_mode=output_mode,
        schema_name=schema_name,
        schema_size_chars=schema_size_chars,
        context_apply_applied=context_apply_applied,
        context_apply_status=context_apply_status,
        optional_sources_missing=optional_sources_missing,
        provider_input_budget_tokens=provider_input_budget_tokens,
        provider_input_tokens_before_context=provider_input_tokens_before_context,
        provider_remaining_input_tokens=provider_remaining_input_tokens,
        effective_context_budget_tokens=effective_context_budget_tokens,
        schema_contract_first=schema_contract_first,
        provider_bound_messages_mutated=provider_bound_messages_mutated,
        trace_call_id=trace_call_id,
        trace_seq=trace_seq,
        accounting=accounting,
        context_items=safe_context_items,
    )
    try:
        report = build_context_usage_report(
            manifest=manifest,
            accounting=accounting,
            context_items=safe_context_items,
            reserved_output_tokens=reserved_output_tokens,
            schema_size_chars=schema_size_chars,
        )
    except (ContextConfigError, ContextUsageError) as exc:
        config = get_context_engineering_config()
        return LLMInputObservation(
            manifest=manifest,
            context_usage_report=None,
            legacy_context_usage=None,
            context_usage_error=context_usage_report_error_payload(
                manifest=manifest,
                exc=exc,
            ),
            blocking_error=exc if config.get("strict") is True else None,
        )
    return LLMInputObservation(
        manifest=manifest,
        context_usage_report=report,
        legacy_context_usage=legacy_context_usage_payload(report),
        context_usage_error=None,
        blocking_error=None,
    )


def raise_for_blocking_input_observation(
    observation: LLMInputObservation,
) -> None:
    """Fail before provider invocation when an input budget contract blocks it."""
    if observation.blocking_error is not None:
        raise observation.blocking_error


def emit_llm_input_usage(
    logger: logging.Logger,
    observation: LLMInputObservation,
    *,
    state: dict | None,
    trace_call_id: str = "",
    trace_seq: int = 0,
) -> None:
    """Emit v2 usage plus the legacy projection without recounting messages."""
    trace_fields = {
        "trace_call_id": str(trace_call_id or ""),
        "trace_seq": max(0, int(trace_seq or 0)),
    }
    if observation.context_usage_report is not None:
        report_payload = observation.context_usage_report.model_dump(mode="json")
        emit_a3_trace(
            logger,
            "context_usage_report",
            {**trace_fields, **report_payload},
            state=state or {},
            env_flag="LOG_A3_TRACE",
            max_items=64,
        )
        emit_a3_trace(
            logger,
            "context_usage",
            {**trace_fields, **(observation.legacy_context_usage or {})},
            state=state or {},
            env_flag="LOG_A3_TRACE",
        )
        return
    error_payload = observation.context_usage_error or {
        "reason": "context_usage_report_unavailable",
        "warning": "context usage report unavailable",
    }
    emit_a3_trace(
        logger,
        "context_usage_report_error",
        {**trace_fields, **error_payload},
        state=state or {},
        env_flag="LOG_A3_TRACE",
    )
    emit_a3_trace(
        logger,
        "context_usage_error",
        {**trace_fields, **error_payload},
        state=state or {},
        env_flag="LOG_A3_TRACE",
    )


def emit_context_usage_trace(
    logger: logging.Logger,
    *,
    observation: LLMInputObservation,
    messages: list[Any] | None = None,
    state: dict | None,
    trace_call_id: str = "",
    trace_seq: int = 0,
) -> None:
    """Compatibility boundary that emits from the unified accounting result."""
    _ = messages
    emit_llm_input_usage(
        logger,
        observation,
        state=state,
        trace_call_id=trace_call_id,
        trace_seq=trace_seq,
    )
