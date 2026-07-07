"""Policy-driven provider supply for CE-3 production context collection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.context_engineering.evidence_normalizer import (
    EvidenceNormalizationStats,
    normalize_evidence_items,
)
from src.context_engineering.providers.base import ContextProvider, ProviderContext
from src.context_engineering.providers.registry import (
    ContextProviderSettings,
    get_context_provider_settings,
    get_default_providers,
    get_registered_provider_sources,
)
from src.context_engineering.schema import (
    ContextItem,
    ContextProviderError,
    ContextSourceType,
    sanitize_error_message,
)
from src.context_engineering.tokenizer import message_content_to_text
from src.context_engineering.trace import (
    build_context_items_collected_event,
    emit_context_provider_error,
)
from src.observability.a3_trace import emit_a3_trace

ProviderMissingReason = str

_MISSING_PROVIDER_NOT_REGISTERED = "provider_not_registered"
_MISSING_PROVIDER_DISABLED = "provider_disabled"
_MISSING_PROVIDER_EMPTY = "provider_empty"
_MISSING_PROVIDER_ERROR = "provider_error"


@dataclass(frozen=True)
class ProviderSupplyPlan:
    """Safe provider supply decision for one CE call."""

    requested_sources: tuple[ContextSourceType, ...]
    required_sources: tuple[ContextSourceType, ...]
    optional_sources: tuple[ContextSourceType, ...]
    enabled_sources: tuple[ContextSourceType, ...]
    disabled_sources: tuple[ContextSourceType, ...]
    unregistered_sources: tuple[ContextSourceType, ...]
    provider_count: int
    provider_sources_missing: dict[str, int] = field(default_factory=dict)
    provider_missing_reasons: dict[str, ProviderMissingReason] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ContextCollectionResult:
    """Collected context items plus safe missing/error telemetry."""

    items: list[ContextItem]
    provider_count: int
    provider_sources_missing: dict[str, int]
    provider_missing_reasons: dict[str, ProviderMissingReason]
    errors: list[ContextProviderError]
    evidence_stats: EvidenceNormalizationStats


def plan_provider_supply(
    *,
    requested_sources: tuple[ContextSourceType, ...],
    required_sources: tuple[ContextSourceType, ...],
    optional_sources: tuple[ContextSourceType, ...],
    settings: ContextProviderSettings | None = None,
) -> ProviderSupplyPlan:
    """Build a provider supply plan without invoking providers."""
    settings = settings or get_context_provider_settings()
    requested = _dedupe_sources(requested_sources)
    registered = set(get_registered_provider_sources())
    configured_enabled = set(settings.enabled_sources) if settings.enabled else set()
    enabled_sources: list[ContextSourceType] = []
    disabled_sources: list[ContextSourceType] = []
    unregistered_sources: list[ContextSourceType] = []
    missing_reasons: dict[str, ProviderMissingReason] = {}

    for source in requested:
        if source not in registered:
            unregistered_sources.append(source)
            missing_reasons[str(source)] = _MISSING_PROVIDER_NOT_REGISTERED
            continue
        if source not in configured_enabled:
            disabled_sources.append(source)
            missing_reasons[str(source)] = _MISSING_PROVIDER_DISABLED
            continue
        enabled_sources.append(source)

    return ProviderSupplyPlan(
        requested_sources=requested,
        required_sources=_dedupe_sources(required_sources),
        optional_sources=_dedupe_sources(optional_sources),
        enabled_sources=tuple(enabled_sources),
        disabled_sources=tuple(disabled_sources),
        unregistered_sources=tuple(unregistered_sources),
        provider_count=len(enabled_sources),
        provider_sources_missing={source: 1 for source in missing_reasons},
        provider_missing_reasons=missing_reasons,
    )


def collect_context_for_policy(
    *,
    node_name: str,
    llm_node: str,
    messages: list[Any],
    state: dict[str, Any],
    requested_sources: tuple[ContextSourceType, ...],
    required_sources: tuple[ContextSourceType, ...],
    optional_sources: tuple[ContextSourceType, ...],
    settings: ContextProviderSettings | None = None,
) -> tuple[ProviderSupplyPlan, ContextCollectionResult]:
    """Collect only policy-requested context from existing state/messages."""
    settings = settings or get_context_provider_settings()
    plan = plan_provider_supply(
        requested_sources=requested_sources,
        required_sources=required_sources,
        optional_sources=optional_sources,
        settings=settings,
    )
    providers_by_source = {
        provider.source_type: provider
        for provider in get_default_providers(settings)
        if provider.source_type in set(plan.enabled_sources)
    }
    user_query, current_user_message_index = _user_query_from_messages(messages)
    provider_context = ProviderContext(
        node_name=node_name,
        llm_node=llm_node,
        user_query=user_query,
        current_user_message_index=current_user_message_index,
        state=state,
        messages=list(messages or []),
        request_id=_optional_string(state.get("request_id")),
        thread_id=_optional_string(state.get("thread_id")),
        max_items_per_provider=settings.max_items_per_provider,
        max_content_chars_per_item=settings.max_content_chars_per_item,
    )

    items: list[ContextItem] = []
    errors: list[ContextProviderError] = []
    provider_missing_reasons = dict(plan.provider_missing_reasons)
    provider_sources_missing = dict(plan.provider_sources_missing)
    provider_count = 0
    for source in plan.enabled_sources:
        provider = providers_by_source.get(source)
        if provider is None:
            provider_missing_reasons[str(source)] = _MISSING_PROVIDER_DISABLED
            provider_sources_missing[str(source)] = 1
            continue
        provider_count += 1
        try:
            collected = _collect_provider(provider, provider_context)
        except ContextProviderError as exc:
            errors.append(exc)
            provider_missing_reasons[str(source)] = _MISSING_PROVIDER_ERROR
            provider_sources_missing[str(source)] = 1
            continue
        if not collected:
            provider_missing_reasons[str(source)] = _MISSING_PROVIDER_EMPTY
            provider_sources_missing[str(source)] = 1
            continue
        items.extend(collected[: settings.max_items_per_provider])

    items, evidence_stats = normalize_evidence_items(items)
    if "evidence" in plan.requested_sources and not any(
        item.source_type == "evidence" for item in items
    ):
        provider_missing_reasons["evidence"] = (
            provider_missing_reasons.get("evidence") or _MISSING_PROVIDER_EMPTY
        )
        provider_sources_missing["evidence"] = 1

    return plan, ContextCollectionResult(
        items=items,
        provider_count=provider_count,
        provider_sources_missing=provider_sources_missing,
        provider_missing_reasons=provider_missing_reasons,
        errors=errors,
        evidence_stats=evidence_stats,
    )


def emit_context_provider_supply_plan(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str,
    plan: ProviderSupplyPlan,
    state: dict[str, Any],
    trace_fields: dict[str, Any],
) -> None:
    """Emit safe provider supply plan telemetry."""
    emit_a3_trace(
        logger,
        "context_provider_supply_plan",
        {
            **trace_fields,
            "node_name": node_name,
            "llm_node": llm_node,
            "requested_sources": _safe_sources(plan.requested_sources),
            "required_sources": _safe_sources(plan.required_sources),
            "optional_sources": _safe_sources(plan.optional_sources),
            "enabled_sources": _safe_sources(plan.enabled_sources),
            "disabled_sources": _safe_sources(plan.disabled_sources),
            "unregistered_sources": _safe_sources(plan.unregistered_sources),
            "provider_count": plan.provider_count,
            "provider_sources_missing": _safe_int_dict(plan.provider_sources_missing),
            "provider_missing_reasons": _safe_reason_dict(
                plan.provider_missing_reasons
            ),
        },
        state=state,
        env_flag="LOG_A3_TRACE",
    )


def emit_context_provider_supply(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str,
    result: ContextCollectionResult,
    state: dict[str, Any],
    trace_fields: dict[str, Any],
) -> None:
    """Emit safe provider supply outcome telemetry."""
    emit_a3_trace(
        logger,
        "context_provider_supply",
        {
            **trace_fields,
            "node_name": node_name,
            "llm_node": llm_node,
            "provider_count": result.provider_count,
            "item_count": len(result.items),
            "source_counts": _source_counts(result.items),
            "provider_sources_missing": _safe_int_dict(result.provider_sources_missing),
            "provider_missing_reasons": _safe_reason_dict(
                result.provider_missing_reasons
            ),
            "provider_error_count": len(result.errors),
            **result.evidence_stats.as_event_fields(),
        },
        state=state,
        env_flag="LOG_A3_TRACE",
    )


def emit_context_items_collected_for_supply(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str,
    result: ContextCollectionResult,
    trace_top_items: int,
    state: dict[str, Any],
    trace_fields: dict[str, Any],
) -> None:
    """Emit safe context_items_collected telemetry for CE-3 collection."""
    emit_a3_trace(
        logger,
        "context_items_collected",
        {
            **trace_fields,
            **build_context_items_collected_event(
                node_name=node_name,
                llm_node=llm_node,
                provider_count=result.provider_count,
                items=result.items,
                trace_top_items=trace_top_items,
                evidence_stats=result.evidence_stats,
            ),
        },
        state=state,
        env_flag="LOG_A3_TRACE",
    )


def emit_provider_errors(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str,
    errors: list[ContextProviderError],
    state: dict[str, Any],
    trace_fields: dict[str, Any] | None = None,
) -> None:
    """Emit individual safe provider error events."""
    for error in errors:
        if trace_fields:
            emit_a3_trace(
                logger,
                "context_provider_error",
                {
                    **trace_fields,
                    "node_name": node_name,
                    "llm_node": llm_node,
                    "provider": error.provider,
                    "source_type": error.source_type,
                    "provider_stage": error.stage,
                    "error_type": error.original_exception_type or type(error).__name__,
                    "error_reason": error.sanitized_message,
                },
                state=state,
                env_flag="LOG_A3_TRACE",
            )
        else:
            emit_context_provider_error(
                logger,
                error=error,
                node_name=node_name,
                llm_node=llm_node,
                state=state,
            )


def _collect_provider(
    provider: ContextProvider,
    context: ProviderContext,
) -> list[ContextItem]:
    collected = provider.collect(context)
    if not isinstance(collected, list):
        raise ContextProviderError(
            provider=provider.name,
            source_type=provider.source_type,
            stage="collect",
            message="provider returned non-list result",
            original_exception_type="TypeError",
        )
    for item in collected:
        if not isinstance(item, ContextItem):
            raise ContextProviderError(
                provider=provider.name,
                source_type=provider.source_type,
                stage="collect",
                message="provider returned non-ContextItem item",
                original_exception_type="TypeError",
            )
    return collected


def _user_query_from_messages(messages: list[Any]) -> tuple[str, int | None]:
    for index in range(len(messages or []) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict):
            if str(message.get("role") or "").lower() == "user":
                return message_content_to_text(message).strip(), index
            continue
        class_name = type(message).__name__.lower()
        if "human" in class_name:
            return message_content_to_text(message).strip(), index
    return "", None


def _dedupe_sources(
    sources: tuple[ContextSourceType, ...],
) -> tuple[ContextSourceType, ...]:
    result: list[ContextSourceType] = []
    for source in sources:
        if source not in result:
            result.append(source)
    return tuple(result)


def _safe_sources(sources: tuple[ContextSourceType, ...]) -> list[str]:
    return [sanitize_error_message(source, max_chars=80) for source in sources]


def _source_counts(items: list[ContextItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        source = str(item.source_type)
        counts[source] = counts.get(source, 0) + 1
    return counts


def _safe_int_dict(value: dict[str, int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            continue
        result[sanitize_error_message(key, max_chars=80)] = item
    return result


def _safe_reason_dict(value: dict[str, str]) -> dict[str, str]:
    return {
        sanitize_error_message(key, max_chars=80): sanitize_error_message(
            item,
            max_chars=120,
        )
        for key, item in value.items()
        if str(key or "").strip() and str(item or "").strip()
    }


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
