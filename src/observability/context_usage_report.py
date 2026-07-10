"""Versioned context usage reporting from a single input accounting snapshot."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from src.context_engineering.budget import build_context_budget
from src.context_engineering.input_accounting import (
    AccountedMessage,
    LLMInputAccounting,
)
from src.context_engineering.schema import ContextItem
from src.context_engineering.workspace import sanitize_workspace_text, utc_now_iso
from src.observability.contracts import (
    CONTEXT_USAGE_REPORT_SCHEMA_VERSION,
    ContextMainCategory,
    ContextUsageCategory,
    ContextUsageReport,
    ContextUsageSegment,
)

CONTEXT_USAGE_REPORT_HISTORY_LIMIT = 30
CONTEXT_USAGE_REPORT_HISTORY_CHAR_LIMIT = 240_000
CONTEXT_USAGE_REPORT_SEGMENT_LIMIT = 24

_TERMINAL_MAIN_ORDER: tuple[ContextMainCategory, ...] = (
    "system_prompt",
    "tool_definitions",
    "rules",
    "skills",
    "subagent_definitions",
    "conversation",
    "unclassified",
)


def build_context_usage_report(
    *,
    manifest: Mapping[str, Any],
    accounting: LLMInputAccounting,
    context_items: Iterable[ContextItem] = (),
    reserved_output_tokens: int | None = None,
    schema_size_chars: int | None = None,
) -> ContextUsageReport:
    """Build a fully reconciled report without reading provider message content."""
    node_name = _required_text(manifest.get("node_name"), "node_name")
    llm_node = _required_text(manifest.get("llm_node"), "llm_node")
    model = _required_text(manifest.get("model"), "model")
    budget = build_context_budget(
        node_name=node_name,
        llm_node=llm_node,
        model=model,
        reserved_output_tokens=reserved_output_tokens,
    )
    items = tuple(context_items)
    raw_segments = _build_segments(
        accounting,
        context_items=items,
        schema_contract_first=bool(manifest.get("schema_contract_first")),
    )
    segments, segments_compacted = _compact_segments(raw_segments)
    main_categories = _rollup_segments(
        raw_segments,
        key="main_category",
        preferred_order=_TERMINAL_MAIN_ORDER,
    )
    detailed_categories = _rollup_segments(raw_segments, key="detailed_category")
    injected_segments = [
        segment
        for segment in segments
        if segment.provenance.get("overlap") == "injected_context"
    ]
    overlap_rollups = (
        [
            ContextUsageCategory(
                category="injected_context",
                estimated_tokens=sum(
                    segment.estimated_tokens for segment in injected_segments
                ),
                segment_count=len(injected_segments),
                message_count=len(
                    {segment.message_index for segment in injected_segments}
                ),
            )
        ]
        if injected_segments
        else []
    )
    input_tokens = accounting.input_estimated_tokens
    unclassified_tokens = sum(
        item.estimated_tokens
        for item in main_categories
        if item.category == "unclassified"
    )
    reconciliation_warnings = []
    if segments_compacted:
        reconciliation_warnings.append("segments_compacted")
    if unclassified_tokens:
        reconciliation_warnings.append("unclassified_tokens_present")
    used_tokens = input_tokens + budget.reserved_output_tokens
    used_ratio = used_tokens / budget.max_context_tokens
    identity = {
        "manifest_id": manifest.get("manifest_id"),
        "message_fingerprint": accounting.message_fingerprint,
        "reserved_output_tokens": budget.reserved_output_tokens,
        "max_context_tokens": budget.max_context_tokens,
        "segment_fingerprints": [segment.fingerprint for segment in segments],
    }
    return ContextUsageReport(
        schema_version=CONTEXT_USAGE_REPORT_SCHEMA_VERSION,
        report_id=f"context_usage:v1:{_stable_digest(identity)}",
        manifest_id=_required_text(manifest.get("manifest_id"), "manifest_id"),
        created_at=utc_now_iso(),
        request_id=_safe_text(manifest.get("request_id"), 120),
        thread_id=_safe_text(manifest.get("thread_id"), 120),
        node_name=node_name,
        llm_node=llm_node,
        provider=_safe_text(manifest.get("provider"), 120),
        model=model,
        input_estimated_tokens=input_tokens,
        output_reserved_tokens=budget.reserved_output_tokens,
        used_tokens=used_tokens,
        max_context_tokens=budget.max_context_tokens,
        available_tokens=max(budget.max_context_tokens - used_tokens, 0),
        used_ratio=round(used_ratio, 6),
        warning_level=_warning_level(
            used_ratio,
            warning_ratio=budget.warning_ratio,
            critical_ratio=budget.critical_ratio,
        ),
        estimated=True,
        tokenizer_mode=accounting.tokenizer_mode,
        message_count=accounting.message_count,
        schema_size_chars=schema_size_chars,
        main_categories=main_categories,
        detailed_categories=detailed_categories,
        overlap_rollups=overlap_rollups,
        segments=segments,
        unclassified_tokens=unclassified_tokens,
        reconciliation_ok=True,
        reconciliation_warnings=reconciliation_warnings,
    )


def legacy_context_usage_payload(report: ContextUsageReport) -> dict[str, Any]:
    """Project v2 accounting into the additive legacy context_usage contract."""
    return {
        "node_name": report.node_name,
        "llm_node": report.llm_node,
        "provider": report.provider,
        "model": report.model,
        "input_estimated_tokens": report.input_estimated_tokens,
        "reserved_output_tokens": report.output_reserved_tokens,
        "used_tokens": report.used_tokens,
        "max_context_tokens": report.max_context_tokens,
        "available_tokens": report.available_tokens,
        "used_ratio": report.used_ratio,
        "warning_level": report.warning_level,
        "estimated": report.estimated,
        "tokenizer_mode": report.tokenizer_mode,
        "message_count": report.message_count,
        "schema_size_chars": report.schema_size_chars,
        "breakdown": {
            "input_estimated_tokens": report.input_estimated_tokens,
            "reserved_output_tokens": report.output_reserved_tokens,
            **(
                {"schema_size_chars": report.schema_size_chars}
                if report.schema_size_chars is not None
                else {}
            ),
        },
    }


def context_usage_report_error_payload(
    *,
    manifest: Mapping[str, Any],
    exc: BaseException,
) -> dict[str, Any]:
    """Return a bounded diagnostic without message content or provider bodies."""
    return {
        "schema_version": "context_usage_report_error_v1",
        "manifest_id": _safe_text(manifest.get("manifest_id"), 180),
        "node_name": _safe_text(manifest.get("node_name"), 120),
        "llm_node": _safe_text(manifest.get("llm_node"), 120),
        "provider": _safe_text(manifest.get("provider"), 120),
        "model": _safe_text(manifest.get("model"), 160),
        "reason": _safe_text(getattr(exc, "reason", "report_unavailable"), 160),
        "warning": _safe_text(
            getattr(exc, "warning", "context usage report unavailable"),
            200,
        ),
        "error_type": type(exc).__name__,
    }


def merge_context_usage_report_history(
    existing: list[dict[str, Any]] | None,
    update: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Merge safe reports idempotently with deterministic item and char bounds."""
    merged: dict[str, ContextUsageReport] = {}
    for raw in [*(existing or []), *(update or [])]:
        if not isinstance(raw, Mapping):
            continue
        try:
            report = ContextUsageReport.model_validate(raw)
        except Exception:
            continue
        merged[report.report_id] = report
    ordered = sorted(
        merged.values(),
        key=lambda report: (report.created_at, report.report_id),
        reverse=True,
    )[:CONTEXT_USAGE_REPORT_HISTORY_LIMIT]
    bounded: list[dict[str, Any]] = []
    total_chars = 2
    for report in ordered:
        payload = report.model_dump(mode="json")
        item_chars = len(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        if (
            bounded
            and total_chars + item_chars > CONTEXT_USAGE_REPORT_HISTORY_CHAR_LIMIT
        ):
            continue
        if not bounded and item_chars > CONTEXT_USAGE_REPORT_HISTORY_CHAR_LIMIT:
            continue
        bounded.append(payload)
        total_chars += item_chars
    return bounded


def _build_segments(
    accounting: LLMInputAccounting,
    *,
    context_items: tuple[ContextItem, ...],
    schema_contract_first: bool,
) -> list[ContextUsageSegment]:
    last_user_index = max(
        (
            message.index
            for message in accounting.messages
            if message.role in {"human", "user"}
        ),
        default=-1,
    )
    segments: list[ContextUsageSegment] = []
    for message in accounting.messages:
        if message.contains_injected_context:
            segments.extend(_ce_segments(message, context_items=context_items))
            continue
        main, detail = _message_categories(
            message,
            schema_contract_first=schema_contract_first,
            last_user_index=last_user_index,
        )
        segments.append(
            _segment(
                message=message,
                ordinal=0,
                main_category=main,
                detailed_category=detail,
                char_count=message.char_count,
                estimated_tokens=message.estimated_tokens,
                provenance={
                    "source": "provider_message",
                    "message_role": message.role,
                },
            )
        )
    return segments


def _ce_segments(
    message: AccountedMessage,
    *,
    context_items: tuple[ContextItem, ...],
) -> list[ContextUsageSegment]:
    total_tokens = message.estimated_tokens
    if total_tokens <= 0:
        return [
            _segment(
                message=message,
                ordinal=0,
                main_category="rules",
                detailed_category="ce_wrapper",
                char_count=message.char_count,
                estimated_tokens=0,
                provenance={
                    "source": "context_engineering",
                    "overlap": "injected_context",
                },
            )
        ]
    positive_items = tuple(item for item in context_items if item.token_estimate > 0)
    allocations = _allocate_ce_tokens(total_tokens, positive_items)
    segments: list[ContextUsageSegment] = []
    allocated = 0
    for ordinal, (item, tokens) in enumerate(zip(positive_items, allocations)):
        if tokens <= 0:
            continue
        allocated += tokens
        detail = _context_item_detail(item)
        main: ContextMainCategory = (
            "rules" if item.source_type == "rules" else "conversation"
        )
        metadata = item.metadata if isinstance(item.metadata, Mapping) else {}
        provenance = {
            "source": "context_item",
            "source_type": item.source_type,
            "source_id": _safe_text(item.id, 96),
            "overlap": "injected_context",
        }
        for key in ("source_node", "purpose"):
            value = _safe_text(metadata.get(key), 96)
            if value:
                provenance[key] = value
        segments.append(
            _segment(
                message=message,
                ordinal=ordinal,
                main_category=main,
                detailed_category=detail,
                char_count=0,
                estimated_tokens=tokens,
                provenance=provenance,
                identity_suffix=item.id,
            )
        )
    remainder = total_tokens - allocated
    if remainder > 0 or not segments:
        segments.append(
            _segment(
                message=message,
                ordinal=len(segments),
                main_category="rules",
                detailed_category="ce_wrapper",
                char_count=message.char_count,
                estimated_tokens=max(remainder, 0),
                provenance={
                    "source": "context_engineering",
                    "overlap": "injected_context",
                },
            )
        )
    return segments


def _compact_segments(
    segments: list[ContextUsageSegment],
) -> tuple[list[ContextUsageSegment], bool]:
    if len(segments) <= CONTEXT_USAGE_REPORT_SEGMENT_LIMIT:
        return segments, False

    categories = sorted(
        {segment.main_category for segment in segments},
        key=lambda category: (
            _TERMINAL_MAIN_ORDER.index(category),
            category,
        ),
    )
    keep_count = max(0, CONTEXT_USAGE_REPORT_SEGMENT_LIMIT - len(categories))
    kept = list(segments[:keep_count])
    overflow = segments[keep_count:]
    grouped: dict[ContextMainCategory, list[ContextUsageSegment]] = defaultdict(list)
    for segment in overflow:
        grouped[segment.main_category].append(segment)

    for category in categories:
        group = grouped.get(category)
        if not group:
            continue
        fingerprint = _stable_digest(
            {
                "main_category": category,
                "component_fingerprints": [item.fingerprint for item in group],
            }
        )
        roles = {item.role for item in group}
        kept.append(
            ContextUsageSegment(
                segment_id=f"context_segment:v1:{fingerprint}",
                fingerprint=fingerprint,
                message_index=min(item.message_index for item in group),
                role=next(iter(roles)) if len(roles) == 1 else "mixed",
                main_category=category,
                detailed_category="segment_overflow",
                char_count=sum(item.char_count for item in group),
                estimated_tokens=sum(item.estimated_tokens for item in group),
                provenance={
                    "source": "aggregated",
                    "reason": "bounded_report",
                    "segment_count": str(len(group)),
                },
            )
        )
    return kept, True


def _allocate_ce_tokens(
    total_tokens: int,
    items: tuple[ContextItem, ...],
) -> list[int]:
    if not items:
        return []
    weights = [max(1, int(item.token_estimate)) for item in items]
    weight_total = sum(weights)
    if weight_total <= total_tokens:
        return weights
    raw = [total_tokens * weight / weight_total for weight in weights]
    result = [int(value) for value in raw]
    remainder = total_tokens - sum(result)
    order = sorted(
        range(len(items)),
        key=lambda index: (-(raw[index] - result[index]), str(items[index].id)),
    )
    for index in order[:remainder]:
        result[index] += 1
    return result


def _message_categories(
    message: AccountedMessage,
    *,
    schema_contract_first: bool,
    last_user_index: int,
) -> tuple[ContextMainCategory, str]:
    if schema_contract_first and message.index == 0:
        return "system_prompt", "schema_contract"
    if message.contains_capability_context:
        return "subagent_definitions", "capability_registry"
    if message.role in {"system", "developer"}:
        return "system_prompt", "business_system_prompt"
    if message.role in {"tool", "function"}:
        return "tool_definitions", "tool_definitions"
    if message.role in {"human", "user"}:
        return (
            "conversation",
            "original_query" if message.index == last_user_index else "recent_messages",
        )
    if message.role in {"ai", "assistant"}:
        return "conversation", "recent_messages"
    return "unclassified", "unclassified"


def _context_item_detail(item: ContextItem) -> str:
    metadata = item.metadata if isinstance(item.metadata, Mapping) else {}
    if item.source_type == "pipeline":
        influence_kind = _safe_text(metadata.get("influence_kind"), 120)
        if influence_kind == "query_rewrite":
            return "rewrite"
        if influence_kind == "retrieval_plan":
            return "retrieval_plan"
        if influence_kind == "planner_output":
            return "planner"
        if influence_kind == "reviewer_output":
            return "reviewer"
        if influence_kind == "agent_output":
            return "agent"
        if influence_kind == "consensus_output":
            return "consensus"
        return "pipeline"
    if item.source_type == "evidence":
        retrieval_mode = _safe_text(metadata.get("retrieval_mode"), 160)
        if retrieval_mode.startswith("local"):
            return "local_evidence"
        if retrieval_mode.startswith("web"):
            return "web_evidence"
        if retrieval_mode.startswith("task_workspace"):
            return "workspace"
        return "judge"
    return {
        "artifact": "artifact",
        "profile": "profile",
        "memory": "memory",
        "rules": "rules",
        "curriculum": "curriculum",
        "trajectory": "trajectory",
        "message": "recent_messages",
        "unknown": "unclassified",
    }.get(item.source_type, "unclassified")


def _segment(
    *,
    message: AccountedMessage,
    ordinal: int,
    main_category: ContextMainCategory,
    detailed_category: str,
    char_count: int,
    estimated_tokens: int,
    provenance: dict[str, str],
    identity_suffix: str = "",
) -> ContextUsageSegment:
    identity = {
        "message_fingerprint": message.fingerprint,
        "ordinal": ordinal,
        "main_category": main_category,
        "detailed_category": detailed_category,
        "identity_suffix": identity_suffix,
    }
    digest = _stable_digest(identity)
    return ContextUsageSegment(
        segment_id=f"context_segment:v1:{digest}",
        fingerprint=digest,
        message_index=message.index,
        role=message.role,
        main_category=main_category,
        detailed_category=detailed_category,
        char_count=max(0, char_count),
        estimated_tokens=max(0, estimated_tokens),
        provenance=provenance,
    )


def _rollup_segments(
    segments: list[ContextUsageSegment],
    *,
    key: str,
    preferred_order: tuple[str, ...] = (),
) -> list[ContextUsageCategory]:
    tokens: dict[str, int] = defaultdict(int)
    segment_counts: dict[str, int] = defaultdict(int)
    message_indexes: dict[str, set[int]] = defaultdict(set)
    for segment in segments:
        category = str(getattr(segment, key))
        tokens[category] += segment.estimated_tokens
        segment_counts[category] += 1
        message_indexes[category].add(segment.message_index)
    order_rank = {category: index for index, category in enumerate(preferred_order)}
    categories = sorted(
        tokens,
        key=lambda category: (order_rank.get(category, len(order_rank)), category),
    )
    return [
        ContextUsageCategory(
            category=category,
            estimated_tokens=tokens[category],
            segment_count=segment_counts[category],
            message_count=len(message_indexes[category]),
        )
        for category in categories
    ]


def _warning_level(
    used_ratio: float,
    *,
    warning_ratio: float,
    critical_ratio: float,
) -> str:
    if used_ratio > 1:
        return "overflow"
    if used_ratio >= critical_ratio:
        return "critical"
    if used_ratio >= warning_ratio:
        return "warning"
    return "ok"


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:32]


def _safe_text(value: object, max_chars: int) -> str:
    return sanitize_workspace_text(value, max_chars=max_chars, fallback="")


def _required_text(value: object, field: str) -> str:
    text = _safe_text(value, 180)
    if not text:
        raise ValueError(f"{field} is required")
    return text
