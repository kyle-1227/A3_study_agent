"""Source-aware context injection filtering for CE-2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.context_engineering.packing.node_policy import SourceBudgetPolicy
from src.context_engineering.schema import ContextItem
from src.observability.node_registry import get_node_runtime_metadata
from src.rag.course_catalog import normalize_subject

_HIGH_RISK_MATCH_SOURCES = {"memory", "trajectory", "artifact", "pipeline"}
_LOW_RISK_MATCH_SOURCES = {"rules", "curriculum", "evidence"}
_USER_ALIASES = ("user_id", "student_id", "learner_id", "profile_user_id")
_THREAD_ALIASES = ("thread_id", "session_id")
_SUBJECT_ALIASES = (
    "normalized_subject",
    "subject",
    "subject_id",
    "course_subject",
    "course_id",
    "knowledge_subject",
)
_TASK_ALIASES = ("task_id", "resource_type", "requested_resource_type", "task_type")
_PURPOSE_ALIASES = ("purpose", "purposes", "context_purpose")
_ARTIFACT_ALIASES = ("artifact_id", "file_id", "document_id")


@dataclass(frozen=True)
class SourceFilterResult:
    """Safe aggregate result from source-aware filtering."""

    kept_items: list[ContextItem]
    dropped_items: list[ContextItem]
    source_counts_before: dict[str, int]
    source_counts_after: dict[str, int]
    source_counts_dropped: dict[str, int]
    source_drop_reasons: dict[str, int]
    budget_drop_reasons: dict[str, int]
    drop_reasons: dict[str, int]
    warnings: list[str]


def filter_context_items_by_source_policy(
    items: list[ContextItem],
    *,
    injectable_sources: tuple[str, ...],
    exclude_message_source: bool,
    source_policies: dict[str, SourceBudgetPolicy],
    state: dict | None,
    policy_mode: str = "strict",
    target_node_name: str = "",
    existing_content_fingerprints: set[str] | None = None,
) -> SourceFilterResult:
    """Filter candidate items by source policy without inspecting new backends."""
    state_values = _state_match_values(state)
    allowed_sources = {str(source) for source in injectable_sources}
    source_drop_reasons: dict[str, int] = {}
    budget_drop_reasons: dict[str, int] = {}
    warnings: list[str] = []
    prelim_kept: list[ContextItem] = []
    dropped: list[ContextItem] = []

    for item in items:
        source = str(item.source_type)
        policy = source_policies.get(source)
        reason = _first_source_drop_reason(
            item,
            allowed_sources=allowed_sources,
            exclude_message_source=exclude_message_source,
            policy=policy,
            state_values=state_values,
            state=state or {},
            policy_mode=policy_mode,
            target_node_name=target_node_name,
            existing_content_fingerprints=existing_content_fingerprints or set(),
            warnings=warnings,
        )
        if reason:
            dropped.append(item)
            _increment(source_drop_reasons, reason)
            continue
        prelim_kept.append(item)

    final_kept, budget_dropped = _apply_source_budgets(
        prelim_kept,
        source_policies=source_policies,
        budget_drop_reasons=budget_drop_reasons,
    )
    dropped.extend(budget_dropped)
    return SourceFilterResult(
        kept_items=final_kept,
        dropped_items=dropped,
        source_counts_before=_source_counts(items),
        source_counts_after=_source_counts(final_kept),
        source_counts_dropped=_source_counts(dropped),
        source_drop_reasons=source_drop_reasons,
        budget_drop_reasons=budget_drop_reasons,
        drop_reasons=_merge_reason_counts(source_drop_reasons, budget_drop_reasons),
        warnings=_unique(warnings),
    )


def _first_source_drop_reason(
    item: ContextItem,
    *,
    allowed_sources: set[str],
    exclude_message_source: bool,
    policy: SourceBudgetPolicy | None,
    state_values: dict[str, str],
    state: dict[str, Any],
    policy_mode: str,
    target_node_name: str,
    existing_content_fingerprints: set[str],
    warnings: list[str],
) -> str:
    source = str(item.source_type)
    if exclude_message_source and source == "message":
        return "message_source_excluded"
    if source not in allowed_sources:
        return "source_not_allowed"
    if source == "evidence" and item.metadata.get("grounding_approved") is not True:
        return "grounding_not_approved"
    if source == "pipeline":
        pipeline_reason = _pipeline_safety_drop_reason(
            item,
            state=state,
            target_node_name=target_node_name,
        )
        if pipeline_reason:
            return pipeline_reason
        fingerprint = str(item.metadata.get("content_fingerprint") or "").strip()
        if fingerprint and fingerprint in existing_content_fingerprints:
            return "duplicate_provider_input"
    if (
        policy_mode == "strict"
        and source == "evidence"
        and item.relevance_score is None
    ):
        return "missing_required_relevance_score"
    if policy is None:
        return ""
    if policy_mode == "broad":
        warnings.append("broad_business_filters_bypassed")
        return _match_drop_reason(
            item,
            policy=policy,
            state_values=state_values,
            warnings=warnings,
            match_keys=("user", "thread"),
        )
    if policy.min_priority is not None and item.priority < policy.min_priority:
        return "quality_below_threshold"
    if policy.min_relevance_score is not None:
        if (
            item.relevance_score is None
            or item.relevance_score < policy.min_relevance_score
        ):
            return "quality_below_threshold"
    if policy.min_trust_level is not None:
        trust_level = _metadata_ratio(item.metadata, "trust_level")
        if trust_level is None or trust_level < policy.min_trust_level:
            return "trust_too_low"
    purpose_reason = _purpose_drop_reason(item, policy=policy, warnings=warnings)
    if purpose_reason:
        return purpose_reason
    stale_reason = _stale_drop_reason(item, policy=policy)
    if stale_reason:
        return stale_reason
    match_reason = _match_drop_reason(
        item,
        policy=policy,
        state_values=state_values,
        warnings=warnings,
    )
    if match_reason:
        return match_reason
    return ""


def _purpose_drop_reason(
    item: ContextItem,
    *,
    policy: SourceBudgetPolicy,
    warnings: list[str],
) -> str:
    if not policy.allowed_purposes:
        return ""
    item_purposes = set(get_metadata_values(item, _PURPOSE_ALIASES))
    if not item_purposes:
        warnings.append("missing_purpose_metadata")
        return ""
    if item_purposes.isdisjoint(policy.allowed_purposes):
        return "purpose_not_allowed"
    return ""


def _stale_drop_reason(item: ContextItem, *, policy: SourceBudgetPolicy) -> str:
    if not _is_stale(item):
        return ""
    if policy.stale_policy == "drop":
        return "stale_context"
    return ""


def _match_drop_reason(
    item: ContextItem,
    *,
    policy: SourceBudgetPolicy,
    state_values: dict[str, str],
    warnings: list[str],
    match_keys: tuple[str, ...] = ("user", "thread", "subject", "task"),
) -> str:
    checks = (
        ("user", policy.require_user_match, "user_mismatch"),
        ("thread", policy.require_thread_match, "thread_mismatch"),
        ("subject", policy.require_subject_match, "subject_mismatch"),
        ("task", policy.require_task_match, "task_mismatch"),
    )
    for key, required, reason in checks:
        if key not in match_keys:
            continue
        if not required:
            continue
        item_value = _metadata_match_value(item, key)
        state_value = state_values.get(key, "")
        if key == "subject":
            item_value = normalize_subject(item_value) if item_value else ""
            state_value = normalize_subject(state_value) if state_value else ""
        if item_value and state_value and item_value == state_value:
            continue
        if item_value and state_value and item_value != state_value:
            return reason
        source = str(item.source_type)
        if source in _HIGH_RISK_MATCH_SOURCES or policy.strict_match:
            return reason
        if source in _LOW_RISK_MATCH_SOURCES:
            warnings.append(f"missing_{key}_match_metadata")
            continue
        warnings.append(f"missing_{key}_match_metadata")
    return ""


def _pipeline_safety_drop_reason(
    item: ContextItem,
    *,
    state: dict[str, Any],
    target_node_name: str,
) -> str:
    source_node = str(item.metadata.get("source_node") or "").strip()
    source_metadata = get_node_runtime_metadata(source_node)
    target_metadata = get_node_runtime_metadata(target_node_name)
    if source_metadata is None or target_metadata is None:
        return "pipeline_stage_metadata_missing"
    if source_metadata.node_id == target_metadata.node_id:
        return "pipeline_self_output"

    item_request_id = str(item.metadata.get("request_id") or "").strip()
    current_request_id = str(state.get("request_id") or "").strip()
    if item_request_id and current_request_id and item_request_id != current_request_id:
        return "pipeline_request_mismatch"
    if (
        source_metadata.workflow
        and target_metadata.workflow
        and source_metadata.workflow != target_metadata.workflow
    ):
        return "pipeline_workflow_mismatch"

    source_iteration = _non_negative_metadata_int(item.metadata, "iteration")
    target_iteration = _target_iteration(target_metadata.iteration_field, state)
    if source_iteration == target_iteration:
        if source_metadata.stage_rank < target_metadata.stage_rank:
            return ""
        if source_metadata.role == "reviewer" and (
            target_metadata.operation == "rewrite"
            or target_metadata.role == "consensus"
        ):
            return ""
        return "pipeline_future_output"
    if (
        source_iteration + 1 == target_iteration
        and source_metadata.role == "reviewer"
        and target_metadata.role == "agent"
    ):
        return ""
    return "pipeline_iteration_mismatch"


def _target_iteration(iteration_field: str, state: dict[str, Any]) -> int:
    if not iteration_field:
        return 0
    value = state.get(iteration_field)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _non_negative_metadata_int(metadata: dict[str, Any], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _apply_source_budgets(
    items: list[ContextItem],
    *,
    source_policies: dict[str, SourceBudgetPolicy],
    budget_drop_reasons: dict[str, int],
) -> tuple[list[ContextItem], list[ContextItem]]:
    kept: list[ContextItem] = []
    dropped: list[ContextItem] = []
    by_source: dict[str, list[ContextItem]] = {}
    source_order: list[str] = []
    for item in items:
        source = str(item.source_type)
        if source not in by_source:
            by_source[source] = []
            source_order.append(source)
        by_source[source].append(item)

    for source in source_order:
        policy = source_policies.get(source)
        source_items = sorted(
            by_source[source],
            key=lambda item: _source_budget_sort_key(item, policy),
        )
        token_total = 0
        item_count = 0
        for item in source_items:
            if policy is not None and policy.max_items is not None:
                if item_count >= policy.max_items:
                    dropped.append(item)
                    _increment(budget_drop_reasons, "source_budget_exceeded")
                    continue
            if policy is not None and policy.max_tokens is not None:
                if token_total + item.token_estimate > policy.max_tokens:
                    dropped.append(item)
                    _increment(budget_drop_reasons, "token_budget_exceeded")
                    continue
            kept.append(item)
            item_count += 1
            token_total += item.token_estimate
    return kept, dropped


def _source_budget_sort_key(
    item: ContextItem,
    policy: SourceBudgetPolicy | None,
) -> tuple[Any, ...]:
    stale_rank = (
        1
        if policy is not None and policy.stale_policy == "downrank" and _is_stale(item)
        else 0
    )
    return (
        stale_rank,
        -item.priority,
        -(item.relevance_score or 0.0),
        -(_metadata_ratio(item.metadata, "trust_level") or 0.0),
        -(_metadata_ratio(item.metadata, "freshness") or 0.0),
        -(item.confidence or 0.0),
        item.token_estimate,
        item.id,
    )


def _state_match_values(state: dict | None) -> dict[str, str]:
    if not isinstance(state, dict):
        return {"user": "", "thread": "", "subject": "", "task": ""}
    resource_task = state.get("resource_task")
    subject = _first_value_from_mapping(state, _SUBJECT_ALIASES)
    if isinstance(resource_task, dict):
        subject = subject or str(resource_task.get("subject") or "").strip()
    return {
        "user": _first_value_from_mapping(state, _USER_ALIASES),
        "thread": _first_value_from_mapping(state, _THREAD_ALIASES),
        "subject": normalize_subject(subject) if subject else "",
        "task": _state_task_match_value(state),
    }


def _state_task_match_value(state: dict[str, Any]) -> str:
    resource_task = state.get("resource_task")
    if isinstance(resource_task, dict):
        resource_type = str(resource_task.get("resource_type") or "").strip()
        if resource_type:
            return resource_type
    requested_resource_types = _string_list(state.get("requested_resource_types"))
    if len(requested_resource_types) > 1:
        return ""
    requested_resource_type = str(state.get("requested_resource_type") or "").strip()
    if requested_resource_type and (
        not requested_resource_types
        or requested_resource_types == [requested_resource_type]
    ):
        return requested_resource_type
    if len(requested_resource_types) == 1:
        return requested_resource_types[0]
    return _first_value_from_mapping(state, ("task_id", "resource_type", "task_type"))


def _metadata_match_value(item: ContextItem, key: str) -> str:
    candidates = {
        "user": _USER_ALIASES,
        "thread": _THREAD_ALIASES,
        "subject": _SUBJECT_ALIASES,
        "task": _TASK_ALIASES,
    }[key]
    values = get_metadata_values(item, candidates)
    if values:
        return values[0]
    return ""


def _string_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item or "").strip() for item in value if str(item or "").strip())


def get_metadata_values(item: ContextItem, aliases: tuple[str, ...]) -> tuple[str, ...]:
    """Return non-empty metadata values for aliases in declaration order."""
    values: list[str] = []
    for alias in aliases:
        raw_value = item.metadata.get(alias)
        raw_values = raw_value if isinstance(raw_value, list) else [raw_value]
        for raw_item in raw_values:
            text = str(raw_item or "").strip()
            if text and text not in values:
                values.append(text)
    return tuple(values)


def _first_value_from_mapping(
    metadata: dict[str, Any],
    aliases: tuple[str, ...],
) -> str:
    for alias in aliases:
        value = metadata.get(alias)
        if isinstance(value, list):
            for item in value:
                text = str(item or "").strip()
                if text:
                    return text
            continue
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _metadata_ratio(metadata: dict[str, Any], key: str) -> float | None:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    ratio = float(value)
    if ratio < 0.0 or ratio > 1.0:
        return None
    return ratio


def _is_stale(item: ContextItem) -> bool:
    stale = item.metadata.get("stale")
    if isinstance(stale, bool):
        return stale
    freshness = _metadata_ratio(item.metadata, "freshness")
    return freshness is not None and freshness <= 0.0


def _source_counts(items: list[ContextItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        source = str(item.source_type)
        counts[source] = counts.get(source, 0) + 1
    return counts


def _merge_reason_counts(
    *reason_maps: dict[str, int],
) -> dict[str, int]:
    merged: dict[str, int] = {}
    for reasons in reason_maps:
        for reason, count in reasons.items():
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                continue
            merged[reason] = merged.get(reason, 0) + count
    return merged


def _increment(counts: dict[str, int], reason: str) -> None:
    counts[reason] = counts.get(reason, 0) + 1


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
