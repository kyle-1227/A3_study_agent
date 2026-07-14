"""Node-aware context apply policy resolver for CE-1."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from src.config import get_setting
from src.context_engineering.packing.apply import (
    ContextApplyError,
    ContextInjectionPolicy,
    get_context_injection_policy,
)
from src.context_engineering.policy_mode import (
    ContextRuntimePolicy,
    resolve_context_runtime_policy,
)
from src.context_engineering.schema import ContextSourceType, sanitize_error_message
from src.observability.node_registry import get_node_runtime_metadata

ContextPolicyMode = Literal["disabled", "observe_only", "active"]
PolicySource = Literal[
    "node_policy",
    "node_group",
    "resource_type_policy",
    "default_policy",
    "legacy_global",
    "disabled_global",
    "config_error_fallback",
]

_VALID_MODES = {"disabled", "observe_only", "active"}
_VALID_POLICY_SOURCES = {
    "node_policy",
    "node_group",
    "resource_type_policy",
    "default_policy",
    "legacy_global",
    "disabled_global",
    "config_error_fallback",
}
_ALLOWED_STALE_POLICIES = {"keep", "drop", "downrank"}
_ALLOWED_SOURCES = {
    "message",
    "memory",
    "evidence",
    "artifact",
    "profile",
    "trajectory",
    "rules",
    "curriculum",
    "pipeline",
    "unknown",
}


@dataclass(frozen=True)
class SourceBudgetPolicy:
    """Per-source admission and budget policy."""

    source_type: ContextSourceType
    max_items: int | None = None
    max_tokens: int | None = None
    min_priority: int | None = None
    min_relevance_score: float | None = None
    min_trust_level: float | None = None
    allowed_purposes: tuple[str, ...] = ()
    require_user_match: bool = False
    require_thread_match: bool = False
    require_subject_match: bool = False
    require_task_match: bool = False
    strict_match: bool = False
    stale_policy: str = "keep"


@dataclass(frozen=True)
class NodeContextPolicy:
    """Resolved node-level context apply policy before legacy adaptation."""

    mode: ContextPolicyMode
    risk_tier: int
    max_injected_context_tokens: int
    max_items_total: int
    min_injectable_items: int
    injectable_sources: tuple[ContextSourceType, ...]
    required_sources: tuple[ContextSourceType, ...]
    optional_sources: tuple[ContextSourceType, ...]
    exclude_message_source: bool
    source_overrides: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ResolvedContextPolicy:
    """Final policy used by the plain LLM integration."""

    mode: ContextPolicyMode
    risk_tier: int
    policy_source: PolicySource
    injection_policy: ContextInjectionPolicy
    source_policies: dict[str, SourceBudgetPolicy]
    legacy_mode_enabled: bool
    node_policy_enabled: bool
    summary: dict[str, Any]
    runtime_policy_mode: str = "strict"
    runtime_environment: str = "unspecified"


def resolve_context_policy(
    *,
    node_name: str,
    llm_node: str,
    state: dict | None,
) -> ResolvedContextPolicy:
    """Resolve node-aware context policy from settings."""
    runtime_policy = resolve_context_runtime_policy()
    legacy_policy = get_context_injection_policy(
        node_name=node_name,
        llm_node=llm_node,
    )
    apply_config = _apply_config()
    summary = build_context_policy_summary(apply_config)
    if not legacy_policy.enabled:
        disabled_policy = replace(
            legacy_policy,
            mode="disabled",
            risk_tier=0,
            policy_source="disabled_global",
        )
        return _apply_runtime_policy(
            ResolvedContextPolicy(
                mode="disabled",
                risk_tier=0,
                policy_source="disabled_global",
                injection_policy=disabled_policy,
                source_policies={},
                legacy_mode_enabled=False,
                node_policy_enabled=_node_policy_schema_configured(apply_config),
                summary=summary,
            ),
            runtime_policy=runtime_policy,
            node_name=node_name,
        )

    if not _node_policy_schema_configured(apply_config):
        return _apply_runtime_policy(
            ResolvedContextPolicy(
                mode="active",
                risk_tier=1,
                policy_source="legacy_global",
                injection_policy=replace(
                    legacy_policy,
                    mode="active",
                    risk_tier=1,
                    policy_source="legacy_global",
                ),
                source_policies=_source_policies_from_config(
                    source_defaults={},
                    source_overrides={},
                    injectable_sources=legacy_policy.injectable_sources,
                    node_name=node_name,
                    llm_node=llm_node,
                ),
                legacy_mode_enabled=True,
                node_policy_enabled=False,
                summary=summary,
            ),
            runtime_policy=runtime_policy,
            node_name=node_name,
        )

    raw_policy, policy_source = _raw_node_policy(
        apply_config,
        legacy_policy=legacy_policy,
        node_name=node_name,
        llm_node=llm_node,
        state=state,
    )
    node_policy = _parse_node_policy(
        raw_policy,
        legacy_policy=legacy_policy,
        node_name=node_name,
        llm_node=llm_node,
    )
    source_policies = _source_policies_from_config(
        source_defaults=_optional_mapping(apply_config.get("source_defaults")),
        source_overrides=node_policy.source_overrides,
        injectable_sources=node_policy.injectable_sources,
        node_name=node_name,
        llm_node=llm_node,
    )
    injection_policy = _policy_from_node_policy(
        legacy_policy=legacy_policy,
        node_policy=node_policy,
        policy_source=policy_source,
        node_name=node_name,
    )
    return _apply_runtime_policy(
        ResolvedContextPolicy(
            mode=node_policy.mode,
            risk_tier=node_policy.risk_tier,
            policy_source=policy_source,
            injection_policy=injection_policy,
            source_policies=source_policies,
            legacy_mode_enabled=False,
            node_policy_enabled=True,
            summary=summary,
        ),
        runtime_policy=runtime_policy,
        node_name=node_name,
    )


def _apply_runtime_policy(
    resolved: ResolvedContextPolicy,
    *,
    runtime_policy: ContextRuntimePolicy,
    node_name: str,
) -> ResolvedContextPolicy:
    summary = {
        **resolved.summary,
        "runtime_policy_mode": runtime_policy.mode,
        "runtime_environment": runtime_policy.environment,
        "broad_max_items_total": runtime_policy.max_items_total,
        "broad_max_injected_context_tokens": (
            runtime_policy.max_injected_context_tokens
        ),
    }
    metadata = get_node_runtime_metadata(node_name)
    broad_eligible = bool(
        metadata
        and metadata.role in set(runtime_policy.eligible_node_roles)
        and resolved.mode == "active"
    )
    if runtime_policy.mode != "broad" or not broad_eligible:
        return replace(
            resolved,
            summary=summary,
            runtime_policy_mode=runtime_policy.mode,
            runtime_environment=runtime_policy.environment,
        )

    policy = resolved.injection_policy
    sources = _dedupe_context_sources(
        (
            *policy.required_sources,
            *runtime_policy.enabled_sources,
            *policy.injectable_sources,
        )
    )
    optional_sources = _dedupe_context_sources(
        (
            *policy.optional_sources,
            *(source for source in sources if source not in policy.required_sources),
        )
    )
    source_policies: dict[str, SourceBudgetPolicy] = {}
    for source in sources:
        source_text = str(source)
        broad_source = runtime_policy.source_policies.get(source_text)
        if broad_source is None:
            continue
        strict_source = resolved.source_policies.get(source_text)
        source_policies[source_text] = SourceBudgetPolicy(
            source_type=source,
            max_items=broad_source.max_items,
            max_tokens=broad_source.max_tokens,
            require_user_match=(
                broad_source.require_user_match
                or bool(strict_source and strict_source.require_user_match)
            ),
            require_thread_match=(
                broad_source.require_thread_match
                or bool(strict_source and strict_source.require_thread_match)
            ),
        )
    broad_policy = replace(
        policy,
        max_injected_context_tokens=runtime_policy.max_injected_context_tokens,
        injectable_sources=sources,
        optional_sources=optional_sources,
        quality=replace(
            policy.quality,
            min_priority=0,
            min_relevance_score=None,
            max_items_total=runtime_policy.max_items_total,
            max_items_per_source={
                source: item.max_items
                for source, item in runtime_policy.source_policies.items()
            },
        ),
        budget=replace(
            policy.budget,
            graceful_degradation_enabled=True,
        ),
        format=replace(
            policy.format,
            max_content_chars_per_item=(runtime_policy.max_content_chars_per_item),
            source_order=sources,
        ),
    )
    return replace(
        resolved,
        injection_policy=broad_policy,
        source_policies=source_policies,
        summary={**summary, "broad_applied": True},
        runtime_policy_mode="broad",
        runtime_environment=runtime_policy.environment,
    )


def _dedupe_context_sources(
    values: tuple[ContextSourceType, ...],
) -> tuple[ContextSourceType, ...]:
    result: list[ContextSourceType] = []
    for source in values:
        if source not in result:
            result.append(source)
    return tuple(result)


def build_context_policy_summary(
    apply_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a safe boot/first-use summary for CE policy settings."""
    config = apply_config if apply_config is not None else _apply_config()
    enabled = isinstance(config, dict) and config.get("enabled") is True
    node_groups = _optional_mapping(config.get("node_groups")) if config else {}
    node_policies = _optional_mapping(config.get("node_policies")) if config else {}
    resource_type_policies = (
        _optional_mapping(config.get("resource_type_policies")) if config else {}
    )
    source_defaults = _optional_mapping(config.get("source_defaults")) if config else {}
    default_policy = _optional_mapping(config.get("default_policy")) if config else {}
    default_mode = str(default_policy.get("mode") or "").strip()
    default_risk_tier = default_policy.get("risk_tier", 0)
    node_policy_schema_configured = _node_policy_schema_configured(config)
    legacy_global_enabled = _legacy_global_configured(config)

    active_nodes: set[str] = set()
    observe_only_nodes: set[str] = set()
    disabled_nodes: set[str] = set()
    for group in node_groups.values():
        if not isinstance(group, dict):
            continue
        mode = str(group.get("mode") or default_policy.get("mode") or "").strip()
        nodes = _string_tuple(group.get("nodes"), path="node_groups.nodes")
        _add_nodes_by_mode(
            mode=mode,
            nodes=nodes,
            active_nodes=active_nodes,
            observe_only_nodes=observe_only_nodes,
            disabled_nodes=disabled_nodes,
        )
    for node, raw in node_policies.items():
        if not isinstance(raw, dict):
            continue
        inherited_mode = ""
        inherit = str(raw.get("inherit") or "").strip()
        group = node_groups.get(inherit)
        if isinstance(group, dict):
            inherited_mode = str(group.get("mode") or "").strip()
        mode = str(
            raw.get("mode") or inherited_mode or default_policy.get("mode") or ""
        ).strip()
        _add_nodes_by_mode(
            mode=mode,
            nodes=(str(node),),
            active_nodes=active_nodes,
            observe_only_nodes=observe_only_nodes,
            disabled_nodes=disabled_nodes,
        )

    importance = _optional_mapping(config.get("importance_scoring")) if config else {}
    return {
        "enabled": enabled,
        "legacy_mode_enabled": enabled and not node_policy_schema_configured,
        "legacy_global_enabled": legacy_global_enabled,
        "node_policy_enabled": node_policy_schema_configured,
        "node_policy_schema_configured": node_policy_schema_configured,
        "node_policy_count": len(node_policies),
        "node_group_count": len(node_groups),
        "resource_type_policy_count": len(resource_type_policies),
        "default_policy_mode": default_mode,
        "default_risk_tier": (
            default_risk_tier
            if isinstance(default_risk_tier, int)
            and not isinstance(default_risk_tier, bool)
            else 0
        ),
        "active_nodes": sorted(active_nodes),
        "observe_only_nodes": sorted(observe_only_nodes),
        "disabled_nodes": sorted(disabled_nodes),
        "source_defaults": sorted(str(key) for key in source_defaults),
        "importance_scoring_enabled": bool(importance.get("enabled") is True),
        "importance_scoring_shadow_mode": bool(importance.get("shadow_mode") is True),
    }


def should_emit_context_policy_summary() -> bool:
    """Return whether the boot summary should be emitted for this request."""
    if os.getenv("CE_POLICY_SUMMARY_EVERY_REQUEST") == "1":
        return True
    return os.getenv("LOG_CONTEXT_POLICY_SUMMARY") == "1"


def _policy_from_node_policy(
    *,
    legacy_policy: ContextInjectionPolicy,
    node_policy: NodeContextPolicy,
    policy_source: PolicySource,
    node_name: str,
) -> ContextInjectionPolicy:
    route = legacy_policy.route_rollout
    route_nodes = tuple(dict.fromkeys((*route.apply_enabled_nodes, node_name)))
    route = replace(
        route,
        apply_enabled_nodes=route_nodes,
        min_injectable_items=node_policy.min_injectable_items,
    )
    quality = replace(
        legacy_policy.quality,
        max_items_total=node_policy.max_items_total,
        max_items_per_source=_source_max_items_caps(node_policy.source_overrides),
    )
    return replace(
        legacy_policy,
        enabled=True,
        apply_enabled_nodes=(node_name,) if node_policy.mode == "active" else (),
        max_injected_context_tokens=node_policy.max_injected_context_tokens,
        injectable_sources=node_policy.injectable_sources,
        required_sources=node_policy.required_sources,
        optional_sources=node_policy.optional_sources,
        exclude_message_source=node_policy.exclude_message_source,
        route_rollout=route,
        quality=quality,
        mode=node_policy.mode,
        risk_tier=node_policy.risk_tier,
        policy_source=policy_source,
    )


def _source_max_items_caps(
    source_overrides: dict[str, dict[str, Any]],
) -> dict[str, int]:
    caps: dict[str, int] = {}
    for source, raw in source_overrides.items():
        value = raw.get("max_items")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            continue
        caps[source] = value
    return caps


def _raw_node_policy(
    apply_config: dict[str, Any],
    *,
    legacy_policy: ContextInjectionPolicy,
    node_name: str,
    llm_node: str,
    state: dict | None,
) -> tuple[dict[str, Any], PolicySource]:
    legacy_base = _legacy_policy_base(legacy_policy)
    node_policies = _optional_mapping(apply_config.get("node_policies"))
    node_groups = _optional_mapping(apply_config.get("node_groups"))
    resource_type_policies = _optional_mapping(
        apply_config.get("resource_type_policies")
    )
    default_policy = _optional_mapping(apply_config.get("default_policy"))
    resource_type = _resource_type_from_state(state)
    raw_resource_policy = (
        _optional_mapping(resource_type_policies.get(resource_type))
        if resource_type
        else {}
    )
    matched_group = _matched_node_group(node_groups, node_name)
    raw_node_policy = _optional_mapping(node_policies.get(node_name))
    if raw_node_policy:
        inherit = str(raw_node_policy.get("inherit") or "").strip()
        if inherit and inherit not in node_groups:
            raise _config_error(
                "context_apply_node_policy_inherit_missing",
                f"context apply node policy inherit group not found: {inherit}",
                node_name=node_name,
                llm_node=llm_node,
            )
        inherited = (
            _optional_mapping(node_groups.get(inherit)) if inherit else matched_group
        )
        return _merge_policy_dicts(
            legacy_base,
            default_policy,
            inherited,
            raw_resource_policy,
            raw_node_policy,
        ), "node_policy"

    if raw_resource_policy:
        return _merge_policy_dicts(
            legacy_base,
            default_policy,
            matched_group,
            raw_resource_policy,
        ), "resource_type_policy"

    if matched_group:
        return (
            _merge_policy_dicts(legacy_base, default_policy, matched_group),
            "node_group",
        )

    if default_policy:
        return (
            _merge_policy_dicts(legacy_base, default_policy),
            "default_policy",
        )
    raise _config_error(
        "context_apply_node_policy_missing",
        "context_engineering.packer.apply.default_policy must be configured when node_policies are enabled",
        node_name=node_name,
        llm_node=llm_node,
    )


def _parse_node_policy(
    raw: dict[str, Any],
    *,
    legacy_policy: ContextInjectionPolicy,
    node_name: str,
    llm_node: str,
) -> NodeContextPolicy:
    mode = _mode(raw.get("mode"), node_name=node_name, llm_node=llm_node)
    risk_tier = _non_negative_int(
        raw.get("risk_tier", legacy_policy.risk_tier),
        field_name="risk_tier",
        node_name=node_name,
        llm_node=llm_node,
    )
    injectable_sources = _source_tuple(
        raw.get("injectable_sources", list(legacy_policy.injectable_sources)),
        node_name=node_name,
        llm_node=llm_node,
        field_name="injectable_sources",
        allow_empty=False,
    )
    return NodeContextPolicy(
        mode=mode,
        risk_tier=risk_tier,
        max_injected_context_tokens=_positive_int(
            raw.get(
                "max_injected_context_tokens",
                legacy_policy.max_injected_context_tokens,
            ),
            field_name="max_injected_context_tokens",
            node_name=node_name,
            llm_node=llm_node,
        ),
        max_items_total=_positive_int(
            raw.get("max_items_total", legacy_policy.quality.max_items_total),
            field_name="max_items_total",
            node_name=node_name,
            llm_node=llm_node,
        ),
        min_injectable_items=_positive_int(
            raw.get(
                "min_injectable_items",
                legacy_policy.route_rollout.min_injectable_items,
            ),
            field_name="min_injectable_items",
            node_name=node_name,
            llm_node=llm_node,
        ),
        injectable_sources=injectable_sources,
        required_sources=_source_tuple(
            raw.get("required_sources", list(legacy_policy.required_sources)),
            node_name=node_name,
            llm_node=llm_node,
            field_name="required_sources",
            allow_empty=True,
        ),
        optional_sources=_source_tuple(
            raw.get(
                "optional_sources",
                list(legacy_policy.optional_sources or injectable_sources),
            ),
            node_name=node_name,
            llm_node=llm_node,
            field_name="optional_sources",
            allow_empty=True,
        ),
        exclude_message_source=_bool_field(
            raw.get("exclude_message_source", legacy_policy.exclude_message_source),
            field_name="exclude_message_source",
            node_name=node_name,
            llm_node=llm_node,
        ),
        source_overrides=_dict_of_dicts(raw.get("source_overrides")),
    )


def _source_policies_from_config(
    *,
    source_defaults: dict[str, Any],
    source_overrides: dict[str, dict[str, Any]],
    injectable_sources: tuple[ContextSourceType, ...],
    node_name: str,
    llm_node: str,
) -> dict[str, SourceBudgetPolicy]:
    policies: dict[str, SourceBudgetPolicy] = {}
    for source in injectable_sources:
        source_text = str(source)
        raw = _merge_policy_dicts(
            _optional_mapping(source_defaults.get(source_text)),
            source_overrides.get(source_text, {}),
        )
        policies[source_text] = _parse_source_policy(
            source=source,
            raw=raw,
            node_name=node_name,
            llm_node=llm_node,
        )
    return policies


def _legacy_policy_base(policy: ContextInjectionPolicy) -> dict[str, Any]:
    source_overrides = {
        source: {"max_items": max_items}
        for source, max_items in policy.quality.max_items_per_source.items()
    }
    return {
        "mode": "active",
        "risk_tier": policy.risk_tier,
        "max_injected_context_tokens": policy.max_injected_context_tokens,
        "max_items_total": policy.quality.max_items_total,
        "min_injectable_items": policy.route_rollout.min_injectable_items,
        "injectable_sources": list(policy.injectable_sources),
        "required_sources": list(policy.required_sources),
        "optional_sources": list(policy.optional_sources),
        "exclude_message_source": policy.exclude_message_source,
        "source_overrides": source_overrides,
    }


def _matched_node_group(
    node_groups: dict[str, Any],
    node_name: str,
) -> dict[str, Any]:
    for raw_group in node_groups.values():
        group = _optional_mapping(raw_group)
        if node_name in _string_tuple(group.get("nodes"), path="node_groups.nodes"):
            return group
    return {}


def _resource_type_from_state(state: dict | None) -> str:
    if not isinstance(state, dict):
        return ""
    resource_task = state.get("resource_task")
    if isinstance(resource_task, dict):
        resource_type = str(resource_task.get("resource_type") or "").strip()
        if resource_type:
            return resource_type
    values = _single_string_list(state.get("requested_resource_types"))
    if len(values) > 1:
        return ""
    requested_resource_type = str(state.get("requested_resource_type") or "").strip()
    if requested_resource_type and (not values or values == [requested_resource_type]):
        return requested_resource_type
    if len(values) == 1:
        return values[0]
    return ""


def _legacy_global_configured(apply_config: dict[str, Any]) -> bool:
    route_rollout = _optional_mapping(apply_config.get("route_rollout"))
    quality = _optional_mapping(apply_config.get("quality"))
    return any(
        (
            isinstance(apply_config.get("apply_enabled_nodes"), list),
            isinstance(route_rollout.get("apply_enabled_nodes"), list),
            "max_injected_context_tokens" in apply_config,
            bool(quality),
        )
    )


def _single_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item or "").strip() for item in value if str(item or "").strip()]


def _parse_source_policy(
    *,
    source: ContextSourceType,
    raw: dict[str, Any],
    node_name: str,
    llm_node: str,
) -> SourceBudgetPolicy:
    stale_policy = str(raw.get("stale_policy") or "keep").strip() or "keep"
    if stale_policy not in _ALLOWED_STALE_POLICIES:
        raise _config_error(
            "context_apply_source_policy_invalid",
            f"unsupported stale_policy: {stale_policy}",
            node_name=node_name,
            llm_node=llm_node,
        )
    return SourceBudgetPolicy(
        source_type=source,
        max_items=_optional_non_negative_int(
            raw.get("max_items"),
            field_name="max_items",
            node_name=node_name,
            llm_node=llm_node,
        ),
        max_tokens=_optional_non_negative_int(
            raw.get("max_tokens"),
            field_name="max_tokens",
            node_name=node_name,
            llm_node=llm_node,
        ),
        min_priority=_optional_non_negative_int(
            raw.get("min_priority"),
            field_name="min_priority",
            node_name=node_name,
            llm_node=llm_node,
        ),
        min_relevance_score=_optional_ratio(
            raw.get("min_relevance_score"),
            field_name="min_relevance_score",
            node_name=node_name,
            llm_node=llm_node,
        ),
        min_trust_level=_optional_ratio(
            raw.get("min_trust_level"),
            field_name="min_trust_level",
            node_name=node_name,
            llm_node=llm_node,
        ),
        allowed_purposes=_string_tuple(
            raw.get("allowed_purposes"),
            path="allowed_purposes",
        ),
        require_user_match=bool(raw.get("require_user_match") is True),
        require_thread_match=bool(raw.get("require_thread_match") is True),
        require_subject_match=bool(raw.get("require_subject_match") is True),
        require_task_match=bool(raw.get("require_task_match") is True),
        strict_match=bool(raw.get("strict_match") is True),
        stale_policy=stale_policy,
    )


def _apply_config() -> dict[str, Any]:
    context_config = get_setting("context_engineering")
    if not isinstance(context_config, dict):
        return {}
    packer_config = context_config.get("packer")
    if not isinstance(packer_config, dict):
        return {}
    apply_config = packer_config.get("apply")
    if not isinstance(apply_config, dict):
        return {}
    return apply_config


def _node_policy_schema_configured(apply_config: dict[str, Any]) -> bool:
    """Return true when new node-aware policy schema is configured.

    Any of default_policy, node_groups, node_policies, or resource_type_policies
    enables the CE-1 resolver. Only when all four are absent does apply use the
    legacy global apply config unchanged.
    """
    return any(
        key in apply_config and isinstance(apply_config.get(key), dict)
        for key in (
            "node_policies",
            "node_groups",
            "default_policy",
            "resource_type_policies",
        )
    )


def _merge_policy_dicts(*values: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in values:
        for key, item in value.items():
            if key == "source_overrides" and isinstance(item, dict):
                current = _dict_of_dicts(merged.get(key))
                for source, override in _dict_of_dicts(item).items():
                    current[source] = _merge_policy_dicts(
                        current.get(source, {}),
                        override,
                    )
                merged[key] = current
            elif key != "nodes":
                merged[key] = item
    return merged


def _optional_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _dict_of_dicts(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for key, item in value.items():
        if isinstance(item, dict):
            result[str(key)] = dict(item)
    return result


def _mode(value: Any, *, node_name: str, llm_node: str) -> ContextPolicyMode:
    text = str(value or "").strip()
    if text not in _VALID_MODES:
        raise _config_error(
            "context_apply_policy_mode_invalid",
            f"context apply mode must be one of: {', '.join(sorted(_VALID_MODES))}",
            node_name=node_name,
            llm_node=llm_node,
        )
    return cast(ContextPolicyMode, text)


def _source_tuple(
    value: Any,
    *,
    node_name: str,
    llm_node: str,
    field_name: str,
    allow_empty: bool,
) -> tuple[ContextSourceType, ...]:
    values = _string_tuple(value, path=field_name)
    if not values and not allow_empty:
        raise _config_error(
            f"context_apply_{field_name}_invalid",
            f"{field_name} must be a non-empty list",
            node_name=node_name,
            llm_node=llm_node,
        )
    sources: list[ContextSourceType] = []
    for source in values:
        if source not in _ALLOWED_SOURCES:
            raise _config_error(
                f"context_apply_{field_name}_invalid",
                f"unknown context source: {source}",
                node_name=node_name,
                llm_node=llm_node,
            )
        sources.append(cast(ContextSourceType, source))
    return tuple(sources)


def _string_tuple(value: Any, *, path: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        return ()
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            result.append(text)
    return tuple(result)


def _positive_int(
    value: Any,
    *,
    field_name: str,
    node_name: str,
    llm_node: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _config_error(
            "context_apply_node_policy_invalid",
            f"{field_name} must be a positive integer",
            node_name=node_name,
            llm_node=llm_node,
        )
    return value


def _bool_field(
    value: Any,
    *,
    field_name: str,
    node_name: str,
    llm_node: str,
) -> bool:
    if not isinstance(value, bool):
        raise _config_error(
            "context_apply_node_policy_invalid",
            f"{field_name} must be a boolean",
            node_name=node_name,
            llm_node=llm_node,
        )
    return value


def _non_negative_int(
    value: Any,
    *,
    field_name: str,
    node_name: str,
    llm_node: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _config_error(
            "context_apply_node_policy_invalid",
            f"{field_name} must be a non-negative integer",
            node_name=node_name,
            llm_node=llm_node,
        )
    return value


def _optional_non_negative_int(
    value: Any,
    *,
    field_name: str,
    node_name: str,
    llm_node: str,
) -> int | None:
    if value is None:
        return None
    return _non_negative_int(
        value,
        field_name=field_name,
        node_name=node_name,
        llm_node=llm_node,
    )


def _optional_ratio(
    value: Any,
    *,
    field_name: str,
    node_name: str,
    llm_node: str,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _config_error(
            "context_apply_source_policy_invalid",
            f"{field_name} must be a number between 0 and 1",
            node_name=node_name,
            llm_node=llm_node,
        )
    ratio = float(value)
    if ratio < 0.0 or ratio > 1.0:
        raise _config_error(
            "context_apply_source_policy_invalid",
            f"{field_name} must be between 0 and 1",
            node_name=node_name,
            llm_node=llm_node,
        )
    return ratio


def _add_nodes_by_mode(
    *,
    mode: str,
    nodes: tuple[str, ...],
    active_nodes: set[str],
    observe_only_nodes: set[str],
    disabled_nodes: set[str],
) -> None:
    if mode == "active":
        active_nodes.update(nodes)
    elif mode == "observe_only":
        observe_only_nodes.update(nodes)
    elif mode == "disabled":
        disabled_nodes.update(nodes)


def _config_error(
    reason: str,
    warning: object,
    *,
    node_name: str,
    llm_node: str,
) -> ContextApplyError:
    return ContextApplyError(
        reason=reason,
        warning=sanitize_error_message(warning),
        node_name=node_name,
        llm_node=llm_node,
        original_exception_type="ContextConfigError",
    )
