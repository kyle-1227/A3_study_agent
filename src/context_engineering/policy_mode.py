"""Validated runtime mode for strict and broad Context Engineering policy."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Mapping, cast, get_args

from src.config import get_setting
from src.context_engineering.schema import ContextConfigError, ContextSourceType

ContextPolicyMode = Literal["strict", "broad"]
_VALID_POLICY_MODES = frozenset({"strict", "broad"})
_PRODUCTION_ENVIRONMENTS = frozenset({"prod", "production"})
_BROAD_BUSINESS_FILTERS = frozenset(
    {"purpose", "quality", "relevance", "subject", "task", "stale"}
)


@dataclass(frozen=True)
class BroadSourcePolicy:
    source_type: ContextSourceType
    max_items: int
    max_tokens: int
    require_user_match: bool
    require_thread_match: bool


@dataclass(frozen=True)
class ContextRuntimePolicy:
    mode: ContextPolicyMode
    environment: str
    max_items_total: int
    max_injected_context_tokens: int
    max_content_chars_per_item: int
    enabled_sources: tuple[ContextSourceType, ...]
    eligible_node_roles: tuple[str, ...]
    source_policies: dict[str, BroadSourcePolicy]
    bypass_business_filters: frozenset[str]


def resolve_context_runtime_policy() -> ContextRuntimePolicy:
    """Resolve and validate one runtime policy without silent defaults."""
    config = get_setting("context_engineering")
    if not isinstance(config, Mapping):
        raise ContextConfigError(
            "context_engineering_missing",
            "context_engineering config is required",
        )
    configured_mode = _policy_mode(
        config.get("policy_mode"),
        reason="context_policy_mode_invalid",
        path="context_engineering.policy_mode",
    )
    env_value = os.getenv("CONTEXT_POLICY_MODE")
    mode = (
        _policy_mode(
            env_value,
            reason="context_policy_mode_invalid",
            path="CONTEXT_POLICY_MODE",
        )
        if env_value is not None and env_value.strip()
        else configured_mode
    )
    environment = _runtime_environment()
    if environment in _PRODUCTION_ENVIRONMENTS and mode == "broad":
        raise ContextConfigError(
            "context_policy_mode_forbidden_in_production",
            "broad context policy is not permitted in production",
        )
    broad = config.get("broad_policy")
    if not isinstance(broad, Mapping):
        raise ContextConfigError(
            "context_broad_policy_missing",
            "context_engineering.broad_policy must be configured",
        )
    enabled_sources = _source_tuple(
        broad.get("enabled_sources"),
        path="context_engineering.broad_policy.enabled_sources",
    )
    eligible_node_roles = _eligible_roles(broad.get("eligible_node_roles"))
    source_policies = _source_policies(
        broad.get("source_caps"),
        enabled_sources=enabled_sources,
    )
    bypass_filters = _business_filter_set(broad.get("bypass_business_filters"))
    return ContextRuntimePolicy(
        mode=mode,
        environment=environment,
        max_items_total=_positive_int(
            broad.get("max_items_total"),
            path="context_engineering.broad_policy.max_items_total",
        ),
        max_injected_context_tokens=_positive_int(
            broad.get("max_injected_context_tokens"),
            path="context_engineering.broad_policy.max_injected_context_tokens",
        ),
        max_content_chars_per_item=_positive_int(
            broad.get("max_content_chars_per_item"),
            path="context_engineering.broad_policy.max_content_chars_per_item",
        ),
        enabled_sources=enabled_sources,
        eligible_node_roles=eligible_node_roles,
        source_policies=source_policies,
        bypass_business_filters=bypass_filters,
    )


def validate_context_runtime_policy() -> ContextRuntimePolicy:
    """Startup validation entry point."""
    return resolve_context_runtime_policy()


def _runtime_environment() -> str:
    values = {
        str(os.getenv("APP_ENV") or "").strip().lower(),
        str(os.getenv("A3_ENV") or "").strip().lower(),
    }
    values.discard("")
    if values & _PRODUCTION_ENVIRONMENTS:
        return "production"
    return sorted(values)[0] if values else "unspecified"


def _policy_mode(value: object, *, reason: str, path: str) -> ContextPolicyMode:
    mode = str(value or "").strip().lower()
    if mode not in _VALID_POLICY_MODES:
        raise ContextConfigError(
            reason,
            f"{path} must be one of: broad, strict",
        )
    return cast(ContextPolicyMode, mode)


def _source_tuple(value: object, *, path: str) -> tuple[ContextSourceType, ...]:
    if not isinstance(value, list) or not value:
        raise ContextConfigError(
            "context_broad_sources_invalid",
            f"{path} must be a non-empty list",
        )
    allowed_sources = set(get_args(ContextSourceType)) - {"unknown"}
    result: list[ContextSourceType] = []
    for item in value:
        source = str(item or "").strip()
        if source not in allowed_sources:
            raise ContextConfigError(
                "context_broad_sources_invalid",
                f"unsupported broad context source: {source}",
            )
        typed_source = cast(ContextSourceType, source)
        if typed_source not in result:
            result.append(typed_source)
    return tuple(result)


def _source_policies(
    value: object,
    *,
    enabled_sources: tuple[ContextSourceType, ...],
) -> dict[str, BroadSourcePolicy]:
    if not isinstance(value, Mapping):
        raise ContextConfigError(
            "context_broad_source_caps_invalid",
            "context_engineering.broad_policy.source_caps must be a mapping",
        )
    expected = {str(source) for source in enabled_sources}
    configured = {str(source or "").strip() for source in value}
    if configured != expected:
        raise ContextConfigError(
            "context_broad_source_caps_invalid",
            "broad source_caps keys must exactly match enabled_sources",
        )
    result: dict[str, BroadSourcePolicy] = {}
    for source in enabled_sources:
        raw = value.get(str(source))
        if not isinstance(raw, Mapping):
            raise ContextConfigError(
                "context_broad_source_caps_invalid",
                f"source cap for {source} must be a mapping",
            )
        result[str(source)] = BroadSourcePolicy(
            source_type=source,
            max_items=_positive_int(
                raw.get("max_items"),
                path=f"context_engineering.broad_policy.source_caps.{source}.max_items",
            ),
            max_tokens=_positive_int(
                raw.get("max_tokens"),
                path=f"context_engineering.broad_policy.source_caps.{source}.max_tokens",
            ),
            require_user_match=_required_bool(
                raw.get("require_user_match"),
                path=(
                    "context_engineering.broad_policy.source_caps."
                    f"{source}.require_user_match"
                ),
            ),
            require_thread_match=_required_bool(
                raw.get("require_thread_match"),
                path=(
                    "context_engineering.broad_policy.source_caps."
                    f"{source}.require_thread_match"
                ),
            ),
        )
    return result


def _business_filter_set(value: object) -> frozenset[str]:
    if not isinstance(value, list):
        raise ContextConfigError(
            "context_broad_business_filters_invalid",
            "broad_policy.bypass_business_filters must be a list",
        )
    configured = frozenset(str(item or "").strip() for item in value)
    if configured != _BROAD_BUSINESS_FILTERS:
        raise ContextConfigError(
            "context_broad_business_filters_invalid",
            "broad business filters must explicitly list the supported filter set",
        )
    return configured


def _eligible_roles(value: object) -> tuple[str, ...]:
    allowed = {"planner", "agent", "reviewer", "consensus"}
    if not isinstance(value, list) or not value:
        raise ContextConfigError(
            "context_broad_eligible_roles_invalid",
            "broad_policy.eligible_node_roles must be a non-empty list",
        )
    roles = tuple(dict.fromkeys(str(item or "").strip() for item in value))
    if any(role not in allowed for role in roles):
        raise ContextConfigError(
            "context_broad_eligible_roles_invalid",
            "broad_policy.eligible_node_roles contains an unsupported role",
        )
    return roles


def _positive_int(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContextConfigError(
            "context_broad_policy_invalid",
            f"{path} must be a positive integer",
        )
    return value


def _required_bool(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise ContextConfigError(
            "context_broad_policy_invalid",
            f"{path} must be a boolean",
        )
    return value
