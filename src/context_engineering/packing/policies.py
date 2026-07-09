"""Configuration and policy helpers for ContextPacker shadow mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from src.config import get_setting
from src.context_engineering.packing.schema import ContextPackingError, PackingStrategy
from src.context_engineering.schema import ContextSourceType

_ALLOWED_SOURCES = {
    "message",
    "memory",
    "evidence",
    "artifact",
    "profile",
    "trajectory",
    "rules",
    "curriculum",
    "unknown",
}


@dataclass(frozen=True)
class PackingPolicy:
    """Explicit context-packing policy."""

    enabled: bool
    shadow_mode: bool
    apply_to_llm: bool
    strategy: PackingStrategy
    max_context_block_tokens: int
    trace_selected_items: int
    trace_dropped_items: int
    enabled_nodes: tuple[str, ...]
    enabled_sources: tuple[ContextSourceType, ...]


def get_packing_policy(*, node_name: str, llm_node: str) -> PackingPolicy:
    """Read and validate explicit context_engineering.packer settings."""
    context_config = get_setting("context_engineering")
    if not isinstance(context_config, dict):
        raise _config_error(
            "context_engineering_missing",
            "context_engineering config is required",
            node_name=node_name,
            llm_node=llm_node,
        )
    if context_config.get("enabled") is False:
        return PackingPolicy(
            enabled=False,
            shadow_mode=True,
            apply_to_llm=False,
            strategy="priority_budget",
            max_context_block_tokens=1,
            trace_selected_items=0,
            trace_dropped_items=0,
            enabled_nodes=(),
            enabled_sources=(),
        )

    raw = context_config.get("packer")
    if not isinstance(raw, dict):
        raise _config_error(
            "context_packer_missing",
            "context_engineering.packer config is required",
            node_name=node_name,
            llm_node=llm_node,
        )

    enabled = _required_bool(raw, "enabled", node_name=node_name, llm_node=llm_node)
    if not enabled:
        return PackingPolicy(
            enabled=False,
            shadow_mode=True,
            apply_to_llm=False,
            strategy="priority_budget",
            max_context_block_tokens=1,
            trace_selected_items=0,
            trace_dropped_items=0,
            enabled_nodes=(),
            enabled_sources=(),
        )

    strategy = raw.get("strategy")
    if strategy != "priority_budget":
        raise _config_error(
            "context_packer_strategy_unsupported",
            "context_engineering.packer.strategy must be priority_budget",
            node_name=node_name,
            llm_node=llm_node,
        )

    return PackingPolicy(
        enabled=enabled,
        shadow_mode=_required_bool(
            raw, "shadow_mode", node_name=node_name, llm_node=llm_node
        ),
        apply_to_llm=_required_bool(
            raw, "apply_to_llm", node_name=node_name, llm_node=llm_node
        ),
        strategy=cast(PackingStrategy, strategy),
        max_context_block_tokens=_required_positive_int(
            raw,
            "max_context_block_tokens",
            node_name=node_name,
            llm_node=llm_node,
        ),
        trace_selected_items=_required_non_negative_int(
            raw,
            "trace_selected_items",
            node_name=node_name,
            llm_node=llm_node,
        ),
        trace_dropped_items=_required_non_negative_int(
            raw,
            "trace_dropped_items",
            node_name=node_name,
            llm_node=llm_node,
        ),
        enabled_nodes=_required_string_tuple(
            raw,
            "enabled_nodes",
            node_name=node_name,
            llm_node=llm_node,
        ),
        enabled_sources=_required_sources(
            raw,
            node_name=node_name,
            llm_node=llm_node,
        ),
    )


def node_enabled(policy: PackingPolicy, *, node_name: str) -> bool:
    """Return whether packer should run for this node."""
    return not policy.enabled_nodes or node_name in policy.enabled_nodes


def _required_bool(
    values: dict[str, Any],
    key: str,
    *,
    node_name: str,
    llm_node: str,
) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise _config_error(
            f"context_packer_{key}_invalid",
            f"context_engineering.packer.{key} must be a boolean",
            node_name=node_name,
            llm_node=llm_node,
        )
    return value


def _required_positive_int(
    values: dict[str, Any],
    key: str,
    *,
    node_name: str,
    llm_node: str,
) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _config_error(
            f"context_packer_{key}_invalid",
            f"context_engineering.packer.{key} must be a positive integer",
            node_name=node_name,
            llm_node=llm_node,
        )
    return value


def _required_non_negative_int(
    values: dict[str, Any],
    key: str,
    *,
    node_name: str,
    llm_node: str,
) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _config_error(
            f"context_packer_{key}_invalid",
            f"context_engineering.packer.{key} must be a non-negative integer",
            node_name=node_name,
            llm_node=llm_node,
        )
    return value


def _required_string_tuple(
    values: dict[str, Any],
    key: str,
    *,
    node_name: str,
    llm_node: str,
) -> tuple[str, ...]:
    value = values.get(key)
    if not isinstance(value, list):
        raise _config_error(
            f"context_packer_{key}_invalid",
            f"context_engineering.packer.{key} must be a list",
            node_name=node_name,
            llm_node=llm_node,
        )
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            raise _config_error(
                f"context_packer_{key}_invalid",
                f"context_engineering.packer.{key} entries must be non-empty",
                node_name=node_name,
                llm_node=llm_node,
            )
        result.append(text)
    return tuple(result)


def _required_sources(
    values: dict[str, Any],
    *,
    node_name: str,
    llm_node: str,
) -> tuple[ContextSourceType, ...]:
    raw = values.get("enabled_sources")
    if not isinstance(raw, list) or not raw:
        raise _config_error(
            "context_packer_enabled_sources_invalid",
            "context_engineering.packer.enabled_sources must be a non-empty list",
            node_name=node_name,
            llm_node=llm_node,
        )
    sources: list[ContextSourceType] = []
    for item in raw:
        source = str(item or "").strip()
        if source not in _ALLOWED_SOURCES:
            raise _config_error(
                "context_packer_enabled_sources_invalid",
                f"unknown context source: {source}",
                node_name=node_name,
                llm_node=llm_node,
            )
        sources.append(cast(ContextSourceType, source))
    return tuple(sources)


def _config_error(
    reason: str,
    warning: object,
    *,
    node_name: str,
    llm_node: str,
) -> ContextPackingError:
    return ContextPackingError(
        reason=reason,
        warning=warning,
        node_name=node_name,
        llm_node=llm_node,
        original_exception_type="ContextConfigError",
    )
