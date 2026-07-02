"""Policy tests for Phase 3B-1 context apply."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.context_engineering.packing import apply as apply_module
from src.context_engineering.packing.apply import (
    ContextApplyError,
    apply_node_enabled,
    get_context_injection_policy,
)


def _settings(apply_config: dict[str, Any] | None) -> dict[str, Any]:
    packer: dict[str, Any] = {"apply_to_llm": False}
    if apply_config is not None:
        packer["apply"] = apply_config
    return {"context_engineering": {"enabled": True, "packer": packer}}


def _lookup(settings: dict[str, Any], key: str, default: Any = None) -> Any:
    current: Any = settings
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _patch_settings(monkeypatch: pytest.MonkeyPatch, settings: dict[str, Any]) -> None:
    monkeypatch.setattr(
        apply_module,
        "get_setting",
        lambda key, default=None: _lookup(settings, key, default),
    )


def _enabled_apply_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "apply_enabled_nodes": ["plain_node"],
        "fallback_on_error": True,
        "allow_structured_output": False,
        "role": "system",
        "position": "after_system",
        "exclude_message_source": True,
        "max_injected_context_tokens": 80000,
        "injectable_sources": ["rules", "evidence", "memory"],
        "route_rollout": {
            "enabled": False,
            "route_name": "single_resource_generation",
        },
        "quality": {
            "min_priority": 0,
            "min_relevance_score": None,
            "max_items_total": 8,
            "max_items_per_source": {},
        },
        "budget": {
            "graceful_degradation_enabled": True,
            "drop_order": ["priority_asc", "token_estimate_desc", "id_asc"],
            "fallback_if_empty_after_drop": True,
        },
        "format": {
            "group_by_source": True,
            "include_untrusted_context_warning": True,
            "include_section_headers": True,
            "max_content_chars_per_item": 4000,
        },
        "importance_scoring": {
            "enabled": False,
            "llm_node": "academic",
        },
    }


def test_apply_missing_config_is_disabled(monkeypatch):
    _patch_settings(monkeypatch, _settings(apply_config=None))

    policy = get_context_injection_policy(node_name="plain_node", llm_node="llm")

    assert policy.enabled is False
    assert apply_node_enabled(policy, node_name="plain_node") is False


def test_apply_enabled_false_is_disabled(monkeypatch):
    _patch_settings(monkeypatch, _settings({"enabled": False}))

    policy = get_context_injection_policy(node_name="plain_node", llm_node="llm")

    assert policy.enabled is False
    assert apply_node_enabled(policy, node_name="plain_node") is False


def test_apply_enabled_nodes_empty_means_apply_none(monkeypatch):
    config = _enabled_apply_config()
    config["apply_enabled_nodes"] = []
    _patch_settings(monkeypatch, _settings(config))

    policy = get_context_injection_policy(node_name="plain_node", llm_node="llm")

    assert policy.enabled is True
    assert apply_node_enabled(policy, node_name="plain_node") is False


def test_node_must_be_explicitly_listed(monkeypatch):
    _patch_settings(monkeypatch, _settings(_enabled_apply_config()))

    policy = get_context_injection_policy(node_name="plain_node", llm_node="llm")

    assert apply_node_enabled(policy, node_name="plain_node") is True
    assert apply_node_enabled(policy, node_name="other_node") is False
    assert policy.role == "system"
    assert policy.position == "after_system"
    assert policy.max_injected_context_tokens == 80000
    assert policy.injectable_sources == ("rules", "evidence", "memory")
    assert policy.route_rollout.enabled is False


def test_enabled_apply_requires_complete_valid_config(monkeypatch):
    config = _enabled_apply_config()
    config.pop("injectable_sources")
    _patch_settings(monkeypatch, _settings(config))

    with pytest.raises(ContextApplyError) as exc_info:
        get_context_injection_policy(node_name="plain_node", llm_node="llm")

    assert exc_info.value.reason == "context_apply_injectable_sources_invalid"


def test_legacy_apply_to_llm_guard_remains_false_in_settings():
    from src.config import clear_cache, get_setting

    clear_cache()
    assert get_setting("context_engineering.packer.apply_to_llm") is False


def test_apply_module_does_not_hardcode_generate_answer():
    source = Path("src/context_engineering/packing/apply.py").read_text(
        encoding="utf-8"
    )

    assert "generate_answer" not in source
