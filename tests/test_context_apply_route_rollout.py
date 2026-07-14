"""Route rollout tests for Phase 3B-2A context apply."""

from __future__ import annotations

from typing import Any

import pytest

from src.context_engineering.packing import apply as apply_module
from src.context_engineering.packing.apply import (
    ContextInjectionPolicy,
    apply_node_enabled,
    detect_single_resource_request,
    evaluate_context_apply_route,
    get_context_injection_policy,
)


def _settings(apply_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "context_engineering": {
            "enabled": True,
            "packer": {"apply_to_llm": False, "apply": apply_config},
        }
    }


def _lookup(settings: dict[str, Any], key: str, default: Any = None) -> Any:
    current: Any = settings
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _patch_settings(monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]) -> None:
    monkeypatch.setattr(
        apply_module,
        "get_setting",
        lambda key, default=None: _lookup(_settings(config), key, default),
    )


def _apply_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "apply_enabled_nodes": ["review_doc_agent"],
        "allow_structured_output": False,
        "role": "system",
        "position": "after_system",
        "exclude_message_source": True,
        "max_injected_context_tokens": 80000,
        "injectable_sources": ["rules", "evidence", "memory"],
        "route_rollout": {
            "enabled": True,
            "route_name": "single_resource_generation",
            "apply_enabled_nodes": ["review_doc_agent"],
            "require_single_resource_request": True,
            "sample_rate": 1.0,
            "min_injectable_items": 1,
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


def test_single_resource_detection_is_conservative():
    assert (
        detect_single_resource_request(
            {
                "requested_resource_types": ["review_doc"],
                "requested_resource_type": "review_doc",
            }
        )
        == "matched_single_resource"
    )
    assert (
        detect_single_resource_request({"requested_resource_types": ["a", "b"]})
        == "multi_resource_request"
    )
    assert (
        detect_single_resource_request({"is_parallel_resource_request": True})
        == "parallel_resource_request"
    )
    assert detect_single_resource_request({}) == "missing_resource_type"
    assert (
        detect_single_resource_request(
            {
                "requested_resource_types": ["review_doc"],
                "requested_resource_type": "exercise",
            }
        )
        == "ambiguous_resource_state"
    )


def test_route_rollout_disabled_skips_even_when_route_lists_node(monkeypatch):
    config = _apply_config()
    config["route_rollout"]["enabled"] = False
    _patch_settings(monkeypatch, config)

    policy = get_context_injection_policy(
        node_name="review_doc_agent",
        llm_node="review_doc",
    )
    enabled, skip_reason, single_result, warnings = evaluate_context_apply_route(
        policy=policy,
        node_name="review_doc_agent",
        state={"requested_resource_types": ["review_doc"]},
    )

    assert apply_node_enabled(policy, node_name="review_doc_agent") is True
    assert enabled is False
    assert skip_reason == "route_rollout_disabled"
    assert single_result == ""
    assert warnings == []


def test_route_node_and_single_resource_must_match(monkeypatch):
    _patch_settings(monkeypatch, _apply_config())
    policy = get_context_injection_policy(
        node_name="review_doc_agent",
        llm_node="review_doc",
    )

    enabled, skip_reason, _single_result, warnings = evaluate_context_apply_route(
        policy=policy,
        node_name="other_node",
        state={"requested_resource_types": ["review_doc"]},
    )
    assert enabled is False
    assert skip_reason == "top_level_node_not_enabled"
    assert warnings == []

    enabled, skip_reason, single_result, warnings = evaluate_context_apply_route(
        policy=policy,
        node_name="review_doc_agent",
        state={"requested_resource_types": ["review_doc", "exercise"]},
    )
    assert enabled is False
    assert skip_reason == "single_resource_not_matched"
    assert single_result == "multi_resource_request"
    assert warnings == []


def test_sampling_missing_stable_id_returns_warning(monkeypatch):
    config = _apply_config()
    config["route_rollout"]["sample_rate"] = 0.5
    _patch_settings(monkeypatch, config)
    policy = get_context_injection_policy(
        node_name="review_doc_agent",
        llm_node="review_doc",
    )

    _enabled, _skip_reason, _single_result, warnings = evaluate_context_apply_route(
        policy=policy,
        node_name="review_doc_agent",
        state={"requested_resource_types": ["review_doc"]},
    )

    assert warnings == ["context_apply_sampling_missing_stable_id"]


def test_importance_scorer_node_conflict_disables_shadow(monkeypatch):
    config = _apply_config()
    config["importance_scoring"] = {
        "enabled": True,
        "shadow_mode": True,
        "mode": "shadow",
        "llm_node": "review_doc_agent",
        "max_items_to_score": 3,
        "max_content_preview_chars": 100,
        "timeout_seconds": 1,
        "emit_shadow_telemetry": True,
        "min_shadow_score_for_analysis": 0.5,
    }
    _patch_settings(monkeypatch, config)

    policy: ContextInjectionPolicy = get_context_injection_policy(
        node_name="review_doc_agent",
        llm_node="review_doc",
    )

    assert policy.importance_scoring.enabled is False
    assert (
        policy.importance_scoring.disabled_reason
        == "context_importance_scorer_node_conflicts_with_apply_nodes"
    )
