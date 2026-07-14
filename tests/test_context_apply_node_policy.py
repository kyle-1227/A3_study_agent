"""CE-1/CE-2 node-aware and source-aware context apply tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.context_engineering.packing import apply as apply_module
from src.context_engineering.packing import node_policy as node_policy_module
from src.context_engineering.packing.apply import (
    ApplyBudgetPolicy,
    ApplyFormatPolicy,
    ApplyQualityPolicy,
    ContextApplyError,
    ContextInjectionPolicy,
    ImportanceScoringPolicy,
    RouteRolloutPolicy,
)
from src.context_engineering.packing.node_policy import (
    ResolvedContextPolicy,
    SourceBudgetPolicy,
    build_context_policy_summary,
    resolve_context_policy,
)
from src.context_engineering.packing.packer import pack_context_items
from src.context_engineering.packing.policies import PackingPolicy
from src.context_engineering.packing.source_policy import (
    filter_context_items_by_source_policy,
)
from src.context_engineering.providers.supply import (
    ContextCollectionResult,
    ProviderSupplyPlan,
)
from src.context_engineering.evidence_normalizer import EvidenceNormalizationStats
from src.context_engineering.schema import ContextItem


def _legacy_apply_config() -> dict[str, Any]:
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


def _patch_settings(monkeypatch: pytest.MonkeyPatch, settings: dict[str, Any]) -> None:
    def getter(key: str, default: Any = None) -> Any:
        return _lookup(settings, key, default)

    monkeypatch.setattr(apply_module, "get_setting", getter)
    monkeypatch.setattr(node_policy_module, "get_setting", getter)


def _policy(
    *,
    mode: str = "active",
    nodes: tuple[str, ...] = ("plain_node",),
    max_tokens: int = 10000,
    required_sources: tuple[str, ...] = (),
) -> ContextInjectionPolicy:
    return ContextInjectionPolicy(
        enabled=True,
        apply_enabled_nodes=nodes if mode == "active" else (),
        allow_structured_output=False,
        role="system",
        position="after_system",
        exclude_message_source=True,
        max_injected_context_tokens=max_tokens,
        injectable_sources=("memory", "evidence", "rules"),
        required_sources=required_sources,
        mode=mode,
        risk_tier=2,
        policy_source="node_policy",
        route_rollout=RouteRolloutPolicy(
            enabled=True,
            route_name="single_resource_generation",
            apply_enabled_nodes=nodes,
            require_single_resource_request=True,
            sample_rate=1.0,
            min_injectable_items=1,
        ),
        quality=ApplyQualityPolicy(
            min_priority=0,
            min_relevance_score=None,
            max_items_total=8,
            max_items_per_source={},
        ),
        budget=ApplyBudgetPolicy(
            graceful_degradation_enabled=True,
            drop_order=("priority_asc", "token_estimate_desc", "id_asc"),
        ),
        format=ApplyFormatPolicy(
            group_by_source=True,
            include_untrusted_context_warning=True,
            include_section_headers=True,
            max_content_chars_per_item=4000,
        ),
        importance_scoring=ImportanceScoringPolicy(
            enabled=False,
            shadow_mode=False,
            mode="disabled",
            llm_node="",
            max_items_to_score=0,
            max_content_preview_chars=0,
            timeout_seconds=0.0,
            emit_shadow_telemetry=False,
            min_shadow_score_for_analysis=0.0,
        ),
    )


def _resolved(policy: ContextInjectionPolicy) -> ResolvedContextPolicy:
    return ResolvedContextPolicy(
        mode=policy.mode,
        risk_tier=policy.risk_tier,
        policy_source="node_policy",
        injection_policy=policy,
        source_policies={},
        legacy_mode_enabled=True,
        node_policy_enabled=True,
        summary={},
    )


def _item(
    item_id: str = "memory-1",
    *,
    source_type: str = "memory",
    token_estimate: int = 5,
    priority: int = 80,
    relevance_score: float | None = 0.8,
    metadata: dict[str, Any] | None = None,
) -> ContextItem:
    item_metadata = dict(metadata or {})
    if source_type == "evidence":
        item_metadata.setdefault("grounding_approved", True)
    return ContextItem(
        id=item_id,
        source_type=source_type,
        title=item_id,
        content=f"{item_id} content",
        token_estimate=token_estimate,
        estimated=True,
        tokenizer_mode="estimated_mixed",
        priority=priority,
        relevance_score=relevance_score,
        recency_score=0.8,
        confidence=0.8,
        scope="turn",
        lifetime="turn",
        compressible=True,
        can_drop=True,
        disclosure_level="snippet",
        metadata=item_metadata,
    )


def _packed(items: list[ContextItem]):
    return pack_context_items(
        node_name="plain_node",
        llm_node="llm",
        items=items,
        max_context_block_tokens=10000,
    )


def _mock_llm(content: str = "answer"):
    mock_llm = MagicMock()
    mock_llm.model_name = "deepseek-v4-pro"
    mock_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content=content))
    return mock_llm


def _packing_policy(*, enabled: bool = True, max_tokens: int = 10000) -> PackingPolicy:
    return PackingPolicy(
        enabled=enabled,
        shadow_mode=False,
        apply_to_llm=True,
        strategy="priority_budget",
        max_context_block_tokens=max_tokens,
        trace_selected_items=0,
        trace_dropped_items=0,
        enabled_nodes=(),
        enabled_sources=("message", "memory", "evidence", "rules"),
    )


def _collection(
    items: list[ContextItem],
    *,
    requested_sources: tuple[str, ...] | None = None,
    required_sources: tuple[str, ...] = (),
    optional_sources: tuple[str, ...] = (),
) -> tuple[ProviderSupplyPlan, ContextCollectionResult]:
    requested = (
        requested_sources
        or tuple(dict.fromkeys(item.source_type for item in items))
        or ("memory",)
    )
    present = {item.source_type for item in items}
    missing = {
        source: "provider_empty" for source in requested if source not in present
    }
    plan = ProviderSupplyPlan(
        requested_sources=requested,
        required_sources=required_sources,
        optional_sources=optional_sources or requested,
        enabled_sources=requested,
        disabled_sources=(),
        unregistered_sources=(),
        provider_count=len(requested),
        provider_sources_missing={source: 1 for source in missing},
        provider_missing_reasons=missing,
    )
    return plan, ContextCollectionResult(
        items=items,
        provider_count=len(requested),
        provider_sources_missing={source: 1 for source in missing},
        provider_missing_reasons=missing,
        errors=[],
        evidence_stats=EvidenceNormalizationStats(),
    )


def _patch_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
    *,
    policy: ContextInjectionPolicy | None = None,
    error: ContextApplyError | None = None,
    items: list[ContextItem] | None = None,
    packing_enabled: bool = True,
    trace_payloads: list[dict] | None = None,
) -> None:
    from src.context_engineering.packing import orchestrator as orchestrator_module

    if error is None:
        assert policy is not None
        monkeypatch.setattr(
            orchestrator_module,
            "resolve_context_policy",
            lambda **_: _resolved(policy),
        )
    else:
        monkeypatch.setattr(
            orchestrator_module,
            "resolve_context_policy",
            lambda **_: (_ for _ in ()).throw(error),
        )
    monkeypatch.setattr(
        orchestrator_module,
        "collect_context_for_policy",
        lambda **kwargs: _collection(
            items or [_item("memory-1")],
            requested_sources=kwargs.get("requested_sources"),
            required_sources=kwargs.get("required_sources", ()),
            optional_sources=kwargs.get("optional_sources", ()),
        ),
    )
    max_tokens = policy.max_injected_context_tokens if policy is not None else 10000
    monkeypatch.setattr(
        orchestrator_module,
        "get_packing_policy",
        lambda **_: _packing_policy(
            enabled=packing_enabled,
            max_tokens=max_tokens,
        ),
    )
    if trace_payloads is not None:
        monkeypatch.setattr(
            orchestrator_module,
            "emit_a3_trace",
            lambda _logger, stage, payload, **_kwargs: trace_payloads.append(
                {"stage": stage, **payload}
            ),
        )


def test_resolver_keeps_legacy_when_node_policies_absent(monkeypatch):
    _patch_settings(monkeypatch, _settings(_legacy_apply_config()))

    resolved = resolve_context_policy(
        node_name="review_doc_agent",
        llm_node="review_doc",
        state={},
    )

    assert resolved.policy_source == "legacy_global"
    assert resolved.mode == "active"
    assert resolved.injection_policy.apply_enabled_nodes == ("review_doc_agent",)


def test_resolver_priority_node_group_default(monkeypatch):
    config = _legacy_apply_config()
    config.update(
        {
            "default_policy": {
                "mode": "observe_only",
                "risk_tier": 2,
                "max_injected_context_tokens": 1000,
                "max_items_total": 3,
                "min_injectable_items": 1,
                "injectable_sources": ["rules"],
            },
            "node_groups": {
                "agents": {
                    "mode": "active",
                    "risk_tier": 1,
                    "nodes": ["group_node"],
                    "max_injected_context_tokens": 2000,
                    "max_items_total": 5,
                    "injectable_sources": ["rules", "evidence"],
                }
            },
            "node_policies": {
                "group_node": {
                    "mode": "disabled",
                    "risk_tier": 4,
                    "max_injected_context_tokens": 500,
                    "max_items_total": 1,
                    "min_injectable_items": 1,
                    "injectable_sources": ["rules"],
                }
            },
        }
    )
    _patch_settings(monkeypatch, _settings(config))

    node_resolved = resolve_context_policy(
        node_name="group_node",
        llm_node="llm",
        state={},
    )
    default_resolved = resolve_context_policy(
        node_name="other_node",
        llm_node="llm",
        state={},
    )

    assert node_resolved.policy_source == "node_policy"
    assert node_resolved.mode == "disabled"
    assert default_resolved.policy_source == "default_policy"
    assert default_resolved.mode == "observe_only"


def test_node_policy_schema_configured_semantics():
    assert node_policy_module._node_policy_schema_configured({"default_policy": {}})
    assert node_policy_module._node_policy_schema_configured({"node_groups": {"g": {}}})
    assert node_policy_module._node_policy_schema_configured(
        {"node_policies": {"n": {}}}
    )
    assert node_policy_module._node_policy_schema_configured(
        {"resource_type_policies": {"quiz": {}}}
    )
    assert not node_policy_module._node_policy_schema_configured({})


def test_context_policy_summary_distinguishes_legacy_and_node_schema():
    legacy_summary = build_context_policy_summary(_legacy_apply_config())

    assert legacy_summary["node_policy_schema_configured"] is False
    assert legacy_summary["legacy_global_enabled"] is True
    assert legacy_summary["legacy_mode_enabled"] is True

    node_config = _legacy_apply_config()
    node_config["default_policy"] = {
        "mode": "observe_only",
        "risk_tier": 0,
        "max_injected_context_tokens": 1000,
        "max_items_total": 2,
        "min_injectable_items": 1,
        "injectable_sources": ["rules"],
    }
    node_summary = build_context_policy_summary(node_config)

    assert node_summary["node_policy_schema_configured"] is True
    assert node_summary["legacy_global_enabled"] is True
    assert node_summary["legacy_mode_enabled"] is False
    assert node_summary["default_policy_mode"] == "observe_only"


def test_resource_type_policy_resolves_from_resource_task_state(monkeypatch):
    config = _legacy_apply_config()
    config.update(
        {
            "default_policy": {
                "mode": "observe_only",
                "risk_tier": 3,
                "max_injected_context_tokens": 1000,
                "max_items_total": 2,
                "min_injectable_items": 1,
                "injectable_sources": ["rules"],
                "source_overrides": {"rules": {"max_items": 1}},
            },
            "node_groups": {
                "agents": {
                    "mode": "observe_only",
                    "risk_tier": 2,
                    "nodes": ["resource_node"],
                    "injectable_sources": ["rules", "evidence"],
                }
            },
            "resource_type_policies": {
                "quiz": {
                    "mode": "active",
                    "risk_tier": 1,
                    "max_items_total": 5,
                    "source_overrides": {"evidence": {"max_items": 3}},
                }
            },
        }
    )
    _patch_settings(monkeypatch, _settings(config))

    resolved = resolve_context_policy(
        node_name="resource_node",
        llm_node="llm",
        state={
            "resource_task": {"resource_type": "quiz"},
            "requested_resource_types": ["quiz", "review_doc"],
        },
    )

    assert resolved.policy_source == "resource_type_policy"
    assert resolved.mode == "active"
    assert resolved.risk_tier == 1
    assert resolved.injection_policy.injectable_sources == ("rules", "evidence")
    assert resolved.source_policies["rules"].max_items == 1
    assert resolved.source_policies["evidence"].max_items == 3


def test_node_policy_overrides_resource_type_policy(monkeypatch):
    config = _legacy_apply_config()
    config.update(
        {
            "default_policy": {
                "mode": "observe_only",
                "risk_tier": 3,
                "max_injected_context_tokens": 1000,
                "max_items_total": 2,
                "min_injectable_items": 1,
                "injectable_sources": ["rules"],
            },
            "resource_type_policies": {"quiz": {"mode": "active", "risk_tier": 1}},
            "node_policies": {
                "resource_node": {
                    "mode": "disabled",
                    "risk_tier": 4,
                    "injectable_sources": ["rules"],
                }
            },
        }
    )
    _patch_settings(monkeypatch, _settings(config))

    resolved = resolve_context_policy(
        node_name="resource_node",
        llm_node="llm",
        state={"resource_task": {"resource_type": "quiz"}},
    )

    assert resolved.policy_source == "node_policy"
    assert resolved.mode == "disabled"
    assert resolved.risk_tier == 4


def test_multi_requested_resource_types_without_resource_task_do_not_match_policy(
    monkeypatch,
):
    config = _legacy_apply_config()
    config.update(
        {
            "default_policy": {
                "mode": "observe_only",
                "risk_tier": 3,
                "max_injected_context_tokens": 1000,
                "max_items_total": 2,
                "min_injectable_items": 1,
                "injectable_sources": ["rules"],
            },
            "resource_type_policies": {"quiz": {"mode": "active", "risk_tier": 1}},
        }
    )
    _patch_settings(monkeypatch, _settings(config))

    resolved = resolve_context_policy(
        node_name="resource_node",
        llm_node="llm",
        state={
            "requested_resource_type": "quiz",
            "requested_resource_types": ["quiz", "review_doc"],
        },
    )

    assert resolved.policy_source == "default_policy"
    assert resolved.mode == "observe_only"


def test_unknown_inherit_raises_context_apply_error(monkeypatch):
    config = _legacy_apply_config()
    config.update(
        {
            "default_policy": {
                "mode": "observe_only",
                "risk_tier": 3,
                "max_injected_context_tokens": 1000,
                "max_items_total": 2,
                "min_injectable_items": 1,
                "injectable_sources": ["rules"],
            },
            "node_policies": {"node": {"inherit": "missing_group"}},
        }
    )
    _patch_settings(monkeypatch, _settings(config))

    with pytest.raises(ContextApplyError) as exc_info:
        resolve_context_policy(node_name="node", llm_node="llm", state={})

    assert exc_info.value.reason == "context_apply_node_policy_inherit_missing"


def test_settings_resolve_expected_default_tiers():
    from src.config import clear_cache

    clear_cache()
    assert (
        resolve_context_policy(
            node_name="supervisor",
            llm_node="supervisor",
            state={},
        ).mode
        == "observe_only"
    )
    planner = resolve_context_policy(
        node_name="review_doc_planner",
        llm_node="review_doc",
        state={},
    )
    assert planner.mode == "active"
    assert planner.injection_policy.required_sources == ("rules",)
    assert "evidence" in planner.injection_policy.optional_sources
    assert planner.injection_policy.max_injected_context_tokens == 1500
    assert planner.source_policies["evidence"].max_items == 2
    assert planner.source_policies["rules"].min_priority == 30
    assert (
        resolve_context_policy(
            node_name="exercise_planner",
            llm_node="quiz",
            state={},
        ).mode
        == "active"
    )
    agent = resolve_context_policy(
        node_name="mindmap_agent",
        llm_node="mindmap",
        state={},
    )
    assert agent.mode == "active"
    assert agent.injection_policy.required_sources == ("evidence",)
    assert (
        resolve_context_policy(
            node_name="video_script_agent",
            llm_node="video_script",
            state={},
        ).mode
        == "active"
    )
    reviewer = resolve_context_policy(
        node_name="review_doc_reviewer",
        llm_node="review_doc",
        state={},
    )
    assert reviewer.mode == "active"
    assert reviewer.injection_policy.required_sources == ("rules",)
    assert reviewer.injection_policy.injectable_sources == ("rules", "evidence")
    summary = resolve_context_policy(
        node_name="conversation_summary",
        llm_node="summary",
        state={},
    )
    assert summary.mode == "active"
    assert summary.injection_policy.exclude_message_source is False
    assert "message" in summary.injection_policy.injectable_sources
    assert summary.injection_policy.required_sources == ("message",)
    assert "message" in summary.source_policies
    assert summary.source_policies["message"].max_items == 4
    assert summary.source_policies["message"].max_tokens == 900
    assert (
        resolve_context_policy(
            node_name="review_doc_output",
            llm_node="review_doc",
            state={},
        ).mode
        == "disabled"
    )
    assert (
        resolve_context_policy(
            node_name="resource_bundle_output",
            llm_node="resource_bundle",
            state={},
        ).mode
        == "disabled"
    )
    assert (
        resolve_context_policy(
            node_name="video_animation_output",
            llm_node="video_animation",
            state={},
        ).mode
        == "disabled"
    )


def test_settings_enable_qa_agent_with_required_rules_and_optional_context(
    monkeypatch,
):
    from src.config import clear_cache

    monkeypatch.setenv("CONTEXT_POLICY_MODE", "strict")
    clear_cache()
    resolved = resolve_context_policy(
        node_name="qa_agent",
        llm_node="qa_agent",
        state={"response_mode": "qa", "qa_scope": "academic"},
    )

    assert resolved.mode == "active"
    assert resolved.injection_policy.required_sources == ("rules",)
    assert resolved.injection_policy.optional_sources == (
        "pipeline",
        "evidence",
        "profile",
        "memory",
        "artifact",
    )
    assert resolved.source_policies["evidence"].max_items == 4


def test_settings_agent_required_sources_are_node_specific():
    from src.config import clear_cache

    clear_cache()
    evidence_agents = (
        "review_doc_agent",
        "exercise_agent",
        "mindmap_agent",
        "code_practice_agent",
        "video_script_agent",
        "video_animation_agent",
    )
    for node_name in evidence_agents:
        resolved = resolve_context_policy(
            node_name=node_name,
            llm_node=node_name.removesuffix("_agent"),
            state={},
        )
        assert resolved.mode == "active"
        assert resolved.injection_policy.required_sources == ("evidence",)

    study_plan = resolve_context_policy(
        node_name="study_plan_agent",
        llm_node="study_plan",
        state={},
    )
    assert study_plan.injection_policy.required_sources == ()
    assert "profile" in study_plan.injection_policy.optional_sources

    adaptive = resolve_context_policy(
        node_name="adaptive_practice_responder",
        llm_node="adaptive_practice",
        state={},
    )
    recommendation = resolve_context_policy(
        node_name="recommendation_provider",
        llm_node="recommendation",
        state={},
    )
    assert adaptive.injection_policy.required_sources == ("rules",)
    assert recommendation.injection_policy.required_sources == ("rules",)
    assert "profile" in adaptive.injection_policy.optional_sources
    assert "trajectory" in adaptive.injection_policy.optional_sources
    assert "profile" in recommendation.injection_policy.optional_sources
    assert "trajectory" in recommendation.injection_policy.optional_sources


def test_agent_injectable_sources_are_node_specific():
    from src.config import clear_cache

    clear_cache()
    expected = {
        "review_doc_agent": {
            "required": ("evidence",),
            "optional": ("rules", "curriculum", "memory", "profile", "artifact"),
            "injectable": (
                "evidence",
                "rules",
                "curriculum",
                "memory",
                "profile",
                "artifact",
            ),
            "excluded": ("trajectory",),
        },
        "exercise_agent": {
            "required": ("evidence",),
            "optional": ("rules", "trajectory", "memory", "curriculum", "artifact"),
            "injectable": (
                "evidence",
                "rules",
                "trajectory",
                "memory",
                "curriculum",
                "artifact",
            ),
            "excluded": (),
        },
        "mindmap_agent": {
            "required": ("evidence",),
            "optional": ("rules", "curriculum", "artifact"),
            "injectable": ("evidence", "rules", "curriculum", "artifact"),
            "excluded": ("trajectory", "memory", "profile"),
        },
        "code_practice_agent": {
            "required": ("evidence",),
            "optional": ("rules", "artifact", "profile"),
            "injectable": ("evidence", "rules", "artifact", "profile"),
            "excluded": ("trajectory",),
        },
        "video_script_agent": {
            "required": ("evidence",),
            "optional": ("rules", "curriculum", "profile", "artifact"),
            "injectable": ("evidence", "rules", "curriculum", "profile", "artifact"),
            "excluded": ("trajectory", "memory"),
        },
        "video_animation_agent": {
            "required": ("evidence",),
            "optional": ("rules", "curriculum", "artifact"),
            "injectable": ("evidence", "rules", "curriculum", "artifact"),
            "excluded": ("trajectory", "memory", "profile"),
        },
        "study_plan_agent": {
            "required": (),
            "optional": (
                "profile",
                "rules",
                "trajectory",
                "memory",
                "curriculum",
                "artifact",
            ),
            "injectable": (
                "profile",
                "rules",
                "trajectory",
                "memory",
                "curriculum",
                "artifact",
            ),
            "excluded": ("evidence",),
        },
        "adaptive_practice_responder": {
            "required": ("rules",),
            "optional": ("profile", "trajectory", "memory"),
            "injectable": ("rules", "profile", "trajectory", "memory"),
            "excluded": ("evidence", "artifact", "curriculum"),
        },
        "error_classifier": {
            "required": ("rules",),
            "optional": (),
            "injectable": ("rules",),
            "excluded": (
                "evidence",
                "artifact",
                "curriculum",
                "memory",
                "profile",
                "trajectory",
            ),
        },
        "practice_generator": {
            "required": ("rules",),
            "optional": (),
            "injectable": ("rules",),
            "excluded": (
                "evidence",
                "artifact",
                "curriculum",
                "memory",
                "profile",
                "trajectory",
            ),
        },
        "recommendation_provider": {
            "required": ("rules",),
            "optional": ("profile", "trajectory", "curriculum"),
            "injectable": ("rules", "profile", "trajectory", "curriculum"),
            "excluded": ("evidence", "artifact", "memory"),
        },
    }

    for node_name, assertion in expected.items():
        resolved = resolve_context_policy(
            node_name=node_name,
            llm_node=node_name,
            state={},
        )
        policy = resolved.injection_policy

        assert policy.required_sources == assertion["required"]
        assert policy.optional_sources == assertion["optional"]
        assert policy.injectable_sources == assertion["injectable"]
        for source in assertion["excluded"]:
            assert source not in policy.injectable_sources


def test_settings_reviewer_sources_are_explicit_and_injectable():
    from src.config import clear_cache

    clear_cache()
    expected = {
        "review_doc_reviewer": ("rules", "evidence"),
        "exercise_reviewer": ("rules", "evidence"),
        "mindmap_reviewer": ("rules",),
        "code_practice_reviewer": ("rules", "artifact"),
        "video_script_reviewer": ("rules",),
        "video_animation_reviewer": ("rules",),
        "study_plan_reviewer_academic": ("rules", "curriculum"),
        "study_plan_reviewer_emotional": ("rules", "profile"),
        "study_plan_consensus": ("rules",),
    }

    for node_name, injectable_sources in expected.items():
        resolved = resolve_context_policy(
            node_name=node_name,
            llm_node=node_name.removesuffix("_reviewer"),
            state={},
        )
        assert resolved.mode == "active"
        assert resolved.injection_policy.required_sources == ("rules",)
        assert resolved.injection_policy.injectable_sources == injectable_sources
        assert set(resolved.injection_policy.optional_sources).issubset(
            set(injectable_sources)
        )


def test_structured_active_rollout_includes_selected_resource_subnodes():
    from src.config import clear_cache, get_setting

    clear_cache()
    active_nodes = set(
        get_setting(
            "context_engineering.packer.apply.structured_output_context.active_nodes",
            [],
        )
    )
    expected_active = {
        "review_doc_planner",
        "exercise_planner",
        "mindmap_planner",
        "code_practice_planner",
        "video_script_planner",
        "video_animation_planner",
        "study_plan_planner",
        "review_doc_agent",
        "exercise_agent",
        "mindmap_agent",
        "code_practice_agent",
        "video_script_agent",
        "video_animation_agent",
        "study_plan_agent",
        "review_doc_reviewer",
        "exercise_reviewer",
        "mindmap_reviewer",
        "code_practice_reviewer",
        "video_script_reviewer",
        "video_animation_reviewer",
        "study_plan_reviewer_academic",
        "study_plan_reviewer_emotional",
        "study_plan_consensus",
        "error_classifier",
        "practice_generator",
    }

    assert expected_active.issubset(active_nodes)
    assert "supervisor" not in active_nodes
    assert "search_query_rewriter" not in active_nodes
    assert "evidence_judge" not in active_nodes
    assert "resource_bundle_output" not in active_nodes


def test_conversation_summary_allows_message_but_generation_nodes_still_exclude():
    from src.config import clear_cache

    clear_cache()
    summary = resolve_context_policy(
        node_name="conversation_summary",
        llm_node="summary",
        state={},
    )
    review_agent = resolve_context_policy(
        node_name="review_doc_agent",
        llm_node="review_doc",
        state={},
    )
    message_item = _item("current-message", source_type="message")

    summary_result = filter_context_items_by_source_policy(
        [message_item],
        injectable_sources=summary.injection_policy.injectable_sources,
        exclude_message_source=summary.injection_policy.exclude_message_source,
        source_policies=summary.source_policies,
        state={},
    )
    review_result = filter_context_items_by_source_policy(
        [message_item],
        injectable_sources=review_agent.injection_policy.injectable_sources,
        exclude_message_source=review_agent.injection_policy.exclude_message_source,
        source_policies=review_agent.source_policies,
        state={},
    )

    assert [item.id for item in summary_result.kept_items] == ["current-message"]
    assert "message_source_excluded" not in summary_result.drop_reasons
    assert review_agent.injection_policy.exclude_message_source is True
    assert review_result.kept_items == []
    assert review_result.source_drop_reasons["message_source_excluded"] == 1


def test_invalid_mode_raises_context_apply_error(monkeypatch):
    config = _legacy_apply_config()
    config["default_policy"] = {
        "mode": "mystery",
        "risk_tier": 1,
        "max_injected_context_tokens": 1000,
        "max_items_total": 2,
        "min_injectable_items": 1,
        "injectable_sources": ["rules"],
    }
    _patch_settings(monkeypatch, _settings(config))

    with pytest.raises(ContextApplyError) as exc_info:
        resolve_context_policy(node_name="node", llm_node="llm", state={})

    assert exc_info.value.reason == "context_apply_policy_mode_invalid"


def test_source_filter_reports_allowlist_message_budget_and_match_reasons():
    policies = {
        "memory": SourceBudgetPolicy(
            source_type="memory",
            max_items=1,
            max_tokens=20,
            allowed_purposes=("continuity",),
            require_user_match=True,
        )
    }
    result = filter_context_items_by_source_policy(
        [
            _item("msg", source_type="message"),
            _item("evidence", source_type="evidence"),
            _item("wrong-user", metadata={"user_id": "other", "purpose": "continuity"}),
            _item("kept", priority=90, metadata={"user_id": "u1"}),
            _item("extra", metadata={"user_id": "u1", "purpose": "continuity"}),
        ],
        injectable_sources=("memory",),
        exclude_message_source=True,
        source_policies=policies,
        state={"user_id": "u1"},
    )

    assert [item.id for item in result.kept_items] == ["kept"]
    assert result.source_drop_reasons["message_source_excluded"] == 1
    assert result.source_drop_reasons["source_not_allowed"] == 1
    assert result.source_drop_reasons["user_mismatch"] == 1
    assert result.budget_drop_reasons["source_budget_exceeded"] == 1
    assert "missing_purpose_metadata" in result.warnings


def test_source_counts_dropped_tracks_each_drop_stage():
    policies = {
        "memory": SourceBudgetPolicy(source_type="memory", max_items=1),
    }
    result = filter_context_items_by_source_policy(
        [
            _item("msg", source_type="message"),
            _item("evidence", source_type="evidence"),
            _item("kept", source_type="memory", priority=90),
            _item("extra", source_type="memory", priority=80),
        ],
        injectable_sources=("memory",),
        exclude_message_source=True,
        source_policies=policies,
        state={},
    )

    assert result.source_counts_dropped["message"] == 1
    assert result.source_counts_dropped["evidence"] == 1
    assert result.source_counts_dropped["memory"] == 1


def test_source_filter_metadata_aliases_match_user_subject_task_and_purpose():
    policies = {
        "memory": SourceBudgetPolicy(
            source_type="memory",
            allowed_purposes=("continuity",),
            require_user_match=True,
            require_subject_match=True,
            require_task_match=True,
        )
    }
    result = filter_context_items_by_source_policy(
        [
            _item(
                "alias-kept",
                metadata={
                    "student_id": "student-1",
                    "course_subject": "math",
                    "requested_resource_type": "quiz",
                    "context_purpose": "continuity",
                },
            )
        ],
        injectable_sources=("memory",),
        exclude_message_source=True,
        source_policies=policies,
        state={
            "user_id": "student-1",
            "resource_task": {"subject": "math", "resource_type": "quiz"},
        },
    )

    assert [item.id for item in result.kept_items] == ["alias-kept"]
    assert result.drop_reasons == {}


def test_require_user_match_does_not_use_thread_or_session_aliases():
    policies = {
        "memory": SourceBudgetPolicy(
            source_type="memory",
            require_user_match=True,
        )
    }
    result = filter_context_items_by_source_policy(
        [
            _item("thread-only", metadata={"thread_id": "thread-1"}),
            _item("user-kept", metadata={"user_id": "user-1"}),
        ],
        injectable_sources=("memory",),
        exclude_message_source=True,
        source_policies=policies,
        state={"thread_id": "thread-1", "user_id": "user-1"},
    )

    assert [item.id for item in result.kept_items] == ["user-kept"]
    assert result.source_drop_reasons["user_mismatch"] == 1


def test_thread_match_keeps_cross_resource_artifact_when_subject_matches():
    policies = {
        "artifact": SourceBudgetPolicy(
            source_type="artifact",
            require_thread_match=True,
            require_subject_match=True,
            require_task_match=False,
            allowed_purposes=("artifact_reference",),
            min_relevance_score=0.4,
        )
    }
    result = filter_context_items_by_source_policy(
        [
            _item(
                "prior-review-doc",
                source_type="artifact",
                relevance_score=0.8,
                metadata={
                    "thread_id": "thread-1",
                    "normalized_subject": "machine_learning",
                    "resource_type": "review_doc",
                    "purpose": "artifact_reference",
                },
            )
        ],
        injectable_sources=("artifact",),
        exclude_message_source=True,
        source_policies=policies,
        state={
            "thread_id": "thread-1",
            "subject": "Machine Learning",
            "requested_resource_type": "quiz",
        },
    )

    assert [item.id for item in result.kept_items] == ["prior-review-doc"]
    assert result.drop_reasons == {}


def test_thread_match_drops_artifact_on_thread_mismatch():
    policies = {
        "artifact": SourceBudgetPolicy(
            source_type="artifact",
            require_thread_match=True,
            allowed_purposes=("artifact_reference",),
        )
    }
    result = filter_context_items_by_source_policy(
        [
            _item(
                "other-thread-artifact",
                source_type="artifact",
                metadata={
                    "thread_id": "thread-other",
                    "purpose": "artifact_reference",
                },
            )
        ],
        injectable_sources=("artifact",),
        exclude_message_source=True,
        source_policies=policies,
        state={"thread_id": "thread-1"},
    )

    assert result.kept_items == []
    assert result.source_drop_reasons["thread_mismatch"] == 1


def test_source_policy_without_thread_match_behaves_as_before():
    policies = {
        "artifact": SourceBudgetPolicy(
            source_type="artifact",
            allowed_purposes=("artifact_reference",),
        )
    }
    result = filter_context_items_by_source_policy(
        [
            _item(
                "artifact",
                source_type="artifact",
                metadata={
                    "thread_id": "thread-other",
                    "purpose": "artifact_reference",
                },
            )
        ],
        injectable_sources=("artifact",),
        exclude_message_source=True,
        source_policies=policies,
        state={"thread_id": "thread-1"},
    )

    assert [item.id for item in result.kept_items] == ["artifact"]


def test_task_match_does_not_use_top_level_requested_resource_type_for_multi_resource():
    policies = {
        "memory": SourceBudgetPolicy(
            source_type="memory",
            require_task_match=True,
        )
    }
    result = filter_context_items_by_source_policy(
        [_item("review-memory", metadata={"task_id": "review_doc"})],
        injectable_sources=("memory",),
        exclude_message_source=True,
        source_policies=policies,
        state={
            "requested_resource_type": "review_doc",
            "requested_resource_types": ["review_doc", "quiz"],
        },
    )

    assert result.kept_items == []
    assert result.source_drop_reasons["task_mismatch"] == 1


def test_task_match_prefers_resource_task_over_multi_resource_top_level_state():
    policies = {
        "memory": SourceBudgetPolicy(
            source_type="memory",
            require_task_match=True,
        )
    }
    result = filter_context_items_by_source_policy(
        [
            _item("review-memory", metadata={"task_id": "review_doc"}),
            _item("quiz-memory", metadata={"task_id": "quiz"}),
        ],
        injectable_sources=("memory",),
        exclude_message_source=True,
        source_policies=policies,
        state={
            "resource_task": {"resource_type": "quiz"},
            "requested_resource_type": "review_doc",
            "requested_resource_types": ["review_doc", "quiz"],
        },
    )

    assert [item.id for item in result.kept_items] == ["quiz-memory"]
    assert result.source_drop_reasons["task_mismatch"] == 1


def test_task_match_allows_single_requested_resource_type():
    policies = {
        "memory": SourceBudgetPolicy(
            source_type="memory",
            require_task_match=True,
        )
    }
    result = filter_context_items_by_source_policy(
        [_item("review-memory", metadata={"requested_resource_type": "review_doc"})],
        injectable_sources=("memory",),
        exclude_message_source=True,
        source_policies=policies,
        state={"requested_resource_types": ["review_doc"]},
    )

    assert [item.id for item in result.kept_items] == ["review-memory"]
    assert result.source_drop_reasons == {}


def test_evidence_without_relevance_score_is_not_injectable():
    result = filter_context_items_by_source_policy(
        [_item("evidence", source_type="evidence", relevance_score=None)],
        injectable_sources=("evidence",),
        exclude_message_source=True,
        source_policies={
            "evidence": SourceBudgetPolicy(
                source_type="evidence",
                min_relevance_score=0.35,
            )
        },
        state={},
    )

    assert result.kept_items == []
    assert result.source_drop_reasons["missing_required_relevance_score"] == 1


def test_source_filter_downranks_stale_without_dropping():
    policies = {
        "memory": SourceBudgetPolicy(
            source_type="memory",
            max_items=1,
            stale_policy="downrank",
        )
    }
    result = filter_context_items_by_source_policy(
        [
            _item("stale", priority=100, metadata={"stale": True}),
            _item("fresh", priority=10, metadata={"stale": False}),
        ],
        injectable_sources=("memory",),
        exclude_message_source=True,
        source_policies=policies,
        state={},
    )

    assert [item.id for item in result.kept_items] == ["fresh"]
    assert result.budget_drop_reasons["source_budget_exceeded"] == 1
    assert "stale_context" not in result.drop_reasons


def test_source_filter_drops_stale_when_policy_is_drop():
    policies = {"memory": SourceBudgetPolicy(source_type="memory", stale_policy="drop")}
    result = filter_context_items_by_source_policy(
        [_item("stale", metadata={"stale": True})],
        injectable_sources=("memory",),
        exclude_message_source=True,
        source_policies=policies,
        state={},
    )

    assert result.kept_items == []
    assert result.source_drop_reasons["stale_context"] == 1


def test_source_filter_keep_policy_does_not_downrank_stale():
    policies = {
        "memory": SourceBudgetPolicy(
            source_type="memory",
            max_items=1,
            stale_policy="keep",
        )
    }
    result = filter_context_items_by_source_policy(
        [
            _item("stale", priority=100, metadata={"stale": True}),
            _item("fresh", priority=10, metadata={"stale": False}),
        ],
        injectable_sources=("memory",),
        exclude_message_source=True,
        source_policies=policies,
        state={},
    )

    assert [item.id for item in result.kept_items] == ["stale"]


@pytest.mark.anyio
async def test_disabled_mode_emits_selection_without_collecting(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    mock_llm = _mock_llm()
    selections: list[str] = []

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    trace_payloads: list[dict] = []
    _patch_orchestrator(
        monkeypatch,
        policy=_policy(mode="disabled"),
        items=[],
        trace_payloads=trace_payloads,
    )
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 0)
    monkeypatch.setattr(llm_module, "emit_context_usage_trace", lambda *_, **__: None)

    result = await invoke_plain_llm_fail_fast(
        node_name="plain_node",
        llm_node="llm",
        messages=messages,
        state={"request_id": "r1", "thread_id": "t1"},
    )

    assert result == "answer"
    selections = [
        payload["skip_reason"]
        for payload in trace_payloads
        if payload["stage"] == "context_apply_selection"
    ]
    assert selections == ["node_policy_disabled"]
    mock_llm.ainvoke.assert_awaited_once_with(messages)


@pytest.mark.anyio
async def test_observe_only_collects_filters_and_keeps_original_messages(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    mock_llm = _mock_llm()
    trace_payloads: list[dict] = []

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    _patch_orchestrator(
        monkeypatch,
        policy=_policy(mode="observe_only"),
        items=[_item("memory-1")],
        trace_payloads=trace_payloads,
    )
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 0)
    monkeypatch.setattr(llm_module, "emit_context_usage_trace", lambda *_, **__: None)

    result = await invoke_plain_llm_fail_fast(
        node_name="plain_node",
        llm_node="llm",
        messages=messages,
        state={"request_id": "r1", "thread_id": "t1"},
    )

    assert result == "answer"
    selections = [
        payload
        for payload in trace_payloads
        if payload["stage"] == "context_apply_selection"
    ]
    applied = [
        payload for payload in trace_payloads if payload["stage"] == "context_applied"
    ]
    assert selections[0]["skip_reason"] == "node_policy_observe_only"
    assert selections[0]["final_injected_count"] == 1
    assert applied == []
    mock_llm.ainvoke.assert_awaited_once_with(messages)


@pytest.mark.anyio
async def test_active_mode_applies_context(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    mock_llm = _mock_llm()

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    _patch_orchestrator(
        monkeypatch,
        policy=_policy(mode="active"),
        items=[_item("memory-1")],
    )
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 0)
    monkeypatch.setattr(llm_module, "emit_context_usage_trace", lambda *_, **__: None)

    result = await invoke_plain_llm_fail_fast(
        node_name="plain_node",
        llm_node="llm",
        messages=messages,
        state={
            "request_id": "r1",
            "thread_id": "t1",
            "requested_resource_types": ["review_doc"],
        },
    )

    assert result == "answer"
    final_messages = mock_llm.ainvoke.await_args.args[0]
    assert final_messages is not messages
    assert "<INJECTED_CONTEXT>" in final_messages[0]["content"]


@pytest.mark.anyio
async def test_active_apply_error_does_not_fallback_to_original_messages(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    mock_llm = _mock_llm()
    trace_payloads: list[dict] = []

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    _patch_orchestrator(
        monkeypatch,
        policy=_policy(mode="active"),
        items=[_item("memory-1")],
        packing_enabled=False,
        trace_payloads=trace_payloads,
    )
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 0)
    monkeypatch.setattr(llm_module, "emit_context_usage_trace", lambda *_, **__: None)

    with pytest.raises(ContextApplyError) as exc_info:
        await invoke_plain_llm_fail_fast(
            node_name="plain_node",
            llm_node="llm",
            messages=messages,
            state={
                "request_id": "r1",
                "thread_id": "t1",
                "requested_resource_types": ["review_doc"],
            },
        )

    assert exc_info.value.reason == "packed_context_missing"
    errors = [
        payload
        for payload in trace_payloads
        if payload["stage"] == "context_apply_error"
    ]
    assert "fallback_used" not in errors[0]
    mock_llm.ainvoke.assert_not_awaited()


@pytest.mark.anyio
async def test_packed_context_missing_emits_plan_and_selection(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    mock_llm = _mock_llm()
    trace_payloads: list[dict] = []

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    _patch_orchestrator(
        monkeypatch,
        policy=_policy(mode="active"),
        items=[_item("memory-1")],
        packing_enabled=False,
        trace_payloads=trace_payloads,
    )
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 0)
    monkeypatch.setattr(llm_module, "emit_context_usage_trace", lambda *_, **__: None)

    with pytest.raises(ContextApplyError):
        await invoke_plain_llm_fail_fast(
            node_name="plain_node",
            llm_node="llm",
            messages=messages,
            state={
                "request_id": "r1",
                "thread_id": "t1",
                "requested_resource_types": ["review_doc"],
            },
        )

    plans = [
        payload
        for payload in trace_payloads
        if payload["stage"] == "context_apply_plan"
    ]
    selections = [
        payload["skip_reason"]
        for payload in trace_payloads
        if payload["stage"] == "context_apply_selection"
    ]
    applied = [
        payload for payload in trace_payloads if payload["stage"] == "context_applied"
    ]
    assert len(plans) == 1
    assert plans[0]["selected_item_count"] == 0
    assert plans[0]["injectable_item_count"] == 0
    assert plans[0]["skipped_item_count"] == 0
    assert selections == ["packed_context_missing"]
    assert applied == []
    mock_llm.ainvoke.assert_not_awaited()


@pytest.mark.anyio
async def test_inherit_missing_fail_fast_before_llm(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    mock_llm = _mock_llm()
    trace_payloads: list[dict] = []

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    _patch_orchestrator(
        monkeypatch,
        error=ContextApplyError(
            reason="context_apply_node_policy_inherit_missing",
            warning="missing group",
            node_name="plain_node",
            llm_node="llm",
        ),
        trace_payloads=trace_payloads,
    )
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 0)
    monkeypatch.setattr(llm_module, "emit_context_usage_trace", lambda *_, **__: None)

    with pytest.raises(ContextApplyError) as exc_info:
        await invoke_plain_llm_fail_fast(
            node_name="plain_node",
            llm_node="llm",
            messages=messages,
            state={"request_id": "r1", "thread_id": "t1"},
        )

    assert exc_info.value.reason == "context_apply_node_policy_inherit_missing"
    errors = [
        payload
        for payload in trace_payloads
        if payload["stage"] == "context_apply_error"
    ]
    assert errors[0]["reason"] == "context_apply_node_policy_inherit_missing"
    assert "fallback_used" not in errors[0]
    mock_llm.ainvoke.assert_not_awaited()


def test_injected_context_header_preserves_user_requested_depth():
    rendered, _tokens = apply_module.render_injected_context(
        items=[_item("memory-1")],
        max_tokens=10000,
    )

    assert "Do not reduce the user's requested depth" in rendered
    assert "examples, structure, self-check items, or deliverables" in rendered
