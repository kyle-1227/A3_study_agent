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
from src.context_engineering.packing.source_policy import (
    filter_context_items_by_source_policy,
)
from src.context_engineering.schema import ContextItem


def _legacy_apply_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "apply_enabled_nodes": ["review_doc_agent"],
        "fallback_on_error": True,
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
    fallback_on_error: bool = True,
    max_tokens: int = 10000,
) -> ContextInjectionPolicy:
    return ContextInjectionPolicy(
        enabled=True,
        apply_enabled_nodes=nodes if mode == "active" else (),
        fallback_on_error=fallback_on_error,
        allow_structured_output=False,
        role="system",
        position="after_system",
        exclude_message_source=True,
        max_injected_context_tokens=max_tokens,
        injectable_sources=("memory", "evidence", "rules"),
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
            fallback_if_empty_after_drop=fallback_on_error,
        ),
        format=ApplyFormatPolicy(
            group_by_source=True,
            include_untrusted_context_warning=True,
            include_section_headers=True,
            max_content_chars_per_item=4000,
        ),
        importance_scoring=ImportanceScoringPolicy(
            enabled=False,
            shadow_mode=True,
            mode="shadow",
            llm_node="",
            max_items_to_score=0,
            max_content_preview_chars=0,
            timeout_seconds=0.0,
            fallback_to_rule_based=True,
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
        metadata=metadata or {},
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
    assert (
        resolve_context_policy(
            node_name="mindmap_agent",
            llm_node="mindmap",
            state={},
        ).mode
        == "active"
    )
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
    assert reviewer.injection_policy.injectable_sources == ("rules", "evidence")
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
    monkeypatch.setattr(
        llm_module,
        "resolve_context_policy",
        lambda **_: _resolved(_policy(mode="disabled")),
    )
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 0)
    monkeypatch.setattr(llm_module, "emit_context_usage_trace", lambda *_, **__: None)
    monkeypatch.setattr(
        llm_module,
        "emit_context_items_shadow",
        lambda *_, **__: pytest.fail("disabled mode must not collect context"),
    )
    monkeypatch.setattr(
        llm_module,
        "emit_context_apply_selection",
        lambda _logger, *, selection, **_kwargs: selections.append(
            selection.skip_reason
        ),
    )

    result = await invoke_plain_llm_fail_fast(
        node_name="plain_node",
        llm_node="llm",
        messages=messages,
        state={"request_id": "r1", "thread_id": "t1"},
    )

    assert result == "answer"
    assert selections == ["node_policy_disabled"]
    mock_llm.ainvoke.assert_awaited_once_with(messages)


@pytest.mark.anyio
async def test_observe_only_collects_filters_and_keeps_original_messages(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    mock_llm = _mock_llm()
    selections: list[Any] = []

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    monkeypatch.setattr(
        llm_module,
        "resolve_context_policy",
        lambda **_: _resolved(_policy(mode="observe_only")),
    )
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 0)
    monkeypatch.setattr(llm_module, "emit_context_usage_trace", lambda *_, **__: None)
    monkeypatch.setattr(
        llm_module, "emit_context_applied", lambda *_, **__: pytest.fail()
    )
    monkeypatch.setattr(
        llm_module,
        "emit_context_items_shadow",
        lambda *_, **__: [_item("memory-1")],
    )
    monkeypatch.setattr(
        llm_module,
        "emit_context_packing_shadow",
        lambda *_, **__: _packed([_item("memory-1")]),
    )
    monkeypatch.setattr(
        llm_module,
        "emit_context_apply_selection",
        lambda _logger, *, selection, **_kwargs: selections.append(selection),
    )

    result = await invoke_plain_llm_fail_fast(
        node_name="plain_node",
        llm_node="llm",
        messages=messages,
        state={"request_id": "r1", "thread_id": "t1"},
    )

    assert result == "answer"
    assert selections[0].skip_reason == "node_policy_observe_only"
    assert selections[0].final_injected_count == 1
    mock_llm.ainvoke.assert_awaited_once_with(messages)


@pytest.mark.anyio
async def test_active_mode_applies_context(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    mock_llm = _mock_llm()

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    monkeypatch.setattr(
        llm_module,
        "resolve_context_policy",
        lambda **_: _resolved(_policy(mode="active")),
    )
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 0)
    monkeypatch.setattr(llm_module, "emit_context_usage_trace", lambda *_, **__: None)
    monkeypatch.setattr(
        llm_module,
        "emit_context_items_shadow",
        lambda *_, **__: [_item("memory-1")],
    )
    monkeypatch.setattr(
        llm_module,
        "emit_context_packing_shadow",
        lambda *_, **__: _packed([_item("memory-1")]),
    )

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
async def test_fallback_on_error_keeps_original_messages(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    mock_llm = _mock_llm()
    errors: list[ContextApplyError] = []

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    monkeypatch.setattr(
        llm_module,
        "resolve_context_policy",
        lambda **_: _resolved(_policy(mode="active", fallback_on_error=True)),
    )
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 0)
    monkeypatch.setattr(llm_module, "emit_context_usage_trace", lambda *_, **__: None)
    monkeypatch.setattr(
        llm_module,
        "emit_context_items_shadow",
        lambda *_, **__: [_item("memory-1")],
    )
    monkeypatch.setattr(
        llm_module, "emit_context_packing_shadow", lambda *_, **__: None
    )
    monkeypatch.setattr(
        llm_module,
        "emit_context_apply_error",
        lambda _logger, *, error, **_kwargs: errors.append(error),
    )

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
    assert errors[0].fallback_used is True
    mock_llm.ainvoke.assert_awaited_once_with(messages)


@pytest.mark.anyio
async def test_packed_context_missing_emits_plan_and_selection(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    mock_llm = _mock_llm()
    plans: list[dict[str, int]] = []
    selections: list[str] = []

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    monkeypatch.setattr(
        llm_module,
        "resolve_context_policy",
        lambda **_: _resolved(_policy(mode="active", fallback_on_error=True)),
    )
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 0)
    monkeypatch.setattr(llm_module, "emit_context_usage_trace", lambda *_, **__: None)
    monkeypatch.setattr(
        llm_module,
        "emit_context_items_shadow",
        lambda *_, **__: [_item("memory-1")],
    )
    monkeypatch.setattr(
        llm_module, "emit_context_packing_shadow", lambda *_, **__: None
    )
    monkeypatch.setattr(
        llm_module,
        "emit_context_apply_plan",
        lambda _logger, **kwargs: plans.append(
            {
                "selected": kwargs["selected_item_count"],
                "injectable": kwargs["injectable_item_count"],
                "skipped": kwargs["skipped_item_count"],
            }
        ),
    )
    monkeypatch.setattr(
        llm_module,
        "emit_context_apply_selection",
        lambda _logger, *, selection, **_kwargs: selections.append(
            selection.skip_reason
        ),
    )
    monkeypatch.setattr(llm_module, "emit_context_apply_error", lambda *_, **__: None)

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
    assert plans == [{"selected": 0, "injectable": 0, "skipped": 0}]
    assert selections == ["packed_context_missing"]
    mock_llm.ainvoke.assert_awaited_once_with(messages)


@pytest.mark.anyio
async def test_inherit_missing_fallback_uses_original_messages(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    mock_llm = _mock_llm()
    errors: list[ContextApplyError] = []

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    monkeypatch.setattr(
        llm_module,
        "resolve_context_policy",
        lambda **_: (_ for _ in ()).throw(
            ContextApplyError(
                reason="context_apply_node_policy_inherit_missing",
                warning="missing group",
                node_name="plain_node",
                llm_node="llm",
            )
        ),
    )
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 0)
    monkeypatch.setattr(llm_module, "emit_context_usage_trace", lambda *_, **__: None)
    monkeypatch.setattr(
        llm_module,
        "emit_context_items_shadow",
        lambda *_, **__: [],
    )
    monkeypatch.setattr(
        llm_module, "emit_context_packing_shadow", lambda *_, **__: None
    )
    monkeypatch.setattr(
        llm_module,
        "emit_context_apply_error",
        lambda _logger, *, error, **_kwargs: errors.append(error),
    )

    result = await invoke_plain_llm_fail_fast(
        node_name="plain_node",
        llm_node="llm",
        messages=messages,
        state={"request_id": "r1", "thread_id": "t1"},
    )

    assert result == "answer"
    assert errors[0].reason == "context_apply_node_policy_inherit_missing"
    assert errors[0].fallback_used is True
    mock_llm.ainvoke.assert_awaited_once_with(messages)


def test_injected_context_header_preserves_user_requested_depth():
    rendered, _tokens = apply_module.render_injected_context(
        items=[_item("memory-1")],
        max_tokens=10000,
    )

    assert "Do not reduce the user's requested depth" in rendered
    assert "examples, structure, self-check items, or deliverables" in rendered
