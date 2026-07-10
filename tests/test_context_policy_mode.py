"""Strict/broad Context Engineering runtime-policy tests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from src.context_engineering.packing import node_policy as node_policy_module
from src.context_engineering.packing import orchestrator as orchestrator_module
from src.context_engineering.packing.node_policy import (
    SourceBudgetPolicy,
)
from src.context_engineering.packing.source_policy import (
    filter_context_items_by_source_policy,
)
from src.context_engineering.policy_mode import (
    BroadSourcePolicy,
    ContextRuntimePolicy,
    resolve_context_runtime_policy,
)
from src.context_engineering.schema import ContextConfigError, ContextItem


def _broad_config(*, mode: str = "strict") -> dict[str, Any]:
    caps = {
        "pipeline": (8, 5000, False, True),
        "evidence": (6, 4000, False, False),
        "artifact": (4, 2500, False, True),
        "memory": (4, 2000, True, True),
        "profile": (2, 1000, True, False),
        "rules": (4, 2000, False, False),
        "curriculum": (4, 2000, False, False),
        "trajectory": (4, 2000, True, True),
    }
    return {
        "policy_mode": mode,
        "broad_policy": {
            "max_items_total": 24,
            "max_injected_context_tokens": 12000,
            "max_content_chars_per_item": 2000,
            "enabled_sources": list(caps),
            "eligible_node_roles": ["planner", "agent", "reviewer", "consensus"],
            "bypass_business_filters": [
                "purpose",
                "quality",
                "relevance",
                "subject",
                "task",
                "stale",
            ],
            "source_caps": {
                source: {
                    "max_items": values[0],
                    "max_tokens": values[1],
                    "require_user_match": values[2],
                    "require_thread_match": values[3],
                }
                for source, values in caps.items()
            },
        },
    }


def _runtime_policy(*, mode: str = "broad") -> ContextRuntimePolicy:
    config = _broad_config(mode=mode)["broad_policy"]
    source_policies = {
        source: BroadSourcePolicy(
            source_type=source,
            max_items=values["max_items"],
            max_tokens=values["max_tokens"],
            require_user_match=values["require_user_match"],
            require_thread_match=values["require_thread_match"],
        )
        for source, values in config["source_caps"].items()
    }
    return ContextRuntimePolicy(
        mode=mode,
        environment="development",
        max_items_total=24,
        max_injected_context_tokens=12000,
        max_content_chars_per_item=2000,
        enabled_sources=tuple(config["enabled_sources"]),
        eligible_node_roles=tuple(config["eligible_node_roles"]),
        source_policies=source_policies,
        bypass_business_filters=frozenset(config["bypass_business_filters"]),
    )


def _item(
    item_id: str,
    *,
    source_type: str,
    metadata: dict[str, Any] | None = None,
    relevance_score: float | None = 0.8,
    content: str = "bounded context",
) -> ContextItem:
    return ContextItem(
        id=item_id,
        source_type=source_type,
        title=item_id,
        content=content,
        token_estimate=8,
        estimated=True,
        tokenizer_mode="estimated_mixed",
        priority=50,
        relevance_score=relevance_score,
        recency_score=0.8,
        confidence=0.8,
        scope="session",
        lifetime="session",
        compressible=True,
        can_drop=True,
        disclosure_level="snippet",
        metadata=dict(metadata or {}),
    )


def test_runtime_policy_uses_explicit_strict_config(monkeypatch):
    from src.context_engineering import policy_mode as policy_mode_module

    monkeypatch.setattr(
        policy_mode_module,
        "get_setting",
        lambda key: _broad_config() if key == "context_engineering" else None,
    )
    monkeypatch.delenv("CONTEXT_POLICY_MODE", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("A3_ENV", raising=False)

    policy = resolve_context_runtime_policy()

    assert policy.mode == "strict"
    assert policy.max_items_total == 24
    assert policy.max_injected_context_tokens == 12000
    assert policy.source_policies["pipeline"].max_tokens == 5000


def test_runtime_policy_env_override_is_validated(monkeypatch):
    from src.context_engineering import policy_mode as policy_mode_module

    monkeypatch.setattr(
        policy_mode_module,
        "get_setting",
        lambda key: _broad_config() if key == "context_engineering" else None,
    )
    monkeypatch.setenv("CONTEXT_POLICY_MODE", "wide")

    with pytest.raises(ContextConfigError) as exc_info:
        resolve_context_runtime_policy()

    assert exc_info.value.reason == "context_policy_mode_invalid"


def test_production_rejects_broad_policy(monkeypatch):
    from src.context_engineering import policy_mode as policy_mode_module

    monkeypatch.setattr(
        policy_mode_module,
        "get_setting",
        lambda key: _broad_config() if key == "context_engineering" else None,
    )
    monkeypatch.setenv("CONTEXT_POLICY_MODE", "broad")
    monkeypatch.setenv("APP_ENV", "production")

    with pytest.raises(ContextConfigError) as exc_info:
        resolve_context_runtime_policy()

    assert exc_info.value.reason == "context_policy_mode_forbidden_in_production"


def test_broad_policy_requires_exact_bounded_source_caps(monkeypatch):
    from src.context_engineering import policy_mode as policy_mode_module

    config = _broad_config()
    del config["broad_policy"]["source_caps"]["pipeline"]
    monkeypatch.setattr(
        policy_mode_module,
        "get_setting",
        lambda key: config if key == "context_engineering" else None,
    )

    with pytest.raises(ContextConfigError) as exc_info:
        resolve_context_runtime_policy()

    assert exc_info.value.reason == "context_broad_source_caps_invalid"


def test_broad_policy_only_expands_active_eligible_nodes(monkeypatch):
    from tests.test_context_apply_node_policy import _policy, _resolved

    monkeypatch.setattr(
        node_policy_module,
        "resolve_context_runtime_policy",
        lambda: _runtime_policy(),
    )
    active = node_policy_module._apply_runtime_policy(
        _resolved(_policy()),
        runtime_policy=_runtime_policy(),
        node_name="mindmap_agent",
    )
    observe = node_policy_module._apply_runtime_policy(
        replace(_resolved(_policy()), mode="observe_only"),
        runtime_policy=_runtime_policy(),
        node_name="mindmap_agent",
    )
    output = node_policy_module._apply_runtime_policy(
        _resolved(_policy()),
        runtime_policy=_runtime_policy(),
        node_name="mindmap_output",
    )

    assert active.runtime_policy_mode == "broad"
    assert active.injection_policy.max_injected_context_tokens == 12000
    assert active.injection_policy.quality.max_items_total == 24
    assert active.injection_policy.format.max_content_chars_per_item == 2000
    assert active.source_policies["pipeline"].max_items == 8
    assert "pipeline" in active.injection_policy.injectable_sources
    assert observe.injection_policy.injectable_sources == (
        "memory",
        "evidence",
        "rules",
    )
    assert output.injection_policy.injectable_sources == ("memory", "evidence", "rules")


@pytest.mark.parametrize("policy_mode", ["strict", "broad"])
def test_evidence_judge_approval_is_never_bypassed(policy_mode):
    result = filter_context_items_by_source_policy(
        [
            _item(
                "unapproved",
                source_type="evidence",
                metadata={"grounding_approved": False},
            ),
            _item(
                "approved",
                source_type="evidence",
                metadata={"grounding_approved": True},
            ),
        ],
        injectable_sources=("evidence",),
        exclude_message_source=True,
        source_policies={"evidence": SourceBudgetPolicy(source_type="evidence")},
        state={},
        policy_mode=policy_mode,
    )

    assert [item.id for item in result.kept_items] == ["approved"]
    assert result.source_drop_reasons == {"grounding_not_approved": 1}


def test_broad_bypasses_business_filters_but_keeps_thread_isolation():
    policy = SourceBudgetPolicy(
        source_type="artifact",
        min_priority=99,
        min_relevance_score=0.99,
        allowed_purposes=("other",),
        require_thread_match=True,
        require_subject_match=True,
        require_task_match=True,
        stale_policy="drop",
    )
    same_thread = _item(
        "same",
        source_type="artifact",
        metadata={
            "thread_id": "thread-1",
            "subject": "different",
            "resource_type": "review_doc",
            "purpose": "artifact_reference",
            "stale": True,
        },
        relevance_score=0.1,
    )
    other_thread = same_thread.model_copy(
        update={
            "id": "other",
            "metadata": {**same_thread.metadata, "thread_id": "thread-2"},
        }
    )

    result = filter_context_items_by_source_policy(
        [same_thread, other_thread],
        injectable_sources=("artifact",),
        exclude_message_source=True,
        source_policies={"artifact": policy},
        state={
            "thread_id": "thread-1",
            "subject": "machine learning",
            "requested_resource_type": "mindmap",
        },
        policy_mode="broad",
    )

    assert [item.id for item in result.kept_items] == ["same"]
    assert result.source_drop_reasons == {"thread_mismatch": 1}
    assert "broad_business_filters_bypassed" in result.warnings


def test_pipeline_safety_allows_upstream_and_rejects_future_self_and_duplicate():
    base_metadata = {
        "request_id": "request-1",
        "thread_id": "thread-1",
        "workflow": "mindmap",
        "iteration": 0,
    }
    items = [
        _item(
            "planner",
            source_type="pipeline",
            metadata={
                **base_metadata,
                "source_node": "mindmap_planner",
                "content_fingerprint": "planner-fingerprint",
            },
        ),
        _item(
            "self",
            source_type="pipeline",
            metadata={**base_metadata, "source_node": "mindmap_agent"},
        ),
        _item(
            "future",
            source_type="pipeline",
            metadata={**base_metadata, "source_node": "mindmap_reviewer"},
        ),
        _item(
            "duplicate",
            source_type="pipeline",
            metadata={
                **base_metadata,
                "source_node": "mindmap_planner",
                "content_fingerprint": "existing-fingerprint",
            },
        ),
    ]
    result = filter_context_items_by_source_policy(
        items,
        injectable_sources=("pipeline",),
        exclude_message_source=True,
        source_policies={
            "pipeline": SourceBudgetPolicy(
                source_type="pipeline",
                require_thread_match=True,
            )
        },
        state={"request_id": "request-1", "thread_id": "thread-1"},
        policy_mode="broad",
        target_node_name="mindmap_agent",
        existing_content_fingerprints={"existing-fingerprint"},
    )

    assert [item.id for item in result.kept_items] == ["planner"]
    assert result.source_drop_reasons == {
        "pipeline_self_output": 1,
        "pipeline_future_output": 1,
        "duplicate_provider_input": 1,
    }


def test_pipeline_reviewer_eligibility_is_iteration_aware():
    reviewer = _item(
        "reviewer",
        source_type="pipeline",
        metadata={
            "request_id": "request-1",
            "thread_id": "thread-1",
            "source_node": "mindmap_reviewer",
            "iteration": 0,
        },
    )
    rewrite = filter_context_items_by_source_policy(
        [reviewer],
        injectable_sources=("pipeline",),
        exclude_message_source=True,
        source_policies={"pipeline": SourceBudgetPolicy(source_type="pipeline")},
        state={"request_id": "request-1", "thread_id": "thread-1"},
        policy_mode="broad",
        target_node_name="mindmap_rewrite",
    )
    next_agent = filter_context_items_by_source_policy(
        [reviewer],
        injectable_sources=("pipeline",),
        exclude_message_source=True,
        source_policies={"pipeline": SourceBudgetPolicy(source_type="pipeline")},
        state={
            "request_id": "request-1",
            "thread_id": "thread-1",
            "mindmap_round": 1,
        },
        policy_mode="broad",
        target_node_name="mindmap_agent",
    )

    assert [item.id for item in rewrite.kept_items] == ["reviewer"]
    assert [item.id for item in next_agent.kept_items] == ["reviewer"]


def test_provider_remaining_input_window_constrains_ce_budget(monkeypatch):
    from tests.test_context_apply_node_policy import _policy, _resolved

    resolved = _resolved(_policy(max_tokens=80))
    monkeypatch.setattr(
        orchestrator_module,
        "build_context_budget",
        lambda **_: SimpleNamespace(max_input_tokens=100),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "estimate_messages_tokens_mixed",
        lambda _messages: 70,
    )

    budget = orchestrator_module._constrain_policy_to_provider_budget(
        resolved=resolved,
        node_name="mindmap_agent",
        llm_node="mindmap",
        model="configured-model",
        messages=[{"role": "user", "content": "request"}],
    )

    assert budget.provider_input_budget_tokens == 100
    assert budget.provider_input_tokens_before_context == 70
    assert budget.provider_remaining_input_tokens == 30
    assert budget.effective_context_budget_tokens == 30
    assert budget.resolved.injection_policy.max_injected_context_tokens == 30


def test_provider_input_overflow_fails_before_context_collection(monkeypatch):
    from src.context_engineering.packing.apply import ContextApplyError
    from tests.test_context_apply_node_policy import _policy, _resolved

    monkeypatch.setattr(
        orchestrator_module,
        "build_context_budget",
        lambda **_: SimpleNamespace(max_input_tokens=100),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "estimate_messages_tokens_mixed",
        lambda _messages: 100,
    )

    with pytest.raises(ContextApplyError) as exc_info:
        orchestrator_module._constrain_policy_to_provider_budget(
            resolved=_resolved(_policy()),
            node_name="mindmap_agent",
            llm_node="mindmap",
            model="configured-model",
            messages=[{"role": "user", "content": "request"}],
        )

    assert exc_info.value.reason == "provider_context_budget_exceeded"
    assert exc_info.value.error_scope == "budget"
