"""Plain LLM integration tests for Phase 3B-1 context apply."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

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
)
from src.context_engineering.packing.packer import pack_context_items
from src.context_engineering.packing.policies import PackingPolicy
from src.context_engineering.providers.supply import (
    ContextCollectionResult,
    ProviderSupplyPlan,
)
from src.context_engineering.evidence_normalizer import EvidenceNormalizationStats
from src.context_engineering.schema import ContextItem


def _policy(
    *,
    enabled: bool = True,
    nodes: tuple[str, ...] = ("plain_node",),
    fallback_on_error: bool = False,
    max_tokens: int = 10000,
    mode: str = "active",
    required_sources: tuple[str, ...] = (),
) -> ContextInjectionPolicy:
    return ContextInjectionPolicy(
        enabled=enabled,
        apply_enabled_nodes=nodes,
        fallback_on_error=fallback_on_error,
        allow_structured_output=False,
        role="system",
        position="after_system",
        exclude_message_source=True,
        max_injected_context_tokens=max_tokens,
        injectable_sources=("memory", "evidence", "rules"),
        required_sources=required_sources,
        mode=mode if enabled else "disabled",
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
            graceful_degradation_enabled=False,
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
            shadow_mode=False,
            mode="disabled",
            llm_node="",
            max_items_to_score=0,
            max_content_preview_chars=0,
            timeout_seconds=0.0,
            fallback_to_rule_based=False,
            emit_shadow_telemetry=False,
            min_shadow_score_for_analysis=0.0,
        ),
    )


def _resolved(policy: ContextInjectionPolicy) -> ResolvedContextPolicy:
    return ResolvedContextPolicy(
        mode=policy.mode if policy.enabled else "disabled",
        risk_tier=policy.risk_tier,
        policy_source=policy.policy_source,
        injection_policy=policy,
        source_policies={},
        legacy_mode_enabled=policy.enabled,
        node_policy_enabled=False,
        summary={},
    )


def _item(
    item_id: str = "memory-1",
    *,
    source_type: str = "memory",
    content: str = "useful memory",
    token_estimate: int = 5,
    priority: int = 80,
    relevance_score: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> ContextItem:
    item_metadata = dict(metadata or {})
    if source_type == "evidence":
        item_metadata.setdefault("grounding_approved", True)
    return ContextItem(
        id=item_id,
        source_type=source_type,
        title=item_id,
        content=content,
        token_estimate=token_estimate,
        estimated=True,
        tokenizer_mode="estimated_mixed",
        priority=priority,
        relevance_score=relevance_score,
        recency_score=None,
        confidence=None,
        scope="turn",
        lifetime="turn",
        compressible=True,
        can_drop=True,
        disclosure_level="snippet",
        metadata=item_metadata,
    )


def _packed(items: list[ContextItem] | None = None):
    return pack_context_items(
        node_name="plain_node",
        llm_node="llm",
        items=items or [_item()],
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
    policy: ContextInjectionPolicy,
    items: list[ContextItem] | None = None,
    source_policies: dict[str, SourceBudgetPolicy] | None = None,
    packing_enabled: bool = True,
    trace_payloads: list[dict] | None = None,
) -> None:
    from src.context_engineering.packing import orchestrator as orchestrator_module

    monkeypatch.setattr(
        orchestrator_module,
        "resolve_context_policy",
        lambda **_: ResolvedContextPolicy(
            mode=policy.mode if policy.enabled else "disabled",
            risk_tier=policy.risk_tier,
            policy_source=policy.policy_source,
            injection_policy=policy,
            source_policies=source_policies or {},
            legacy_mode_enabled=policy.enabled,
            node_policy_enabled=False,
            summary={},
        ),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "collect_context_for_policy",
        lambda **kwargs: _collection(
            items or [_item()],
            requested_sources=kwargs.get("requested_sources"),
            required_sources=kwargs.get("required_sources", ()),
            optional_sources=kwargs.get("optional_sources", ()),
        ),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "get_packing_policy",
        lambda **_: _packing_policy(
            enabled=packing_enabled,
            max_tokens=policy.max_injected_context_tokens,
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


def test_optional_provider_error_with_valid_context_is_degraded_applied(monkeypatch):
    from src.context_engineering.input_manifest import build_llm_input_manifest
    from src.context_engineering.packing import orchestrator as orchestrator_module

    policy = replace(
        _policy(),
        injectable_sources=("rules", "memory"),
        optional_sources=("rules", "memory"),
    )
    _patch_orchestrator(
        monkeypatch,
        policy=policy,
        items=[_item("rules-1", source_type="rules")],
    )

    def collect_with_optional_error(**kwargs):
        plan, collection = _collection(
            [_item("rules-1", source_type="rules")],
            requested_sources=kwargs["requested_sources"],
            required_sources=kwargs["required_sources"],
            optional_sources=kwargs["optional_sources"],
        )
        reasons = {**collection.provider_missing_reasons, "memory": "provider_error"}
        return plan, replace(collection, provider_missing_reasons=reasons)

    monkeypatch.setattr(
        orchestrator_module,
        "collect_context_for_policy",
        collect_with_optional_error,
    )

    prepared = orchestrator_module.prepare_messages_with_context_policy(
        MagicMock(),
        node_name="plain_node",
        llm_node="llm",
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": "question"}],
        state={"request_id": "r1", "thread_id": "t1"},
    )
    manifest = build_llm_input_manifest(
        node_name="plain_node",
        llm_node="llm",
        provider="configured-provider",
        model="deepseek-v4-pro",
        messages=prepared.messages_for_llm,
        state={"request_id": "r1", "thread_id": "t1"},
        call_purpose="plain_llm",
        context_apply_applied=prepared.context_apply_applied,
        context_apply_status=prepared.context_apply_status,
        optional_sources_missing=prepared.optional_sources_missing,
        provider_input_budget_tokens=prepared.provider_input_budget_tokens,
        provider_input_tokens_before_context=(
            prepared.provider_input_tokens_before_context
        ),
        provider_remaining_input_tokens=prepared.provider_remaining_input_tokens,
        effective_context_budget_tokens=prepared.effective_context_budget_tokens,
    )

    assert prepared.context_apply_applied is True
    assert prepared.context_apply_status == "degraded_applied"
    assert prepared.optional_sources_missing == ("memory",)
    assert prepared.apply_result is not None
    assert "optional_provider_error:memory" in prepared.apply_result.warnings
    assert manifest["context_apply_status"] == "degraded_applied"
    assert manifest["optional_sources_missing"] == ["memory"]
    assert manifest["effective_context_budget_tokens"] > 0


@pytest.mark.anyio
async def test_apply_disabled_keeps_original_messages_without_collecting(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    order: list[str] = []
    mock_llm = _mock_llm()

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    _patch_orchestrator(monkeypatch, policy=_policy(enabled=False), items=[])
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 0)
    monkeypatch.setattr(
        llm_module,
        "emit_context_usage_trace",
        lambda *_, **__: order.append("usage"),
    )

    result = await invoke_plain_llm_fail_fast(
        node_name="plain_node",
        llm_node="llm",
        messages=messages,
        state={"request_id": "r1", "thread_id": "t1"},
    )

    assert result == "answer"
    assert order == ["usage"]
    mock_llm.ainvoke.assert_awaited_once_with(messages)
    assert messages == [{"role": "user", "content": "question"}]


@pytest.mark.anyio
async def test_active_resolved_policy_applies_context(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    mock_llm = _mock_llm()

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    _patch_orchestrator(monkeypatch, policy=_policy(nodes=("other_node",)))
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
async def test_observe_only_keeps_original_messages_and_emits_selection(
    monkeypatch,
):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    trace_payloads: list[dict] = []
    mock_llm = _mock_llm()
    policy = _policy(mode="observe_only")

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    _patch_orchestrator(
        monkeypatch,
        policy=policy,
        trace_payloads=trace_payloads,
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
    selections = [
        payload
        for payload in trace_payloads
        if payload["stage"] == "context_apply_selection"
    ]
    assert selections[0]["skip_reason"] == "node_policy_observe_only"
    mock_llm.ainvoke.assert_awaited_once_with(messages)


@pytest.mark.anyio
async def test_apply_enabled_node_uses_final_messages_and_context_usage(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    usage_messages: list[list] = []
    mock_llm = _mock_llm()

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    _patch_orchestrator(monkeypatch, policy=_policy())
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 0)
    monkeypatch.setattr(
        llm_module,
        "emit_context_usage_trace",
        lambda *_, **kwargs: usage_messages.append(kwargs["messages"]),
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
    assert final_messages is usage_messages[0]
    assert final_messages is not messages
    assert "<INJECTED_CONTEXT>" in final_messages[0]["content"]
    assert "useful memory" in final_messages[0]["content"]
    assert messages == [{"role": "user", "content": "question"}]


@pytest.mark.anyio
async def test_plain_llm_output_success_does_not_trace_raw_output(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    trace_payloads: list[dict] = []
    mock_llm = _mock_llm("answer with private details")

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    _patch_orchestrator(monkeypatch, policy=_policy(enabled=False), items=[])
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 0)
    monkeypatch.setattr(llm_module, "emit_context_usage_trace", lambda *_, **__: None)
    monkeypatch.setattr(
        llm_module,
        "emit_a3_trace",
        lambda _logger, stage, payload, **_kwargs: trace_payloads.append(
            {"stage": stage, **payload}
        ),
    )

    result = await invoke_plain_llm_fail_fast(
        node_name="plain_node",
        llm_node="llm",
        messages=messages,
        state={"request_id": "r1", "thread_id": "t1"},
    )

    assert result == "answer with private details"
    output = next(
        payload for payload in trace_payloads if payload["stage"] == "plain_llm_output"
    )
    assert output["raw_output_chars"] == len("answer with private details")
    for forbidden in (
        "raw_output",
        "provider_error_body",
        "final_messages",
        "rendered_context",
        "injected_context",
    ):
        assert forbidden not in output


@pytest.mark.anyio
async def test_plain_llm_output_failure_does_not_trace_raw_provider_response(
    monkeypatch,
):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    trace_payloads: list[dict] = []
    mock_llm = MagicMock()
    mock_llm.model_name = "deepseek-v4-pro"
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("provider failed"))

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    _patch_orchestrator(monkeypatch, policy=_policy(enabled=False), items=[])
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 0)
    monkeypatch.setattr(llm_module, "emit_context_usage_trace", lambda *_, **__: None)
    monkeypatch.setattr(
        llm_module,
        "emit_a3_trace",
        lambda _logger, stage, payload, **_kwargs: trace_payloads.append(
            {"stage": stage, **payload}
        ),
    )

    with pytest.raises(RuntimeError):
        await invoke_plain_llm_fail_fast(
            node_name="plain_node",
            llm_node="llm",
            messages=messages,
            state={"request_id": "r1", "thread_id": "t1"},
        )

    output = next(
        payload for payload in trace_payloads if payload["stage"] == "plain_llm_output"
    )
    assert output["success"] is False
    assert output["raw_output_chars"] == 0
    for forbidden in (
        "raw_output",
        "provider_error_body",
        "final_messages",
        "rendered_context",
        "injected_context",
    ):
        assert forbidden not in output


@pytest.mark.anyio
async def test_apply_error_does_not_fallback_to_original_messages(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    usage_messages: list[list] = []
    trace_payloads: list[dict] = []
    mock_llm = _mock_llm()

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    _patch_orchestrator(
        monkeypatch,
        policy=_policy(max_tokens=1, fallback_on_error=False),
        trace_payloads=trace_payloads,
    )
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 0)
    monkeypatch.setattr(
        llm_module,
        "emit_context_usage_trace",
        lambda *_, **kwargs: usage_messages.append(kwargs["messages"]),
    )

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

    assert exc_info.value.reason == "rendered_context_over_budget"
    assert usage_messages == []
    mock_llm.ainvoke.assert_not_awaited()
    error_payload = next(
        payload
        for payload in trace_payloads
        if payload["stage"] == "context_apply_error"
    )
    assert "fallback_used" not in error_payload
    assert error_payload["recoverable"] is False


@pytest.mark.anyio
async def test_required_source_missing_fail_fast_before_llm(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    trace_payloads: list[dict] = []
    mock_llm = _mock_llm()

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    _patch_orchestrator(
        monkeypatch,
        policy=_policy(required_sources=("evidence",)),
        items=[_item("memory-1", source_type="memory")],
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

    assert exc_info.value.reason == "required_sources_missing"
    assert exc_info.value.error_scope == "provider"
    assert exc_info.value.recoverable is False
    assert exc_info.value.required_sources_missing == ("evidence",)
    assert exc_info.value.provider_missing_reasons == {"evidence": "provider_empty"}
    errors = [
        payload
        for payload in trace_payloads
        if payload["stage"] == "context_apply_error"
    ]
    assert errors[0]["required_sources_missing"] == ["evidence"]
    assert errors[0]["provider_missing_reasons"] == {"evidence": "provider_empty"}
    mock_llm.ainvoke.assert_not_awaited()


@pytest.mark.anyio
async def test_required_evidence_present_applies_and_calls_llm(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    trace_payloads: list[dict] = []
    mock_llm = _mock_llm()

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    _patch_orchestrator(
        monkeypatch,
        policy=_policy(required_sources=("evidence",)),
        items=[
            _item(
                "graded-evidence-1",
                source_type="evidence",
                content="LLM-graded evidence text.",
                relevance_score=0.88,
            )
        ],
        trace_payloads=trace_payloads,
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
    mock_llm.ainvoke.assert_awaited_once()
    final_messages = mock_llm.ainvoke.await_args.args[0]
    assert final_messages is not messages
    assert "<INJECTED_CONTEXT>" in final_messages[0]["content"]
    assert not [
        payload
        for payload in trace_payloads
        if payload["stage"] == "context_apply_error"
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("items", "source_policies", "expected_budget_reasons", "expected_source_reasons"),
    [
        (
            [_item("evidence-low", source_type="evidence", relevance_score=0.2)],
            {
                "evidence": SourceBudgetPolicy(
                    source_type="evidence",
                    min_relevance_score=0.35,
                )
            },
            {},
            {"quality_below_threshold": 1},
        ),
        (
            [_item("rules-low", source_type="rules", priority=10)],
            {
                "rules": SourceBudgetPolicy(
                    source_type="rules",
                    min_priority=30,
                )
            },
            {},
            {"quality_below_threshold": 1},
        ),
        (
            [_item("evidence-too-large", source_type="evidence", relevance_score=0.9)],
            {
                "evidence": SourceBudgetPolicy(
                    source_type="evidence",
                    max_items=0,
                )
            },
            {"source_budget_exceeded": 1},
            {},
        ),
    ],
)
async def test_required_source_filtered_out_fails_fast_before_llm(
    monkeypatch,
    items,
    source_policies,
    expected_budget_reasons,
    expected_source_reasons,
):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    required_source = items[0].source_type
    messages = [{"role": "user", "content": "question"}]
    trace_payloads: list[dict] = []
    mock_llm = _mock_llm()

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    _patch_orchestrator(
        monkeypatch,
        policy=_policy(required_sources=(required_source,)),
        items=items,
        source_policies=source_policies,
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

    assert exc_info.value.reason == "required_sources_filtered_out"
    assert exc_info.value.error_scope == "source_filter"
    assert exc_info.value.recoverable is False
    assert exc_info.value.required_sources_filtered_out == (required_source,)
    assert exc_info.value.source_counts_before == {required_source: 1}
    assert exc_info.value.source_counts_after == {}
    assert exc_info.value.source_counts_dropped == {required_source: 1}
    assert exc_info.value.source_drop_reasons == expected_source_reasons
    assert exc_info.value.budget_drop_reasons == expected_budget_reasons
    mock_llm.ainvoke.assert_not_awaited()

    error_payload = next(
        payload
        for payload in trace_payloads
        if payload["stage"] == "context_apply_error"
    )
    assert error_payload["required_sources_filtered_out"] == [required_source]
    assert error_payload["source_counts_before"] == {required_source: 1}
    assert error_payload["source_counts_after"] == {}
    assert error_payload["source_counts_dropped"] == {required_source: 1}
    assert error_payload["source_drop_reasons"] == expected_source_reasons
    assert error_payload["budget_drop_reasons"] == expected_budget_reasons


@pytest.mark.anyio
async def test_apply_error_without_fallback_raises_before_llm_and_emits_safe_error(
    monkeypatch,
):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    apply_errors: list[ContextApplyError] = []
    mock_llm = _mock_llm()

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    _patch_orchestrator(
        monkeypatch,
        policy=_policy(max_tokens=1, fallback_on_error=False),
        items=[_item(content="api_key=sk-secret-value cookie=session must not leak")],
    )
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 0)
    monkeypatch.setattr(llm_module, "emit_context_usage_trace", lambda *_, **__: None)
    from src.context_engineering.packing import orchestrator as orchestrator_module

    original_emit_error = orchestrator_module._emit_context_apply_error

    def capture_error(*args, error: ContextApplyError, **kwargs):
        apply_errors.append(error)
        return original_emit_error(*args, error=error, **kwargs)

    monkeypatch.setattr(orchestrator_module, "_emit_context_apply_error", capture_error)

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

    assert exc_info.value.reason == "rendered_context_over_budget"
    mock_llm.ainvoke.assert_not_awaited()
    assert len(apply_errors) == 1
    serialized_error = repr(
        {
            "reason": apply_errors[0].reason,
            "warning": apply_errors[0].warning,
        }
    ).lower()
    assert not hasattr(apply_errors[0], "fallback_used")
    assert "api_key" not in serialized_error
    assert "cookie" not in serialized_error
    assert "sk-secret-value" not in serialized_error
    assert "must not leak" not in serialized_error


@pytest.mark.anyio
async def test_apply_retry_uses_same_final_messages_object(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    mock_llm = MagicMock()
    mock_llm.model_name = "deepseek-v4-pro"
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            SimpleNamespace(content=""),
            SimpleNamespace(content="answer"),
        ]
    )

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    _patch_orchestrator(monkeypatch, policy=_policy())
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 1)
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
    first_messages = mock_llm.ainvoke.await_args_list[0].args[0]
    second_messages = mock_llm.ainvoke.await_args_list[1].args[0]
    assert first_messages is second_messages
    assert "<INJECTED_CONTEXT>" in first_messages[0]["content"]
