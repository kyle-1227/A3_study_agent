"""Structured output guard tests for Phase 3B-1 context apply."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from src.context_engineering.schema import ContextItem


class _TinySchema(BaseModel):
    answer: str


def _item() -> ContextItem:
    return ContextItem(
        id="memory-1",
        source_type="memory",
        title="memory",
        content="memory content",
        token_estimate=5,
        estimated=True,
        tokenizer_mode="estimated_mixed",
        priority=80,
        relevance_score=None,
        recency_score=None,
        confidence=None,
        scope="turn",
        lifetime="turn",
        compressible=True,
        can_drop=True,
        disclosure_level="snippet",
        metadata={},
    )


@pytest.mark.anyio
async def test_structured_output_observe_only_never_mutates_provider_messages(
    monkeypatch,
):
    from src.context_engineering.packing import apply as apply_module
    from src.graph import llm as graph_llm
    from src.llm import structured_output
    from src.llm.structured_output import _invoke_one_mode

    original_messages = [{"role": "user", "content": "question"}]
    captured_runnable_messages: list[list] = []
    trace_stages: list[str] = []
    scorer_calls: list[object] = []

    class FakeRunnable:
        async def ainvoke(self, messages):
            captured_runnable_messages.append(messages)
            return SimpleNamespace(content='{"answer": "ok"}')

    class FakeLLM:
        def bind(self, **_kwargs):
            return FakeRunnable()

    async def fake_transport_retry(operation, **_kwargs):
        return await operation(), 0

    def fail_if_apply_called(*_args, **_kwargs):
        raise AssertionError("structured output must not call build_applied_messages")

    async def fail_if_raw_scorer_called(*args, **_kwargs):
        scorer_calls.append(args)
        raise AssertionError("structured output must not call raw importance scorer")

    monkeypatch.setattr(apply_module, "build_applied_messages", fail_if_apply_called)
    monkeypatch.setattr(
        graph_llm,
        "invoke_context_importance_scorer_raw",
        fail_if_raw_scorer_called,
    )
    monkeypatch.setattr(
        structured_output,
        "emit_a3_trace",
        lambda _logger, stage, _payload, **_kwargs: trace_stages.append(stage),
    )
    monkeypatch.setattr(
        structured_output,
        "emit_context_usage_trace",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        structured_output,
        "emit_context_items_shadow",
        lambda *_, **__: [_item()],
    )
    monkeypatch.setattr(
        structured_output,
        "emit_context_packing_shadow",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        structured_output,
        "_structured_context_apply_config",
        lambda: {
            "enabled": True,
            "mode": "observe_only",
            "active_nodes": (),
            "allow_structured_output": False,
            "diagnostics": [],
        },
    )
    monkeypatch.setattr(structured_output, "_provider", lambda _node: "deepseek")
    monkeypatch.setattr(
        structured_output,
        "_model",
        lambda _node: "deepseek-v4-pro",
    )
    monkeypatch.setattr(structured_output, "get_node_llm", lambda _node: FakeLLM())
    monkeypatch.setattr(
        structured_output,
        "invoke_with_provider_transport_retry",
        fake_transport_retry,
    )

    parsed, _raw_output, _metrics = await _invoke_one_mode(
        node_name="study_plan_agent",
        llm_node="study_plan",
        schema=_TinySchema,
        messages=list(original_messages),
        mode="native_json_schema_pydantic",
        state={"request_id": "r1", "thread_id": "t1"},
    )

    assert parsed.answer == "ok"
    serialized_messages = repr(captured_runnable_messages[0])
    assert "<INJECTED_CONTEXT>" not in serialized_messages
    assert "injected_context" not in serialized_messages
    assert "rendered_context" not in serialized_messages
    assert captured_runnable_messages[0] != original_messages
    assert original_messages == [{"role": "user", "content": "question"}]
    assert scorer_calls == []
    assert "context_importance_scored" not in trace_stages
    assert "llm_input_manifest.built" in trace_stages


@pytest.mark.parametrize(
    ("node_name", "llm_node"),
    [
        ("supervisor", "supervisor"),
        ("search_query_rewriter", "search_query_rewriter"),
        ("evidence_sufficiency_judge", "evidence_sufficiency_judge"),
        ("resource_bundle_output", "resource_bundle_output"),
    ],
)
def test_structured_output_active_rollout_skips_non_resource_agent_nodes(
    monkeypatch,
    node_name,
    llm_node,
):
    from src.llm import structured_output

    messages = [{"role": "user", "content": "question"}]
    shadow_calls: list[str] = []

    monkeypatch.setattr(
        structured_output,
        "_structured_context_apply_config",
        lambda: {
            "enabled": True,
            "mode": "active",
            "active_nodes": ("mindmap_agent",),
            "allow_structured_output": True,
            "diagnostics": [],
        },
    )
    monkeypatch.setattr(
        structured_output,
        "prepare_messages_with_context_policy",
        lambda *_, **__: pytest.fail("non-active structured node must not apply CE"),
    )
    monkeypatch.setattr(
        structured_output,
        "emit_context_items_shadow",
        lambda *_, **__: shadow_calls.append("items") or [_item()],
    )
    monkeypatch.setattr(
        structured_output,
        "emit_context_packing_shadow",
        lambda *_, **__: shadow_calls.append("packing"),
    )
    monkeypatch.setattr(
        structured_output,
        "emit_a3_trace",
        lambda *_, **__: None,
    )

    result = structured_output._prepare_structured_messages_with_context(
        node_name=node_name,
        llm_node=llm_node,
        messages=messages,
        state={"request_id": "r1", "thread_id": "t1"},
    )

    assert result.messages == messages
    assert result.debug["structured_context_apply_status"] == "observed"
    assert (
        result.debug["structured_context_apply_skip_reason"]
        == "node_not_in_active_structured_context_rollout"
    )
    assert result.debug["provider_bound_messages_mutated"] is False
    assert result.debug["context_apply_applied"] is False
    assert shadow_calls == ["items", "packing"]


def test_structured_output_active_rollout_applies_for_reviewer_node(monkeypatch):
    from src.context_engineering.packing import ContextPreparedMessages
    from src.llm import structured_output

    messages = [
        {"role": "system", "content": "contract"},
        {"role": "user", "content": "review"},
    ]
    final_messages = [
        messages[0],
        {"role": "system", "content": "<INJECTED_CONTEXT>rules</INJECTED_CONTEXT>"},
        messages[1],
    ]

    def fake_prepare(*_args, **_kwargs):
        return ContextPreparedMessages(
            messages_for_llm=final_messages,
            original_messages=messages,
            trace_call_id="trace-1",
            next_trace_seq=1,
            context_apply_applied=True,
            apply_result=SimpleNamespace(
                injected_items_count=1,
                injected_context_tokens=4,
            ),
        )

    monkeypatch.setattr(
        structured_output,
        "_structured_context_apply_config",
        lambda: {
            "enabled": True,
            "mode": "active",
            "active_nodes": ("study_plan_reviewer_academic",),
            "allow_structured_output": True,
            "diagnostics": [],
        },
    )
    monkeypatch.setattr(
        structured_output,
        "prepare_messages_with_context_policy",
        fake_prepare,
    )
    monkeypatch.setattr(
        structured_output,
        "emit_context_items_shadow",
        lambda *_, **__: pytest.fail("active node must not use observe shadow"),
    )
    monkeypatch.setattr(
        structured_output,
        "emit_context_packing_shadow",
        lambda *_, **__: pytest.fail("active node must not use observe shadow"),
    )
    monkeypatch.setattr(structured_output, "emit_a3_trace", lambda *_, **__: None)

    result = structured_output._prepare_structured_messages_with_context(
        node_name="study_plan_reviewer_academic",
        llm_node="study_plan",
        messages=messages,
        state={"request_id": "r1", "thread_id": "t1"},
    )

    assert result.messages == final_messages
    assert result.debug["structured_context_apply_status"] == "applied"
    assert result.debug["provider_bound_messages_mutated"] is True
    assert result.debug["context_apply_applied"] is True
