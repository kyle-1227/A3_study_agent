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
async def test_structured_output_never_calls_apply_even_when_config_would_allow(
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
