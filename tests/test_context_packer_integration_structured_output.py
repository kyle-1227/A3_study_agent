"""Structured-output integration tests for ContextPacker shadow mode."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from src.context_engineering.schema import ContextItem


class _TinySchema(BaseModel):
    answer: str


def _item() -> ContextItem:
    return ContextItem(
        id="item-1",
        source_type="message",
        title="query",
        content="content",
        token_estimate=10,
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
async def test_structured_output_runs_packing_after_schema_injection(monkeypatch):
    from src.llm import structured_output
    from src.llm.structured_output import _invoke_one_mode

    original_messages = [{"role": "user", "content": "question"}]
    captured_item_messages: list[list] = []
    captured_packing_items: list[list[ContextItem]] = []
    context_items = [_item()]

    def fake_items_shadow(*_, **kwargs):
        captured_item_messages.append(kwargs["messages"])
        return context_items

    def fake_packing_shadow(*_, **kwargs):
        captured_packing_items.append(kwargs["items"])

    monkeypatch.setattr(
        structured_output,
        "emit_context_items_shadow",
        fake_items_shadow,
    )
    monkeypatch.setattr(
        structured_output,
        "emit_context_packing_shadow",
        fake_packing_shadow,
    )

    with pytest.raises(Exception, match="constrained_decoding"):
        await _invoke_one_mode(
            node_name="study_plan_agent",
            llm_node="study_plan",
            schema=_TinySchema,
            messages=list(original_messages),
            mode="constrained_decoding",
            state={"request_id": "r1", "thread_id": "t1"},
        )

    assert captured_packing_items == [context_items]
    assert captured_item_messages
    assert captured_item_messages[0] != original_messages
    assert original_messages == [{"role": "user", "content": "question"}]


@pytest.mark.anyio
async def test_structured_output_runnable_receives_schema_messages_without_context_pack(
    monkeypatch,
):
    from src.llm import structured_output
    from src.llm.structured_output import _invoke_one_mode

    original_messages = [{"role": "user", "content": "question"}]
    captured_runnable_messages: list[list] = []
    context_items = [_item()]

    class FakeRunnable:
        async def ainvoke(self, messages):
            captured_runnable_messages.append(messages)
            return SimpleNamespace(content='{"answer": "ok"}')

    class FakeLLM:
        def __init__(self) -> None:
            self.runnable = FakeRunnable()

        def bind(self, **_kwargs):
            return self.runnable

    async def fake_transport_retry(operation, **_kwargs):
        return await operation(), 0

    monkeypatch.setattr(
        structured_output,
        "emit_context_usage_trace",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        structured_output,
        "emit_context_items_shadow",
        lambda *_, **__: context_items,
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

    parsed, raw_output, metrics = await _invoke_one_mode(
        node_name="study_plan_agent",
        llm_node="study_plan",
        schema=_TinySchema,
        messages=list(original_messages),
        mode="native_json_schema_pydantic",
        state={"request_id": "r1", "thread_id": "t1"},
    )

    assert parsed.answer == "ok"
    assert raw_output == '{"answer": "ok"}'
    assert metrics.extra_debug["structured_context_apply_status"] == "observed"
    assert metrics.extra_debug["structured_context_apply_skip_reason"] == "observe_only"
    assert len(captured_runnable_messages) == 1
    serialized_messages = repr(captured_runnable_messages[0])
    assert "<CONTEXT_PACK>" not in serialized_messages
    assert "rendered_context" not in serialized_messages
    assert "answer" in serialized_messages
    assert captured_runnable_messages[0] != original_messages
    assert original_messages == [{"role": "user", "content": "question"}]
