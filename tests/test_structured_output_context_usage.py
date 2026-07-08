"""Context usage tests for structured-output LLM calls."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from src.context_engineering.tokenizer import count_messages_tokens
from src.observability.a3_trace import reset_trace_event_sink, set_trace_event_sink


class _TinySchema(BaseModel):
    answer: str


@pytest.mark.anyio
async def test_structured_output_counts_after_schema_contract_injection_without_leaking_schema(
    monkeypatch,
):
    from src.llm import structured_output
    from src.llm.structured_output import _invoke_one_mode

    original_messages = [{"role": "user", "content": "question"}]
    original_count = count_messages_tokens(original_messages).value
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
    sink: list[dict] = []
    token = set_trace_event_sink(sink)
    try:
        with pytest.raises(Exception, match="constrained_decoding"):
            await _invoke_one_mode(
                node_name="study_plan_agent",
                llm_node="study_plan",
                schema=_TinySchema,
                messages=list(original_messages),
                mode="constrained_decoding",
                state={"request_id": "r1", "thread_id": "t1"},
            )
    finally:
        reset_trace_event_sink(token)

    usage_events = [event for event in sink if event.get("stage") == "context_usage"]
    assert len(usage_events) == 1
    event = usage_events[0]
    assert event["schema_size_chars"] > 0
    assert event["input_estimated_tokens"] > original_count

    serialized = repr(event).lower()
    assert "messages" not in serialized
    assert "raw_output" not in serialized
    assert 'schema": {' not in serialized
    assert "answer" not in serialized
