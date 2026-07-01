"""SSE tests for Context Engineering usage and error events."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.observability.a3_trace import emit_a3_trace


class AsyncIteratorMock:
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


def _payloads(collected: list[str]) -> list[dict]:
    return [json.loads(item.removeprefix("data: ").strip()) for item in collected]


def _snapshot(values: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(next=(), tasks=[], values=values or {})


@pytest.mark.anyio
async def test_context_usage_sse_uses_canonical_fields_and_updates_state():
    from app import generate_sse

    async def events():
        emit_a3_trace(
            logging.getLogger("test_context_usage_sse"),
            "context_usage",
            {
                "node_name": "study_plan_agent",
                "llm_node": "study_plan",
                "provider": "deepseek_official",
                "model": "deepseek-v4-pro",
                "input_estimated_tokens": 100,
                "reserved_output_tokens": 20,
                "used_tokens": 120,
                "max_context_tokens": 1000,
                "available_tokens": 880,
                "used_ratio": 0.12,
                "warning_level": "ok",
                "estimated": True,
                "tokenizer_mode": "estimated_mixed",
                "message_count": 2,
                "schema_size_chars": None,
                "breakdown": {
                    "input_estimated_tokens": 100,
                    "reserved_output_tokens": 20,
                },
                "prompt_tokens": 100,
                "usage_ratio": 0.12,
            },
            state={"request_id": "r1", "thread_id": "thread-1"},
        )
        yield {
            "event": "on_chain_start",
            "name": "study_plan_agent",
            "metadata": {"langgraph_node": "study_plan_agent"},
            "data": {"input": {}},
        }

    graph = MagicMock()
    graph.astream_events = MagicMock(return_value=events())
    graph.aget_state = AsyncMock(
        return_value=_snapshot({"schema_version": "run_control_v1"})
    )
    graph.aupdate_state = AsyncMock()

    collected = []
    async for item in generate_sse("q", graph, thread_id="thread-1"):
        collected.append(item)

    usage_events = [
        payload
        for payload in _payloads(collected)
        if payload.get("type") == "context_usage"
    ]
    assert len(usage_events) == 1
    event = usage_events[0]
    assert event["input_estimated_tokens"] == 100
    assert event["reserved_output_tokens"] == 20
    assert event["available_tokens"] == 880
    assert event["used_ratio"] == 0.12
    assert event["warning_level"] == "ok"
    assert event["tokenizer_mode"] == "estimated_mixed"
    assert "prompt_tokens" not in event
    assert "output_reserved_tokens" not in event
    assert "usage_ratio" not in event
    assert "remaining_tokens" not in event
    assert "level" not in event

    state_updates = [call.args[1] for call in graph.aupdate_state.await_args_list]
    context_updates = [update for update in state_updates if "context_usage" in update]
    assert context_updates
    assert context_updates[-1]["context_usage"]["input_estimated_tokens"] == 100


@pytest.mark.anyio
async def test_context_usage_error_sse_does_not_break_stream():
    from app import generate_sse

    async def events():
        emit_a3_trace(
            logging.getLogger("test_context_usage_error_sse"),
            "context_usage_error",
            {
                "node_name": "study_plan_agent",
                "llm_node": "study_plan",
                "provider": "deepseek_official",
                "model": "unknown-model",
                "reason": "model_window_unknown",
                "warning": "model context window is unknown",
            },
            state={"request_id": "r1", "thread_id": "thread-1"},
        )
        yield {
            "event": "on_chain_start",
            "name": "study_plan_agent",
            "metadata": {"langgraph_node": "study_plan_agent"},
            "data": {"input": {}},
        }

    graph = MagicMock()
    graph.astream_events = MagicMock(return_value=events())
    graph.aget_state = AsyncMock(
        return_value=_snapshot({"schema_version": "run_control_v1"})
    )
    graph.aupdate_state = AsyncMock()

    collected = []
    async for item in generate_sse("q", graph, thread_id="thread-1"):
        collected.append(item)

    payloads = _payloads(collected)
    error_events = [
        payload for payload in payloads if payload.get("type") == "context_usage_error"
    ]
    assert error_events == [
        {
            "type": "context_usage_error",
            "node": "study_plan_agent",
            "llm_node": "study_plan",
            "provider": "deepseek_official",
            "model": "unknown-model",
            "reason": "model_window_unknown",
            "warning": "model context window is unknown",
        }
    ]
    assert any(payload.get("type") == "node_event" for payload in payloads)
    assert payloads[-1] == {"type": "done"}
