"""Phase 0 regression tests for existing SSE and context-usage behavior."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


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


def _node_start(node_name: str) -> dict:
    return {
        "event": "on_chain_start",
        "name": node_name,
        "metadata": {"langgraph_node": node_name},
        "data": {"input": {}},
    }


def _node_end(node_name: str, output: dict | None = None) -> dict:
    return {
        "event": "on_chain_end",
        "name": node_name,
        "metadata": {"langgraph_node": node_name},
        "data": {"output": output or {}},
    }


def _usage(node_name: str) -> dict:
    return {
        "event": "on_chat_model_end",
        "name": "ChatOpenAI",
        "metadata": {"langgraph_node": node_name},
        "data": {
            "output": SimpleNamespace(
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                }
            )
        },
    }


@pytest.mark.anyio
async def test_stream_keeps_node_usage_resource_final_events():
    from app import generate_sse

    final_state = {
        "requested_resource_type": "quiz",
        "messages": [SimpleNamespace(content="quiz ready")],
        "exercise_items": [{"question": "Q1"}],
        "exercise_artifact": {"title": "Python quiz"},
    }
    graph = MagicMock()
    graph.astream_events = MagicMock(
        return_value=AsyncIteratorMock(
            [
                _node_start("generate_answer"),
                _usage("generate_answer"),
                _node_end("generate_answer"),
            ]
        )
    )
    graph.aget_state = AsyncMock(return_value=_snapshot(final_state))
    graph.aupdate_state = AsyncMock()

    collected = []
    async for event in generate_sse("q", graph, thread_id="thread-1"):
        collected.append(event)

    payloads = _payloads(collected)
    assert any(payload.get("type") == "node_event" for payload in payloads)
    assert any(payload.get("type") == "usage" for payload in payloads)
    resource_events = [
        payload for payload in payloads if payload.get("type") == "resource_final"
    ]
    assert resource_events and resource_events[0]["resource_type"] == "quiz"
    assert payloads[-1] == {"type": "done"}


@pytest.mark.anyio
async def test_resume_keeps_basic_stream_path():
    from app import generate_resume_sse

    final_snapshot = _snapshot({"schema_version": "run_control_v1"})
    graph = MagicMock()
    graph.astream_events = MagicMock(
        return_value=AsyncIteratorMock([_node_start("study_plan_output")])
    )
    graph.aget_state = AsyncMock(
        side_effect=[
            _snapshot({"schema_version": "run_control_v1"}),
            final_snapshot,
            final_snapshot,
        ]
    )
    graph.aupdate_state = AsyncMock()

    collected = []
    async for event in generate_resume_sse("approved", None, graph, "thread-1"):
        collected.append(event)

    payloads = _payloads(collected)
    assert payloads[0]["run_status"] == "continuing"
    assert any(payload.get("type") == "node_event" for payload in payloads)
    assert payloads[-1] == {"type": "done"}


def test_deepseek_model_window_is_unknown_after_phase0_config_cleanup():
    from src.config import clear_cache
    from src.observability.context_usage import build_context_usage_payload

    clear_cache()
    stage, payload = build_context_usage_payload(
        node_name="study_plan_agent",
        llm_node="study_plan",
        provider="deepseek_official",
        model="deepseek-v4-pro",
        messages=[],
    )

    assert stage == "context_usage_error"
    assert payload is not None
    assert payload["reason"] == "model_window_unknown"
