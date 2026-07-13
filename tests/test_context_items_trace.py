"""Trace and SSE tests for Phase 2 context item events."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.stream_draft_helpers import draft_payloads
from src.context_engineering.itemizer import make_context_item
from src.context_engineering.schema import ContextProviderError
from src.context_engineering.trace import (
    build_context_items_collected_event,
    build_context_provider_error_event,
    emit_context_items_collected,
)
from src.observability.a3_trace import (
    emit_a3_trace,
    reset_trace_event_sink,
    set_trace_event_sink,
)


def _item():
    return make_context_item(
        source_type="message",
        title="query",
        content="secret prompt content",
        priority=100,
        scope="turn",
        lifetime="turn",
        compressible=False,
        can_drop=False,
        disclosure_level="full",
        metadata={"safe": "metadata"},
        max_content_chars=4000,
    )


def _payloads(collected) -> list[dict]:
    return draft_payloads(collected)


def _snapshot(values: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(next=(), tasks=[], values=values or {})


def test_context_items_collected_event_contains_only_safe_preview_fields():
    event = build_context_items_collected_event(
        node_name="node",
        llm_node="llm",
        provider_count=1,
        items=[_item()],
        trace_top_items=10,
    )

    assert event["item_count"] == 1
    assert event["source_counts"] == {"message": 1}
    assert event["top_items"] == [
        {
            "id": _item().id,
            "source_type": "message",
            "title": "query",
            "token_estimate": _item().token_estimate,
            "priority": 100,
            "scope": "turn",
            "lifetime": "turn",
            "disclosure_level": "full",
        }
    ]
    serialized = repr(event).lower()
    assert "secret prompt content" not in serialized
    assert "metadata" not in serialized
    assert "content" not in event["top_items"][0]


def test_context_provider_error_event_redacts_secret_like_values():
    error = ContextProviderError(
        provider="memory_provider",
        source_type="memory",
        stage="collect",
        message="failed with api_key=sk-secret and cookie=session",
        original_exception_type="RuntimeError",
    )

    event = build_context_provider_error_event(
        node_name="node",
        llm_node="llm",
        error=error,
    )

    serialized = repr(event).lower()
    assert event["error_type"] == "RuntimeError"
    assert "sk-secret" not in serialized
    assert "api_key" not in serialized
    assert "cookie" not in serialized


def test_emit_context_items_collected_writes_safe_trace_sink():
    sink: list[dict] = []
    token = set_trace_event_sink(sink)
    try:
        emit_context_items_collected(
            logging.getLogger("test_context_items_trace"),
            node_name="node",
            llm_node="llm",
            provider_count=1,
            items=[_item()],
            trace_top_items=10,
            state={"request_id": "r1", "thread_id": "t1"},
        )
    finally:
        reset_trace_event_sink(token)

    assert sink[0]["stage"] == "context_items_collected"
    assert "secret prompt content" not in repr(sink[0]).lower()


@pytest.mark.anyio
async def test_context_item_events_are_forwarded_as_safe_sse():
    from app import generate_stream_drafts

    async def events():
        emit_a3_trace(
            logging.getLogger("test_context_items_trace"),
            "context_items_collected",
            {
                "node_name": "generate_answer",
                "llm_node": "academic",
                "provider_count": 1,
                "item_count": 1,
                "source_counts": {"message": 1},
                "total_estimated_tokens": 5,
                "top_items": [
                    {
                        "id": "message:1",
                        "source_type": "message",
                        "title": "query",
                        "token_estimate": 5,
                        "priority": 100,
                        "scope": "turn",
                        "lifetime": "turn",
                        "disclosure_level": "full",
                        "content": "must not forward",
                    }
                ],
            },
            state={"request_id": "r1", "thread_id": "thread-1"},
        )
        emit_a3_trace(
            logging.getLogger("test_context_items_trace"),
            "context_provider_error",
            {
                "node_name": "generate_answer",
                "llm_node": "academic",
                "provider": "memory_provider",
                "source_type": "memory",
                "provider_stage": "collect",
                "error_type": "RuntimeError",
                "error_reason": "provider unavailable",
                "messages": [{"content": "must not forward"}],
            },
            state={"request_id": "r1", "thread_id": "thread-1"},
        )
        yield {
            "event": "on_chain_start",
            "name": "generate_answer",
            "metadata": {"langgraph_node": "generate_answer"},
            "data": {"input": {}},
        }

    graph = MagicMock()
    graph.astream_events = MagicMock(return_value=events())
    graph.aget_state = AsyncMock(
        return_value=_snapshot({"schema_version": "run_control_v1"})
    )
    graph.aupdate_state = AsyncMock()

    collected = []
    async for item in generate_stream_drafts("q", graph, thread_id="thread-1"):
        collected.append(item)

    payloads = _payloads(collected)
    collected_events = [
        payload
        for payload in payloads
        if payload.get("type") == "context_items_collected"
    ]
    error_events = [
        payload
        for payload in payloads
        if payload.get("type") == "context_provider_error"
    ]
    assert len(collected_events) == 1
    assert len(error_events) == 1
    serialized = repr(collected_events + error_events).lower()
    assert "must not forward" not in serialized
    assert "messages" not in serialized
    assert "content" not in serialized
    assert not [payload for payload in payloads if payload.get("type") == "stream_done"]
