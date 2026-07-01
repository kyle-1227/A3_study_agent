"""Trace tests for Phase 3A ContextPacker."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.context_engineering.packing.packer import pack_context_items
from src.context_engineering.packing.policies import PackingPolicy
from src.context_engineering.packing.schema import ContextPackingError
from src.context_engineering.packing.trace import (
    build_context_packed_event,
    build_context_packing_error_event,
    build_context_packing_plan_event,
)
from src.context_engineering.schema import ContextItem
from src.observability.a3_trace import emit_a3_trace


def _item(item_id: str = "item-1") -> ContextItem:
    return ContextItem(
        id=item_id,
        source_type="message",
        title="title api_key=sk-secret-value",
        content="secret prompt content",
        token_estimate=10,
        estimated=True,
        tokenizer_mode="estimated_mixed",
        priority=80,
        relevance_score=0.8,
        recency_score=None,
        confidence=None,
        scope="turn",
        lifetime="turn",
        compressible=True,
        can_drop=True,
        disclosure_level="snippet",
        metadata={},
    )


def _policy() -> PackingPolicy:
    return PackingPolicy(
        enabled=True,
        shadow_mode=True,
        apply_to_llm=False,
        strategy="priority_budget",
        max_context_block_tokens=1000,
        trace_selected_items=10,
        trace_dropped_items=10,
        enabled_nodes=(),
        enabled_sources=("message",),
    )


def _snapshot(values: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(next=(), tasks=[], values=values or {})


def _payloads(collected: list[str]) -> list[dict]:
    return [json.loads(item.removeprefix("data: ").strip()) for item in collected]


def test_context_packing_plan_event_is_safe():
    event = build_context_packing_plan_event(
        node_name="node",
        llm_node="llm",
        items=[_item()],
        max_context_block_tokens=1000,
        strategy="priority_budget",
    )

    serialized = repr(event).lower()
    assert event["candidate_count"] == 1
    assert "secret prompt content" not in serialized
    assert "content" not in serialized


def test_context_packed_event_uses_safe_preview_only():
    packed = pack_context_items(
        node_name="node",
        llm_node="llm",
        items=[_item()],
        max_context_block_tokens=1000,
    )

    event = build_context_packed_event(
        packed,
        trace_selected_items=10,
        trace_dropped_items=10,
    )

    serialized = repr(event).lower()
    assert event["selected_count"] == 1
    assert "secret prompt content" not in serialized
    assert "rendered_context" not in serialized
    assert "metadata" not in serialized
    assert "api_key" not in serialized
    assert set(event["selected_items_preview"][0]) == {
        "id",
        "source_type",
        "title",
        "token_estimate",
        "priority",
        "can_drop",
        "reason",
    }


def test_context_packing_error_event_is_redacted():
    event = build_context_packing_error_event(
        ContextPackingError(
            reason="context_packing_error",
            warning="failed api_key=sk-secret-value cookie=session",
            node_name="node",
            llm_node="llm",
            selected_tokens=1,
            budget_tokens=2,
            original_exception_type="RuntimeError",
        )
    )

    serialized = repr(event).lower()
    assert "api_key" not in serialized
    assert "cookie" not in serialized
    assert "sk-secret-value" not in serialized


@pytest.mark.anyio
async def test_context_packing_events_are_forwarded_as_safe_sse():
    from app import generate_sse

    async def events():
        emit_a3_trace(
            logging.getLogger("test_context_packer_trace"),
            "context_packed",
            {
                "node_name": "generate_answer",
                "llm_node": "academic",
                "strategy": "priority_budget",
                "selected_count": 1,
                "dropped_count": 0,
                "selected_tokens": 10,
                "dropped_tokens": 0,
                "required_tokens": 0,
                "optional_tokens": 10,
                "remaining_tokens": 990,
                "overflow": False,
                "selected_items_preview": [
                    {
                        "id": "item-1",
                        "source_type": "message",
                        "title": (
                            "query api_key=sk-secret-value "
                            "cookie=session authorization: bearer token-value "
                            "db_uri=postgresql://user:pass@localhost/db"
                        ),
                        "token_estimate": 10,
                        "priority": 80,
                        "can_drop": True,
                        "reason": "fits_budget",
                        "content": "must not forward",
                    }
                ],
                "rendered_context": "must not forward",
                "warnings": [],
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
    async for item in generate_sse("q", graph, thread_id="thread-1"):
        collected.append(item)

    payloads = _payloads(collected)
    packed_events = [
        payload for payload in payloads if payload.get("type") == "context_packed"
    ]
    assert len(packed_events) == 1
    serialized = repr(packed_events).lower()
    assert "must not forward" not in serialized
    assert "rendered_context" not in serialized
    assert "content" not in serialized
    assert "api_key" not in serialized
    assert "cookie" not in serialized
    assert "authorization" not in serialized
    assert "db_uri" not in serialized
    assert "sk-secret-value" not in serialized
    assert "postgresql://" not in serialized
