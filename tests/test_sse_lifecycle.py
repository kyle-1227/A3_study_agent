"""Unit tests for SSE node lifecycle events in generate_sse.

Tests cover: node start/end events, token streaming coexistence,
sub-chain filtering, internal node filtering, and event ordering.
All tests mock astream_events — no real graph execution required.

NOTE: generate_sse now emits a thread_id event first (REQ-08 HIL),
and calls graph.aget_state() after streaming to detect interrupts.
All mock graphs must provide aget_state as AsyncMock.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helper: reusable async iterator from a list
# ---------------------------------------------------------------------------

class AsyncIteratorMock:
    """Create an async iterator from a list of items."""

    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


def _make_mock_graph(events=None):
    """Create a mock graph with astream_events and aget_state (no interrupt)."""
    mock_graph = MagicMock()
    mock_graph.astream_events = MagicMock(
        return_value=AsyncIteratorMock(events or []),
    )
    mock_graph.aget_state = AsyncMock(
        return_value=SimpleNamespace(next=(), tasks=[]),
    )
    return mock_graph


def _parse_payloads(collected):
    """Parse SSE lines into JSON payloads, skipping thread_id and done events."""
    all_payloads = [json.loads(s.removeprefix("data: ").strip()) for s in collected]
    # First payload is always thread_id; filter it and the trailing done event
    assert all_payloads[0]["type"] == "thread_id"
    return [p for p in all_payloads[1:] if p.get("type") != "done"]


# ---------------------------------------------------------------------------
# Helpers: build mock events matching astream_events v2 format
# ---------------------------------------------------------------------------

def _node_start(node_name: str) -> dict:
    """Build an on_chain_start event for a graph node."""
    return {
        "event": "on_chain_start",
        "name": node_name,
        "metadata": {"langgraph_node": node_name},
        "data": {"input": {}},
    }


def _node_end(node_name: str) -> dict:
    """Build an on_chain_end event for a graph node."""
    return {
        "event": "on_chain_end",
        "name": node_name,
        "metadata": {"langgraph_node": node_name},
        "data": {"output": {}},
    }


def _sub_chain_start(chain_name: str, parent_node: str) -> dict:
    """Build an on_chain_start for an internal sub-chain (not a graph node)."""
    return {
        "event": "on_chain_start",
        "name": chain_name,
        "metadata": {"langgraph_node": parent_node},
        "data": {"input": {}},
    }


def _token_event(node_name: str, content: str) -> dict:
    """Build an on_chat_model_stream event with a token chunk."""
    chunk = SimpleNamespace(content=content)
    return {
        "event": "on_chat_model_stream",
        "name": "ChatOpenAI",
        "metadata": {"langgraph_node": node_name},
        "data": {"chunk": chunk},
    }


# ---------------------------------------------------------------------------
# TestSSENodeLifecycle
# ---------------------------------------------------------------------------

class TestSSENodeLifecycle:
    """Tests that generate_sse emits node lifecycle events."""

    @pytest.mark.anyio
    async def test_yields_node_start_event(self):
        """on_chain_start for a graph node → {"type": "node_event", "status": "start"}."""
        from app import generate_sse

        mock_graph = _make_mock_graph([_node_start("supervisor")])

        collected = []
        async for sse in generate_sse("hello", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert len(payloads) == 1
        assert payloads[0] == {"type": "node_event", "status": "start", "node": "supervisor"}

    @pytest.mark.anyio
    async def test_yields_node_end_event(self):
        """on_chain_end for a graph node → {"type": "node_event", "status": "end"}."""
        from app import generate_sse

        mock_graph = _make_mock_graph([_node_end("rag_retrieve")])

        collected = []
        async for sse in generate_sse("hello", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert len(payloads) == 1
        assert payloads[0]["type"] == "node_event"
        assert payloads[0]["status"] == "end"
        assert payloads[0]["node"] == "rag_retrieve"
        assert "duration_ms" in payloads[0]
        assert payloads[0]["error"] is None

    @pytest.mark.anyio
    async def test_ignores_sub_chain_events(self):
        """Sub-chain events (name != metadata.langgraph_node) must be dropped."""
        from app import generate_sse

        events = [
            _sub_chain_start("RunnableSequence", "supervisor"),
            _sub_chain_start("ChatPromptTemplate", "generate_answer"),
        ]
        mock_graph = _make_mock_graph(events)

        collected = []
        async for sse in generate_sse("hello", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert payloads == []

    @pytest.mark.anyio
    async def test_ignores_langgraph_internal_nodes(self):
        """LangGraph internal nodes like __start__ must be filtered out."""
        from app import generate_sse

        events = [
            {
                "event": "on_chain_start",
                "name": "__start__",
                "metadata": {"langgraph_node": "__start__"},
                "data": {},
            },
        ]
        mock_graph = _make_mock_graph(events)

        collected = []
        async for sse in generate_sse("hello", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert payloads == []


# ---------------------------------------------------------------------------
# TestSSETokenStreamingPreserved
# ---------------------------------------------------------------------------

class TestSSETokenStreamingPreserved:
    """Ensure the original token streaming logic is not broken."""

    @pytest.mark.anyio
    async def test_token_from_allowed_node(self):
        """Tokens from ALLOWED_NODES should still be emitted."""
        from app import generate_sse

        mock_graph = _make_mock_graph([_token_event("generate_answer", "Hello")])

        collected = []
        async for sse in generate_sse("hi", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert len(payloads) == 1
        assert payloads[0] == {"type": "token", "content": "Hello"}

    @pytest.mark.anyio
    async def test_token_from_disallowed_node_dropped(self):
        """Tokens from nodes NOT in ALLOWED_NODES should be dropped."""
        from app import generate_sse

        mock_graph = _make_mock_graph([_token_event("supervisor", "thinking...")])

        collected = []
        async for sse in generate_sse("hi", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert payloads == []

    @pytest.mark.anyio
    async def test_empty_token_dropped(self):
        """Empty token content should not produce an SSE payload."""
        from app import generate_sse

        mock_graph = _make_mock_graph([_token_event("generate_answer", "")])

        collected = []
        async for sse in generate_sse("hi", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert payloads == []


# ---------------------------------------------------------------------------
# TestSSEMixedEventOrdering
# ---------------------------------------------------------------------------

class TestSSEMixedEventOrdering:
    """Tests correct ordering when lifecycle and token events are interleaved."""

    @pytest.mark.anyio
    async def test_full_academic_flow(self):
        """Simulate supervisor → rag_retrieve → generate_answer with tokens."""
        from app import generate_sse

        events = [
            _node_start("supervisor"),
            _sub_chain_start("RunnableSequence", "supervisor"),
            _node_end("supervisor"),
            _node_start("rag_retrieve"),
            _node_end("rag_retrieve"),
            _node_start("generate_answer"),
            _token_event("generate_answer", "The"),
            _token_event("generate_answer", " answer"),
            _node_end("generate_answer"),
        ]
        mock_graph = _make_mock_graph(events)

        collected = []
        async for sse in generate_sse("What is calculus?", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        # Sub-chain event dropped → 8 graph events
        assert len(payloads) == 8

        assert payloads[0] == {"type": "node_event", "status": "start", "node": "supervisor"}
        assert payloads[1]["type"] == "node_event"
        assert payloads[1]["status"] == "end"
        assert payloads[1]["node"] == "supervisor"
        assert payloads[2] == {"type": "node_event", "status": "start", "node": "rag_retrieve"}
        assert payloads[3]["type"] == "node_event"
        assert payloads[3]["status"] == "end"
        assert payloads[3]["node"] == "rag_retrieve"
        assert payloads[4] == {"type": "node_event", "status": "start", "node": "generate_answer"}
        assert payloads[5] == {"type": "token", "content": "The"}
        assert payloads[6] == {"type": "token", "content": " answer"}
        assert payloads[7]["type"] == "node_event"
        assert payloads[7]["status"] == "end"
        assert payloads[7]["node"] == "generate_answer"

    @pytest.mark.anyio
    async def test_emotional_flow(self):
        """Simulate supervisor → emotional_response with tokens."""
        from app import generate_sse

        events = [
            _node_start("supervisor"),
            _node_end("supervisor"),
            _node_start("emotional_response"),
            _token_event("emotional_response", "I understand"),
            _node_end("emotional_response"),
        ]
        mock_graph = _make_mock_graph(events)

        collected = []
        async for sse in generate_sse("I'm stressed", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert len(payloads) == 5
        assert payloads[0]["type"] == "node_event"
        assert payloads[3]["type"] == "token"
        assert payloads[3]["content"] == "I understand"


# ---------------------------------------------------------------------------
# TestSSEAllGraphNodes
# ---------------------------------------------------------------------------

class TestSSEAllGraphNodes:
    """Ensure every known graph node can produce lifecycle events."""

    ALL_NODES = [
        "supervisor",
        "academic_router",
        "search_query_rewriter",
        "rag_retrieve",
        "web_search",
        "generate_answer",
        "evaluate_hallucination",
        "rewrite_query",
        "search_policy",
        "gather_intel",
        "drafter",
        "reviewer_academic",
        "reviewer_emotional",
        "consensus_check",
        "adv_rewrite",
        "plan_output",
        "feedback_router",
        "plan_tweak",
        "mindmap_planner",
        "mindmap_agent",
        "mindmap_reviewer",
        "mindmap_rewrite",
        "mindmap_output",
        "exercise_planner",
        "exercise_agent",
        "exercise_reviewer",
        "exercise_rewrite",
        "exercise_output",
        "emotional_response",
        "handle_unknown",
    ]

    @pytest.mark.anyio
    @pytest.mark.parametrize("node_name", ALL_NODES)
    async def test_each_node_emits_start(self, node_name):
        """Every graph node should produce a start event."""
        from app import generate_sse

        mock_graph = _make_mock_graph([_node_start(node_name)])

        collected = []
        async for sse in generate_sse("q", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert len(payloads) == 1
        assert payloads[0]["node"] == node_name
        assert payloads[0]["status"] == "start"

    @pytest.mark.anyio
    @pytest.mark.parametrize("node_name", ALL_NODES)
    async def test_each_node_emits_end(self, node_name):
        """Every graph node should produce an end event with duration_ms and error."""
        from app import generate_sse

        mock_graph = _make_mock_graph([_node_start(node_name), _node_end(node_name)])

        collected = []
        async for sse in generate_sse("q", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert len(payloads) == 2
        assert payloads[1]["node"] == node_name
        assert payloads[1]["status"] == "end"
        assert isinstance(payloads[1]["duration_ms"], int)
        assert payloads[1]["error"] is None


# ---------------------------------------------------------------------------
# TestSSENodeTiming
# ---------------------------------------------------------------------------

class TestSSENodeTiming:
    """Tests that node end events include duration_ms."""

    @pytest.mark.anyio
    async def test_end_has_duration_ms(self):
        """A start+end pair should produce a non-negative duration_ms."""
        from app import generate_sse

        mock_graph = _make_mock_graph([_node_start("supervisor"), _node_end("supervisor")])

        collected = []
        async for sse in generate_sse("q", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert payloads[1]["duration_ms"] is not None
        assert payloads[1]["duration_ms"] >= 0

    @pytest.mark.anyio
    async def test_end_without_start_has_null_duration(self):
        """An end event without a preceding start should have duration_ms=None."""
        from app import generate_sse

        mock_graph = _make_mock_graph([_node_end("supervisor")])

        collected = []
        async for sse in generate_sse("q", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert payloads[0]["duration_ms"] is None


# ---------------------------------------------------------------------------
# TestSSEErrorCapture
# ---------------------------------------------------------------------------

class TestSSEErrorCapture:
    """Tests that node end events capture errors."""

    @pytest.mark.anyio
    async def test_error_null_on_success(self):
        """Normal end → error is null."""
        from app import generate_sse

        mock_graph = _make_mock_graph([_node_start("supervisor"), _node_end("supervisor")])

        collected = []
        async for sse in generate_sse("q", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert payloads[1]["error"] is None

    @pytest.mark.anyio
    async def test_error_captured_from_output(self):
        """End event with error in output → error field populated."""
        from app import generate_sse

        end_event = {
            "event": "on_chain_end",
            "name": "web_search",
            "metadata": {"langgraph_node": "web_search"},
            "data": {"output": {"error": "TimeoutError: request timed out"}},
        }
        mock_graph = _make_mock_graph([_node_start("web_search"), end_event])

        collected = []
        async for sse in generate_sse("q", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert payloads[1]["error"] == "TimeoutError: request timed out"


# ---------------------------------------------------------------------------
# TestSSEUsageEvents
# ---------------------------------------------------------------------------

def _chat_model_end(node_name: str, input_tokens: int, output_tokens: int, total_tokens: int) -> dict:
    """Build an on_chat_model_end event with usage_metadata."""
    output = SimpleNamespace(
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
    )
    return {
        "event": "on_chat_model_end",
        "name": "ChatOpenAI",
        "metadata": {"langgraph_node": node_name},
        "data": {"output": output},
    }


def _chat_model_end_no_usage(node_name: str) -> dict:
    """Build an on_chat_model_end event without usage_metadata."""
    output = SimpleNamespace(usage_metadata=None)
    return {
        "event": "on_chat_model_end",
        "name": "ChatOpenAI",
        "metadata": {"langgraph_node": node_name},
        "data": {"output": output},
    }


class TestSSEUsageEvents:
    """Tests for token usage SSE events."""

    @pytest.mark.anyio
    async def test_emits_usage_event(self):
        """on_chat_model_end with usage_metadata → usage SSE event."""
        from app import generate_sse

        mock_graph = _make_mock_graph([_chat_model_end("generate_answer", 100, 50, 150)])

        collected = []
        async for sse in generate_sse("q", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert len(payloads) == 1
        assert payloads[0] == {
            "type": "usage",
            "node": "generate_answer",
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }

    @pytest.mark.anyio
    async def test_no_usage_event_when_no_metadata(self):
        """on_chat_model_end without usage_metadata → no event emitted."""
        from app import generate_sse

        mock_graph = _make_mock_graph([_chat_model_end_no_usage("generate_answer")])

        collected = []
        async for sse in generate_sse("q", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert payloads == []

    @pytest.mark.anyio
    async def test_usage_interleaved_with_node_events(self):
        """Usage events appear alongside node lifecycle events."""
        from app import generate_sse

        events = [
            _node_start("generate_answer"),
            _token_event("generate_answer", "Hi"),
            _chat_model_end("generate_answer", 200, 100, 300),
            _node_end("generate_answer"),
        ]
        mock_graph = _make_mock_graph(events)

        collected = []
        async for sse in generate_sse("q", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        types = [p["type"] for p in payloads]
        assert types == ["node_event", "token", "usage", "node_event"]


# ---------------------------------------------------------------------------
# TestSSETextEvent — "text" SSE event for non-streaming nodes (AC-02)
# ---------------------------------------------------------------------------

class TestSSETextEvent:
    """Tests that TEXT_EMIT_NODES produce a 'text' SSE event on chain end."""

    @pytest.mark.anyio
    async def test_text_event_emitted_for_plan_output(self):
        """on_chain_end for plan_output with AIMessage → text SSE event."""
        from langchain_core.messages import AIMessage
        from app import generate_sse

        end_event = {
            "event": "on_chain_end",
            "name": "plan_output",
            "metadata": {"langgraph_node": "plan_output"},
            "data": {"output": {"messages": [AIMessage(content="## 最终计划")]}},
        }
        mock_graph = _make_mock_graph([_node_start("plan_output"), end_event])

        collected = []
        async for sse in generate_sse("q", mock_graph):
            collected.append(sse)

        all_payloads = [json.loads(s.removeprefix("data: ").strip()) for s in collected]
        text_events = [p for p in all_payloads if p.get("type") == "text"]
        assert len(text_events) == 1
        assert text_events[0]["content"] == "## 最终计划"
        assert text_events[0]["node"] == "plan_output"

    @pytest.mark.anyio
    async def test_text_event_emitted_for_handle_unknown(self):
        """on_chain_end for handle_unknown with AIMessage → text SSE event."""
        from langchain_core.messages import AIMessage
        from app import generate_sse

        end_event = {
            "event": "on_chain_end",
            "name": "handle_unknown",
            "metadata": {"langgraph_node": "handle_unknown"},
            "data": {"output": {"messages": [AIMessage(content="我不太理解您的问题")]}},
        }
        mock_graph = _make_mock_graph([_node_start("handle_unknown"), end_event])

        collected = []
        async for sse in generate_sse("q", mock_graph):
            collected.append(sse)

        all_payloads = [json.loads(s.removeprefix("data: ").strip()) for s in collected]
        text_events = [p for p in all_payloads if p.get("type") == "text"]
        assert len(text_events) == 1
        assert text_events[0]["content"] == "我不太理解您的问题"

    @pytest.mark.anyio
    async def test_no_text_event_for_non_text_emit_node(self):
        """on_chain_end for a node NOT in TEXT_EMIT_NODES → no text event."""
        from langchain_core.messages import AIMessage
        from app import generate_sse

        end_event = {
            "event": "on_chain_end",
            "name": "generate_answer",
            "metadata": {"langgraph_node": "generate_answer"},
            "data": {"output": {"messages": [AIMessage(content="some answer")]}},
        }
        mock_graph = _make_mock_graph([_node_start("generate_answer"), end_event])

        collected = []
        async for sse in generate_sse("q", mock_graph):
            collected.append(sse)

        all_payloads = [json.loads(s.removeprefix("data: ").strip()) for s in collected]
        text_events = [p for p in all_payloads if p.get("type") == "text"]
        assert len(text_events) == 0


class TestSSEMindmapResult:
    """Tests that mindmap_output emits structured mindmap_result SSE payloads."""

    @pytest.mark.anyio
    async def test_mindmap_result_emitted(self):
        from langchain_core.messages import AIMessage
        from app import generate_sse

        end_event = {
            "event": "on_chain_end",
            "name": "mindmap_output",
            "metadata": {"langgraph_node": "mindmap_output"},
            "data": {
                "output": {
                    "messages": [AIMessage(content="已生成思维导图")],
                    "mindmap_artifact": {
                        "title": "过拟合",
                        "tree": {"title": "过拟合", "children": []},
                        "xmind_url": "/artifacts/mindmaps/a/mindmap.xmind",
                    },
                },
            },
        }
        mock_graph = _make_mock_graph([_node_start("mindmap_output"), end_event])

        collected = []
        async for sse in generate_sse("q", mock_graph):
            collected.append(sse)

        all_payloads = [json.loads(s.removeprefix("data: ").strip()) for s in collected]
        mindmap_events = [p for p in all_payloads if p.get("type") == "mindmap_result"]
        assert len(mindmap_events) == 1
        assert mindmap_events[0]["title"] == "过拟合"
        assert mindmap_events[0]["tree"]["title"] == "过拟合"


# ---------------------------------------------------------------------------
# TestSSEDoneEvent — "done" SSE event at stream completion (BUG-09)
# ---------------------------------------------------------------------------

class TestSSEDoneEvent:
    """Tests that the last SSE event after normal completion is 'done'."""

    @pytest.mark.anyio
    async def test_done_event_emitted_on_normal_completion(self):
        """After normal stream completion, the last event should be 'done'."""
        from app import generate_sse

        mock_graph = _make_mock_graph([_node_start("supervisor"), _node_end("supervisor")])

        collected = []
        async for sse in generate_sse("q", mock_graph):
            collected.append(sse)

        all_payloads = [json.loads(s.removeprefix("data: ").strip()) for s in collected]
        assert all_payloads[-1] == {"type": "done"}

    @pytest.mark.anyio
    async def test_no_done_event_on_interrupt(self):
        """When graph is interrupted, no 'done' event should be emitted."""
        from app import generate_sse

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(return_value=AsyncIteratorMock([]))

        interrupt_obj = SimpleNamespace(value="## 计划草稿")
        task = SimpleNamespace(interrupts=[interrupt_obj])
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=("plan_output",), tasks=[task]),
        )

        collected = []
        async for sse in generate_sse("q", mock_graph, thread_id="t-1"):
            collected.append(sse)

        all_payloads = [json.loads(s.removeprefix("data: ").strip()) for s in collected]
        done_events = [p for p in all_payloads if p.get("type") == "done"]
        assert len(done_events) == 0
