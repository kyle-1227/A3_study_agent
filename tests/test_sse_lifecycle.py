"""Unit tests for SSE node lifecycle events in generate_stream_drafts.

Tests cover: node start/end events, token streaming coexistence,
sub-chain filtering, internal node filtering, and event ordering.
All tests mock astream_events - no real graph execution required.

NOTE: generate_stream_drafts now emits a thread_id event first (REQ-08 HIL),
and calls graph.aget_state() after streaming to detect interrupts.
All mock graphs must provide aget_state as AsyncMock.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.stream_draft_helpers import draft_payloads
from src.observability.a3_trace import emit_a3_trace


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
    mock_graph.aupdate_state = AsyncMock()
    node_ids = {"supervisor"} | {
        str(event.get("metadata", {}).get("langgraph_node") or "").strip()
        for event in (events or [])
        if isinstance(event, dict)
    }
    mock_graph._a3_node_ids = frozenset(
        node_id
        for node_id in node_ids
        if node_id and not (node_id.startswith("__") and node_id.endswith("__"))
    )
    return mock_graph


def _parse_payloads(collected):
    """Unwrap native drafts, skipping thread and run-control progress."""
    all_payloads = draft_payloads(collected)
    assert all_payloads[0]["type"] == "thread_id"
    return [p for p in all_payloads[1:] if p.get("type") != "run_status"]


def _parse_all_payloads(collected):
    """Unwrap native drafts without filtering run-control progress."""
    return draft_payloads(collected)


async def _trace_only_events(stage: str, payload: dict | None = None):
    emit_a3_trace(
        logging.getLogger(__name__),
        stage,
        payload or {},
        state={},
        env_flag="LOG_GENERATION_SUMMARY",
    )
    if False:
        yield {}


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
    """Tests that generate_stream_drafts emits node lifecycle events."""

    @pytest.mark.anyio
    async def test_yields_node_start_event(self):
        """on_chain_start for a graph node -> {"type": "node_event", "status": "start"}."""
        from app import generate_stream_drafts

        mock_graph = _make_mock_graph([_node_start("supervisor")])

        collected = []
        async for sse in generate_stream_drafts("hello", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert len(payloads) == 1
        assert payloads[0] == {
            "type": "node_event",
            "status": "start",
            "node": "supervisor",
        }

    @pytest.mark.anyio
    async def test_yields_node_end_event(self):
        """on_chain_end for a graph node -> {"type": "node_event", "status": "end"}."""
        from app import generate_stream_drafts

        mock_graph = _make_mock_graph([_node_end("rag_retrieve")])

        collected = []
        async for sse in generate_stream_drafts("hello", mock_graph):
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
        from app import generate_stream_drafts

        events = [
            _sub_chain_start("RunnableSequence", "supervisor"),
            _sub_chain_start("ChatPromptTemplate", "generate_answer"),
        ]
        mock_graph = _make_mock_graph(events)

        collected = []
        async for sse in generate_stream_drafts("hello", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert payloads == []

    @pytest.mark.anyio
    async def test_ignores_langgraph_internal_nodes(self):
        """LangGraph internal nodes like __start__ must be filtered out."""
        from app import generate_stream_drafts

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
        async for sse in generate_stream_drafts("hello", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert payloads == []


# ---------------------------------------------------------------------------
# TestContentBlockStreaming
# ---------------------------------------------------------------------------


class TestContentBlockStreaming:
    """Ensure provider deltas use native provisional content blocks."""

    @pytest.mark.anyio
    async def test_delta_from_allowed_node(self):
        """Deltas from ALLOWED_NODES form one ordered content block."""
        from app import generate_stream_drafts

        mock_graph = _make_mock_graph([_token_event("generate_answer", "Hello")])

        collected = []
        async for sse in generate_stream_drafts("hi", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert [payload["type"] for payload in payloads] == [
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
        ]
        assert payloads[1]["delta"] == "Hello"
        assert {payload["block_id"] for payload in payloads} == {
            payloads[0]["block_id"]
        }

    @pytest.mark.anyio
    async def test_delta_from_disallowed_node_dropped(self):
        """Deltas from nodes NOT in ALLOWED_NODES should be dropped."""
        from app import generate_stream_drafts

        mock_graph = _make_mock_graph([_token_event("supervisor", "thinking...")])

        collected = []
        async for sse in generate_stream_drafts("hi", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert payloads == []

    @pytest.mark.anyio
    async def test_empty_delta_dropped(self):
        """Empty delta content should not produce a content block."""
        from app import generate_stream_drafts

        mock_graph = _make_mock_graph([_token_event("generate_answer", "")])

        collected = []
        async for sse in generate_stream_drafts("hi", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert payloads == []


# ---------------------------------------------------------------------------
# TestSSEMixedEventOrdering
# ---------------------------------------------------------------------------


class TestSSEMixedEventOrdering:
    """Tests ordering when lifecycle and content-block events are interleaved."""

    @pytest.mark.anyio
    async def test_full_academic_flow(self):
        """Simulate supervisor -> rag_retrieve -> generate_answer with tokens."""
        from app import generate_stream_drafts

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
        async for sse in generate_stream_drafts("What is calculus?", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        # Sub-chain event is dropped; one block wraps both answer deltas.
        assert len(payloads) == 10

        assert payloads[0] == {
            "type": "node_event",
            "status": "start",
            "node": "supervisor",
        }
        assert payloads[1]["type"] == "node_event"
        assert payloads[1]["status"] == "end"
        assert payloads[1]["node"] == "supervisor"
        assert payloads[2] == {
            "type": "node_event",
            "status": "start",
            "node": "rag_retrieve",
        }
        assert payloads[3]["type"] == "node_event"
        assert payloads[3]["status"] == "end"
        assert payloads[3]["node"] == "rag_retrieve"
        assert payloads[4] == {
            "type": "node_event",
            "status": "start",
            "node": "generate_answer",
        }
        assert payloads[5]["type"] == "content_block_start"
        assert payloads[6]["type"] == "content_block_delta"
        assert payloads[6]["delta"] == "The"
        assert payloads[7]["type"] == "content_block_delta"
        assert payloads[7]["delta"] == " answer"
        assert payloads[8]["type"] == "node_event"
        assert payloads[8]["status"] == "end"
        assert payloads[8]["node"] == "generate_answer"
        assert payloads[9]["type"] == "content_block_stop"

    @pytest.mark.anyio
    async def test_emotional_flow(self):
        """Simulate supervisor -> emotional_response with tokens."""
        from app import generate_stream_drafts

        events = [
            _node_start("supervisor"),
            _node_end("supervisor"),
            _node_start("emotional_response"),
            _token_event("emotional_response", "I understand"),
            _node_end("emotional_response"),
        ]
        mock_graph = _make_mock_graph(events)

        collected = []
        async for sse in generate_stream_drafts("I'm stressed", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert len(payloads) == 7
        assert payloads[0]["type"] == "node_event"
        assert payloads[3]["type"] == "content_block_start"
        assert payloads[4]["type"] == "content_block_delta"
        assert payloads[4]["delta"] == "I understand"
        assert payloads[-1]["type"] == "content_block_stop"


# ---------------------------------------------------------------------------
# TestSSEAllGraphNodes
# ---------------------------------------------------------------------------


class TestSSEAllGraphNodes:
    """Ensure every known graph node can produce lifecycle events."""

    ALL_NODES = [
        "supervisor",
        "episodic_memory_retriever",
        "episodic_memory_writer",
        "memory_use_decider",
        "academic_router",
        "search_query_rewriter",
        "rag_retrieve",
        "web_search",
        "evidence_judge",
        "evidence_summary_output",
        "generate_answer",
        "evaluate_hallucination",
        "rewrite_query",
        "resource_orchestrator",
        "resource_worker",
        "resource_bundle_output",
        "emotional_response",
        "handle_unknown",
    ]

    @pytest.mark.anyio
    @pytest.mark.parametrize("node_name", ALL_NODES)
    async def test_each_node_emits_start(self, node_name):
        """Every graph node should produce a start event."""
        from app import generate_stream_drafts

        mock_graph = _make_mock_graph([_node_start(node_name)])

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert len(payloads) == 1
        assert payloads[0]["node"] == node_name
        assert payloads[0]["status"] == "start"

    @pytest.mark.anyio
    @pytest.mark.parametrize("node_name", ALL_NODES)
    async def test_each_node_emits_end(self, node_name):
        """Every graph node should produce an end event with duration_ms and error."""
        from app import generate_stream_drafts

        mock_graph = _make_mock_graph([_node_start(node_name), _node_end(node_name)])

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph):
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
        from app import generate_stream_drafts

        mock_graph = _make_mock_graph(
            [_node_start("supervisor"), _node_end("supervisor")]
        )

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert payloads[1]["duration_ms"] is not None
        assert payloads[1]["duration_ms"] >= 0

    @pytest.mark.anyio
    async def test_end_without_start_has_null_duration(self):
        """An end event without a preceding start should have duration_ms=None."""
        from app import generate_stream_drafts

        mock_graph = _make_mock_graph([_node_end("supervisor")])

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph):
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
        """Normal end -> error is null."""
        from app import generate_stream_drafts

        mock_graph = _make_mock_graph(
            [_node_start("supervisor"), _node_end("supervisor")]
        )

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert payloads[1]["error"] is None

    @pytest.mark.anyio
    async def test_error_captured_from_output(self):
        """End event with error in output -> error field populated."""
        from app import generate_stream_drafts

        end_event = {
            "event": "on_chain_end",
            "name": "web_search",
            "metadata": {"langgraph_node": "web_search"},
            "data": {"output": {"error": "TimeoutError: request timed out"}},
        }
        mock_graph = _make_mock_graph([_node_start("web_search"), end_event])

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert payloads[1]["error"] == "TimeoutError: request timed out"

    @pytest.mark.anyio
    async def test_exception_emits_synthetic_error_for_active_node(self):
        """If streaming fails mid-node, emit a synthetic node error before global error."""
        from app import generate_stream_drafts

        class RaisingIterator:
            def __init__(self):
                self._started = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._started:
                    self._started = True
                    return _node_start("evidence_judge")
                raise RuntimeError("Evidence Judge timed out")

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(return_value=RaisingIterator())
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=(), tasks=[])
        )
        mock_graph.aupdate_state = AsyncMock()
        mock_graph._a3_node_ids = frozenset({"supervisor", "evidence_judge"})

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert payloads[0] == {
            "type": "node_event",
            "status": "start",
            "node": "evidence_judge",
        }
        assert payloads[1]["type"] == "node_event"
        assert payloads[1]["status"] == "end"
        assert payloads[1]["node"] == "evidence_judge"
        assert payloads[1]["synthetic"] is True
        assert payloads[1]["error_type"] == "RuntimeError"
        assert payloads[2]["type"] == "stream_error"
        assert payloads[2]["failed_node"] == "evidence_judge"
        assert "evidence_judge" in payloads[2]["active_nodes"]


# ---------------------------------------------------------------------------
# TestSSEUsageEvents
# ---------------------------------------------------------------------------


def _chat_model_end(
    node_name: str, input_tokens: int, output_tokens: int, total_tokens: int
) -> dict:
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
        """on_chat_model_end with usage_metadata -> usage SSE event."""
        from app import generate_stream_drafts

        mock_graph = _make_mock_graph(
            [_chat_model_end("generate_answer", 100, 50, 150)]
        )

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph):
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
        """on_chat_model_end without usage_metadata -> no event emitted."""
        from app import generate_stream_drafts

        mock_graph = _make_mock_graph([_chat_model_end_no_usage("generate_answer")])

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        assert payloads == []

    @pytest.mark.anyio
    async def test_usage_interleaved_with_node_events(self):
        """Usage events appear alongside node lifecycle events."""
        from app import generate_stream_drafts

        events = [
            _node_start("generate_answer"),
            _token_event("generate_answer", "Hi"),
            _chat_model_end("generate_answer", 200, 100, 300),
            _node_end("generate_answer"),
        ]
        mock_graph = _make_mock_graph(events)

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph):
            collected.append(sse)

        payloads = _parse_payloads(collected)
        types = [p["type"] for p in payloads]
        assert types == [
            "node_event",
            "content_block_start",
            "content_block_delta",
            "usage",
            "node_event",
            "content_block_stop",
        ]


# ---------------------------------------------------------------------------
# TestNonStreamingContentBlocks
# ---------------------------------------------------------------------------


class TestNonStreamingContentBlocks:
    """Tests that configured non-streaming nodes produce one content block."""

    @pytest.mark.anyio
    async def test_content_block_emitted_for_handle_unknown(self):
        """handle_unknown emits its final AIMessage as one content block."""
        from langchain_core.messages import AIMessage
        from app import generate_stream_drafts

        end_event = {
            "event": "on_chain_end",
            "name": "handle_unknown",
            "metadata": {"langgraph_node": "handle_unknown"},
            "data": {
                "output": {
                    "messages": [
                        AIMessage(content="I could not understand the request.")
                    ]
                }
            },
        }
        mock_graph = _make_mock_graph([_node_start("handle_unknown"), end_event])

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph):
            collected.append(sse)

        all_payloads = draft_payloads(collected)
        content_events = [
            payload
            for payload in all_payloads
            if str(payload.get("type") or "").startswith("content_block_")
        ]
        assert [payload["type"] for payload in content_events] == [
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
        ]
        assert content_events[1]["delta"] == "I could not understand the request."

    @pytest.mark.anyio
    async def test_content_block_emitted_for_resource_bundle_output(self):
        """resource_bundle_output emits the bundle message as one block."""
        from langchain_core.messages import AIMessage
        from app import generate_stream_drafts

        end_event = {
            "event": "on_chain_end",
            "name": "resource_bundle_output",
            "metadata": {"langgraph_node": "resource_bundle_output"},
            "data": {
                "output": {"messages": [AIMessage(content="## 学习资源生成结果")]}
            },
        }
        mock_graph = _make_mock_graph(
            [_node_start("resource_bundle_output"), end_event]
        )

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph):
            collected.append(sse)

        all_payloads = draft_payloads(collected)
        content_events = [
            payload
            for payload in all_payloads
            if str(payload.get("type") or "").startswith("content_block_")
        ]
        assert [payload["type"] for payload in content_events] == [
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
        ]
        assert "resource_bundle_output" in content_events[0]["block_id"]

    @pytest.mark.anyio
    async def test_resource_bundle_final_uses_terminal_output_when_snapshot_is_stale(
        self,
    ):
        """A stale checkpoint must not drop current successful resource downloads."""
        from langchain_core.messages import AIMessage
        from app import generate_stream_drafts

        resource_output = {
            "resource_generation_status": "partial_success",
            "resource_bundle_artifact": {
                "type": "resource_bundle",
                "status": "partial_success",
                "message": "# 已生成多类学习资源",
                "resources": [
                    {"resource_type": "mindmap", "status": "success"},
                    {"resource_type": "quiz", "status": "success"},
                    {"resource_type": "video_script", "status": "success"},
                ],
                "errors": [
                    {
                        "resource_type": "review_doc",
                        "status": "failed",
                        "error_type": "TimeoutError",
                    }
                ],
            },
            "mindmap_artifact": {
                "title": "机器学习复习思维导图",
                "tree": {"title": "机器学习", "children": []},
                "xmind_url": "/artifacts/mindmaps/m1/map.xmind",
            },
            "mindmap_tree": {"title": "机器学习", "children": []},
            "exercise_items": [{"question": "Q1"}],
            "exercise_artifact": {
                "title": "机器学习练习题",
                "markdown_url": "/artifacts/exercises/e1/quiz.md",
                "docx_url": "/artifacts/exercises/e1/quiz.docx",
            },
            "video_script_artifact": {
                "title": "10分钟理解 Python 类与对象",
                "markdown_url": "/artifacts/video-scripts/v1/script.md",
                "docx_url": "/artifacts/video-scripts/v1/script.docx",
                "srt_url": "/artifacts/video-scripts/v1/script.srt",
            },
            "messages": [AIMessage(content="# 已生成多类学习资源")],
        }
        end_event = {
            "event": "on_chain_end",
            "name": "resource_bundle_output",
            "metadata": {"langgraph_node": "resource_bundle_output"},
            "data": {"output": resource_output},
        }
        mock_graph = _make_mock_graph(
            [_node_start("resource_bundle_output"), end_event]
        )
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(
                next=(),
                tasks=[],
                values={
                    "requested_resource_type": "code_practice",
                    "code_practice_artifact": {
                        "title": "stale code practice",
                        "markdown_url": "/artifacts/code-practice/old.md",
                    },
                    "messages": [AIMessage(content="stale checkpoint")],
                },
            ),
        )

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph, thread_id="t-1"):
            collected.append(sse)

        all_payloads = draft_payloads(collected)
        resource_events = [p for p in all_payloads if p.get("type") == "resource_final"]

        assert len(resource_events) == 1
        assert all_payloads[-1] is resource_events[0]
        payload = resource_events[0]
        assert payload["resource_type"] == "bundle"
        assert payload["resource_id"].startswith("resource:v1:")
        assert payload["payload_hash"].startswith("payload:v1:")
        assert payload["schema_version"] == 2
        assert payload["thread_id"] == "t-1"
        assert payload["request_id"]
        assert payload["terminal_status"] == "partial_success"
        assert payload["resource"]["kind"] == "bundle"
        assert payload["resource_generation_status"] == "partial_success"
        assert payload["answer"] == "# 已生成多类学习资源"
        assert payload["mindmap"]["xmind_url"] == "/artifacts/mindmaps/m1/map.xmind"
        assert (
            payload["exercise_artifact"]["markdown_url"]
            == "/artifacts/exercises/e1/quiz.md"
        )
        assert (
            payload["video_script_artifact"]["srt_url"]
            == "/artifacts/video-scripts/v1/script.srt"
        )
        assert payload["errors"][0]["resource_type"] == "review_doc"
        assert "review_doc" not in payload
        assert "code_practice_artifact" not in payload

    @pytest.mark.anyio
    async def test_no_content_block_for_non_emit_node(self):
        """A non-configured chain-end message does not create a content block."""
        from langchain_core.messages import AIMessage
        from app import generate_stream_drafts

        end_event = {
            "event": "on_chain_end",
            "name": "generate_answer",
            "metadata": {"langgraph_node": "generate_answer"},
            "data": {"output": {"messages": [AIMessage(content="some answer")]}},
        }
        mock_graph = _make_mock_graph([_node_start("generate_answer"), end_event])

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph):
            collected.append(sse)

        all_payloads = draft_payloads(collected)
        content_events = [
            payload
            for payload in all_payloads
            if str(payload.get("type") or "").startswith("content_block_")
        ]
        assert content_events == []


class TestSSEEvidenceSummaryResourceFinal:
    """Evidence controlled stop should emit a normal resource_final event."""

    @pytest.mark.anyio
    async def test_evidence_summary_resource_final_emitted(self):
        from app import generate_stream_drafts

        final_state = {
            "evidence_controlled_stop": True,
            "final_response_type": "evidence_summary",
            "requested_resource_type": "study_plan",
            "evidence_controlled_stop_reason": "evidence_insufficient",
            "plan": "## Evidence summary\nEvidence is insufficient for a full resource.",
            "study_plan_artifact": {},
        }
        mock_graph = _make_mock_graph([])
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=(), tasks=[], values=final_state),
        )

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph, thread_id="t-1"):
            collected.append(sse)

        all_payloads = draft_payloads(collected)
        resource_events = [p for p in all_payloads if p.get("type") == "resource_final"]
        assert len(resource_events) == 1
        assert all_payloads[-1] is resource_events[0]
        assert resource_events[0]["resource_type"] == "evidence_summary"
        assert resource_events[0]["resource_id"].startswith("resource:v1:")
        assert resource_events[0]["controlled_stop"] is True
        assert "study_plan" not in resource_events[0]
        assert all_payloads[-1]["type"] == "resource_final"


class TestSSEQAFinal:
    @pytest.mark.anyio
    async def test_qa_final_is_the_authoritative_producer_terminal(self):
        from app import generate_stream_drafts
        from src.graph.qa import QAResponse, QASuggestion, build_qa_final_payload

        qa_payload = build_qa_final_payload(
            response=QAResponse(
                answer="A grounded QA answer.",
                uncertainty_note="",
                grounding_status="general_knowledge",
                suggestions=[
                    QASuggestion(
                        label="Continue",
                        action="continue_qa",
                        resource_type="",
                    )
                ],
            ),
            qa_scope="general",
            thread_id="t-1",
            request_id="request-1",
        )
        final_state = {
            "request_id": "request-1",
            "thread_id": "t-1",
            "response_mode": "qa",
            "final_response_type": "qa",
            "last_qa_response": qa_payload,
        }
        mock_graph = _make_mock_graph([])
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=(), tasks=[], values=final_state),
        )

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph, thread_id="t-1"):
            collected.append(sse)

        payloads = _parse_all_payloads(collected)
        qa_events = [item for item in payloads if item.get("type") == "qa_final"]
        assert len(qa_events) == 1
        assert payloads[-1] is qa_events[0]
        assert not [
            item
            for item in payloads
            if item.get("type") in {"resource_final", "resource_final_diagnostic"}
        ]


class TestSSEResourceFinalDiagnostics:
    @pytest.mark.anyio
    async def test_plain_answer_does_not_emit_authoritative_resource_completion(self):
        from langchain_core.messages import AIMessage
        from app import generate_stream_drafts

        final_state = {
            "requested_resource_type": "",
            "resource_generation_status": "",
            "messages": [AIMessage(content="plain answer")],
        }
        mock_graph = _make_mock_graph([])
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=(), tasks=[], values=final_state),
        )

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph, thread_id="t-1"):
            collected.append(sse)

        all_payloads = draft_payloads(collected)
        assert not [
            payload
            for payload in all_payloads
            if payload.get("type") == "resource_final_diagnostic"
        ]
        assert not [
            payload
            for payload in all_payloads
            if payload.get("type") == "resource_final"
        ]
        assert not [
            payload for payload in all_payloads if payload.get("type") == "stream_done"
        ]


class TestSSEProviderRetryEvents:
    """Provider transport retry traces should be visible to the frontend."""

    @pytest.mark.anyio
    async def test_provider_retry_trace_is_emitted_as_sse_event(self):
        from app import generate_stream_drafts

        async def events():
            emit_a3_trace(
                logging.getLogger("test_sse_provider_retry"),
                "provider_transport_retry_attempt",
                {
                    "node_name": "evidence_judge",
                    "llm_node": "evidence_judge",
                    "provider": "openrouter",
                    "model": "test-model",
                    "retry_count": 1,
                    "max_retries": 2,
                    "next_attempt": 2,
                    "fallback_used": False,
                    "error_type": "ConnectError",
                    "error_message": "connection failed",
                    "status_code": None,
                },
                state={"request_id": "r1"},
            )
            yield _node_start("evidence_judge")

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(return_value=events())
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=(), tasks=[], values={}),
        )
        mock_graph.aupdate_state = AsyncMock()

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph, thread_id="t-1"):
            collected.append(sse)

        payloads = draft_payloads(collected)
        retry_events = [
            payload for payload in payloads if payload.get("type") == "provider_retry"
        ]
        assert len(retry_events) == 1
        assert retry_events[0]["stage"] == "provider_transport_retry_attempt"
        assert retry_events[0]["node"] == "evidence_judge"
        assert retry_events[0]["retry_count"] == 1
        assert retry_events[0]["max_retries"] == 2


class TestSSEResourceSubnodeEvents:
    """Resource worker internal subnode traces should be visible safely."""

    @pytest.mark.anyio
    async def test_resource_subnode_trace_is_emitted_as_sse_event(self):
        from app import generate_stream_drafts

        async def events():
            emit_a3_trace(
                logging.getLogger("test_sse_resource_subnode"),
                "resource_subnode.start",
                {
                    "resource_type": "quiz",
                    "subnode": "exercise_agent api_key=secret",
                    "elapsed_ms": 0,
                    "status": "start",
                    "error_type": "",
                    "content": "must-not-leak",
                },
                state={"request_id": "r1"},
            )
            yield _node_start("resource_worker")

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(return_value=events())
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=(), tasks=[], values={}),
        )
        mock_graph.aupdate_state = AsyncMock()

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph, thread_id="t-1"):
            collected.append(sse)

        payloads = draft_payloads(collected)
        subnode_events = [
            payload for payload in payloads if payload.get("type") == "resource_subnode"
        ]
        assert len(subnode_events) == 1
        assert subnode_events[0]["stage"] == "resource_subnode.start"
        assert subnode_events[0]["resource_type"] == "quiz"
        assert subnode_events[0]["subnode"].startswith("exercise_agent")
        assert "[REDACTED]" in subnode_events[0]["subnode"]
        assert subnode_events[0]["status"] == "start"
        assert "content" not in subnode_events[0]
        assert "secret" not in json.dumps(subnode_events[0], ensure_ascii=False)


class TestSSEContextErrorEvents:
    """CE fail-fast traces should become explicit frontend-safe SSE errors."""

    @pytest.mark.anyio
    async def test_required_source_missing_emits_context_error_and_failed_status(self):
        from app import generate_stream_drafts

        async def events():
            emit_a3_trace(
                logging.getLogger("test_sse_context_error"),
                "context_apply_error",
                {
                    "node_name": "review_doc_agent",
                    "llm_node": "review_doc",
                    "trace_call_id": "call-1",
                    "trace_seq": 4,
                    "reason": "required_sources_missing",
                    "warning": "required context sources are unavailable",
                    "fallback_used": False,
                    "error_scope": "provider",
                    "recoverable": False,
                    "required_sources_missing": ["evidence"],
                    "provider_missing_reasons": {"evidence": "provider_empty"},
                    "content": "api_key=sk-secret must not leak",
                },
                state={"request_id": "r1"},
            )
            yield _node_start("review_doc_agent")

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(return_value=events())
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=(), tasks=[], values={}),
        )
        mock_graph.aupdate_state = AsyncMock()

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph, thread_id="t-1"):
            collected.append(sse)

        payloads = draft_payloads(collected)
        context_errors = [
            payload for payload in payloads if payload.get("type") == "context_error"
        ]
        assert len(context_errors) == 1
        assert context_errors[0] == {
            "type": "context_error",
            "stage": "context_apply_error",
            "node": "review_doc_agent",
            "llm_node": "review_doc",
            "trace_call_id": "call-1",
            "trace_seq": 4,
            "reason": "required_sources_missing",
            "required_sources_missing": ["evidence"],
            "required_sources_filtered_out": [],
            "recoverable": False,
            "provider_missing_reasons": {"evidence": "provider_empty"},
            "source_drop_reasons": {},
            "budget_drop_reasons": {},
            "source_counts_before": {},
            "source_counts_after": {},
            "source_counts_dropped": {},
        }
        mock_graph.aupdate_state.assert_any_await(
            {"configurable": {"thread_id": "t-1"}},
            {
                "run_status": "failed",
                "resume_available": False,
                "pending_interrupt_type": "",
            },
            as_node="supervisor",
        )
        serialized = json.dumps(context_errors[0], ensure_ascii=False).lower()
        assert "api_key" not in serialized
        assert "sk-secret" not in serialized
        assert "must not leak" not in serialized

    @pytest.mark.anyio
    async def test_required_source_filtered_out_emits_safe_root_cause(self):
        from app import generate_stream_drafts

        async def events():
            emit_a3_trace(
                logging.getLogger("test_sse_context_filtered_out"),
                "context_apply_error",
                {
                    "node_name": "review_doc_agent",
                    "llm_node": "review_doc",
                    "trace_call_id": "call-2",
                    "trace_seq": 6,
                    "reason": "required_sources_filtered_out",
                    "warning": "required context sources were filtered out",
                    "fallback_used": False,
                    "error_scope": "source_filter",
                    "recoverable": False,
                    "required_sources_missing": [],
                    "required_sources_filtered_out": ["evidence"],
                    "provider_missing_reasons": {},
                    "source_drop_reasons": {"quality_below_threshold": 1},
                    "budget_drop_reasons": {},
                    "source_counts_before": {"evidence": 1},
                    "source_counts_after": {},
                    "source_counts_dropped": {"evidence": 1},
                    "content": "api_key=sk-secret must not leak",
                },
                state={"request_id": "r1"},
            )
            yield _node_start("review_doc_agent")

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(return_value=events())
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=(), tasks=[], values={}),
        )
        mock_graph.aupdate_state = AsyncMock()

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph, thread_id="t-1"):
            collected.append(sse)

        payloads = draft_payloads(collected)
        context_error = next(
            payload for payload in payloads if payload.get("type") == "context_error"
        )

        assert context_error["reason"] == "required_sources_filtered_out"
        assert context_error["required_sources_filtered_out"] == ["evidence"]
        assert context_error["source_drop_reasons"] == {"quality_below_threshold": 1}
        assert context_error["source_counts_before"] == {"evidence": 1}
        assert context_error["source_counts_after"] == {}
        assert context_error["source_counts_dropped"] == {"evidence": 1}
        serialized = json.dumps(context_error, ensure_ascii=False).lower()
        assert "api_key" not in serialized
        assert "sk-secret" not in serialized
        assert "must not leak" not in serialized


# ---------------------------------------------------------------------------
# TestProducerTerminalOwnership
# ---------------------------------------------------------------------------


class TestProducerTerminalOwnership:
    """The producer emits drafts; the session alone appends stream_done."""

    @pytest.mark.anyio
    async def test_producer_does_not_emit_stream_done(self):
        from app import generate_stream_drafts

        mock_graph = _make_mock_graph(
            [_node_start("supervisor"), _node_end("supervisor")]
        )

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph):
            collected.append(sse)

        all_payloads = draft_payloads(collected)
        assert all_payloads[-1]["type"] == "node_event"
        assert not [
            payload for payload in all_payloads if payload.get("type") == "stream_done"
        ]

    @pytest.mark.anyio
    async def test_no_done_event_on_interrupt(self):
        """When graph is interrupted, no 'done' event should be emitted."""
        from app import generate_stream_drafts

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(return_value=AsyncIteratorMock([]))
        mock_graph.aupdate_state = AsyncMock()

        interrupt_obj = SimpleNamespace(value="## 请确认是否继续生成学习计划")
        task = SimpleNamespace(interrupts=[interrupt_obj])
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(
                next=("resource_bundle_output",), tasks=[task]
            ),
        )

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph, thread_id="t-1"):
            collected.append(sse)

        all_payloads = draft_payloads(collected)
        done_events = [p for p in all_payloads if p.get("type") == "stream_done"]
        assert len(done_events) == 0

    @pytest.mark.anyio
    async def test_memory_confirmation_interrupt_payload_is_typed(self):
        """Memory confirmation interrupt should be distinguishable from plan review."""
        from app import generate_stream_drafts

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(return_value=AsyncIteratorMock([]))
        mock_graph.aupdate_state = AsyncMock()

        interrupt_obj = SimpleNamespace(
            value={
                "type": "memory_confirmation",
                "question": "Use memory?",
                "reason": "ambiguous",
                "selected_memory_count": 2,
                "options": [
                    {"label": "Use", "value": "use"},
                    {"label": "Ignore", "value": "ignore"},
                ],
            }
        )
        task = SimpleNamespace(interrupts=[interrupt_obj])
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=("memory_use_decider",), tasks=[task]),
        )

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph, thread_id="t-1"):
            collected.append(sse)

        all_payloads = draft_payloads(collected)
        interrupt_events = [p for p in all_payloads if p.get("type") == "interrupt"]
        assert interrupt_events == [
            {
                "type": "interrupt",
                "interrupt_type": "memory_confirmation",
                "question": "Use memory?",
                "reason": "ambiguous",
                "selected_memory_count": 2,
                "options": [
                    {"label": "Use", "value": "use"},
                    {"label": "Ignore", "value": "ignore"},
                ],
                "thread_id": "t-1",
            }
        ]
        assert not [p for p in all_payloads if p.get("type") == "stream_done"]
        assert mock_graph.aupdate_state.await_count == 2
        interrupt_update = mock_graph.aupdate_state.await_args_list[-1].args[1]
        assert interrupt_update["pending_interrupt_type"] == "memory_confirmation"
        assert "activity_timeline" in interrupt_update

    @pytest.mark.anyio
    async def test_profile_completion_interrupt_payload_is_typed(self):
        """Profile completion interrupt should be distinguishable and resumable."""
        from app import generate_stream_drafts

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(return_value=AsyncIteratorMock([]))
        mock_graph.aupdate_state = AsyncMock()

        request_payload = {
            "title": "生成学习计划前需要补充学习信息",
            "fields": [
                {
                    "key": "learning_goal",
                    "label": "学习目标",
                    "required": True,
                    "max_chars": 400,
                }
            ],
        }
        interrupt_obj = SimpleNamespace(
            value={
                "type": "profile_completion_required",
                "profile_completion_request": request_payload,
                "resume_available": True,
            }
        )
        task = SimpleNamespace(interrupts=[interrupt_obj])
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=("resource_worker",), tasks=[task]),
        )

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph, thread_id="t-1"):
            collected.append(sse)

        all_payloads = draft_payloads(collected)
        interrupt_events = [p for p in all_payloads if p.get("type") == "interrupt"]
        assert interrupt_events == [
            {
                "type": "interrupt",
                "interrupt_type": "profile_completion_required",
                "title": "生成学习计划前需要补充学习信息",
                "fields": request_payload["fields"],
                "profile_completion_request": request_payload,
                "resume_available": True,
                "thread_id": "t-1",
            }
        ]
        assert not [p for p in all_payloads if p.get("type") == "stream_done"]
        # persist_checkpoint=True for profile_completion_required now causes
        # an additional aupdate_state call (initial + interrupt).
        assert mock_graph.aupdate_state.await_count == 2


# ---------------------------------------------------------------------------
# Interrupt from Send-dispatched worker with empty state_snapshot.next
# ---------------------------------------------------------------------------


class TestSSEInterruptWithEmptyNext:
    """When GraphInterrupt is raised inside a Send-dispatched worker,
    state_snapshot.next is () but tasks[].interrupts[] still holds the
    interrupt value. The producer must emit an authoritative interrupt, not
    resource_final or stream_done."""

    @pytest.mark.anyio
    async def test_profile_completion_interrupt_empty_next_emits_interrupt_only(self):
        """next=() with profile_completion_required interrupt must emit
        interrupt and MUST NOT emit resource_final or stream_done."""
        from app import generate_stream_drafts

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(return_value=AsyncIteratorMock([]))
        mock_graph.aupdate_state = AsyncMock()

        request_payload = {
            "title": "Need profile before study plan",
            "fields": [
                {
                    "key": "learning_goal",
                    "label": "Learning goal",
                    "required": True,
                    "max_chars": 400,
                }
            ],
        }
        interrupt_obj = SimpleNamespace(
            value={
                "type": "profile_completion_required",
                "profile_completion_request": request_payload,
                "resume_available": True,
            }
        )
        task = SimpleNamespace(interrupts=[interrupt_obj])
        # KEY: next is empty tuple (Send-dispatched worker interrupt)
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=(), tasks=[task]),
        )

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph, thread_id="t-1"):
            collected.append(sse)

        all_payloads = draft_payloads(collected)
        interrupt_events = [p for p in all_payloads if p.get("type") == "interrupt"]
        assert len(interrupt_events) == 1, (
            f"Expected 1 interrupt event, got {len(interrupt_events)}"
        )
        assert interrupt_events[0]["interrupt_type"] == "profile_completion_required"
        assert interrupt_events[0]["title"] == "Need profile before study plan"

        resource_final_events = [
            p for p in all_payloads if p.get("type") == "resource_final"
        ]
        assert resource_final_events == []
        completed_events = [
            p
            for p in all_payloads
            if p.get("type") == "run_status" and p.get("run_status") == "completed"
        ]
        assert completed_events == []
        done_events = [p for p in all_payloads if p.get("type") == "stream_done"]
        assert done_events == []
        # persist_checkpoint=True for profile_completion causes an additional
        # aupdate_state call (initial running state + interrupt state).
        assert mock_graph.aupdate_state.await_count == 2

    @pytest.mark.anyio
    async def test_user_stop_interrupt_empty_next_emits_stopped_only(self):
        """next=() with user_stop interrupt must emit stopped status
        and MUST NOT emit resource_final or stream_done."""
        from app import generate_stream_drafts

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(return_value=AsyncIteratorMock([]))
        mock_graph.aupdate_state = AsyncMock()

        interrupt_obj = SimpleNamespace(
            value={
                "type": "user_stop",
                "node": "resource_worker",
                "reason": "user requested stop",
            }
        )
        task = SimpleNamespace(interrupts=[interrupt_obj])
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=(), tasks=[task]),
        )

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph, thread_id="t-1"):
            collected.append(sse)

        all_payloads = draft_payloads(collected)
        stopped_events = [p for p in all_payloads if p.get("type") == "stopped"]
        assert len(stopped_events) == 1
        resource_final_events = [
            p for p in all_payloads if p.get("type") == "resource_final"
        ]
        assert resource_final_events == []
        done_events = [p for p in all_payloads if p.get("type") == "stream_done"]
        assert done_events == []

    @pytest.mark.anyio
    async def test_interrupt_prevents_completed_without_resource(self):
        """When an interrupt is pending, completed_without_resource diagnostic
        must NOT be emitted even if resource was requested."""
        from app import generate_stream_drafts

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(return_value=AsyncIteratorMock([]))
        mock_graph.aupdate_state = AsyncMock()

        interrupt_obj = SimpleNamespace(
            value={
                "type": "profile_completion_required",
                "profile_completion_request": {
                    "title": "Need profile",
                    "fields": [
                        {"key": "learning_goal", "label": "Goal", "required": True}
                    ],
                },
                "resume_available": True,
            }
        )
        task = SimpleNamespace(interrupts=[interrupt_obj])
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(
                next=(),
                tasks=[task],
                values={
                    "requested_resource_type": "study_plan",
                    "requested_resource_types": ["study_plan"],
                    "resource_generation_plan": {
                        "tasks": [{"resource_type": "study_plan"}]
                    },
                    "resource_generation_status": "running",
                },
            ),
        )

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph, thread_id="t-1"):
            collected.append(sse)

        all_payloads = draft_payloads(collected)
        diagnostic_events = [
            p for p in all_payloads if p.get("type") == "resource_final_diagnostic"
        ]
        assert diagnostic_events == [], (
            "completed_without_resource must not be emitted when interrupt is pending"
        )
        interrupt_events = [p for p in all_payloads if p.get("type") == "interrupt"]
        assert len(interrupt_events) == 1

    @pytest.mark.anyio
    async def test_lost_interrupt_emits_terminal_error_without_completion_events(self):
        """If worker interrupt evidence exists but checkpoint interrupts are empty,
        SSE must fail closed with a terminal non-completed error."""
        from app import generate_stream_drafts

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=_trace_only_events(
                "resource_generation.worker.interrupted",
                {"resource_type": "study_plan", "elapsed_ms": 42},
            )
        )
        mock_graph.aupdate_state = AsyncMock()
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(
                next=(),
                tasks=[],
                values={
                    "requested_resource_type": "study_plan",
                    "requested_resource_types": ["study_plan"],
                    "resource_generation_plan": {
                        "tasks": [{"resource_type": "study_plan"}]
                    },
                },
            )
        )

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph, thread_id="t-1"):
            collected.append(sse)

        all_payloads = _parse_all_payloads(collected)
        error_events = [p for p in all_payloads if p.get("type") == "stream_error"]
        assert len(error_events) == 1
        assert error_events[0]["error_type"] == "interrupt_lost"
        assert error_events[0]["terminal_non_completed"] is True
        assert "Completion was blocked" in error_events[0]["message"]
        assert [p for p in all_payloads if p.get("type") == "resource_final"] == []
        assert [
            p
            for p in all_payloads
            if p.get("type") == "resource_final_diagnostic"
            and p.get("status") == "completed_without_resource"
        ] == []
        assert [
            p
            for p in all_payloads
            if p.get("type") == "run_status" and p.get("run_status") == "completed"
        ] == []
        assert [p for p in all_payloads if p.get("type") == "stream_done"] == []

    @pytest.mark.anyio
    async def test_lost_interrupt_recovers_profile_completion_interrupt_from_state(
        self,
    ):
        """If checkpoint interrupt is missing but a complete request exists in state,
        SSE should emit recovered profile_completion_required interrupt."""
        from app import generate_stream_drafts

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=_trace_only_events(
                "resource_subnode.end",
                {
                    "resource_type": "study_plan",
                    "subnode": "study_plan_profile_gate",
                    "status": "interrupted",
                    "error_type": "GraphInterrupt",
                },
            )
        )
        mock_graph.aupdate_state = AsyncMock()
        recovered_request = {
            "title": "Need profile before study plan",
            "fields": [
                {
                    "key": "learning_goal",
                    "label": "Learning goal",
                    "required": True,
                    "max_chars": 400,
                }
            ],
        }
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(
                next=(),
                tasks=[],
                values={
                    "requested_resource_type": "study_plan",
                    "requested_resource_types": ["study_plan"],
                    "resource_generation_plan": {
                        "tasks": [{"resource_type": "study_plan"}]
                    },
                    "profile_completion_request": recovered_request,
                },
            )
        )

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph, thread_id="t-1"):
            collected.append(sse)

        all_payloads = _parse_all_payloads(collected)
        interrupt_events = [p for p in all_payloads if p.get("type") == "interrupt"]
        assert len(interrupt_events) == 1
        assert interrupt_events[0]["interrupt_type"] == "profile_completion_required"
        assert interrupt_events[0]["profile_completion_request"] == recovered_request
        assert interrupt_events[0]["resume_available"] is True
        assert [p for p in all_payloads if p.get("type") == "resource_final"] == []
        assert [
            p
            for p in all_payloads
            if p.get("type") == "resource_final_diagnostic"
            and p.get("status") == "completed_without_resource"
        ] == []
        assert [
            p
            for p in all_payloads
            if p.get("type") == "run_status" and p.get("run_status") == "completed"
        ] == []
        assert [p for p in all_payloads if p.get("type") == "stream_done"] == []

        run_state_updates = [
            call.args[1]
            for call in mock_graph.aupdate_state.await_args_list
            if len(call.args) >= 2 and isinstance(call.args[1], dict)
        ]
        assert any(
            update.get("pending_interrupt_type") == "profile_completion_required"
            and update.get("resume_available") is True
            and update.get("profile_completion_request") == recovered_request
            for update in run_state_updates
        )

    def test_profile_completion_trace_event_recovers_complete_request(self):
        from app import _profile_completion_request_from_trace_event

        event = {
            "stage": "profile_completion.required",
            "profile_completion_request": {
                "title": "生成学习计划前需要补充学习信息",
                "fields": [
                    {
                        "key": "learning_goal",
                        "label": "学习目标",
                        "required": True,
                        "max_chars": 400,
                    }
                ],
                "missing_required_keys": ["current_foundation"],
            },
        }

        recovered = _profile_completion_request_from_trace_event(event)
        assert recovered == {
            "title": "生成学习计划前需要补充学习信息",
            "fields": [
                {
                    "key": "learning_goal",
                    "label": "学习目标",
                    "required": True,
                    "max_chars": 400,
                }
            ],
        }

    @pytest.mark.anyio
    async def test_lost_interrupt_recovers_profile_completion_interrupt_from_trace(
        self,
    ):
        """When checkpoint interrupt is missing but trace carries a complete
        profile_completion_request, SSE should recover to profile completion interrupt."""
        from app import generate_stream_drafts

        async def trace_events():
            emit_a3_trace(
                logging.getLogger(__name__),
                "profile_completion.required",
                {
                    "profile_completion_request": {
                        "title": "Need profile before study plan",
                        "fields": [
                            {
                                "key": "learning_goal",
                                "label": "Learning goal",
                                "required": True,
                                "max_chars": 400,
                            }
                        ],
                        "missing_required_keys": ["current_foundation"],
                    }
                },
                state={},
                env_flag="LOG_GENERATION_SUMMARY",
            )
            emit_a3_trace(
                logging.getLogger(__name__),
                "resource_generation.worker.interrupted",
                {"resource_type": "study_plan", "elapsed_ms": 42},
                state={},
                env_flag="LOG_GENERATION_SUMMARY",
            )
            if False:
                yield {}

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(return_value=trace_events())
        mock_graph.aupdate_state = AsyncMock()
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(
                next=(),
                tasks=[],
                values={
                    "requested_resource_type": "study_plan",
                    "requested_resource_types": ["study_plan"],
                    "resource_generation_plan": {
                        "tasks": [{"resource_type": "study_plan"}]
                    },
                },
            )
        )

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph, thread_id="t-1"):
            collected.append(sse)

        all_payloads = _parse_all_payloads(collected)
        interrupt_events = [p for p in all_payloads if p.get("type") == "interrupt"]
        assert len(interrupt_events) == 1
        assert interrupt_events[0]["interrupt_type"] == "profile_completion_required"
        assert interrupt_events[0]["profile_completion_request"]["title"] == (
            "Need profile before study plan"
        )
        assert [
            p for p in all_payloads if p.get("error_type") == "interrupt_lost"
        ] == []
        assert [p for p in all_payloads if p.get("type") == "resource_final"] == []
        assert [
            p
            for p in all_payloads
            if p.get("type") == "resource_final_diagnostic"
            and p.get("status") == "completed_without_resource"
        ] == []
        assert [
            p
            for p in all_payloads
            if p.get("type") == "run_status" and p.get("run_status") == "completed"
        ] == []
        assert [p for p in all_payloads if p.get("type") == "stream_done"] == []

    @pytest.mark.anyio
    async def test_pending_checkpoint_without_interrupt_blocks_completion(self):
        """A non-empty snapshot.next without an interrupt is not completion."""
        from app import generate_stream_drafts

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(return_value=AsyncIteratorMock([]))
        mock_graph.aupdate_state = AsyncMock()
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(
                next=("evidence_judge", "study_plan_profile_gate_main"),
                tasks=[],
                values={"requested_resource_type": "study_plan"},
            )
        )

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph, thread_id="t-1"):
            collected.append(sse)

        payloads = _parse_all_payloads(collected)
        errors = [
            item
            for item in payloads
            if item.get("error_type") == "pending_checkpoint_without_interrupt"
        ]
        assert len(errors) == 1
        assert errors[0]["terminal_non_completed"] is True
        assert errors[0]["pending_nodes"] == [
            "evidence_judge",
            "study_plan_profile_gate_main",
        ]
        assert [item for item in payloads if item.get("type") == "resource_final"] == []
        assert [item for item in payloads if item.get("type") == "stream_done"] == []
        assert [
            item
            for item in payloads
            if item.get("type") == "run_status"
            and item.get("run_status") == "completed"
        ] == []

    @pytest.mark.anyio
    async def test_request_local_provisional_events_reach_sse_without_trace_logging(
        self,
    ):
        from app import generate_stream_drafts
        from src.streaming.provisional import emit_provisional_event

        async def events():
            emit_provisional_event(
                "qa_provisional_start",
                node_name="qa_agent",
                request_id="request-local",
                thread_id="t-1",
            )
            emit_provisional_event(
                "qa_provisional_delta",
                node_name="qa_agent",
                request_id="request-local",
                thread_id="t-1",
                delta="增量答案",
                answer_chars=4,
            )
            emit_provisional_event(
                "qa_provisional_stop",
                node_name="qa_agent",
                request_id="request-local",
                thread_id="t-1",
                answer_chars=4,
            )
            if False:
                yield {}

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(return_value=events())
        mock_graph.aupdate_state = AsyncMock()
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=(), tasks=[], values={})
        )

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph, thread_id="t-1"):
            collected.append(sse)

        payloads = _parse_all_payloads(collected)
        provisional = [
            item
            for item in payloads
            if item.get("type")
            in {"content_block_start", "content_block_delta", "content_block_stop"}
            and item.get("block_index") == 0
        ]
        assert [item["type"] for item in provisional] == [
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
        ]
        assert provisional[1]["delta"] == "增量答案"

    @pytest.mark.anyio
    async def test_plan_review_interrupt_empty_next_draft_is_raw_value(self):
        """plan_review with a raw string interrupt must emit draft as
        the raw value, not the normalized wrapper dict."""
        from app import generate_stream_drafts

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(return_value=AsyncIteratorMock([]))
        mock_graph.aupdate_state = AsyncMock()

        raw_text = "## Please confirm whether to continue"
        interrupt_obj = SimpleNamespace(value=raw_text)
        task = SimpleNamespace(interrupts=[interrupt_obj])
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=(), tasks=[task]),
        )

        collected = []
        async for sse in generate_stream_drafts("q", mock_graph, thread_id="t-1"):
            collected.append(sse)

        all_payloads = draft_payloads(collected)
        interrupt_events = [p for p in all_payloads if p.get("type") == "interrupt"]
        assert len(interrupt_events) == 1
        assert interrupt_events[0]["interrupt_type"] == "plan_review"
        # draft must be the raw string, not {"type": "plan_review", "value": ...}
        assert interrupt_events[0]["draft"] == raw_text
