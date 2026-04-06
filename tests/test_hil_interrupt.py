"""Unit tests for Human-in-the-loop interrupt/resume (REQ-08).

Tests cover:
- plan_adversarial_node calls interrupt() and respects resumed value
- generate_sse emits interrupt SSE event when graph is interrupted
- generate_sse emits thread_id SSE event at start
- generate_resume_sse resumes execution and streams remaining events
- POST /resume endpoint wiring
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage


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


# ---------------------------------------------------------------------------
# plan_adversarial_node — interrupt / resume
# ---------------------------------------------------------------------------


class TestPlanOutputNodeInterrupt:
    """Test that plan_output_node calls interrupt() and uses resumed value."""

    @patch("src.graph.plan_adversarial.interrupt")
    async def test_calls_interrupt_with_draft(self, mock_interrupt):
        """plan_output_node calls interrupt() with the draft text."""
        mock_interrupt.return_value = "用户编辑后的计划"

        from src.graph.plan_adversarial import plan_output_node

        state = {
            "messages": [HumanMessage(content="帮我做复习计划")],
            "intent": "planning",
            "subject": "math",
            "keypoints": [],
            "context": [],
            "search_results": [],
            "plan": "",
            "retry_count": 0,
            "hallucination_detected": False,
            "rewritten_query": "",
            "hallucination_reason": "",
            "emotional_intel": "",
            "resource_intel": "",
            "intel_summary": "情报摘要",
            "draft": "## 原始计划",
            "academic_verdict": "approve",
            "academic_reason": "",
            "emotional_verdict": "approve",
            "emotional_reason": "",
            "adv_round": 1,
            "consensus": True,
            "revision_notes": "",
        }
        result = await plan_output_node(state)

        mock_interrupt.assert_called_once_with("## 原始计划")

    @patch("src.graph.plan_adversarial.interrupt")
    async def test_uses_resumed_plan(self, mock_interrupt):
        """When interrupt() returns user's edited plan, node uses it."""
        edited_plan = "## 用户修改后的计划\n- 周一：数学（减轻强度）"
        mock_interrupt.return_value = edited_plan

        from src.graph.plan_adversarial import plan_output_node

        state = {
            "messages": [HumanMessage(content="帮我做复习计划")],
            "intent": "planning",
            "subject": "math",
            "keypoints": [],
            "context": [],
            "search_results": [],
            "plan": "",
            "retry_count": 0,
            "hallucination_detected": False,
            "rewritten_query": "",
            "hallucination_reason": "",
            "emotional_intel": "",
            "resource_intel": "",
            "intel_summary": "情报摘要",
            "draft": "## 原始计划",
            "academic_verdict": "approve",
            "academic_reason": "",
            "emotional_verdict": "approve",
            "emotional_reason": "",
            "adv_round": 1,
            "consensus": True,
            "revision_notes": "",
        }
        result = await plan_output_node(state)

        assert result["plan"] == edited_plan
        assert result["messages"][0].content == edited_plan

    @patch("src.graph.plan_adversarial.interrupt")
    async def test_falls_back_to_original_when_resume_empty(self, mock_interrupt):
        """If interrupt() returns empty/None, use original draft."""
        mock_interrupt.return_value = None

        from src.graph.plan_adversarial import plan_output_node

        state = {
            "messages": [HumanMessage(content="帮我做复习计划")],
            "intent": "planning",
            "subject": "math",
            "keypoints": [],
            "context": [],
            "search_results": [],
            "plan": "",
            "retry_count": 0,
            "hallucination_detected": False,
            "rewritten_query": "",
            "hallucination_reason": "",
            "emotional_intel": "",
            "resource_intel": "",
            "intel_summary": "情报摘要",
            "draft": "## 原始计划",
            "academic_verdict": "approve",
            "academic_reason": "",
            "emotional_verdict": "approve",
            "emotional_reason": "",
            "adv_round": 1,
            "consensus": True,
            "revision_notes": "",
        }
        result = await plan_output_node(state)

        assert result["plan"] == "## 原始计划"


# ---------------------------------------------------------------------------
# Full interrupt/resume integration with LangGraph MemorySaver
# ---------------------------------------------------------------------------


class TestHILIntegration:
    """Integration test: graph hits interrupt, state persisted, resume works."""

    async def test_interrupt_persists_and_resume_completes(self):
        """Graph reaches interrupt → state saved → Command(resume=...) completes."""
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, StateGraph
        from langgraph.types import Command, interrupt
        from typing_extensions import TypedDict

        class MiniState(TypedDict):
            plan: str

        async def plan_node(state):
            edited = interrupt(state["plan"])
            return {"plan": edited if isinstance(edited, str) and edited else state["plan"]}

        g = StateGraph(MiniState)
        g.add_node("plan", plan_node)
        g.set_entry_point("plan")
        g.add_edge("plan", END)
        app = g.compile(checkpointer=MemorySaver())

        cfg = {"configurable": {"thread_id": "hil-test-1"}}

        # First invocation — hits interrupt
        result = await app.ainvoke({"plan": "原始计划"}, config=cfg)
        snap = await app.aget_state(cfg)
        assert snap.next == ("plan",)
        assert len(snap.tasks) > 0
        assert snap.tasks[0].interrupts[0].value == "原始计划"

        # Resume with edited plan
        result2 = await app.ainvoke(Command(resume="修改后的计划"), config=cfg)
        assert result2["plan"] == "修改后的计划"

        # Graph completed
        snap2 = await app.aget_state(cfg)
        assert snap2.next == ()


# ---------------------------------------------------------------------------
# generate_sse — thread_id and interrupt events
# ---------------------------------------------------------------------------


class TestSSEThreadIdEvent:
    """generate_sse should emit thread_id at the start of the stream."""

    @pytest.mark.anyio
    async def test_emits_thread_id_event(self):
        from app import generate_sse

        events = []  # empty graph events
        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock(events),
        )
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=(), tasks=[]),
        )

        collected = []
        async for sse in generate_sse("hello", mock_graph, thread_id="tid-123"):
            collected.append(sse)

        # First event should be thread_id
        assert len(collected) >= 1
        data = json.loads(collected[0].removeprefix("data: ").strip())
        assert data["type"] == "thread_id"
        assert data["thread_id"] == "tid-123"

    @pytest.mark.anyio
    async def test_generates_thread_id_when_none(self):
        from app import generate_sse

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock([]),
        )
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=(), tasks=[]),
        )

        collected = []
        async for sse in generate_sse("hello", mock_graph, thread_id=None):
            collected.append(sse)

        data = json.loads(collected[0].removeprefix("data: ").strip())
        assert data["type"] == "thread_id"
        assert len(data["thread_id"]) > 0  # auto-generated UUID


class TestSSEInterruptEvent:
    """generate_sse should emit interrupt event when graph is interrupted."""

    @pytest.mark.anyio
    async def test_emits_interrupt_event(self):
        from app import generate_sse

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock([]),
        )

        # Simulate interrupted state
        interrupt_obj = SimpleNamespace(value="## 学习计划草稿")
        task = SimpleNamespace(interrupts=[interrupt_obj])
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=("plan_adversarial",), tasks=[task]),
        )

        collected = []
        async for sse in generate_sse("做计划", mock_graph, thread_id="t-42"):
            collected.append(sse)

        # Find interrupt event (after thread_id event)
        payloads = [json.loads(s.removeprefix("data: ").strip()) for s in collected]
        interrupt_events = [p for p in payloads if p["type"] == "interrupt"]
        assert len(interrupt_events) == 1
        assert interrupt_events[0]["draft"] == "## 学习计划草稿"
        assert interrupt_events[0]["thread_id"] == "t-42"

    @pytest.mark.anyio
    async def test_no_interrupt_event_on_normal_completion(self):
        from app import generate_sse

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock([]),
        )
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=(), tasks=[]),
        )

        collected = []
        async for sse in generate_sse("hi", mock_graph, thread_id="t-99"):
            collected.append(sse)

        payloads = [json.loads(s.removeprefix("data: ").strip()) for s in collected]
        interrupt_events = [p for p in payloads if p.get("type") == "interrupt"]
        assert len(interrupt_events) == 0


# ---------------------------------------------------------------------------
# generate_resume_sse
# ---------------------------------------------------------------------------


class TestResumeSSE:
    """generate_resume_sse should resume graph and stream remaining events."""

    @pytest.mark.anyio
    async def test_resumes_and_streams(self):
        from app import generate_resume_sse

        def _node_event(name, status):
            return {
                "event": f"on_chain_{status}",
                "name": name,
                "metadata": {"langgraph_node": name},
                "data": {"output": {}},
            }

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock([
                _node_event("plan_output", "end"),
            ]),
        )
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=(), tasks=[]),
        )

        collected = []
        async for sse in generate_resume_sse("修改后计划", None, mock_graph, "t-42"):
            collected.append(sse)

        # Should have at least the node_end event
        payloads = [json.loads(s.removeprefix("data: ").strip()) for s in collected]
        node_events = [p for p in payloads if p.get("type") == "node_event"]
        assert len(node_events) >= 1

    @pytest.mark.anyio
    async def test_resume_uses_command(self):
        """Verify graph.astream_events is called with Command(resume=...) for confirm."""
        from app import generate_resume_sse
        from langgraph.types import Command

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock([]),
        )
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=(), tasks=[]),
        )

        async for _ in generate_resume_sse("edited", None, mock_graph, "t-1"):
            pass

        call_args = mock_graph.astream_events.call_args
        first_arg = call_args[0][0]
        assert isinstance(first_arg, Command)
        assert first_arg.resume == "edited"

    @pytest.mark.anyio
    async def test_resume_with_feedback_sends_dict(self):
        """When feedback is provided, Command(resume=dict) is constructed."""
        from app import generate_resume_sse
        from langgraph.types import Command

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock([]),
        )
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=(), tasks=[]),
        )

        async for _ in generate_resume_sse("", "把周三改成物理", mock_graph, "t-1"):
            pass

        call_args = mock_graph.astream_events.call_args
        first_arg = call_args[0][0]
        assert isinstance(first_arg, Command)
        assert first_arg.resume == {"action": "feedback", "text": "把周三改成物理"}

    @pytest.mark.anyio
    async def test_resume_without_feedback_sends_string(self):
        """When feedback is None, Command(resume=string) is constructed (backward compat)."""
        from app import generate_resume_sse
        from langgraph.types import Command

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock([]),
        )
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=(), tasks=[]),
        )

        async for _ in generate_resume_sse("edited plan", None, mock_graph, "t-1"):
            pass

        call_args = mock_graph.astream_events.call_args
        first_arg = call_args[0][0]
        assert isinstance(first_arg, Command)
        assert first_arg.resume == "edited plan"


# ---------------------------------------------------------------------------
# POST /resume endpoint
# ---------------------------------------------------------------------------


class TestResumeEndpoint:
    """Tests for the /resume FastAPI endpoint."""

    def test_resume_request_schema(self):
        from src.schemas import ResumeRequest

        req = ResumeRequest(thread_id="abc-123", edited_plan="## 修改计划")
        assert req.thread_id == "abc-123"
        assert req.edited_plan == "## 修改计划"

    def test_resume_request_requires_thread_id(self):
        from src.schemas import ResumeRequest

        with pytest.raises(Exception):
            ResumeRequest(edited_plan="test")

    def test_resume_request_edited_plan_defaults_empty(self):
        """edited_plan now defaults to '' (not required)."""
        from src.schemas import ResumeRequest

        req = ResumeRequest(thread_id="abc")
        assert req.edited_plan == ""

    def test_resume_request_feedback_defaults_none(self):
        from src.schemas import ResumeRequest

        req = ResumeRequest(thread_id="abc")
        assert req.feedback is None

    def test_resume_request_feedback_accepts_value(self):
        from src.schemas import ResumeRequest

        req = ResumeRequest(thread_id="abc", feedback="改一下时间")
        assert req.feedback == "改一下时间"

    def test_resume_request_feedback_max_length(self):
        from pydantic import ValidationError
        from src.schemas import ResumeRequest

        with pytest.raises(ValidationError):
            ResumeRequest(thread_id="abc", feedback="x" * 5000)

    @pytest.mark.anyio
    async def test_resume_endpoint_exists(self):
        """POST /resume should exist in the app."""
        from fastapi.testclient import TestClient

        from app import app

        routes = [r.path for r in app.routes]
        assert "/resume" in routes
