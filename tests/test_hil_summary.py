"""Unit tests for hil_summary behavior in feedback_router.

Tests cover:
- Summary creation from empty state
- Summary compression with existing history
- Summary truncation (bounded growth)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.graph.plan_adversarial import FeedbackClassification, feedback_router
from src.graph.state import TutorState


def _base_state(**overrides) -> TutorState:
    """Return a TutorState dict with sensible defaults, overridden by kwargs."""
    state: TutorState = {
        "messages": [HumanMessage(content="帮我做课程补基础计划")],
        "intent": "planning",
        "subject": "math",
        "keypoints": ["高等数学", "课程补基础计划"],
        "context": [],
        "search_results": [],
        "plan": "",
        "retry_count": 0,
        "hallucination_detected": False,
        "rewritten_query": "",
        "hallucination_reason": "",
        "emotional_intel": "",
        "resource_intel": "",
        "intel_summary": "学生数学薄弱",
        "draft": "## 每日学习计划\n- 周一: 高等数学\n- 周二: Python 基础",
        "academic_verdict": "",
        "academic_reason": "",
        "emotional_verdict": "",
        "emotional_reason": "",
        "adv_round": 0,
        "consensus": False,
        "revision_notes": "",
        "hil_action": "",
        "hil_feedback": "",
        "hil_summary": "",
        "feedback_route": "",
    }
    state.update(overrides)
    return state


class TestHilSummaryEmpty:
    """Test feedback_router with empty hil_summary."""

    @patch("src.graph.plan_adversarial.get_node_llm")
    async def test_creates_summary_from_feedback(self, mock_get_llm):
        mock_llm = MagicMock()
        structured = MagicMock()
        structured.ainvoke = AsyncMock(
            return_value=FeedbackClassification(route="tweak", reason="局部修改")
        )
        mock_llm.with_structured_output.return_value = structured
        mock_get_llm.return_value = mock_llm

        state = _base_state(
            hil_feedback="把周三改成物理",
            hil_summary="",
        )
        result = await feedback_router(state)

        assert result["hil_summary"] == "用户反馈: 把周三改成物理"


class TestHilSummaryExisting:
    """Test feedback_router with existing hil_summary."""

    @patch("src.graph.plan_adversarial.get_node_llm")
    async def test_compresses_old_summary_with_new_feedback(self, mock_get_llm):
        mock_llm = MagicMock()
        structured = MagicMock()
        structured.ainvoke = AsyncMock(
            return_value=FeedbackClassification(route="tweak", reason="局部修改")
        )
        mock_llm.with_structured_output.return_value = structured
        mock_get_llm.return_value = mock_llm

        state = _base_state(
            hil_feedback="再把周五改成化学",
            hil_summary="用户反馈: 把周三改成物理",
        )
        result = await feedback_router(state)

        assert "历史修改摘要" in result["hil_summary"]
        assert "把周三改成物理" in result["hil_summary"]
        assert "再把周五改成化学" in result["hil_summary"]


class TestHilSummaryTruncation:
    """Test that hil_summary doesn't grow unbounded."""

    @patch("src.graph.plan_adversarial.get_node_llm")
    async def test_long_summary_is_truncated(self, mock_get_llm):
        mock_llm = MagicMock()
        structured = MagicMock()
        structured.ainvoke = AsyncMock(
            return_value=FeedbackClassification(route="tweak", reason="局部修改")
        )
        mock_llm.with_structured_output.return_value = structured
        mock_get_llm.return_value = mock_llm

        long_summary = "这是一段很长的摘要。" * 100  # ~900 chars
        state = _base_state(
            hil_feedback="新反馈",
            hil_summary=long_summary,
        )
        result = await feedback_router(state)

        # Old summary truncated to 200 chars, feedback to 500
        # Total should be bounded
        assert len(result["hil_summary"]) < len(long_summary)

    @patch("src.graph.plan_adversarial.get_node_llm")
    async def test_long_feedback_is_truncated(self, mock_get_llm):
        mock_llm = MagicMock()
        structured = MagicMock()
        structured.ainvoke = AsyncMock(
            return_value=FeedbackClassification(route="tweak", reason="局部修改")
        )
        mock_llm.with_structured_output.return_value = structured
        mock_get_llm.return_value = mock_llm

        long_feedback = "反馈内容" * 200  # 800 chars
        state = _base_state(
            hil_feedback=long_feedback,
            hil_summary="",
        )
        result = await feedback_router(state)

        # Feedback truncated to 500 chars in summary
        assert len(result["hil_summary"]) <= len("用户反馈: ") + 500
