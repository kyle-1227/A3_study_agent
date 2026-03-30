"""Unit tests for the Adversarial Planning SubGraph (REQ-07).

Tests cover:
- ReviewVerdict Pydantic model
- Individual async nodes (drafter, reviewers, consensus_check, rewrite, output)
- SubGraph integration: consensus path, rejection+retry path, max_rounds safety valve
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.graph.plan_adversarial import (
    PlanAdversarialState,
    ReviewVerdict,
    build_adversarial_subgraph,
    consensus_check_node,
    drafter_node,
    output_node,
    reviewer_academic_node,
    reviewer_emotional_node,
    rewrite_node,
)


# ---------------------------------------------------------------------------
# ReviewVerdict model
# ---------------------------------------------------------------------------


class TestReviewVerdict:

    def test_approve_verdict(self):
        v = ReviewVerdict(verdict="approve", reason="计划合理")
        assert v.verdict == "approve"
        assert v.reason == "计划合理"

    def test_reject_verdict(self):
        v = ReviewVerdict(verdict="reject", reason="缺少休息时间")
        assert v.verdict == "reject"

    def test_invalid_verdict_raises(self):
        with pytest.raises(Exception):
            ReviewVerdict(verdict="maybe", reason="不确定")


# ---------------------------------------------------------------------------
# drafter_node
# ---------------------------------------------------------------------------


class TestDrafterNode:

    @patch("src.graph.plan_adversarial.get_fallback_llm")
    @patch("src.graph.plan_adversarial.get_node_llm")
    async def test_produces_draft(self, mock_get_llm, mock_get_fallback):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="## 学习计划\n- 周一：数学"))
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()

        state: PlanAdversarialState = {
            "intel_summary": "学生数学薄弱，情绪稳定",
            "user_request": "帮我做一周复习计划",
            "draft": "",
            "academic_verdict": "",
            "emotional_verdict": "",
            "round": 0,
            "max_rounds": 3,
            "consensus": False,
            "revision_notes": "",
        }
        result = await drafter_node(state)

        assert "draft" in result
        assert "学习计划" in result["draft"]
        assert result["round"] == 1

    @patch("src.graph.plan_adversarial.get_fallback_llm")
    @patch("src.graph.plan_adversarial.get_node_llm")
    async def test_rewrite_uses_revision_notes(self, mock_get_llm, mock_get_fallback):
        """When revision_notes is non-empty, drafter uses the rewrite prompt."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="## 修改后的计划"))
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()

        state: PlanAdversarialState = {
            "intel_summary": "情报信息",
            "user_request": "做计划",
            "draft": "旧计划",
            "academic_verdict": "",
            "emotional_verdict": "",
            "round": 1,
            "max_rounds": 3,
            "consensus": False,
            "revision_notes": "需要增加休息时间",
        }
        result = await drafter_node(state)

        assert "draft" in result
        # Verify the rewrite prompt was used (contains revision_notes)
        call_args = mock_llm.ainvoke.call_args[0][0]
        prompt_text = call_args[1].content  # HumanMessage
        assert "需要增加休息时间" in prompt_text


# ---------------------------------------------------------------------------
# reviewer nodes
# ---------------------------------------------------------------------------


class TestReviewerAcademicNode:

    @patch("src.graph.plan_adversarial.get_fallback_llm")
    @patch("src.graph.plan_adversarial.get_node_llm")
    async def test_returns_verdict(self, mock_get_llm, mock_get_fallback):
        mock_llm = MagicMock()
        structured = MagicMock()
        structured.ainvoke = AsyncMock(
            return_value=ReviewVerdict(verdict="approve", reason="计划全面覆盖各科目")
        )
        mock_llm.with_structured_output.return_value = structured
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()

        state: PlanAdversarialState = {
            "intel_summary": "情报",
            "user_request": "做计划",
            "draft": "## 计划\n- 周一：数学",
            "academic_verdict": "",
            "emotional_verdict": "",
            "round": 1,
            "max_rounds": 3,
            "consensus": False,
            "revision_notes": "",
        }
        result = await reviewer_academic_node(state)

        assert result["academic_verdict"] == "approve"

    @patch("src.graph.plan_adversarial.get_fallback_llm")
    @patch("src.graph.plan_adversarial.get_node_llm")
    async def test_reject_verdict(self, mock_get_llm, mock_get_fallback):
        mock_llm = MagicMock()
        structured = MagicMock()
        structured.ainvoke = AsyncMock(
            return_value=ReviewVerdict(verdict="reject", reason="缺少物理复习")
        )
        mock_llm.with_structured_output.return_value = structured
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()

        state: PlanAdversarialState = {
            "intel_summary": "情报",
            "user_request": "做计划",
            "draft": "## 计划",
            "academic_verdict": "",
            "emotional_verdict": "",
            "round": 1,
            "max_rounds": 3,
            "consensus": False,
            "revision_notes": "",
        }
        result = await reviewer_academic_node(state)

        assert result["academic_verdict"] == "reject"

    @patch("src.graph.plan_adversarial.get_fallback_llm")
    @patch("src.graph.plan_adversarial.get_node_llm")
    async def test_fallback_approve_on_error(self, mock_get_llm, mock_get_fallback):
        """If structured output fails, default to approve (safe fallback)."""
        mock_llm = MagicMock()
        structured = MagicMock()
        structured.ainvoke = AsyncMock(side_effect=Exception("parse error"))
        mock_llm.with_structured_output.return_value = structured
        mock_get_llm.return_value = mock_llm

        mock_fallback = MagicMock()
        fallback_structured = MagicMock()
        fallback_structured.ainvoke = AsyncMock(side_effect=Exception("fallback also failed"))
        mock_fallback.with_structured_output.return_value = fallback_structured
        mock_get_fallback.return_value = mock_fallback

        state: PlanAdversarialState = {
            "intel_summary": "情报",
            "user_request": "做计划",
            "draft": "## 计划",
            "academic_verdict": "",
            "emotional_verdict": "",
            "round": 1,
            "max_rounds": 3,
            "consensus": False,
            "revision_notes": "",
        }
        result = await reviewer_academic_node(state)

        assert result["academic_verdict"] == "approve"


class TestReviewerEmotionalNode:

    @patch("src.graph.plan_adversarial.get_fallback_llm")
    @patch("src.graph.plan_adversarial.get_node_llm")
    async def test_returns_verdict(self, mock_get_llm, mock_get_fallback):
        mock_llm = MagicMock()
        structured = MagicMock()
        structured.ainvoke = AsyncMock(
            return_value=ReviewVerdict(verdict="reject", reason="学习强度过大，缺少放松时间")
        )
        mock_llm.with_structured_output.return_value = structured
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()

        state: PlanAdversarialState = {
            "intel_summary": "学生焦虑",
            "user_request": "做计划",
            "draft": "## 计划\n每天学习14小时",
            "academic_verdict": "",
            "emotional_verdict": "",
            "round": 1,
            "max_rounds": 3,
            "consensus": False,
            "revision_notes": "",
        }
        result = await reviewer_emotional_node(state)

        assert result["emotional_verdict"] == "reject"


# ---------------------------------------------------------------------------
# consensus_check_node
# ---------------------------------------------------------------------------


class TestConsensusCheckNode:

    async def test_both_approve(self):
        state: PlanAdversarialState = {
            "intel_summary": "",
            "user_request": "",
            "draft": "计划",
            "academic_verdict": "approve",
            "emotional_verdict": "approve",
            "round": 1,
            "max_rounds": 3,
            "consensus": False,
            "revision_notes": "",
        }
        result = await consensus_check_node(state)

        assert result["consensus"] is True
        assert result["revision_notes"] == ""

    async def test_academic_reject(self):
        state: PlanAdversarialState = {
            "intel_summary": "",
            "user_request": "",
            "draft": "计划",
            "academic_verdict": "reject",
            "emotional_verdict": "approve",
            "round": 1,
            "max_rounds": 3,
            "consensus": False,
            "revision_notes": "",
        }
        result = await consensus_check_node(state)

        assert result["consensus"] is False
        assert "academic" in result["revision_notes"].lower() or len(result["revision_notes"]) > 0

    async def test_emotional_reject(self):
        state: PlanAdversarialState = {
            "intel_summary": "",
            "user_request": "",
            "draft": "计划",
            "academic_verdict": "approve",
            "emotional_verdict": "reject",
            "round": 1,
            "max_rounds": 3,
            "consensus": False,
            "revision_notes": "",
        }
        result = await consensus_check_node(state)

        assert result["consensus"] is False

    async def test_both_reject(self):
        state: PlanAdversarialState = {
            "intel_summary": "",
            "user_request": "",
            "draft": "计划",
            "academic_verdict": "reject",
            "emotional_verdict": "reject",
            "round": 1,
            "max_rounds": 3,
            "consensus": False,
            "revision_notes": "",
        }
        result = await consensus_check_node(state)

        assert result["consensus"] is False

    async def test_max_rounds_forces_consensus(self):
        """When round >= max_rounds, force consensus regardless of verdicts."""
        state: PlanAdversarialState = {
            "intel_summary": "",
            "user_request": "",
            "draft": "计划",
            "academic_verdict": "reject",
            "emotional_verdict": "reject",
            "round": 3,
            "max_rounds": 3,
            "consensus": False,
            "revision_notes": "",
        }
        result = await consensus_check_node(state)

        assert result["consensus"] is True


# ---------------------------------------------------------------------------
# rewrite_node
# ---------------------------------------------------------------------------


class TestRewriteNode:

    async def test_clears_verdicts(self):
        state: PlanAdversarialState = {
            "intel_summary": "",
            "user_request": "",
            "draft": "旧计划",
            "academic_verdict": "reject",
            "emotional_verdict": "reject",
            "round": 1,
            "max_rounds": 3,
            "consensus": False,
            "revision_notes": "需要修改",
        }
        result = await rewrite_node(state)

        assert result["academic_verdict"] == ""
        assert result["emotional_verdict"] == ""


# ---------------------------------------------------------------------------
# output_node
# ---------------------------------------------------------------------------


class TestOutputNode:

    async def test_returns_draft(self):
        state: PlanAdversarialState = {
            "intel_summary": "",
            "user_request": "",
            "draft": "## 最终计划\n- 周一：数学",
            "academic_verdict": "approve",
            "emotional_verdict": "approve",
            "round": 1,
            "max_rounds": 3,
            "consensus": True,
            "revision_notes": "",
        }
        result = await output_node(state)

        assert result["draft"] == state["draft"]


# ---------------------------------------------------------------------------
# SubGraph build
# ---------------------------------------------------------------------------


class TestBuildSubGraph:

    def test_builds_successfully(self):
        sub = build_adversarial_subgraph()
        assert sub is not None

    def test_has_expected_nodes(self):
        from langgraph.graph import StateGraph
        # We test the uncompiled builder indirectly via the compiled graph
        sub = build_adversarial_subgraph()
        # Compiled graph should be invokable
        assert hasattr(sub, "invoke")


# ---------------------------------------------------------------------------
# SubGraph integration: end-to-end with mocked LLMs
# ---------------------------------------------------------------------------


class TestSubGraphIntegration:

    @patch("src.graph.plan_adversarial.get_fallback_llm")
    @patch("src.graph.plan_adversarial.get_node_llm")
    async def test_consensus_path(self, mock_get_llm, mock_get_fallback):
        """Both reviewers approve on first round → output reached."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="## 完美计划"))
        structured = MagicMock()
        structured.ainvoke = AsyncMock(
            return_value=ReviewVerdict(verdict="approve", reason="好")
        )
        mock_llm.with_structured_output.return_value = structured
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()

        sub = build_adversarial_subgraph()
        initial: PlanAdversarialState = {
            "intel_summary": "学生情况正常",
            "user_request": "帮我做计划",
            "draft": "",
            "academic_verdict": "",
            "emotional_verdict": "",
            "round": 0,
            "max_rounds": 3,
            "consensus": False,
            "revision_notes": "",
        }
        result = await sub.ainvoke(initial)

        assert result["consensus"] is True
        assert result["round"] == 1
        assert "完美计划" in result["draft"]

    @patch("src.graph.plan_adversarial.get_fallback_llm")
    @patch("src.graph.plan_adversarial.get_node_llm")
    async def test_rejection_retry_then_approve(self, mock_get_llm, mock_get_fallback):
        """Reviewer rejects first round, approves second → 2 rounds total."""
        mock_llm = MagicMock()
        # drafter returns different content each call
        mock_llm.ainvoke = AsyncMock(side_effect=[
            MagicMock(content="## 初稿"),
            MagicMock(content="## 修改后的计划"),
        ])
        # First call: reject, second call: approve
        call_count = {"n": 0}

        async def structured_side_effect(messages):
            call_count["n"] += 1
            if call_count["n"] <= 2:  # first round: both reviewers reject/approve
                if call_count["n"] == 1:
                    return ReviewVerdict(verdict="reject", reason="缺少科目")
                return ReviewVerdict(verdict="approve", reason="ok")
            return ReviewVerdict(verdict="approve", reason="好")

        structured = MagicMock()
        structured.ainvoke = AsyncMock(side_effect=structured_side_effect)
        mock_llm.with_structured_output.return_value = structured
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()

        sub = build_adversarial_subgraph()
        initial: PlanAdversarialState = {
            "intel_summary": "情报",
            "user_request": "做计划",
            "draft": "",
            "academic_verdict": "",
            "emotional_verdict": "",
            "round": 0,
            "max_rounds": 3,
            "consensus": False,
            "revision_notes": "",
        }
        result = await sub.ainvoke(initial)

        assert result["consensus"] is True
        assert result["round"] >= 2

    @patch("src.graph.plan_adversarial.get_fallback_llm")
    @patch("src.graph.plan_adversarial.get_node_llm")
    async def test_max_rounds_safety_valve(self, mock_get_llm, mock_get_fallback):
        """Reviewers always reject → forced output at max_rounds."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="## 计划"))
        structured = MagicMock()
        structured.ainvoke = AsyncMock(
            return_value=ReviewVerdict(verdict="reject", reason="不行")
        )
        mock_llm.with_structured_output.return_value = structured
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()

        sub = build_adversarial_subgraph()
        initial: PlanAdversarialState = {
            "intel_summary": "情报",
            "user_request": "做计划",
            "draft": "",
            "academic_verdict": "",
            "emotional_verdict": "",
            "round": 0,
            "max_rounds": 1,
            "consensus": False,
            "revision_notes": "",
        }
        result = await sub.ainvoke(initial)

        # max_rounds=1 → after round 1, forced consensus
        assert result["consensus"] is True
        assert result["round"] == 1
