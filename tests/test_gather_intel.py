"""Unit tests for the gather_intel node (REQ-07 Phase2a).

Tests cover:
- Parallel emotional + resource intel gathering
- Integration with TutorState
- Error handling (graceful degradation)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.planner import gather_intel


class TestGatherIntel:

    @patch("src.graph.planner.retrieve")
    @patch("src.graph.planner.web_search_fn")
    @patch("src.graph.planner.get_fallback_llm")
    @patch("src.graph.planner.get_node_llm")
    async def test_produces_all_intel_fields(
        self, mock_get_llm, mock_get_fallback, mock_web_search, mock_retrieve
    ):
        """gather_intel should produce emotional_intel, resource_intel, and intel_summary."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="学生情绪稳定，学习动力较强。")
        )
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()

        mock_retrieve.return_value = {
            "docs": [
                {"content": "高考数学重点：函数与导数", "source": "math.pdf", "score": 0.8},
            ],
            "is_hit": True,
        }
        mock_web_search.return_value = [
            {"content": "2026高考政策变化", "title": "政策", "url": "https://example.com"},
        ]

        state = {
            "messages": [HumanMessage(content="帮我制定数学复习计划")],
            "intent": "planning",
            "subject": "math",
            "keypoints": ["数学", "复习计划"],
            "context": [],
            "search_results": [],
            "plan": "",
            "retry_count": 0,
            "hallucination_detected": False,
            "rewritten_query": "",
            "hallucination_reason": "",
            "emotional_intel": "",
            "resource_intel": "",
            "intel_summary": "",
        }
        result = await gather_intel(state)

        assert "emotional_intel" in result
        assert "resource_intel" in result
        assert "intel_summary" in result
        assert len(result["emotional_intel"]) > 0
        assert len(result["resource_intel"]) > 0
        assert len(result["intel_summary"]) > 0
        # Adversarial init fields (AC-01 Step 3)
        assert result["adv_round"] == 0
        assert result["draft"] == ""
        assert result["academic_verdict"] == ""
        assert result["academic_reason"] == ""
        assert result["emotional_verdict"] == ""
        assert result["emotional_reason"] == ""
        assert result["consensus"] is False
        assert result["revision_notes"] == ""

    @patch("src.graph.planner.retrieve")
    @patch("src.graph.planner.web_search_fn")
    @patch("src.graph.planner.get_fallback_llm")
    @patch("src.graph.planner.get_node_llm")
    async def test_emotional_intel_from_llm(
        self, mock_get_llm, mock_get_fallback, mock_web_search, mock_retrieve
    ):
        """emotional_intel should come from LLM analysis of conversation history."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="学生表现出考前焦虑。")
        )
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()
        mock_retrieve.return_value = {"docs": [], "is_hit": False}
        mock_web_search.return_value = []

        state = {
            "messages": [
                HumanMessage(content="我好焦虑，数学总是考不好"),
                AIMessage(content="我理解你的心情"),
                HumanMessage(content="帮我制定复习计划吧"),
            ],
            "intent": "planning",
            "subject": "",
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
            "intel_summary": "",
        }
        result = await gather_intel(state)

        assert "焦虑" in result["emotional_intel"]
        # Adversarial init fields
        assert result["adv_round"] == 0
        assert result["consensus"] is False

    @patch("src.graph.planner.retrieve")
    @patch("src.graph.planner.web_search_fn")
    @patch("src.graph.planner.get_fallback_llm")
    @patch("src.graph.planner.get_node_llm")
    async def test_resource_intel_from_rag_and_web(
        self, mock_get_llm, mock_get_fallback, mock_web_search, mock_retrieve
    ):
        """resource_intel should combine RAG and web search results."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="情绪正常"))
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()

        mock_retrieve.return_value = {
            "docs": [
                {"content": "导数是重点", "source": "math.pdf", "score": 0.9},
            ],
            "is_hit": True,
        }
        mock_web_search.return_value = [
            {"content": "2026高考6月7日", "title": "高考日期", "url": "https://example.com"},
        ]

        state = {
            "messages": [HumanMessage(content="帮我做复习计划")],
            "intent": "planning",
            "subject": "math",
            "keypoints": ["复习计划"],
            "context": [],
            "search_results": [],
            "plan": "",
            "retry_count": 0,
            "hallucination_detected": False,
            "rewritten_query": "",
            "hallucination_reason": "",
            "emotional_intel": "",
            "resource_intel": "",
            "intel_summary": "",
        }
        result = await gather_intel(state)

        assert "导数" in result["resource_intel"]
        assert "2026" in result["resource_intel"]
        # Adversarial init fields
        assert result["adv_round"] == 0
        assert result["draft"] == ""
        assert result["consensus"] is False

    @patch("src.graph.planner.retrieve", side_effect=Exception("chromadb down"))
    @patch("src.graph.planner.web_search_fn", side_effect=Exception("network error"))
    @patch("src.graph.planner.get_fallback_llm")
    @patch("src.graph.planner.get_node_llm")
    async def test_graceful_degradation_on_resource_errors(
        self, mock_get_llm, mock_get_fallback, mock_web_search, mock_retrieve
    ):
        """When both RAG and web search fail, resource_intel should degrade gracefully."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="情绪正常"))
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()

        state = {
            "messages": [HumanMessage(content="帮我做计划")],
            "intent": "planning",
            "subject": "",
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
            "intel_summary": "",
        }
        result = await gather_intel(state)

        # Should not raise; resource_intel should be a fallback string
        assert "emotional_intel" in result
        assert "resource_intel" in result
        assert "intel_summary" in result
        # Adversarial init fields
        assert result["adv_round"] == 0
        assert result["consensus"] is False

    @patch("src.graph.planner.retrieve")
    @patch("src.graph.planner.web_search_fn")
    @patch("src.graph.planner.get_fallback_llm")
    @patch("src.graph.planner.get_node_llm")
    async def test_emotional_llm_failure_degrades_gracefully(
        self, mock_get_llm, mock_get_fallback, mock_web_search, mock_retrieve
    ):
        """When emotional LLM call fails, should return a fallback emotional_intel."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM down"))
        mock_get_llm.return_value = mock_llm

        mock_fallback = MagicMock()
        mock_fallback.ainvoke = AsyncMock(side_effect=Exception("fallback also down"))
        mock_get_fallback.return_value = mock_fallback

        mock_retrieve.return_value = {"docs": [], "is_hit": False}
        mock_web_search.return_value = []

        state = {
            "messages": [HumanMessage(content="帮我做计划")],
            "intent": "planning",
            "subject": "",
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
            "intel_summary": "",
        }
        result = await gather_intel(state)

        assert "emotional_intel" in result
        assert "intel_summary" in result
        # Adversarial init fields
        assert result["adv_round"] == 0
        assert result["consensus"] is False
