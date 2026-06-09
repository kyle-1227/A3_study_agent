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

from src.graph.planner import _build_planning_retrieval_query, _gather_resource_intel, gather_intel


class TestGatherIntel:
    def test_build_planning_retrieval_query_priority(self):
        state = {
            "messages": [HumanMessage(content="原始问题")],
            "search_rag_query": "bilingual rag query",
            "expanded_keypoints": ["扩展", "expanded"],
            "keypoints": ["原始关键词"],
        }
        assert _build_planning_retrieval_query(state) == "bilingual rag query"

        state["search_rag_query"] = ""
        assert _build_planning_retrieval_query(state) == "扩展 expanded"

        state["expanded_keypoints"] = []
        assert _build_planning_retrieval_query(state) == "原始关键词"

        state["keypoints"] = []
        assert _build_planning_retrieval_query(state) == "原始问题"

    @patch("src.graph.planner.retrieve")
    @patch("src.graph.planner.web_search_fn")
    @patch("src.graph.planner.get_fallback_llm")
    @patch("src.graph.planner.get_node_llm")
    async def test_resource_intel_uses_rewritten_planning_queries(
        self, mock_get_llm, mock_get_fallback, mock_web_search, mock_retrieve
    ):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="情绪正常"))
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()
        mock_retrieve.return_value = {"docs": [], "is_hit": False}
        mock_web_search.return_value = []

        state = {
            "messages": [HumanMessage(content="帮我制定 Python 学习计划")],
            "intent": "planning",
            "subject": "python",
            "keypoints": ["Python"],
            "search_rag_query": "Python 函数 function 作用域 scope",
            "search_web_query": "Python function scope course notes",
        }

        await gather_intel(state)

        mock_retrieve.assert_called_once_with(
            query="Python 函数 function 作用域 scope",
            subject="python",
        )
        mock_web_search.assert_called_once_with("Python function scope course notes")

    @patch("src.graph.planner.retrieve")
    @patch("src.graph.planner.web_search_fn")
    async def test_resource_intel_uses_retrieval_plan_by_subject(self, mock_web_search, mock_retrieve):
        def fake_retrieve(query, subject, top_k):
            return {
                "docs": [
                    {
                        "content": f"{subject} doc for {query}",
                        "source": f"{subject}.pdf",
                        "score": 0.8,
                    },
                ],
                "is_hit": True,
            }

        mock_retrieve.side_effect = fake_retrieve
        mock_web_search.return_value = [
            {"title": "course notes", "content": "web result", "url": "https://example.com"},
        ]

        state = {
            "messages": [HumanMessage(content="用 Python 做机器学习过拟合检测")],
            "intent": "planning",
            "subject": "python",
            "search_rag_query": "overall query",
            "search_web_query": "overall web query",
            "retrieval_plan": [
                {
                    "subject": "python",
                    "role": "implementation_tool",
                    "rag_query": "Python sklearn code",
                    "purpose": "实现工具",
                    "relation_to_goal": "承载实践",
                    "priority": 0.4,
                },
                {
                    "subject": "machine_learning",
                    "role": "core_concept",
                    "rag_query": "overfitting regularization",
                    "purpose": "核心概念",
                    "relation_to_goal": "解释过拟合",
                    "priority": 0.9,
                },
            ],
        }

        with patch("src.graph.planner.get_setting") as mock_setting:
            mock_setting.side_effect = lambda key, default=None: {
                "rag.multi_subject_per_subject_top_k": 2,
            }.get(key, default)
            result = await _gather_resource_intel(state)

        assert "【知识库资源｜python｜implementation_tool】" in result
        assert "【知识库资源｜machine_learning｜core_concept】" in result
        assert "用途：实现工具" in result
        assert "关系：解释过拟合" in result
        assert "【网络搜索】" in result
        mock_retrieve.assert_any_call(query="Python sklearn code", subject="python", top_k=2)
        mock_retrieve.assert_any_call(query="overfitting regularization", subject="machine_learning", top_k=2)
        mock_web_search.assert_called_once_with("overall web query")

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
                {"content": "高等数学课程重点：函数与导数", "source": "math.pdf", "score": 0.8},
            ],
            "is_hit": True,
        }
        mock_web_search.return_value = [
            {"content": "2026数据科学学习路线实践趋势", "title": "学习规划", "url": "https://example.com"},
        ]

        state = {
            "messages": [HumanMessage(content="帮我制定高等数学课程补基础计划")],
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
            return_value=MagicMock(content="学习者表现出课程压力和学习焦虑。")
        )
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()
        mock_retrieve.return_value = {"docs": [], "is_hit": False}
        mock_web_search.return_value = []

        state = {
            "messages": [
                HumanMessage(content="我好焦虑，高等数学作业和测验总是跟不上"),
                AIMessage(content="我理解你的心情"),
                HumanMessage(content="帮我制定课程补基础计划吧"),
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
            {"content": "2026机器学习课程项目实践安排", "title": "课程项目", "url": "https://example.com"},
        ]

        state = {
            "messages": [HumanMessage(content="帮我做学习计划")],
            "intent": "planning",
            "subject": "math",
            "keypoints": ["学习计划"],
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
