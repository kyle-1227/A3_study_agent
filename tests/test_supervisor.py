"""Unit tests for the Supervisor node (intent routing + keypoint extraction)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.supervisor import (
    SupervisorOutput,
    _VALID_INTENTS,
    _detect_requested_resource_type,
    handle_unknown,
    route_by_intent,
    supervisor_node,
)


def _mock_supervisor_output(intent="academic", keywords=None, confidence=0.9, subject_candidates=None):
    """Helper to create a SupervisorOutput instance for mocking."""
    return SupervisorOutput(
        intent=intent,
        keywords=keywords or [],
        confidence=confidence,
        subject_candidates=subject_candidates or [],
    )


class TestSupervisorNode:

    @patch("src.graph.supervisor.get_node_llm")
    async def test_academic_intent(self, mock_get_llm):
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=_mock_supervisor_output(
            intent="academic",
            keywords=["Python", "函数/function"],
            confidence=0.95,
            subject_candidates=["python"],
        ))
        mock_llm.with_structured_output.return_value = structured_llm
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="Python 函数怎么理解？")]}
        with patch("src.graph.supervisor.get_available_subjects_from_data", return_value=["python", "math"]):
            result = await supervisor_node(state)

        assert result["intent"] == "academic"
        assert result["subject"] == "python"
        assert result["subject_candidates"] == ["python"]
        assert "函数/function" in result["keypoints"]
        mock_llm.with_structured_output.assert_called_once_with(SupervisorOutput, method="json_mode")

    @patch("src.graph.supervisor.get_node_llm")
    async def test_planning_intent(self, mock_get_llm):
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=_mock_supervisor_output(
            intent="planning", keywords=[], confidence=0.9,
        ))
        mock_llm.with_structured_output.return_value = structured_llm
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="帮我制定复习计划")]}
        result = await supervisor_node(state)

        assert result["intent"] == "planning"
        assert result["keypoints"] == []

    @patch("src.graph.supervisor.get_node_llm")
    async def test_emotional_intent(self, mock_get_llm):
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=_mock_supervisor_output(
            intent="emotional", keywords=[], confidence=0.85,
        ))
        mock_llm.with_structured_output.return_value = structured_llm
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="我好焦虑")]}
        result = await supervisor_node(state)

        assert result["intent"] == "emotional"

    @patch("src.graph.supervisor.get_node_llm")
    async def test_unknown_intent(self, mock_get_llm):
        """Unknown intent is returned when the query is off-topic."""
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=_mock_supervisor_output(
            intent="unknown", keywords=[], confidence=0.3,
        ))
        mock_llm.with_structured_output.return_value = structured_llm
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="今天天气怎么样")]}
        result = await supervisor_node(state)

        assert result["intent"] == "unknown"

    @patch("src.graph.supervisor.get_node_llm")
    async def test_subject_candidates_select_available_subject(self, mock_get_llm):
        """Supervisor subject selection is based on available subjects, not keyword maps."""
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=_mock_supervisor_output(
            intent="academic",
            keywords=["Python", "函数/function"],
            confidence=0.95,
            subject_candidates=["python", "math"],
        ))
        mock_llm.with_structured_output.return_value = structured_llm
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="Python 函数 参数 返回值")]}
        with patch("src.graph.supervisor.get_available_subjects_from_data", return_value=["math", "python"]):
            result = await supervisor_node(state)

        assert result["intent"] == "academic"
        assert result["subject"] == "python"
        assert result["subject_candidates"] == ["python", "math"]

    @patch("src.graph.supervisor.get_node_llm")
    async def test_structured_output_failure_falls_back(self, mock_get_llm):
        """When structured output fails, fall back to academic defaults."""
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
        mock_llm.with_structured_output.return_value = structured_llm
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="test")]}
        result = await supervisor_node(state)

        assert result["intent"] == "academic"
        assert result["subject"] == "other"
        assert result["keypoints"] == []

    @patch("src.graph.supervisor.get_node_llm")
    async def test_uses_with_structured_output(self, mock_get_llm):
        """Verify with_structured_output is called (no json.loads)."""
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=_mock_supervisor_output())
        mock_llm.with_structured_output.return_value = structured_llm
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="test")]}
        await supervisor_node(state)

        mock_llm.with_structured_output.assert_called_once_with(SupervisorOutput, method="json_mode")
        structured_llm.ainvoke.assert_called_once()

    @patch("src.graph.supervisor.get_node_llm")
    async def test_keywords_mapped_to_keypoints(self, mock_get_llm):
        """SupervisorOutput.keywords should be mapped to state keypoints."""
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=_mock_supervisor_output(
            intent="academic", keywords=["椭圆", "离心率"],
        ))
        mock_llm.with_structured_output.return_value = structured_llm
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="椭圆的离心率怎么求？")]}
        result = await supervisor_node(state)

        assert result["keypoints"] == ["椭圆", "离心率"]

    @patch("src.graph.supervisor.get_node_llm")
    async def test_python_function_not_hardcoded_as_math(self, mock_get_llm):
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=_mock_supervisor_output(
            intent="academic",
            keywords=["Python", "函数/function"],
            subject_candidates=["python"],
        ))
        mock_llm.with_structured_output.return_value = structured_llm
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="Python 函数 参数 返回值 作用域")]}
        with patch("src.graph.supervisor.get_available_subjects_from_data", return_value=["math", "python"]):
            result = await supervisor_node(state)

        assert result["subject"] == "python"

    @patch("src.graph.supervisor.get_node_llm")
    async def test_unavailable_subject_candidates_are_filtered(self, mock_get_llm):
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=_mock_supervisor_output(
            intent="academic",
            keywords=["合同法/contract law"],
            subject_candidates=["law", "python"],
        ))
        mock_llm.with_structured_output.return_value = structured_llm
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="合同法的要约和承诺是什么？")]}
        with patch("src.graph.supervisor.get_available_subjects_from_data", return_value=["python"]):
            result = await supervisor_node(state)

        assert result["subject"] == "python"
        assert result["subject_candidates"] == ["python"]

    @patch("src.graph.supervisor.get_node_llm")
    async def test_no_matching_subject_candidates_returns_other(self, mock_get_llm):
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=_mock_supervisor_output(
            intent="academic",
            keywords=["合同法/contract law"],
            subject_candidates=["law"],
        ))
        mock_llm.with_structured_output.return_value = structured_llm
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="合同法的要约和承诺是什么？")]}
        with patch("src.graph.supervisor.get_available_subjects_from_data", return_value=["python"]):
            result = await supervisor_node(state)

        assert result["subject"] == "other"
        assert result["subject_candidates"] == []

    @patch("src.graph.supervisor.get_node_llm")
    async def test_ignores_llm_mindmap_flag_for_plain_question(self, mock_get_llm):
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        llm_result = MagicMock(
            intent="academic",
            keywords=["思维导图"],
            confidence=0.95,
            requested_resource_type="mindmap",
            needs_mindmap=True,
        )
        structured_llm.ainvoke = AsyncMock(return_value=llm_result)
        mock_llm.with_structured_output.return_value = structured_llm
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="思维导图是什么？")]}
        result = await supervisor_node(state)

        assert result["needs_mindmap"] is False
        assert result["requested_resource_type"] == ""

    @patch("src.graph.supervisor.get_node_llm")
    async def test_mindmap_request_sets_route_flag(self, mock_get_llm):
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=_mock_supervisor_output(
            intent="academic", keywords=["数据结构", "栈", "队列"],
        ))
        mock_llm.with_structured_output.return_value = structured_llm
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="给我做一个数据结构栈和队列的思维导图")]}
        result = await supervisor_node(state)

        assert result["needs_mindmap"] is True
        assert result["requested_resource_type"] == "mindmap"

    @patch("src.graph.supervisor.get_node_llm")
    async def test_plain_mindmap_question_does_not_route_to_mindmap(self, mock_get_llm):
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=_mock_supervisor_output(
            intent="academic", keywords=["思维导图"],
        ))
        mock_llm.with_structured_output.return_value = structured_llm
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="思维导图是什么？")]}
        result = await supervisor_node(state)

        assert result["needs_mindmap"] is False
        assert result["requested_resource_type"] == ""

    @patch("src.graph.supervisor.get_node_llm")
    async def test_other_resource_request_is_detected_without_mindmap_route(self, mock_get_llm):
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=_mock_supervisor_output(
            intent="academic", keywords=["机器学习", "练习题"],
        ))
        mock_llm.with_structured_output.return_value = structured_llm
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="请生成机器学习过拟合的分层练习题")]}
        result = await supervisor_node(state)

        assert result["needs_mindmap"] is False
        assert result["requested_resource_type"] == "quiz"


class TestResourceTypeDetection:

    def test_detects_explicit_mindmap_generation(self):
        assert _detect_requested_resource_type("生成机器学习过拟合的思维导图") == "mindmap"

    def test_does_not_detect_mindmap_explanation_question(self):
        assert _detect_requested_resource_type("思维导图是什么？") == ""

    def test_detects_other_resource_types(self):
        assert _detect_requested_resource_type("请生成数据库第一章练习题") == "quiz"
        assert _detect_requested_resource_type("帮我制作人工智能导论 PPT") == "ppt"
        assert _detect_requested_resource_type("给我做一个 Python 代码案例") == "code_case"


class TestRouteByIntent:

    def test_routes_academic(self):
        assert route_by_intent({"intent": "academic"}) == "academic"

    def test_routes_planning(self):
        assert route_by_intent({"intent": "planning"}) == "planning"

    def test_routes_emotional(self):
        assert route_by_intent({"intent": "emotional"}) == "emotional"

    def test_routes_unknown(self):
        assert route_by_intent({"intent": "unknown"}) == "unknown"

    def test_routes_mindmap_request_as_academic(self):
        assert route_by_intent({"intent": "academic", "needs_mindmap": True}) == "academic"

    def test_missing_intent_defaults_to_academic(self):
        assert route_by_intent({}) == "academic"


class TestValidIntents:

    def test_valid_intents_includes_unknown(self):
        assert "unknown" in _VALID_INTENTS

    def test_valid_intents_set(self):
        assert _VALID_INTENTS == {"academic", "planning", "emotional", "unknown"}


class TestHandleUnknown:

    async def test_returns_friendly_message(self):
        state = {"messages": [HumanMessage(content="今天天气怎么样")]}
        result = await handle_unknown(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        # Should contain a friendly message in Chinese
        assert len(result["messages"][0].content) > 0

    async def test_message_is_ai_message(self):
        state = {"messages": [HumanMessage(content="帮我订外卖")]}
        result = await handle_unknown(state)

        msg = result["messages"][0]
        assert isinstance(msg, AIMessage)


class TestSupervisorOutput:

    def test_valid_output(self):
        output = SupervisorOutput(
            intent="academic",
            keywords=["数学/mathematics"],
            confidence=0.9,
            subject_candidates=["math"],
        )
        assert output.intent == "academic"
        assert output.keywords == ["数学/mathematics"]
        assert output.confidence == 0.9
        assert output.subject_candidates == ["math"]

    def test_unknown_intent_valid(self):
        output = SupervisorOutput(intent="unknown", keywords=[], confidence=0.1)
        assert output.intent == "unknown"

    def test_invalid_intent_raises(self):
        with pytest.raises(Exception):
            SupervisorOutput(intent="invalid", keywords=[], confidence=0.5)
