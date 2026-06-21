"""Unit tests for the supervisor node and routing normalization."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.supervisor import (
    SupervisorOutput,
    _VALID_INTENTS,
    _detect_requested_resource_type,
    _detect_requested_resource_types,
    handle_unknown,
    route_by_intent,
    supervisor_node,
)


def _result(
    intent: str = "academic",
    keywords: list[str] | None = None,
    confidence: float = 0.9,
    subject_candidates: list[str] | None = None,
    requested_resource_type: str = "",
    requested_resource_types: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        parsed=SupervisorOutput(
            intent=intent,
            keywords=keywords or [],
            confidence=confidence,
            subject_candidates=subject_candidates or [],
            requested_resource_type=requested_resource_type,
            requested_resource_types=requested_resource_types or [],
        ),
        raw_output="{}",
    )


class TestSupervisorNode:
    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_academic_intent(self, mock_invoke):
        mock_invoke.return_value = _result(
            intent="academic",
            keywords=["Python", "function"],
            confidence=0.95,
            subject_candidates=["python"],
        )

        state = {"messages": [HumanMessage(content="How do Python functions work?")]}
        with patch("src.graph.supervisor.get_available_subjects_from_data", return_value=["python", "math"]):
            result = await supervisor_node(state)

        assert result["intent"] == "academic"
        assert result["subject"] == "python"
        assert result["subject_candidates"] == ["python"]
        assert "function" in result["keypoints"]
        mock_invoke.assert_awaited_once()

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_emits_a3_trace_when_enabled(self, mock_invoke, caplog, monkeypatch):
        monkeypatch.setenv("LOG_SUPERVISOR_RESULT", "true")
        mock_invoke.return_value = _result(
            intent="academic",
            keywords=["Python"],
            confidence=0.95,
            subject_candidates=["python"],
        )
        state = {
            "messages": [HumanMessage(content="Explain Python functions")],
            "request_id": "req-1",
            "session_id": "sess-1",
            "thread_id": "thread-1",
        }

        with patch("src.graph.supervisor.get_available_subjects_from_data", return_value=["python"]):
            with caplog.at_level("WARNING"):
                await supervisor_node(state)

        record = next(r for r in caplog.records if r.getMessage().startswith("A3_TRACE "))
        payload = json.loads(record.getMessage().removeprefix("A3_TRACE "))
        assert payload["stage"] == "supervisor"
        assert payload["request_id"] == "req-1"
        assert payload["subject"] == "python"

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_academic_with_study_plan_resource_type(self, mock_invoke):
        """academic intent with requested_resource_type=study_plan stays academic."""
        mock_invoke.return_value = _result(
            intent="academic", keywords=["learning plan"], confidence=0.9,
            requested_resource_type="study_plan",
        )

        state = {"messages": [HumanMessage(content="Help me make a learning plan")]}
        result = await supervisor_node(state)

        assert result["intent"] == "academic"
        assert result["requested_resource_type"] == "study_plan"
        assert result["requested_resource_types"] == ["study_plan"]

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_emotional_intent(self, mock_invoke):
        mock_invoke.return_value = _result(intent="emotional", keywords=[], confidence=0.85)

        state = {"messages": [HumanMessage(content="I feel overwhelmed by coursework")]}
        result = await supervisor_node(state)

        assert result["intent"] == "emotional"

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_unknown_intent(self, mock_invoke):
        mock_invoke.return_value = _result(intent="unknown", keywords=[], confidence=0.3)

        state = {"messages": [HumanMessage(content="What is the weather today?")]}
        result = await supervisor_node(state)

        assert result["intent"] == "unknown"

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_subject_candidates_select_available_subject(self, mock_invoke):
        mock_invoke.return_value = _result(
            intent="academic",
            keywords=["Python", "function"],
            confidence=0.95,
            subject_candidates=["python", "math"],
        )

        state = {"messages": [HumanMessage(content="Python function parameters and return values")]}
        with patch("src.graph.supervisor.get_available_subjects_from_data", return_value=["math", "python"]):
            result = await supervisor_node(state)

        assert result["subject"] == "python"
        assert result["subject_candidates"] == ["python", "math"]

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_structured_output_failure_raises(self, mock_invoke):
        mock_invoke.side_effect = RuntimeError("structured failure")

        state = {"messages": [HumanMessage(content="test")]}
        with pytest.raises(RuntimeError, match="structured failure"):
            await supervisor_node(state)

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_uses_structured_runtime(self, mock_invoke):
        mock_invoke.return_value = _result()

        state = {"messages": [HumanMessage(content="test")]}
        await supervisor_node(state)

        kwargs = mock_invoke.await_args.kwargs
        assert kwargs["node_name"] == "supervisor"
        assert kwargs["llm_node"] == "supervisor"
        assert kwargs["schema"] is SupervisorOutput

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_unavailable_subject_candidates_are_filtered(self, mock_invoke):
        mock_invoke.return_value = _result(
            intent="academic",
            keywords=["contract law"],
            subject_candidates=["law", "python"],
        )

        state = {"messages": [HumanMessage(content="What are contract law requirements?")]}
        with patch("src.graph.supervisor.get_available_subjects_from_data", return_value=["python"]):
            result = await supervisor_node(state)

        assert result["subject"] == "python"
        assert result["subject_candidates"] == ["python"]

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_no_matching_subject_candidates_returns_other(self, mock_invoke):
        mock_invoke.return_value = _result(
            intent="academic",
            keywords=["contract law"],
            subject_candidates=["law"],
        )

        state = {"messages": [HumanMessage(content="What are contract law requirements?")]}
        with patch("src.graph.supervisor.get_available_subjects_from_data", return_value=["python"]):
            result = await supervisor_node(state)

        assert result["subject"] == "other"
        assert result["subject_candidates"] == []

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_study_plan_request_sets_resource_type(self, mock_invoke):
        mock_invoke.return_value = _result(intent="academic", keywords=["machine learning"])

        state = {"messages": [HumanMessage(content="Please create a machine learning study plan")]}
        result = await supervisor_node(state)

        assert result["intent"] == "academic"
        assert result["requested_resource_type"] == "study_plan"
        assert result["requested_resource_types"] == ["study_plan"]

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_mindmap_request_sets_route_flag(self, mock_invoke):
        mock_invoke.return_value = _result(intent="academic", keywords=["data structures"])

        state = {"messages": [HumanMessage(content="Please create a data structures mindmap")]}
        result = await supervisor_node(state)

        assert result["needs_mindmap"] is True
        assert result["requested_resource_type"] == "mindmap"
        assert result["requested_resource_types"] == ["mindmap"]

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_multi_resource_request_sets_ordered_resource_types(self, mock_invoke):
        mock_invoke.return_value = _result(intent="academic", keywords=["big data"])

        state = {"messages": [HumanMessage(content="请帮我生成一份大数据复习文档和练习题")]}
        result = await supervisor_node(state)

        assert result["requested_resource_type"] == "review_doc"
        assert result["requested_resource_types"] == ["review_doc", "quiz"]

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_plain_mindmap_question_does_not_route_to_mindmap(self, mock_invoke):
        mock_invoke.return_value = _result(intent="academic", keywords=["mindmap"])

        state = {"messages": [HumanMessage(content="What is a mindmap?")]}
        result = await supervisor_node(state)

        assert result["needs_mindmap"] is False
        assert result["requested_resource_type"] == ""

    @pytest.mark.parametrize(
        ("query", "expected_type", "expected_types"),
        [
            ("Python 的 list 和 tuple 有什么区别？", "", []),
            ("给我一份 Python 复习资料", "review_doc", ["review_doc"]),
            ("帮我生成 Python 思维导图", "mindmap", ["mindmap"]),
            ("给我一份 Python 练习题", "quiz", ["quiz"]),
            ("帮我生成一份 Python 的复习资料和思维导图", "review_doc", ["review_doc", "mindmap"]),
            ("帮我生成一份 Python 的复习资料和练习题", "review_doc", ["review_doc", "quiz"]),
            ("帮我生成一份 Python 的复习资料、思维导图和练习题", "review_doc", ["review_doc", "mindmap", "quiz"]),
        ],
    )
    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_resource_type_detection_outputs_list(self, mock_invoke, query, expected_type, expected_types):
        mock_invoke.return_value = _result(intent="academic", keywords=["Python"])

        result = await supervisor_node({"messages": [HumanMessage(content=query)]})

        assert result["requested_resource_type"] == expected_type
        assert result["requested_resource_types"] == expected_types
        assert result["multi_resource_mode"] is (len(expected_types) > 1)


class TestResourceTypeDetection:
    def test_detects_explicit_mindmap_generation(self):
        assert _detect_requested_resource_type("Please create a machine learning mindmap") == "mindmap"

    def test_does_not_detect_mindmap_explanation_question(self):
        assert _detect_requested_resource_type("What is a mindmap?") == ""

    def test_detects_study_plan_requests(self):
        assert _detect_requested_resource_type("Please create a Python study plan") == "study_plan"
        assert _detect_requested_resource_type("Give me a machine learning roadmap") == "study_plan"

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("Python 的 list 和 tuple 有什么区别？", []),
            ("给我一份 Python 复习资料", ["review_doc"]),
            ("帮我生成 Python 思维导图", ["mindmap"]),
            ("给我一份 Python 练习题", ["quiz"]),
            ("帮我生成一份 Python 的复习资料和思维导图", ["review_doc", "mindmap"]),
            ("帮我生成一份 Python 的复习资料和练习题", ["review_doc", "quiz"]),
            ("帮我生成一份 Python 的复习资料、思维导图和练习题", ["review_doc", "mindmap", "quiz"]),
        ],
    )
    def test_detects_requested_resource_types(self, query, expected):
        assert _detect_requested_resource_types(query) == expected

    def test_detects_multiple_resource_types_in_order(self):
        assert _detect_requested_resource_types("请帮我生成机器学习思维导图和练习题") == ["mindmap", "quiz"]

    def test_does_not_detect_resource_list_for_explanation_question(self):
        assert _detect_requested_resource_types("什么是思维导图？") == []


class TestRouteByIntent:
    def test_routes_academic(self):
        assert route_by_intent({"intent": "academic"}) == "academic"

    def test_routes_planning_to_unknown(self):
        """Planning is no longer a valid intent — routes to unknown."""
        assert route_by_intent({"intent": "planning"}) == "unknown"

    def test_routes_emotional(self):
        assert route_by_intent({"intent": "emotional"}) == "emotional"

    def test_routes_unknown(self):
        assert route_by_intent({"intent": "unknown"}) == "unknown"

    def test_missing_intent_defaults_to_academic(self):
        assert route_by_intent({}) == "academic"


class TestValidIntents:
    def test_valid_intents_includes_unknown(self):
        assert "unknown" in _VALID_INTENTS

    def test_valid_intents_no_longer_includes_planning(self):
        """Planning is no longer a valid intent — supervisor sanitizes it."""
        assert _VALID_INTENTS == {"academic", "emotional", "unknown"}
        assert "planning" not in _VALID_INTENTS


class TestHandleUnknown:
    async def test_returns_friendly_message(self):
        state = {"messages": [HumanMessage(content="What is the weather today?")]}
        result = await handle_unknown(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert len(result["messages"][0].content) > 0


class TestSupervisorOutput:
    def test_valid_output(self):
        output = SupervisorOutput(
            intent="academic",
            keywords=["mathematics"],
            confidence=0.9,
            subject_candidates=["math"],
            requested_resource_types=["quiz"],
        )
        assert output.intent == "academic"
        assert output.keywords == ["mathematics"]
        assert output.confidence == 0.9
        assert output.subject_candidates == ["math"]
        assert output.requested_resource_types == ["quiz"]

    def test_unknown_intent_valid(self):
        output = SupervisorOutput(intent="unknown", keywords=[], confidence=0.1)
        assert output.intent == "unknown"

    def test_invalid_intent_raises(self):
        with pytest.raises(Exception):
            SupervisorOutput(intent="invalid", keywords=[], confidence=0.5)
