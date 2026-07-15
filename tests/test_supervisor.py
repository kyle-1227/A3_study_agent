"""Unit tests for the supervisor node and routing normalization."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.supervisor import (
    SupervisorConfigurationError,
    SupervisorOutput,
    _VALID_INTENTS,
    _load_valid_intents_config,
    _validate_valid_intents_config,
    handle_unknown,
    route_after_supervisor,
    supervisor_node,
    validate_supervisor_output,
)


class TestSupervisorIntentConfiguration:
    def test_accepts_exact_schema_intents(self):
        configured = ["emotional", "unknown", "academic"]

        assert _validate_valid_intents_config(configured) == frozenset(configured)

    @pytest.mark.parametrize(
        "configured",
        [
            None,
            "academic",
            [],
            ["academic", "emotional"],
            ["academic", "emotional", "unknown", "planning"],
            ["academic", "emotional", "unknown", "unknown"],
            ["academic", "emotional", " unknown"],
            ["academic", "emotional", 3],
        ],
    )
    def test_rejects_missing_drifted_or_repaired_values(self, configured):
        with pytest.raises(SupervisorConfigurationError):
            _validate_valid_intents_config(configured)

    def test_loader_does_not_supply_a_default(self):
        with patch("src.graph.supervisor.get_setting", return_value=None):
            with pytest.raises(SupervisorConfigurationError):
                _load_valid_intents_config()


def _result(
    intent: str = "academic",
    keywords: list[str] | None = None,
    confidence: float = 0.9,
    subject_candidates: list[str] | None = None,
    requested_resource_type: str = "",
    requested_resource_types: list[str] | None = None,
    response_mode: str | None = None,
    qa_scope: str | None = None,
    requires_live_verification: bool = False,
) -> SimpleNamespace:
    has_resources = bool(requested_resource_type or requested_resource_types)
    resolved_mode = response_mode or (
        "emotional" if intent == "emotional" else "resource" if has_resources else "qa"
    )
    resolved_scope = qa_scope
    if resolved_scope is None:
        resolved_scope = (
            ""
            if resolved_mode != "qa"
            else "academic"
            if intent == "academic"
            else "general"
        )
    return SimpleNamespace(
        parsed=SupervisorOutput(
            intent=intent,
            response_mode=resolved_mode,
            qa_scope=resolved_scope,
            requires_live_verification=requires_live_verification,
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
        with patch(
            "src.graph.supervisor.get_available_subjects_from_data",
            return_value=["python", "math"],
        ):
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

        with patch(
            "src.graph.supervisor.get_available_subjects_from_data",
            return_value=["python"],
        ):
            with caplog.at_level("WARNING"):
                await supervisor_node(state)

        record = next(
            r
            for r in caplog.records
            if r.getMessage().startswith('A3_TRACE {"stage": "supervisor"')
        )
        payload = json.loads(record.getMessage().removeprefix("A3_TRACE "))
        assert payload["stage"] == "supervisor"
        assert payload["request_id"] == "req-1"
        assert payload["subject"] == "python"

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_academic_with_study_plan_resource_type(self, mock_invoke):
        """academic intent with requested_resource_type=study_plan stays academic."""
        mock_invoke.return_value = _result(
            intent="academic",
            keywords=["learning plan"],
            confidence=0.9,
            requested_resource_type="study_plan",
        )

        state = {"messages": [HumanMessage(content="Help me make a learning plan")]}
        result = await supervisor_node(state)

        assert result["intent"] == "academic"
        assert result["requested_resource_type"] == "study_plan"
        assert result["requested_resource_types"] == ["study_plan"]

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_emotional_intent(self, mock_invoke):
        mock_invoke.return_value = _result(
            intent="emotional", keywords=[], confidence=0.85
        )

        state = {"messages": [HumanMessage(content="I feel overwhelmed by coursework")]}
        result = await supervisor_node(state)

        assert result["intent"] == "emotional"

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_unknown_intent(self, mock_invoke):
        mock_invoke.return_value = _result(
            intent="unknown", keywords=[], confidence=0.3
        )

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

        state = {
            "messages": [
                HumanMessage(content="Python function parameters and return values")
            ]
        }
        with patch(
            "src.graph.supervisor.get_available_subjects_from_data",
            return_value=["math", "python"],
        ):
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
        assert kwargs["business_validator"] is validate_supervisor_output

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_unavailable_subject_candidates_are_filtered(self, mock_invoke):
        mock_invoke.return_value = _result(
            intent="academic",
            keywords=["contract law"],
            subject_candidates=["law", "python"],
        )

        state = {
            "messages": [HumanMessage(content="What are contract law requirements?")]
        }
        with patch(
            "src.graph.supervisor.get_available_subjects_from_data",
            return_value=["python"],
        ):
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

        state = {
            "messages": [HumanMessage(content="What are contract law requirements?")]
        }
        with patch(
            "src.graph.supervisor.get_available_subjects_from_data",
            return_value=["python"],
        ):
            result = await supervisor_node(state)

        assert result["subject"] == "other"
        assert result["subject_candidates"] == []

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_recommendation_without_subject_keeps_missing_scope(
        self, mock_invoke
    ):
        mock_invoke.return_value = _result(
            intent="academic",
            keywords=["recommendation"],
            subject_candidates=[],
            response_mode="recommendation",
            qa_scope="",
        )

        result = await supervisor_node(
            {"messages": [HumanMessage(content="Recommend a learning resource")]}
        )

        assert result["response_mode"] == "recommendation"
        assert result["subject"] == ""
        assert result["subject_candidates"] == []

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_study_plan_request_sets_resource_type(self, mock_invoke):
        mock_invoke.return_value = _result(
            intent="academic",
            keywords=["machine learning"],
            requested_resource_type="study_plan",
            requested_resource_types=["study_plan"],
        )

        state = {
            "messages": [
                HumanMessage(content="Please create a machine learning study plan")
            ]
        }
        result = await supervisor_node(state)

        assert result["intent"] == "academic"
        assert result["requested_resource_type"] == "study_plan"
        assert result["requested_resource_types"] == ["study_plan"]

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_mindmap_request_sets_route_flag(self, mock_invoke):
        mock_invoke.return_value = _result(
            intent="academic",
            keywords=["data structures"],
            requested_resource_type="mindmap",
            requested_resource_types=["mindmap"],
        )

        state = {
            "messages": [
                HumanMessage(content="Please create a data structures mindmap")
            ]
        }
        result = await supervisor_node(state)

        assert result["needs_mindmap"] is True
        assert result["requested_resource_type"] == "mindmap"
        assert result["requested_resource_types"] == ["mindmap"]

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_multi_resource_request_sets_ordered_resource_types(
        self, mock_invoke
    ):
        mock_invoke.return_value = _result(
            intent="academic",
            keywords=["big data"],
            requested_resource_type="review_doc",
            requested_resource_types=["review_doc", "quiz"],
        )

        state = {
            "messages": [HumanMessage(content="请帮我生成一份大数据复习文档和练习题")]
        }
        result = await supervisor_node(state)

        assert result["requested_resource_type"] == "review_doc"
        assert result["requested_resource_types"] == ["review_doc", "quiz"]

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_single_review_doc_with_excluded_resources_stays_single(
        self, mock_invoke
    ):
        mock_invoke.return_value = _result(
            intent="academic",
            keywords=["overfitting", "regularization", "cross-validation"],
            confidence=0.98,
            subject_candidates=["machine_learning"],
            requested_resource_type="review_doc",
            requested_resource_types=["review_doc"],
        )

        state = {
            "messages": [
                HumanMessage(
                    content=(
                        "请只生成一个 review_doc 复习文档资源，主题是机器学习中的过拟合、"
                        "欠拟合、正则化与交叉验证。不要生成思维导图、练习题、"
                        "代码题或视频脚本。最终输出 Markdown 复习文档。"
                    )
                )
            ]
        }
        with patch(
            "src.graph.supervisor.get_available_subjects_from_data",
            return_value=["machine_learning"],
        ):
            result = await supervisor_node(state)

        assert result["requested_resource_type"] == "review_doc"
        assert result["requested_resource_types"] == ["review_doc"]
        assert result["is_parallel_resource_request"] is False
        assert result["needs_mindmap"] is False

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_plain_mindmap_question_does_not_route_to_mindmap(self, mock_invoke):
        mock_invoke.return_value = _result(intent="academic", keywords=["mindmap"])

        state = {"messages": [HumanMessage(content="What is a mindmap?")]}
        result = await supervisor_node(state)

        assert result["needs_mindmap"] is False
        assert result["requested_resource_type"] == ""

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_query_phrase_cannot_override_structured_qa_contract(
        self, mock_invoke
    ):
        mock_invoke.return_value = _result(
            intent="unknown",
            keywords=["mindmap"],
            response_mode="qa",
            qa_scope="general",
        )

        result = await supervisor_node(
            {
                "messages": [
                    HumanMessage(
                        content="Please create a Python mindmap and practice questions"
                    )
                ]
            }
        )

        assert result["intent"] == "unknown"
        assert result["response_mode"] == "qa"
        assert result["qa_scope"] == "general"
        assert result["requested_resource_type"] == ""
        assert result["requested_resource_types"] == []
        assert result["needs_mindmap"] is False
        assert route_after_supervisor(result) == "qa"

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_resource_only_request_inherits_workspace_subject(self, mock_invoke):
        mock_invoke.return_value = _result(
            intent="academic",
            keywords=["mindmap"],
            requested_resource_type="mindmap",
            requested_resource_types=["mindmap"],
        )
        state = {
            "messages": [HumanMessage(content="make another mindmap")],
            "thread_id": "thread-1",
            "session_id": "thread-1",
            "request_id": "request-2",
            "task_workspace": {
                "schema_version": 1,
                "workspace_id": "workspace:v1:ml",
                "thread_id": "thread-1",
                "active_subject": "Machine Learning",
                "normalized_subject": "machine_learning",
                "active_learning_goal": "Review machine learning concepts",
            },
        }

        with patch(
            "src.graph.supervisor.get_available_subjects_from_data",
            return_value=["machine_learning"],
        ):
            result = await supervisor_node(state)

        assert result["subject"] == "machine_learning"
        assert result["subject_candidates"] == []
        assert result["workspace_continuation_applied"] is True
        assert (
            result["workspace_continuation"]["normalized_subject"] == "machine_learning"
        )

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_recommendation_inherits_workspace_subject_without_candidates(
        self,
        mock_invoke,
    ):
        mock_invoke.return_value = _result(
            intent="academic",
            keywords=["recommendation"],
            response_mode="recommendation",
            qa_scope="",
            subject_candidates=[],
        )
        state = {
            "messages": [HumanMessage(content="Recommend something for this topic")],
            "thread_id": "thread-1",
            "session_id": "thread-1",
            "request_id": "request-3",
            "task_workspace": {
                "schema_version": 1,
                "workspace_id": "workspace:v1:ml",
                "thread_id": "thread-1",
                "active_subject": "Machine Learning",
                "normalized_subject": "machine_learning",
                "active_learning_goal": "Review machine learning concepts",
            },
        }

        with patch(
            "src.graph.supervisor.get_available_subjects_from_data",
            return_value=["machine_learning"],
        ):
            result = await supervisor_node(state)

        assert result["response_mode"] == "recommendation"
        assert result["subject"] == "machine_learning"
        assert result["subject_candidates"] == []
        assert result["requested_resource_types"] == []
        assert result["workspace_continuation_applied"] is True
        assert route_after_supervisor(result) == "recommendation"

    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_explicit_subject_does_not_inherit_workspace_subject(
        self, mock_invoke
    ):
        mock_invoke.return_value = _result(
            intent="academic",
            keywords=["python", "mindmap"],
            subject_candidates=["python"],
            requested_resource_type="mindmap",
            requested_resource_types=["mindmap"],
        )
        state = {
            "messages": [HumanMessage(content="make a Python mindmap")],
            "thread_id": "thread-1",
            "session_id": "thread-1",
            "task_workspace": {
                "schema_version": 1,
                "workspace_id": "workspace:v1:ml",
                "thread_id": "thread-1",
                "active_subject": "Machine Learning",
                "normalized_subject": "machine_learning",
            },
        }

        with patch(
            "src.graph.supervisor.get_available_subjects_from_data",
            return_value=["python", "machine_learning"],
        ):
            result = await supervisor_node(state)

        assert result["subject"] == "python"
        assert result["workspace_continuation_applied"] is False
        assert result["workspace_continuation_reason"] == "current_subject_present"

    @pytest.mark.parametrize(
        ("query", "expected_type", "expected_types"),
        [
            ("Python 的 list 和 tuple 有什么区别？", "", []),
            ("给我一份 Python 复习资料", "review_doc", ["review_doc"]),
            ("帮我生成 Python 思维导图", "mindmap", ["mindmap"]),
            ("给我一份 Python 练习题", "quiz", ["quiz"]),
            (
                "帮我生成一份 Python 的复习资料和思维导图",
                "review_doc",
                ["review_doc", "mindmap"],
            ),
            (
                "帮我生成一份 Python 的复习资料和练习题",
                "review_doc",
                ["review_doc", "quiz"],
            ),
            (
                "帮我生成一份 Python 的复习资料、思维导图和练习题",
                "review_doc",
                ["review_doc", "mindmap", "quiz"],
            ),
        ],
    )
    @patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
    async def test_structured_resource_types_output_list(
        self, mock_invoke, query, expected_type, expected_types
    ):
        mock_invoke.return_value = _result(
            intent="academic",
            keywords=["Python"],
            requested_resource_type=expected_type,
            requested_resource_types=expected_types,
        )

        result = await supervisor_node({"messages": [HumanMessage(content=query)]})

        assert result["requested_resource_type"] == expected_type
        assert result["requested_resource_types"] == expected_types
        assert (len(result["requested_resource_types"]) > 1) is (
            len(expected_types) > 1
        )
        assert result["requested_resource_type"] != "multi_resource"


class TestRouteAfterSupervisor:
    def test_routes_general_and_a3_qa_directly(self):
        assert (
            route_after_supervisor(
                {"intent": "unknown", "response_mode": "qa", "qa_scope": "general"}
            )
            == "qa"
        )
        assert (
            route_after_supervisor(
                {"intent": "unknown", "response_mode": "qa", "qa_scope": "a3_agent"}
            )
            == "qa"
        )

    def test_routes_academic_qa_and_resource_to_retrieval(self):
        assert (
            route_after_supervisor(
                {"intent": "academic", "response_mode": "qa", "qa_scope": "academic"}
            )
            == "academic"
        )

    def test_routes_explicit_recommendation_directly(self):
        assert (
            route_after_supervisor(
                {
                    "intent": "academic",
                    "response_mode": "recommendation",
                    "qa_scope": "",
                }
            )
            == "recommendation"
        )
        assert (
            route_after_supervisor(
                {"intent": "academic", "response_mode": "resource", "qa_scope": ""}
            )
            == "academic"
        )

    def test_invalid_contract_routes_to_technical_handler(self):
        assert route_after_supervisor({"intent": "unknown"}) == "invalid"


class TestValidIntents:
    def test_valid_intents_includes_unknown(self):
        assert "unknown" in _VALID_INTENTS

    def test_valid_intents_no_longer_includes_planning(self):
        """Planning is no longer a valid intent — supervisor sanitizes it."""
        assert _VALID_INTENTS == {"academic", "emotional", "unknown"}
        assert "planning" not in _VALID_INTENTS


class TestHandleUnknown:
    async def test_returns_friendly_message(self):
        state = {
            "messages": [HumanMessage(content="What is the weather today?")],
            "thread_id": "thread-1",
            "request_id": "request-1",
        }
        result = await handle_unknown(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert len(result["messages"][0].content) > 0
        assert result["last_qa_response"]["type"] == "qa_final"


class TestSupervisorOutput:
    def test_valid_output(self):
        output = SupervisorOutput(
            intent="academic",
            response_mode="resource",
            qa_scope="",
            requires_live_verification=False,
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
        output = SupervisorOutput(
            intent="unknown",
            response_mode="qa",
            qa_scope="general",
            requires_live_verification=False,
            keywords=[],
            confidence=0.1,
        )
        assert output.intent == "unknown"

    def test_invalid_intent_raises(self):
        with pytest.raises(Exception):
            SupervisorOutput(
                intent="invalid",
                response_mode="qa",
                qa_scope="general",
                requires_live_verification=False,
                keywords=[],
                confidence=0.5,
            )

    def test_new_routing_fields_are_required(self):
        with pytest.raises(Exception):
            SupervisorOutput(intent="academic", keywords=[], confidence=0.5)


class TestSupervisorBusinessValidation:
    def test_rejects_resource_mode_without_resource(self):
        parsed = SupervisorOutput(
            intent="academic",
            response_mode="resource",
            qa_scope="",
            requires_live_verification=False,
            keywords=["topic"],
            confidence=0.9,
        )
        assert "requires requested_resource_types" in validate_supervisor_output(parsed)

    def test_rejects_qa_with_resource(self):
        parsed = SupervisorOutput(
            intent="academic",
            response_mode="qa",
            qa_scope="academic",
            requires_live_verification=False,
            keywords=["topic"],
            confidence=0.9,
            requested_resource_type="mindmap",
            requested_resource_types=["mindmap"],
        )
        assert "may not carry" in validate_supervisor_output(parsed)

    def test_accepts_strict_explicit_recommendation(self):
        parsed = SupervisorOutput(
            intent="academic",
            response_mode="recommendation",
            qa_scope="",
            requires_live_verification=False,
            keywords=["python", "practice"],
            confidence=0.9,
            subject_candidates=["python"],
        )

        assert validate_supervisor_output(parsed) == ""

    @pytest.mark.parametrize(
        ("overrides", "error_text"),
        [
            ({"intent": "unknown"}, "requires academic intent"),
            (
                {
                    "requested_resource_type": "quiz",
                    "requested_resource_types": ["quiz"],
                },
                "may not carry requested_resource_types",
            ),
            ({"qa_scope": "academic"}, "requires empty qa_scope"),
            (
                {"requires_live_verification": True},
                "may not require live verification",
            ),
        ],
    )
    def test_rejects_invalid_explicit_recommendation_contract(
        self,
        overrides,
        error_text,
    ):
        values = {
            "intent": "academic",
            "response_mode": "recommendation",
            "qa_scope": "",
            "requires_live_verification": False,
            "keywords": ["python"],
            "confidence": 0.9,
            "subject_candidates": ["python"],
        }
        values.update(overrides)

        parsed = SupervisorOutput(**values)

        assert error_text in validate_supervisor_output(parsed)
