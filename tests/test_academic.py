"""Unit tests for the current academic evidence path."""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.config import get_setting
from src.graph.academic import (
    _best_doc_score,
    _build_local_evidence_candidates,
    _dedupe_and_merge_local_candidates,
    _deterministic_memory_use_decision,
    _evaluate_retrieval_branch,
    _format_retrieved,
    _format_web_research_context,
    _context_item_from_evidence,
    _normalize_retrieval_plan,
    _order_evidence_candidates_for_grading,
    _safe_evidence_id_part,
    _select_docs_with_subject_quota,
    MemoryUseDecisionOutput,
    RetrievalPlanItem,
    SearchQueryRewriteOutput,
    academic_router,
    build_evidence_memory_summary,
    generate_answer,
    memory_use_decider,
    rag_retrieve,
    rewrite_query,
    search_query_rewriter,
    select_relevant_memory_summaries,
    validate_search_query_rewrite_output,
    web_search,
)
from src.graph.evidence import EvidenceCandidate, EvidenceJudgeItem, EvidenceJudgeOutput
from src.graph.state import CONTEXT_CLEAR
from src.graph.web_research import (
    WebResearchPlan,
    WebResearchTask,
    WebSourceSummary,
    WebSourceSummaryBatch,
)
from src.llm.structured_output import (
    StructuredLLMResult,
    StructuredOutputError,
    get_llm_output_mode,
)
from src.observability.a3_trace import reset_trace_event_sink, set_trace_event_sink


def _valid_query_rewrite_output() -> SearchQueryRewriteOutput:
    return SearchQueryRewriteOutput(
        local_retrieval_query="Python functions 参数 return value scope",
        web_research_seed_query="Python function parameters return value course notes tutorial",
        expanded_keypoints=["Python functions", "参数 parameter", "return value"],
        reason="Expanded concise bilingual retrieval terms.",
        learning_goal="Understand Python functions",
        primary_subject="python",
        subject_relation_summary="single subject",
        retrieval_plan=[
            RetrievalPlanItem(
                subject="python",
                role="core_concept",
                local_retrieval_query="Python functions 参数 return value scope",
                web_research_seed_query="Python function parameters tutorial course notes",
                priority=0.9,
                retrieval_coverage_goals=["function definition", "parameter passing"],
            )
        ],
    )


def _structured_output_error(
    *,
    phase: str = "parsing_error",
    error_type: str = "JSONDecodeError",
    error_message: str = "invalid json",
    raw_output: str = "{bad",
) -> StructuredOutputError:
    return StructuredOutputError(
        StructuredLLMResult(
            success=False,
            parsed=None,
            node_name="search_query_rewriter",
            llm_node="query_rewrite",
            schema_name="SearchQueryRewriteOutput",
            provider="test",
            model="test",
            output_mode="native_json_schema_pydantic",
            raw_output=raw_output,
            failure_phase=phase,
            error_type=error_type,
            error_message=error_message,
            parsing_error=error_message if phase == "parsing_error" else "",
            validation_error=error_message if phase == "validation_error" else "",
            business_validation_error=error_message
            if phase == "business_validation_error"
            else "",
        )
    )


class TestAcademicRouterRetry:
    async def test_returns_empty_on_first_run(self):
        result = await academic_router(
            {"messages": [HumanMessage(content="test")], "retry_count": 0}
        )
        assert "context" not in result

    async def test_clears_context_on_retry(self):
        result = await academic_router(
            {"messages": [HumanMessage(content="test")], "retry_count": 1}
        )
        assert result["context"] is CONTEXT_CLEAR


class TestRewriteQuery:
    @patch("src.graph.llm.invoke_plain_llm_fail_fast")
    async def test_produces_rewritten_query(self, mock_invoke_plain):
        mock_invoke_plain.return_value = "improved retrieval query"

        result = await rewrite_query(
            {
                "messages": [HumanMessage(content="original question")],
                "hallucination_reason": "fabricated detail",
                "retry_count": 1,
                "request_id": "test-req",
                "thread_id": "test-thread",
            }
        )

        assert result["rewritten_query"] == "improved retrieval query"
        assert result["retrieval_plan"] == []

    @patch("src.graph.llm.invoke_plain_llm_fail_fast")
    async def test_fail_fast_on_retry_rewrite_failure(self, mock_invoke_plain):
        """Rewrite query now fails fast - no fallback to original query."""
        mock_invoke_plain.side_effect = RuntimeError("LLM error")

        with pytest.raises(RuntimeError, match="LLM error"):
            await rewrite_query(
                {
                    "messages": [HumanMessage(content="original question")],
                    "hallucination_reason": "bad",
                    "retry_count": 1,
                    "request_id": "test-req",
                    "thread_id": "test-thread",
                }
            )


class TestMemoryUseDecision:
    def test_memory_use_decider_has_explicit_deepseek_official_config(self):
        assert get_setting("llm.memory_use_decider.provider") == "deepseek_official"
        assert (
            get_setting("llm.memory_use_decider.base_url") == "https://api.deepseek.com"
        )
        assert (
            get_setting("llm.memory_use_decider.beta_base_url")
            == "https://api.deepseek.com/beta"
        )
        assert get_setting("llm.memory_use_decider.api_key_env") == "DEEPSEEK_API_KEY"
        assert get_llm_output_mode("memory_use_decider") == "deepseek_tool_call_strict"

    def test_empty_memory_ignores_without_prompt(self):
        decision = _deterministic_memory_use_decision(
            "重新给我一份学习计划", selected_memory_count=0
        )
        assert decision is not None
        assert decision.decision == "ignore"

    def test_explicit_history_reference_uses_memory(self):
        decision = _deterministic_memory_use_decision(
            "结合之前的内容，给我一份学习计划", selected_memory_count=1
        )
        assert decision is not None
        assert decision.decision == "use"

    def test_explicit_history_exclusion_ignores_memory(self):
        decision = _deterministic_memory_use_decision(
            "不要参考之前，给我一份学习计划", selected_memory_count=1
        )
        assert decision is not None
        assert decision.decision == "ignore"

    def test_ambiguous_revision_asks_user_when_memory_exists(self):
        decision = _deterministic_memory_use_decision(
            "重新给我一份学习计划", selected_memory_count=1
        )
        assert decision is not None
        assert decision.decision == "ask_user"
        assert decision.question_to_user

    async def test_memory_use_decider_ignores_when_no_memory(self):
        result = await memory_use_decider(
            {
                "messages": [HumanMessage(content="重新给我一份学习计划")],
                "evidence_summary_memory": [],
                "subject": "other",
                "requested_resource_type": "study_plan",
                "request_id": "req",
                "thread_id": "thread",
            }
        )
        assert result["memory_use_policy"] == "ignore"
        assert result["selected_evidence_memory_summaries"] == []

    def test_irrelevant_recent_memory_is_ineligible(self):
        selected = select_relevant_memory_summaries(
            {
                "evidence_summary_memory": [
                    {
                        "memory_id": "recent-math",
                        "subject": "math",
                        "resource_type": "mindmap",
                        "summary": "linear algebra vector spaces matrix decomposition",
                    }
                ],
                "request_id": "req",
                "thread_id": "thread",
            },
            current_query="Python functions quiz practice",
            subject="python",
            requested_resource_type="quiz",
        )

        assert selected == []

    @patch("src.graph.academic.interrupt", return_value={"choice": "ignore"})
    async def test_relevant_memory_plus_ambiguous_query_asks_user(self, mock_interrupt):
        result = await memory_use_decider(
            {
                "messages": [HumanMessage(content="重新给我一份学习计划")],
                "evidence_summary_memory": [
                    {
                        "memory_id": "python-plan",
                        "subject": "python",
                        "resource_type": "study_plan",
                        "summary": "Python functions parameters return values learning plan",
                    }
                ],
                "subject": "python",
                "requested_resource_type": "study_plan",
                "request_id": "req",
                "thread_id": "thread",
            }
        )

        mock_interrupt.assert_called_once()
        assert result["memory_use_policy"] == "ignore"
        assert result["eligible_evidence_memory_count"] == 1

    @patch("src.graph.academic.interrupt")
    async def test_explicit_history_use_bypasses_popup(self, mock_interrupt):
        result = await memory_use_decider(
            {
                "messages": [HumanMessage(content="结合之前的内容，给我一份学习计划")],
                "evidence_summary_memory": [
                    {
                        "memory_id": "python-plan",
                        "subject": "python",
                        "resource_type": "study_plan",
                        "summary": "Python functions parameters return values learning plan",
                    }
                ],
                "subject": "python",
                "requested_resource_type": "study_plan",
                "request_id": "req",
                "thread_id": "thread",
            }
        )

        mock_interrupt.assert_not_called()
        assert result["memory_use_policy"] == "use"
        assert result["selected_evidence_memory_summaries"]

    async def test_memory_use_decider_prompt_discourages_ask_user_for_explicit_mismatch(
        self,
    ):
        captured: dict = {}

        async def fake_invoke_structured_llm(**kwargs):
            captured.update(kwargs)
            return StructuredLLMResult(
                success=True,
                parsed=MemoryUseDecisionOutput(
                    decision="ignore",
                    reason="Current query is explicit and memory mismatches.",
                    question_to_user="",
                ),
                node_name=kwargs["node_name"],
                llm_node=kwargs["llm_node"],
                schema_name=kwargs["schema"].__name__,
                provider="unit",
                model="unit",
                output_mode=kwargs["output_mode"],
            )

        with (
            patch(
                "src.graph.academic._deterministic_memory_use_decision",
                return_value=None,
            ),
            patch(
                "src.graph.academic.invoke_structured_llm", fake_invoke_structured_llm
            ),
        ):
            result = await memory_use_decider(
                {
                    "messages": [HumanMessage(content="我还想要一份大数据的练习题")],
                    "evidence_summary_memory": [
                        {
                            "memory_id": "python-quiz",
                            "subject": "python",
                            "resource_type": "quiz",
                            "summary": "Python functions practice quiz",
                        }
                    ],
                    "subject": "big_data",
                    "requested_resource_type": "quiz",
                    "request_id": "req",
                    "thread_id": "thread",
                }
            )

        system_prompt = captured["messages"][0].content
        assert "explicit subject and explicit requested resource type" in system_prompt
        assert "subject/resource mismatched" in system_prompt
        assert "choose ignore" in system_prompt
        assert (
            "Choose ask_user only when the current query contains a genuinely ambiguous history reference"
            in system_prompt
        )
        assert result["memory_use_policy"] == "ignore"


class TestSearchQueryRewriter:
    def test_query_rewrite_schema_exposes_length_limits(self):
        schema = SearchQueryRewriteOutput.model_json_schema()
        props = schema["properties"]
        plan_props = schema["$defs"]["RetrievalPlanItem"]["properties"]

        assert schema["additionalProperties"] is False
        assert schema["$defs"]["RetrievalPlanItem"]["additionalProperties"] is False
        assert props["local_retrieval_query"]["maxLength"] == 240
        assert props["web_research_seed_query"]["maxLength"] == 180
        assert props["expanded_keypoints"]["maxItems"] == 8
        assert props["expanded_keypoints"]["items"]["maxLength"] == 120
        assert props["retrieval_plan"]["maxItems"] == 4
        assert props["memory_context_notes"]["maxItems"] == 5
        assert props["memory_context_notes"]["items"]["maxLength"] == 240
        assert plan_props["retrieval_coverage_goals"]["maxItems"] == 8
        assert plan_props["retrieval_coverage_goals"]["items"]["maxLength"] == 120

    def test_search_query_rewriter_uses_unified_structured_retry(self):
        assert get_setting("llm_outputs.search_query_rewriter.max_retries") == 2
        assert (
            get_setting("llm_outputs.search_query_rewriter.output_mode")
            == "deepseek_tool_call_strict"
        )
        assert get_setting("provider_transport_retry.max_retries") == 2

    def test_overlong_query_fields_fail_schema_validation(self):
        with pytest.raises(Exception):
            SearchQueryRewriteOutput(
                local_retrieval_query="x" * 241,
                web_research_seed_query="Python function tutorial",
                expanded_keypoints=["Python"],
                reason="too long query",
            )

    def test_combined_query_rewrite_keys_fail_schema_validation(self):
        with pytest.raises(Exception):
            SearchQueryRewriteOutput.model_validate(
                {
                    "expanded_keypoints": ["big data quiz"],
                    "reason": "bad combined keys",
                    "learning_goal_primary_subject": "big_data",
                    "primary_subject_relation_summary": "single subject",
                    "local_retrieval_query_web_research_seed_query": "big data quiz",
                    "retrieval_plan_subject_role_local_retrieval_query_web_research_seed_query_purpose_relation_to_goal_retrieval_coverage_hint_retrieval_coverage_goals_priority": [],
                }
            )

    def test_repeated_chinese_phrases_fail_business_validation(self):
        parsed = _valid_query_rewrite_output()
        parsed.local_retrieval_query = (
            "检索意图 资源类型 练习题 答案 解析 实操任务 " * 3
        )

        error = validate_search_query_rewrite_output(parsed, memory_use_policy="ignore")

        assert "repeated query phrase" in error
        assert "检索意图" in error

    def test_repeated_english_ngram_fails_business_validation(self):
        parsed = _valid_query_rewrite_output()
        parsed.web_research_seed_query = "python function parameter return " * 3

        error = validate_search_query_rewrite_output(parsed, memory_use_policy="ignore")

        assert "repeated query ngram" in error

    def test_valid_concise_query_passes_business_validation(self):
        assert (
            validate_search_query_rewrite_output(
                _valid_query_rewrite_output(),
                memory_use_policy="ignore",
            )
            == ""
        )

    def test_duplicate_retrieval_plan_subject_fails_business_validation(self):
        parsed = _valid_query_rewrite_output()
        parsed.retrieval_plan.append(
            RetrievalPlanItem(
                subject="python",
                role="application_example",
                local_retrieval_query="Python function application examples",
                web_research_seed_query="Python function application tutorial",
                purpose="Retrieve application examples for the same subject.",
                relation_to_goal="Apply the function concepts in code.",
                retrieval_coverage_goals=["application examples"],
                priority=0.7,
            )
        )

        error = validate_search_query_rewrite_output(
            parsed,
            memory_use_policy="ignore",
        )

        assert error == (
            "retrieval_plan.1.subject duplicates retrieval_plan.0.subject: python"
        )

    def test_distinct_retrieval_plan_subjects_pass_business_validation(self):
        parsed = _valid_query_rewrite_output()
        parsed.retrieval_plan.append(
            RetrievalPlanItem(
                subject="machine_learning",
                role="application_context",
                local_retrieval_query="Machine learning Python function applications",
                web_research_seed_query="Machine learning Python implementation tutorial",
                purpose="Retrieve a distinct application context.",
                relation_to_goal="Connect Python functions to model implementation.",
                retrieval_coverage_goals=["model implementation"],
                priority=0.7,
            )
        )

        assert (
            validate_search_query_rewrite_output(
                parsed,
                memory_use_policy="ignore",
            )
            == ""
        )

    @patch("src.graph.academic.get_available_subjects_from_data")
    @patch("src.graph.academic.invoke_structured_llm", new_callable=AsyncMock)
    async def test_produces_rag_web_queries_and_plan(
        self, mock_invoke, mock_available_subjects
    ):
        mock_available_subjects.return_value = ["python", "machine_learning"]
        parsed = SearchQueryRewriteOutput(
            local_retrieval_query="Python functions parameters return values",
            web_research_seed_query="Python functions course notes tutorial",
            expanded_keypoints=["Python", "functions"],
            reason="expanded for bilingual retrieval",
            learning_goal="Understand Python functions",
            primary_subject="python",
            subject_relation_summary="single subject",
            retrieval_plan=[
                RetrievalPlanItem(
                    subject="python",
                    role="core_concept",
                    local_retrieval_query="Python functions",
                    web_research_seed_query="Python functions tutorial",
                    priority=0.8,
                ),
            ],
        )
        mock_invoke.return_value = SimpleNamespace(
            parsed=parsed, raw_output='{"ok": true}'
        )

        result = await search_query_rewriter(
            {
                "messages": [HumanMessage(content="Explain Python functions")],
                "keypoints": ["Python"],
                "subject": "python",
                "subject_candidates": ["python"],
                "memory_use_policy": "ignore",
            }
        )

        assert (
            result["local_retrieval_query"]
            == "Python functions parameters return values"
        )
        assert (
            result["web_research_seed_query"]
            == "Python functions course notes tutorial"
        )
        assert result["retrieval_plan"][0]["subject"] == "python"
        assert result["primary_subject"] == "python"
        mock_invoke.assert_awaited_once()

    @patch("src.graph.academic.get_available_subjects_from_data")
    @patch("src.graph.academic.invoke_structured_llm", new_callable=AsyncMock)
    @patch("src.graph.academic._maintain_conversation_summary", new_callable=AsyncMock)
    async def test_workspace_continuation_is_prompted_even_when_memory_ignored(
        self,
        mock_summary,
        mock_invoke,
        mock_available_subjects,
    ):
        mock_available_subjects.return_value = ["machine_learning"]
        mock_summary.return_value = ""
        parsed = SearchQueryRewriteOutput(
            local_retrieval_query="machine learning core concepts mindmap structure",
            web_research_seed_query="machine learning concept map prerequisites pitfalls",
            expanded_keypoints=["machine learning", "concept relationships"],
            reason="Use same-thread workspace subject for resource continuity.",
            learning_goal="Review machine learning concepts",
            primary_subject="machine_learning",
            subject_relation_summary="single subject",
            retrieval_plan=[
                RetrievalPlanItem(
                    subject="machine_learning",
                    role="core_concept",
                    local_retrieval_query="machine learning core concepts",
                    web_research_seed_query="machine learning concept map",
                    priority=0.9,
                ),
            ],
        )
        mock_invoke.return_value = SimpleNamespace(
            parsed=parsed, raw_output='{"ok": true}'
        )

        result = await search_query_rewriter(
            {
                "messages": [HumanMessage(content="make another mindmap")],
                "keypoints": ["mindmap"],
                "requested_resource_type": "mindmap",
                "subject": "machine_learning",
                "subject_candidates": [],
                "memory_use_policy": "ignore",
                "workspace_continuation_applied": True,
                "workspace_continuation_reason": "",
                "workspace_continuation": {
                    "can_continue": True,
                    "continuation_applied": True,
                    "workspace_id": "workspace:v1:ml",
                    "thread_id": "thread-1",
                    "active_subject": "Machine Learning",
                    "normalized_subject": "machine_learning",
                    "active_learning_goal": "Review machine learning concepts",
                    "resource_types": ["mindmap"],
                },
            }
        )

        messages = mock_invoke.await_args.kwargs["messages"]
        prompt = messages[1].content
        assert "Task Workspace Continuation" in prompt
        assert "machine_learning" in prompt
        assert "Review machine learning concepts" in prompt
        assert result["primary_subject"] == "machine_learning"

    @patch("src.graph.academic.get_available_subjects_from_data")
    @patch("src.graph.academic.invoke_structured_llm", new_callable=AsyncMock)
    @patch("src.graph.academic._maintain_conversation_summary", new_callable=AsyncMock)
    async def test_query_rewrite_delegates_compliance_retry_to_structured_runtime(
        self,
        mock_summary,
        mock_invoke,
        mock_available_subjects,
    ):
        mock_available_subjects.return_value = ["python"]
        mock_summary.return_value = ""
        parsed = _valid_query_rewrite_output()
        mock_invoke.return_value = SimpleNamespace(
            parsed=parsed, raw_output='{"ok": true}'
        )
        events: list[dict] = []
        token = set_trace_event_sink(events)
        try:
            result = await search_query_rewriter(
                {
                    "messages": [HumanMessage(content="Explain Python functions")],
                    "keypoints": ["Python"],
                    "subject": "python",
                    "subject_candidates": ["python"],
                    "memory_use_policy": "ignore",
                    "request_id": "req",
                    "thread_id": "thread",
                }
            )
        finally:
            reset_trace_event_sink(token)

        assert result["local_retrieval_query"] == parsed.local_retrieval_query
        assert mock_invoke.await_count == 1
        assert not any(
            event["stage"] == "query_rewrite_compliance_retry" for event in events
        )
        memory_event = next(
            event for event in events if event["stage"] == "query_rewrite_memory_use"
        )
        assert memory_event["memory_prompt_injected"] is False
        assert (
            memory_event["memory_used_for_retrieval"]
            == memory_event["llm_reported_memory_used_for_retrieval"]
        )

    @patch("src.graph.academic.get_available_subjects_from_data")
    @patch("src.graph.academic.invoke_structured_llm", new_callable=AsyncMock)
    @patch("src.graph.academic._maintain_conversation_summary", new_callable=AsyncMock)
    async def test_query_rewrite_structured_failure_is_not_retried_locally(
        self,
        mock_summary,
        mock_invoke,
        mock_available_subjects,
    ):
        mock_available_subjects.return_value = ["python"]
        mock_summary.return_value = ""
        mock_invoke.side_effect = _structured_output_error(
            phase="validation_error",
            error_type="ValidationError",
            error_message="local_retrieval_query maxLength",
        )
        events: list[dict] = []
        token = set_trace_event_sink(events)
        try:
            with pytest.raises(StructuredOutputError):
                await search_query_rewriter(
                    {
                        "messages": [HumanMessage(content="Explain Python functions")],
                        "keypoints": ["Python"],
                        "subject": "python",
                        "subject_candidates": ["python"],
                        "memory_use_policy": "ignore",
                        "request_id": "req",
                        "thread_id": "thread",
                    }
                )
        finally:
            reset_trace_event_sink(token)

        assert mock_invoke.await_count == 1
        assert not any(
            event["stage"] == "query_rewrite_compliance_retry" for event in events
        )

    @patch("src.graph.academic.get_available_subjects_from_data")
    @patch("src.graph.academic.invoke_structured_llm", new_callable=AsyncMock)
    @patch("src.graph.academic._maintain_conversation_summary", new_callable=AsyncMock)
    async def test_unimplemented_output_mode_does_not_compliance_retry(
        self,
        mock_summary,
        mock_invoke,
        mock_available_subjects,
    ):
        mock_available_subjects.return_value = ["python"]
        mock_summary.return_value = ""
        mock_invoke.side_effect = _structured_output_error(
            phase="second_layer_native_json_schema_pydantic_unsupported",
            error_type="NotImplementedError",
            error_message="unsupported output mode",
        )

        with pytest.raises(StructuredOutputError):
            await search_query_rewriter(
                {
                    "messages": [HumanMessage(content="Explain Python functions")],
                    "keypoints": ["Python"],
                    "subject": "python",
                    "subject_candidates": ["python"],
                    "memory_use_policy": "ignore",
                }
            )

        assert mock_invoke.await_count == 1

    @patch("src.graph.academic.get_available_subjects_from_data")
    @patch("src.graph.academic.invoke_structured_llm", new_callable=AsyncMock)
    @patch("src.graph.academic._maintain_conversation_summary", new_callable=AsyncMock)
    async def test_always_rewrites_even_with_stale_rewritten_query(
        self, mock_summary, mock_invoke, mock_available_subjects
    ):
        """Query rewrite always runs for every new request - stale
        rewritten_query from a previous turn does NOT skip it."""
        mock_available_subjects.return_value = ["python"]
        mock_summary.return_value = ""
        parsed = SearchQueryRewriteOutput(
            local_retrieval_query="fresh rag query",
            web_research_seed_query="fresh web query",
            expanded_keypoints=["fresh"],
            reason="rewritten for new request",
            learning_goal="",
            primary_subject="python",
            subject_relation_summary="",
            retrieval_plan=[
                RetrievalPlanItem(
                    subject="python",
                    role="core_concept",
                    local_retrieval_query="fresh rag query",
                    web_research_seed_query="fresh web query",
                    priority=1.0,
                ),
            ],
        )
        mock_invoke.return_value = SimpleNamespace(
            parsed=parsed, raw_output='{"ok": true}'
        )

        result = await search_query_rewriter(
            {
                "messages": [HumanMessage(content="new request")],
                "rewritten_query": "stale retry query from previous turn",
                "subject": "python",
                "subject_candidates": ["python"],
                "memory_use_policy": "ignore",
            }
        )

        assert result["local_retrieval_query"] == "fresh rag query"
        assert result["web_research_seed_query"] == "fresh web query"
        assert result["retrieval_plan"][0]["subject"] == "python"
        # Stale rewritten_query does NOT suppress the fresh retrieval plan
        assert result["primary_subject"] == "python"
        mock_invoke.assert_awaited_once()

    @patch("src.graph.academic.get_available_subjects_from_data")
    def test_normalize_retrieval_plan_returns_debug(self, mock_available_subjects):
        mock_available_subjects.return_value = ["python", "machine_learning"]

        plan, debug = _normalize_retrieval_plan(
            [
                RetrievalPlanItem(
                    subject="", role="core_concept", local_retrieval_query="x"
                ),
                RetrievalPlanItem(
                    subject="python",
                    role="bad_role",
                    local_retrieval_query="old",
                    priority=0.1,
                ),
                RetrievalPlanItem(
                    subject="python",
                    role="implementation_tool",
                    local_retrieval_query="new",
                    priority=0.9,
                ),
                RetrievalPlanItem(
                    subject="law",
                    role="core_concept",
                    local_retrieval_query="law",
                    priority=0.8,
                ),
                RetrievalPlanItem(
                    subject="machine_learning",
                    role="core_concept",
                    local_retrieval_query="",
                    priority=0.8,
                ),
            ],
            {"subject": "python"},
        )

        assert len(plan) == 1
        assert plan[0] == {
            "subject": "python",
            "role": "implementation_tool",
            "local_retrieval_query": "new",
            "web_research_seed_query": "",
            "purpose": "",
            "relation_to_goal": "",
            "priority": 0.9,
            "retrieval_coverage_hint": "",
            "retrieval_coverage_goals": [],
            "_parent_child_priority_explicit": True,
        }
        assert debug["raw_plan_count"] == 5
        assert debug["normalized_plan_count"] == 1
        assert debug["accepted_subjects"] == ["python"]
        assert any(item["reason"] == "invalid_role" for item in debug["rejected_items"])

    @patch("src.graph.academic.get_available_subjects_from_data")
    def test_normalize_retrieval_plan_keeps_core_and_practice_roles(
        self,
        mock_available_subjects,
    ):
        mock_available_subjects.return_value = ["math"]

        plan, debug = _normalize_retrieval_plan(
            [
                RetrievalPlanItem(
                    subject="math",
                    role="core_concept",
                    local_retrieval_query="math concepts",
                    priority=0.9,
                ),
                RetrievalPlanItem(
                    subject="math",
                    role="practice",
                    local_retrieval_query="math practice exercises",
                    priority=0.8,
                ),
            ],
            {"subject": "math"},
        )

        assert debug["normalized_plan_count"] == 2
        assert set(debug["accepted_plan_keys"]) == {
            "math/core_concept",
            "math/practice",
        }
        assert {(item["subject"], item["role"]) for item in plan} == {
            ("math", "core_concept"),
            ("math", "practice"),
        }

    @patch("src.graph.academic.get_available_subjects_from_data")
    def test_normalize_retrieval_plan_dedupes_by_subject_and_role(
        self,
        mock_available_subjects,
    ):
        mock_available_subjects.return_value = ["math"]

        plan, debug = _normalize_retrieval_plan(
            [
                RetrievalPlanItem(
                    subject="math",
                    role="practice",
                    local_retrieval_query="old practice",
                    priority=0.1,
                ),
                RetrievalPlanItem(
                    subject="math",
                    role="practice",
                    local_retrieval_query="new practice",
                    priority=0.9,
                ),
                RetrievalPlanItem(
                    subject="math",
                    role="exercise",
                    local_retrieval_query="math exercise set",
                    priority=0.8,
                ),
            ],
            {"subject": "math"},
        )

        assert debug["normalized_plan_count"] == 2
        assert {(item["subject"], item["role"]) for item in plan} == {
            ("math", "practice"),
            ("math", "exercise"),
        }
        assert any(
            item["reason"] == "duplicate_subject_role_lower_priority"
            for item in debug["rejected_items"]
        )
        assert all(item["reason"] != "invalid_role" for item in debug["rejected_items"])
        assert (
            next(item for item in plan if item["role"] == "practice")[
                "local_retrieval_query"
            ]
            == "new practice"
        )

    @patch("src.graph.academic.invoke_structured_llm", new_callable=AsyncMock)
    async def test_structured_runtime_failure_raises(self, mock_invoke):
        mock_invoke.side_effect = RuntimeError("structured failure")

        with pytest.raises(RuntimeError, match="structured failure"):
            await search_query_rewriter(
                {
                    "messages": [HumanMessage(content="Python practice")],
                    "keypoints": ["Python"],
                    "subject": "python",
                    "memory_use_policy": "ignore",
                }
            )


class TestRetrievalBranchQuality:
    def test_best_doc_score_prefers_rerank_score(self):
        assert (
            _best_doc_score(
                [
                    {"raw_vector_score": 0.9, "rerank_score": 0.2},
                    {"raw_vector_score": 0.4, "rerank_score": 0.8},
                ]
            )
            == 0.8
        )

    def test_evaluates_strong_usable_weak_missing(self):
        assert (
            _evaluate_retrieval_branch(
                subject="python",
                role="core_concept",
                docs=[{"rerank_score": 0.8}],
                is_hit=True,
                subject_mismatch_count=0,
            )["branch_status"]
            == "strong"
        )
        assert (
            _evaluate_retrieval_branch(
                subject="python",
                role="core_concept",
                docs=[],
                is_hit=False,
                subject_mismatch_count=0,
            )["branch_status"]
            == "missing"
        )

    def test_select_docs_with_subject_quota_caps_subject_and_weak_docs(self):
        docs = [
            {
                "content": f"ml {i}",
                "source": f"ml{i}.pdf",
                "rerank_score": 0.9 - i * 0.01,
                "retrieval_subject": "machine_learning",
                "retrieval_priority": 0.9,
                "branch_status": "strong",
            }
            for i in range(5)
        ] + [
            {
                "content": "python weak",
                "source": "py.pdf",
                "rerank_score": 0.2,
                "retrieval_subject": "python",
                "retrieval_priority": 0.6,
                "branch_status": "weak",
            },
        ]

        selected, debug = _select_docs_with_subject_quota(
            docs, 4, primary_subject="machine_learning"
        )

        assert len(selected) == 4
        assert debug["weak_subjects"] == ["python"]


class TestRagRetrieveDualSource:
    @patch("src.graph.academic.retrieve")
    async def test_returns_local_candidates_not_context(self, mock_retrieve):
        mock_retrieve.return_value = {
            "docs": [
                {
                    "content": "Python functions",
                    "source": "python.pdf",
                    "rerank_score": 0.9,
                }
            ],
            "is_hit": True,
            "reranker_failed": False,
        }

        result = await rag_retrieve(
            {
                "messages": [HumanMessage(content="Explain Python functions")],
                "keypoints": ["Python", "functions"],
                "subject": "python",
            }
        )

        assert "context" not in result
        assert len(result["local_evidence_candidates"]) == 1
        assert result["local_evidence_candidates"][0]["source_type"] == "local_rag"
        assert result["local_evidence_originals"]
        mock_retrieve.assert_called_once_with(
            query="Python functions", subject="python", top_k=3
        )

    @patch("src.graph.academic.retrieve")
    async def test_core_and_practice_branches_generate_unique_stable_local_ids(
        self, mock_retrieve
    ):
        mock_retrieve.side_effect = [
            {
                "docs": [
                    {
                        "content": "Python functions explain parameters.",
                        "source": "python.pdf",
                        "metadata": {"chunk_id": "intro"},
                        "rerank_score": 0.9,
                    }
                ],
                "is_hit": True,
            },
            {
                "docs": [
                    {
                        "content": "Python function practice exercises.",
                        "source": "python.pdf",
                        "metadata": {"chunk_id": "practice"},
                        "rerank_score": 0.8,
                    }
                ],
                "is_hit": True,
            },
        ]

        result = await rag_retrieve(
            {
                "messages": [HumanMessage(content="给我 Python mindmap 和 quiz")],
                "requested_resource_types": ["mindmap", "quiz"],
                "retrieval_plan": [
                    {
                        "subject": "python",
                        "role": "core_concept",
                        "local_retrieval_query": "Python functions concept",
                        "priority": 0.9,
                    },
                    {
                        "subject": "python",
                        "role": "practice",
                        "local_retrieval_query": "Python functions practice",
                        "priority": 0.8,
                    },
                ],
            }
        )

        candidate_ids = [
            candidate["evidence_id"]
            for candidate in result["local_evidence_candidates"]
        ]
        assert len(candidate_ids) == len(set(candidate_ids)) == 2
        assert all(
            candidate_id.startswith("local:python:") for candidate_id in candidate_ids
        )
        assert "local:python:0" not in candidate_ids
        assert {
            candidate["role"] for candidate in result["local_evidence_candidates"]
        } == {
            "core_concept",
            "practice",
        }

    @patch("src.graph.academic.retrieve")
    async def test_math_core_and_practice_branches_do_not_reuse_rank_ids(
        self, mock_retrieve
    ):
        mock_retrieve.side_effect = [
            {
                "docs": [
                    {
                        "content": "线性代数核心概念",
                        "source": "math.pdf",
                        "metadata": {"chunk_id": "core"},
                        "rerank_score": 0.9,
                    }
                ],
                "is_hit": True,
            },
            {
                "docs": [
                    {
                        "content": "线性代数练习题",
                        "source": "math.pdf",
                        "metadata": {"chunk_id": "practice"},
                        "rerank_score": 0.8,
                    }
                ],
                "is_hit": True,
            },
        ]

        result = await rag_retrieve(
            {
                "messages": [HumanMessage(content="线性代数思维导图和练习题")],
                "requested_resource_types": ["mindmap", "quiz"],
                "retrieval_plan": [
                    {
                        "subject": "math",
                        "role": "core_concept",
                        "local_retrieval_query": "linear algebra concept",
                        "priority": 0.9,
                    },
                    {
                        "subject": "math",
                        "role": "practice",
                        "local_retrieval_query": "linear algebra practice",
                        "priority": 0.8,
                    },
                ],
            }
        )

        candidate_ids = [
            candidate["evidence_id"]
            for candidate in result["local_evidence_candidates"]
        ]
        assert len(candidate_ids) == len(set(candidate_ids)) == 2
        assert all(
            candidate_id.startswith("local:math:") for candidate_id in candidate_ids
        )
        assert "local:math:0" not in candidate_ids


class TestEvidenceCandidateIdentity:
    def test_same_chunk_across_branches_merges_and_preserves_metadata(self):
        docs = [
            {
                "content": "Python functions shared chunk.",
                "source": "python.pdf",
                "metadata": {"chunk_id": "shared"},
                "retrieval_query": "Python functions concept",
                "retrieval_priority": 0.9,
                "rerank_score": 0.9,
            }
        ]
        core_candidates = _build_local_evidence_candidates(
            docs=docs,
            subject="python",
            role="core_concept",
            branch_index=0,
            branch_status="strong",
            branch_status_score_source="rerank_score",
        )
        practice_candidates = _build_local_evidence_candidates(
            docs=[
                {
                    **docs[0],
                    "retrieval_query": "Python functions practice",
                    "retrieval_priority": 0.8,
                }
            ],
            subject="python",
            role="practice",
            branch_index=1,
            branch_status="strong",
            branch_status_score_source="rerank_score",
        )

        merged, debug = _dedupe_and_merge_local_candidates(
            [*core_candidates, *practice_candidates]
        )

        assert len(merged) == 1
        metadata = merged[0].metadata
        assert set(metadata["covered_roles"]) == {"core_concept", "practice"}
        assert metadata["covered_branch_indices"] == [0, 1]
        assert set(metadata["covered_retrieval_queries"]) == {
            "Python functions concept",
            "Python functions practice",
        }
        assert metadata["max_priority"] == 0.9
        assert metadata["source_type"] == "local_rag"
        assert len(metadata["merged_from_ids"]) == 2
        assert all(":branch" in origin_id for origin_id in metadata["merged_from_ids"])
        assert debug["local_candidate_raw_count"] == 2
        assert debug["local_candidate_deduped_count"] == 1
        assert debug["deduped_chunk_count"] == 1

    def test_same_evidence_id_with_different_identity_fails_fast(self):
        first = EvidenceCandidate(
            evidence_id="local:python:source:chunk",
            source_type="local_rag",
            subject="python",
            role="core_concept",
            content_preview="First chunk.",
            metadata={
                "dedupe_key": "python:source-a:chunk-a",
                "source_hash": "source-a",
                "content_hash": "content-a",
            },
        )
        second = EvidenceCandidate(
            evidence_id="local:python:source:chunk",
            source_type="local_rag",
            subject="python",
            role="practice",
            content_preview="Different chunk.",
            metadata={
                "dedupe_key": "python:source-b:chunk-b",
                "source_hash": "source-b",
                "content_hash": "content-b",
            },
        )

        with pytest.raises(RuntimeError, match="duplicate evidence_id collision"):
            _dedupe_and_merge_local_candidates([first, second])

    def test_full_content_hash_fallback_does_not_use_truncated_preview(self):
        shared_prefix = "A" * 900
        candidates = _build_local_evidence_candidates(
            docs=[
                {"content": f"{shared_prefix} first tail", "source": "python.pdf"},
                {"content": f"{shared_prefix} second tail", "source": "python.pdf"},
            ],
            subject="python",
            role="core_concept",
            branch_index=0,
            branch_status="strong",
            branch_status_score_source="rerank_score",
        )

        assert candidates[0].content_preview == candidates[1].content_preview
        assert candidates[0].evidence_id != candidates[1].evidence_id
        assert candidates[0].metadata["chunk_identity_kind"] == "content_hash"
        assert (
            candidates[0].metadata["content_hash8"]
            != candidates[1].metadata["content_hash8"]
        )

    def test_safe_id_part_does_not_alias_normalize_roles(self):
        assert _safe_evidence_id_part("practice") == "practice"
        assert _safe_evidence_id_part("exercise") == "exercise"
        assert _safe_evidence_id_part("Practice Quiz") == "practice_quiz"

    def test_first_batch_reorders_for_valid_web_and_core_practice_without_dropping_candidates(
        self,
    ):
        local_core = EvidenceCandidate(
            evidence_id="local:python:source1:chunk1",
            source_type="local_rag",
            subject="python",
            role="core_concept",
            content_preview="Core concepts.",
            rerank_score=0.9,
            metadata={"covered_roles": ["core_concept"], "source_type": "local_rag"},
        )
        local_practice = EvidenceCandidate(
            evidence_id="local:python:source1:chunk2",
            source_type="local_rag",
            subject="python",
            role="practice",
            content_preview="Practice exercises.",
            rerank_score=0.8,
            metadata={"covered_roles": ["practice"], "source_type": "local_rag"},
        )
        invalid_web = EvidenceCandidate(
            evidence_id="web:invalid",
            source_type="web",
            subject="python",
            role="practice",
            content_preview="",
            tavily_score=1.0,
            metadata={
                "fetch_status": "failed",
                "content_chars": 0,
                "covered_roles": ["practice"],
            },
        )
        valid_web = EvidenceCandidate(
            evidence_id="web:valid",
            source_type="web",
            subject="python",
            role="practice",
            content_preview="Useful web evidence.",
            tavily_score=0.1,
            metadata={
                "fetch_status": "success",
                "content_chars": 120,
                "covered_roles": ["practice"],
            },
        )

        ordered, debug = _order_evidence_candidates_for_grading(
            [invalid_web, local_core, valid_web, local_practice],
            batch_size=3,
            requested_resource_types=["mindmap", "quiz"],
        )

        assert len(ordered) == 4
        assert {candidate.evidence_id for candidate in ordered} == {
            "local:python:source1:chunk1",
            "local:python:source1:chunk2",
            "web:valid",
            "web:invalid",
        }
        assert debug["candidate_count_preserved"] is True
        assert "web:valid" in debug["first_batch_evidence_ids"]
        assert "web:invalid" not in debug["first_batch_evidence_ids"]
        assert len(debug["first_batch_evidence_ids"]) == len(
            set(debug["first_batch_evidence_ids"])
        )
        assert set(debug["first_batch_role_coverage"]) >= {"core_concept", "practice"}
        assert debug["first_batch_source_type_counts"]["web"] == 1


def _web_research_structured_result(
    parsed, *, node_name: str, schema_name: str
) -> StructuredLLMResult:
    return StructuredLLMResult(
        success=True,
        parsed=parsed,
        node_name=node_name,
        llm_node=node_name,
        schema_name=schema_name,
        provider="test",
        model="test",
        output_mode="deepseek_tool_call_strict",
        raw_output=parsed.model_dump_json() if parsed is not None else "{}",
    )


async def _fake_web_research_v2_llm(**kwargs):
    if kwargs["node_name"] == "web_research_planner":
        planner_prompt = str(kwargs["messages"][-1]["content"])
        subject_matches = re.findall(r'"subject":\s*"([^"]+)"', planner_prompt)
        subject = subject_matches[-1] if subject_matches else "python"
        plan = WebResearchPlan(
            tasks=[
                WebResearchTask(
                    task_id=f"task-{subject}-0",
                    subject=subject,
                    role="supporting_context",
                    purpose=f"Find {subject} tutorial material.",
                    search_query=f"{subject} tutorial",
                    reason=f"Need web evidence for the requested {subject} topic.",
                    priority=0.8,
                )
            ]
        )
        return _web_research_structured_result(
            plan,
            node_name="web_research_planner",
            schema_name="WebResearchPlan",
        )

    source_ids = re.findall(
        r'"source_id":\s*"([^"]+)"', str(kwargs["messages"][-1]["content"])
    )
    batch = WebSourceSummaryBatch(
        summaries=[
            WebSourceSummary(
                source_id=source_id,
                keep=True,
                summary="Python tutorial source.",
                coverage_points=["Python function basics"],
                reason="Relevant tutorial result.",
                evidence_type="tutorial",
                use_case="background_context",
                relevance="medium",
                usefulness="medium",
                risk="low",
            )
            for source_id in source_ids
        ]
    )
    return _web_research_structured_result(
        batch,
        node_name="web_source_summarizer",
        schema_name="WebSourceSummaryBatch",
    )


class TestWebSearchDualSource:
    @patch("src.graph.academic.web_search_fn")
    async def test_returns_web_candidates_not_context(self, mock_search, monkeypatch):
        mock_search.return_value = {
            "provider": "tavily",
            "ok": True,
            "status_code": 200,
            "elapsed_ms": 10,
            "results": [
                {
                    "content": "Python tutorial",
                    "title": "Python",
                    "url": "https://example.com",
                }
            ],
        }
        monkeypatch.setattr(
            "src.graph.academic.invoke_structured_llm", _fake_web_research_v2_llm
        )

        result = await web_search(
            {
                "messages": [HumanMessage(content="Explain Python functions")],
                "web_research_seed_query": "Python functions tutorial",
                "subject": "python",
            }
        )

        assert "context" not in result
        assert len(result["web_evidence_candidates"]) == 1
        assert result["web_evidence_candidates"][0]["source_type"] == "web"
        assert result["web_evidence_originals"]

    @patch(
        "src.graph.academic.web_search_fn", side_effect=RuntimeError("network error")
    )
    async def test_returns_empty_candidates_on_search_exception(
        self, mock_search, monkeypatch
    ):
        monkeypatch.setattr(
            "src.graph.academic.invoke_structured_llm", _fake_web_research_v2_llm
        )

        with pytest.raises(RuntimeError, match="fallback is disabled") as exc_info:
            await web_search({"messages": [HumanMessage(content="test")]})

        debug = getattr(exc_info.value, "web_research_debug")
        assert debug["status"] == "failed"
        assert debug["used_fallback"] is False


class TestEvidenceMemorySummary:
    def test_context_item_from_evidence_exports_llm_score_for_ce(self):
        candidate = EvidenceCandidate(
            evidence_id="local:python:score",
            source_type="local_rag",
            provider="chroma_rag",
            subject="python",
            role="core_concept",
            title="Python functions",
            content_preview="Function notes.",
        )
        judge_item = EvidenceJudgeItem(
            evidence_id="local:python:score",
            keep=True,
            final_quality="high",
            relevance="high",
            authority="medium",
            usefulness="high",
            risk="low",
            evidence_score=0.87,
            score_reason="Directly supports the requested review material.",
            evidence_type="local_textbook_chunk",
            use_case="core_evidence",
            coverage_contribution="Covers Python function concepts.",
            reason="Useful course evidence.",
        )

        doc = _context_item_from_evidence(
            candidate=candidate,
            judge_item=judge_item,
            original={"content": "Function notes."},
        )

        assert doc["evidence_score"] == pytest.approx(0.87)
        assert doc["relevance_score"] == pytest.approx(0.87)
        assert doc["score"] == pytest.approx(0.87)
        assert doc["score_source"] == "evidence_item_grader"
        assert doc["score_scale"] == "0-1"
        assert doc["score_type"] == "task_relevance"

    def test_builder_uses_current_call_candidates_and_originals(self):
        parsed = EvidenceJudgeOutput(
            overall_evidence_state="sufficient",
            need_more_web_research=False,
            judged_evidence=[
                EvidenceJudgeItem(
                    evidence_id="current",
                    keep=True,
                    final_quality="high",
                    evidence_score=0.93,
                    score_reason="Directly supports the current quiz resource.",
                    use_case="core_evidence",
                    coverage_contribution="covers Python function basics",
                    reason="useful course note",
                )
            ],
            decision_summary="Current evidence is enough.",
        )
        candidate = EvidenceCandidate(
            evidence_id="current",
            source_type="web",
            provider="tavily",
            subject="python",
            source="Current source",
            url="",
            content_preview="This preview must not be persisted as raw content.",
        )
        state = {
            "subject": "python",
            "requested_resource_type": "quiz",
            "evidence_candidates": [
                {
                    "evidence_id": "current",
                    "source_type": "local_rag",
                    "subject": "math",
                    "source": "Stale source",
                    "url": "https://stale.example",
                    "content": "stale raw doc",
                }
            ],
            "request_id": "req",
            "thread_id": "thread",
        }
        events: list[dict] = []
        token = set_trace_event_sink(events)
        try:
            evidence_entries, gap_entries = build_evidence_memory_summary(
                state=state,
                parsed=parsed,
                candidates=[candidate],
                originals={
                    "current": {
                        "source": "Original source fallback",
                        "url": "https://current.example",
                        "content": "raw original content",
                    }
                },
                request_id="req",
                thread_id="thread",
            )
        finally:
            reset_trace_event_sink(token)

        assert gap_entries == []
        entry = evidence_entries[0]
        kept = entry["kept_evidence_summary"][0]
        assert kept["source"] == "Current source"
        assert kept["url"] == "https://current.example"
        assert kept["subject"] == "python"
        assert kept["source_type"] == "web"
        assert kept["final_quality"] == "high"
        assert kept["use_case"] == "core_evidence"
        assert kept["coverage_contribution"] == "covers Python function basics"
        assert "content" not in entry
        assert "raw_docs" not in entry
        assert "content_preview" not in kept
        trace = next(
            event
            for event in events
            if event["stage"] == "evidence_memory_summary_build"
        )
        assert trace["candidate_metadata_source"] == "current_call_arguments"
        assert trace["candidate_count"] == 1
        assert trace["original_count"] == 1


class TestFormatHelpers:
    def test_format_retrieved_empty(self):
        assert _format_retrieved([]).strip()

    def test_format_retrieved_with_docs(self):
        output = _format_retrieved(
            [
                {"content": "Doc one", "source": "one.pdf", "rerank_score": 0.9},
                {"content": "Doc two", "source": "two.pdf", "rerank_score": 0.8},
            ]
        )
        assert "[1]" in output
        assert "one.pdf" in output

    def test_format_web_research_context_empty(self):
        assert _format_web_research_context([]).strip()

    def test_format_web_research_context_with_results(self):
        output = _format_web_research_context(
            [
                {
                    "type": "web_evidence",
                    "title": "Course project plan",
                    "url": "https://example.com/1",
                    "content": "project plan",
                },
            ]
        )
        assert "[1]" in output
        assert "Course project plan" in output


class TestGenerateAnswer:
    @patch("src.graph.academic.invoke_plain_llm_fail_fast")
    async def test_generates_ai_message(self, mock_invoke_plain):
        mock_invoke_plain.return_value = "answer"

        result = await generate_answer(
            {
                "messages": [HumanMessage(content="question")],
                "context": [{"type": "rag", "content": "doc"}],
            }
        )

        assert result["messages"][0].content == "answer"
        assert mock_invoke_plain.await_args.kwargs["llm_node"] == "academic"

    @patch("src.graph.academic.invoke_plain_llm_fail_fast")
    async def test_handles_empty_context(self, mock_invoke_plain):
        mock_invoke_plain.return_value = "answer without context"

        result = await generate_answer(
            {
                "messages": [HumanMessage(content="question")],
                "context": [],
            }
        )

        assert result["messages"][0].content == "answer without context"
