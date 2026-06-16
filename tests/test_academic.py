"""Unit tests for the current academic evidence path."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.config import get_setting
from src.graph.academic import (
    _best_doc_score,
    _deterministic_memory_use_decision,
    _evaluate_retrieval_branch,
    _format_retrieved,
    _format_search,
    _normalize_retrieval_plan,
    _select_docs_with_subject_quota,
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
from src.llm.structured_output import StructuredLLMResult, StructuredOutputError, get_llm_output_mode
from src.observability.a3_trace import reset_trace_event_sink, set_trace_event_sink


def _valid_query_rewrite_output() -> SearchQueryRewriteOutput:
    return SearchQueryRewriteOutput(
        rag_query="Python functions 参数 return value scope",
        web_search_query="Python function parameters return value course notes tutorial",
        expanded_keypoints=["Python functions", "参数 parameter", "return value"],
        reason="Expanded concise bilingual retrieval terms.",
        learning_goal="Understand Python functions",
        primary_subject="python",
        subject_relation_summary="single subject",
        retrieval_plan=[
            RetrievalPlanItem(
                subject="python",
                role="core_concept",
                rag_query="Python functions 参数 return value scope",
                web_search_query="Python function parameters tutorial course notes",
                priority=0.9,
                expected_coverage=["function definition", "parameter passing"],
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
            business_validation_error=error_message if phase == "business_validation_error" else "",
        )
    )


class TestAcademicRouterRetry:
    async def test_returns_empty_on_first_run(self):
        result = await academic_router({"messages": [HumanMessage(content="test")], "retry_count": 0})
        assert "context" not in result

    async def test_clears_context_on_retry(self):
        result = await academic_router({"messages": [HumanMessage(content="test")], "retry_count": 1})
        assert result["context"] is CONTEXT_CLEAR


class TestRewriteQuery:
    @patch("src.graph.llm.get_node_llm")
    async def test_produces_rewritten_query(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="improved retrieval query"))
        mock_get_llm.return_value = mock_llm

        result = await rewrite_query({
            "messages": [HumanMessage(content="original question")],
            "hallucination_reason": "fabricated detail",
            "retry_count": 1,
            "request_id": "test-req",
            "thread_id": "test-thread",
        })

        assert result["rewritten_query"] == "improved retrieval query"
        assert result["retrieval_plan"] == []

    @patch("src.graph.llm.get_node_llm")
    async def test_fail_fast_on_retry_rewrite_failure(self, mock_get_llm):
        """Rewrite query now fails fast — no fallback to original query."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM error"))
        mock_get_llm.return_value = mock_llm

        with pytest.raises(RuntimeError, match="LLM error"):
            await rewrite_query({
                "messages": [HumanMessage(content="original question")],
                "hallucination_reason": "bad",
                "retry_count": 1,
                "request_id": "test-req",
                "thread_id": "test-thread",
            })


class TestMemoryUseDecision:
    def test_memory_use_decider_has_explicit_openrouter_config(self):
        assert get_setting("llm.memory_use_decider.provider") == "openrouter"
        assert get_setting("llm.memory_use_decider.base_url") == "https://openrouter.ai/api/v1"
        assert get_setting("llm.memory_use_decider.api_key_env") == "OPENROUTER_API_KEY"
        assert get_llm_output_mode("memory_use_decider") == "native_json_schema_pydantic"

    def test_empty_memory_ignores_without_prompt(self):
        decision = _deterministic_memory_use_decision("重新给我一份学习计划", selected_memory_count=0)
        assert decision is not None
        assert decision.decision == "ignore"

    def test_explicit_history_reference_uses_memory(self):
        decision = _deterministic_memory_use_decision("结合之前的内容，给我一份学习计划", selected_memory_count=1)
        assert decision is not None
        assert decision.decision == "use"

    def test_explicit_history_exclusion_ignores_memory(self):
        decision = _deterministic_memory_use_decision("不要参考之前，给我一份学习计划", selected_memory_count=1)
        assert decision is not None
        assert decision.decision == "ignore"

    def test_ambiguous_revision_asks_user_when_memory_exists(self):
        decision = _deterministic_memory_use_decision("重新给我一份学习计划", selected_memory_count=1)
        assert decision is not None
        assert decision.decision == "ask_user"
        assert decision.question_to_user

    async def test_memory_use_decider_ignores_when_no_memory(self):
        result = await memory_use_decider({
            "messages": [HumanMessage(content="重新给我一份学习计划")],
            "evidence_summary_memory": [],
            "subject": "other",
            "requested_resource_type": "study_plan",
            "request_id": "req",
            "thread_id": "thread",
        })
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
        result = await memory_use_decider({
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
        })

        mock_interrupt.assert_called_once()
        assert result["memory_use_policy"] == "ignore"
        assert result["eligible_evidence_memory_count"] == 1

    @patch("src.graph.academic.interrupt")
    async def test_explicit_history_use_bypasses_popup(self, mock_interrupt):
        result = await memory_use_decider({
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
        })

        mock_interrupt.assert_not_called()
        assert result["memory_use_policy"] == "use"
        assert result["selected_evidence_memory_summaries"]


class TestSearchQueryRewriter:
    def test_query_rewrite_schema_exposes_length_limits(self):
        schema = SearchQueryRewriteOutput.model_json_schema()
        props = schema["properties"]
        plan_props = schema["$defs"]["RetrievalPlanItem"]["properties"]

        assert schema["additionalProperties"] is False
        assert schema["$defs"]["RetrievalPlanItem"]["additionalProperties"] is False
        assert props["rag_query"]["maxLength"] == 240
        assert props["web_search_query"]["maxLength"] == 180
        assert props["expanded_keypoints"]["maxItems"] == 8
        assert props["expanded_keypoints"]["items"]["maxLength"] == 120
        assert props["retrieval_plan"]["maxItems"] == 4
        assert props["memory_context_notes"]["maxItems"] == 5
        assert props["memory_context_notes"]["items"]["maxLength"] == 240
        assert plan_props["expected_coverage"]["maxItems"] == 8
        assert plan_props["expected_coverage"]["items"]["maxLength"] == 120

    def test_search_query_rewriter_structured_retry_is_disabled_for_local_retry(self):
        assert get_setting("llm_outputs.search_query_rewriter.max_retries") == 0
        assert get_setting("provider_transport_retry.max_retries") == 2

    def test_overlong_query_fields_fail_schema_validation(self):
        with pytest.raises(Exception):
            SearchQueryRewriteOutput(
                rag_query="x" * 241,
                web_search_query="Python function tutorial",
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
                    "rag_query_web_search_query": "big data quiz",
                    "retrieval_plan_subject_role_rag_query_web_search_query_purpose_relation_to_goal_coverage_hint_expected_coverage_priority": [],
                }
            )

    def test_repeated_chinese_phrases_fail_business_validation(self):
        parsed = _valid_query_rewrite_output()
        parsed.rag_query = "检索意图 资源类型 练习题 答案 解析 实操任务 " * 3

        error = validate_search_query_rewrite_output(parsed, memory_use_policy="ignore")

        assert "repeated query phrase" in error
        assert "检索意图" in error

    def test_repeated_english_ngram_fails_business_validation(self):
        parsed = _valid_query_rewrite_output()
        parsed.web_search_query = "python function parameter return " * 3

        error = validate_search_query_rewrite_output(parsed, memory_use_policy="ignore")

        assert "repeated query ngram" in error

    def test_valid_concise_query_passes_business_validation(self):
        assert validate_search_query_rewrite_output(
            _valid_query_rewrite_output(),
            memory_use_policy="ignore",
        ) == ""

    @patch("src.graph.academic.get_available_subjects_from_data")
    @patch("src.graph.academic.invoke_structured_llm", new_callable=AsyncMock)
    async def test_produces_rag_web_queries_and_plan(self, mock_invoke, mock_available_subjects):
        mock_available_subjects.return_value = ["python", "machine_learning"]
        parsed = SearchQueryRewriteOutput(
            rag_query="Python functions parameters return values",
            web_search_query="Python functions course notes tutorial",
            expanded_keypoints=["Python", "functions"],
            reason="expanded for bilingual retrieval",
            learning_goal="Understand Python functions",
            primary_subject="python",
            subject_relation_summary="single subject",
            retrieval_plan=[
                RetrievalPlanItem(
                    subject="python",
                    role="core_concept",
                    rag_query="Python functions",
                    web_search_query="Python functions tutorial",
                    priority=0.8,
                ),
            ],
        )
        mock_invoke.return_value = SimpleNamespace(parsed=parsed, raw_output='{"ok": true}')

        result = await search_query_rewriter({
            "messages": [HumanMessage(content="Explain Python functions")],
            "keypoints": ["Python"],
            "subject": "python",
            "subject_candidates": ["python"],
            "memory_use_policy": "ignore",
        })

        assert result["search_rag_query"] == "Python functions parameters return values"
        assert result["search_web_query"] == "Python functions course notes tutorial"
        assert result["retrieval_plan"][0]["subject"] == "python"
        assert result["primary_subject"] == "python"
        mock_invoke.assert_awaited_once()

    @patch("src.graph.academic.get_available_subjects_from_data")
    @patch("src.graph.academic.invoke_structured_llm", new_callable=AsyncMock)
    @patch("src.graph.academic._maintain_conversation_summary", new_callable=AsyncMock)
    async def test_local_compliance_retry_succeeds_once(
        self,
        mock_summary,
        mock_invoke,
        mock_available_subjects,
    ):
        mock_available_subjects.return_value = ["python"]
        mock_summary.return_value = ""
        parsed = _valid_query_rewrite_output()
        mock_invoke.side_effect = [
            _structured_output_error(
                phase="business_validation_error",
                error_type="BusinessValidationError",
                error_message="rag_query repeated query phrase: 检索意图",
            ),
            SimpleNamespace(parsed=parsed, raw_output='{"ok": true}'),
        ]
        events: list[dict] = []
        token = set_trace_event_sink(events)
        try:
            result = await search_query_rewriter({
                "messages": [HumanMessage(content="Explain Python functions")],
                "keypoints": ["Python"],
                "subject": "python",
                "subject_candidates": ["python"],
                "memory_use_policy": "ignore",
                "request_id": "req",
                "thread_id": "thread",
            })
        finally:
            reset_trace_event_sink(token)

        assert result["search_rag_query"] == parsed.rag_query
        assert mock_invoke.await_count == 2
        assert mock_invoke.await_args_list[1].kwargs["fallback_modes"] == []
        retry_event = next(event for event in events if event["stage"] == "query_rewrite_compliance_retry")
        assert retry_event["success"] is True
        memory_event = next(event for event in events if event["stage"] == "query_rewrite_memory_use")
        assert memory_event["memory_prompt_injected"] is False
        assert memory_event["memory_used_for_retrieval"] == memory_event["llm_reported_memory_used_for_retrieval"]

    @patch("src.graph.academic.get_available_subjects_from_data")
    @patch("src.graph.academic.invoke_structured_llm", new_callable=AsyncMock)
    @patch("src.graph.academic._maintain_conversation_summary", new_callable=AsyncMock)
    async def test_local_compliance_retry_fails_fast_after_retry_failure(
        self,
        mock_summary,
        mock_invoke,
        mock_available_subjects,
    ):
        mock_available_subjects.return_value = ["python"]
        mock_summary.return_value = ""
        mock_invoke.side_effect = [
            _structured_output_error(
                phase="parsing_error",
                error_type="JSONDecodeError",
                error_message="invalid json",
            ),
            _structured_output_error(
                phase="validation_error",
                error_type="ValidationError",
                error_message="rag_query maxLength",
            ),
        ]
        events: list[dict] = []
        token = set_trace_event_sink(events)
        try:
            with pytest.raises(StructuredOutputError):
                await search_query_rewriter({
                    "messages": [HumanMessage(content="Explain Python functions")],
                    "keypoints": ["Python"],
                    "subject": "python",
                    "subject_candidates": ["python"],
                    "memory_use_policy": "ignore",
                    "request_id": "req",
                    "thread_id": "thread",
                })
        finally:
            reset_trace_event_sink(token)

        assert mock_invoke.await_count == 2
        retry_event = next(event for event in events if event["stage"] == "query_rewrite_compliance_retry")
        assert retry_event["success"] is False
        assert retry_event["final_failure_phase"] == "validation_error"

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
            await search_query_rewriter({
                "messages": [HumanMessage(content="Explain Python functions")],
                "keypoints": ["Python"],
                "subject": "python",
                "subject_candidates": ["python"],
                "memory_use_policy": "ignore",
            })

        assert mock_invoke.await_count == 1

    @patch("src.graph.academic.get_available_subjects_from_data")
    @patch("src.graph.academic.invoke_structured_llm", new_callable=AsyncMock)
    @patch("src.graph.academic._maintain_conversation_summary", new_callable=AsyncMock)
    async def test_always_rewrites_even_with_stale_rewritten_query(
        self, mock_summary, mock_invoke, mock_available_subjects
    ):
        """Query rewrite always runs for every new request — stale
        rewritten_query from a previous turn does NOT skip it."""
        mock_available_subjects.return_value = ["python"]
        mock_summary.return_value = ""
        parsed = SearchQueryRewriteOutput(
            rag_query="fresh rag query",
            web_search_query="fresh web query",
            expanded_keypoints=["fresh"],
            reason="rewritten for new request",
            learning_goal="",
            primary_subject="python",
            subject_relation_summary="",
            retrieval_plan=[
                RetrievalPlanItem(
                    subject="python",
                    role="core_concept",
                    rag_query="fresh rag query",
                    web_search_query="fresh web query",
                    priority=1.0,
                ),
            ],
        )
        mock_invoke.return_value = SimpleNamespace(parsed=parsed, raw_output='{"ok": true}')

        result = await search_query_rewriter({
            "messages": [HumanMessage(content="new request")],
            "rewritten_query": "stale retry query from previous turn",
            "subject": "python",
            "subject_candidates": ["python"],
            "memory_use_policy": "ignore",
        })

        assert result["search_rag_query"] == "fresh rag query"
        assert result["search_web_query"] == "fresh web query"
        assert result["retrieval_plan"][0]["subject"] == "python"
        # Stale rewritten_query does NOT suppress the fresh retrieval plan
        assert result["primary_subject"] == "python"
        mock_invoke.assert_awaited_once()

    @patch("src.graph.academic.get_available_subjects_from_data")
    def test_normalize_retrieval_plan_returns_debug(self, mock_available_subjects):
        mock_available_subjects.return_value = ["python", "machine_learning"]

        plan, debug = _normalize_retrieval_plan(
            [
                RetrievalPlanItem(subject="", role="core_concept", rag_query="x"),
                RetrievalPlanItem(subject="python", role="bad_role", rag_query="old", priority=0.1),
                RetrievalPlanItem(subject="python", role="implementation_tool", rag_query="new", priority=0.9),
                RetrievalPlanItem(subject="law", role="core_concept", rag_query="law", priority=0.8),
                RetrievalPlanItem(subject="machine_learning", role="core_concept", rag_query="", priority=0.8),
            ],
            {"subject": "python"},
        )

        assert len(plan) == 1
        assert plan[0] == {
            "subject": "python",
            "role": "implementation_tool",
            "rag_query": "new",
            "web_search_query": "",
            "purpose": "",
            "relation_to_goal": "",
            "priority": 0.9,
            "coverage_hint": "",
            "expected_coverage": [],
        }
        assert debug["raw_plan_count"] == 5
        assert debug["normalized_plan_count"] == 1
        assert debug["accepted_subjects"] == ["python"]

    @patch("src.graph.academic.invoke_structured_llm", new_callable=AsyncMock)
    async def test_structured_runtime_failure_raises(self, mock_invoke):
        mock_invoke.side_effect = RuntimeError("structured failure")

        with pytest.raises(RuntimeError, match="structured failure"):
            await search_query_rewriter({
                "messages": [HumanMessage(content="Python practice")],
                "keypoints": ["Python"],
                "subject": "python",
                "memory_use_policy": "ignore",
            })


class TestRetrievalBranchQuality:
    def test_best_doc_score_prefers_rerank_score(self):
        assert _best_doc_score([
            {"raw_vector_score": 0.9, "rerank_score": 0.2},
            {"raw_vector_score": 0.4, "rerank_score": 0.8},
        ]) == 0.8

    def test_evaluates_strong_usable_weak_missing(self):
        assert _evaluate_retrieval_branch(
            subject="python",
            role="core_concept",
            docs=[{"rerank_score": 0.8}],
            is_hit=True,
            subject_mismatch_count=0,
        )["branch_status"] == "strong"
        assert _evaluate_retrieval_branch(
            subject="python",
            role="core_concept",
            docs=[],
            is_hit=False,
            subject_mismatch_count=0,
        )["branch_status"] == "missing"

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

        selected, debug = _select_docs_with_subject_quota(docs, 4, primary_subject="machine_learning")

        assert len(selected) == 4
        assert debug["weak_subjects"] == ["python"]


class TestRagRetrieveDualSource:
    @patch("src.graph.academic.retrieve")
    async def test_returns_local_candidates_not_context(self, mock_retrieve):
        mock_retrieve.return_value = {
            "docs": [{"content": "Python functions", "source": "python.pdf", "rerank_score": 0.9}],
            "is_hit": True,
            "reranker_failed": False,
        }

        result = await rag_retrieve({
            "messages": [HumanMessage(content="Explain Python functions")],
            "keypoints": ["Python", "functions"],
            "subject": "python",
        })

        assert "context" not in result
        assert len(result["local_evidence_candidates"]) == 1
        assert result["local_evidence_candidates"][0]["source_type"] == "local_rag"
        assert result["local_evidence_originals"]
        mock_retrieve.assert_called_once_with(query="Python functions", subject="python", top_k=3)


class TestWebSearchDualSource:
    @patch("src.graph.academic.web_search_fn")
    async def test_returns_web_candidates_not_context(self, mock_search):
        mock_search.return_value = {
            "provider": "tavily",
            "ok": True,
            "status_code": 200,
            "elapsed_ms": 10,
            "results": [{"content": "Python tutorial", "title": "Python", "url": "https://example.com"}],
        }

        result = await web_search({
            "messages": [HumanMessage(content="Explain Python functions")],
            "search_web_query": "Python functions tutorial",
            "subject": "python",
        })

        assert "context" not in result
        assert len(result["web_evidence_candidates"]) == 1
        assert result["web_evidence_candidates"][0]["source_type"] == "web"
        assert result["web_evidence_originals"]

    @patch("src.graph.academic.web_search_fn", side_effect=RuntimeError("network error"))
    async def test_returns_empty_candidates_on_search_exception(self, mock_search):
        result = await web_search({"messages": [HumanMessage(content="test")]})

        assert result["web_evidence_candidates"] == []
        assert result["web_evidence_originals"] == {}


class TestEvidenceMemorySummary:
    def test_builder_uses_current_call_candidates_and_originals(self):
        parsed = EvidenceJudgeOutput(
            overall_evidence_state="sufficient",
            need_more_web_search=False,
            judged_evidence=[
                EvidenceJudgeItem(
                    evidence_id="current",
                    keep=True,
                    final_quality="high",
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
        trace = next(event for event in events if event["stage"] == "evidence_memory_summary_build")
        assert trace["candidate_metadata_source"] == "current_call_arguments"
        assert trace["candidate_count"] == 1
        assert trace["original_count"] == 1


class TestFormatHelpers:
    def test_format_retrieved_empty(self):
        assert _format_retrieved([]).strip()

    def test_format_retrieved_with_docs(self):
        output = _format_retrieved([
            {"content": "Doc one", "source": "one.pdf", "rerank_score": 0.9},
            {"content": "Doc two", "source": "two.pdf", "rerank_score": 0.8},
        ])
        assert "[1]" in output
        assert "one.pdf" in output

    def test_format_search_empty(self):
        assert _format_search([]).strip()

    def test_format_search_with_results(self):
        output = _format_search([
            {"title": "Course project plan", "url": "https://example.com/1", "content": "project plan"},
        ])
        assert "[1]" in output
        assert "Course project plan" in output


class TestGenerateAnswer:
    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    async def test_generates_ai_message(self, mock_get_llm, mock_get_fallback):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="answer"))
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = None

        result = await generate_answer({
            "messages": [HumanMessage(content="question")],
            "context": [{"type": "rag", "content": "doc"}],
        })

        assert result["messages"][0].content == "answer"

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    async def test_handles_empty_context(self, mock_get_llm, mock_get_fallback):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="answer without context"))
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = None

        result = await generate_answer({
            "messages": [HumanMessage(content="question")],
            "context": [],
        })

        assert result["messages"][0].content == "answer without context"
