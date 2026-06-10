"""Unit tests for SubGraph A — Academic Learning Assistant nodes."""

from __future__ import annotations

import json
from collections import Counter
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.academic import (
    _best_doc_score,
    _evaluate_retrieval_branch,
    _format_retrieved,
    _format_search,
    _normalize_retrieval_plan,
    _select_docs_with_subject_quota,
    RetrievalPlanItem,
    SearchQueryRewriteOutput,
    academic_router,
    generate_answer,
    rag_retrieve,
    rewrite_query,
    search_query_rewriter,
    web_search,
)
from src.config import load_prompt
from src.graph.state import CONTEXT_CLEAR


class TestAcademicRouterRetry:
    """academic_router clears context on retry path."""

    async def test_returns_empty_on_first_run(self):
        """First run (retry_count=0): no context clearing."""
        state = {
            "messages": [HumanMessage(content="test")],
            "retry_count": 0,
        }
        result = await academic_router(state)
        assert "context" not in result

    async def test_clears_context_on_retry(self):
        """On retry (retry_count > 0): returns CONTEXT_CLEAR to reset context."""
        state = {
            "messages": [HumanMessage(content="test")],
            "retry_count": 1,
            "context": [{"type": "rag", "content": "stale"}],
        }
        result = await academic_router(state)
        assert result["context"] is CONTEXT_CLEAR

    async def test_clears_context_on_second_retry(self):
        """retry_count=2 also triggers context clearing."""
        state = {
            "messages": [HumanMessage(content="test")],
            "retry_count": 2,
        }
        result = await academic_router(state)
        assert result["context"] is CONTEXT_CLEAR


class TestRewriteQuery:
    """rewrite_query node rewrites the user's question on retry."""

    @patch("src.graph.academic.get_node_llm")
    async def test_produces_rewritten_query(self, mock_get_llm):
        """Should call LLM and store result in rewritten_query."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="改进后的问题：判别式的具体用法"))
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="判别式怎么用")],
            "hallucination_reason": "答案未基于上下文",
            "retry_count": 1,
        }
        result = await rewrite_query(state)

        assert "rewritten_query" in result
        assert len(result["rewritten_query"]) > 0
        mock_get_llm.assert_called_once_with("supervisor")

    @patch("src.graph.academic.get_node_llm")
    async def test_uses_hallucination_reason_in_prompt(self, mock_get_llm):
        """The LLM prompt should include the hallucination reason."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="rewritten"))
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="original question")],
            "hallucination_reason": "fabricated formula",
            "retry_count": 1,
        }
        await rewrite_query(state)

        call_args = mock_llm.ainvoke.call_args[0][0]
        prompt_text = " ".join(m.content for m in call_args)
        assert "fabricated formula" in prompt_text
        assert "original question" in prompt_text
        assert "只输出一行检索查询" in prompt_text
        assert "英文术语" in prompt_text

    @patch("src.graph.academic.get_node_llm")
    async def test_falls_back_to_original_on_failure(self, mock_get_llm):
        """On LLM failure, rewritten_query should be the original question."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="original question")],
            "hallucination_reason": "bad",
            "retry_count": 1,
        }
        result = await rewrite_query(state)

        assert result["rewritten_query"] == "original question"


class TestSearchQueryRewriter:
    """search_query_rewriter rewrites initial retrieval queries."""

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    async def test_produces_rag_and_web_queries(self, mock_get_llm, mock_get_fallback):
        structured = MagicMock()
        parsed = SearchQueryRewriteOutput(
            rag_query="Python 函数 function 参数 parameter argument 返回值 return value 作用域 scope def local variable global variable",
            web_search_query="Python function parameters arguments return value scope course notes tutorial",
            expanded_keypoints=["函数", "function", "参数", "parameter", "argument", "返回值", "return value", "作用域", "scope"],
            reason="用户输入包含中文课程术语，补充英文教材常用术语以召回中英双语资料",
        )
        structured.ainvoke = AsyncMock(
            return_value={
                "raw": AIMessage(content='{"rag_query":"Python 变量 条件判断 循环 课程知识点"}'),
                "parsed": parsed,
                "parsing_error": None,
            }
        )
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        mock_get_llm.return_value = llm

        fallback_llm = MagicMock()
        fallback_llm.with_structured_output.return_value = MagicMock()
        mock_get_fallback.return_value = fallback_llm

        result = await search_query_rewriter({
            "messages": [HumanMessage(content="生成一份 Python 练习题")],
            "keypoints": ["Python", "练习题"],
            "requested_resource_type": "quiz",
            "subject": "computer_science",
        })

        assert result["search_rag_query"] == (
            "Python 函数 function 参数 parameter argument 返回值 return value 作用域 scope def local variable global variable"
        )
        assert result["search_web_query"] == "Python function parameters arguments return value scope course notes tutorial"
        assert result["expanded_keypoints"] == [
            "函数", "function", "参数", "parameter", "argument", "返回值", "return value", "作用域", "scope",
        ]
        assert result["search_query_rewrite_reason"]
        assert result["search_query_rewrite_error"] == ""
        assert result["search_query_rewrite_raw_preview"].startswith('{"rag_query"')
        mock_get_llm.assert_called_once_with("query_rewrite", temperature=0.0)
        llm.with_structured_output.assert_called_once_with(
            SearchQueryRewriteOutput,
            method="json_mode",
            include_raw=True,
        )
        fallback_llm.with_structured_output.assert_called_once_with(
            SearchQueryRewriteOutput,
            method="json_mode",
            include_raw=True,
        )

    def test_search_query_rewriter_prompt_requires_bilingual_retrieval(self):
        prompt = load_prompt("search_query_rewriter")
        assert "中文和英文高校课程资料" in prompt
        assert "中英双语" in prompt
        assert "英文教材常用术语" in prompt
        assert "不要只复述用户原始 query" in prompt
        assert "available subjects" in prompt
        assert "subject_candidates" in prompt
        assert "retrieval_plan" in prompt
        assert "core_concept" in prompt

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

        assert plan == [
            {
                "subject": "python",
                "role": "implementation_tool",
                "rag_query": "new",
                "web_search_query": "",
                "purpose": "",
                "relation_to_goal": "",
                "priority": 0.9,
            },
        ]
        assert debug["raw_plan_count"] == 5
        assert debug["normalized_plan_count"] == 1
        assert debug["accepted_subjects"] == ["python"]
        reasons = {item["reason"] for item in debug["rejected_items"]}
        assert "empty_subject" in reasons
        assert "invalid_role_fallback_to_supporting_context" in reasons
        assert "duplicate_subject_lower_priority" in reasons
        assert "subject_not_in_available_subjects" in reasons
        assert "empty_rag_query" in reasons

    async def test_noops_when_retry_rewritten_query_exists(self):
        result = await search_query_rewriter({
            "messages": [HumanMessage(content="original")],
            "rewritten_query": "retry query",
        })
        assert result == {
            "retrieval_plan": [],
            "learning_goal": "",
            "primary_subject": "",
            "subject_relation_summary": "",
        }

    @patch("src.graph.academic.get_available_subjects_from_data")
    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    async def test_retrieval_plan_uses_available_subjects_not_only_candidates(
        self, mock_get_llm, mock_get_fallback, mock_available_subjects,
    ):
        mock_available_subjects.return_value = ["python", "machine_learning", "big_data"]
        structured = MagicMock()
        parsed = SearchQueryRewriteOutput(
            rag_query="Python machine learning overfitting code",
            web_search_query="Python machine learning overfitting course notes",
            expanded_keypoints=["Python", "overfitting", "机器学习"],
            reason="用户问题涉及实现工具和机器学习核心概念",
            learning_goal="用 Python 理解和检测机器学习过拟合",
            primary_subject="machine_learning",
            subject_relation_summary="machine_learning 提供核心概念，python 提供实现工具",
            retrieval_plan=[
                RetrievalPlanItem(
                    subject="python",
                    role="implementation_tool",
                    rag_query="Python code function sklearn overfitting detection",
                    web_search_query="Python sklearn overfitting code example",
                    purpose="检索实现工具与代码资料",
                    relation_to_goal="用 Python 承载机器学习检测实践",
                    priority=0.8,
                ),
                RetrievalPlanItem(
                    subject="machine_learning",
                    role="core_concept",
                    rag_query="机器学习 overfitting 正则化 regularization 泛化 generalization",
                    web_search_query="machine learning overfitting regularization course notes",
                    purpose="检索过拟合核心概念",
                    relation_to_goal="解释检测与改进依据",
                    priority=0.95,
                ),
            ],
        )
        structured.ainvoke = AsyncMock(return_value={"raw": AIMessage(content="{}"), "parsed": parsed, "parsing_error": None})
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        mock_get_llm.return_value = llm
        fallback_llm = MagicMock()
        fallback_llm.with_structured_output.return_value = MagicMock()
        mock_get_fallback.return_value = fallback_llm

        result = await search_query_rewriter({
            "messages": [HumanMessage(content="用 Python 做机器学习过拟合检测")],
            "keypoints": ["Python", "机器学习", "过拟合"],
            "requested_resource_type": "",
            "subject": "python",
            "subject_candidates": ["python"],
        })

        subjects = [item["subject"] for item in result["retrieval_plan"]]
        assert subjects == ["machine_learning", "python"]
        assert result["primary_subject"] == "machine_learning"
        assert result["learning_goal"] == "用 Python 理解和检测机器学习过拟合"
        assert result["subject_relation_summary"] == "machine_learning 提供核心概念，python 提供实现工具"

    @patch("src.graph.academic.get_available_subjects_from_data")
    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    async def test_query_rewrite_emits_a3_trace(
        self, mock_get_llm, mock_get_fallback, mock_available_subjects, caplog, monkeypatch,
    ):
        monkeypatch.setenv("LOG_QUERY_REWRITE_RESULT", "true")
        mock_available_subjects.return_value = ["python"]
        structured = MagicMock()
        parsed = SearchQueryRewriteOutput(
            rag_query="Python function",
            web_search_query="Python function course notes",
            expanded_keypoints=["Python", "function"],
            reason="test",
            retrieval_plan=[
                RetrievalPlanItem(
                    subject="python",
                    role="implementation_tool",
                    rag_query="Python function",
                    priority=0.8,
                ),
            ],
        )
        structured.ainvoke = AsyncMock(return_value={"raw": AIMessage(content="raw"), "parsed": parsed, "parsing_error": None})
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        mock_get_llm.return_value = llm
        fallback_llm = MagicMock()
        fallback_llm.with_structured_output.return_value = MagicMock()
        mock_get_fallback.return_value = fallback_llm

        with caplog.at_level("WARNING"):
            await search_query_rewriter({
                "messages": [HumanMessage(content="Python 函数")],
                "subject": "python",
                "request_id": "req-qr",
                "thread_id": "thread-qr",
            })

        payload = json.loads(
            next(r.getMessage() for r in caplog.records if '"stage": "query_rewrite"' in r.getMessage()).removeprefix("A3_TRACE ")
        )
        assert payload["request_id"] == "req-qr"
        assert payload["thread_id"] == "thread-qr"
        assert payload["retrieval_plan_count"] == 1

    @patch("src.graph.academic.get_available_subjects_from_data")
    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    async def test_retrieval_plan_filters_and_normalizes_subjects(
        self, mock_get_llm, mock_get_fallback, mock_available_subjects,
    ):
        mock_available_subjects.return_value = ["python", "machine_learning", "big_data", "math"]
        structured = MagicMock()
        parsed = SearchQueryRewriteOutput(
            rag_query="multi subject query",
            web_search_query="multi subject web query",
            expanded_keypoints=["multi"],
            reason="test",
            primary_subject="law",
            retrieval_plan=[
                RetrievalPlanItem(subject="Python", role="implementation_tool", rag_query="old python", priority=0.2),
                RetrievalPlanItem(subject="python", role="implementation_tool", rag_query="best python", priority=1.8),
                RetrievalPlanItem(subject="machine learning", role="core_concept", rag_query="ml query", priority=0.9),
                RetrievalPlanItem(subject="law", role="core_concept", rag_query="law query", priority=0.99),
                RetrievalPlanItem(subject="big-data", role="bad_role", rag_query="big data query", priority=-2),
                RetrievalPlanItem(subject="math", role="prerequisite", rag_query="math query", priority=0.7),
            ],
        )
        structured.ainvoke = AsyncMock(return_value={"raw": AIMessage(content="{}"), "parsed": parsed, "parsing_error": None})
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        mock_get_llm.return_value = llm
        fallback_llm = MagicMock()
        fallback_llm.with_structured_output.return_value = MagicMock()
        mock_get_fallback.return_value = fallback_llm

        result = await search_query_rewriter({
            "messages": [HumanMessage(content="test")],
            "subject": "python",
            "subject_candidates": ["python"],
        })

        plan = result["retrieval_plan"]
        assert [item["subject"] for item in plan] == ["python", "machine_learning", "math", "big_data"]
        assert plan[0]["rag_query"] == "best python"
        assert plan[0]["priority"] == 1.0
        assert plan[-1]["role"] == "supporting_context"
        assert plan[-1]["priority"] == 0.0
        assert result["primary_subject"] == "python"

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    async def test_returns_empty_queries_on_failure(self, mock_get_llm, mock_get_fallback):
        structured = MagicMock()
        structured.ainvoke = AsyncMock(side_effect=RuntimeError("structured failure"))
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        mock_get_llm.return_value = llm

        fallback_llm = MagicMock()
        fallback_llm.with_structured_output.return_value = MagicMock()
        mock_get_fallback.return_value = fallback_llm

        result = await search_query_rewriter({
            "messages": [HumanMessage(content="Python 练习题")],
            "keypoints": ["Python"],
            "requested_resource_type": "quiz",
            "subject": "computer_science",
        })

        assert result["search_rag_query"] == ""
        assert result["search_web_query"] == ""
        assert result["expanded_keypoints"] == []
        assert "structured failure" in result["search_query_rewrite_error"]
        assert result["retrieval_plan"] == []
        assert result["learning_goal"] == ""
        assert result["primary_subject"] == ""
        assert result["subject_relation_summary"] == ""

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    async def test_structured_path_returns_parsing_error_with_raw_preview(self, mock_get_llm, mock_get_fallback):
        structured = MagicMock()
        structured.ainvoke = AsyncMock(
            return_value={
                "raw": AIMessage(content="bad structured output"),
                "parsed": None,
                "parsing_error": ValueError("invalid JSON"),
            }
        )
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        mock_get_llm.return_value = llm

        fallback_llm = MagicMock()
        fallback_llm.with_structured_output.return_value = MagicMock()
        mock_get_fallback.return_value = fallback_llm

        result = await search_query_rewriter({
            "messages": [HumanMessage(content="Python 练习题")],
            "keypoints": ["Python"],
            "requested_resource_type": "quiz",
            "subject": "computer_science",
        })

        assert "search_query_rewriter parsing_error" in result["search_query_rewrite_error"]
        assert result["search_query_rewrite_raw_preview"] == "bad structured output"
        assert result["search_rag_query"] == ""
        assert result["search_web_query"] == ""
        assert result["retrieval_plan"] == []

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    async def test_structured_path_rejects_none_parsed_with_raw_preview(self, mock_get_llm, mock_get_fallback):
        structured = MagicMock()
        structured.ainvoke = AsyncMock(
            return_value={
                "raw": AIMessage(content="{}"),
                "parsed": None,
                "parsing_error": None,
            }
        )
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        mock_get_llm.return_value = llm

        fallback_llm = MagicMock()
        fallback_llm.with_structured_output.return_value = MagicMock()
        mock_get_fallback.return_value = fallback_llm

        result = await search_query_rewriter({
            "messages": [HumanMessage(content="Python 练习题")],
            "keypoints": ["Python"],
            "requested_resource_type": "quiz",
            "subject": "computer_science",
        })

        assert "parsed result is None" in result["search_query_rewrite_error"]
        assert result["search_query_rewrite_raw_preview"] == "{}"
        assert result["search_rag_query"] == ""
        assert result["search_web_query"] == ""
        assert result["retrieval_plan"] == []

class TestRetrievalBranchQuality:
    def test_best_doc_score_prefers_rerank_score(self):
        assert _best_doc_score([
            {"raw_vector_score": 0.9, "rerank_score": 0.2},
            {"raw_vector_score": 0.4, "rerank_score": 0.8},
        ]) == 0.8

    def test_evaluates_strong_usable_weak_missing(self):
        strong = _evaluate_retrieval_branch(
            subject="python",
            role="implementation_tool",
            docs=[{"rerank_score": 0.8}],
            is_hit=True,
            subject_mismatch_count=0,
        )
        usable = _evaluate_retrieval_branch(
            subject="python",
            role="implementation_tool",
            docs=[{"rerank_score": 0.5}],
            is_hit=True,
            subject_mismatch_count=0,
        )
        weak = _evaluate_retrieval_branch(
            subject="python",
            role="implementation_tool",
            docs=[{"rerank_score": 0.8}],
            is_hit=False,
            subject_mismatch_count=0,
        )
        missing = _evaluate_retrieval_branch(
            subject="python",
            role="implementation_tool",
            docs=[],
            is_hit=False,
            subject_mismatch_count=0,
        )

        assert strong["branch_status"] == "strong"
        assert usable["branch_status"] == "usable"
        assert weak["branch_status"] == "weak"
        assert weak["weak_reason"] == "retrieve_is_hit_false"
        assert missing["branch_status"] == "missing"
        assert missing["weak_reason"] == "no_docs"

    def test_evaluate_branch_marks_reranker_failure_as_fallback(self):
        result = _evaluate_retrieval_branch(
            subject="python",
            role="core_concept",
            docs=[{"raw_vector_score": 57.7}],
            is_hit=False,
            subject_mismatch_count=0,
            reranker_failed=True,
        )

        assert result["branch_status"] == "weak"
        assert result["weak_reason"] == "reranker_failed"
        assert result["branch_status_score_source"] == "fallback_raw_retrieval_signal"
        assert result["reranker_failed"] is True

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
            {
                "type": "rag_diagnostic",
                "content": "missing",
                "source": "local_rag_diagnostic",
                "retrieval_subject": "big_data",
                "retrieval_priority": 0.5,
                "branch_status": "missing",
            },
        ]

        selected, debug = _select_docs_with_subject_quota(
            docs,
            8,
            primary_subject="machine_learning",
        )

        distribution = Counter(doc["retrieval_subject"] for doc in selected)
        assert distribution["machine_learning"] == 4
        assert distribution["python"] == 1
        assert distribution["big_data"] == 1
        assert debug["weak_subjects"] == ["python"]
        assert debug["missing_subjects"] == ["big_data"]


class TestRagRetrieveWithRewrittenQuery:
    """rag_retrieve uses rewritten_query when available."""

    @patch("src.graph.academic.retrieve")
    async def test_uses_rewritten_query(self, mock_retrieve):
        """When rewritten_query is set, use it instead of keypoints."""
        mock_retrieve.return_value = {"docs": []}

        state = {
            "messages": [HumanMessage(content="original")],
            "keypoints": ["original"],
            "subject": "math",
            "rewritten_query": "improved question about discriminant",
        }
        await rag_retrieve(state)

        mock_retrieve.assert_called_once_with(
            query="improved question about discriminant", subject="math",
        )

    @patch("src.graph.academic.retrieve")
    async def test_ignores_empty_rewritten_query(self, mock_retrieve):
        """When rewritten_query is empty, fall back to keypoints."""
        mock_retrieve.return_value = {"docs": []}

        state = {
            "messages": [HumanMessage(content="原始问题")],
            "keypoints": ["判别式"],
            "subject": "math",
            "rewritten_query": "",
        }
        await rag_retrieve(state)

        mock_retrieve.assert_called_once_with(query="判别式", subject="math")

    @patch("src.graph.academic.retrieve")
    async def test_uses_search_rag_query_before_keypoints(self, mock_retrieve):
        mock_retrieve.return_value = {"docs": []}

        state = {
            "messages": [HumanMessage(content="original")],
            "keypoints": ["original"],
            "subject": "computer_science",
            "rewritten_query": "",
            "search_rag_query": "Python variables loops course concepts",
        }
        await rag_retrieve(state)

        mock_retrieve.assert_called_once_with(
            query="Python variables loops course concepts", subject="computer_science",
        )

    @patch("src.graph.academic.retrieve")
    async def test_uses_expanded_keypoints_before_original_keypoints(self, mock_retrieve):
        mock_retrieve.return_value = {"docs": []}

        state = {
            "messages": [HumanMessage(content="original")],
            "keypoints": ["Python"],
            "expanded_keypoints": ["变量", "条件判断", "循环"],
            "subject": "computer_science",
            "rewritten_query": "",
            "search_rag_query": "",
        }
        await rag_retrieve(state)

        mock_retrieve.assert_called_once_with(query="变量 条件判断 循环", subject="computer_science")

    @patch("src.graph.academic.retrieve")
    async def test_uses_retrieval_plan_by_subject_with_quota(self, mock_retrieve):
        def fake_retrieve(query, subject, top_k):
            docs_by_subject = {
                "python": [
                    {"content": f"Python doc {i}", "source": f"py{i}.pdf", "rerank_score": 0.8, "metadata": {"subject": "python"}}
                    for i in range(1, 6)
                ],
                "machine_learning": [
                    {"content": f"ML doc {i}", "source": f"ml{i}.pdf", "rerank_score": 0.9, "metadata": {"subject": "machine_learning"}}
                    for i in range(1, 6)
                ],
            }
            return {"docs": docs_by_subject[subject], "is_hit": True, "reranker_failed": False}

        mock_retrieve.side_effect = fake_retrieve

        state = {
            "messages": [HumanMessage(content="用 Python 做过拟合检测")],
            "subject": "python",
            "rewritten_query": "",
            "search_rag_query": "overall query",
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

        with patch("src.graph.academic.get_setting") as mock_setting:
            mock_setting.side_effect = lambda key, default=None: {
                "rag.multi_subject_per_subject_top_k": 2,
                "rag.multi_subject_max_docs": 3,
            }.get(key, default)
            result = await rag_retrieve(state)

        mock_retrieve.assert_has_calls([
            call(query="Python sklearn code", subject="python", top_k=2),
            call(query="overfitting regularization", subject="machine_learning", top_k=2),
        ])
        context = result["context"]
        subjects = {doc["retrieval_subject"] for doc in context}
        assert {"python", "machine_learning"}.issubset(subjects)
        assert len(context) == 3
        assert all(doc["type"] == "rag" for doc in context)
        assert all("doc 4" not in doc["content"] and "doc 5" not in doc["content"] for doc in context)
        assert all(doc["branch_status"] == "strong" for doc in context)
        assert any(doc["retrieval_role"] == "implementation_tool" for doc in context)

    @patch("src.graph.academic.retrieve")
    async def test_missing_retrieval_plan_branch_adds_diagnostic_context(self, mock_retrieve):
        mock_retrieve.return_value = {"docs": [], "is_hit": False}

        state = {
            "messages": [HumanMessage(content="test")],
            "subject": "python",
            "rewritten_query": "",
            "search_rag_query": "overall query",
            "retrieval_plan": [
                {
                    "subject": "python",
                    "role": "implementation_tool",
                    "rag_query": "Python code",
                    "purpose": "实现工具",
                    "relation_to_goal": "支持实践",
                    "priority": 0.8,
                },
            ],
        }

        result = await rag_retrieve(state)

        assert len(result["context"]) == 1
        diagnostic = result["context"][0]
        assert diagnostic["type"] == "rag_diagnostic"
        assert diagnostic["branch_status"] == "missing"
        assert diagnostic["weak_reason"] == "no_docs"
        assert "暂未检索到" in diagnostic["content"]

    @patch("src.graph.academic.retrieve")
    async def test_rewritten_query_ignores_retrieval_plan(self, mock_retrieve):
        mock_retrieve.return_value = {"docs": []}

        state = {
            "messages": [HumanMessage(content="original")],
            "subject": "python",
            "rewritten_query": "retry query",
            "retrieval_plan": [
                {"subject": "machine_learning", "rag_query": "should not run"},
            ],
        }
        await rag_retrieve(state)

        mock_retrieve.assert_called_once_with(query="retry query", subject="python")


class TestWebSearchWithRewrittenQuery:
    """web_search uses rewritten_query when available."""

    @patch("src.graph.academic.web_search_fn")
    async def test_uses_rewritten_query(self, mock_search):
        """When rewritten_query is set, use it for web search."""
        mock_search.return_value = []

        state = {
            "messages": [HumanMessage(content="original")],
            "rewritten_query": "improved search query",
        }
        await web_search(state)

        mock_search.assert_called_once_with("improved search query")

    @patch("src.graph.academic.web_search_fn")
    async def test_ignores_empty_rewritten_query(self, mock_search):
        """When rewritten_query is empty, fall back to last human message."""
        mock_search.return_value = []

        state = {
            "messages": [HumanMessage(content="the real question")],
            "rewritten_query": "",
        }
        await web_search(state)

        mock_search.assert_called_once_with("the real question")

    @patch("src.graph.academic.web_search_fn")
    async def test_uses_search_web_query_before_original_question(self, mock_search):
        mock_search.return_value = []

        state = {
            "messages": [HumanMessage(content="the real question")],
            "rewritten_query": "",
            "search_web_query": "Python practice problems answers explanations",
        }
        await web_search(state)

        mock_search.assert_called_once_with("Python practice problems answers explanations")

    @patch("src.graph.academic.web_search_fn")
    async def test_uses_highest_priority_retrieval_plan_query_once(self, mock_search):
        mock_search.return_value = []

        state = {
            "messages": [HumanMessage(content="the real question")],
            "rewritten_query": "",
            "search_web_query": "",
            "retrieval_plan": [
                {
                    "subject": "python",
                    "web_search_query": "Python code examples",
                    "rag_query": "Python code",
                    "priority": 0.4,
                },
                {
                    "subject": "machine_learning",
                    "web_search_query": "machine learning overfitting course notes",
                    "rag_query": "overfitting",
                    "priority": 0.9,
                },
            ],
        }
        await web_search(state)

        mock_search.assert_called_once_with("machine learning overfitting course notes")


class TestRagRetrieve:

    @patch("src.graph.academic.retrieve")
    async def test_uses_keypoints_as_query(self, mock_retrieve):
        mock_retrieve.return_value = {
            "docs": [{"content": "test", "source": "f.pdf", "rerank_score": 0.9}],
            "is_hit": True,
            "reranker_failed": False,
        }

        state = {
            "messages": [HumanMessage(content="什么是判别式")],
            "keypoints": ["二次函数", "判别式"],
            "subject": "math",
        }
        result = await rag_retrieve(state)

        mock_retrieve.assert_called_once_with(query="二次函数 判别式", subject="math")
        assert len(result["context"]) == 1
        assert result["context"][0]["type"] == "rag"

    @patch("src.graph.academic.retrieve")
    async def test_falls_back_to_message_when_no_keypoints(self, mock_retrieve):
        mock_retrieve.return_value = {"docs": []}

        state = {
            "messages": [HumanMessage(content="告诉我关于椭圆的知识")],
            "keypoints": [],
            "subject": "math",
        }
        await rag_retrieve(state)

        mock_retrieve.assert_called_once_with(query="告诉我关于椭圆的知识", subject="math")

    @patch("src.graph.academic.retrieve")
    async def test_subject_other_passes_none(self, mock_retrieve):
        mock_retrieve.return_value = {"docs": []}

        state = {
            "messages": [HumanMessage(content="test")],
            "keypoints": ["test"],
            "subject": "other",
        }
        await rag_retrieve(state)

        mock_retrieve.assert_called_once_with(query="test", subject=None)


class TestWebSearch:

    @patch("src.graph.academic.web_search_fn")
    async def test_returns_context_results(self, mock_search):
        mock_search.return_value = [{"content": "result", "title": "t", "url": "u"}]

        state = {"messages": [HumanMessage(content="量子力学")]}
        result = await web_search(state)

        assert len(result["context"]) == 1
        assert result["context"][0]["type"] == "web"
        mock_search.assert_called_once_with("量子力学")

    @patch("src.graph.academic.web_search_fn", side_effect=Exception("network error"))
    async def test_returns_empty_on_exception(self, mock_search):
        state = {"messages": [HumanMessage(content="test")]}
        result = await web_search(state)

        assert result["context"] == []


class TestFormatHelpers:

    def test_format_retrieved_empty(self):
        assert _format_retrieved([]) == "无相关参考资料。"

    def test_format_retrieved_with_docs(self, sample_retrieved_docs):
        output = _format_retrieved(sample_retrieved_docs)
        assert "[1]" in output
        assert "[2]" in output
        assert "math_2024.pdf" in output

    def test_format_retrieved_with_retrieval_plan_metadata(self):
        docs = [
            {
                "content": "Overfitting means poor generalization.",
                "source": "ml.pdf",
                "rerank_score": 0.9,
                "retrieval_subject": "machine_learning",
                "retrieval_role": "core_concept",
                "retrieval_purpose": "解释核心概念",
                "relation_to_goal": "支撑过拟合检测",
                "retrieval_query": "overfitting generalization",
            },
        ]

        output = _format_retrieved(docs)

        assert "machine_learning｜core_concept｜依据" in output
        assert "用途：解释核心概念" in output
        assert "关系：支撑过拟合检测" in output
        assert "检索 query：overfitting generalization" in output

    def test_format_retrieved_marks_weak_and_missing_evidence(self):
        docs = [
            {
                "content": "Weak Python material.",
                "source": "py.pdf",
                "rerank_score": 0.1,
                "retrieval_subject": "python",
                "retrieval_role": "implementation_tool",
                "branch_status": "weak",
                "weak_reason": "low_rerank_score",
            },
            {
                "content": "本地知识库中暂未检索到该学科分支的有效资料。",
                "source": "local_rag_diagnostic",
                "retrieval_subject": "big_data",
                "retrieval_role": "application_context",
                "branch_status": "missing",
                "weak_reason": "no_docs",
            },
        ]

        output = _format_retrieved(docs)

        assert "弱证据" in output
        assert "本地资料不足" in output
        assert "不要当作强课程依据" in output

    def test_format_search_empty(self):
        assert _format_search([]) == "无网络搜索结果。"

    def test_format_search_with_results(self, sample_search_results):
        output = _format_search(sample_search_results)
        assert "[1]" in output
        assert "课程安排" in output


class TestGenerateAnswer:

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    async def test_generates_ai_message(self, mock_get_llm, mock_get_fallback, mock_llm_response):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response("判别式 Δ=b²-4ac 的作用是..."))
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()

        state = {
            "messages": [HumanMessage(content="判别式怎么用")],
            "context": [{"type": "rag", "content": "Δ=b²-4ac", "source": "test.pdf", "rerank_score": 0.9}],
        }
        result = await generate_answer(state)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "判别式" in result["messages"][0].content

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    async def test_handles_empty_context(self, mock_get_llm, mock_get_fallback, mock_llm_response):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response("I can help with that."))
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()

        state = {
            "messages": [HumanMessage(content="test")],
            "context": [],
        }
        result = await generate_answer(state)

        assert len(result["messages"]) == 1

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    async def test_plain_answer_prompt_offers_followup_resources(
        self, mock_get_llm, mock_get_fallback, mock_llm_response,
    ):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response("answer"))
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()

        state = {
            "messages": [HumanMessage(content="机器学习里的过拟合是什么意思？")],
            "context": [
                {"type": "rag", "content": "过拟合是泛化不足", "source": "ml.md", "rerank_score": 0.9},
                {"type": "web", "content": "正则化可缓解过拟合", "title": "ML", "url": "https://example.com"},
            ],
            "requested_resource_type": "",
            "needs_mindmap": False,
        }
        await generate_answer(state)

        prompt_text = mock_llm.ainvoke.call_args[0][0][-1].content
        assert "本地课程知识库" in prompt_text
        assert "网络或外部搜索补充资料" in prompt_text
        assert "过拟合是泛化不足" in prompt_text
        assert "正则化可缓解过拟合" in prompt_text
        assert "## 还可以继续生成的个性化学习资源" in prompt_text

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    async def test_resource_request_prompt_does_not_offer_followup_resources(
        self, mock_get_llm, mock_get_fallback, mock_llm_response,
    ):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response("resource answer"))
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()

        state = {
            "messages": [HumanMessage(content="请生成机器学习过拟合的分层练习题")],
            "context": [],
            "requested_resource_type": "quiz",
            "needs_mindmap": False,
        }
        await generate_answer(state)

        prompt_text = mock_llm.ainvoke.call_args[0][0][-1].content
        assert "不要追加“还可以继续生成的个性化学习资源”小节" in prompt_text
        assert "## 还可以继续生成的个性化学习资源" not in prompt_text
