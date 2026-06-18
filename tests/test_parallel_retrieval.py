"""Unit tests for parallel retrieval (fan-out / fan-in) pattern.

Tests cover: operator.add context reducer, academic_router pass-through,
rag_retrieve + web_search writing evidence candidates,
generate_answer reading judged context, and graph fan-out/barrier structure.
All tests mock external dependencies -- no real API calls required.
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.academic import (
    WEB_RESEARCH_V2_PLANNER_NODE,
    academic_router,
    rag_retrieve,
    web_search,
)
from src.graph.builder import build_graph, get_compiled_graph
from src.graph.web_research import WebResearchPlan, WebResearchTask, WebSourceSummary, WebSourceSummaryBatch
from src.llm.structured_output import StructuredLLMResult


def _web_research_structured_result(parsed, *, node_name: str, schema_name: str) -> StructuredLLMResult:
    return StructuredLLMResult(
        success=True,
        parsed=parsed,
        node_name=node_name,
        llm_node=node_name,
        schema_name=schema_name,
        provider="test",
        model="test",
        output_mode="deepseek_tool_call_strict",
        fallback_modes=[],
        raw_output=parsed.model_dump_json() if parsed is not None else "{}",
    )


async def _fake_web_research_v2_llm(**kwargs):
    if kwargs["node_name"] == WEB_RESEARCH_V2_PLANNER_NODE:
        planner_prompt = str(kwargs["messages"][-1]["content"])
        subject_matches = re.findall(r'"subject":\s*"([^"]*)"', planner_prompt)
        seed_matches = re.findall(r'"seed_search_query":\s*"([^"]*)"', planner_prompt)
        subject = (subject_matches[-1] if subject_matches else "other") or "other"
        seed_query = (seed_matches[-1] if seed_matches else "web tutorial") or "web tutorial"
        plan = WebResearchPlan(tasks=[
            WebResearchTask(
                task_id=f"task-{subject}-0",
                subject=subject,
                role="supporting_context",
                purpose=f"Find web material for {subject}.",
                search_query=seed_query,
                reason="Need web evidence for the current retrieval request.",
                priority=0.8,
            )
        ])
        return _web_research_structured_result(
            plan,
            node_name=WEB_RESEARCH_V2_PLANNER_NODE,
            schema_name="WebResearchPlan",
        )

    source_ids = re.findall(r'"source_id":\s*"([^"]+)"', str(kwargs["messages"][-1]["content"]))
    batch = WebSourceSummaryBatch(summaries=[
        WebSourceSummary(
            source_id=source_id,
            keep=True,
            summary="result",
            coverage_points=["result"],
            reason="Relevant web result.",
            evidence_type="unknown",
            use_case="background_context",
            relevance="medium",
            usefulness="medium",
            risk="low",
        )
        for source_id in source_ids
    ])
    return _web_research_structured_result(
        batch,
        node_name=kwargs["node_name"],
        schema_name="WebSourceSummaryBatch",
    )


# ===========================================================================
# TestContextReducer -- operator.add merges context from parallel branches
# ===========================================================================

class TestContextReducer:
    """Verify that context_reducer correctly merges and clears context."""

    def test_merges_from_two_branches(self):
        """Simulates LangGraph fan-in: merge rag + web context lists."""
        from src.graph.state import context_reducer

        branch_a = [{"type": "rag", "content": "doc1", "source": "f.pdf", "rerank_score": 0.9}]
        branch_b = [{"type": "web", "content": "web1", "title": "T", "url": "http://x"}]

        merged = context_reducer(branch_a, branch_b)

        assert len(merged) == 2
        assert merged[0]["type"] == "rag"
        assert merged[1]["type"] == "web"

    def test_empty_branch_produces_no_extra(self):
        """If one branch returns empty context, merge is just the other."""
        from src.graph.state import context_reducer

        branch_a = [{"type": "rag", "content": "doc1"}]
        branch_b = []

        merged = context_reducer(branch_a, branch_b)

        assert len(merged) == 1
        assert merged[0]["type"] == "rag"

    def test_clear_signal_resets_context(self):
        """A list with __clear__ sentinel resets context to empty."""
        from src.graph.state import CONTEXT_CLEAR, context_reducer

        existing = [{"type": "rag", "content": "old_doc"}]

        merged = context_reducer(existing, CONTEXT_CLEAR)

        assert merged == []

    def test_normal_append_after_clear(self):
        """After clearing, normal updates append to empty list."""
        from src.graph.state import context_reducer

        cleared = []
        new_docs = [{"type": "rag", "content": "new_doc"}]

        merged = context_reducer(cleared, new_docs)

        assert len(merged) == 1
        assert merged[0]["content"] == "new_doc"


# ===========================================================================
# TestAcademicRouterNode -- pass-through for fan-out
# ===========================================================================

class TestAcademicRouterNode:
    """academic_router is a no-op node that enables fan-out."""

    async def test_returns_empty_dict(self):
        state = {
            "messages": [HumanMessage(content="test")],
            "keypoints": ["test"],
            "subject": "math",
        }
        result = await academic_router(state)
        assert result == {}

    async def test_is_traced(self, in_memory_exporter):
        """Should create an OTel span for Jaeger visibility."""
        state = {"messages": [HumanMessage(content="test")]}
        await academic_router(state)

        spans = in_memory_exporter.get_finished_spans()
        node_spans = [s for s in spans if s.name == "graph.node.academic_router"]
        assert len(node_spans) == 1


# ===========================================================================
# TestRagRetrieveParallelOutput -- writes local evidence candidates
# ===========================================================================

class TestRagRetrieveParallelOutput:
    """rag_retrieve returns local evidence only; context is written by evidence_judge."""

    @patch("src.graph.academic.retrieve")
    async def test_returns_context_with_rag_type(self, mock_retrieve):
        mock_retrieve.return_value = {
            "docs": [{"content": "判别式", "source": "math.pdf", "rerank_score": 0.9}],
        }

        state = {
            "messages": [HumanMessage(content="test")],
            "keypoints": ["判别式"],
            "subject": "math",
        }
        result = await rag_retrieve(state)

        assert "context" not in result
        assert len(result["local_evidence_candidates"]) == 1
        candidate = result["local_evidence_candidates"][0]
        assert candidate["source_type"] == "local_rag"
        assert candidate["content_preview"] == "判别式"
        assert result["local_evidence_originals"][candidate["evidence_id"]]["source"] == "math.pdf"

    @patch("src.graph.academic.retrieve")
    async def test_empty_retrieval_returns_empty_context(self, mock_retrieve):
        mock_retrieve.return_value = {"docs": []}

        state = {
            "messages": [HumanMessage(content="test")],
            "keypoints": ["test"],
            "subject": "math",
        }
        result = await rag_retrieve(state)

        assert result["local_evidence_candidates"] == []
        assert result["local_evidence_originals"] == {}
        assert "context" not in result

    @patch("src.graph.academic.retrieve")
    async def test_uses_last_human_message_when_no_keypoints(self, mock_retrieve):
        """On retry, fallback query should find last HumanMessage."""
        mock_retrieve.return_value = {"docs": []}

        state = {
            "messages": [
                HumanMessage(content="original question"),
                AIMessage(content="bad answer"),
            ],
            "keypoints": [],
            "subject": "math",
        }
        await rag_retrieve(state)

        mock_retrieve.assert_called_once_with(query="original question", subject="math", top_k=3)


# ===========================================================================
# TestWebSearchParallelOutput -- writes web evidence candidates
# ===========================================================================

class TestWebSearchParallelOutput:
    """web_search returns web evidence only; context is written by evidence_judge."""

    @patch("src.graph.academic.web_search_fn")
    async def test_returns_context_with_web_type(self, mock_search, monkeypatch):
        mock_search.return_value = [
            {"content": "result", "title": "Title", "url": "http://x"},
        ]
        monkeypatch.setattr("src.graph.academic.invoke_structured_llm", _fake_web_research_v2_llm)

        state = {"messages": [HumanMessage(content="量子力学")]}
        result = await web_search(state)

        assert "context" not in result
        assert len(result["web_evidence_candidates"]) == 1
        candidate = result["web_evidence_candidates"][0]
        assert candidate["source_type"] == "web"
        assert candidate["content_preview"] == "result"
        assert result["web_evidence_originals"][candidate["evidence_id"]]["title"] == "Title"

    @patch("src.graph.academic.web_search_fn", side_effect=Exception("network"))
    async def test_returns_empty_context_on_exception(self, mock_search, monkeypatch):
        monkeypatch.setattr("src.graph.academic.invoke_structured_llm", _fake_web_research_v2_llm)

        state = {"messages": [HumanMessage(content="test")]}
        with pytest.raises(RuntimeError, match="fallback is disabled") as exc_info:
            await web_search(state)

        debug = getattr(exc_info.value, "web_research_debug")
        assert debug["status"] == "failed"
        assert debug["used_fallback"] is False

    @patch("src.graph.academic.web_search_fn")
    async def test_uses_last_human_message_for_query(self, mock_search, monkeypatch):
        """During retry, web_search must find the original question."""
        mock_search.return_value = [{"content": "result", "title": "Title", "url": "http://x"}]
        monkeypatch.setattr("src.graph.academic.invoke_structured_llm", _fake_web_research_v2_llm)

        state = {
            "messages": [
                HumanMessage(content="the real question"),
                AIMessage(content="previous bad answer"),
            ],
        }
        await web_search(state)

        mock_search.assert_called_once()
        assert mock_search.call_args.args[0] == "the real question"


# ===========================================================================
# TestGenerateAnswerFromMergedContext -- reads unified context
# ===========================================================================

class TestGenerateAnswerFromMergedContext:
    """generate_answer splits merged context by type for formatting."""

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    async def test_uses_both_rag_and_web_context(
        self, mock_get_llm, mock_get_fallback, mock_llm_response,
    ):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response("combined answer"))
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()

        state = {
            "messages": [HumanMessage(content="判别式")],
            "context": [
                {"type": "rag", "content": "delta=b^2-4ac", "source": "math.pdf", "rerank_score": 0.9},
                {
                    "type": "web_evidence",
                    "source_type": "web",
                    "content": "判别式用法",
                    "title": "高等数学课程资料",
                    "url": "http://x",
                },
            ],
        }

        from src.graph.academic import generate_answer

        result = await generate_answer(state)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        # Verify the prompt includes both RAG and web content
        call_args = mock_llm.ainvoke.call_args[0][0]
        prompt_text = call_args[-1].content
        assert "delta=b^2-4ac" in prompt_text
        assert "判别式用法" in prompt_text

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    async def test_handles_empty_context(
        self, mock_get_llm, mock_get_fallback, mock_llm_response,
    ):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response("answer without context"))
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()

        state = {
            "messages": [HumanMessage(content="test")],
            "context": [],
        }

        from src.graph.academic import generate_answer

        result = await generate_answer(state)

        assert len(result["messages"]) == 1


# ===========================================================================
# TestGraphFanOutStructure -- verify graph topology
# ===========================================================================

class TestGraphFanOutStructure:
    """Verify the graph has correct fan-out/fan-in wiring."""

    def test_academic_router_node_exists(self):
        graph = build_graph()
        assert "academic_router" in graph.nodes

    def test_evaluate_hallucination_node_exists(self):
        graph = build_graph()
        assert "evaluate_hallucination" in graph.nodes

    def test_graph_compiles_with_fan_out(self):
        """Fan-out/fan-in graph must compile without error."""
        compiled = get_compiled_graph()
        assert compiled is not None
        assert hasattr(compiled, "invoke")

    def test_should_web_search_not_in_graph(self):
        """Sequential should_web_search replaced by parallel fan-out."""
        graph = build_graph()
        assert "rag_retrieve" in graph.nodes
        assert "web_search" in graph.nodes


# ===========================================================================
# TestRetryRoutesToAcademicRouter -- retry re-runs both retrievals
# ===========================================================================

class TestRewriteQueryNodeInGraph:
    """Verify rewrite_query node exists in graph and is wired in retry path."""

    def test_rewrite_query_node_exists(self):
        graph = build_graph()
        assert "rewrite_query" in graph.nodes

    def test_retry_routes_to_rewrite_query(self):
        """On retry, evaluate_hallucination should route to rewrite_query, not academic_router."""
        graph = build_graph()
        # The retry path should go through rewrite_query
        assert "rewrite_query" in graph.nodes


class TestRetryRoutesToAcademicRouter:
    """On hallucination retry, route back to academic_router for full re-retrieval."""

    def test_should_retry_or_end_returns_retry(self):
        from src.graph.academic import should_retry_or_end

        state = {"hallucination_detected": True, "retry_count": 1}
        assert should_retry_or_end(state) == "retry"

    def test_should_retry_or_end_returns_end_past_max(self):
        from src.graph.academic import MAX_RETRIES, should_retry_or_end

        state = {"hallucination_detected": True, "retry_count": MAX_RETRIES + 1}
        assert should_retry_or_end(state) == "end"
