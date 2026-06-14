"""Unit tests for graph construction and compilation."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.builder import (
    build_graph,
    get_compiled_graph,
    route_after_academic_retrieval,
    route_after_evidence_judge,
    route_after_query_rewrite,
)


class TestBuildGraph:

    def test_returns_state_graph(self):
        from langgraph.graph import StateGraph
        graph = build_graph()
        assert isinstance(graph, StateGraph)

    def test_graph_has_all_nodes(self):
        graph = build_graph()
        node_names = set(graph.nodes.keys())
        expected = {
            "supervisor",
            "academic_router",
            "search_query_rewriter",
            "rag_retrieve",
            "web_search",
            "evidence_judge",
            "generate_answer",
            "evaluate_hallucination",
            "rewrite_query",
            "study_plan_emotional_intel",
            "study_plan_planner",
            "study_plan_agent",
            "study_plan_reviewer_academic",
            "study_plan_reviewer_emotional",
            "study_plan_consensus",
            "study_plan_rewrite",
            "study_plan_output",
            "mindmap_planner",
            "mindmap_agent",
            "mindmap_reviewer",
            "mindmap_rewrite",
            "mindmap_output",
            "exercise_planner",
            "exercise_agent",
            "exercise_reviewer",
            "exercise_rewrite",
            "exercise_output",
            "review_doc_planner",
            "review_doc_agent",
            "review_doc_reviewer",
            "review_doc_rewrite",
            "review_doc_output",
            "emotional_response",
            "handle_unknown",
        }
        assert expected.issubset(node_names), f"Missing nodes: {expected - node_names}"

    def test_graph_compiles_without_error(self):
        graph = build_graph()
        compiled = graph.compile()
        assert compiled is not None

    def test_get_compiled_graph_returns_compiled(self):
        compiled = get_compiled_graph()
        assert hasattr(compiled, "invoke")
        assert hasattr(compiled, "stream")

    def test_route_after_evidence_judge_routes_resource_chains(self):
        assert route_after_evidence_judge({"requested_resource_type": "mindmap"}) == "mindmap"
        assert route_after_evidence_judge({"needs_mindmap": False, "requested_resource_type": "quiz"}) == "exercise"
        assert route_after_evidence_judge({"needs_mindmap": False, "requested_resource_type": ""}) == "answer"
        assert route_after_evidence_judge({"needs_mindmap": True, "requested_resource_type": "quiz"}) == "exercise"
        assert route_after_evidence_judge({"requested_resource_type": "review_doc"}) == "review_doc"
        assert route_after_evidence_judge({"requested_resource_type": "study_plan"}) == "study_plan"
        assert route_after_evidence_judge({}) == "answer"
        assert route_after_academic_retrieval({}) == "answer"

    def test_route_after_query_rewrite_routes_academic(self):
        assert route_after_query_rewrite({"intent": "academic"}) == "academic"
        assert route_after_query_rewrite({}) == "academic"

    def test_academic_router_no_longer_points_to_query_rewriter(self):
        graph = build_graph()
        assert ("academic_router", "search_query_rewriter") not in graph.edges
        assert ("academic_router", "rag_retrieve") in graph.edges
        assert ("academic_router", "web_search") in graph.edges

    def test_academic_retrieval_uses_evidence_judge_barrier(self):
        graph = build_graph()
        assert "evidence_judge" in graph.nodes
        assert (("rag_retrieve", "web_search"), "evidence_judge") in graph.waiting_edges
        old_direct_edges = {
            ("rag_retrieve", "generate_answer"),
            ("web_search", "generate_answer"),
            ("rag_retrieve", "mindmap_planner"),
            ("web_search", "mindmap_planner"),
            ("rag_retrieve", "exercise_planner"),
            ("web_search", "exercise_planner"),
            ("rag_retrieve", "review_doc_planner"),
            ("web_search", "review_doc_planner"),
        }
        assert old_direct_edges.isdisjoint(graph.edges)
        assert "route_after_evidence_judge" in graph.branches["evidence_judge"]

    def test_search_query_rewriter_is_shared_after_supervisor(self):
        graph = build_graph()
        assert "search_query_rewriter" in graph.branches
        assert "route_after_query_rewrite" in graph.branches["search_query_rewriter"]

    def test_study_plan_reviewer_fan_in_uses_barrier(self):
        graph = build_graph()
        assert (
            ("study_plan_reviewer_academic", "study_plan_reviewer_emotional"),
            "study_plan_consensus",
        ) in graph.waiting_edges
        assert ("study_plan_reviewer_academic", "study_plan_consensus") not in graph.edges
        assert ("study_plan_reviewer_emotional", "study_plan_consensus") not in graph.edges

    @pytest.mark.anyio
    async def test_evidence_judge_runs_once_after_local_and_web_candidates(self):
        calls = []

        async def fake_supervisor(state):
            return {"intent": "academic"}

        async def fake_search_query_rewriter(state):
            return {}

        async def fake_academic_router(state):
            return {}

        async def fake_rag_retrieve(state):
            return {
                "local_evidence_candidates": [{"evidence_id": "local:math:0", "source_type": "local_rag"}],
                "local_evidence_originals": {"local:math:0": {"content": "local"}},
            }

        async def fake_web_search(state):
            return {
                "web_evidence_candidates": [{"evidence_id": "web:math:0", "source_type": "web"}],
                "web_evidence_originals": {"web:math:0": {"content": "web"}},
            }

        async def fake_evidence_judge(state):
            calls.append({
                "local": list(state.get("local_evidence_candidates") or []),
                "web": list(state.get("web_evidence_candidates") or []),
            })
            return {"context": [{"type": "rag", "content": "judged"}]}

        async def fake_generate_answer(state):
            return {"messages": [AIMessage(content="answer")]}

        async def fake_evaluate_hallucination(state):
            return {"hallucination_detected": False}

        with (
            patch("src.graph.builder.supervisor_node", fake_supervisor),
            patch("src.graph.builder.search_query_rewriter", fake_search_query_rewriter),
            patch("src.graph.builder.academic_router", fake_academic_router),
            patch("src.graph.builder.rag_retrieve", fake_rag_retrieve),
            patch("src.graph.builder.web_search", fake_web_search),
            patch("src.graph.builder.evidence_judge", fake_evidence_judge),
            patch("src.graph.builder.generate_answer", fake_generate_answer),
            patch("src.graph.builder.evaluate_hallucination", fake_evaluate_hallucination),
        ):
            compiled = build_graph().compile()
            result = await compiled.ainvoke({"messages": [HumanMessage(content="test")]})

        assert len(calls) == 1
        assert calls[0]["local"] == [{"evidence_id": "local:math:0", "source_type": "local_rag"}]
        assert calls[0]["web"] == [{"evidence_id": "web:math:0", "source_type": "web"}]
        assert result["context"] == [{"type": "rag", "content": "judged"}]

    @pytest.mark.anyio
    async def test_evidence_judge_failure_blocks_generation(self):
        generate_called = False

        async def fake_supervisor(state):
            return {"intent": "academic"}

        async def fake_search_query_rewriter(state):
            return {}

        async def fake_academic_router(state):
            return {}

        async def fake_rag_retrieve(state):
            return {"local_evidence_candidates": [{"evidence_id": "local:math:0", "source_type": "local_rag"}]}

        async def fake_web_search(state):
            return {"web_evidence_candidates": [{"evidence_id": "web:math:0", "source_type": "web"}]}

        async def fake_evidence_judge(state):
            raise RuntimeError("judge failed")

        async def fake_generate_answer(state):
            nonlocal generate_called
            generate_called = True
            return {"messages": [AIMessage(content="should not happen")]}

        with (
            patch("src.graph.builder.supervisor_node", fake_supervisor),
            patch("src.graph.builder.search_query_rewriter", fake_search_query_rewriter),
            patch("src.graph.builder.academic_router", fake_academic_router),
            patch("src.graph.builder.rag_retrieve", fake_rag_retrieve),
            patch("src.graph.builder.web_search", fake_web_search),
            patch("src.graph.builder.evidence_judge", fake_evidence_judge),
            patch("src.graph.builder.generate_answer", fake_generate_answer),
        ):
            compiled = build_graph().compile()
            with pytest.raises(RuntimeError, match="judge failed"):
                await compiled.ainvoke({"messages": [HumanMessage(content="test")]})

        assert generate_called is False
