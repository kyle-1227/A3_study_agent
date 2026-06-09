"""Unit tests for graph construction and compilation."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.graph.builder import (
    build_graph,
    get_compiled_graph,
    route_after_academic_retrieval,
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
            "generate_answer",
            "evaluate_hallucination",
            "rewrite_query",
            "gather_planning_context",
            "gather_intel",
            "drafter",
            "reviewer_academic",
            "reviewer_emotional",
            "consensus_check",
            "adv_rewrite",
            "plan_output",
            "feedback_router",
            "plan_tweak",
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

    def test_route_after_academic_retrieval_routes_resource_chains(self):
        assert route_after_academic_retrieval({"requested_resource_type": "mindmap"}) == "mindmap"
        assert route_after_academic_retrieval({"needs_mindmap": False, "requested_resource_type": "quiz"}) == "exercise"
        assert route_after_academic_retrieval({"needs_mindmap": False, "requested_resource_type": ""}) == "answer"
        assert route_after_academic_retrieval({"needs_mindmap": True, "requested_resource_type": "quiz"}) == "exercise"
        assert route_after_academic_retrieval({}) == "answer"

    def test_route_after_query_rewrite_routes_planning_and_academic(self):
        assert route_after_query_rewrite({"intent": "planning"}) == "planning"
        assert route_after_query_rewrite({"intent": "academic"}) == "academic"
        assert route_after_query_rewrite({}) == "academic"

    def test_academic_router_no_longer_points_to_query_rewriter(self):
        graph = build_graph()
        assert ("academic_router", "search_query_rewriter") not in graph.edges
        assert ("academic_router", "rag_retrieve") in graph.edges
        assert ("academic_router", "web_search") in graph.edges

    def test_search_query_rewriter_is_shared_after_supervisor(self):
        graph = build_graph()
        assert "search_query_rewriter" in graph.branches
        assert "route_after_query_rewrite" in graph.branches["search_query_rewriter"]
