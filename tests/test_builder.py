"""Unit tests for graph construction and compilation."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.graph.builder import build_graph, get_compiled_graph, route_after_academic_retrieval


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
            "rag_retrieve",
            "web_search",
            "generate_answer",
            "evaluate_hallucination",
            "rewrite_query",
            "search_policy",
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
