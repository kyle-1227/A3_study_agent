"""Graph construction: assemble Supervisor branches and compile."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.graph.academic import (
    academic_router,
    episodic_memory_retriever,
    episodic_memory_writer,
    evidence_judge,
    evidence_summary_output,
    evaluate_hallucination,
    generate_answer,
    memory_use_decider,
    rag_retrieve,
    rewrite_query,
    search_query_rewriter,
    should_retry_or_end,
    web_search,
)
from src.graph.emotional import emotional_response
from src.graph.exercises import (
    exercise_agent,
    exercise_output,
    exercise_planner,
    exercise_reviewer,
    exercise_rewrite,
    should_rewrite_exercise,
)
from src.graph.mindmap import (
    mindmap_agent,
    mindmap_output,
    mindmap_planner,
    mindmap_reviewer,
    mindmap_rewrite,
    should_rewrite_mindmap,
)
from src.graph.review_doc import (
    review_doc_agent,
    review_doc_output,
    review_doc_planner,
    review_doc_reviewer,
    review_doc_rewrite,
    should_rewrite_review_doc,
)
from src.graph.resource_generation import (
    dispatch_resource_workers,
    normalize_requested_resource_types,
    resource_bundle_output,
    resource_orchestrator,
    resource_worker,
)
from src.graph.state import LearningState
from src.graph.study_plan import (
    route_after_study_plan_consensus,
    study_plan_agent,
    study_plan_consensus,
    study_plan_emotional_intel,
    study_plan_output,
    study_plan_planner,
    study_plan_reviewer_academic,
    study_plan_reviewer_emotional,
    study_plan_rewrite,
)
from src.graph.supervisor import handle_unknown, route_by_intent, supervisor_node


def build_graph() -> StateGraph:
    """Construct the full LangGraph StateGraph (uncompiled)."""

    # Build graph
    graph = StateGraph(LearningState)

    # Nodes
    graph.add_node("supervisor", supervisor_node)

    # SubGraph A: Academic (parallel retrieval + answer generation)
    graph.add_node("episodic_memory_retriever", episodic_memory_retriever)
    graph.add_node("episodic_memory_writer", episodic_memory_writer)
    graph.add_node("academic_router", academic_router)
    graph.add_node("memory_use_decider", memory_use_decider)
    graph.add_node("search_query_rewriter", search_query_rewriter)
    graph.add_node("rag_retrieve", rag_retrieve)
    graph.add_node("web_search", web_search)
    graph.add_node("evidence_judge", evidence_judge)
    graph.add_node("evidence_summary_output", evidence_summary_output)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("evaluate_hallucination", evaluate_hallucination)
    graph.add_node("rewrite_query", rewrite_query)

    # Emotional support
    graph.add_node("emotional_response", emotional_response)

    # Mindmap resource generation
    graph.add_node("mindmap_planner", mindmap_planner)
    graph.add_node("mindmap_agent", mindmap_agent)
    graph.add_node("mindmap_reviewer", mindmap_reviewer)
    graph.add_node("mindmap_rewrite", mindmap_rewrite)
    graph.add_node("mindmap_output", mindmap_output)

    # Exercise resource generation
    graph.add_node("exercise_planner", exercise_planner)
    graph.add_node("exercise_agent", exercise_agent)
    graph.add_node("exercise_reviewer", exercise_reviewer)
    graph.add_node("exercise_rewrite", exercise_rewrite)
    graph.add_node("exercise_output", exercise_output)

    # Review document resource generation
    graph.add_node("review_doc_planner", review_doc_planner)
    graph.add_node("review_doc_agent", review_doc_agent)
    graph.add_node("review_doc_reviewer", review_doc_reviewer)
    graph.add_node("review_doc_rewrite", review_doc_rewrite)
    graph.add_node("review_doc_output", review_doc_output)

    # Study plan resource generation
    graph.add_node("resource_orchestrator", resource_orchestrator)
    graph.add_node("resource_worker", resource_worker)
    graph.add_node("resource_bundle_output", resource_bundle_output)
    graph.add_node("study_plan_emotional_intel", study_plan_emotional_intel)
    graph.add_node("study_plan_planner", study_plan_planner)
    graph.add_node("study_plan_agent", study_plan_agent)
    graph.add_node("study_plan_reviewer_academic", study_plan_reviewer_academic)
    graph.add_node("study_plan_reviewer_emotional", study_plan_reviewer_emotional)
    graph.add_node("study_plan_consensus", study_plan_consensus)
    graph.add_node("study_plan_rewrite", study_plan_rewrite)
    graph.add_node("study_plan_output", study_plan_output)

    # Unknown / off-topic
    graph.add_node("handle_unknown", handle_unknown)

    # Edges
    graph.set_entry_point("supervisor")

    # Conditional fork edges
    graph.add_conditional_edges(
        "supervisor",
        route_by_intent,    # judge users intent
        {
            "academic": "episodic_memory_retriever",
            "emotional": "emotional_response",
            "unknown": "handle_unknown",
        },
    )

    # Retrieve long-term episodic/semantic memories before memory use decision.
    graph.add_edge("episodic_memory_retriever", "memory_use_decider")
    # Decide whether historical evidence memory may influence retrieval.
    graph.add_edge("memory_use_decider", "search_query_rewriter")

    # Shared initial query rewrite, then route into academic evidence flow.
    graph.add_conditional_edges(
        "search_query_rewriter",
        route_after_query_rewrite,
        {
            "academic": "academic_router",
        },
    )

    # Academic flow: fan-out/fan-in parallel retrieval
    graph.add_edge("academic_router", "rag_retrieve")
    graph.add_edge("academic_router", "web_search")

    # Barrier fan-in: Evidence Judge runs once after Local RAG and Tavily both finish.
    graph.add_edge(["rag_retrieve", "web_search"], "evidence_judge")

    # Fan-in routing: only judged context may enter answer/resource generation.
    graph.add_conditional_edges(
        "evidence_judge",
        route_after_evidence_judge,
        {
            "answer": "generate_answer",
            "resources": "resource_orchestrator",
            "evidence_summary_output": "evidence_summary_output",
        },
    )
    graph.add_edge("evidence_summary_output", END)

    graph.add_conditional_edges("resource_orchestrator", dispatch_resource_workers)
    graph.add_edge("resource_worker", "resource_bundle_output")
    graph.add_edge("resource_bundle_output", END)

    # Hallucination evaluation with retry loop
    graph.add_edge("generate_answer", "evaluate_hallucination")
    graph.add_conditional_edges(
        "evaluate_hallucination",
        should_retry_or_end,
        {
            "retry": "rewrite_query",
            "end": "episodic_memory_writer",
        },
    )
    graph.add_edge("episodic_memory_writer", END)
    graph.add_edge("rewrite_query", "academic_router")

    # Emotional support ends after the response node.
    graph.add_edge("emotional_response", END)

    # Mindmap resource generation: plan -> JSON tree -> review -> export
    graph.add_edge("mindmap_planner", "mindmap_agent")
    graph.add_edge("mindmap_agent", "mindmap_reviewer")
    graph.add_conditional_edges(
        "mindmap_reviewer",
        should_rewrite_mindmap,
        {
            "rewrite": "mindmap_rewrite",
            "output": "mindmap_output",
        },
    )
    graph.add_edge("mindmap_rewrite", "mindmap_agent")
    graph.add_edge("mindmap_output", END)

    # Exercise resource generation: plan -> structured exercises -> review -> output
    graph.add_edge("exercise_planner", "exercise_agent")
    graph.add_edge("exercise_agent", "exercise_reviewer")
    graph.add_conditional_edges(
        "exercise_reviewer",
        should_rewrite_exercise,
        {
            "rewrite": "exercise_rewrite",
            "output": "exercise_output",
        },
    )
    graph.add_edge("exercise_rewrite", "exercise_agent")
    graph.add_edge("exercise_output", END)

    # Review document resource generation: plan -> Markdown -> review -> output
    graph.add_edge("review_doc_planner", "review_doc_agent")
    graph.add_edge("review_doc_agent", "review_doc_reviewer")
    graph.add_conditional_edges(
        "review_doc_reviewer",
        should_rewrite_review_doc,
        {
            "rewrite": "review_doc_rewrite",
            "output": "review_doc_output",
        },
    )
    graph.add_edge("review_doc_rewrite", "review_doc_agent")
    graph.add_edge("review_doc_output", END)

    graph.add_edge("study_plan_emotional_intel", "study_plan_planner")
    graph.add_edge("study_plan_planner", "study_plan_agent")
    graph.add_edge("study_plan_agent", "study_plan_reviewer_academic")
    graph.add_edge("study_plan_agent", "study_plan_reviewer_emotional")
    graph.add_edge(["study_plan_reviewer_academic", "study_plan_reviewer_emotional"], "study_plan_consensus")
    graph.add_conditional_edges(
        "study_plan_consensus",
        route_after_study_plan_consensus,
        {
            "rewrite": "study_plan_rewrite",
            "output": "study_plan_output",
        },
    )
    graph.add_edge("study_plan_rewrite", "study_plan_agent")
    graph.add_edge("study_plan_output", END)

    # Unknown: direct to END
    graph.add_edge("handle_unknown", END)

    return graph


def route_after_evidence_judge(state: LearningState) -> str:
    """Route judged evidence to answer generation, resource chains, or controlled stop."""
    if state.get("evidence_controlled_stop"):
        return "evidence_summary_output"

    resource_types = normalize_requested_resource_types(
        state.get("requested_resource_types") or [],
        state.get("requested_resource_type") or "",
    )
    if resource_types:
        return "resources"
    return "answer"


# Backward-compatible import alias for older tests/tools; do not attach this to retrieval nodes.
route_after_academic_retrieval = route_after_evidence_judge


def route_after_query_rewrite(state: LearningState) -> str:
    """Route shared query rewrite output into the academic evidence path."""
    return "academic"


def get_compiled_graph(checkpointer=None):
    """Build and compile the graph, ready for invocation.

    Args:
        checkpointer: Optional LangGraph checkpointer for persistent state.
                      When provided, the graph saves/restores state per thread_id.
    """
    return build_graph().compile(checkpointer=checkpointer)
