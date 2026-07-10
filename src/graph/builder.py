"""Graph construction: assemble Supervisor branches and compile."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.context_engineering.influence_runtime import wrap_context_influence_node
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
from src.graph.resource_generation import (
    SUPPORTED_RESOURCE_TYPES,
    dispatch_resource_workers,
    normalize_requested_resource_types,
    resource_preflight_router,
    resource_bundle_output,
    resource_orchestrator,
    resource_worker,
    route_after_resource_preflight,
)
from src.graph.qa import qa_agent
from src.graph.state import LearningState
from src.graph.study_plan import study_plan_profile_gate_main
from src.graph.supervisor import handle_unknown, route_after_supervisor, supervisor_node
from src.run_control import wrap_interruptible_node


def build_graph() -> StateGraph:
    """Construct the full LangGraph StateGraph (uncompiled)."""

    # Build graph
    graph = StateGraph(LearningState)

    def add_interruptible_node(node_name: str, node_fn) -> None:
        interruptible = wrap_interruptible_node(node_name, node_fn)
        graph.add_node(
            node_name,
            wrap_context_influence_node(node_name, interruptible),
        )

    # Nodes
    add_interruptible_node("supervisor", supervisor_node)

    # SubGraph A: Academic (parallel retrieval + answer generation)
    add_interruptible_node("episodic_memory_retriever", episodic_memory_retriever)
    add_interruptible_node("episodic_memory_writer", episodic_memory_writer)
    add_interruptible_node("academic_router", academic_router)
    add_interruptible_node("memory_use_decider", memory_use_decider)
    add_interruptible_node("search_query_rewriter", search_query_rewriter)
    add_interruptible_node("rag_retrieve", rag_retrieve)
    add_interruptible_node("web_search", web_search)
    add_interruptible_node("evidence_judge", evidence_judge)
    add_interruptible_node("evidence_summary_output", evidence_summary_output)
    add_interruptible_node("generate_answer", generate_answer)
    add_interruptible_node("evaluate_hallucination", evaluate_hallucination)
    add_interruptible_node("rewrite_query", rewrite_query)

    # Emotional support
    add_interruptible_node("emotional_response", emotional_response)

    # First-class structured question answering
    add_interruptible_node("qa_agent", qa_agent)

    # Unified resource generation
    add_interruptible_node("resource_preflight_router", resource_preflight_router)
    add_interruptible_node(
        "study_plan_profile_gate_main",
        study_plan_profile_gate_main,
    )
    add_interruptible_node("resource_orchestrator", resource_orchestrator)
    add_interruptible_node("resource_worker", resource_worker)
    add_interruptible_node("resource_bundle_output", resource_bundle_output)

    # Unknown / off-topic
    add_interruptible_node("handle_unknown", handle_unknown)

    # Edges
    graph.set_entry_point("supervisor")

    # Conditional fork edges
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "academic": "episodic_memory_retriever",
            "emotional": "emotional_response",
            "qa": "qa_agent",
            "invalid": "handle_unknown",
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
            "qa": "qa_agent",
            "resources": "resource_preflight_router",
            "evidence_summary_output": "evidence_summary_output",
        },
    )
    graph.add_edge("evidence_summary_output", END)

    graph.add_conditional_edges(
        "resource_preflight_router",
        route_after_resource_preflight,
        {
            "study_plan_profile_gate_main": "study_plan_profile_gate_main",
            "resource_orchestrator": "resource_orchestrator",
        },
    )
    graph.add_edge("study_plan_profile_gate_main", "resource_orchestrator")
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
    graph.add_edge("qa_agent", END)

    # Unknown: legacy handler remains available for compatibility.
    graph.add_edge("handle_unknown", END)

    return graph


def route_after_evidence_judge(state: LearningState) -> str:
    """Route judged evidence to answer generation, resource chains, or controlled stop."""
    response_mode = str(state.get("response_mode") or "").strip()
    qa_scope = str(state.get("qa_scope") or "").strip()
    if response_mode == "qa":
        if qa_scope != "academic":
            raise ValueError("retrieval-backed QA requires qa_scope=academic")
        return "qa"
    if response_mode not in {"", "resource"}:
        raise ValueError("evidence judge received an invalid response_mode")

    requested_resource_type = str(state.get("requested_resource_type") or "").strip()
    requested_resource_types = [
        str(item or "").strip()
        for item in state.get("requested_resource_types", []) or []
        if str(item or "").strip()
    ]
    has_explicit_resource_request = bool(
        requested_resource_type in SUPPORTED_RESOURCE_TYPES
        or any(item in SUPPORTED_RESOURCE_TYPES for item in requested_resource_types)
    )
    if state.get("evidence_controlled_stop") and not has_explicit_resource_request:
        return "evidence_summary_output"

    resource_types = normalize_requested_resource_types(
        requested_resource_types,
        requested_resource_type,
    )
    if resource_types:
        return "resources"
    return "answer"


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
