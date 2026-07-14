"""Graph construction: assemble Supervisor branches and compile."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
from src.graph.learning_guidance import (
    make_learner_path_planner_node,
    make_resource_recommendation_node,
)
from src.graph.resource_contracts import (
    SUPPORTED_RESOURCE_TYPES,
    normalize_requested_resource_types,
)
from src.graph.resource_generation import (
    dispatch_resource_workers,
    dispatch_resource_workers_to_recommendation_aggregator,
    make_resource_bundle_output_with_recommendations_node,
    resource_preflight_router,
    resource_bundle_aggregator,
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

if TYPE_CHECKING:
    from src.graph.evidence_orchestration import EvidenceOrchestrationRuntime
    from src.graph.parent_child_nodes import ParentChildGraphRuntime
else:
    EvidenceOrchestrationRuntime = object
    ParentChildGraphRuntime = object


def _build_graph_with_academic_nodes(
    *,
    rag_node,
    post_evidence_node,
) -> StateGraph:
    """Construct one graph with explicitly supplied local retrieval boundaries."""

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
    add_interruptible_node("rag_retrieve", rag_node)
    add_interruptible_node("web_search", web_search)
    add_interruptible_node("evidence_judge", evidence_judge)
    if post_evidence_node is not None:
        add_interruptible_node("parent_child_parent_hydration", post_evidence_node)
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
    evidence_route_source = "evidence_judge"
    if post_evidence_node is not None:
        graph.add_edge("evidence_judge", "parent_child_parent_hydration")
        evidence_route_source = "parent_child_parent_hydration"
    graph.add_conditional_edges(
        evidence_route_source,
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


def build_graph() -> StateGraph:
    """Construct the legacy-served graph without enabling the candidate route."""

    return _build_graph_with_academic_nodes(
        rag_node=rag_retrieve,
        post_evidence_node=None,
    )


def build_parent_child_graph(runtime: ParentChildGraphRuntime) -> StateGraph:
    """Construct the explicit candidate graph pinned to one generation runtime."""

    from src.graph.parent_child_nodes import (
        ParentChildGraphRuntime as ParentChildGraphRuntimeType,
        make_parent_child_hydration_node,
        make_parent_child_rag_node,
    )

    if not isinstance(runtime, ParentChildGraphRuntimeType):
        raise TypeError("runtime must be a validated ParentChildGraphRuntime")
    return _build_graph_with_academic_nodes(
        rag_node=make_parent_child_rag_node(runtime),
        post_evidence_node=make_parent_child_hydration_node(runtime),
    )


def build_resource_evidence_parent_child_graph(
    runtime: EvidenceOrchestrationRuntime,
) -> StateGraph:
    """Build the joint Parent-Child plus resource-evidence candidate graph."""

    from src.graph.evidence_orchestration import (
        EvidenceOrchestrationRuntime as EvidenceOrchestrationRuntimeType,
        make_evidence_repair_planner_node,
        make_local_rag_search_batch_node,
        make_requirement_evidence_judge_node,
        make_resource_evidence_assignment_node,
        make_resource_evidence_planner_node,
        make_retrieval_round_merge_node,
        make_retrieval_round_router_node,
        make_terminal_parent_hydration_node,
        make_web_research_search_batch_node,
        route_after_requirement_evidence_judge,
        validate_evidence_orchestration_runtime_binding,
    )
    from src.graph.parent_child_nodes import (
        make_parent_child_hydration_node,
        make_parent_child_rag_node,
    )

    if not isinstance(runtime, EvidenceOrchestrationRuntimeType):
        raise TypeError("runtime must be a validated EvidenceOrchestrationRuntime")

    graph = StateGraph(LearningState)

    def add_interruptible_node(node_name: str, node_fn) -> None:
        interruptible = wrap_interruptible_node(node_name, node_fn)
        graph.add_node(
            node_name,
            wrap_context_influence_node(node_name, interruptible),
        )

    def add_candidate_runtime_node(node_name: str, node_fn) -> None:
        async def runtime_bound_node(state: LearningState) -> dict:
            validate_evidence_orchestration_runtime_binding(state, runtime)
            return await node_fn(state)

        add_interruptible_node(node_name, runtime_bound_node)

    academic_parent_hydration = make_parent_child_hydration_node(runtime.parent_child)
    resource_parent_hydration = make_terminal_parent_hydration_node(runtime)

    add_interruptible_node("supervisor", supervisor_node)
    add_interruptible_node("episodic_memory_retriever", episodic_memory_retriever)
    add_interruptible_node("episodic_memory_writer", episodic_memory_writer)
    add_interruptible_node("memory_use_decider", memory_use_decider)
    add_interruptible_node("search_query_rewriter", search_query_rewriter)
    add_interruptible_node("academic_router", academic_router)
    add_interruptible_node(
        "parent_child_retrieve",
        make_parent_child_rag_node(runtime.parent_child),
    )
    add_interruptible_node("web_research", web_search)
    add_interruptible_node("evidence_judge", evidence_judge)
    add_interruptible_node(
        "resource_evidence_planner",
        make_resource_evidence_planner_node(runtime),
    )
    add_interruptible_node(
        "learner_path_planner",
        make_learner_path_planner_node(runtime.learning_guidance),
    )
    add_interruptible_node(
        "retrieval_round_router",
        make_retrieval_round_router_node(runtime),
    )
    add_interruptible_node(
        "local_rag_search_batch",
        make_local_rag_search_batch_node(runtime),
    )
    add_interruptible_node(
        "web_research_search_batch",
        make_web_research_search_batch_node(runtime),
    )
    add_interruptible_node(
        "retrieval_round_merge",
        make_retrieval_round_merge_node(runtime),
    )
    add_interruptible_node(
        "requirement_evidence_judge",
        make_requirement_evidence_judge_node(runtime),
    )
    add_interruptible_node(
        "evidence_repair_planner",
        make_evidence_repair_planner_node(runtime),
    )
    add_interruptible_node(
        "academic_parent_hydration",
        academic_parent_hydration,
    )
    add_interruptible_node(
        "resource_parent_hydration",
        resource_parent_hydration,
    )
    add_interruptible_node(
        "resource_evidence_assignment",
        make_resource_evidence_assignment_node(runtime),
    )
    add_interruptible_node("evidence_summary_output", evidence_summary_output)
    add_interruptible_node("generate_answer", generate_answer)
    add_interruptible_node("evaluate_hallucination", evaluate_hallucination)
    add_interruptible_node("rewrite_query", rewrite_query)
    add_interruptible_node("emotional_response", emotional_response)
    add_interruptible_node("qa_agent", qa_agent)
    add_candidate_runtime_node("resource_preflight_router", resource_preflight_router)
    add_candidate_runtime_node(
        "study_plan_profile_gate_main",
        study_plan_profile_gate_main,
    )
    add_candidate_runtime_node("resource_orchestrator", resource_orchestrator)
    add_candidate_runtime_node("resource_worker", resource_worker)
    add_candidate_runtime_node(
        "resource_bundle_aggregator",
        resource_bundle_aggregator,
    )
    add_candidate_runtime_node(
        "resource_recommendation_auto",
        make_resource_recommendation_node(
            runtime.learning_guidance,
            mode="automatic_after_generation",
        ),
    )
    add_candidate_runtime_node(
        "resource_bundle_output",
        make_resource_bundle_output_with_recommendations_node(
            runtime.learning_guidance,
        ),
    )
    add_interruptible_node("handle_unknown", handle_unknown)

    graph.set_entry_point("supervisor")
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
    graph.add_edge("episodic_memory_retriever", "memory_use_decider")
    graph.add_edge("memory_use_decider", "search_query_rewriter")
    graph.add_conditional_edges(
        "search_query_rewriter",
        route_after_candidate_query_rewrite,
        {
            "academic": "academic_router",
            "resource_evidence": "learner_path_planner",
        },
    )

    graph.add_edge("academic_router", "parent_child_retrieve")
    graph.add_edge("academic_router", "web_research")
    graph.add_edge(
        ["parent_child_retrieve", "web_research"],
        "evidence_judge",
    )
    graph.add_edge("evidence_judge", "academic_parent_hydration")
    graph.add_conditional_edges(
        "academic_parent_hydration",
        route_after_evidence_judge,
        {
            "answer": "generate_answer",
            "qa": "qa_agent",
            "resources": "resource_evidence_assignment",
            "evidence_summary_output": "evidence_summary_output",
        },
    )

    graph.add_edge("learner_path_planner", "resource_evidence_planner")
    graph.add_edge("resource_evidence_planner", "retrieval_round_router")
    graph.add_edge("retrieval_round_router", "local_rag_search_batch")
    graph.add_edge("retrieval_round_router", "web_research_search_batch")
    graph.add_edge(
        ["local_rag_search_batch", "web_research_search_batch"],
        "retrieval_round_merge",
    )
    graph.add_edge("retrieval_round_merge", "requirement_evidence_judge")
    graph.add_conditional_edges(
        "requirement_evidence_judge",
        route_after_requirement_evidence_judge,
        {
            "repair": "evidence_repair_planner",
            "terminal": "resource_parent_hydration",
        },
    )
    graph.add_edge("evidence_repair_planner", "retrieval_round_router")
    graph.add_edge("resource_parent_hydration", "resource_evidence_assignment")
    graph.add_edge("resource_evidence_assignment", "resource_preflight_router")
    graph.add_conditional_edges(
        "resource_preflight_router",
        route_after_resource_preflight,
        {
            "study_plan_profile_gate_main": "study_plan_profile_gate_main",
            "resource_orchestrator": "resource_orchestrator",
        },
    )
    graph.add_edge("study_plan_profile_gate_main", "resource_orchestrator")
    graph.add_conditional_edges(
        "resource_orchestrator",
        dispatch_resource_workers_to_recommendation_aggregator,
    )
    graph.add_edge("resource_worker", "resource_bundle_aggregator")
    graph.add_edge("resource_bundle_aggregator", "resource_recommendation_auto")
    graph.add_edge("resource_recommendation_auto", "resource_bundle_output")
    graph.add_edge("resource_bundle_output", END)

    graph.add_edge("evidence_summary_output", END)
    graph.add_edge("generate_answer", "evaluate_hallucination")
    graph.add_conditional_edges(
        "evaluate_hallucination",
        should_retry_or_end,
        {"retry": "rewrite_query", "end": "episodic_memory_writer"},
    )
    graph.add_edge("episodic_memory_writer", END)
    graph.add_edge("rewrite_query", "academic_router")
    graph.add_edge("emotional_response", END)
    graph.add_edge("qa_agent", END)
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


def route_after_candidate_query_rewrite(state: LearningState) -> str:
    """Route explicit resource requests to the joint evidence candidate only."""

    response_mode = str(state.get("response_mode") or "").strip()
    requested = normalize_requested_resource_types(
        state.get("requested_resource_types") or [],
        state.get("requested_resource_type") or "",
    )
    if response_mode == "resource":
        if not requested:
            raise ValueError("resource response mode requires canonical resource types")
        return "resource_evidence"
    if response_mode in {"", "qa"}:
        return "academic"
    raise ValueError("candidate query rewrite received an invalid response_mode")


def get_compiled_graph(checkpointer=None):
    """Build and compile the graph, ready for invocation.

    Args:
        checkpointer: Optional LangGraph checkpointer for persistent state.
                      When provided, the graph saves/restores state per thread_id.
    """
    return build_graph().compile(checkpointer=checkpointer)


def get_compiled_parent_child_graph(
    runtime: ParentChildGraphRuntime,
    *,
    checkpointer,
):
    """Compile the explicit candidate graph; callers must supply all dependencies."""

    return build_parent_child_graph(runtime).compile(checkpointer=checkpointer)


def get_compiled_resource_evidence_parent_child_graph(
    runtime: EvidenceOrchestrationRuntime,
    *,
    checkpointer,
):
    """Compile the joint candidate; callers must inject every strict dependency."""

    return build_resource_evidence_parent_child_graph(runtime).compile(
        checkpointer=checkpointer
    )
