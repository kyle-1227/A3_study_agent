"""Graph construction — assemble Supervisor + 3 branches, compile."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.graph.academic import (
    academic_router,
    evaluate_hallucination,
    generate_answer,
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
from src.graph.plan_adversarial import (
    adv_rewrite_node,
    consensus_check_node,
    drafter_node,
    feedback_router,
    plan_output_node,
    plan_tweak_node,
    reviewer_academic_node,
    reviewer_emotional_node,
    route_after_hil,
    route_feedback,
    should_output_or_revise,
)
from src.graph.planner import gather_intel, search_policy
from src.graph.state import TutorState
from src.graph.supervisor import handle_unknown, route_by_intent, supervisor_node


def build_graph() -> StateGraph:
    """Construct the full LangGraph StateGraph (uncompiled)."""

    # Build graph
    graph = StateGraph(TutorState)

    # ── Nodes ────────────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor_node)

    # SubGraph A — Academic (parallel retrieval + answer generation)
    graph.add_node("academic_router", academic_router)
    graph.add_node("search_query_rewriter", search_query_rewriter)
    graph.add_node("rag_retrieve", rag_retrieve)
    graph.add_node("web_search", web_search)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("evaluate_hallucination", evaluate_hallucination)
    graph.add_node("rewrite_query", rewrite_query)

    # Planner (gather intel → flattened adversarial planning)
    graph.add_node("search_policy", search_policy)
    graph.add_node("gather_intel", gather_intel)
    graph.add_node("drafter", drafter_node)
    graph.add_node("reviewer_academic", reviewer_academic_node)
    graph.add_node("reviewer_emotional", reviewer_emotional_node)
    graph.add_node("consensus_check", consensus_check_node)
    graph.add_node("adv_rewrite", adv_rewrite_node)
    graph.add_node("plan_output", plan_output_node)
    graph.add_node("feedback_router", feedback_router)
    graph.add_node("plan_tweak", plan_tweak_node)

    # Emotional
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

    # Unknown / off-topic
    graph.add_node("handle_unknown", handle_unknown)

    # ── Edges ────────────────────────────────────────────────────────
    graph.set_entry_point("supervisor")

    # Conditional fork edges
    graph.add_conditional_edges(
        "supervisor",
        route_by_intent,    # judge users intent
        {
            "academic": "search_query_rewriter",
            "planning": "search_query_rewriter",
            "emotional": "emotional_response",
            "unknown": "handle_unknown",
        },
    )

    # Shared initial query rewrite, then route into academic or planning flow.
    graph.add_conditional_edges(
        "search_query_rewriter",
        route_after_query_rewrite,
        {
            "academic": "academic_router",
            "planning": "search_policy",
        },
    )

    # Academic flow — fan-out/fan-in parallel retrieval
    graph.add_edge("academic_router", "rag_retrieve")
    graph.add_edge("academic_router", "web_search")

    # Fan-in: ordinary academic requests converge at answer generation;
    # resource requests reuse retrieval first, then enter sibling resource chains.
    graph.add_conditional_edges(
        "rag_retrieve",
        route_after_academic_retrieval,
        {
            "answer": "generate_answer",
            "mindmap": "mindmap_planner",
            "exercise": "exercise_planner",
        },
    )
    graph.add_conditional_edges(
        "web_search",
        route_after_academic_retrieval,
        {
            "answer": "generate_answer",
            "mindmap": "mindmap_planner",
            "exercise": "exercise_planner",
        },
    )

    # Hallucination evaluation with retry loop
    graph.add_edge("generate_answer", "evaluate_hallucination")
    graph.add_conditional_edges(
        "evaluate_hallucination",
        should_retry_or_end,
        {
            "retry": "rewrite_query",
            "end": END,
        },
    )
    graph.add_edge("rewrite_query", "academic_router")

    # Planner flow: search_policy → gather_intel → adversarial loop → plan_output → END
    graph.add_edge("search_policy", "gather_intel")
    graph.add_edge("gather_intel", "drafter")
    graph.add_edge("drafter", "reviewer_academic")
    graph.add_edge("drafter", "reviewer_emotional")
    graph.add_edge("reviewer_academic", "consensus_check")
    graph.add_edge("reviewer_emotional", "consensus_check")
    graph.add_conditional_edges(
        "consensus_check",
        should_output_or_revise,
        {
            "output": "plan_output",
            "revise": "adv_rewrite",
        },
    )
    graph.add_edge("adv_rewrite", "drafter")
    graph.add_conditional_edges(
        "plan_output",
        route_after_hil,
        {"end": END, "feedback": "feedback_router"},
    )
    graph.add_conditional_edges(
        "feedback_router",
        route_feedback,
        {"tweak": "plan_tweak", "rewrite": "drafter"},
    )
    graph.add_edge("plan_tweak", "plan_output")

    # Emotional — direct to END
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

    # Unknown — direct to END
    graph.add_edge("handle_unknown", END)

    return graph


def route_after_academic_retrieval(state: TutorState) -> str:
    """Route retrieval fan-in to answer generation or resource chains."""
    resource_type = state.get("requested_resource_type")
    if resource_type == "mindmap":
        return "mindmap"
    if resource_type == "quiz":
        return "exercise"
    return "answer"


def route_after_query_rewrite(state: TutorState) -> str:
    """Route shared query rewrite output to planning or academic flow."""
    return "planning" if state.get("intent") == "planning" else "academic"


def get_compiled_graph(checkpointer=None):
    """Build and compile the graph, ready for invocation.

    Args:
        checkpointer: Optional LangGraph checkpointer for persistent state.
                      When provided, the graph saves/restores state per thread_id.
    """
    return build_graph().compile(checkpointer=checkpointer)
