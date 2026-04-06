"""Graph construction — assemble Supervisor + 3 branches, compile."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.graph.academic import (
    academic_router,
    evaluate_hallucination,
    generate_answer,
    rag_retrieve,
    rewrite_query,
    should_retry_or_end,
    web_search,
)
from src.graph.emotional import emotional_response
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

    # Unknown / off-topic
    graph.add_node("handle_unknown", handle_unknown)

    # ── Edges ────────────────────────────────────────────────────────
    graph.set_entry_point("supervisor")

    # Conditional fork edges
    graph.add_conditional_edges(
        "supervisor",
        route_by_intent,    # judge users intent
        {
            "academic": "academic_router",
            "planning": "search_policy",
            "emotional": "emotional_response",
            "unknown": "handle_unknown",
        },
    )

    # Academic flow — fan-out/fan-in parallel retrieval
    graph.add_edge("academic_router", "rag_retrieve")
    graph.add_edge("academic_router", "web_search")

    # Fan-in: both converge at generate_answer
    graph.add_edge("rag_retrieve", "generate_answer")
    graph.add_edge("web_search", "generate_answer")

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

    # Unknown — direct to END
    graph.add_edge("handle_unknown", END)

    return graph


def get_compiled_graph(checkpointer=None):
    """Build and compile the graph, ready for invocation.

    Args:
        checkpointer: Optional LangGraph checkpointer for persistent state.
                      When provided, the graph saves/restores state per thread_id.
    """
    return build_graph().compile(checkpointer=checkpointer)
