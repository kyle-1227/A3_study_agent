"""Adversarial Planning SubGraph (REQ-07).

Drafter → [Reviewer Academic ∥ Reviewer Emotional] → Consensus Check → (loop/output)

The SubGraph runs an adversarial review loop: a drafter produces a study plan,
two parallel reviewers (academic quality + emotional wellbeing) evaluate it,
and a consensus check decides whether to accept or request revisions.
A safety valve (max_rounds) forces output after N iterations.

ADR-001: SubGraph encapsulation with independent state.
ADR-002: Reviewer verdicts via with_structured_output() + Pydantic.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from src.config import get_setting, load_prompt
from src.graph.llm import async_invoke_with_fallback, get_fallback_llm, get_node_llm
from src.tracing import traced_llm_call, traced_node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State & models
# ---------------------------------------------------------------------------


class PlanAdversarialState(TypedDict):
    intel_summary: str
    user_request: str
    draft: str
    academic_verdict: str
    emotional_verdict: str
    round: int
    max_rounds: int
    consensus: bool
    revision_notes: str


class ReviewVerdict(BaseModel):
    verdict: Literal["approve", "reject"]
    reason: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


@traced_node
async def drafter_node(state: PlanAdversarialState) -> dict[str, Any]:
    """Draft or rewrite a study plan based on intel and any revision notes."""
    llm = get_node_llm("planner")
    temperature = get_setting("planner.temperature", 0.7)
    fallback = get_fallback_llm(temperature=temperature)

    user_request = state.get("user_request", "")
    intel_summary = state.get("intel_summary", "")
    revision_notes = state.get("revision_notes", "")

    if revision_notes:
        # Rewrite path: incorporate reviewer feedback
        prompt_text = load_prompt("plan_rewrite").format(
            user_request=user_request,
            intel_summary=intel_summary,
            current_draft=state.get("draft", ""),
            revision_notes=revision_notes,
        )
    else:
        # First draft
        prompt_text = load_prompt("plan_drafter").format(
            user_request=user_request,
            intel_summary=intel_summary,
        )

    messages = [
        SystemMessage(content=load_prompt("plan_drafter_system")),
        HumanMessage(content=prompt_text),
    ]

    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        node_name="drafter_node",
        temperature=temperature,
    ) as span:
        response = await async_invoke_with_fallback(
            llm, messages, fallback=fallback, span=span,
        )

    return {
        "draft": response.content,
        "round": state.get("round", 0) + 1,
    }


async def _run_reviewer(
    state: PlanAdversarialState,
    *,
    system_prompt_name: str,
    node_name: str,
) -> ReviewVerdict:
    """Shared logic for academic and emotional reviewers."""
    reviewer_temp = get_setting("planner.reviewer_temperature", 0.0)
    llm = get_node_llm("planner", temperature=reviewer_temp)
    structured_primary = llm.with_structured_output(ReviewVerdict)

    fallback_llm = get_fallback_llm(temperature=reviewer_temp)
    structured_fallback = fallback_llm.with_structured_output(ReviewVerdict)

    review_prompt = (
        f"## 学习计划\n\n{state.get('draft', '')}\n\n"
        f"## 学生情况\n\n{state.get('intel_summary', '')}"
    )
    messages = [
        SystemMessage(content=load_prompt(system_prompt_name)),
        HumanMessage(content=review_prompt),
    ]

    with traced_llm_call(
        model_name=get_setting("planner.model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
        node_name=node_name,
        temperature=reviewer_temp,
    ) as span:
        try:
            verdict = await async_invoke_with_fallback(
                structured_primary, messages,
                fallback=structured_fallback, span=span,
            )
            return verdict
        except Exception:
            logger.warning("Reviewer %s failed, defaulting to approve", node_name, exc_info=True)
            return ReviewVerdict(verdict="approve", reason="审查异常，默认通过")


@traced_node
async def reviewer_academic_node(state: PlanAdversarialState) -> dict[str, Any]:
    """Academic quality reviewer."""
    verdict = await _run_reviewer(
        state,
        system_prompt_name="plan_reviewer_academic_system",
        node_name="reviewer_academic",
    )
    return {"academic_verdict": verdict.verdict}


@traced_node
async def reviewer_emotional_node(state: PlanAdversarialState) -> dict[str, Any]:
    """Emotional wellbeing reviewer."""
    verdict = await _run_reviewer(
        state,
        system_prompt_name="plan_reviewer_emotional_system",
        node_name="reviewer_emotional",
    )
    return {"emotional_verdict": verdict.verdict}


@traced_node
async def consensus_check_node(state: PlanAdversarialState) -> dict[str, Any]:
    """Check if both reviewers approved, or force output at max_rounds."""
    current_round = state.get("round", 0)
    max_rounds = state.get("max_rounds", 3)
    academic = state.get("academic_verdict", "")
    emotional = state.get("emotional_verdict", "")

    both_approve = academic == "approve" and emotional == "approve"

    # Safety valve: force consensus at max_rounds
    if current_round >= max_rounds:
        if not both_approve:
            logger.warning(
                "Max rounds (%d) reached with unresolved rejections, forcing output",
                max_rounds,
            )
        return {"consensus": True, "revision_notes": ""}

    if both_approve:
        return {"consensus": True, "revision_notes": ""}

    # Collect rejection reasons for revision
    notes_parts: list[str] = []
    if academic == "reject":
        notes_parts.append(f"[学术审查] {academic}")
    if emotional == "reject":
        notes_parts.append(f"[情绪审查] {emotional}")

    return {
        "consensus": False,
        "revision_notes": "; ".join(notes_parts) if notes_parts else "需要修改",
    }


@traced_node
async def rewrite_node(state: PlanAdversarialState) -> dict[str, Any]:
    """Reset verdicts before sending back to drafter for revision."""
    return {
        "academic_verdict": "",
        "emotional_verdict": "",
    }


@traced_node
async def output_node(state: PlanAdversarialState) -> dict[str, Any]:
    """Final output node — returns the approved draft."""
    return {"draft": state.get("draft", "")}


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------


def _should_output_or_revise(state: PlanAdversarialState) -> str:
    """Conditional edge after consensus_check: output or revise."""
    if state.get("consensus", False):
        return "output"
    return "revise"


# ---------------------------------------------------------------------------
# SubGraph construction
# ---------------------------------------------------------------------------


def build_adversarial_subgraph():
    """Build and compile the adversarial planning SubGraph.

    Flow: drafter → [reviewer_academic ∥ reviewer_emotional] →
          consensus_check → output | rewrite → drafter (loop)
    """
    builder = StateGraph(PlanAdversarialState)

    # Nodes
    builder.add_node("drafter", drafter_node)
    builder.add_node("reviewer_academic", reviewer_academic_node)
    builder.add_node("reviewer_emotional", reviewer_emotional_node)
    builder.add_node("consensus_check", consensus_check_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("output", output_node)

    # Entry
    builder.set_entry_point("drafter")

    # Drafter → parallel reviewers (fan-out)
    builder.add_edge("drafter", "reviewer_academic")
    builder.add_edge("drafter", "reviewer_emotional")

    # Reviewers → consensus_check (fan-in)
    builder.add_edge("reviewer_academic", "consensus_check")
    builder.add_edge("reviewer_emotional", "consensus_check")

    # Consensus → output or revise
    builder.add_conditional_edges(
        "consensus_check",
        _should_output_or_revise,
        {
            "output": "output",
            "revise": "rewrite",
        },
    )

    # Rewrite → drafter (loop back)
    builder.add_edge("rewrite", "drafter")

    # Output → END
    builder.add_edge("output", END)

    return builder.compile()
