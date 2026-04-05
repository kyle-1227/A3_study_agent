"""Adversarial Planning Nodes (flattened into parent graph — AC-01).

Drafter → [Reviewer Academic ∥ Reviewer Emotional] → Consensus Check → (loop/output)

The adversarial review loop: a drafter produces a study plan,
two parallel reviewers (academic quality + emotional wellbeing) evaluate it,
and a consensus check decides whether to accept or request revisions.
A safety valve (max_rounds from config) forces output after N iterations.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel

from src.config import get_setting, load_prompt
from src.graph.llm import async_invoke_with_fallback, get_fallback_llm, get_node_llm
from src.graph.state import TutorState
from src.tracing import traced_llm_call, traced_node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ReviewVerdict(BaseModel):
    verdict: Literal["approve", "reject"]
    reason: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _last_human_query(state: TutorState) -> str:
    """Extract the last HumanMessage content from state messages."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


@traced_node
async def drafter_node(state: TutorState) -> dict[str, Any]:
    """Draft or rewrite a study plan based on intel and any revision notes."""
    llm = get_node_llm("planner")
    temperature = get_setting("planner.temperature", 0.7)
    fallback = get_fallback_llm(temperature=temperature)

    user_request = _last_human_query(state)
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
        "adv_round": state.get("adv_round", 0) + 1,
    }


async def _run_reviewer(
    state: TutorState,
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
async def reviewer_academic_node(state: TutorState) -> dict[str, Any]:
    """Academic quality reviewer."""
    verdict = await _run_reviewer(
        state,
        system_prompt_name="plan_reviewer_academic_system",
        node_name="reviewer_academic",
    )
    return {"academic_verdict": verdict.verdict, "academic_reason": verdict.reason}


@traced_node
async def reviewer_emotional_node(state: TutorState) -> dict[str, Any]:
    """Emotional wellbeing reviewer."""
    verdict = await _run_reviewer(
        state,
        system_prompt_name="plan_reviewer_emotional_system",
        node_name="reviewer_emotional",
    )
    return {"emotional_verdict": verdict.verdict, "emotional_reason": verdict.reason}


@traced_node
async def consensus_check_node(state: TutorState) -> dict[str, Any]:
    """Check if both reviewers approved, or force output at max_rounds."""
    current_round = state.get("adv_round", 0)
    max_rounds = get_setting("planner.adversarial_max_rounds", 3)
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

    # Collect rejection reasons for revision (AC-03)
    notes_parts: list[str] = []
    if academic == "reject":
        reason = state.get("academic_reason", "未提供原因")
        notes_parts.append(f"[学术审查] {reason}")
    if emotional == "reject":
        reason = state.get("emotional_reason", "未提供原因")
        notes_parts.append(f"[情绪审查] {reason}")

    return {
        "consensus": False,
        "revision_notes": "; ".join(notes_parts) if notes_parts else "需要修改",
    }


@traced_node
async def adv_rewrite_node(state: TutorState) -> dict[str, Any]:
    """Reset verdicts before sending back to drafter for revision."""
    return {
        "academic_verdict": "",
        "academic_reason": "",
        "emotional_verdict": "",
        "emotional_reason": "",
    }


@traced_node
async def plan_output_node(state: TutorState) -> dict:
    """Final plan output — interrupt for HIL review if checkpointer available."""
    plan_text = state.get("draft", "")

    # HIL: pause for human review. Skip if running without checkpointer (stateless mode).
    try:
        edited_plan = interrupt(plan_text)
    except ValueError:
        # No checkpointer — skip HIL, use draft as-is
        logger.warning("interrupt() failed (no checkpointer?), skipping HIL review")
        edited_plan = plan_text

    final_plan = edited_plan if isinstance(edited_plan, str) and edited_plan else plan_text
    return {
        "plan": final_plan,
        "messages": [AIMessage(content=final_plan)],
    }


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------


def should_output_or_revise(state: TutorState) -> str:
    """Conditional edge after consensus_check: output or revise."""
    if state.get("consensus", False):
        return "output"
    return "revise"
