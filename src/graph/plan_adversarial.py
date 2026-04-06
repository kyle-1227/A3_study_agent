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


class FeedbackClassification(BaseModel):
    """Classify user feedback on a study plan."""
    route: Literal["tweak", "rewrite"]
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
    structured_primary = llm.with_structured_output(ReviewVerdict, method="json_mode")

    fallback_llm = get_fallback_llm(temperature=reviewer_temp)
    structured_fallback = fallback_llm.with_structured_output(ReviewVerdict, method="json_mode")

    review_prompt = (
        f"## 学习计划\n\n{state.get('draft', '')}\n\n"
        f"## 学生情况\n\n{state.get('intel_summary', '')}\n\n"
        f"请以 json 格式返回你的审查结论。"
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
        user_response = interrupt(plan_text)
    except ValueError:
        # No checkpointer — skip HIL, use draft as-is
        logger.warning("interrupt() failed (no checkpointer?), skipping HIL review")
        user_response = plan_text

    # ── User provided feedback (dict) → route to feedback_router ──
    if isinstance(user_response, dict) and user_response.get("action") == "feedback":
        return {
            "hil_action": "feedback",
            "hil_feedback": user_response.get("text", ""),
        }

    # ── User confirmed (string) → finalize ──
    final_plan = user_response if isinstance(user_response, str) and user_response else plan_text
    return {
        "plan": final_plan,
        "messages": [AIMessage(content=final_plan)],
        "hil_action": "confirm",
    }


@traced_node
async def feedback_router(state: TutorState) -> dict[str, Any]:
    """Classify user's HIL feedback as 'tweak' (minor edit) or 'rewrite' (full redo).

    Uses the supervisor's fast model for quick classification.
    Also updates hil_summary: compresses old summary + new feedback into one string.
    """
    llm = get_node_llm("supervisor")
    structured_llm = llm.with_structured_output(FeedbackClassification, method="json_mode")

    feedback = state.get("hil_feedback", "")
    draft = state.get("draft", "")
    old_summary = state.get("hil_summary", "")

    # ── Step 1: Classify feedback ──
    classify_prompt = (
        f"学生对以下学习计划提出了修改意见。\n\n"
        f"## 当前计划（前500字）\n{draft[:500]}\n\n"
        f"## 学生反馈\n{feedback}\n\n"
        f"判断这个反馈需要的修改程度：\n"
        f"- tweak: 只需要局部微调（如调整某天科目、修改时间、增删某个小项）\n"
        f"- rewrite: 需要重新规划（如整体思路不对、完全不符合需求、需要换方向）\n\n"
        f"请以 json 格式返回你的分类结果。"
    )

    try:
        result = await structured_llm.ainvoke([
            SystemMessage(content="你是一个学习计划修改分类器。根据学生反馈判断需要微调还是重写。"),
            HumanMessage(content=classify_prompt),
        ])
        route = result.route
    except Exception:
        logger.warning("Feedback classification failed, defaulting to tweak")
        route = "tweak"

    # ── Step 2: Compress summary (overwrite, not append) ──
    if old_summary:
        new_summary = f"历史修改摘要: {old_summary[:200]}\n最新反馈: {feedback[:500]}"
    else:
        new_summary = f"用户反馈: {feedback[:500]}"

    if route == "rewrite":
        # REWRITE: Clear adversarial state, treat feedback as fresh direction
        return {
            "feedback_route": "rewrite",
            "hil_summary": new_summary,
            "revision_notes": feedback,
            "adv_round": 0,
            "draft": "",
            "academic_verdict": "",
            "academic_reason": "",
            "emotional_verdict": "",
            "emotional_reason": "",
            "consensus": False,
        }
    else:
        # TWEAK: Keep draft intact, pass feedback to plan_tweak
        return {
            "feedback_route": "tweak",
            "hil_summary": new_summary,
        }


@traced_node
async def plan_tweak_node(state: TutorState) -> dict[str, Any]:
    """Make a targeted edit to the plan based on user feedback.

    Single LLM call — no reviewer loop needed for minor edits.
    """
    llm = get_node_llm("planner")
    temperature = get_setting("planner.temperature", 0.7)
    fallback = get_fallback_llm(temperature=temperature)

    draft = state.get("draft", "")
    feedback = state.get("hil_feedback", "")
    summary = state.get("hil_summary", "")

    prompt = (
        f"请根据学生的反馈对以下学习计划进行**局部微调**。\n"
        f"只修改学生提到的部分，保持其他内容不变。\n\n"
        f"## 当前计划\n{draft}\n\n"
        f"## 学生反馈\n{feedback}\n\n"
    )
    if summary:
        prompt += f"## 修改历史摘要\n{summary}\n\n"
    prompt += "请输出修改后的完整计划："

    messages = [
        SystemMessage(content=load_prompt("plan_drafter_system")),
        HumanMessage(content=prompt),
    ]

    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        node_name="plan_tweak",
        temperature=temperature,
    ) as span:
        response = await async_invoke_with_fallback(
            llm, messages, fallback=fallback, span=span,
        )

    return {"draft": response.content}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def should_output_or_revise(state: TutorState) -> str:
    """Conditional edge after consensus_check: output or revise."""
    if state.get("consensus", False):
        return "output"
    return "revise"


def route_after_hil(state: TutorState) -> str:
    """Conditional edge after plan_output: confirm → end, feedback → feedback_router."""
    return "feedback" if state.get("hil_action") == "feedback" else "end"


def route_feedback(state: TutorState) -> str:
    """Conditional edge after feedback_router: tweak or rewrite."""
    return state.get("feedback_route", "tweak")
