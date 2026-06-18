"""Study-plan resource-generation nodes."""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.config import get_setting
from src.graph.llm import invoke_plain_llm_fail_fast
from src.graph.state import LearningState
from src.llm.structured_output import (
    get_fallback_modes,
    get_llm_output_mode,
    get_max_raw_chars,
    invoke_structured_llm,
)
from src.observability.a3_trace import emit_a3_trace
from src.tools.document_tool import create_markdown_artifact
from src.tracing import traced_llm_call, traced_node

logger = logging.getLogger(__name__)


class StudyPlanEmotionalProfile(BaseModel):
    """Learner emotional and workload context for study-plan generation."""

    summary: str = Field(..., min_length=1)
    workload_risk: Literal["low", "medium", "high"]
    motivation_state: str = Field(..., min_length=1)
    support_suggestions: list[str] = Field(default_factory=list)


class StudyPlanPhase(BaseModel):
    """A phase in a personalized study plan."""

    title: str = Field(..., min_length=1)
    duration: str = Field(..., min_length=1)
    goals: list[str] = Field(..., min_length=1)
    tasks: list[str] = Field(..., min_length=1)
    resources: list[str] = Field(..., min_length=1)
    practice: list[str] = Field(..., min_length=1)
    checkpoints: list[str] = Field(..., min_length=1)


class StudyPlanArtifact(BaseModel):
    """Structured personalized study-plan artifact."""

    title: str = Field(..., min_length=1)
    learner_profile_summary: str = Field(..., min_length=1)
    overall_goal: str = Field(..., min_length=1)
    phases: list[StudyPlanPhase] = Field(..., min_length=2)
    weekly_schedule: list[str] = Field(..., min_length=1)
    milestones: list[str] = Field(..., min_length=1)
    practice_tasks: list[str] = Field(..., min_length=1)
    risk_warnings: list[str] = Field(default_factory=list)
    evidence_usage: list[str] = Field(..., min_length=1)


class StudyPlanReviewVerdict(BaseModel):
    """Structured study-plan reviewer verdict."""

    verdict: Literal["approve", "reject"]
    reason: str = Field(..., min_length=1)


def _last_human_query(state: LearningState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _model_to_dict(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _format_keypoints(state: LearningState) -> str:
    keypoints = state.get("keypoints", [])
    return ", ".join(str(item) for item in keypoints if str(item).strip()) or "No explicit keypoints extracted."


def _format_context(context: list[dict]) -> str:
    if not context:
        return "No judged evidence is available. Do not invent citations or course materials."
    parts: list[str] = []
    for idx, item in enumerate(context[:8], 1):
        source = item.get("source") or item.get("title") or item.get("url") or "learning material"
        content = str(item.get("content") or item.get("snippet") or item.get("text") or "")[:900]
        if content:
            parts.append(f"[{idx}] Source: {source}\n{content}")
    return "\n\n".join(parts) or "Judged evidence exists but has no readable body. Use only general learning-planning guidance."


def validate_emotional_profile(parsed: BaseModel) -> str:
    if not isinstance(parsed, StudyPlanEmotionalProfile):
        return "root expected StudyPlanEmotionalProfile"
    if not parsed.summary.strip():
        return "summary must be non-empty"
    return ""


def validate_study_plan_artifact(parsed: BaseModel) -> str:
    if not isinstance(parsed, StudyPlanArtifact):
        return "root expected StudyPlanArtifact"
    if not parsed.title.strip():
        return "title must be non-empty"
    if len(parsed.phases or []) < 2:
        return "phases must contain at least 2 items"
    for idx, phase in enumerate(parsed.phases or []):
        prefix = f"phases.{idx}"
        if not phase.duration.strip():
            return f"{prefix}.duration must be non-empty"
        if not phase.goals:
            return f"{prefix}.goals must be non-empty"
        if not phase.tasks:
            return f"{prefix}.tasks must be non-empty"
        if not phase.checkpoints:
            return f"{prefix}.checkpoints must be non-empty"
    if not parsed.weekly_schedule:
        return "weekly_schedule must be non-empty"
    if not parsed.evidence_usage:
        return "evidence_usage must be non-empty"
    return ""


def validate_study_plan_review(parsed: BaseModel) -> str:
    if not isinstance(parsed, StudyPlanReviewVerdict):
        return "root expected StudyPlanReviewVerdict"
    if parsed.verdict not in {"approve", "reject"}:
        return "verdict must be approve or reject"
    if not parsed.reason.strip():
        return "reason must be non-empty"
    return ""


def _render_artifact_markdown(artifact: dict) -> str:
    lines = [
        f"# {artifact.get('title') or 'Personalized Study Plan'}",
        "",
        "## Learner Profile",
        str(artifact.get("learner_profile_summary") or ""),
        "",
        "## Overall Goal",
        str(artifact.get("overall_goal") or ""),
        "",
        "## Phases",
    ]
    for idx, phase in enumerate(artifact.get("phases") or [], 1):
        lines.extend(
            [
                "",
                f"### {idx}. {phase.get('title') or 'Phase'}",
                f"- Duration: {phase.get('duration') or ''}",
                "- Goals:",
                *[f"  - {item}" for item in phase.get("goals") or []],
                "- Tasks:",
                *[f"  - {item}" for item in phase.get("tasks") or []],
                "- Resources:",
                *[f"  - {item}" for item in phase.get("resources") or []],
                "- Practice:",
                *[f"  - {item}" for item in phase.get("practice") or []],
                "- Checkpoints:",
                *[f"  - {item}" for item in phase.get("checkpoints") or []],
            ]
        )
    lines.extend(["", "## Weekly Schedule", *[f"- {item}" for item in artifact.get("weekly_schedule") or []]])
    lines.extend(["", "## Milestones", *[f"- {item}" for item in artifact.get("milestones") or []]])
    lines.extend(["", "## Practice Tasks", *[f"- {item}" for item in artifact.get("practice_tasks") or []]])
    risks = artifact.get("risk_warnings") or []
    if risks:
        lines.extend(["", "## Risk Warnings", *[f"- {item}" for item in risks]])
    lines.extend(["", "## Evidence Usage", *[f"- {item}" for item in artifact.get("evidence_usage") or []]])
    return "\n".join(lines).strip()


@traced_node
async def study_plan_emotional_intel(state: LearningState) -> dict:
    """Analyze learner workload and emotional context for study-plan generation."""
    query = _last_human_query(state)
    history = "\n".join(str(getattr(msg, "content", msg)) for msg in state.get("messages", [])[-8:])
    prompt = (
        "Analyze the learner's emotional state and workload risk for a personalized university study plan.\n"
        "Do not provide therapy or medical diagnosis. Focus on study burden, motivation, pacing, and support needs.\n\n"
        f"User query:\n{query}\n\nConversation excerpt:\n{history}"
    )
    model_name = get_setting("llm.study_plan.model", get_setting("study_plan.model", ""))
    with traced_llm_call(model_name=model_name, node_name="study_plan_emotional_intel", temperature=0.0):
        structured_result = await invoke_structured_llm(
            node_name="study_plan_emotional_intel",
            llm_node="study_plan",
            schema=StudyPlanEmotionalProfile,
            messages=[
                SystemMessage(content="You analyze learner workload for a study-plan agent. Return only JSON."),
                HumanMessage(content=prompt),
            ],
            output_mode=get_llm_output_mode("study_plan_emotional_intel"),
            fallback_modes=get_fallback_modes("study_plan_emotional_intel"),
            business_validator=validate_emotional_profile,
            state=state,
            max_raw_chars=get_max_raw_chars("study_plan_emotional_intel"),
        )
    result = structured_result.parsed
    if not isinstance(result, StudyPlanEmotionalProfile):
        raise TypeError("study_plan_emotional_intel parsed result is not StudyPlanEmotionalProfile")
    profile = _model_to_dict(result)
    emit_a3_trace(
        logger,
        "study_plan_emotional_intel",
        {"success": True, "workload_risk": profile.get("workload_risk"), "summary_chars": len(result.summary)},
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "study_plan_emotional_intel": result.summary,
        "study_plan_emotional_profile": profile,
    }


@traced_node
async def study_plan_planner(state: LearningState) -> dict:
    """Create a non-empty outline for the study-plan artifact."""
    query = _last_human_query(state)
    context = state.get("context", [])
    prompt = (
        "Create a concise outline for a personalized university learning plan.\n"
        "Use the judged evidence only as support. Do not invent course materials or citations.\n\n"
        f"User query:\n{query}\n\nKeypoints:\n{_format_keypoints(state)}\n\n"
        f"Emotional/workload intel:\n{state.get('study_plan_emotional_intel', '')}\n\n"
        f"Judged evidence:\n{_format_context(context)}"
    )
    outline = await invoke_plain_llm_fail_fast(
        node_name="study_plan_planner",
        llm_node="study_plan",
        messages=[
            SystemMessage(content="You plan personalized university study resources. Return a concrete outline only."),
            HumanMessage(content=prompt),
        ],
        state=state,
        temperature=get_setting("llm.study_plan.temperature", 0.2),
        max_raw_chars=get_setting("llm_outputs.study_plan_planner.max_raw_chars", 12000),
    )
    if not outline.strip():
        raise ValueError("study_plan_planner produced empty outline")
    return {
        "study_plan_outline": outline,
        "study_plan_artifact": {},
        "study_plan_markdown": "",
        "study_plan_round": 0,
        "study_plan_academic_verdict": "",
        "study_plan_academic_reason": "",
        "study_plan_emotional_verdict": "",
        "study_plan_emotional_reason": "",
        "study_plan_consensus": False,
        "study_plan_revision_notes": "",
        "study_plan_document_artifact": {},
    }


@traced_node
async def study_plan_agent(state: LearningState) -> dict:
    """Generate a structured study-plan artifact from the outline."""
    outline = state.get("study_plan_outline", "")
    if not outline.strip():
        raise ValueError("study_plan outline is empty")
    round_no = int(state.get("study_plan_round", 0) or 0) + 1
    prompt = (
        "Generate a personalized study plan as structured JSON.\n"
        "Avoid inventing unavailable resources. Use evidence_usage to explain how judged evidence informed the plan.\n\n"
        f"User query:\n{_last_human_query(state)}\n\nOutline:\n{outline}\n\n"
        f"Revision notes:\n{state.get('study_plan_revision_notes', '') or 'None'}\n\n"
        f"Emotional/workload intel:\n{state.get('study_plan_emotional_intel', '')}\n\n"
        f"Judged evidence:\n{_format_context(state.get('context', []))}"
    )
    model_name = get_setting("llm.study_plan.model", get_setting("study_plan.model", ""))
    with traced_llm_call(model_name=model_name, node_name="study_plan_agent", temperature=0.2):
        structured_result = await invoke_structured_llm(
            node_name="study_plan_agent",
            llm_node="study_plan",
            schema=StudyPlanArtifact,
            messages=[
                SystemMessage(content="You generate structured personalized study plans. Return only JSON."),
                HumanMessage(content=prompt),
            ],
            output_mode=get_llm_output_mode("study_plan_agent"),
            fallback_modes=get_fallback_modes("study_plan_agent"),
            business_validator=validate_study_plan_artifact,
            state=state,
            max_raw_chars=get_max_raw_chars("study_plan_agent"),
        )
    result = structured_result.parsed
    if not isinstance(result, StudyPlanArtifact):
        raise TypeError("study_plan_agent parsed result is not StudyPlanArtifact")
    return {
        "study_plan_artifact": _model_to_dict(result),
        "study_plan_round": round_no,
        "study_plan_academic_verdict": "",
        "study_plan_academic_reason": "",
        "study_plan_emotional_verdict": "",
        "study_plan_emotional_reason": "",
    }


async def _review_study_plan(state: LearningState, *, reviewer_kind: str) -> dict:
    artifact = state.get("study_plan_artifact") or {}
    if not artifact:
        raise ValueError("study_plan artifact is empty")
    focus = (
        "academic soundness, phase progression, evidence consistency, and avoiding fabricated resources"
        if reviewer_kind == "academic"
        else "workload, pacing, review/rest balance, and fit with emotional intel"
    )
    prompt = (
        f"Review this study plan for {focus}. Return approve or reject with a concise reason.\n\n"
        f"Plan:\n{artifact}\n\nEmotional intel:\n{state.get('study_plan_emotional_intel', '')}"
    )
    node_name = f"study_plan_reviewer_{reviewer_kind}"
    with traced_llm_call(model_name=get_setting("llm.study_plan.model", ""), node_name=node_name, temperature=0.0):
        structured_result = await invoke_structured_llm(
            node_name=node_name,
            llm_node="study_plan",
            schema=StudyPlanReviewVerdict,
            messages=[
                SystemMessage(content="You are a strict study-plan quality reviewer. Return only JSON."),
                HumanMessage(content=prompt),
            ],
            output_mode=get_llm_output_mode(node_name),
            fallback_modes=get_fallback_modes(node_name),
            business_validator=validate_study_plan_review,
            state=state,
            max_raw_chars=get_max_raw_chars(node_name),
        )
    result = structured_result.parsed
    if not isinstance(result, StudyPlanReviewVerdict):
        raise TypeError(f"{node_name} parsed result is not StudyPlanReviewVerdict")
    return result.model_dump() if hasattr(result, "model_dump") else result.dict()


@traced_node
async def study_plan_reviewer_academic(state: LearningState) -> dict:
    result = await _review_study_plan(state, reviewer_kind="academic")
    return {
        "study_plan_academic_verdict": result["verdict"],
        "study_plan_academic_reason": result["reason"],
    }


@traced_node
async def study_plan_reviewer_emotional(state: LearningState) -> dict:
    result = await _review_study_plan(state, reviewer_kind="emotional")
    return {
        "study_plan_emotional_verdict": result["verdict"],
        "study_plan_emotional_reason": result["reason"],
    }


@traced_node
async def study_plan_consensus(state: LearningState) -> dict:
    academic_ok = state.get("study_plan_academic_verdict") == "approve"
    emotional_ok = state.get("study_plan_emotional_verdict") == "approve"
    if academic_ok and emotional_ok:
        return {"study_plan_consensus": True, "study_plan_revision_notes": ""}
    max_rounds = int(get_setting("study_plan.max_generation_rounds", 3) or 3)
    current_round = int(state.get("study_plan_round", 0) or 0)
    notes = "\n".join(
        item
        for item in [
            state.get("study_plan_academic_reason", ""),
            state.get("study_plan_emotional_reason", ""),
        ]
        if item
    )
    if current_round >= max_rounds:
        raise RuntimeError(f"study_plan rejected after max rounds: {notes}")
    return {"study_plan_consensus": False, "study_plan_revision_notes": notes}


def route_after_study_plan_consensus(state: LearningState) -> str:
    return "output" if state.get("study_plan_consensus") else "rewrite"


@traced_node
async def study_plan_rewrite(state: LearningState) -> dict:
    notes = state.get("study_plan_revision_notes", "")
    if not notes.strip():
        raise ValueError("study_plan rewrite requested without revision notes")
    return {"study_plan_revision_notes": f"Revise the study plan according to reviewer feedback:\n{notes}"}


@traced_node
async def study_plan_output(state: LearningState) -> dict:
    artifact = state.get("study_plan_artifact") or {}
    if not artifact:
        raise ValueError("study_plan artifact is empty")
    markdown = _render_artifact_markdown(artifact)
    if not markdown.strip():
        raise ValueError("study_plan markdown is empty")
    document = create_markdown_artifact(markdown, str(artifact.get("title") or "Personalized Study Plan"))
    return {
        "study_plan_markdown": markdown,
        "study_plan_document_artifact": document,
        "messages": [AIMessage(content=markdown)],
    }
