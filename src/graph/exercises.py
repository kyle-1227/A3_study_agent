"""Collaborative leveled-exercise resource-generation nodes."""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.config import get_setting, load_prompt
from src.graph.llm import invoke_plain_llm_fail_fast
from src.graph.state import LearningState
from src.llm.structured_output import (
    get_fallback_modes,
    get_llm_output_mode,
    get_max_raw_chars,
    invoke_structured_llm,
)
from src.observability.a3_trace import emit_a3_trace
from src.tools.document_tool import create_document_artifact
from src.tracing import traced_llm_call, traced_node

logger = logging.getLogger(__name__)

REQUIRED_LEVELS = {"basic", "intermediate", "application", "self_check"}


class ExerciseItem(BaseModel):
    """A single exercise item with answer and teaching feedback."""

    level: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    explanation: str = Field(..., min_length=1)
    pitfall: str = Field(..., min_length=1)
    tags: list[str] = Field(..., min_length=1)


class ExerciseArtifact(BaseModel):
    """Structured exercise resource produced by exercise_agent."""

    title: str = Field(..., min_length=1)
    items: list[ExerciseItem] = Field(..., min_length=1)


class ExerciseReviewVerdict(BaseModel):
    """Structured quality gate output for exercise_reviewer."""

    verdict: Literal["approve", "reject"]
    reason: str


def validate_exercise_artifact(parsed: BaseModel) -> str:
    if not isinstance(parsed, ExerciseArtifact):
        return "root expected ExerciseArtifact"
    if len(parsed.items or []) < 4:
        return f"items expected at least 4, got {len(parsed.items or [])}"
    questions = [item.question.strip() for item in parsed.items or []]
    if len(questions) != len(set(questions)):
        return "duplicate questions detected"
    return ""


def validate_review_verdict(parsed: BaseModel) -> str:
    if not isinstance(parsed, ExerciseReviewVerdict):
        return "root expected ExerciseReviewVerdict"
    if parsed.verdict not in {"approve", "reject"}:
        return "verdict must be approve or reject"
    if not str(parsed.reason or "").strip():
        return "reason must be non-empty"
    return ""


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
    return ", ".join(str(item) for item in keypoints if str(item).strip()) or "No explicit keypoints."


def _format_context(context: list[dict]) -> str:
    if not context:
        return "No judged evidence is available. Do not invent citations."
    parts: list[str] = []
    for idx, item in enumerate(context[:8], 1):
        source = item.get("source") or item.get("title") or item.get("url") or "learning material"
        content = str(item.get("content") or item.get("snippet") or item.get("text") or "")[:800]
        if content:
            parts.append(f"[{idx}] Source: {source}\n{content}")
    return "\n\n".join(parts) or "Judged evidence has no readable body."


def _render_prompt(prompt_name: str, replacements: dict[str, str]) -> str:
    prompt = load_prompt(prompt_name)
    for key, value in replacements.items():
        prompt = prompt.replace("{" + key + "}", value)
    return prompt


def _is_web_evidence(item: dict) -> bool:
    return (
        item.get("source_type") == "web"
        or item.get("type") == "web_evidence"
    )


def _web_evidence_items(context: list[dict]) -> list[dict]:
    return [item for item in context if _is_web_evidence(item)]


def _normalize_items(items: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "level": str(item.get("level") or "").strip(),
                "question": str(item.get("question") or "").strip(),
                "answer": str(item.get("answer") or "").strip(),
                "explanation": str(item.get("explanation") or "").strip(),
                "pitfall": str(item.get("pitfall") or "").strip(),
                "tags": list(item.get("tags") or []),
            }
        )
    return normalized


def _local_review_failure(items: list[dict], _query: str) -> str:
    if len(items) < 4:
        return f"exercise item count is too low: {len(items)}"
    for idx, item in enumerate(items, 1):
        for field in ("level", "question", "answer", "explanation", "pitfall"):
            if not str(item.get(field) or "").strip():
                return f"item {idx} missing {field}"
    return ""


def _render_exercise_markdown(title: str, items: list[dict], *, review_reason: str = "", quality_warning: bool = False) -> str:
    lines = [f"## {title}", ""]
    if quality_warning:
        lines.extend([f"> Quality warning: {review_reason}", ""])
    for idx, item in enumerate(items, 1):
        tags = ", ".join(str(tag) for tag in item.get("tags", []) if tag)
        lines.extend(
            [
                f"### {idx}. {item.get('level', 'exercise')}",
                f"**Question:** {item.get('question', '')}",
                f"**Answer:** {item.get('answer', '')}",
                f"**Explanation:** {item.get('explanation', '')}",
                f"**Pitfall:** {item.get('pitfall', '')}",
            ]
        )
        if tags:
            lines.append(f"**Tags:** {tags}")
        lines.append("")
    return "\n".join(lines).strip()


@traced_node
async def exercise_planner(state: LearningState) -> dict:
    query = _last_human_query(state)
    context = state.get("context", [])
    web_evidence = _web_evidence_items(context)
    emit_a3_trace(
        logger,
        "exercise_planner",
        {
            "context_count": len(context),
            "context_web_count": len(web_evidence),
            "dual_source_mode": bool(state.get("dual_source_mode")),
            "evidence_judge_state": state.get("evidence_judge_state", ""),
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    prompt = _render_prompt(
        "exercise_planner",
        {"question": query, "keypoints": _format_keypoints(state), "context": _format_context(context)},
    )
    outline = await invoke_plain_llm_fail_fast(
        node_name="exercise_planner",
        llm_node="exercise",
        messages=[
            SystemMessage(content="You are a university course exercise planner. Return a concrete exercise outline only."),
            HumanMessage(content=prompt),
        ],
        state=state,
        temperature=get_setting("exercise.temperature", 0.2),
    )
    if not outline.strip():
        raise ValueError("exercise_planner produced empty outline")
    return {
        "exercise_outline": outline,
        "exercise_items": [],
        "exercise_artifact": {},
        "exercise_review_verdict": "",
        "exercise_review_reason": "",
        "exercise_revision_notes": "",
        "exercise_round": 0,
    }


@traced_node
async def exercise_agent(state: LearningState) -> dict:
    outline = state.get("exercise_outline", "")
    if not outline.strip():
        raise ValueError("exercise outline is empty")
    round_no = int(state.get("exercise_round", 0) or 0) + 1
    prompt = _render_prompt(
        "exercise_agent",
        {
            "question": _last_human_query(state),
            "keypoints": _format_keypoints(state),
            "context": _format_context(state.get("context", [])),
            "exercise_outline": outline,
            "revision_notes": state.get("exercise_revision_notes", "") or "None",
        },
    )
    model_name = get_setting("llm.exercise.model", get_setting("exercise.model", ""))
    with traced_llm_call(model_name=model_name, node_name="exercise_agent", temperature=get_setting("exercise.temperature", 0.2)):
        structured_result = await invoke_structured_llm(
            node_name="exercise_agent",
            llm_node="exercise",
            schema=ExerciseArtifact,
            messages=[
                SystemMessage(content="You are a leveled exercise generator. Return only JSON for ExerciseArtifact."),
                HumanMessage(content=prompt),
            ],
            output_mode=get_llm_output_mode("exercise_agent"),
            fallback_modes=get_fallback_modes("exercise_agent"),
            business_validator=validate_exercise_artifact,
            state=state,
            max_raw_chars=get_max_raw_chars("exercise_agent"),
        )
    result = structured_result.parsed
    if not isinstance(result, ExerciseArtifact):
        raise TypeError("exercise_agent parsed result is not ExerciseArtifact")
    return {
        "exercise_items": _normalize_items([_model_to_dict(item) for item in result.items]),
        "exercise_artifact": {"title": result.title.strip()},
        "exercise_round": round_no,
        "exercise_review_verdict": "",
        "exercise_review_reason": "",
    }


@traced_node
async def exercise_reviewer(state: LearningState) -> dict:
    items = state.get("exercise_items") or []
    local_failure = _local_review_failure(items, _last_human_query(state))
    if local_failure:
        return {
            "exercise_review_verdict": "reject",
            "exercise_review_reason": local_failure,
            "exercise_revision_notes": f"Please rewrite: {local_failure}",
        }
    prompt = _render_prompt(
        "exercise_reviewer",
        {
            "question": _last_human_query(state),
            "exercise_outline": state.get("exercise_outline", ""),
            "exercise_items": str(items),
        },
    )
    model_name = get_setting("llm.exercise.model", get_setting("exercise.model", ""))
    with traced_llm_call(model_name=model_name, node_name="exercise_reviewer", temperature=0.0):
        structured_result = await invoke_structured_llm(
            node_name="exercise_reviewer",
            llm_node="exercise",
            schema=ExerciseReviewVerdict,
            messages=[
                SystemMessage(content="You are a course exercise quality reviewer. Return only JSON."),
                HumanMessage(content=prompt),
            ],
            output_mode=get_llm_output_mode("exercise_reviewer"),
            fallback_modes=get_fallback_modes("exercise_reviewer"),
            business_validator=validate_review_verdict,
            state=state,
            max_raw_chars=get_max_raw_chars("exercise_reviewer"),
        )
    result = structured_result.parsed
    if not isinstance(result, ExerciseReviewVerdict):
        raise TypeError("exercise_reviewer parsed result is not ExerciseReviewVerdict")
    return {
        "exercise_review_verdict": result.verdict,
        "exercise_review_reason": result.reason.strip(),
        "exercise_revision_notes": "" if result.verdict == "approve" else f"Please rewrite: {result.reason.strip()}",
    }


@traced_node
async def exercise_rewrite(state: LearningState) -> dict:
    reason = state.get("exercise_review_reason", "")
    if not reason.strip():
        raise ValueError("exercise rewrite requested without review reason")
    return {
        "exercise_revision_notes": f"Revise the exercise artifact according to reviewer feedback:\n{reason}",
        "exercise_outline": state.get("exercise_outline", ""),
    }


@traced_node
async def exercise_output(state: LearningState) -> dict:
    items = state.get("exercise_items") or []
    if not items:
        raise ValueError("exercise items are empty")
    title = str((state.get("exercise_artifact") or {}).get("title") or "Leveled exercises")
    content = _render_exercise_markdown(
        title,
        items,
        review_reason=str(state.get("exercise_review_reason") or ""),
        quality_warning=False,
    )
    artifact: dict = {}
    try:
        artifact = create_document_artifact(
            markdown_text=content,
            title=title,
            artifact_kind="exercises",
        )
    except Exception:
        logger.exception("exercise_output document artifact generation failed")

    emit_a3_trace(
        logger,
        "exercise_output",
        {
            "item_count": len(items),
            "markdown_chars": len(content),
            "markdown_url": artifact.get("markdown_url", ""),
            "docx_url": artifact.get("docx_url", ""),
            "has_document_artifact": bool(artifact),
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )

    return {
        "exercise_artifact": {
            **artifact,
            "title": title,
            "items": items,
            "quality_warning": False,
            "review_reason": state.get("exercise_review_reason", ""),
        },
        "messages": [AIMessage(content=content)],
    }


def should_rewrite_exercise(state: LearningState) -> str:
    if state.get("exercise_review_verdict") != "reject":
        return "output"
    max_rounds = int(get_setting("exercise.max_generation_rounds", 3) or 3)
    current_round = int(state.get("exercise_round", 0) or 0)
    if current_round < max_rounds:
        return "rewrite"
    raise RuntimeError(f"exercise rejected after max rounds: {state.get('exercise_review_reason', '')}")
