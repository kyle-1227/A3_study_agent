"""Collaborative Markdown review-document resource-generation nodes."""

from __future__ import annotations

import logging
import os
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

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
from src.tools.document_tool import create_markdown_artifact
from src.tracing import traced_llm_call, traced_node

logger = logging.getLogger(__name__)


class ReviewDocReviewVerdict(BaseModel):
    """Structured quality gate output for review_doc_reviewer."""

    verdict: Literal["approve", "reject"]
    reason: str


def validate_review_doc_verdict(parsed: BaseModel) -> str:
    if not isinstance(parsed, ReviewDocReviewVerdict):
        return "root expected ReviewDocReviewVerdict"
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


def _format_keypoints(state: LearningState) -> str:
    keypoints = state.get("keypoints", [])
    return ", ".join(str(item) for item in keypoints if str(item).strip()) or "No explicit keypoints."


def _format_context(context: list[dict]) -> str:
    if not context:
        return "No judged evidence is available. Do not invent citations."
    parts: list[str] = []
    for idx, item in enumerate(context[:8], 1):
        source = item.get("source") or item.get("title") or item.get("url") or "learning material"
        content = str(item.get("content") or item.get("snippet") or item.get("text") or "")[:900]
        if content:
            parts.append(f"[{idx}] Source: {source}\n{content}")
    return "\n\n".join(parts) or "Judged evidence has no readable body."


def _render_prompt(prompt_name: str, replacements: dict[str, str]) -> str:
    prompt = load_prompt(prompt_name)
    for key, value in replacements.items():
        prompt = prompt.replace("{" + key + "}", value)
    return prompt


def _extract_markdown_title(markdown: str) -> str:
    for line in markdown.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()
    return "Markdown Review Document"


def _local_review_failure(markdown: str, _query: str) -> str:
    text = markdown.strip()
    if not text:
        return "review document markdown is empty"
    if not re.search(r"(?m)^#\s+\S+", text):
        return "review document is missing an H1 title"
    if len(re.findall(r"(?m)^##\s+", text)) < 3:
        return "review document needs at least three H2 sections"
    return ""


@traced_node
async def review_doc_planner(state: LearningState) -> dict:
    query = _last_human_query(state)
    context = state.get("context", [])
    emit_a3_trace(
        logger,
        "review_doc_planner",
        {
            "context_count": len(context),
            "dual_source_mode": bool(state.get("dual_source_mode")),
            "evidence_judge_state": state.get("evidence_judge_state", ""),
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    prompt = _render_prompt(
        "review_doc_planner",
        {"question": query, "keypoints": _format_keypoints(state), "context": _format_context(context)},
    )
    outline = await invoke_plain_llm_fail_fast(
        node_name="review_doc_planner",
        llm_node="review_doc",
        messages=[
            SystemMessage(content="You are a university course Markdown review-document planner. Return a concrete outline only."),
            HumanMessage(content=prompt),
        ],
        state=state,
        temperature=get_setting("review_doc.temperature", 0.2),
    )
    if not outline.strip():
        raise ValueError("review_doc_planner produced empty outline")
    return {
        "review_doc_outline": outline,
        "review_doc_markdown": "",
        "review_doc_artifact": {},
        "review_doc_review_verdict": "",
        "review_doc_review_reason": "",
        "review_doc_revision_notes": "",
        "review_doc_round": 0,
    }


@traced_node
async def review_doc_agent(state: LearningState) -> dict:
    outline = state.get("review_doc_outline", "")
    if not outline.strip():
        raise ValueError("review_doc outline is empty")
    round_no = int(state.get("review_doc_round", 0) or 0) + 1
    prompt = _render_prompt(
        "review_doc_agent",
        {
            "question": _last_human_query(state),
            "keypoints": _format_keypoints(state),
            "context": _format_context(state.get("context", [])),
            "review_doc_outline": outline,
            "revision_notes": state.get("review_doc_revision_notes", "") or "None",
        },
    )
    markdown = await invoke_plain_llm_fail_fast(
        node_name="review_doc_agent",
        llm_node="review_doc",
        messages=[
            SystemMessage(content="You are a university course Markdown review-document generator. Output Markdown only."),
            HumanMessage(content=prompt),
        ],
        state=state,
        temperature=get_setting("review_doc.temperature", 0.2),
    )
    if not markdown.strip():
        raise ValueError("review_doc_agent produced empty Markdown")
    emit_a3_trace(
        logger,
        "review_doc_agent",
        {
            "markdown_chars": len(markdown),
            "round": round_no,
            "fallback_used": False,
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "review_doc_markdown": markdown,
        "review_doc_round": round_no,
        "review_doc_artifact": {},
        "review_doc_review_verdict": "",
        "review_doc_review_reason": "",
    }


@traced_node
async def review_doc_reviewer(state: LearningState) -> dict:
    markdown = state.get("review_doc_markdown", "")
    local_failure = _local_review_failure(markdown, _last_human_query(state))
    if local_failure:
        emit_a3_trace(
            logger,
            "review_doc_reviewer",
            {
                "verdict": "reject",
                "reason": local_failure,
                "markdown_chars": len(markdown),
                "local_check_passed": False,
                "fallback_used": False,
            },
            state=state,
            env_flag="LOG_GENERATION_SUMMARY",
        )
        return {
            "review_doc_review_verdict": "reject",
            "review_doc_review_reason": local_failure,
            "review_doc_revision_notes": f"Please rewrite: {local_failure}",
        }

    prompt = _render_prompt(
        "review_doc_reviewer",
        {
            "question": _last_human_query(state),
            "review_doc_outline": state.get("review_doc_outline", ""),
            "review_doc_markdown": markdown,
        },
    )
    model_name = get_setting("llm.review_doc.model", os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    with traced_llm_call(model_name=model_name, node_name="review_doc_reviewer", temperature=0.0):
        structured_result = await invoke_structured_llm(
            node_name="review_doc_reviewer",
            llm_node="review_doc",
            schema=ReviewDocReviewVerdict,
            messages=[
                SystemMessage(content="You are a strict Markdown review-document quality reviewer. Return only JSON."),
                HumanMessage(content=prompt),
            ],
            output_mode=get_llm_output_mode("review_doc_reviewer"),
            fallback_modes=get_fallback_modes("review_doc_reviewer"),
            business_validator=validate_review_doc_verdict,
            state=state,
            max_raw_chars=get_max_raw_chars("review_doc_reviewer"),
        )
    result = structured_result.parsed
    if not isinstance(result, ReviewDocReviewVerdict):
        raise TypeError("review_doc_reviewer parsed result is not ReviewDocReviewVerdict")
    emit_a3_trace(
        logger,
        "review_doc_reviewer",
        {
            "verdict": result.verdict,
            "reason": result.reason,
            "markdown_chars": len(markdown),
            "local_check_passed": True,
            "fallback_used": False,
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "review_doc_review_verdict": result.verdict,
        "review_doc_review_reason": result.reason.strip(),
        "review_doc_revision_notes": "" if result.verdict == "approve" else f"Please rewrite: {result.reason.strip()}",
    }


@traced_node
async def review_doc_rewrite(state: LearningState) -> dict:
    reason = state.get("review_doc_review_reason", "")
    if not reason.strip():
        raise ValueError("review_doc rewrite requested without review reason")
    return {
        "review_doc_revision_notes": f"Revise the Markdown document according to reviewer feedback:\n{reason}",
        "review_doc_outline": state.get("review_doc_outline", ""),
    }


@traced_node
async def review_doc_output(state: LearningState) -> dict:
    markdown = state.get("review_doc_markdown", "")
    if not markdown.strip():
        raise ValueError("review_doc markdown is empty")
    title = _extract_markdown_title(markdown)
    artifact = create_markdown_artifact(markdown, title)
    emit_a3_trace(
        logger,
        "review_doc_output",
        {
            "markdown_chars": len(markdown),
            "fallback_used": False,
            "markdown_url": artifact.get("markdown_url", ""),
            "docx_url": artifact.get("docx_url", ""),
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "review_doc_artifact": {
            **artifact,
            "markdown": markdown,
            "quality_warning": False,
            "review_reason": state.get("review_doc_review_reason", ""),
        },
        "messages": [AIMessage(content=markdown)],
    }


def should_rewrite_review_doc(state: LearningState) -> str:
    if state.get("review_doc_review_verdict") != "reject":
        return "output"
    max_rounds = int(get_setting("review_doc.max_generation_rounds", 3) or 3)
    current_round = int(state.get("review_doc_round", 0) or 0)
    if current_round < max_rounds:
        return "rewrite"
    raise RuntimeError(f"review_doc rejected after max rounds: {state.get('review_doc_review_reason', '')}")
