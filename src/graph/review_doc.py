"""Collaborative Markdown review-document resource-generation nodes."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from src.config import get_setting, load_prompt
from src.graph.llm import invoke_plain_llm_fail_fast
from src.graph.state import LearningState
from src.llm.structured_output import (
    StructuredOutputError,
    get_llm_output_mode,
    get_max_raw_chars,
    invoke_structured_llm,
)
from src.observability.a3_trace import emit_a3_trace
from src.tools.document_tool import create_markdown_artifact
from src.tracing import traced_llm_call, traced_node

logger = logging.getLogger(__name__)


REQUIRED_REVIEW_DOC_SECTIONS = {
    "title": r"(?m)^#\s+\S+",
    "goal": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?复习目标",
    "core": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?核心知识点",
    "pitfall": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?易错点",
    "self_check": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?自测清单",
}

REVIEW_DOC_SUBJECT_TITLES = {
    "python": "Python",
    "computer": "计算机科学导论",
    "big_data": "大数据",
}


class ReviewDocReviewVerdict(BaseModel):
    """Structured quality gate output for review_doc_reviewer."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    verdict: Literal["approve", "reject"]
    reason: str = Field(min_length=1)


class ReviewDocGenerationError(RuntimeError):
    """Raised when no provider-produced review document can be accepted."""


class ReviewDocApprovalError(RuntimeError):
    """Raised when an unapproved review document reaches artifact output."""


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
    return (
        ", ".join(str(item) for item in keypoints if str(item).strip())
        or "No explicit keypoints."
    )


def _format_context(context: list[dict]) -> str:
    if not context:
        return "No judged evidence is available. Do not invent citations."
    parts: list[str] = []
    for idx, item in enumerate(context[:8], 1):
        source = (
            item.get("source")
            or item.get("title")
            or item.get("url")
            or "learning material"
        )
        content = str(
            item.get("content") or item.get("snippet") or item.get("text") or ""
        )[:900]
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
            title = match.group(1).strip()
            if title:
                return title
            break
    raise ReviewDocGenerationError("review document Markdown title is missing")


def _local_review_failure(markdown: str, _query: str) -> str:
    text = markdown.strip()
    if not text:
        return "review document markdown is empty"

    missing: list[str] = []
    section_names = {
        "title": "标题",
        "goal": "复习目标",
        "core": "核心知识点",
        "pitfall": "易错点",
        "self_check": "自测清单",
    }
    for key, pattern in REQUIRED_REVIEW_DOC_SECTIONS.items():
        if not re.search(pattern, text):
            missing.append(section_names[key])

    if missing:
        return f"Markdown missing required sections: {', '.join(missing)}"

    return ""


def _review_doc_bundle_local_failure(state: LearningState, markdown: str) -> str:
    documents = state.get("review_doc_markdowns") or []
    if not documents:
        return _local_review_failure(markdown, _last_human_query(state))
    if not isinstance(documents, list):
        return "review_doc_markdowns must be a list"

    seen_subjects: set[str] = set()
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            return f"review document {index} must be an object"
        subject = str(document.get("subject") or "").strip()
        title = str(document.get("title") or "").strip()
        document_markdown = str(document.get("markdown") or "").strip()
        if not subject:
            return f"review document {index} subject is empty"
        if subject in seen_subjects:
            return f"duplicate review document subject: {subject}"
        seen_subjects.add(subject)
        expected_title = _review_doc_title_for_subject(subject)
        if title != expected_title:
            return (
                f"review document {index} title mismatch: "
                f"expected={expected_title!r}, actual={title!r}"
            )
        try:
            _require_markdown_title(document_markdown, expected_title)
        except ReviewDocGenerationError as exc:
            return str(exc)
        local_failure = _local_review_failure(
            document_markdown, _last_human_query(state)
        )
        if local_failure:
            return f"review document {index}: {local_failure}"
    return ""


def _review_doc_length_instruction() -> str:
    return (
        "\n\n## Length and stability requirements\n"
        "1. Keep the first version within 1800-2500 Chinese characters unless the user explicitly asks for a long document.\n"
        "2. Generate a medium-length review document by default.\n"
        "3. Prefer complete structure, clear concepts, and concise tables over excessive length.\n"
    )


def _subject_display_name(subject: str) -> str:
    subject_key = str(subject or "").strip()
    if not subject_key:
        return "课程"
    if subject_key in REVIEW_DOC_SUBJECT_TITLES:
        return REVIEW_DOC_SUBJECT_TITLES[subject_key]
    return subject_key.replace("_", " ").strip().title()


def _review_doc_title_for_subject(subject: str) -> str:
    return f"{_subject_display_name(subject)} 复习资料"


def _retrieval_subjects(state: LearningState) -> list[str]:
    subjects: list[str] = []
    for item in state.get("retrieval_plan") or []:
        subject = str((item or {}).get("subject") or "").strip()
        if subject and subject != "other" and subject not in subjects:
            subjects.append(subject)
    return subjects


def _doc_subject_values(item: dict) -> set[str]:
    metadata = item.get("metadata") or {}
    values = {
        item.get("retrieval_subject"),
        item.get("subject"),
        metadata.get("subject") if isinstance(metadata, dict) else "",
    }
    return {str(value).strip() for value in values if str(value or "").strip()}


def _context_for_subject(context: list[dict], subject: str) -> list[dict]:
    return [item for item in context if subject in _doc_subject_values(item)]


def _require_markdown_title(markdown: str, expected_title: str = "") -> str:
    text = markdown.strip()
    if not text:
        raise ReviewDocGenerationError("review document Markdown is empty")
    actual_title = _extract_markdown_title(text)
    if expected_title and actual_title != expected_title:
        raise ReviewDocGenerationError(
            "review document title mismatch: "
            f"expected={expected_title!r}, actual={actual_title!r}"
        )
    return text


def _review_doc_model_name() -> str:
    configured_model = get_setting("llm.review_doc.model", None)
    if not isinstance(configured_model, str) or not configured_model.strip():
        raise ValueError("llm.review_doc.model must be explicitly configured")
    return configured_model.strip()


def _review_doc_temperature() -> float:
    configured_temperature = get_setting("llm.review_doc.temperature", None)
    if isinstance(configured_temperature, bool) or not isinstance(
        configured_temperature, (int, float)
    ):
        raise ValueError("llm.review_doc.temperature must be explicitly configured")
    temperature = float(configured_temperature)
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("llm.review_doc.temperature must be between 0 and 2")
    return temperature


def _review_doc_timeout_seconds() -> float:
    configured_timeout = get_setting("llm.review_doc.timeout_seconds", None)
    if isinstance(configured_timeout, bool) or not isinstance(
        configured_timeout, (int, float)
    ):
        raise ValueError("llm.review_doc.timeout_seconds must be explicitly configured")
    timeout_seconds = float(configured_timeout)
    if timeout_seconds <= 0:
        raise ValueError("llm.review_doc.timeout_seconds must be greater than zero")
    return timeout_seconds


def _review_doc_max_generation_rounds() -> int:
    configured_rounds = get_setting("llm.review_doc.max_generation_rounds", None)
    if isinstance(configured_rounds, bool) or not isinstance(configured_rounds, int):
        raise ValueError(
            "llm.review_doc.max_generation_rounds must be explicitly configured"
        )
    if configured_rounds < 1:
        raise ValueError("llm.review_doc.max_generation_rounds must be at least one")
    return configured_rounds


async def _generate_review_doc_markdown(
    *,
    state: LearningState,
    query: str,
    outline: str,
    revision_notes: str,
    context: list[dict],
    round_no: int,
    subject: str = "",
    title: str = "",
) -> dict:
    temperature = _review_doc_temperature()
    timeout_seconds = _review_doc_timeout_seconds()
    model_name = _review_doc_model_name()

    subject_note = ""
    if subject and title:
        subject_note = (
            f"\n\nOnly generate the independent document titled '{title}'. "
            f"Focus on subject={subject}; do not merge other subjects into this document."
        )
    prompt = _render_prompt(
        "review_doc_agent",
        {
            "question": f"{query}{subject_note}",
            "keypoints": _format_keypoints(state),
            "context": _format_context(context),
            "review_doc_outline": outline,
            "revision_notes": revision_notes or "None",
        },
    )
    prompt += _review_doc_length_instruction()

    messages = [
        SystemMessage(
            content="You are a university course Markdown review-document generator. Output Markdown only."
        ),
        HumanMessage(content=prompt),
    ]

    try:
        with traced_llm_call(
            model_name=model_name, node_name="review_doc_agent", temperature=temperature
        ):
            markdown = await asyncio.wait_for(
                invoke_plain_llm_fail_fast(
                    node_name="review_doc_agent",
                    llm_node="review_doc",
                    messages=messages,
                    state=state,
                    temperature=temperature,
                ),
                timeout=timeout_seconds,
            )
        markdown = str(markdown or "").strip()
    except Exception as exc:
        emit_a3_trace(
            logger,
            "review_doc_agent_failed",
            {
                "review_doc_agent_error_type": type(exc).__name__,
                "error_message": str(exc)[:300],
                "timeout_seconds": timeout_seconds,
                "subject": subject,
            },
            state=state,
            env_flag="LOG_GENERATION_SUMMARY",
        )
        raise

    if not markdown:
        error_message = "LLM returned empty Markdown content"
        emit_a3_trace(
            logger,
            "review_doc_agent_failed",
            {
                "review_doc_agent_error_type": "EmptyResponse",
                "error_message": error_message,
                "timeout_seconds": timeout_seconds,
                "subject": subject,
            },
            state=state,
            env_flag="LOG_GENERATION_SUMMARY",
        )
        raise ReviewDocGenerationError(error_message)

    markdown = _require_markdown_title(markdown, title)

    return {
        "markdown": markdown,
        "round": round_no,
        "timeout_seconds": timeout_seconds,
    }


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
        {
            "question": query,
            "keypoints": _format_keypoints(state),
            "context": _format_context(context),
        },
    )
    outline = await invoke_plain_llm_fail_fast(
        node_name="review_doc_planner",
        llm_node="review_doc",
        messages=[
            SystemMessage(
                content="You are a university course Markdown review-document planner. Return a concrete outline only."
            ),
            HumanMessage(content=prompt),
        ],
        state=state,
        temperature=_review_doc_temperature(),
    )
    if not outline.strip():
        raise ValueError("review_doc_planner produced empty outline")
    return {
        "review_doc_outline": outline,
        "review_doc_markdown": "",
        "review_doc_markdowns": [],
        "review_doc_artifact": {},
        "review_doc_artifacts": [],
        "review_doc_review_verdict": "",
        "review_doc_review_reason": "",
        "review_doc_revision_notes": "",
        "review_doc_round": 0,
    }


@traced_node
async def review_doc_agent(state: LearningState) -> dict:
    query = _last_human_query(state)
    context = state.get("context", [])
    revision_notes = state.get("review_doc_revision_notes", "")
    outline = state.get("review_doc_outline", "")
    if not outline.strip():
        raise ValueError("review_doc outline is empty")
    round_no = int(state.get("review_doc_round", 0) or 0) + 1

    subjects = _retrieval_subjects(state)
    multi_document = len(subjects) > 1
    if multi_document:
        documents: list[dict] = []
        timeout_seconds = _review_doc_timeout_seconds()

        for subject in subjects:
            title = _review_doc_title_for_subject(subject)
            subject_context = _context_for_subject(context, subject)
            subject_outline = (
                f"# {title}\n\n{outline}\n\n"
                f"Generate this as an independent review document for subject={subject} only."
            )
            result = await _generate_review_doc_markdown(
                state=state,
                query=query,
                outline=subject_outline,
                revision_notes=revision_notes,
                context=subject_context,
                round_no=round_no,
                subject=subject,
                title=title,
            )
            markdown = _require_markdown_title(str(result.get("markdown") or ""), title)
            documents.append({"subject": subject, "title": title, "markdown": markdown})

        combined_markdown = "\n\n---\n\n".join(doc["markdown"] for doc in documents)
        emit_a3_trace(
            logger,
            "review_doc_agent",
            {
                "multi_document": True,
                "document_count": len(documents),
                "subjects": subjects,
                "markdown_chars": len(combined_markdown),
                "round": round_no,
                "context_count": len(context),
                "timeout_seconds": timeout_seconds,
            },
            state=state,
            env_flag="LOG_GENERATION_SUMMARY",
        )
        return {
            "review_doc_markdown": combined_markdown,
            "review_doc_markdowns": documents,
            "review_doc_round": round_no,
            "review_doc_artifact": {
                "multi_document": True,
                "document_count": len(documents),
                "subjects": subjects,
            },
            "review_doc_artifacts": [],
            "review_doc_review_verdict": "",
            "review_doc_review_reason": "",
        }

    result = await _generate_review_doc_markdown(
        state=state,
        query=query,
        outline=outline,
        revision_notes=revision_notes,
        context=context,
        round_no=round_no,
    )
    markdown = _require_markdown_title(str(result.get("markdown") or ""))
    emit_a3_trace(
        logger,
        "review_doc_agent",
        {
            "multi_document": False,
            "document_count": 1 if markdown else 0,
            "subjects": subjects,
            "markdown_chars": len(markdown),
            "round": round_no,
            "context_count": len(context),
            "timeout_seconds": result.get("timeout_seconds", ""),
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "review_doc_markdown": markdown,
        "review_doc_markdowns": [],
        "review_doc_round": round_no,
        "review_doc_artifact": {},
        "review_doc_artifacts": [],
        "review_doc_review_verdict": "",
        "review_doc_review_reason": "",
    }


@traced_node
async def review_doc_reviewer(state: LearningState) -> dict:
    markdown = state.get("review_doc_markdown", "")
    local_failure = _review_doc_bundle_local_failure(state, markdown)
    if local_failure:
        emit_a3_trace(
            logger,
            "review_doc_reviewer",
            {
                "verdict": "reject",
                "reason": local_failure,
                "markdown_chars": len(markdown),
                "local_check_passed": False,
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
    model_name = _review_doc_model_name()
    with traced_llm_call(
        model_name=model_name, node_name="review_doc_reviewer", temperature=0.0
    ):
        structured_result = await invoke_structured_llm(
            node_name="review_doc_reviewer",
            llm_node="review_doc",
            schema=ReviewDocReviewVerdict,
            messages=[
                SystemMessage(
                    content="You are a strict Markdown review-document quality reviewer. Return only JSON."
                ),
                HumanMessage(content=prompt),
            ],
            output_mode=get_llm_output_mode("review_doc_reviewer"),
            business_validator=validate_review_doc_verdict,
            state=state,
            max_raw_chars=get_max_raw_chars("review_doc_reviewer"),
        )
    if not structured_result.success:
        raise StructuredOutputError(structured_result)
    result = structured_result.parsed
    if not isinstance(result, ReviewDocReviewVerdict):
        raise TypeError(
            "review_doc_reviewer parsed result is not ReviewDocReviewVerdict"
        )
    emit_a3_trace(
        logger,
        "review_doc_reviewer",
        {
            "verdict": result.verdict,
            "reason": result.reason,
            "markdown_chars": len(markdown),
            "local_check_passed": True,
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "review_doc_review_verdict": result.verdict,
        "review_doc_review_reason": result.reason.strip(),
        "review_doc_revision_notes": ""
        if result.verdict == "approve"
        else f"Please rewrite: {result.reason.strip()}",
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
        raise ReviewDocApprovalError("review_doc markdown is empty")
    review_verdict = str(state.get("review_doc_review_verdict") or "").strip()
    review_reason = str(state.get("review_doc_review_reason") or "").strip()
    if review_verdict != "approve":
        detail = f": {review_reason}" if review_reason else ""
        raise ReviewDocApprovalError(
            f"review_doc output requires an approve verdict{detail}"
        )
    local_failure = _review_doc_bundle_local_failure(state, markdown)
    if local_failure:
        raise ReviewDocApprovalError(
            f"review_doc output failed local quality check: {local_failure}"
        )
    review_doc_markdowns = state.get("review_doc_markdowns") or []
    if review_doc_markdowns:
        review_doc_artifacts: list[dict] = []
        for doc in review_doc_markdowns:
            doc_markdown = str(doc.get("markdown") or "").strip()
            doc_title = str(doc.get("title") or "").strip()
            artifact = create_markdown_artifact(doc_markdown, doc_title)
            review_doc_artifacts.append(
                {
                    **artifact,
                    "subject": str(doc.get("subject") or ""),
                    "title": doc_title,
                    "markdown": doc_markdown,
                    "review_reason": review_reason,
                }
            )

        if len(review_doc_artifacts) != len(review_doc_markdowns):
            raise ReviewDocApprovalError(
                "review_doc artifact count does not match document count"
            )
        first_artifact = review_doc_artifacts[0]
        combined_markdown = "\n\n---\n\n".join(
            artifact["markdown"] for artifact in review_doc_artifacts
        )
        subjects = [
            artifact.get("subject", "")
            for artifact in review_doc_artifacts
            if artifact.get("subject")
        ]
        emit_a3_trace(
            logger,
            "review_doc_output",
            {
                "multi_document": True,
                "document_count": len(review_doc_markdowns),
                "artifact_count": len(review_doc_artifacts),
                "subjects": subjects,
                "markdown_chars": len(combined_markdown),
                "review_reason": review_reason,
                "emits_ai_message": True,
            },
            state=state,
            env_flag="LOG_GENERATION_SUMMARY",
        )
        return {
            "review_doc_markdown": combined_markdown,
            "review_doc_artifact": {
                **first_artifact,
                "multi_document": True,
                "document_count": len(review_doc_artifacts),
                "subjects": subjects,
            },
            "review_doc_artifacts": review_doc_artifacts,
            "messages": [AIMessage(content=combined_markdown)],
        }

    title = _extract_markdown_title(markdown)
    artifact = create_markdown_artifact(markdown, title)
    emit_a3_trace(
        logger,
        "review_doc_output",
        {
            "multi_document": False,
            "document_count": 1,
            "artifact_count": 1,
            "markdown_chars": len(markdown),
            "review_reason": review_reason,
            "emits_ai_message": True,
            "markdown_url": artifact.get("markdown_url", ""),
            "docx_url": artifact.get("docx_url", ""),
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    final_artifact = {
        **artifact,
        "markdown": markdown,
        "review_reason": review_reason,
    }
    return {
        "review_doc_artifact": final_artifact,
        "review_doc_artifacts": [final_artifact],
        "messages": [AIMessage(content=markdown)],
    }


def should_rewrite_review_doc(state: LearningState) -> str:
    verdict = str(state.get("review_doc_review_verdict") or "").strip()
    if verdict == "approve":
        return "output"
    if verdict != "reject":
        raise ReviewDocApprovalError(
            "review_doc routing requires an explicit approve or reject verdict"
        )
    max_rounds = _review_doc_max_generation_rounds()
    current_round = int(state.get("review_doc_round", 0) or 0)
    if current_round < max_rounds:
        return "rewrite"
    raise ReviewDocApprovalError(
        "review_doc rejected after max rounds: "
        f"{state.get('review_doc_review_reason', '')}"
    )
