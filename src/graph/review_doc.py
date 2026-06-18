"""Collaborative Markdown review-document resource-generation nodes."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Literal

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from src.config import get_setting, load_prompt
from src.graph.llm import async_invoke_with_fallback, get_fallback_llm, get_node_llm, invoke_plain_llm_fail_fast
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


REQUIRED_REVIEW_DOC_SECTIONS = {
    "title": r"(?m)^#\s+\S+",
    "goal": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?复习目标",
    "core": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?核心知识点",
    "pitfall": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?易错点",
    "self_check": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?自测清单",
}

REVIEW_DOC_AGENT_RETRY_ERRORS = (
    TimeoutError,
    asyncio.TimeoutError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.TimeoutException,
)

REVIEW_DOC_SUBJECT_TITLES = {
    "python": "Python",
    "computer": "计算机科学导论",
    "big_data": "大数据",
}


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
    return "Markdown复习文档"


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


def _review_doc_error_type(exc: BaseException) -> str:
    return type(exc).__name__


def _is_retriable_review_doc_error(exc: BaseException) -> bool:
    if isinstance(exc, REVIEW_DOC_AGENT_RETRY_ERRORS):
        return True
    message = f"{type(exc).__name__}: {exc}".lower()
    retry_markers = (
        "incomplete chunked read",
        "peer closed connection",
        "remoteprotocolerror",
        "readerror",
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
    )
    if any(marker in message for marker in retry_markers):
        return True
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    return bool(cause and cause is not exc and _is_retriable_review_doc_error(cause))


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
    filtered = [item for item in context if subject in _doc_subject_values(item)]
    return filtered or context


def _ensure_markdown_title(markdown: str, title: str) -> str:
    text = markdown.strip()
    if not text:
        return f"# {title}\n"
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        if line.lstrip().startswith("# "):
            lines[idx] = f"# {title}"
            return "\n".join(lines).strip()
        return f"# {title}\n\n{text}"
    return f"# {title}\n"


def _extract_outline_items(outline: str) -> list[str]:
    items: list[str] = []
    for line in outline.splitlines():
        cleaned = re.sub(r"^[\s#\-*0-9一二三四五六七八九十、.．]+", "", line).strip()
        cleaned = re.sub(r"[:：].*$", "", cleaned).strip()
        if 2 <= len(cleaned) <= 30 and cleaned not in items:
            items.append(cleaned)
    return items


def _fallback_review_doc_markdown(
    *,
    title: str,
    keypoints: list[str],
    outline: str,
    context: list[dict],
    reason: str,
) -> str:
    keypoint_items = [str(item).strip() for item in keypoints if str(item).strip()]
    if not keypoint_items:
        keypoint_items = _extract_outline_items(outline)[:6]
    if not keypoint_items:
        keypoint_items = [title, "核心概念", "常见应用", "易错点"]

    overview_lines = "\n".join(f"- {item}" for item in keypoint_items[:8])
    concept_lines = "\n".join(f"- **{item}**：结合课程资料重点理解其定义、用途和适用边界。" for item in keypoint_items[:6])
    self_check_lines = "\n".join(f"- [ ] 我能说明“{item}”的含义和典型用法。" for item in keypoint_items[:5])
    evidence_note = _format_context(context)[:360]

    return (
        f"# {title}\n\n"
        "## 一、复习目标\n"
        "- 梳理课程核心知识点，形成可快速回顾的复习框架。\n"
        "- 明确重点概念、常见用法、易错点和自测方向。\n\n"
        "## 二、核心知识点总览\n"
        f"{overview_lines}\n\n"
        "## 三、重点概念解释\n"
        f"{concept_lines}\n\n"
        "## 四、易错点整理\n"
        "- 只记结论、不理解适用条件，容易在变式题或实践场景中误用。\n"
        "- 忽略概念之间的边界，容易把相似术语、函数、流程或方法混在一起。\n\n"
        "## 五、自测清单\n"
        f"{self_check_lines}\n"
        "- [ ] 我能用自己的话总结本章节的复习路线。\n\n"
        "## 六、参考依据与说明\n"
        f"- 资料依据摘要：{evidence_note}\n"
        f"- 本次由简化模式生成，原因是模型生成长文档时连接中断。错误类型：{reason}。\n"
    )


async def _generate_review_doc_markdown(
    *,
    state: LearningState,
    query: str,
    keypoints: list[str],
    outline: str,
    revision_notes: str,
    context: list[dict],
    round_no: int,
    subject: str = "",
    title: str = "",
) -> dict:
    temperature = get_setting("review_doc.temperature", 0.2)
    timeout_seconds = float(get_setting("review_doc.timeout_seconds", 90) or 90)
    max_retries = int(get_setting("review_doc.agent_max_retries", 2) or 2)
    model_name = get_setting(
        "llm.review_doc.model",
        get_setting("review_doc.model", os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")),
    )
    llm = get_node_llm("review_doc")
    fallback = get_fallback_llm(temperature=temperature)

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
        SystemMessage(content="You are a university course Markdown review-document generator. Output Markdown only."),
        HumanMessage(content=prompt),
    ]

    markdown = ""
    fallback_used = False
    last_error_type = ""
    last_error_message = ""
    retries_used = 0

    for attempt in range(max_retries + 1):
        try:
            with traced_llm_call(model_name=model_name, node_name="review_doc_agent", temperature=temperature) as span:
                result = await asyncio.wait_for(
                    async_invoke_with_fallback(llm, messages, fallback=fallback, span=span),
                    timeout=timeout_seconds,
                )
            markdown = str(getattr(result, "content", result) or "").strip()
            if markdown:
                break
            last_error_type = "EmptyResponse"
            last_error_message = "LLM returned empty Markdown content"
        except Exception as exc:
            last_error_type = _review_doc_error_type(exc)
            last_error_message = str(exc)
            if attempt < max_retries and _is_retriable_review_doc_error(exc):
                retries_used = attempt + 1
                logger.warning(
                    "review_doc_agent retry %s/%s after %s: %s",
                    retries_used,
                    max_retries,
                    last_error_type,
                    last_error_message,
                )
                emit_a3_trace(
                    logger,
                    "review_doc_agent_retry",
                    {
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "error_type": last_error_type,
                        "error_message": last_error_message[:300],
                        "timeout_seconds": timeout_seconds,
                        "subject": subject,
                    },
                    state=state,
                    env_flag="LOG_GENERATION_SUMMARY",
                )
                await asyncio.sleep(min(0.5 * (attempt + 1), 2.0))
                continue
            logger.warning(
                "review_doc_agent generation failed; using fallback Markdown (%s: %s)",
                last_error_type,
                last_error_message,
            )
            break

    if not markdown:
        fallback_used = True
        fallback_reason = last_error_type or "UnknownGenerationError"
        markdown = _fallback_review_doc_markdown(
            title=title or _extract_markdown_title(outline),
            keypoints=keypoints,
            outline=outline,
            context=context,
            reason=fallback_reason,
        )
        emit_a3_trace(
            logger,
            "review_doc_agent_fallback",
            {
                "fallback_used": True,
                "review_doc_agent_error_type": fallback_reason,
                "error_message": last_error_message[:300],
                "retries_used": retries_used,
                "timeout_seconds": timeout_seconds,
                "markdown_chars": len(markdown),
                "subject": subject,
            },
            state=state,
            env_flag="LOG_GENERATION_SUMMARY",
        )

    if title:
        markdown = _ensure_markdown_title(markdown, title)

    return {
        "markdown": markdown,
        "fallback_used": fallback_used,
        "last_error_type": last_error_type,
        "last_error_message": last_error_message,
        "retries_used": retries_used,
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
    keypoints = state.get("keypoints", [])
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
        fallback_used_any = False
        last_error_types: list[str] = []
        retries_used_total = 0
        timeout_seconds = float(get_setting("review_doc.timeout_seconds", 90) or 90)

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
                keypoints=keypoints,
                outline=subject_outline,
                revision_notes=revision_notes,
                context=subject_context,
                round_no=round_no,
                subject=subject,
                title=title,
            )
            markdown = str(result.get("markdown") or "").strip()
            documents.append({"subject": subject, "title": title, "markdown": markdown})
            fallback_used_any = fallback_used_any or bool(result.get("fallback_used"))
            if result.get("last_error_type"):
                last_error_types.append(str(result.get("last_error_type")))
            retries_used_total += int(result.get("retries_used") or 0)

        combined_markdown = "\n\n---\n\n".join(doc["markdown"] for doc in documents if doc.get("markdown"))
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
                "fallback_used": fallback_used_any,
                "review_doc_agent_error_type": ",".join(last_error_types),
                "retries_used": retries_used_total,
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
                "fallback_used": fallback_used_any,
                "fallback_reason": ",".join(last_error_types),
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
        keypoints=keypoints,
        outline=outline,
        revision_notes=revision_notes,
        context=context,
        round_no=round_no,
    )
    markdown = str(result.get("markdown") or "").strip()
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
            "fallback_used": bool(result.get("fallback_used")),
            "review_doc_agent_error_type": result.get("last_error_type", ""),
            "retries_used": int(result.get("retries_used") or 0),
            "timeout_seconds": result.get("timeout_seconds", ""),
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "review_doc_markdown": markdown,
        "review_doc_markdowns": [],
        "review_doc_round": round_no,
        "review_doc_artifact": {
            "fallback_used": bool(result.get("fallback_used")),
            "fallback_reason": result.get("last_error_type", ""),
        },
        "review_doc_artifacts": [],
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

    if (state.get("review_doc_artifact") or {}).get("fallback_used"):
        reason = "Fallback Markdown passed local structure checks."
        emit_a3_trace(
            logger,
            "review_doc_reviewer",
            {
                "verdict": "approve",
                "reason": reason,
                "markdown_chars": len(markdown),
                "local_check_passed": True,
                "fallback_used": True,
                "skipped_llm_reviewer": True,
            },
            state=state,
            env_flag="LOG_GENERATION_SUMMARY",
        )
        return {
            "review_doc_review_verdict": "approve",
            "review_doc_review_reason": reason,
            "review_doc_revision_notes": "",
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
    review_verdict = state.get("review_doc_review_verdict", "")
    review_reason = state.get("review_doc_review_reason", "")
    prior_artifact = state.get("review_doc_artifact") or {}
    review_doc_markdowns = state.get("review_doc_markdowns") or []
    if review_doc_markdowns:
        review_doc_artifacts: list[dict] = []
        for doc in review_doc_markdowns:
            doc_markdown = str(doc.get("markdown") or "").strip()
            if not doc_markdown:
                continue
            doc_title = str(doc.get("title") or _extract_markdown_title(doc_markdown))
            artifact = create_markdown_artifact(doc_markdown, doc_title)
            review_doc_artifacts.append(
                {
                    **artifact,
                    "subject": str(doc.get("subject") or ""),
                    "title": doc_title,
                    "markdown": doc_markdown,
                    "quality_warning": review_verdict == "reject",
                    "review_reason": review_reason,
                }
            )

        first_artifact = review_doc_artifacts[0] if review_doc_artifacts else {}
        combined_markdown = "\n\n---\n\n".join(
            artifact["markdown"] for artifact in review_doc_artifacts if artifact.get("markdown")
        )
        subjects = [artifact.get("subject", "") for artifact in review_doc_artifacts if artifact.get("subject")]
        emit_a3_trace(
            logger,
            "review_doc_output",
            {
                "multi_document": True,
                "document_count": len(review_doc_markdowns),
                "artifact_count": len(review_doc_artifacts),
                "subjects": subjects,
                "markdown_chars": len(combined_markdown),
                "quality_warning": review_verdict == "reject",
                "review_reason": review_reason,
                "emits_ai_message": True,
                "fallback_used": bool(prior_artifact.get("fallback_used")),
            },
            state=state,
            env_flag="LOG_GENERATION_SUMMARY",
        )
        return {
            "review_doc_markdown": combined_markdown,
            "review_doc_artifact": {
                **prior_artifact,
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
            "quality_warning": review_verdict == "reject",
            "review_reason": review_reason,
            "emits_ai_message": True,
            "fallback_used": bool(prior_artifact.get("fallback_used")),
            "markdown_url": artifact.get("markdown_url", ""),
            "docx_url": artifact.get("docx_url", ""),
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    final_artifact = {
        **prior_artifact,
        **artifact,
        "markdown": markdown,
        "quality_warning": review_verdict == "reject",
        "review_reason": review_reason,
    }
    return {
        "review_doc_artifact": final_artifact,
        "review_doc_artifacts": [final_artifact],
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
