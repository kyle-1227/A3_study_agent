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
from src.graph.llm import async_invoke_with_fallback, get_fallback_llm, get_node_llm
from src.graph.state import  LearningState
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


class ReviewDocReviewVerdict(BaseModel):
    """Structured quality gate output for review_doc_reviewer."""

    verdict: Literal["approve", "reject"]
    reason: str


def _last_human_query(state: LearningState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _format_keypoints(state: LearningState) -> str:
    keypoints = state.get("keypoints", [])
    return "、".join(keypoints) if keypoints else "未提取到明确关键词"


def _subjects_used(context: list[dict]) -> list[str]:
    return sorted({str(item.get("retrieval_subject")) for item in context if item.get("retrieval_subject")})


def _roles_used(context: list[dict]) -> list[str]:
    return sorted({str(item.get("retrieval_role")) for item in context if item.get("retrieval_role")})


def _format_context(context: list[dict]) -> str:
    if not context:
        return (
            "当前课程知识库和外部资料未返回可用依据。可以按高校课程通用知识整理复习文档，"
            "但不得编造教材页码、文件名、章节号、网址或不存在的引用来源；文末必须说明资料依据不足。"
        )

    parts: list[str] = []
    for idx, item in enumerate(context[:8], 1):
        source = item.get("source") or item.get("title") or item.get("url") or "课程资料"
        content = str(item.get("content") or item.get("snippet") or item.get("text") or "")[:700]
        if content:
            parts.append(f"[{idx}] 来源：{source}\n{content}")
    return "\n\n".join(parts) or (
        "已有资料缺少可读正文，请结合用户请求整理通用课程复习文档，"
        "并在文末说明资料依据不足。"
    )


def _render_prompt(prompt_name: str, replacements: dict[str, str]) -> str:
    """Render named placeholders without interpreting literal braces in prompts."""
    prompt = load_prompt(prompt_name)
    for key, value in replacements.items():
        prompt = prompt.replace("{" + key + "}", value)
    return prompt


def _fallback_outline(query: str, keypoints: list[str], context: list[dict]) -> str:
    topic = "、".join(keypoints[:4]) or query[:40] or "课程知识点"
    evidence_note = _format_context(context)[:500]
    return (
        f"# {topic}复习文档大纲\n"
        "## 一、复习目标\n"
        "- 明确本主题需要掌握的概念、方法和应用能力。\n"
        "## 二、核心知识点总览\n"
        "- 按基础概念、关键机制、应用场景和复习优先级组织知识点。\n"
        "## 三、重点概念解释\n"
        "- 解释关键词、公式、算法、代码/API 或课程术语。\n"
        "## 四、知识点对比表\n"
        "- 规划“知识点/含义/适用场景/易混点/复习建议”表格。\n"
        "## 五、典型例题或示例\n"
        "- 设计贴合主题的例题、示例场景、代码片段或分析步骤。\n"
        "## 六、易错点整理\n"
        "- 总结常见误解、边界条件、考试陷阱和实践错误。\n"
        "## 七、复习路线建议\n"
        "- 给出从概念到应用再到自测的复习顺序。\n"
        "## 八、自测清单\n"
        "- 规划可勾选的掌握度检查项。\n"
        "## 九、参考依据\n"
        f"- 依据摘要：{evidence_note}"
    )


def _local_review_failure(markdown: str, _query: str) -> str:
    text = markdown.strip()
    if not text:
        return "复习文档为空。"

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
        return f"Markdown 缺少必要部分：{'、'.join(missing)}。"

    return ""


def _extract_markdown_title(markdown: str) -> str:
    """Extract the first H1 title from Markdown for artifact naming."""
    for line in markdown.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()
    return "Markdown复习文档"


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
        "\n\n## 长度与稳定性要求\n"
        "1. 第一版复习文档控制在 1800-2500 中文字以内。\n"
        "2. 如果用户没有明确要求长文档，默认生成中等长度复习文档。\n"
        "3. 优先保证结构完整、概念清楚、表格简洁，不要一次性生成过长内容。\n"
    )


def _topic_from_query(query: str, keypoints: list[str]) -> str:
    joined = " ".join([query, " ".join(keypoints)]).lower()
    if "python" in joined:
        return "Python"
    for keypoint in keypoints:
        cleaned = str(keypoint).strip()
        if cleaned:
            return cleaned
    return query[:24].strip() or "课程"


def _fallback_review_doc_markdown(
    *,
    query: str,
    keypoints: list[str],
    outline: str,
    context: list[dict],
    reason: str,
) -> str:
    topic = _topic_from_query(query, keypoints)
    keypoint_items = [str(item).strip() for item in keypoints if str(item).strip()]
    if not keypoint_items:
        keypoint_items = _extract_outline_items(outline)[:6]
    if not keypoint_items:
        keypoint_items = [topic, "核心概念", "常见应用", "易错点"]

    concept_lines = "\n".join(f"- **{item}**：结合课程资料重点理解其定义、用途和适用边界。" for item in keypoint_items[:6])
    overview_lines = "\n".join(f"- {item}" for item in keypoint_items[:8])
    self_check_lines = "\n".join(f"- [ ] 我能说明“{item}”的含义和典型用法。" for item in keypoint_items[:5])
    evidence_note = _format_context(context)[:360]

    return (
        f"# {topic} 复习资料\n\n"
        "## 一、复习目标\n"
        f"- 梳理 {topic} 相关章节的核心知识点，形成可快速回顾的复习框架。\n"
        "- 明确重点概念、常见用法、易错点和自测方向。\n"
        "- 在资料有限时，优先掌握通用课程知识和可验证内容。\n\n"
        "## 二、核心知识点总览\n"
        f"{overview_lines}\n\n"
        "## 三、重点概念解释\n"
        f"{concept_lines}\n\n"
        "## 四、易错点整理\n"
        "- 只记结论、不理解适用条件，容易在变式题或实践场景中误用。\n"
        "- 忽略概念之间的边界，容易把相似术语、函数、流程或方法混在一起。\n"
        "- 复习时只看示例不做自测，容易出现“看懂但不会用”的问题。\n\n"
        "## 五、自测清单\n"
        f"{self_check_lines}\n"
        "- [ ] 我能用自己的话总结本章节的复习路线。\n"
        "- [ ] 我能列出至少 3 个容易出错的点，并说明如何避免。\n\n"
        "## 六、参考依据与说明\n"
        f"- 资料依据摘要：{evidence_note}\n"
        f"- 本次由简化模式生成，原因是模型生成长文档时连接中断。错误类型：{reason}。\n"
    )


def _extract_outline_items(outline: str) -> list[str]:
    items: list[str] = []
    for line in outline.splitlines():
        cleaned = re.sub(r"^[\s#\-*0-9一二三四五六七八九十、.．]+", "", line).strip()
        cleaned = re.sub(r"[:：].*$", "", cleaned).strip()
        if 2 <= len(cleaned) <= 30 and cleaned not in items:
            items.append(cleaned)
    return items


@traced_node
async def review_doc_planner(state: TutorState) -> dict:
    """Plan a Markdown review document from the user request and context."""
    query = _last_human_query(state)
    keypoints = state.get("keypoints", [])
    context = state.get("context", [])
    web_supplements = [
        item
        for item in context
        if item.get("type") in {"web_supplement", "web_evidence"} or item.get("source_type") == "web"
    ]

    emit_a3_trace(
        logger,
        "review_doc_planner",
        {
            "subjects_used": _subjects_used(context),
            "roles_used": _roles_used(context),
            "learning_goal": state.get("learning_goal", ""),
            "primary_subject": state.get("primary_subject", ""),
            "context_count": len(context),
            "web_supplement_count": len(web_supplements),
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

    llm = get_node_llm("review_doc")
    fallback = get_fallback_llm(temperature=get_setting("review_doc.temperature", 0.2))
    temperature = get_setting("review_doc.temperature", 0.2)
    model_name = get_setting("review_doc.model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))

    try:
        with traced_llm_call(model_name=model_name, node_name="review_doc_planner", temperature=temperature) as span:
            result = await async_invoke_with_fallback(
                llm,
                [
                    SystemMessage(content="你是高校课程 Markdown 复习文档规划智能体，负责生成复习文档大纲。"),
                    HumanMessage(content=prompt),
                ],
                fallback=fallback,
                span=span,
            )
        outline = str(getattr(result, "content", result)).strip()
    except Exception:
        logger.warning("Review document planner failed, using fallback outline", exc_info=True)
        outline = _fallback_outline(query, keypoints, context)

    if not outline:
        outline = _fallback_outline(query, keypoints, context)

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
async def review_doc_agent(state: TutorState) -> dict:
    """Generate a Markdown review document from the planner outline."""
    query = _last_human_query(state)
    keypoints = state.get("keypoints", [])
    outline = state.get("review_doc_outline", "")
    revision_notes = state.get("review_doc_revision_notes", "")
    context = state.get("context", [])
    round_no = int(state.get("review_doc_round", 0) or 0) + 1

    if not outline.strip():
        return {
            "error": "复习文档大纲为空，无法生成有质量保障的 Markdown 复习文档。",
            "review_doc_round": round_no,
        }

    prompt = _render_prompt(
        "review_doc_agent",
        {
            "question": query,
            "keypoints": _format_keypoints(state),
            "context": _format_context(context),
            "review_doc_outline": outline,
            "revision_notes": revision_notes or "暂无审查修订意见。",
        },
    )
    prompt += _review_doc_length_instruction()

    llm = get_node_llm("review_doc")
    fallback = get_fallback_llm(temperature=get_setting("review_doc.temperature", 0.2))
    temperature = get_setting("review_doc.temperature", 0.2)
    model_name = get_setting("review_doc.model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    timeout_seconds = float(get_setting("review_doc.timeout_seconds", 90) or 90)
    max_retries = int(get_setting("review_doc.agent_max_retries", 2) or 2)
    messages = [
        SystemMessage(content="你是高校课程 Markdown 复习文档生成智能体。只输出 Markdown 正文，默认生成中等长度复习文档。"),
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
                    async_invoke_with_fallback(
                        llm,
                        messages,
                        fallback=fallback,
                        span=span,
                    ),
                    timeout=timeout_seconds,
                )
            markdown = str(getattr(result, "content", result)).strip()
            if markdown:
                break
            last_error_type = "EmptyResponse"
            last_error_message = "LLM returned empty Markdown content"
        except Exception as exc:
            last_error_type = _review_doc_error_type(exc)
            last_error_message = str(exc)
            retryable = _is_retriable_review_doc_error(exc)
            if attempt < max_retries and retryable:
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
            query=query,
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
            },
            state=state,
            env_flag="LOG_GENERATION_SUMMARY",
        )

    emit_a3_trace(
        logger,
        "review_doc_agent",
        {
            "markdown_chars": len(markdown),
            "round": round_no,
            "has_outline": bool(outline.strip()),
            "has_revision_notes": bool(revision_notes.strip()),
            "context_count": len(context),
            "fallback_used": fallback_used,
            "review_doc_agent_error_type": last_error_type,
            "retries_used": retries_used,
            "timeout_seconds": timeout_seconds,
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )

    return {
        "review_doc_markdown": markdown,
        "review_doc_round": round_no,
        "review_doc_artifact": {
            "fallback_used": fallback_used,
            "fallback_reason": last_error_type,
        },
        "review_doc_review_verdict": "",
        "review_doc_review_reason": "",
    }


@traced_node
async def review_doc_reviewer(state: TutorState) -> dict:
    """Review the Markdown document for structure, usefulness, and source discipline."""
    query = _last_human_query(state)
    outline = state.get("review_doc_outline", "")
    markdown = state.get("review_doc_markdown", "")

    local_failure = _local_review_failure(markdown, query)
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
            "review_doc_revision_notes": f"请据此重写：{local_failure}",
        }

    if (state.get("review_doc_artifact") or {}).get("fallback_used"):
        reason = "简化模式 Markdown 已通过本地结构质量检查。"
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
            "question": query,
            "review_doc_outline": outline,
            "review_doc_markdown": markdown,
        },
    )

    llm = get_node_llm("review_doc", temperature=get_setting("review_doc.reviewer_temperature", 0.0))
    structured_llm = llm.with_structured_output(ReviewDocReviewVerdict, method="json_mode")
    fallback = get_fallback_llm(temperature=get_setting("review_doc.reviewer_temperature", 0.0))
    structured_fallback = fallback.with_structured_output(ReviewDocReviewVerdict, method="json_mode")
    model_name = get_setting("review_doc.model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))

    try:
        with traced_llm_call(model_name=model_name, node_name="review_doc_reviewer", temperature=0.0) as span:
            result = await async_invoke_with_fallback(
                structured_llm,
                [
                    SystemMessage(content="你是高校课程 Markdown 复习文档质量审查智能体，只返回 JSON 审查结论。"),
                    HumanMessage(content=prompt),
                ],
                fallback=structured_fallback,
                span=span,
            )
        verdict = result.verdict
        reason = result.reason.strip()
    except Exception:
        logger.warning("Review document reviewer failed, approving document that passed local checks", exc_info=True)
        verdict = "approve"
        reason = "已通过本地 Markdown 结构质量检查。"

    emit_a3_trace(
        logger,
        "review_doc_reviewer",
        {
            "verdict": verdict,
            "reason": reason,
            "markdown_chars": len(markdown),
            "local_check_passed": True,
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )

    return {
        "review_doc_review_verdict": verdict,
        "review_doc_review_reason": reason,
        "review_doc_revision_notes": "" if verdict == "approve" else f"请据此重写：{reason}",
    }


@traced_node
async def review_doc_rewrite(state: TutorState) -> dict:
    """Prepare reviewer feedback for the next Markdown generation attempt."""
    reason = state.get("review_doc_review_reason", "")
    outline = state.get("review_doc_outline", "")
    notes = (
        f"{reason}\n"
        "重写要求：必须输出 Markdown 正文；必须包含标题、复习目标、核心知识点总览、重点概念解释、"
        "知识点对比表、典型例题或示例、易错点整理、复习路线建议、自测清单、参考依据；"
        "不得编造教材页码、文件名或不存在的来源。"
    )
    return {
        "review_doc_revision_notes": notes,
        "review_doc_outline": outline,
    }


@traced_node
async def review_doc_output(state: TutorState) -> dict:
    """Emit the final Markdown review document as the user-facing message."""
    markdown = state.get("review_doc_markdown", "")
    if not markdown:
        return {
            "error": "当前知识依据不足，未能生成可用的 Markdown 复习文档。",
            "messages": [
                AIMessage(content="当前知识依据不足，暂时无法生成质量可靠的 Markdown 复习文档。请补充课程主题、章节或材料后重试。")
            ],
        }

    review_verdict = state.get("review_doc_review_verdict", "")
    review_reason = state.get("review_doc_review_reason", "")
    prior_artifact = state.get("review_doc_artifact") or {}
    title = _extract_markdown_title(markdown)
    artifact = create_markdown_artifact(markdown, title)
    emit_a3_trace(
        logger,
        "review_doc_output",
        {
            "markdown_chars": len(markdown),
            "quality_warning": review_verdict == "reject",
            "review_reason": review_reason,
            "emits_ai_message": True,
            "markdown_url": artifact.get("markdown_url", ""),
            "docx_url": artifact.get("docx_url", ""),
            "fallback_used": bool(prior_artifact.get("fallback_used")),
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )

    return {
        "review_doc_artifact": {
            **prior_artifact,
            **artifact,
            "markdown": markdown,
            "quality_warning": review_verdict == "reject",
            "review_reason": review_reason,
        },
        "messages": [AIMessage(content=markdown)],
    }


def should_rewrite_review_doc(state: TutorState) -> str:
    """Route reviewer output to rewrite or final Markdown output."""
    if state.get("review_doc_review_verdict") != "reject":
        return "output"
    max_rounds = int(get_setting("review_doc.max_generation_rounds", 3) or 3)
    current_round = int(state.get("review_doc_round", 0) or 0)
    return "rewrite" if current_round < max_rounds else "output"
