"""Teaching video / animation script resource-generation nodes."""

from __future__ import annotations

import logging
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from src.config import get_setting
from src.graph.llm import invoke_plain_llm_fail_fast
from src.graph.state import LearningState
from src.llm.structured_output import (
    StructuredOutputError,
    get_llm_output_mode,
    get_max_raw_chars,
    invoke_structured_llm,
)
from src.observability.a3_trace import emit_a3_trace
from src.tools.document_tool import create_video_script_artifact
from src.tracing import traced_llm_call, traced_node

logger = logging.getLogger(__name__)


GENERIC_TITLE_LABEL = "\u6807\u9898"

REQUIRED_VIDEO_SCRIPT_SECTIONS = {
    "title": r"(?m)^#\s+\S+",
    "basic": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?视频基本信息",
    "knowledge": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?知识点拆解",
    "storyboard": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?视频分镜脚本",
    "narration": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?完整旁白文案",
    "srt": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?字幕\s*SRT",
    "animation": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?动画设计说明",
    "board": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?板书内容",
    "interaction": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?互动提问",
    "summary": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?结尾总结",
    "practice": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?拓展练习",
}

REQUIRED_VIDEO_SCRIPT_SECTION_NAMES = {
    "basic": "视频基本信息",
    "knowledge": "知识点拆解",
    "storyboard": "视频分镜脚本",
    "narration": "完整旁白文案",
    "srt": "字幕 SRT",
    "animation": "动画设计说明",
    "board": "板书内容",
    "interaction": "互动提问",
    "summary": "结尾总结",
    "practice": "拓展练习",
}

STORYBOARD_HEADER_RE = re.compile(
    r"\|\s*镜头\s*\|\s*时间\s*\|\s*画面内容\s*\|\s*旁白\s*\|\s*字幕\s*\|\s*动画说明\s*\|"
)
SRT_TIMECODE_RE = re.compile(
    r"(?m)^\d+\s*\n\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}\s*\n.+"
)


class VideoScriptReviewVerdict(BaseModel):
    """Structured teaching-quality gate output for video_script_reviewer."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    verdict: Literal["approve", "reject"]
    reason: str = Field(min_length=1)


class VideoScriptGenerationError(RuntimeError):
    """Raised when no real provider-produced video script can be accepted."""


class VideoScriptApprovalError(RuntimeError):
    """Raised when an unapproved video script reaches artifact output."""


def _video_script_model_name() -> str:
    configured_model = get_setting("llm.video_script.model", None)
    if not isinstance(configured_model, str) or not configured_model.strip():
        raise ValueError("llm.video_script.model must be explicitly configured")
    return configured_model.strip()


def _video_script_temperature() -> float:
    configured_temperature = get_setting("llm.video_script.temperature", None)
    if isinstance(configured_temperature, bool) or not isinstance(
        configured_temperature, (int, float)
    ):
        raise ValueError("llm.video_script.temperature must be explicitly configured")
    temperature = float(configured_temperature)
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("llm.video_script.temperature must be between 0 and 2")
    return temperature


def validate_video_script_verdict(parsed: BaseModel) -> str:
    if not isinstance(parsed, VideoScriptReviewVerdict):
        return "root expected VideoScriptReviewVerdict"
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
    values = [
        str(item).strip() for item in state.get("keypoints", []) if str(item).strip()
    ]
    expanded = [
        str(item).strip()
        for item in state.get("expanded_keypoints", [])
        if str(item).strip()
    ]
    merged = values + [item for item in expanded if item not in values]
    return ", ".join(merged) or "No explicit keypoints."


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


def _extract_markdown_title(markdown: str) -> str:
    for line in markdown.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            raw_title = re.sub(r"\s+", " ", match.group(1)).strip()
            book_title = re.search(
                f"{re.escape(chr(0x300A))}\\s*(?P<title>.+?)\\s*{re.escape(chr(0x300B))}",
                raw_title,
            )
            if book_title:
                title = book_title.group("title").strip()
                if title:
                    return title

            title = re.sub(
                rf"^{re.escape(GENERIC_TITLE_LABEL)}\s*[:：\-—]*\s*",
                "",
                raw_title,
            ).strip()
            if title and title != GENERIC_TITLE_LABEL:
                return title
            break
    raise ValueError("video script Markdown title is missing or generic")


def _section_body(markdown: str, section_key: str) -> str:
    pattern = REQUIRED_VIDEO_SCRIPT_SECTIONS.get(section_key)
    if not pattern:
        return ""
    heading_match = re.search(pattern, markdown or "")
    if not heading_match:
        return ""
    start = heading_match.end()
    next_heading = re.search(r"(?m)^##\s+", markdown[start:])
    end = start + next_heading.start() if next_heading else len(markdown)
    return markdown[start:end].strip()


def _extract_srt_from_markdown(markdown: str) -> str:
    body = _section_body(markdown or "", "srt")
    if not body:
        return ""
    body = re.sub(r"(?m)^```(?:srt|text)?\s*$", "", body).strip()
    body = re.sub(r"(?m)^```\s*$", "", body).strip()
    return body


def _topic_terms(state: dict) -> list[str]:
    values: list[str] = [_last_human_query(state)]
    for key in ("primary_subject", "learning_goal"):
        value = str(state.get(key) or "").strip()
        if value:
            values.append(value)
    values.extend(str(item) for item in state.get("keypoints", []) if str(item).strip())
    values.extend(
        str(item) for item in state.get("expanded_keypoints", []) if str(item).strip()
    )

    joined = " ".join(values).lower()
    terms: list[str] = []
    for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_+#.-]{1,}", joined):
        if term not in {"video", "script", "animation", "python"} and term not in terms:
            terms.append(term)
    for phrase in (
        "面向对象",
        "类",
        "对象",
        "函数",
        "数据结构",
        "机器学习",
        "大数据",
        "计算机科学",
    ):
        if phrase in joined and phrase not in terms:
            terms.append(phrase)
    return terms


def _is_topic_relevant(markdown: str, state: dict) -> bool:
    subject = str(state.get("primary_subject") or "").strip().lower()
    combined = str(markdown or "").lower()
    if subject == "python":
        return "python" in combined or "类" in combined or "对象" in combined
    if subject and subject != "other":
        candidates = {subject, subject.replace("_", " "), subject.replace("_", "")}
        return any(item and item in combined for item in candidates)
    terms = _topic_terms(state)
    return bool(terms) and any(term.lower() in combined for term in terms)


def _local_check_video_script(markdown: str, state: dict) -> dict:
    text = str(markdown or "").strip()
    srt = _extract_srt_from_markdown(text)
    missing_sections = [
        name
        for key, name in REQUIRED_VIDEO_SCRIPT_SECTION_NAMES.items()
        if not re.search(REQUIRED_VIDEO_SCRIPT_SECTIONS[key], text)
    ]
    has_title = bool(re.search(REQUIRED_VIDEO_SCRIPT_SECTIONS["title"], text))
    storyboard_body = _section_body(text, "storyboard")
    narration_body = _section_body(text, "narration")
    animation_body = _section_body(text, "animation")
    interaction_body = _section_body(text, "interaction")
    summary_body = _section_body(text, "summary")
    has_storyboard_table = bool(STORYBOARD_HEADER_RE.search(storyboard_body))
    has_narration = len(narration_body) >= 40
    has_srt = bool(SRT_TIMECODE_RE.search(srt))
    has_animation = len(animation_body) >= 20
    has_interaction = bool(interaction_body)
    has_summary = bool(summary_body)
    topic_relevant = _is_topic_relevant(text, state)

    failed_reasons: list[str] = []
    if not text:
        failed_reasons.append("视频脚本 Markdown 为空")
    if not has_title:
        failed_reasons.append("缺少一级标题")
    if missing_sections:
        failed_reasons.append(f"缺少必要章节: {', '.join(missing_sections)}")
    if not has_storyboard_table:
        failed_reasons.append("缺少标准视频分镜表格")
    if not has_narration:
        failed_reasons.append("缺少完整旁白文案")
    if not has_srt:
        failed_reasons.append("缺少标准 SRT 字幕")
    if not has_animation:
        failed_reasons.append("缺少动画设计说明")
    if not has_interaction:
        failed_reasons.append("缺少互动提问")
    if not has_summary:
        failed_reasons.append("缺少结尾总结")
    if not topic_relevant:
        failed_reasons.append("内容和用户主题相关性不足")

    return {
        "passed": not failed_reasons,
        "failed_reasons": failed_reasons,
        "missing_sections": missing_sections,
        "has_storyboard_table": has_storyboard_table,
        "has_narration": has_narration,
        "has_srt": has_srt,
        "has_animation": has_animation,
        "has_interaction": has_interaction,
        "has_summary": has_summary,
        "topic_relevant": topic_relevant,
        "srt_chars": len(srt),
    }


def _planner_prompt(state: LearningState, query: str, context: list[dict]) -> str:
    return (
        "请根据用户问题、学习目标、关键词和检索资料，规划一份教学视频 / 动画脚本大纲。\n\n"
        f"## 用户问题\n{query}\n\n"
        f"## learning_goal\n{state.get('learning_goal', '') or '未提供'}\n\n"
        f"## keypoints / expanded_keypoints\n{_format_keypoints(state)}\n\n"
        f"## context\n{_format_context(context)}\n\n"
        "## 输出要求\n"
        "只输出大纲，不要生成完整正文。大纲必须包含：\n"
        "- 视频主题\n"
        "- 适用学生\n"
        "- 预计时长\n"
        "- 学习目标\n"
        "- 知识点拆解\n"
        "- 分镜数量\n"
        "- 动画表现方式\n"
        "- 互动问题\n"
        "- 字幕设计\n"
        "不得编造教材页码、文件名或不存在的来源。"
    )


def _agent_prompt(state: LearningState, outline: str) -> str:
    return (
        "请根据视频脚本大纲生成一份 Markdown 教学视频 / 动画脚本文档。\n\n"
        f"## 用户问题\n{_last_human_query(state)}\n\n"
        f"## 视频脚本大纲\n{outline}\n\n"
        f"## 检索资料\n{_format_context(state.get('context', []))}\n\n"
        f"## 修订意见\n{state.get('video_script_revision_notes', '') or 'None'}\n\n"
        "## 必须满足的 Markdown 结构\n"
        "# 标题\n"
        "## 一、视频基本信息\n"
        "## 二、知识点拆解\n"
        "## 三、视频分镜脚本\n"
        "## 四、完整旁白文案\n"
        "## 五、字幕 SRT\n"
        "## 六、动画设计说明\n"
        "## 七、板书内容\n"
        "## 八、互动提问\n"
        "## 九、结尾总结\n"
        "## 十、拓展练习\n\n"
        "视频分镜脚本必须是 Markdown 表格，表头必须严格为：\n"
        "| 镜头 | 时间 | 画面内容 | 旁白 | 字幕 | 动画说明 |\n\n"
        "字幕 SRT 必须包含标准 SRT 时间轴，例如：\n"
        "1\n00:00:00,000 --> 00:00:05,000\n欢迎来到本节课……\n\n"
        "请输出可直接交给视频制作人员使用的中等长度脚本。"
        "如果资料不足，请在文末说明资料依据不足。"
    )


def _create_video_script_artifact(markdown: str, title: str, srt: str) -> dict:
    return create_video_script_artifact(
        markdown_text=markdown, title=title, srt_text=srt
    )


@traced_node
async def video_script_planner(state: LearningState) -> dict:
    query = _last_human_query(state)
    context = state.get("context", [])
    outline = await invoke_plain_llm_fail_fast(
        node_name="video_script_planner",
        llm_node="video_script",
        messages=[
            SystemMessage(
                content="You are a university teaching-video planner. Return a concrete blueprint only."
            ),
            HumanMessage(content=_planner_prompt(state, query, context)),
        ],
        state=state,
        temperature=_video_script_temperature(),
    )
    if not outline.strip():
        raise ValueError("video_script_planner produced empty outline")

    emit_a3_trace(
        logger,
        "video_script_planner",
        {
            "outline_chars": len(outline),
            "context_count": len(context),
            "dual_source_mode": bool(state.get("dual_source_mode")),
            "evidence_judge_state": state.get("evidence_judge_state", ""),
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "video_script_outline": outline.strip(),
        "video_script_markdown": "",
        "video_script_srt": "",
        "video_script_artifact": {},
        "video_script_review_verdict": "",
        "video_script_review_reason": "",
        "video_script_revision_notes": "",
        "video_script_round": 0,
    }


@traced_node
async def video_script_agent(state: LearningState) -> dict:
    outline = state.get("video_script_outline", "")
    if not outline.strip():
        raise VideoScriptGenerationError("video script outline is empty")

    round_no = int(state.get("video_script_round", 0) or 0) + 1
    if (
        state.get("degraded_generation") is True
        and state.get("evidence_judge_state") == "insufficient"
    ):
        raise VideoScriptGenerationError(
            "video script generation blocked because evidence is insufficient"
        )
    markdown = await invoke_plain_llm_fail_fast(
        node_name="video_script_agent",
        llm_node="video_script",
        messages=[
            SystemMessage(
                content="You are a teaching-video and animation script writer. Return Markdown only."
            ),
            HumanMessage(content=_agent_prompt(state, outline)),
        ],
        state=state,
        temperature=_video_script_temperature(),
    )
    if not markdown.strip():
        raise VideoScriptGenerationError("video_script_agent produced empty markdown")

    srt = _extract_srt_from_markdown(markdown)
    emit_a3_trace(
        logger,
        "video_script_agent",
        {
            "markdown_chars": len(markdown),
            "srt_chars": len(srt),
            "round": round_no,
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "video_script_markdown": markdown.strip(),
        "video_script_srt": srt,
        "video_script_artifact": {"title": _extract_markdown_title(markdown)},
        "video_script_round": round_no,
        "video_script_review_verdict": "",
        "video_script_review_reason": "",
    }


@traced_node
async def video_script_reviewer(state: LearningState) -> dict:
    markdown = state.get("video_script_markdown", "")
    local_check = _local_check_video_script(markdown, state)

    def trace_payload(verdict: str, reason: str) -> dict:
        return {
            "local_check_passed": bool(local_check.get("passed")),
            "missing_sections": local_check.get("missing_sections", []),
            "has_storyboard_table": bool(local_check.get("has_storyboard_table")),
            "has_narration": bool(local_check.get("has_narration")),
            "has_srt": bool(local_check.get("has_srt")),
            "has_animation": bool(local_check.get("has_animation")),
            "has_interaction": bool(local_check.get("has_interaction")),
            "has_summary": bool(local_check.get("has_summary")),
            "topic_relevant": bool(local_check.get("topic_relevant")),
            "verdict": verdict,
            "reason": reason,
        }

    if not local_check["passed"]:
        reason = "; ".join(str(item) for item in local_check["failed_reasons"])
        emit_a3_trace(
            logger,
            "video_script_reviewer",
            trace_payload("reject", reason),
            state=state,
            env_flag="LOG_GENERATION_SUMMARY",
        )
        return {
            "video_script_review_verdict": "reject",
            "video_script_review_reason": reason,
            "video_script_revision_notes": reason,
            "video_script_local_check": local_check,
        }

    model_name = _video_script_model_name()
    with traced_llm_call(
        model_name=model_name, node_name="video_script_reviewer", temperature=0.0
    ):
        structured_result = await invoke_structured_llm(
            node_name="video_script_reviewer",
            llm_node="video_script",
            schema=VideoScriptReviewVerdict,
            messages=[
                SystemMessage(
                    content="You are a strict teaching-video script reviewer. Return only JSON."
                ),
                HumanMessage(
                    content=(
                        "Review this Markdown teaching-video / animation script for teaching quality.\n"
                        "The deterministic local check has already verified required structure, storyboard table, narration, SRT, animation design, interaction, summary, and topic relevance.\n"
                        'Return JSON only: {"verdict": "approve" or "reject", "reason": "..."}\n\n'
                        f"## User question\n{_last_human_query(state)}\n\n"
                        f"## Outline\n{state.get('video_script_outline', '')}\n\n"
                        f"## Markdown\n{markdown}"
                    )
                ),
            ],
            output_mode=get_llm_output_mode("video_script_reviewer"),
            business_validator=validate_video_script_verdict,
            state=state,
            max_raw_chars=get_max_raw_chars("video_script_reviewer"),
        )
    if not structured_result.success:
        raise StructuredOutputError(structured_result)
    result = structured_result.parsed
    if not isinstance(result, VideoScriptReviewVerdict):
        raise TypeError(
            "video_script_reviewer parsed result is not VideoScriptReviewVerdict"
        )
    verdict = result.verdict
    reason = result.reason.strip()
    emit_a3_trace(
        logger,
        "video_script_reviewer",
        trace_payload(verdict, reason),
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "video_script_review_verdict": verdict,
        "video_script_review_reason": reason,
        "video_script_revision_notes": "" if verdict == "approve" else reason,
        "video_script_local_check": local_check,
    }


@traced_node
async def video_script_rewrite(state: LearningState) -> dict:
    reason = state.get("video_script_review_reason", "")
    return {
        "video_script_revision_notes": f"Revise the video-script Markdown according to reviewer feedback:\n{reason}",
        "video_script_outline": state.get("video_script_outline", ""),
    }


@traced_node
async def video_script_output(state: LearningState) -> dict:
    markdown = state.get("video_script_markdown", "")
    if not markdown.strip():
        raise VideoScriptApprovalError("video script output received empty markdown")
    review_verdict = str(state.get("video_script_review_verdict") or "").strip()
    review_reason = str(state.get("video_script_review_reason") or "").strip()
    if review_verdict != "approve":
        detail = f": {review_reason}" if review_reason else ""
        raise VideoScriptApprovalError(
            f"video script output requires an approve verdict{detail}"
        )
    local_check = _local_check_video_script(markdown, state)
    if not local_check["passed"]:
        raise VideoScriptApprovalError(
            "video script output failed local quality check: "
            + "; ".join(str(item) for item in local_check["failed_reasons"])
        )

    title = _extract_markdown_title(markdown)
    srt = _extract_srt_from_markdown(markdown)
    artifact = _create_video_script_artifact(markdown, title, srt)
    artifact = {
        **(state.get("video_script_artifact") or {}),
        **artifact,
        "review_reason": review_reason,
    }
    emit_a3_trace(
        logger,
        "video_script_output",
        {
            "title": title,
            "markdown_chars": len(markdown),
            "srt_chars": len(srt),
            "markdown_url": artifact.get("markdown_url", ""),
            "docx_url": artifact.get("docx_url", ""),
            "srt_url": artifact.get("srt_url", ""),
            "emits_ai_message": True,
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "video_script_artifact": artifact,
        "video_script_markdown": markdown,
        "video_script_srt": srt,
        "messages": [AIMessage(content=markdown)],
    }


def should_rewrite_video_script(state: LearningState) -> str:
    verdict = str(state.get("video_script_review_verdict") or "").strip()
    if verdict == "approve":
        return "output"
    if verdict != "reject":
        raise VideoScriptApprovalError(
            "video script routing requires an explicit approve or reject verdict"
        )
    current_round = int(state.get("video_script_round", 0) or 0)
    if current_round < 2:
        return "rewrite"
    raise VideoScriptApprovalError(
        "video script remained rejected after the maximum rewrite rounds"
    )
