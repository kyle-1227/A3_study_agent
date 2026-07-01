"""Teaching video / animation script resource-generation nodes."""

from __future__ import annotations

import logging
import os
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

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
from src.tools.document_tool import create_video_script_artifact
from src.tracing import traced_llm_call, traced_node

logger = logging.getLogger(__name__)


VIDEO_SCRIPT_DEFAULT_MODEL = "deepseek-v4-flash"
GENERIC_TITLE_LABEL = "\u6807\u9898"
FALLBACK_VIDEO_SCRIPT_TITLE = "Python-\u6559\u5b66\u89c6\u9891\u811a\u672c"

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

    verdict: Literal["approve", "reject"]
    reason: str


def _video_script_model_name() -> str:
    configured_model = get_setting(
        "llm.video_script.model",
        get_setting("video_script.model", None),
    )
    return str(
        configured_model or os.getenv("DEEPSEEK_MODEL") or VIDEO_SCRIPT_DEFAULT_MODEL
    )


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
    return FALLBACK_VIDEO_SCRIPT_TITLE


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
    return True if not terms else any(term.lower() in combined for term in terms)


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


def _fallback_video_script_markdown(state: LearningState, reason: str = "") -> str:
    return f"""# 《10分钟理解 Python 类与对象》

## 一、视频基本信息
- 视频主题：Python 面向对象中的类与对象
- 适用学生：已经了解变量、函数和基本控制流的 Python 初学者
- 预计时长：10 分钟
- 学习目标：理解类是模板、对象是实例，并能说出属性和方法的作用

## 二、知识点拆解
- 类：描述一类事物的共同结构和行为
- 对象：由类创建出来的具体实例
- 属性：对象保存的数据
- 方法：对象可以执行的操作
- `__init__`：创建对象时初始化属性

## 三、视频分镜脚本
| 镜头 | 时间 | 画面内容 | 旁白 | 字幕 | 动画说明 |
|---|---|---|---|---|---|
| 1 | 00:00-00:20 | 标题页与 Python 图标 | 欢迎来到本节课，我们用 10 分钟理解 Python 类与对象。 | 10分钟理解 Python 类与对象 | 标题淡入，图标轻微缩放 |
| 2 | 00:20-01:20 | 展示“图纸”和“汽车”类比 | 类像图纸，对象像根据图纸造出来的一辆具体汽车。 | 类是模板，对象是实例 | 图纸变成多辆汽车 |
| 3 | 01:20-03:10 | 代码展示 `class Student` | 我们用学生作为例子，类中保存姓名和分数。 | class Student 定义学生模板 | 代码逐行高亮 |
| 4 | 03:10-05:20 | 展示创建对象 `alice = Student(...)` | 当我们调用类名时，就创建了一个对象。 | 对象保存自己的属性 | 箭头连接变量和对象卡片 |
| 5 | 05:20-07:20 | 展示方法 `average_score` | 方法是写在类里的函数，用来处理对象自己的数据。 | 方法 = 对象的行为 | 方法调用动画 |
| 6 | 07:20-09:20 | 总结类、对象、属性、方法关系 | 类、对象、属性和方法组合起来，就能表达更复杂的程序模型。 | 类 + 对象 + 属性 + 方法 | 四个概念组成结构图 |
| 7 | 09:20-10:00 | 结尾提问与练习 | 请尝试写一个 Book 类，保存书名和作者。 | 拓展练习：Book 类 | 练习卡片弹出 |

## 四、完整旁白文案
欢迎来到本节课，我们将用 10 分钟理解 Python 面向对象中最核心的两个概念：类与对象。你可以先把类想象成一张图纸，把对象想象成根据图纸制造出来的具体物品。图纸本身不是汽车，但它规定了汽车应该有哪些结构；同样，类本身不是某个具体学生，但它规定了学生对象应该有哪些属性和方法。

接下来我们用 `Student` 类作为例子。`name` 和 `scores` 是学生对象保存的数据，也就是属性。`average_score` 是学生对象可以执行的操作，也就是方法。当我们写出 `alice = Student("Alice", [90, 88, 95])` 时，Python 会根据 `Student` 这个模板创建出一个具体对象。每个对象都有自己的属性，所以 Alice 和 Bob 可以有不同的成绩。

最后请记住：类负责定义模板，对象负责保存具体数据，属性描述对象是什么，方法描述对象能做什么。理解这一点，你就能开始用面向对象的方式组织更复杂的 Python 程序。

## 五、字幕 SRT
1
00:00:00,000 --> 00:00:05,000
欢迎来到本节课，我们用 10 分钟理解 Python 类与对象。

2
00:00:05,000 --> 00:00:12,000
类可以理解为模板，对象是根据模板创建出来的具体实例。

3
00:00:12,000 --> 00:00:20,000
在 Student 类中，name 和 scores 是属性，average_score 是方法。

4
00:00:20,000 --> 00:00:30,000
当我们调用 Student 时，就会创建一个保存具体数据的学生对象。

5
00:00:30,000 --> 00:00:40,000
请记住：类定义模板，对象保存数据，方法描述对象能做什么。

## 六、动画设计说明
- 用“图纸生成汽车”的动画解释类和对象的关系。
- 用对象卡片展示不同学生拥有不同属性。
- 用代码逐行高亮强调 `class`、`__init__`、属性和方法。

## 七、板书内容
- 类：对象的模板
- 对象：类的实例
- 属性：对象保存的数据
- 方法：对象可以执行的操作

## 八、互动提问
1. 如果 `Student` 是类，那么 `alice` 是什么？
2. `name` 和 `scores` 为什么适合作为属性？
3. `average_score()` 为什么适合作为方法？

## 九、结尾总结
本节课的重点是建立面向对象的基本心智模型。类不是具体数据，而是模板；对象才是具体实例。属性记录对象状态，方法描述对象行为。

## 十、拓展练习
- 写一个 `Book` 类，包含书名、作者和价格。
- 给 `Book` 类增加一个 `discount()` 方法。
- 尝试创建三个 Book 对象并打印它们的信息。

> 本脚本由简化模式生成。原因：{reason or "模型或证据不足，系统使用 fallback 视频脚本结构生成资源。"}
"""


def _fallback_video_script_outline(state: LearningState, reason: str = "") -> str:
    return (
        "视频主题：10分钟理解 Python 类与对象\n"
        "适用学生：Python 初学者\n"
        "预计时长：10 分钟\n"
        "学习目标：理解类、对象、属性、方法和 __init__\n"
        "知识点拆解：类是模板；对象是实例；属性保存数据；方法描述行为\n"
        "分镜数量：7 个镜头\n"
        "动画表现方式：图纸类比、对象卡片、代码高亮、结构图总结\n"
        "互动问题：区分类和对象；判断属性与方法\n"
        "字幕设计：短句字幕，配合关键概念高亮\n"
        f"生成说明：{reason or 'fallback outline'}"
    )


def _create_video_script_artifact(markdown: str, title: str, srt: str) -> dict:
    return create_video_script_artifact(
        markdown_text=markdown, title=title, srt_text=srt
    )


@traced_node
async def video_script_planner(state: LearningState) -> dict:
    query = _last_human_query(state)
    context = state.get("context", [])
    try:
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
            temperature=get_setting("video_script.temperature", 0.2),
        )
    except Exception as exc:
        logger.warning("video_script_planner fallback used: %s", exc)
        outline = _fallback_video_script_outline(state, f"{type(exc).__name__}: {exc}")
    if not outline.strip():
        outline = _fallback_video_script_outline(
            state, "planner produced empty outline"
        )

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
        outline = _fallback_video_script_outline(state, "outline is empty")

    round_no = int(state.get("video_script_round", 0) or 0) + 1
    fallback_used = False
    fallback_reason = ""
    if (
        state.get("degraded_generation") is True
        and state.get("evidence_judge_state") == "insufficient"
    ):
        fallback_used = True
        fallback_reason = str(
            state.get("degraded_reason")
            or "Evidence is insufficient; generating video script with fallback structure."
        )
        markdown = _fallback_video_script_markdown(state, fallback_reason)
    else:
        try:
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
                temperature=get_setting("video_script.temperature", 0.2),
            )
        except Exception as exc:
            fallback_used = True
            fallback_reason = f"{type(exc).__name__}: {exc}"
            logger.warning("video_script_agent fallback used: %s", fallback_reason)
            markdown = _fallback_video_script_markdown(
                state, "模型生成视频脚本失败，系统使用 fallback 结构生成资源。"
            )
    if not markdown.strip():
        fallback_used = True
        fallback_reason = "video_script_agent produced empty markdown"
        markdown = _fallback_video_script_markdown(state, fallback_reason)

    srt = _extract_srt_from_markdown(markdown)
    emit_a3_trace(
        logger,
        "video_script_agent",
        {
            "markdown_chars": len(markdown),
            "srt_chars": len(srt),
            "round": round_no,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
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

    def trace_payload(
        verdict: str, reason: str, *, llm_fallback_used: bool = False
    ) -> dict:
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
            "llm_fallback_used": llm_fallback_used,
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
    try:
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
                fallback_modes=get_fallback_modes("video_script_reviewer"),
                business_validator=validate_video_script_verdict,
                state=state,
                max_raw_chars=get_max_raw_chars("video_script_reviewer"),
            )
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
    except Exception as exc:
        reason = (
            "LLM reviewer failed; local check passed, so video script is approved by deterministic fallback. "
            f"error={type(exc).__name__}: {exc}"
        )
        logger.warning("video_script_reviewer LLM fallback used: %s", exc)

    emit_a3_trace(
        logger,
        "video_script_reviewer",
        trace_payload("approve", reason, llm_fallback_used=True),
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "video_script_review_verdict": "approve",
        "video_script_review_reason": reason,
        "video_script_revision_notes": "",
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
        markdown = _fallback_video_script_markdown(
            state, "output received empty markdown"
        )
    local_check = _local_check_video_script(markdown, state)
    if not local_check["passed"]:
        markdown = _fallback_video_script_markdown(
            state,
            "generated video script failed local quality check; fallback was used.",
        )

    title = _extract_markdown_title(markdown)
    srt = _extract_srt_from_markdown(markdown)
    review_verdict = state.get("video_script_review_verdict", "")
    review_reason = state.get("video_script_review_reason", "")
    try:
        artifact = _create_video_script_artifact(markdown, title, srt)
    except Exception as exc:
        logger.exception("video_script_output failed to create artifact")
        artifact = {
            "title": title,
            "artifact_id": "",
            "markdown_url": "",
            "docx_url": "",
            "srt_url": "",
            "filename": "",
            "docx_filename": "",
            "srt_filename": "",
            "markdown": markdown,
            "srt": srt,
            "artifact_error": f"{type(exc).__name__}: {exc}",
        }
    artifact = {
        **(state.get("video_script_artifact") or {}),
        **artifact,
        "quality_warning": review_verdict not in {"", "approve"},
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
            "quality_warning": review_verdict not in {"", "approve"},
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
    if state.get("video_script_review_verdict") == "approve":
        return "output"
    current_round = int(state.get("video_script_round", 0) or 0)
    if current_round < 2:
        return "rewrite"
    return "output"
