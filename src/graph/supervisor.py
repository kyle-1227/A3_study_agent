"""Supervisor node — LLM-based intent classification, subject detection, and keypoint extraction.

Combines routing and academic keypoint extraction into a single LLM call
to eliminate a redundant API roundtrip on the academic path.
Uses structured output (Pydantic) instead of manual JSON parsing.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from src.config import get_setting, load_prompt
from src.graph.state import LearningState
from src.llm.structured_output import (
    get_fallback_modes,
    get_llm_output_mode,
    get_max_raw_chars,
    invoke_structured_llm,
)
from src.rag.course_catalog import get_available_subjects_from_data, normalize_subject
from src.observability.a3_trace import emit_a3_trace
from src.tracing import traced_llm_call, traced_node

logger = logging.getLogger(__name__)


class SupervisorOutput(BaseModel):
    """Structured output for supervisor intent classification."""
    intent: Literal["academic", "emotional", "unknown"]
    keywords: list[str]
    confidence: float
    subject_candidates: list[str] = []
    requested_resource_type: str = ""
    requested_resource_types: list[str] = []


_VALID_INTENTS: set[str] = set()

_VALID_RESOURCE_TYPES = frozenset({"study_plan", "mindmap", "quiz", "review_doc", "multi_resource"})
_VALID_INDIVIDUAL_RESOURCE_TYPES = frozenset({"study_plan", "mindmap", "quiz", "review_doc"})


def _sanitize_valid_intents() -> set[str]:
    """Sanitize supervisor.valid_intents config — planning is no longer legal."""
    configured = get_setting(
        "supervisor.valid_intents",
        ["academic", "emotional", "unknown"],
    )
    if not isinstance(configured, list):
        configured = ["academic", "emotional", "unknown"]
    removed = [i for i in configured if i == "planning"]
    sanitized = [i for i in configured if i != "planning"]
    if removed:
        logger.warning(
            "supervisor.valid_intents contains 'planning' — sanitized. "
            "Removed intents: %s. Effective intents: %s",
            removed,
            sanitized,
        )
        emit_a3_trace(
            logger,
            "supervisor_config_sanitize",
            {
                "configured": configured,
                "removed_intents": removed,
                "effective_intents": sanitized,
            },
            state={},
            env_flag="LOG_A3_TRACE",
        )
    return set(sanitized)


_VALID_INTENTS = _sanitize_valid_intents()


def validate_supervisor_output(parsed: BaseModel) -> str:
    """Business validation for supervisor structured routing."""
    if not isinstance(parsed, SupervisorOutput):
        return "root expected SupervisorOutput"
    if parsed.intent not in _VALID_INTENTS:
        return f"intent invalid: {parsed.intent}"
    if not 0 <= float(parsed.confidence) <= 1:
        return "confidence must be between 0 and 1"
    if not isinstance(parsed.keywords, list):
        return "keywords must be a list"
    if not isinstance(parsed.subject_candidates, list):
        return "subject_candidates must be a list"
    if not isinstance(parsed.requested_resource_types, list):
        return "requested_resource_types must be a list"
    # ── Intent/resource combination validation ─────────────────────
    resource_type = (parsed.requested_resource_type or "").strip()
    resource_types = [str(item).strip() for item in parsed.requested_resource_types or [] if str(item).strip()]
    has_resource = bool(resource_type or resource_types)
    if has_resource:
        if parsed.intent in ("emotional", "unknown"):
            if resource_type in _VALID_RESOURCE_TYPES or any(item in _VALID_INDIVIDUAL_RESOURCE_TYPES for item in resource_types):
                return (
                    f"intent={parsed.intent} may not carry "
                    f"requested_resource_type={resource_type}. "
                    f"Only academic intent supports resource generation."
                )
    return ""


@traced_node
async def supervisor_node(state: LearningState) -> dict:
    """Classify intent, detect subject, and extract keypoints in one LLM call.

    Uses the fail-fast structured-output runtime for reliable parsing.

    Returns:
        Dict with ``intent``, ``subject``, and ``keypoints`` for state update.
    """
    last_msg = state["messages"][-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    available_subjects = get_available_subjects_from_data()
    available_subject_set = set(available_subjects)
    available_subjects_text = (
        "\n".join(f"- {subject}" for subject in available_subjects)
        if available_subjects
        else "当前 data/ 目录下未发现可用课程 subject。"
    )
    user_message = (
        "## 当前知识库 available subjects\n"
        f"{available_subjects_text}\n\n"
        "## 用户输入\n"
        f"{user_text}"
    )

    temperature = get_setting("supervisor.temperature", 0.0)
    model_name = get_setting("supervisor.model", os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    with traced_llm_call(
        model_name=model_name,
        node_name="supervisor",
        temperature=temperature,
    ):
        structured_result = await invoke_structured_llm(
            node_name="supervisor",
            llm_node="supervisor",
            schema=SupervisorOutput,
            messages=[
                SystemMessage(content=load_prompt("supervisor_system")),
                HumanMessage(content=user_message),
            ],
            output_mode=get_llm_output_mode("supervisor"),
            fallback_modes=get_fallback_modes("supervisor"),
            business_validator=validate_supervisor_output,
            state=state,
            max_raw_chars=get_max_raw_chars("supervisor"),
        )
    result = structured_result.parsed
    if not isinstance(result, SupervisorOutput):
        raise TypeError("supervisor parsed result is not SupervisorOutput")
    intent = result.intent
    keypoints = result.keywords
    subject_candidates = _filter_subject_candidates(
        result.subject_candidates,
        available_subject_set,
    )
    subject = subject_candidates[0] if subject_candidates else "other"

    # Deterministic resource detection has priority over LLM output.
    llm_resource = (result.requested_resource_type or "").strip()
    llm_resource_types = _normalize_requested_resource_types(
        result.requested_resource_types or ([llm_resource] if llm_resource else [])
    )
    deterministic_resource_types = _detect_requested_resource_types(user_text)
    if len(deterministic_resource_types) > 1:
        requested_resource_type = "multi_resource"
        requested_resource_types = deterministic_resource_types
    elif len(deterministic_resource_types) == 1:
        requested_resource_type = deterministic_resource_types[0]
        requested_resource_types = deterministic_resource_types
    elif len(llm_resource_types) > 1:
        requested_resource_type = "multi_resource"
        requested_resource_types = llm_resource_types
    elif len(llm_resource_types) == 1:
        requested_resource_type = llm_resource_types[0]
        requested_resource_types = llm_resource_types
    elif llm_resource in _VALID_RESOURCE_TYPES:
        requested_resource_type = llm_resource
        requested_resource_types = [] if llm_resource == "multi_resource" else [llm_resource]
    else:
        requested_resource_type = ""
        requested_resource_types = []
    multi_resource_mode = requested_resource_type == "multi_resource"

    # academic intent with resource type stays academic
    # emotional/unknown with resource type was already blocked by validation
    needs_mindmap = requested_resource_type == "mindmap" or "mindmap" in requested_resource_types

    # TEMP A3_TRACE: remove after multi-subject retrieval validation.
    emit_a3_trace(
        logger,
        "supervisor",
        {
            "intent": intent,
            "subject": subject,
            "subject_candidates": subject_candidates,
            "keypoints": keypoints,
            "requested_resource_type": requested_resource_type,
            "requested_resource_types": requested_resource_types,
            "multi_resource_mode": multi_resource_mode,
            "needs_mindmap": needs_mindmap,
            "confidence": result.confidence if "result" in locals() else 0.0,
            "available_subjects": available_subjects,
            "user_query_preview": user_text,
        },
        state=state,
        env_flag="LOG_SUPERVISOR_RESULT",
        max_chars=200,
    )

    return {
        "intent": intent,
        "subject": subject,
        "subject_candidates": subject_candidates,
        "keypoints": keypoints,
        "requested_resource_type": requested_resource_type,
        "requested_resource_types": requested_resource_types,
        "multi_resource_mode": multi_resource_mode,
        "multi_resource_results": [],
        "multi_resource_summary": "",
        "needs_mindmap": needs_mindmap,
    }


def _filter_subject_candidates(candidates: list[str], available_subjects: set[str]) -> list[str]:
    """Keep normalized subject candidates that exist in the current course catalog."""
    filtered: list[str] = []
    for candidate in candidates or []:
        subject = normalize_subject(str(candidate))
        if subject and subject in available_subjects and subject not in filtered:
            filtered.append(subject)
    return filtered


@traced_node
async def handle_unknown(state: LearningState) -> dict:
    """Handle off-topic queries with a friendly redirect message."""
    return {
        "messages": [AIMessage(
            content=(
                "抱歉，这个问题超出了我的辅导范围。我是你的高校学习助手，"
                "可以帮你探索专业方向、解答课程知识、制定学习路径、生成学习资源，"
                "或者聊聊学习中的困惑。请问有什么需要帮助的吗？"
            ),
        )],
    }


def route_by_intent(state: LearningState) -> str:
    """Conditional edge function: route to the appropriate subgraph."""
    intent = state.get("intent", "academic")
    if intent not in ("academic", "emotional", "unknown"):
        intent = "unknown"
    return intent


_RESOURCE_ACTION_MARKERS = (
    "生成",
    "给我",
    "帮我",
    "制作",
    "创建",
    "导出",
    "整理",
    "整理一份",
    "整理成",
    "汇总",
    "总结成",
    "转成",
    "输出",
    "来一份",
    "做一个",
    "做一份",
    "画一个",
    "画一份",
    "设计一份",
    "帮我做",
    "帮我生成",
    "给我做",
    "给我生成",
    "generate",
    "create",
    "export",
    "make",
)

_WEAK_REQUEST_MARKERS = ("帮我", "给我", "我要", "我想要", "请")

_EXPLANATION_MARKERS = (
    "是什么",
    "什么是",
    "怎么理解",
    "如何理解",
    "为什么",
    "讲讲",
    "解释",
    "介绍",
    "原理",
    "区别",
    "作用",
    "用途",
    "有什么用",
    "有啥用",
    "应该怎么",
    "怎么整理",
    "如何整理",
)

_RESOURCE_GENERATION_ACTION_MARKERS = (
    "生成",
    "给我",
    "帮我",
    "做一份",
    "整理一份",
    "输出",
    "制作",
    "创建",
    "来一份",
    "画一个",
    "画一份",
    "设计一份",
    "generate",
    "create",
    "make",
)

_RESOURCE_TYPE_MARKERS_FOR_DETECTION: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "review_doc",
        (
            "复习资料",
            "复习文档",
            "学习资料",
            "学习文档",
            "知识点整理",
            "知识整理",
            "课程资料",
            "讲义",
            "笔记",
            "期末复习",
            "考前复习",
        ),
    ),
    ("mindmap", ("思维导图", "脑图", "知识图谱", "mindmap", "xmind")),
    ("quiz", ("练习题", "习题", "题库", "测试题", "测验", "quiz", "exercises")),
    ("study_plan", ("学习计划", "学习路径", "学习路线", "roadmap")),
)

_RESOURCE_TYPE_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mindmap", ("思维导图", "知识图谱", "脑图", "结构图", "mindmap", "mind map", "markmap", "xmind")),
    ("quiz", ("练习题", "分层练习", "题库", "习题", "测验", "测试题", "题目")),
    ("ppt", ("ppt", "幻灯片", "演示文稿", "课件")),
    ("code_case", ("代码案例", "代码实操", "实操案例", "编程案例", "示例代码", "demo")),
    ("project_case", ("项目案例", "实践项目", "项目实战", "课程项目", "实验项目")),
    ("video_script", ("视频脚本", "动画脚本", "讲解视频", "教学视频", "分镜")),
    ("review_doc", ("复习资料", "复习文档", "学习文档", "学习材料", "考试讲义", "复习讲义", "知识整理", "知识点整理", "章节复习", "期末复习")),
    ("study_plan", ("学习计划", "学习路径", "学习路线", "入门路线", "怎么学习", "如何学习", "怎么安排", "学习规划", "学习方案", "study plan", "learning path", "roadmap")),
    ("reading", ("拓展阅读", "阅读材料", "参考资料", "文献清单", "资料清单")),
    ("volunteer", ("志愿填报", "高考志愿", "填报志愿", "志愿", "择校", "选专业", "分数线", "院校推荐", "专业推荐")),
    ("other", ("讲义", "学习资源", "资源清单", "知识卡片")),
)


def _normalize_requested_resource_types(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        resource_type = str(value or "").strip()
        if resource_type == "multi_resource":
            continue
        if resource_type in _VALID_INDIVIDUAL_RESOURCE_TYPES and resource_type not in normalized:
            normalized.append(resource_type)
    return normalized


def _detect_requested_resource_types(text: str) -> list[str]:
    """Deterministically identify explicit one-turn resource requests."""
    lowered = text.lower()
    has_strong_action = any(marker.lower() in lowered for marker in _RESOURCE_ACTION_MARKERS)
    has_weak_request = any(marker.lower() in lowered for marker in _WEAK_REQUEST_MARKERS)
    asks_explanation = any(marker.lower() in lowered for marker in _EXPLANATION_MARKERS)
    has_generation_action = any(marker.lower() in lowered for marker in _RESOURCE_GENERATION_ACTION_MARKERS)

    if asks_explanation and not has_generation_action:
        return []
    if not has_strong_action and not has_weak_request:
        return []

    detected: list[str] = []
    for resource_type, markers in _RESOURCE_TYPE_MARKERS_FOR_DETECTION:
        if any(marker.lower() in lowered for marker in markers):
            detected.append(resource_type)
    return detected


def _detect_requested_resource_type(text: str) -> str:
    """Backward-compatible single-resource detector."""
    detected = _detect_requested_resource_types(text)
    if len(detected) > 1:
        return "multi_resource"
    return detected[0] if detected else ""
