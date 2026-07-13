"""Supervisor node — LLM-based intent classification, subject detection, and keypoint extraction.

Combines routing and academic keypoint extraction into a single LLM call
to eliminate a redundant API roundtrip on the academic path.
Uses structured output (Pydantic) instead of manual JSON parsing.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from src.context_engineering.workspace import (
    workspace_continuation_context,
    workspace_continuation_trace_payload,
)
from src.config import get_setting, load_prompt
from src.graph.resource_generation import (
    SUPPORTED_RESOURCE_TYPES,
    normalize_requested_resource_types as _normalize_supported_requested_resource_types,
)
from src.graph.capability_registry import get_registered_resource_types
from src.graph.qa import build_general_qa_node_output
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

    model_config = ConfigDict(extra="forbid")

    intent: Literal["academic", "emotional", "unknown"]
    response_mode: Literal["qa", "resource", "emotional"]
    qa_scope: Literal["academic", "general", "a3_agent", ""]
    requires_live_verification: bool
    keywords: list[str]
    confidence: float
    subject_candidates: list[str] = Field(default_factory=list)
    requested_resource_type: str = ""
    requested_resource_types: list[str] = Field(default_factory=list)


_VALID_INTENTS: set[str] = set()

_VALID_RESOURCE_TYPES = set(SUPPORTED_RESOURCE_TYPES)

_SUPERVISOR_RESOURCE_ALIASES = {
    "code_case": "code_practice",
    "project_case": "code_practice",
    "coding_practice": "code_practice",
    "code practice": "code_practice",
    "hands-on project": "code_practice",
    "hands_on project": "code_practice",
    "hands_on_project": "code_practice",
    "animation script": "video_script",
    "video script": "video_script",
    "storyboard": "video_script",
    "narration script": "video_script",
    "video animation": "video_animation",
    "animation video": "video_animation",
    "animation_video": "video_animation",
    "mp4": "video_animation",
    "mp4 video": "video_animation",
    "mp4_video": "video_animation",
    "render video": "video_animation",
    "render_video": "video_animation",
}


def _normalize_supervisor_resource_type(value: Any) -> str:
    """Normalize resource aliases known at supervisor time."""
    text = str(value or "").strip().lower().replace("-", "_")
    if not text:
        return ""
    text = _SUPERVISOR_RESOURCE_ALIASES.get(text, text)
    if text in {"code_practice", "video_script", "video_animation"}:
        return text
    supported = _normalize_supported_requested_resource_types(text)
    return supported[0] if supported else ""


def normalize_requested_resource_types(*values: Any) -> list[str]:
    """Return ordered, deduplicated resource types accepted by the supervisor."""
    normalized: list[str] = []

    def add_one(item: Any) -> None:
        resource_type = _normalize_supervisor_resource_type(item)
        if resource_type and resource_type not in normalized:
            normalized.append(resource_type)

    for value in values:
        if isinstance(value, (list, tuple, set)):
            for item in value:
                add_one(item)
        else:
            add_one(value)
    return normalized


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
    resource_types = normalize_requested_resource_types(
        parsed.requested_resource_types,
        parsed.requested_resource_type,
    )
    raw_resource_values = [
        str(item or "").strip()
        for item in [
            parsed.requested_resource_type,
            *(parsed.requested_resource_types or []),
        ]
        if str(item or "").strip()
    ]
    invalid_resources = [
        item
        for item in raw_resource_values
        if item not in _VALID_RESOURCE_TYPES
        and not normalize_requested_resource_types(item)
    ]
    if invalid_resources:
        return f"invalid requested_resource_types: {invalid_resources}"
    if parsed.response_mode == "resource":
        if parsed.intent != "academic":
            return "resource response_mode requires academic intent"
        if not resource_types:
            return "resource response_mode requires requested_resource_types"
        if parsed.qa_scope:
            return "resource response_mode requires empty qa_scope"
    elif parsed.response_mode == "emotional":
        if parsed.intent != "emotional":
            return "emotional response_mode requires emotional intent"
        if resource_types or parsed.qa_scope:
            return "emotional response_mode may not carry resources or qa_scope"
        if parsed.requires_live_verification:
            return "emotional response_mode may not require live verification"
    elif parsed.response_mode == "qa":
        if resource_types:
            return "qa response_mode may not carry requested_resource_types"
        if parsed.qa_scope not in {"academic", "general", "a3_agent"}:
            return "qa response_mode requires a non-empty valid qa_scope"
        if parsed.qa_scope == "academic" and parsed.intent != "academic":
            return "academic qa_scope requires academic intent"
        if parsed.qa_scope in {"general", "a3_agent"} and parsed.intent not in {
            "academic",
            "unknown",
        }:
            return "general and a3_agent qa_scope may not use emotional intent"
    else:
        return "response_mode is invalid"
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
    available_resource_types_text = "\n".join(
        f"- {resource_type}" for resource_type in get_registered_resource_types()
    )
    user_message = (
        "## 当前知识库 available subjects\n"
        f"{available_subjects_text}\n\n"
        "## Available resource types\n"
        f"{available_resource_types_text}\n\n"
        "## 用户输入\n"
        f"{user_text}"
    )

    temperature = get_setting("supervisor.temperature", 0.0)
    model_name = get_setting(
        "llm.supervisor.model", get_setting("supervisor.model", "")
    )
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
    response_mode = result.response_mode
    qa_scope = result.qa_scope
    requires_live_verification = result.requires_live_verification
    keypoints = result.keywords
    subject_candidates = _filter_subject_candidates(
        result.subject_candidates,
        available_subject_set,
    )
    subject = subject_candidates[0] if subject_candidates else "other"

    requested_resource_types = normalize_requested_resource_types(
        result.requested_resource_types,
        result.requested_resource_type,
    )
    requested_resource_type = (
        requested_resource_types[0] if requested_resource_types else ""
    )
    is_parallel_resource_request = len(requested_resource_types) > 1
    needs_mindmap = (
        requested_resource_type == "mindmap" or "mindmap" in requested_resource_types
    )
    continuation_state = {
        **state,
        "subject": subject,
        "subject_candidates": subject_candidates,
        "requested_resource_type": requested_resource_type,
        "requested_resource_types": requested_resource_types,
    }
    workspace_continuation = workspace_continuation_context(continuation_state)
    workspace_continuation_applied = False
    if workspace_continuation.get("can_continue"):
        continuation_subject = str(
            workspace_continuation.get("normalized_subject")
            or workspace_continuation.get("active_subject")
            or ""
        ).strip()
        if available_subject_set and continuation_subject not in available_subject_set:
            workspace_continuation["can_continue"] = False
            workspace_continuation["skip_reason"] = "subject_unavailable"
        elif continuation_subject:
            subject = continuation_subject
            workspace_continuation["continuation_applied"] = True
            workspace_continuation_applied = True

    continuation_payload = workspace_continuation_trace_payload(workspace_continuation)
    emit_a3_trace(
        logger,
        "task_workspace.continuation_checked",
        continuation_payload,
        state=state,
        env_flag="LOG_A3_TRACE",
        max_chars=200,
    )
    if workspace_continuation.get("workspace_id") or requested_resource_types:
        emit_a3_trace(
            logger,
            "task_workspace.continuation_applied"
            if workspace_continuation_applied
            else "task_workspace.continuation_skipped",
            continuation_payload,
            state=state,
            env_flag="LOG_A3_TRACE",
            max_chars=200,
        )

    # TEMP A3_TRACE: remove after multi-subject retrieval validation.
    emit_a3_trace(
        logger,
        "supervisor",
        {
            "intent": intent,
            "response_mode": response_mode,
            "qa_scope": qa_scope,
            "requires_live_verification": requires_live_verification,
            "subject": subject,
            "subject_candidates": subject_candidates,
            "keypoints": keypoints,
            "requested_resource_type": requested_resource_type,
            "requested_resource_types": requested_resource_types,
            "is_parallel_resource_request": is_parallel_resource_request,
            "needs_mindmap": needs_mindmap,
            "workspace_continuation_applied": workspace_continuation_applied,
            "workspace_continuation_reason": workspace_continuation.get(
                "skip_reason", ""
            ),
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
        "response_mode": response_mode,
        "qa_scope": qa_scope,
        "requires_live_verification": requires_live_verification,
        "subject": subject,
        "subject_candidates": subject_candidates,
        "keypoints": keypoints,
        "requested_resource_type": requested_resource_type,
        "requested_resource_types": requested_resource_types,
        "is_parallel_resource_request": is_parallel_resource_request,
        "needs_mindmap": needs_mindmap,
        "workspace_continuation": workspace_continuation,
        "workspace_continuation_applied": workspace_continuation_applied,
        "workspace_continuation_reason": workspace_continuation.get("skip_reason", ""),
    }


def _filter_subject_candidates(
    candidates: list[str], available_subjects: set[str]
) -> list[str]:
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
    result = {
        "messages": [
            AIMessage(
                content=(
                    "抱歉，这个问题超出了我的辅导范围。我是你的高校学习助手，"
                    "可以帮你探索专业方向、解答课程知识、制定学习路径、生成学习资源，"
                    "或者聊聊学习中的困惑。请问有什么需要帮助的吗？"
                ),
            )
        ],
    }
    return build_general_qa_node_output(
        answer=str(result["messages"][0].content),
        state=state,
    )


def route_by_intent(state: LearningState) -> str:
    """Conditional edge function: route to the appropriate subgraph."""
    intent = state.get("intent", "academic")
    if intent not in ("academic", "emotional", "unknown"):
        intent = "unknown"
    return intent


def route_after_supervisor(state: LearningState) -> str:
    """Route the strict supervisor response contract without phrase heuristics."""
    response_mode = str(state.get("response_mode") or "").strip()
    qa_scope = str(state.get("qa_scope") or "").strip()
    intent = str(state.get("intent") or "").strip()
    if response_mode == "emotional" and intent == "emotional":
        return "emotional"
    if response_mode == "resource" and intent == "academic":
        return "academic"
    if response_mode == "qa" and qa_scope == "academic" and intent == "academic":
        return "academic"
    if response_mode == "qa" and qa_scope in {"general", "a3_agent"}:
        return "qa"
    return "invalid"


_READABLE_RESOURCE_ACTION_MARKERS = (
    "生成",
    "制作",
    "创建",
    "导出",
    "整理",
    "汇总",
    "总结",
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
    "give me",
)

_CODE_PRACTICE_ACTION_MARKERS = (
    "生成",
    "给我",
    "帮我做",
    "帮我生成",
    "创建",
    "输出",
    "制作",
    "generate",
    "give me",
    "create",
    "make",
)

_READABLE_WEAK_REQUEST_MARKERS = (
    "帮我",
    "给我",
    "我要",
    "我想要",
    "我还想要",
    "请",
)

_READABLE_EXPLANATION_MARKERS = (
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
    "what is",
)

_READABLE_RESOURCE_TYPE_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "mindmap",
        (
            "思维导图",
            "知识图谱",
            "脑图",
            "结构图",
            "mindmap",
            "mind map",
            "markmap",
            "xmind",
        ),
    ),
    (
        "quiz",
        (
            "练习题",
            "分层练习",
            "题库",
            "习题",
            "测验",
            "测试题",
            "题目",
            "quiz",
            "exercise",
            "practice questions",
        ),
    ),
    (
        "code_practice",
        (
            "代码实操",
            "代码案例",
            "实操案例",
            "编程实战",
            "项目实战",
            "项目案例",
            "完整代码",
            "可运行代码",
            "代码练习",
            "coding practice",
            "code practice",
            "hands-on project",
        ),
    ),
    (
        "video_animation",
        (
            "教学动画",
            "动画视频",
            "mp4",
            "真实视频",
            "生成视频",
            "教学视频 mp4",
            "动画演示",
            "可播放动画",
            "video animation",
            "animation video",
            "mp4 video",
            "render video",
        ),
    ),
    (
        "video_script",
        (
            "视频脚本",
            "动画脚本",
            "分镜脚本",
            "旁白文案",
            "字幕脚本",
            "视频分镜",
            "animation script",
            "video script",
            "storyboard",
            "narration script",
        ),
    ),
    (
        "review_doc",
        (
            "复习资料",
            "复习文档",
            "学习资料",
            "课程讲解文档",
            "讲义",
            "知识点整理",
            "复习笔记",
            "课程文档",
            "review doc",
            "review document",
        ),
    ),
    (
        "study_plan",
        (
            "学习计划",
            "学习路径",
            "学习路线",
            "入门路线",
            "怎么学习",
            "如何学习",
            "怎么安排",
            "学习规划",
            "学习方案",
            "study plan",
            "learning path",
            "roadmap",
        ),
    ),
)


def _detect_requested_resource_types(text: str) -> list[str]:
    """Deterministically identify explicit single or multi-resource requests."""
    lowered = str(text or "").lower()
    asks_explanation = any(
        marker.lower() in lowered for marker in _READABLE_EXPLANATION_MARKERS
    )
    has_action = any(
        marker.lower() in lowered for marker in _READABLE_RESOURCE_ACTION_MARKERS
    )
    has_weak_request = any(
        marker.lower() in lowered for marker in _READABLE_WEAK_REQUEST_MARKERS
    )
    has_code_practice_action = any(
        marker.lower() in lowered for marker in _CODE_PRACTICE_ACTION_MARKERS
    )

    detected: list[tuple[int, str]] = []
    for resource_type, markers in _READABLE_RESOURCE_TYPE_MARKERS:
        positions = [
            lowered.find(marker.lower())
            for marker in markers
            if marker.lower() in lowered
        ]
        if positions:
            if resource_type == "code_practice" and not has_code_practice_action:
                continue
            detected.append((min(positions), resource_type))

    if not detected:
        return []

    has_study_plan = any(resource_type == "study_plan" for _, resource_type in detected)
    if (
        not has_study_plan
        and not has_action
        and (not has_weak_request or asks_explanation)
    ):
        return []
    if asks_explanation and not has_action and not has_weak_request:
        return []

    ordered: list[str] = []
    for _, resource_type in sorted(detected, key=lambda item: item[0]):
        if resource_type not in ordered:
            ordered.append(resource_type)
    return ordered


def _detect_requested_resource_type(text: str) -> str:
    """Backward-compatible single resource detector."""
    resources = _detect_requested_resource_types(text)
    return resources[0] if resources else ""
