"""Parallel learning-resource generation orchestration.

The graph-level resource node uses LangGraph dynamic fan-out/fan-in while each
worker reuses the existing resource-generation node functions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage
from langgraph.types import Send

from src.graph.code_practice import (
    code_practice_agent,
    code_practice_output,
    code_practice_planner,
    code_practice_reviewer,
    code_practice_rewrite,
    should_rewrite_code_practice,
)
from src.graph.exercises import (
    exercise_agent,
    exercise_output,
    exercise_planner,
    exercise_reviewer,
    exercise_rewrite,
    should_rewrite_exercise,
)
from src.graph.mindmap import (
    mindmap_agent,
    mindmap_output,
    mindmap_planner,
    mindmap_reviewer,
    mindmap_rewrite,
    should_rewrite_mindmap,
)
from src.graph.review_doc import (
    review_doc_agent,
    review_doc_output,
    review_doc_planner,
    review_doc_reviewer,
    review_doc_rewrite,
    should_rewrite_review_doc,
)
from src.graph.state import LearningState, RESOURCE_RESULTS_CLEAR
from src.graph.study_plan import (
    route_after_study_plan_consensus,
    study_plan_agent,
    study_plan_consensus,
    study_plan_emotional_intel,
    study_plan_output,
    study_plan_planner,
    study_plan_reviewer_academic,
    study_plan_reviewer_emotional,
    study_plan_rewrite,
)
from src.graph.video_animation import (
    should_rewrite_video_animation,
    video_animation_agent,
    video_animation_output,
    video_animation_planner,
    video_animation_reviewer,
    video_animation_rewrite,
)
from src.graph.video_script import (
    should_rewrite_video_script,
    video_script_agent,
    video_script_output,
    video_script_planner,
    video_script_reviewer,
    video_script_rewrite,
)
from src.observability.a3_trace import emit_a3_trace
from src.tools.search_tool import sanitize_error_message
from src.tracing import traced_node

logger = logging.getLogger(__name__)

RESOURCE_TYPE_ORDER = (
    "review_doc",
    "mindmap",
    "quiz",
    "code_practice",
    "video_script",
    "video_animation",
    "study_plan",
)
SUPPORTED_RESOURCE_TYPES = frozenset(RESOURCE_TYPE_ORDER)
RESOURCE_ALIASES = {
    "exercise": "quiz",
    "exercises": "quiz",
    "practice": "quiz",
    "practice_questions": "quiz",
    "review": "review_doc",
    "review_document": "review_doc",
    "doc": "review_doc",
    "document": "review_doc",
    "learning_plan": "study_plan",
    "roadmap": "study_plan",
    "mind_map": "mindmap",
    "xmind": "mindmap",
    "code": "code_practice",
    "coding_practice": "code_practice",
    "video": "video_animation",
    "animation": "video_animation",
    "video_animation": "video_animation",
    "video_script": "video_script",
}

RESOURCE_OUTPUT_STATE_KEYS: dict[str, tuple[str, ...]] = {
    "mindmap": (
        "mindmap_outline",
        "mindmap_tree",
        "mindmap_artifact",
        "mindmap_review_verdict",
        "mindmap_review_reason",
        "mindmap_revision_notes",
        "mindmap_round",
    ),
    "quiz": (
        "exercise_outline",
        "exercise_items",
        "exercise_artifact",
        "exercise_review_verdict",
        "exercise_review_reason",
        "exercise_revision_notes",
        "exercise_round",
    ),
    "review_doc": (
        "review_doc_outline",
        "review_doc_markdown",
        "review_doc_markdowns",
        "review_doc_artifact",
        "review_doc_artifacts",
        "review_doc_review_verdict",
        "review_doc_review_reason",
        "review_doc_revision_notes",
        "review_doc_round",
    ),
    "study_plan": (
        "study_plan_emotional_intel",
        "study_plan_emotional_profile",
        "study_plan_outline",
        "study_plan_artifact",
        "study_plan_markdown",
        "study_plan_round",
        "study_plan_academic_verdict",
        "study_plan_academic_reason",
        "study_plan_emotional_verdict",
        "study_plan_emotional_reason",
        "study_plan_consensus",
        "study_plan_revision_notes",
        "study_plan_document_artifact",
    ),
    "code_practice": (
        "code_practice_outline",
        "code_practice_markdown",
        "code_practice_artifact",
        "code_practice_review_verdict",
        "code_practice_review_reason",
        "code_practice_revision_notes",
        "code_practice_round",
    ),
    "video_script": (
        "video_script_outline",
        "video_script_markdown",
        "video_script_srt",
        "video_script_artifact",
        "video_script_review_verdict",
        "video_script_review_reason",
        "video_script_revision_notes",
        "video_script_round",
    ),
    "video_animation": (
        "video_animation_spec",
        "video_animation_html",
        "video_animation_artifact",
        "video_animation_review_verdict",
        "video_animation_review_reason",
        "video_animation_revision_notes",
        "video_animation_round",
        "video_animation_render_log",
    ),
}


def normalize_resource_type(value: Any) -> str:
    """Normalize public resource aliases to the graph's canonical resource type."""
    text = str(value or "").strip().lower().replace("-", "_")
    if not text:
        return ""
    text = RESOURCE_ALIASES.get(text, text)
    return text if text in SUPPORTED_RESOURCE_TYPES else ""


def normalize_requested_resource_types(*values: Any) -> list[str]:
    """Return ordered, deduplicated canonical resource types."""
    normalized: list[str] = []

    def add_one(item: Any) -> None:
        resource_type = normalize_resource_type(item)
        if resource_type and resource_type not in normalized:
            normalized.append(resource_type)

    for value in values:
        if isinstance(value, (list, tuple, set)):
            for item in value:
                add_one(item)
        else:
            add_one(value)
    return normalized


def _resource_plan_from_state(state: LearningState) -> list[dict]:
    resources = normalize_requested_resource_types(
        state.get("requested_resource_types") or [],
        state.get("requested_resource_type") or "",
    )
    return [
        {
            "task_id": f"resource:{resource_type}",
            "resource_type": resource_type,
            "status": "pending",
        }
        for resource_type in resources
    ]


def _debug_base(state: LearningState, tasks: list[dict]) -> dict:
    run_id = str(state.get("request_id") or uuid4())
    resource_types = [task["resource_type"] for task in tasks]
    return {
        "run_id": run_id,
        "status": "running" if tasks else "skipped",
        "selected_resource_types": resource_types,
        "success_count": 0,
        "failed_count": 0,
        "partial_success": False,
        "developer_warnings": [],
        "stages": [
            {
                "stage": "resource_generation.orchestrator.start",
                "status": "success" if tasks else "skipped",
                "selected_resource_types": resource_types,
                "task_count": len(tasks),
            }
        ],
    }


@traced_node
async def resource_orchestrator(state: LearningState) -> dict:
    """Plan resource worker tasks after Evidence Judge V2 has approved context."""
    tasks = _resource_plan_from_state(state)
    resource_types = [task["resource_type"] for task in tasks]
    debug = _debug_base(state, tasks)
    debug["stages"].append(
        {
            "stage": "resource_generation.orchestrator.success",
            "status": "success" if tasks else "skipped",
            "task_count": len(tasks),
            "selected_resource_types": resource_types,
        }
    )
    emit_a3_trace(
        logger,
        "resource_generation.orchestrator.success",
        {
            "task_count": len(tasks),
            "selected_resource_types": resource_types,
            "status": debug["status"],
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "requested_resource_type": resource_types[0] if resource_types else "",
        "requested_resource_types": resource_types,
        "resource_generation_plan": {"tasks": tasks},
        "resource_branch_results": RESOURCE_RESULTS_CLEAR,
        "resource_bundle_artifact": {},
        "resource_generation_debug": debug,
        "resource_generation_status": "running" if tasks else "skipped",
    }


def dispatch_resource_workers(state: LearningState) -> list[Send]:
    """Dynamically fan out one worker per planned resource task."""
    tasks = (state.get("resource_generation_plan") or {}).get("tasks") or []
    if not tasks:
        return [Send("resource_bundle_output", dict(state))]
    return [
        Send("resource_worker", {**dict(state), "resource_task": task})
        for task in tasks
    ]


def _merge_node_output(local_state: dict, output: dict | None) -> str:
    if not output:
        return ""
    message_content = ""
    for message in output.get("messages") or []:
        if isinstance(message, AIMessage):
            message_content = str(message.content or "")
        elif hasattr(message, "content"):
            message_content = str(getattr(message, "content") or "")
    for key, value in output.items():
        if key != "messages":
            local_state[key] = value
    return message_content


async def _run_mindmap_resource(local_state: dict) -> str:
    _merge_node_output(local_state, await mindmap_planner(local_state))
    _merge_node_output(local_state, await mindmap_agent(local_state))
    _merge_node_output(local_state, await mindmap_reviewer(local_state))
    while should_rewrite_mindmap(local_state) == "rewrite":
        _merge_node_output(local_state, await mindmap_rewrite(local_state))
        _merge_node_output(local_state, await mindmap_agent(local_state))
        _merge_node_output(local_state, await mindmap_reviewer(local_state))
    return _merge_node_output(local_state, await mindmap_output(local_state))


async def _run_quiz_resource(local_state: dict) -> str:
    _merge_node_output(local_state, await exercise_planner(local_state))
    _merge_node_output(local_state, await exercise_agent(local_state))
    _merge_node_output(local_state, await exercise_reviewer(local_state))
    while should_rewrite_exercise(local_state) == "rewrite":
        _merge_node_output(local_state, await exercise_rewrite(local_state))
        _merge_node_output(local_state, await exercise_agent(local_state))
        _merge_node_output(local_state, await exercise_reviewer(local_state))
    return _merge_node_output(local_state, await exercise_output(local_state))


async def _run_review_doc_resource(local_state: dict) -> str:
    _merge_node_output(local_state, await review_doc_planner(local_state))
    _merge_node_output(local_state, await review_doc_agent(local_state))
    _merge_node_output(local_state, await review_doc_reviewer(local_state))
    while should_rewrite_review_doc(local_state) == "rewrite":
        _merge_node_output(local_state, await review_doc_rewrite(local_state))
        _merge_node_output(local_state, await review_doc_agent(local_state))
        _merge_node_output(local_state, await review_doc_reviewer(local_state))
    return _merge_node_output(local_state, await review_doc_output(local_state))


async def _run_study_plan_resource(local_state: dict) -> str:
    _merge_node_output(local_state, await study_plan_emotional_intel(local_state))
    _merge_node_output(local_state, await study_plan_planner(local_state))
    while True:
        _merge_node_output(local_state, await study_plan_agent(local_state))
        academic_update, emotional_update = await asyncio.gather(
            study_plan_reviewer_academic(local_state),
            study_plan_reviewer_emotional(local_state),
        )
        _merge_node_output(local_state, academic_update)
        _merge_node_output(local_state, emotional_update)
        _merge_node_output(local_state, await study_plan_consensus(local_state))
        if route_after_study_plan_consensus(local_state) == "output":
            return _merge_node_output(local_state, await study_plan_output(local_state))
        _merge_node_output(local_state, await study_plan_rewrite(local_state))


async def _run_code_practice_resource(local_state: dict) -> str:
    _merge_node_output(local_state, await code_practice_planner(local_state))
    _merge_node_output(local_state, await code_practice_agent(local_state))
    _merge_node_output(local_state, await code_practice_reviewer(local_state))
    while should_rewrite_code_practice(local_state) == "rewrite":
        _merge_node_output(local_state, await code_practice_rewrite(local_state))
        _merge_node_output(local_state, await code_practice_agent(local_state))
        _merge_node_output(local_state, await code_practice_reviewer(local_state))
    return _merge_node_output(local_state, await code_practice_output(local_state))


async def _run_video_script_resource(local_state: dict) -> str:
    _merge_node_output(local_state, await video_script_planner(local_state))
    _merge_node_output(local_state, await video_script_agent(local_state))
    _merge_node_output(local_state, await video_script_reviewer(local_state))
    while should_rewrite_video_script(local_state) == "rewrite":
        _merge_node_output(local_state, await video_script_rewrite(local_state))
        _merge_node_output(local_state, await video_script_agent(local_state))
        _merge_node_output(local_state, await video_script_reviewer(local_state))
    return _merge_node_output(local_state, await video_script_output(local_state))


async def _run_video_animation_resource(local_state: dict) -> str:
    _merge_node_output(local_state, await video_animation_planner(local_state))
    _merge_node_output(local_state, await video_animation_agent(local_state))
    _merge_node_output(local_state, await video_animation_reviewer(local_state))
    while should_rewrite_video_animation(local_state) == "rewrite":
        _merge_node_output(local_state, await video_animation_rewrite(local_state))
        _merge_node_output(local_state, await video_animation_agent(local_state))
        _merge_node_output(local_state, await video_animation_reviewer(local_state))
    return _merge_node_output(local_state, await video_animation_output(local_state))


RESOURCE_RUNNERS = {
    "mindmap": _run_mindmap_resource,
    "quiz": _run_quiz_resource,
    "review_doc": _run_review_doc_resource,
    "code_practice": _run_code_practice_resource,
    "video_script": _run_video_script_resource,
    "video_animation": _run_video_animation_resource,
    "study_plan": _run_study_plan_resource,
}


def _state_updates_for_resource(resource_type: str, local_state: dict) -> dict:
    return {
        key: local_state.get(key)
        for key in RESOURCE_OUTPUT_STATE_KEYS.get(resource_type, ())
        if key in local_state
    }


def _primary_artifact(resource_type: str, local_state: dict) -> dict:
    if resource_type == "mindmap":
        return dict(local_state.get("mindmap_artifact") or {})
    if resource_type == "quiz":
        return dict(local_state.get("exercise_artifact") or {})
    if resource_type == "review_doc":
        return dict(local_state.get("review_doc_artifact") or {})
    if resource_type == "study_plan":
        artifact = dict(local_state.get("study_plan_artifact") or {})
        document = dict(local_state.get("study_plan_document_artifact") or {})
        return {**artifact, "document": document}
    if resource_type == "code_practice":
        return dict(local_state.get("code_practice_artifact") or {})
    if resource_type == "video_script":
        return dict(local_state.get("video_script_artifact") or {})
    if resource_type == "video_animation":
        return dict(local_state.get("video_animation_artifact") or {})
    return {}


def _resource_title(resource_type: str, artifact: dict, local_state: dict) -> str:
    if resource_type == "mindmap":
        return str(
            artifact.get("title")
            or (local_state.get("mindmap_tree") or {}).get("title")
            or "Mindmap"
        )
    if resource_type == "quiz":
        return str(artifact.get("title") or "Leveled exercises")
    if resource_type == "review_doc":
        return str(artifact.get("title") or "Review document")
    if resource_type == "study_plan":
        return str(artifact.get("title") or "Personalized Study Plan")
    if resource_type == "code_practice":
        return str(artifact.get("title") or "Code practice")
    if resource_type == "video_script":
        return str(artifact.get("title") or "Teaching video script")
    if resource_type == "video_animation":
        return str(artifact.get("title") or "Teaching animation")
    return resource_type


def _success_result(
    resource_type: str, local_state: dict, message_content: str, elapsed_ms: int
) -> dict:
    artifact = _primary_artifact(resource_type, local_state)
    return {
        "resource_type": resource_type,
        "status": "success",
        "title": _resource_title(resource_type, artifact, local_state),
        "artifact": artifact,
        "artifacts": list(local_state.get("review_doc_artifacts") or [])
        if resource_type == "review_doc"
        else [],
        "state_updates": _state_updates_for_resource(resource_type, local_state),
        "message_content": message_content,
        "message_preview": message_content[:500],
        "error_type": None,
        "error_message_sanitized": None,
        "elapsed_ms": elapsed_ms,
    }


def _failed_result(resource_type: str, exc: BaseException, elapsed_ms: int) -> dict:
    return {
        "resource_type": resource_type,
        "status": "failed",
        "title": resource_type,
        "artifact": {},
        "artifacts": [],
        "state_updates": {},
        "message_content": "",
        "message_preview": "",
        "error_type": type(exc).__name__,
        "error_message_sanitized": sanitize_error_message(str(exc), max_chars=1200),
        "elapsed_ms": elapsed_ms,
    }


def _count_mindmap_nodes(tree: Any) -> int:
    """Count mindmap nodes without assuming a perfect tree shape."""
    try:
        if isinstance(tree, dict):
            children = tree.get("children") or []
            return 1 + _count_mindmap_nodes(children)
        if isinstance(tree, list):
            return sum(_count_mindmap_nodes(child) for child in tree)
    except Exception:
        return 0
    return 0


@traced_node
async def resource_worker(state: LearningState) -> dict:
    """Generate exactly one resource branch and return a reducer-safe result."""
    task = dict(state.get("resource_task") or {})
    resource_type = normalize_resource_type(task.get("resource_type"))
    start = time.perf_counter()
    emit_a3_trace(
        logger,
        "resource_generation.worker.start",
        {"resource_type": resource_type, "task_id": task.get("task_id", "")},
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    try:
        if resource_type not in RESOURCE_RUNNERS:
            raise ValueError(f"unsupported resource_type: {task.get('resource_type')}")
        local_state = dict(state)
        local_state["requested_resource_type"] = resource_type
        local_state["requested_resource_types"] = [resource_type]
        message_content = await RESOURCE_RUNNERS[resource_type](local_state)
        result = _success_result(
            resource_type,
            local_state,
            message_content,
            int((time.perf_counter() - start) * 1000),
        )
        emit_a3_trace(
            logger,
            "resource_generation.worker.success",
            {
                "resource_type": resource_type,
                "elapsed_ms": result["elapsed_ms"],
                "message_chars": len(message_content),
            },
            state=state,
            env_flag="LOG_GENERATION_SUMMARY",
        )
    except Exception as exc:
        logger.exception("resource_worker failed for resource_type=%s", resource_type)
        result = _failed_result(
            resource_type or str(task.get("resource_type") or "unknown"),
            exc,
            int((time.perf_counter() - start) * 1000),
        )
        emit_a3_trace(
            logger,
            "resource_generation.worker.failed",
            {
                "resource_type": result["resource_type"],
                "elapsed_ms": result["elapsed_ms"],
                "error_type": result["error_type"],
                "error_message_sanitized": result["error_message_sanitized"],
            },
            state=state,
            env_flag="LOG_GENERATION_SUMMARY",
        )
    return {"resource_branch_results": [result]}


def _resource_display_name(resource_type: str) -> str:
    return {
        "review_doc": "复习资料",
        "mindmap": "思维导图",
        "quiz": "练习题",
        "code_practice": "代码实操案例",
        "video_script": "教学视频 / 动画脚本",
        "video_animation": "教学动画 / MP4 视频",
        "study_plan": "学习计划",
    }.get(resource_type, resource_type)


def _yes_no(value: bool) -> str:
    return "是" if value else "否"


def _bundle_display_order(result: dict) -> int:
    order = {
        "review_doc": 0,
        "mindmap": 1,
        "quiz": 2,
        "code_practice": 3,
        "video_script": 4,
        "video_animation": 5,
        "study_plan": 6,
    }
    return order.get(str(result.get("resource_type") or ""), len(order))


def _compose_resource_section(result: dict) -> list[str]:
    resource_type = str(result.get("resource_type") or "")
    title = str(result.get("title") or _resource_display_name(resource_type))
    metrics = _resource_metrics(result)
    lines = [f"### {_resource_display_name(resource_type)}"]

    if resource_type == "review_doc":
        lines.extend(
            [
                "已生成 Markdown / Word 版本。",
                f"- 标题：{title}",
                f"- 文档数量：{metrics.get('artifact_count', 0)}",
                f"- Markdown 字数：{metrics.get('markdown_chars', 0)}",
            ]
        )
    elif resource_type == "mindmap":
        lines.extend(
            [
                "已生成 XMind 版本。",
                f"- 标题：{title}",
                f"- 节点数量：{metrics.get('node_count', 0)}",
            ]
        )
    elif resource_type == "quiz":
        lines.extend(
            [
                "已生成 Markdown / Word 版本。",
                f"- 标题：{title}",
                f"- 题目数量：{metrics.get('item_count', 0)}",
            ]
        )
    elif resource_type == "code_practice":
        lines.extend(
            [
                "已生成 Markdown / Word / Python 源码版本。",
                f"- 标题：{title}",
                f"- 包含 Python 源码：{_yes_no(bool(metrics.get('has_python')))}",
            ]
        )
    elif resource_type == "video_script":
        lines.extend(
            [
                "已生成 Markdown / Word / SRT 字幕版本。",
                f"- 标题：{title}",
                f"- SRT 字符数：{metrics.get('srt_chars', 0)}",
            ]
        )
    elif resource_type == "video_animation":
        generated_formats = "HTML 预览 / JSON / SRT"
        if bool(metrics.get("render_success")):
            generated_formats += " / MP4"
        lines.extend(
            [
                f"已生成 {generated_formats} 版本。",
                f"- 标题：{title}",
                f"- MP4 渲染成功：{_yes_no(bool(metrics.get('render_success')))}",
            ]
        )
    elif resource_type == "study_plan":
        lines.extend(
            [
                "已生成 Markdown / Word 学习计划文档。",
                f"- 标题：{title}",
                f"- 是否包含文档：{_yes_no(bool(metrics.get('has_document')))}",
            ]
        )
    else:
        lines.extend([f"- 标题：{title}"])
    return lines


def _compose_bundle_message(
    status: str, successes: list[dict], failures: list[dict]
) -> str:
    if len(successes) == 1 and not failures:
        content = str(successes[0].get("message_content") or "").strip()
        if content:
            return content

    lines = [
        "# 已生成多类学习资源",
        "",
        "本次已根据你的请求生成以下学习资源，可分别查看或下载：",
        "",
    ]

    if successes:
        lines.extend(["## 已生成", ""])
        for result in sorted(successes, key=_bundle_display_order):
            lines.extend(_compose_resource_section(result))
            lines.append("")

    if failures:
        lines.extend(["## 未完成", ""])
        for result in sorted(failures, key=_bundle_display_order):
            resource_type = result.get("resource_type") or "unknown"
            reason = (
                result.get("error_message_sanitized")
                or result.get("error_type")
                or "unknown error"
            )
            lines.append(f"- {resource_type}: {reason}")
        lines.append("")

    if status == "failed":
        lines.append("所有请求的学习资源都生成失败，请稍后重试或缩小资源范围。")
    elif status == "partial_success":
        lines.append("部分资源已生成，失败的资源可以稍后单独重试。")

    return "\n".join(lines).strip()

def _resource_metrics(result: dict) -> dict:
    resource_type = result.get("resource_type")
    artifact = (
        result.get("artifact") if isinstance(result.get("artifact"), dict) else {}
    )
    artifacts = (
        result.get("artifacts") if isinstance(result.get("artifacts"), list) else []
    )
    state_updates = (
        result.get("state_updates")
        if isinstance(result.get("state_updates"), dict)
        else {}
    )

    if resource_type == "review_doc":
        markdown = (
            state_updates.get("review_doc_markdown") or artifact.get("markdown") or ""
        )
        return {
            "artifact_count": len(artifacts) or (1 if artifact else 0),
            "markdown_chars": len(str(markdown)),
            "has_markdown": bool(
                artifact.get("markdown_url") or artifact.get("markdown")
            ),
            "has_docx": bool(artifact.get("docx_url") or artifact.get("docx_filename")),
        }

    if resource_type == "mindmap":
        tree = state_updates.get("mindmap_tree") or artifact.get("tree") or {}
        return {
            "has_xmind": bool(artifact.get("xmind_url")),
            "node_count": _count_mindmap_nodes(tree),
            "has_png": bool(artifact.get("png_url")),
            "has_svg": bool(artifact.get("svg_url")),
        }

    if resource_type == "quiz":
        exercise_items = state_updates.get("exercise_items")
        if not isinstance(exercise_items, list):
            exercise_items = []
        return {
            "item_count": len(exercise_items),
            "has_markdown": bool(
                artifact.get("markdown_url") or artifact.get("markdown")
            ),
            "has_docx": bool(artifact.get("docx_url") or artifact.get("docx_filename")),
            "has_pdf": bool(artifact.get("pdf_url") or artifact.get("pdf_filename")),
        }

    if resource_type == "code_practice":
        markdown = (
            state_updates.get("code_practice_markdown")
            or artifact.get("markdown")
            or ""
        )
        return {
            "markdown_chars": len(str(markdown)),
            "has_markdown": bool(artifact.get("markdown_url") or markdown),
            "has_docx": bool(artifact.get("docx_url") or artifact.get("docx_filename")),
            "has_python": bool(
                artifact.get("python_url") or artifact.get("python_filename")
            ),
        }

    if resource_type == "video_script":
        markdown = (
            state_updates.get("video_script_markdown") or artifact.get("markdown") or ""
        )
        srt = state_updates.get("video_script_srt") or artifact.get("srt") or ""
        return {
            "markdown_chars": len(str(markdown)),
            "srt_chars": len(str(srt)),
            "has_markdown": bool(artifact.get("markdown_url") or markdown),
            "has_docx": bool(artifact.get("docx_url") or artifact.get("docx_filename")),
            "has_srt": bool(
                artifact.get("srt_url") or artifact.get("srt_filename") or srt
            ),
        }

    if resource_type == "video_animation":
        return {
            "duration_seconds": artifact.get("duration_seconds", 0),
            "render_success": bool(artifact.get("render_success")),
            "mp4_available": bool(
                artifact.get("mp4_available") or artifact.get("mp4_url")
            ),
            "has_html": bool(artifact.get("html_url") or artifact.get("html_filename")),
            "has_json": bool(artifact.get("json_url") or artifact.get("json_filename")),
            "has_srt": bool(artifact.get("srt_url") or artifact.get("srt_filename")),
            "has_mp4": bool(artifact.get("mp4_url") and artifact.get("render_success")),
        }

    if resource_type == "study_plan":
        document = (
            artifact.get("document")
            if isinstance(artifact.get("document"), dict)
            else {}
        )
        markdown = (
            state_updates.get("study_plan_markdown")
            or document.get("markdown")
            or artifact.get("markdown")
            or ""
        )
        return {
            "has_markdown": bool(
                document.get("markdown_url") or document.get("markdown") or markdown
            ),
            "has_docx": bool(document.get("docx_url") or document.get("docx_filename")),
            "has_document": bool(document),
        }

    return {}


def _resource_summary(result: dict) -> dict:
    return {
        "resource_type": result.get("resource_type"),
        "status": result.get("status"),
        "title": result.get("title"),
        "artifact": result.get("artifact") or {},
        "artifacts": result.get("artifacts") or [],
        "message_preview": result.get("message_preview") or "",
        "error_type": result.get("error_type"),
        "error_message_sanitized": result.get("error_message_sanitized"),
        "elapsed_ms": result.get("elapsed_ms"),
        "metrics": _resource_metrics(result),
    }


@traced_node
async def resource_bundle_output(state: LearningState) -> dict:
    """Aggregate resource worker results into one final user-visible bundle."""
    requested = normalize_requested_resource_types(
        state.get("requested_resource_types") or [],
        state.get("requested_resource_type") or "",
    )
    results = [
        result
        for result in state.get("resource_branch_results") or []
        if isinstance(result, dict) and result.get("resource_type")
    ]
    successes = [result for result in results if result.get("status") == "success"]
    failures = [result for result in results if result.get("status") == "failed"]

    if successes and failures:
        status = "partial_success"
    elif successes:
        status = "success"
    elif requested or results:
        status = "failed"
    else:
        status = "skipped"

    state_updates: dict = {}
    for result in successes:
        state_updates.update(result.get("state_updates") or {})

    bundle = {
        "type": "resource_bundle",
        "status": status,
        "requested_resource_types": requested,
        "success_count": len(successes),
        "failed_count": len(failures),
        "resources": [_resource_summary(result) for result in successes],
        "errors": [_resource_summary(result) for result in failures],
    }
    debug = dict(state.get("resource_generation_debug") or {})
    stages = list(debug.get("stages") or [])
    stages.append(
        {
            "stage": "resource_generation.bundle.complete",
            "status": status,
            "success_count": len(successes),
            "failed_count": len(failures),
            "resource_count": len(results),
        }
    )
    debug.update(
        {
            "status": status,
            "success_count": len(successes),
            "failed_count": len(failures),
            "partial_success": status == "partial_success",
            "branch_results": [_resource_summary(result) for result in results],
            "stages": stages,
        }
    )
    message = _compose_bundle_message(status, successes, failures)
    bundle["message"] = message
    emit_a3_trace(
        logger,
        "resource_generation.bundle.complete",
        {
            "status": status,
            "requested_resource_types": requested,
            "success_count": len(successes),
            "failed_count": len(failures),
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        **state_updates,
        "resource_bundle_artifact": bundle,
        "resource_generation_debug": debug,
        "resource_generation_status": status,
        "messages": [AIMessage(content=message)] if message else [],
    }
