"""Sequential multi-resource orchestration node."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage

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
from src.graph.state import LearningState
from src.observability.a3_trace import emit_a3_trace

logger = logging.getLogger(__name__)

ResourceNode = Callable[[LearningState], Awaitable[dict]]

RESOURCE_ORDER = ("review_doc", "mindmap", "quiz", "code_practice")
SUPPORTED_RESOURCE_TYPES = set(RESOURCE_ORDER)


def _apply_update(working_state: dict, update: dict) -> dict:
    """Merge a node update into a working state snapshot."""
    if not update:
        return working_state

    next_state = dict(working_state)
    for key, value in update.items():
        if key == "messages":
            existing_messages = list(next_state.get("messages") or [])
            new_messages = value if isinstance(value, list) else [value]
            next_state["messages"] = existing_messages + [msg for msg in new_messages if msg is not None]
        else:
            next_state[key] = value
    return next_state


async def _run_node(working_state: dict, node: ResourceNode) -> dict:
    update = await node(working_state)  # type: ignore[arg-type]
    return _apply_update(working_state, update or {})


async def _run_review_doc_chain(working_state: dict) -> dict:
    working_state = await _run_node(working_state, review_doc_planner)
    while True:
        working_state = await _run_node(working_state, review_doc_agent)
        working_state = await _run_node(working_state, review_doc_reviewer)
        route = should_rewrite_review_doc(working_state)  # type: ignore[arg-type]
        if route != "rewrite":
            break
        working_state = await _run_node(working_state, review_doc_rewrite)
    return await _run_node(working_state, review_doc_output)


async def _run_mindmap_chain(working_state: dict) -> dict:
    working_state = await _run_node(working_state, mindmap_planner)
    while True:
        working_state = await _run_node(working_state, mindmap_agent)
        working_state = await _run_node(working_state, mindmap_reviewer)
        route = should_rewrite_mindmap(working_state)  # type: ignore[arg-type]
        if route != "rewrite":
            break
        working_state = await _run_node(working_state, mindmap_rewrite)
    return await _run_node(working_state, mindmap_output)


async def _run_quiz_chain(working_state: dict) -> dict:
    working_state = await _run_node(working_state, exercise_planner)
    while True:
        working_state = await _run_node(working_state, exercise_agent)
        working_state = await _run_node(working_state, exercise_reviewer)
        route = should_rewrite_exercise(working_state)  # type: ignore[arg-type]
        if route != "rewrite":
            break
        working_state = await _run_node(working_state, exercise_rewrite)
    return await _run_node(working_state, exercise_output)


async def _run_code_practice_chain(working_state: dict) -> dict:
    working_state = await _run_node(working_state, code_practice_planner)
    while True:
        working_state = await _run_node(working_state, code_practice_agent)
        working_state = await _run_node(working_state, code_practice_reviewer)
        route = should_rewrite_code_practice(working_state)  # type: ignore[arg-type]
        if route != "rewrite":
            break
        working_state = await _run_node(working_state, code_practice_rewrite)
    return await _run_node(working_state, code_practice_output)


def _requested_resources(state: dict) -> list[str]:
    requested = [
        str(item).strip()
        for item in state.get("requested_resource_types") or []
        if str(item or "").strip()
    ]
    if not requested:
        fallback = str(state.get("requested_resource_type") or "").strip()
        if fallback and fallback != "multi_resource":
            requested = [fallback]
    normalized: list[str] = []
    for resource_type in requested:
        if resource_type in {"exercise", "exercises"}:
            resource_type = "quiz"
        if resource_type and resource_type not in normalized:
            normalized.append(resource_type)
    return normalized


def _resource_result(resource_type: str, working_state: dict) -> dict[str, Any]:
    if resource_type == "review_doc":
        artifacts = working_state.get("review_doc_artifacts") or []
        artifact = working_state.get("review_doc_artifact") or {}
        return {
            "resource_type": resource_type,
            "status": "completed",
            "title": artifact.get("title", "Review Document"),
            "artifact_count": len(artifacts),
            "markdown_chars": len(str(working_state.get("review_doc_markdown") or "")),
        }
    if resource_type == "mindmap":
        artifact = working_state.get("mindmap_artifact") or {}
        tree = working_state.get("mindmap_tree") or {}
        return {
            "resource_type": resource_type,
            "status": "completed",
            "title": artifact.get("title") or tree.get("title", "Mindmap"),
            "has_xmind": bool(artifact.get("xmind_url")),
            "node_count": _count_mindmap_nodes(tree),
        }
    if resource_type == "quiz":
        artifact = working_state.get("exercise_artifact") or {}
        items = working_state.get("exercise_items") or []
        return {
            "resource_type": resource_type,
            "status": "completed",
            "title": artifact.get("title", "Exercises"),
            "item_count": len(items),
            "has_markdown": bool(artifact.get("markdown_url")),
            "has_docx": bool(artifact.get("docx_url")),
        }
    if resource_type == "code_practice":
        artifact = working_state.get("code_practice_artifact") or {}
        return {
            "resource_type": resource_type,
            "status": "completed",
            "title": artifact.get("title", "代码实操案例"),
            "markdown_chars": len(str(working_state.get("code_practice_markdown") or "")),
            "has_markdown": bool(artifact.get("markdown_url")),
            "has_docx": bool(artifact.get("docx_url")),
            "has_python": bool(artifact.get("python_url")),
        }
    return {"resource_type": resource_type, "status": "completed"}


def _count_mindmap_nodes(tree: Any) -> int:
    if not isinstance(tree, dict):
        return 0
    return 1 + sum(_count_mindmap_nodes(child) for child in tree.get("children") or [])


def _build_combined_answer(completed: list[str], results: list[dict]) -> str:
    lines = [
        "# 已生成多类学习资源",
        "",
        "本次已根据你的请求一次性生成以下资源，可分别下载：",
        "",
    ]

    if "review_doc" in completed:
        lines.extend(["## 一、复习资料", "已生成 Markdown / Word / PDF 打印版本。", ""])
    if "mindmap" in completed:
        lines.extend(["## 二、思维导图", "已生成 XMind / Markdown / SVG / PNG 导出版本。", ""])
    if "quiz" in completed:
        lines.extend(["## 三、练习题", "已生成 Markdown / Word / PDF 打印版本。", ""])
    if "code_practice" in completed:
        lines.extend(["## 四、代码实操案例", "已生成 Markdown / Word / Python 源码版本，可下载后直接运行。", ""])

    lines.extend(["---", "", "## 资源摘要"])
    for result in results:
        if result.get("status") != "completed":
            continue
        resource_type = result.get("resource_type")
        title = result.get("title", "")
        if resource_type == "review_doc":
            lines.append(f"- 复习资料：{title}，共 {result.get('artifact_count', 0)} 份文档。")
        elif resource_type == "mindmap":
            lines.append(f"- 思维导图：{title}，共 {result.get('node_count', 0)} 个节点。")
        elif resource_type == "quiz":
            lines.append(f"- 练习题：{title}，共 {result.get('item_count', 0)} 道题。")
        elif resource_type == "code_practice":
            lines.append(f"- 代码实操案例：{title}，已生成 Markdown、Word 和 Python 源码。")
    return "\n".join(lines).strip()


async def multi_resource_runner(state: LearningState) -> dict:
    working_state: dict = dict(state)
    requested = _requested_resources(working_state)
    if not requested:
        raise RuntimeError("multi_resource_runner has no requested resources")
    requested_supported = [item for item in RESOURCE_ORDER if item in requested]
    skipped = [item for item in requested if item not in SUPPORTED_RESOURCE_TYPES]

    completed: list[str] = []
    failed: list[str] = []
    results: list[dict] = []

    for resource_type in skipped:
        logger.warning("multi_resource_runner skipping unsupported resource type: %s", resource_type)
        results.append(
            {
                "resource_type": resource_type,
                "status": "skipped",
                "warning": "Multi-resource mode does not support this resource type yet.",
            }
        )

    runners = {
        "review_doc": _run_review_doc_chain,
        "mindmap": _run_mindmap_chain,
        "quiz": _run_quiz_chain,
        "code_practice": _run_code_practice_chain,
    }

    for resource_type in requested_supported:
        try:
            working_state = await runners[resource_type](working_state)
            completed.append(resource_type)
            results.append(_resource_result(resource_type, working_state))
        except Exception as exc:
            logger.exception("multi_resource_runner failed for resource type %s", resource_type)
            failed.append(resource_type)
            results.append(
                {
                    "resource_type": resource_type,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    if requested_supported and not completed:
        raise RuntimeError(f"multi_resource_runner failed all requested resources: {failed}")
    if not requested_supported and skipped:
        raise RuntimeError(f"multi_resource_runner has no supported resources: {skipped}")

    combined_answer = _build_combined_answer(completed, results)
    review_doc_artifacts = working_state.get("review_doc_artifacts") or []
    mindmap_artifact = working_state.get("mindmap_artifact") or {}
    exercise_artifact = working_state.get("exercise_artifact") or {}
    code_practice_artifact = working_state.get("code_practice_artifact") or {}
    code_practice_markdown = working_state.get("code_practice_markdown", "")

    emit_a3_trace(
        logger,
        "multi_resource_runner",
        {
            "requested_resource_types": requested,
            "completed_resource_types": completed,
            "failed_resource_types": failed,
            "has_review_doc_artifacts": bool(review_doc_artifacts),
            "review_doc_artifacts_count": len(review_doc_artifacts),
            "has_mindmap": bool(mindmap_artifact),
            "has_exercise": bool(exercise_artifact),
            "has_code_practice": bool(code_practice_artifact),
            "code_practice_artifact_exists": bool(code_practice_artifact),
            "answer_chars": len(combined_answer),
        },
        state=working_state,
        env_flag="LOG_GENERATION_SUMMARY",
    )

    return {
        "review_doc_artifact": working_state.get("review_doc_artifact") or {},
        "review_doc_artifacts": review_doc_artifacts,
        "review_doc_markdown": working_state.get("review_doc_markdown", ""),
        "mindmap_artifact": mindmap_artifact,
        "mindmap_tree": working_state.get("mindmap_tree") or {},
        "exercise_artifact": exercise_artifact,
        "exercise_items": working_state.get("exercise_items") or [],
        "code_practice_artifact": code_practice_artifact,
        "code_practice_markdown": code_practice_markdown,
        "multi_resource_results": results,
        "multi_resource_summary": combined_answer,
        "messages": [AIMessage(content=combined_answer)],
    }
