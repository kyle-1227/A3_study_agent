"""Strict rendered teaching-animation resource-generation nodes."""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from src.config import get_setting
from src.graph.state import LearningState
from src.llm.structured_output import (
    StructuredOutputError,
    get_llm_output_mode,
    get_max_raw_chars,
    invoke_structured_llm,
)
from src.observability.a3_trace import emit_a3_trace
from src.tools.video_animation_contracts import (
    AnimationReviewVerdictV1,
    VideoAnimationSpecV1,
    validate_animation_review_verdict,
    validate_video_animation_spec,
)
from src.tools.video_animation_tool import (
    create_video_animation_artifact_async,
    get_video_animation_artifact_dir,
)
from src.tracing import traced_llm_call, traced_node

logger = logging.getLogger(__name__)


class VideoAnimationGenerationError(RuntimeError):
    """Raised when no provider-produced animation specification can be accepted."""


class VideoAnimationApprovalError(RuntimeError):
    """Raised when an unapproved animation reaches artifact output."""


class VideoAnimationRenderError(RuntimeError):
    """Raised when the renderer cannot produce the requested real artifact."""


def _last_human_query(state: LearningState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _format_context(context: list[dict]) -> str:
    if not context:
        return "No judged evidence is available. Do not invent facts or sources."
    parts: list[str] = []
    for index, item in enumerate(context[:6], 1):
        source = (
            item.get("source")
            or item.get("title")
            or item.get("url")
            or "learning material"
        )
        content = str(
            item.get("content") or item.get("snippet") or item.get("text") or ""
        )[:700]
        if content:
            parts.append(f"[{index}] Source: {source}\n{content}")
    return "\n\n".join(parts) or "Judged evidence has no readable body."


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


def _video_animation_model_name() -> str:
    configured_model = get_setting("llm.video_animation.model", None)
    if not isinstance(configured_model, str) or not configured_model.strip():
        raise ValueError("llm.video_animation.model must be explicitly configured")
    return configured_model.strip()


def _video_animation_temperature() -> float:
    configured_temperature = get_setting("llm.video_animation.temperature", None)
    if isinstance(configured_temperature, bool) or not isinstance(
        configured_temperature, (int, float)
    ):
        raise ValueError(
            "llm.video_animation.temperature must be explicitly configured"
        )
    temperature = float(configured_temperature)
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("llm.video_animation.temperature must be between 0 and 2")
    return temperature


def _video_animation_max_generation_rounds() -> int:
    configured_rounds = get_setting("video_animation.max_generation_rounds", None)
    if isinstance(configured_rounds, bool) or not isinstance(configured_rounds, int):
        raise ValueError(
            "video_animation.max_generation_rounds must be explicitly configured"
        )
    if configured_rounds < 1:
        raise ValueError("video_animation.max_generation_rounds must be at least one")
    return configured_rounds


def _video_animation_max_duration_seconds() -> int:
    configured_duration = get_setting("video_animation.max_duration_seconds", None)
    if isinstance(configured_duration, bool) or not isinstance(
        configured_duration, int
    ):
        raise ValueError(
            "video_animation.max_duration_seconds must be explicitly configured"
        )
    if configured_duration != 90:
        raise ValueError("video_animation.max_duration_seconds must equal 90 for V1")
    return configured_duration


def _video_animation_render_mode() -> str:
    configured_mode = get_setting("video_animation.render_mode", None)
    if configured_mode not in {"production", "test"}:
        raise ValueError(
            "video_animation.render_mode must be explicitly configured as production or test"
        )
    return str(configured_mode)


def _raise_if_insufficient_evidence(state: LearningState) -> None:
    if (
        state.get("degraded_generation") is True
        and state.get("evidence_judge_state") == "insufficient"
    ):
        raise VideoAnimationGenerationError(
            "video animation generation blocked because evidence is insufficient"
        )


def _validation_error_summary(exc: ValidationError) -> str:
    details: list[str] = []
    for error in exc.errors(include_url=False, include_input=False)[:8]:
        location = ".".join(str(part) for part in error.get("loc", ())) or "root"
        details.append(f"{location}: {error.get('msg', 'invalid value')}")
    return "; ".join(details) or "animation specification validation failed"


def _validate_spec_payload(payload: object) -> VideoAnimationSpecV1:
    try:
        spec = VideoAnimationSpecV1.model_validate(payload)
    except ValidationError as exc:
        raise VideoAnimationGenerationError(_validation_error_summary(exc)) from exc
    business_error = validate_video_animation_spec(spec)
    if business_error:
        raise VideoAnimationGenerationError(business_error)
    return spec


def _spec_payload(spec: VideoAnimationSpecV1) -> dict:
    return spec.model_dump(mode="json")


def _planner_prompt(state: LearningState) -> str:
    existing_script = str(state.get("video_script_markdown") or "").strip()
    return (
        "Plan a concise, evidence-grounded teaching animation.\n"
        "Return the exact video_animation_spec_v1 structured contract.\n\n"
        f"## User question\n{_last_human_query(state)}\n\n"
        f"## Expanded keypoints\n{_format_keypoints(state)}\n\n"
        f"## Judged evidence\n{_format_context(state.get('context', []))}\n\n"
        f"## Existing video script\n{existing_script[:5000] if existing_script else 'None'}\n\n"
        "Use exactly 5-8 non-overlapping scenes within 90 seconds. "
        "Every scene must contain all six animation steps exactly once. "
        "Box/text/circle elements require text, x, y, width, and height. "
        "Arrow elements require source, target, and non-empty text."
    )


def _agent_prompt(state: LearningState, draft_spec: VideoAnimationSpecV1) -> str:
    return (
        "Produce the final evidence-grounded teaching animation specification.\n"
        "Return the exact video_animation_spec_v1 structured contract.\n\n"
        f"## User question\n{_last_human_query(state)}\n\n"
        f"## Draft specification\n{json.dumps(_spec_payload(draft_spec), ensure_ascii=False, indent=2)}\n\n"
        f"## Existing video script\n{str(state.get('video_script_markdown') or '')[:5000] or 'None'}\n\n"
        f"## Revision notes\n{state.get('video_animation_revision_notes', '') or 'None'}"
    )


async def _invoke_animation_spec(
    *, node_name: str, prompt: str, state: LearningState
) -> VideoAnimationSpecV1:
    model_name = _video_animation_model_name()
    temperature = _video_animation_temperature()
    with traced_llm_call(
        model_name=model_name, node_name=node_name, temperature=temperature
    ):
        structured_result = await invoke_structured_llm(
            node_name=node_name,
            llm_node="video_animation",
            schema=VideoAnimationSpecV1,
            messages=[
                SystemMessage(
                    content=(
                        "You create strict JSON specifications for deterministic "
                        "teaching-animation rendering."
                    )
                ),
                HumanMessage(content=prompt),
            ],
            output_mode=get_llm_output_mode(node_name),
            business_validator=validate_video_animation_spec,
            state=state,
            max_raw_chars=get_max_raw_chars(node_name),
        )
    if not structured_result.success:
        raise StructuredOutputError(structured_result)
    if not isinstance(structured_result.parsed, VideoAnimationSpecV1):
        raise VideoAnimationGenerationError(
            f"{node_name} parsed result is not VideoAnimationSpecV1"
        )
    return structured_result.parsed


def _topic_terms(state: LearningState) -> list[str]:
    values = [
        _last_human_query(state),
        str(state.get("primary_subject") or ""),
        str(state.get("learning_goal") or ""),
    ]
    values.extend(str(item) for item in state.get("keypoints", []) if str(item).strip())
    values.extend(
        str(item) for item in state.get("expanded_keypoints", []) if str(item).strip()
    )
    joined = " ".join(values).lower()
    terms: list[str] = []
    for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_+#.-]{1,}", joined):
        if term not in {"video", "animation", "mp4"} and term not in terms:
            terms.append(term)
    for phrase in ("面向对象", "类", "对象", "__init__", "继承", "多态", "封装"):
        if phrase.lower() in joined and phrase not in terms:
            terms.append(phrase)
    return terms


def _is_topic_relevant(spec: VideoAnimationSpecV1, state: LearningState) -> bool:
    text = json.dumps(_spec_payload(spec), ensure_ascii=False).lower()
    subject = str(state.get("primary_subject") or "").strip().lower()
    if subject == "python":
        return any(term in text for term in ("python", "类", "对象", "面向对象"))
    if subject and subject != "other":
        candidates = {subject, subject.replace("_", " "), subject.replace("_", "")}
        return any(candidate and candidate in text for candidate in candidates)
    terms = _topic_terms(state)
    return bool(terms) and any(term.lower() in text for term in terms)


def _local_check_video_animation(payload: object, state: LearningState) -> dict:
    try:
        spec = _validate_spec_payload(payload)
    except VideoAnimationGenerationError as exc:
        return {
            "passed": False,
            "failed_reasons": [str(exc)],
            "scene_count": 0,
            "duration_seconds": 0,
            "topic_relevant": False,
        }
    topic_relevant = _is_topic_relevant(spec, state)
    failed_reasons = (
        [] if topic_relevant else ["animation spec is not relevant to user topic"]
    )
    return {
        "passed": not failed_reasons,
        "failed_reasons": failed_reasons,
        "scene_count": len(spec.scenes),
        "duration_seconds": spec.duration_seconds,
        "topic_relevant": topic_relevant,
    }


@traced_node
async def video_animation_planner(state: LearningState) -> dict:
    _raise_if_insufficient_evidence(state)
    draft_spec = await _invoke_animation_spec(
        node_name="video_animation_planner",
        prompt=_planner_prompt(state),
        state=state,
    )
    payload = _spec_payload(draft_spec)
    emit_a3_trace(
        logger,
        "video_animation_planner",
        {
            "has_video_script_markdown": bool(state.get("video_script_markdown")),
            "scene_count": len(draft_spec.scenes),
            "duration_seconds": draft_spec.duration_seconds,
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "video_animation_spec": payload,
        "video_animation_html": "",
        "video_animation_artifact": {},
        "video_animation_review_verdict": "",
        "video_animation_review_reason": "",
        "video_animation_revision_notes": "",
        "video_animation_round": 0,
        "video_animation_render_log": "",
    }


@traced_node
async def video_animation_agent(state: LearningState) -> dict:
    _raise_if_insufficient_evidence(state)
    draft_spec = _validate_spec_payload(state.get("video_animation_spec"))
    spec = await _invoke_animation_spec(
        node_name="video_animation_agent",
        prompt=_agent_prompt(state, draft_spec),
        state=state,
    )
    round_no = int(state.get("video_animation_round", 0) or 0) + 1
    emit_a3_trace(
        logger,
        "video_animation_agent",
        {
            "round": round_no,
            "scene_count": len(spec.scenes),
            "duration_seconds": spec.duration_seconds,
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "video_animation_spec": _spec_payload(spec),
        "video_animation_round": round_no,
        "video_animation_review_verdict": "",
        "video_animation_review_reason": "",
    }


@traced_node
async def video_animation_reviewer(state: LearningState) -> dict:
    local_check = _local_check_video_animation(state.get("video_animation_spec"), state)
    if not local_check["passed"]:
        verdict = "reject"
        reason = "; ".join(local_check["failed_reasons"])
    else:
        model_name = _video_animation_model_name()
        with traced_llm_call(
            model_name=model_name,
            node_name="video_animation_reviewer",
            temperature=0.0,
        ):
            structured_result = await invoke_structured_llm(
                node_name="video_animation_reviewer",
                llm_node="video_animation",
                schema=AnimationReviewVerdictV1,
                messages=[
                    SystemMessage(
                        content=(
                            "You are a strict teaching-animation quality reviewer. "
                            "Return the exact structured verdict."
                        )
                    ),
                    HumanMessage(
                        content=(
                            "Review the teaching quality, clarity, topic relevance, and "
                            "scene progression of this validated animation specification.\n\n"
                            f"## User question\n{_last_human_query(state)}\n\n"
                            "## Animation specification\n"
                            f"{json.dumps(state.get('video_animation_spec'), ensure_ascii=False, indent=2)}"
                        )
                    ),
                ],
                output_mode=get_llm_output_mode("video_animation_reviewer"),
                business_validator=validate_animation_review_verdict,
                state=state,
                max_raw_chars=get_max_raw_chars("video_animation_reviewer"),
            )
        if not structured_result.success:
            raise StructuredOutputError(structured_result)
        result = structured_result.parsed
        if not isinstance(result, AnimationReviewVerdictV1):
            raise VideoAnimationApprovalError(
                "video_animation_reviewer parsed result is not AnimationReviewVerdictV1"
            )
        verdict = result.verdict
        reason = result.reason.strip()
    emit_a3_trace(
        logger,
        "video_animation_reviewer",
        {
            "local_check_passed": bool(local_check["passed"]),
            "scene_count": local_check.get("scene_count", 0),
            "duration_seconds": local_check.get("duration_seconds", 0),
            "topic_relevant": bool(local_check.get("topic_relevant")),
            "verdict": verdict,
            "reason": reason,
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "video_animation_review_verdict": verdict,
        "video_animation_review_reason": reason,
        "video_animation_revision_notes": "" if verdict == "approve" else reason,
        "video_animation_local_check": local_check,
    }


def should_rewrite_video_animation(state: LearningState) -> str:
    verdict = str(state.get("video_animation_review_verdict") or "").strip()
    if verdict == "approve":
        return "output"
    if verdict != "reject":
        raise VideoAnimationApprovalError(
            "video animation routing requires an explicit approve or reject verdict"
        )
    current_round = int(state.get("video_animation_round", 0) or 0)
    if current_round < _video_animation_max_generation_rounds():
        return "rewrite"
    raise VideoAnimationApprovalError(
        "video animation remained rejected after the maximum rewrite rounds"
    )


@traced_node
async def video_animation_rewrite(state: LearningState) -> dict:
    reason = str(state.get("video_animation_review_reason") or "").strip()
    if not reason:
        raise VideoAnimationApprovalError(
            "video animation rewrite requires a review reason"
        )
    spec = _validate_spec_payload(state.get("video_animation_spec"))
    return {
        "video_animation_spec": _spec_payload(spec),
        "video_animation_revision_notes": reason,
    }


def _read_artifact_html(artifact: dict) -> str:
    artifact_id = str(artifact.get("artifact_id") or "").strip()
    filename = str(artifact.get("html_filename") or "").strip()
    if not artifact_id or not filename:
        raise VideoAnimationRenderError(
            "video animation artifact is missing HTML identity"
        )
    root = get_video_animation_artifact_dir()
    path = (root / artifact_id / filename).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise VideoAnimationRenderError(
            "video animation HTML path escapes the artifact root"
        ) from exc
    if not path.is_file():
        raise VideoAnimationRenderError("video animation HTML artifact is missing")
    return path.read_text(encoding="utf-8")


def _validate_rendered_artifact(artifact: object, render_mode: str) -> dict:
    if not isinstance(artifact, dict):
        raise VideoAnimationRenderError("video animation renderer returned no artifact")
    required = {
        "artifact_id": artifact.get("artifact_id"),
        "html_url": artifact.get("html_url"),
        "json_url": artifact.get("json_url"),
        "srt_url": artifact.get("srt_url"),
        "html_filename": artifact.get("html_filename"),
        "json_filename": artifact.get("json_filename"),
        "srt_filename": artifact.get("srt_filename"),
    }
    missing = [name for name, value in required.items() if not str(value or "").strip()]
    if missing:
        raise VideoAnimationRenderError(
            f"video animation artifact missing fields: {', '.join(missing)}"
        )
    for availability_field in ("html_available", "json_available", "srt_available"):
        if artifact.get(availability_field) is not True:
            raise VideoAnimationRenderError(
                f"video animation artifact is not readable: {availability_field}"
            )
    if artifact.get("render_success"):
        if not str(artifact.get("mp4_url") or "").strip():
            raise VideoAnimationRenderError(
                "successful video animation render is missing mp4_url"
            )
        if (
            not artifact.get("mp4_exists")
            or int(artifact.get("mp4_file_size") or 0) <= 0
        ):
            raise VideoAnimationRenderError(
                "successful video animation MP4 file is missing or empty"
            )
    if render_mode != str(artifact.get("render_mode") or "").strip():
        raise VideoAnimationRenderError(
            "video animation artifact render_mode does not match runtime config"
        )
    return artifact


def _is_formal_full_video(artifact: dict, render_mode: str) -> bool:
    full_duration = int(artifact.get("full_duration_seconds") or 0)
    render_duration = int(artifact.get("render_duration_seconds") or 0)
    return bool(
        render_mode == "production"
        and artifact.get("render_success") is True
        and artifact.get("is_preview_video") is False
        and artifact.get("video_valid_for_teaching") is True
        and full_duration > 0
        and render_duration == full_duration
    )


@traced_node
async def video_animation_output(state: LearningState) -> dict:
    verdict = str(state.get("video_animation_review_verdict") or "").strip()
    reason = str(state.get("video_animation_review_reason") or "").strip()
    if verdict != "approve":
        detail = f": {reason}" if reason else ""
        raise VideoAnimationApprovalError(
            f"video animation output requires an approve verdict{detail}"
        )
    spec = _validate_spec_payload(state.get("video_animation_spec"))
    local_check = _local_check_video_animation(_spec_payload(spec), state)
    if not local_check["passed"]:
        raise VideoAnimationApprovalError(
            "video animation output failed local quality check: "
            + "; ".join(local_check["failed_reasons"])
        )
    render_mode = _video_animation_render_mode()
    render_fps = 12 if render_mode == "test" else 24
    artifact = await create_video_animation_artifact_async(
        animation_spec=_spec_payload(spec),
        title=spec.title,
        srt_text=None,
        fps=render_fps,
        width=spec.resolution.width,
        height=spec.resolution.height,
        max_duration_seconds=_video_animation_max_duration_seconds(),
        render_mode=render_mode,
    )
    artifact = _validate_rendered_artifact(artifact, render_mode)
    render_log = str(artifact.get("render_log") or "")
    html = _read_artifact_html(artifact)
    formal_full_video = _is_formal_full_video(artifact, render_mode)
    status_text = (
        "已生成并验证完整正式教学动画。"
        if formal_full_video
        else "已生成可验证的 HTML、JSON 与字幕资源；MP4 未达到完整正式视频标准。"
    )
    mp4_text = (
        str(artifact.get("mp4_url"))
        if formal_full_video
        else "不可作为完整正式教学视频"
    )
    content = (
        f"# {spec.title}\n\n"
        f"{status_text}\n\n"
        f"- HTML 预览：{artifact['html_url']}\n"
        f"- JSON 规格：{artifact['json_url']}\n"
        f"- 字幕 SRT：{artifact['srt_url']}\n"
        f"- MP4 视频：{mp4_text}\n"
    )
    emit_a3_trace(
        logger,
        "video_animation_output",
        {
            "title": spec.title,
            "artifact_id": artifact["artifact_id"],
            "render_success": bool(artifact.get("render_success")),
            "formal_full_video": formal_full_video,
            "render_mode": render_mode,
            "is_preview_video": bool(artifact.get("is_preview_video")),
            "video_valid_for_teaching": bool(artifact.get("video_valid_for_teaching")),
            "render_log_preview": render_log[:500],
            "ffmpeg_path": artifact.get("ffmpeg_path", ""),
            "playwright_available": bool(artifact.get("playwright_available")),
            "frame_count": int(artifact.get("frame_count") or 0),
            "render_duration_seconds": int(
                artifact.get("render_duration_seconds") or 0
            ),
            "full_duration_seconds": int(artifact.get("full_duration_seconds") or 0),
            "fps": int(artifact.get("fps") or 0),
            "mp4_exists": bool(artifact.get("mp4_exists")),
            "mp4_file_size": int(artifact.get("mp4_file_size") or 0),
            "html_url": artifact["html_url"],
            "mp4_url": artifact["mp4_url"],
            "render_log_chars": len(render_log),
            "emits_ai_message": True,
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "video_animation_artifact": artifact,
        "video_animation_spec": _spec_payload(spec),
        "video_animation_html": html,
        "video_animation_render_log": render_log,
        "messages": [AIMessage(content=content)],
    }
