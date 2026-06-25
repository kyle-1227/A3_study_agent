"""Rendered teaching animation / MP4 resource-generation nodes."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.config import get_setting
from src.graph.llm import invoke_plain_llm_fail_fast
from src.graph.state import LearningState
from src.observability.a3_trace import emit_a3_trace
from src.tools.video_animation_tool import (
    create_video_animation_artifact_async,
    get_video_animation_artifact_dir,
)
from src.tracing import traced_node

logger = logging.getLogger(__name__)
DEFAULT_ANIMATION_STEPS = [
    "fade_in",
    "move",
    "highlight",
    "arrow_draw",
    "code_highlight",
    "fade_out",
]


def _last_human_query(state: LearningState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _format_context(context: list[dict]) -> str:
    if not context:
        return "No judged evidence is available. Use a fallback educational animation structure."
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


def _extract_markdown_title(markdown: str) -> str:
    for line in str(markdown or "").splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line.strip())
        if match:
            title = match.group(1).strip()
            book_title = re.search(r"《\s*(?P<title>.+?)\s*》", title)
            return (book_title.group("title") if book_title else title).strip()
    return ""


def _topic_title(state: LearningState) -> str:
    script_title = _extract_markdown_title(state.get("video_script_markdown", ""))
    if script_title:
        return script_title.replace("脚本", "动画演示")
    query = _last_human_query(state)
    if "面向对象" in query or "oop" in query.lower():
        return "Python 面向对象编程动画演示"
    subject = str(state.get("primary_subject") or state.get("subject") or "").strip()
    if subject and subject != "other":
        return f"{subject} 教学动画演示"
    return "教学动画演示"


def _default_scene_elements(index: int, left: str, right: str) -> list[dict]:
    return [
        {"type": "box", "text": left, "x": 120, "y": 180},
        {"type": "box", "text": right, "x": 680, "y": 120 + (index % 3) * 70},
        {"type": "arrow", "from": left, "to": right},
    ]


def _fallback_animation_spec(state: LearningState, reason: str = "") -> dict:
    title = _topic_title(state)
    query = _last_human_query(state).lower()
    subject = str(state.get("primary_subject") or state.get("subject") or "").lower()
    if ("python" in query or subject == "python") and (
        "面向对象" in query or "oop" in query
    ):
        title = "Python 面向对象编程核心概念动画"
    scene_blueprint = [
        (
            "类与对象",
            "类是蓝图，对象是实例",
            "在 Python 中，类像蓝图，对象是根据蓝图创建出的具体实例。",
            "class Student",
            "student_1",
        ),
        (
            "__init__ 初始化",
            "__init__ 负责设置初始属性",
            "创建对象时，__init__ 会接收参数并把数据保存到对象属性中。",
            "__init__(name)",
            "self.name",
        ),
        (
            "实例方法调用",
            "方法描述对象能做什么",
            "实例方法通过 self 访问对象自己的属性，并完成与该对象相关的行为。",
            "student.say_hi()",
            "self",
        ),
        (
            "继承",
            "子类复用并扩展父类",
            "继承让子类获得父类的属性和方法，同时可以加入自己的新能力。",
            "Person",
            "Student",
        ),
        (
            "多态",
            "同一接口，不同行为",
            "多态让不同对象响应同一个方法调用，并表现出各自的行为。",
            "speak()",
            "不同对象",
        ),
        (
            "封装",
            "隐藏细节，暴露接口",
            "封装把内部数据和实现细节收拢起来，只通过清晰的方法对外使用。",
            "_balance",
            "deposit()",
        ),
        (
            "总结",
            "类、对象、属性、方法协同建模",
            "掌握这些概念后，就可以用面向对象方式组织更复杂的 Python 程序。",
            "OOP",
            "可维护程序",
        ),
    ]
    scenes: list[dict] = []
    duration = 28
    for index, (scene_title, subtitle, narration, left, right) in enumerate(
        scene_blueprint
    ):
        start = index * 4
        end = start + 4
        scenes.append(
            {
                "scene_id": f"scene_{index + 1}",
                "start": start,
                "end": end,
                "title": scene_title,
                "subtitle": subtitle,
                "narration": narration,
                "visual_type": "class_object_diagram"
                if index == 0
                else "concept_diagram",
                "elements": _default_scene_elements(index, left, right),
                "animation_steps": DEFAULT_ANIMATION_STEPS,
            }
        )
    spec = {
        "title": title,
        "duration_seconds": duration,
        "resolution": {"width": 1280, "height": 720},
        "style": {
            "theme": "clean academic",
            "background": "#f8fafc",
            "font": "Microsoft YaHei, Arial, sans-serif",
        },
        "scenes": scenes,
    }
    if reason:
        spec["generation_note"] = reason
    return spec


def _planner_prompt(state: LearningState) -> str:
    script = str(state.get("video_script_markdown") or "").strip()
    return (
        "Plan a concise teaching animation / MP4 specification.\n"
        "Prefer the existing video_script_markdown when available. Return JSON only.\n\n"
        f"## User question\n{_last_human_query(state)}\n\n"
        f"## expanded_keypoints\n{_format_keypoints(state)}\n\n"
        f"## context\n{_format_context(state.get('context', []))}\n\n"
        f"## existing video_script_markdown\n{script[:5000] if script else 'None'}\n\n"
        "## Required JSON shape\n"
        "{\n"
        '  "title": "...",\n'
        '  "duration_seconds": 60,\n'
        '  "resolution": {"width": 1280, "height": 720},\n'
        '  "style": {"theme": "clean academic", "background": "#f8fafc", "font": "Microsoft YaHei, Arial, sans-serif"},\n'
        '  "scenes": [\n'
        '    {"scene_id":"scene_1","start":0,"end":8,"title":"...","subtitle":"...",'
        '"narration":"...","visual_type":"concept_diagram","elements":[{"type":"box","text":"...","x":120,"y":180}],'
        '"animation_steps":["fade_in","move","highlight","arrow_draw","code_highlight","fade_out"]}\n'
        "  ]\n"
        "}\n"
        "Limit duration_seconds to <= 90. Use 5-8 scenes when possible."
    )


def _agent_prompt(state: LearningState, draft_spec: dict) -> str:
    return (
        "Generate the final animation_spec JSON for a rendered teaching animation.\n"
        "Return JSON only. No Markdown, no explanation.\n\n"
        "Hard requirements:\n"
        "- duration_seconds <= 90\n"
        "- scenes count between 5 and 8\n"
        "- every scene has start/end/title/subtitle/narration/visual_type/elements\n"
        "- every scene has animation_steps including fade_in, move, highlight, arrow_draw, code_highlight, fade_out\n"
        "- elements support box, arrow, text, circle\n"
        "- times must be ordered and non-overlapping\n\n"
        "For Python OOP, include these scene topics by default: 类与对象, __init__ 初始化, 实例方法调用, 继承, 多态, 封装, 总结.\n\n"
        f"## User question\n{_last_human_query(state)}\n\n"
        f"## Draft spec\n{json.dumps(draft_spec, ensure_ascii=False, indent=2)}\n\n"
        f"## Existing video script\n{str(state.get('video_script_markdown') or '')[:5000] or 'None'}"
    )


def _parse_json_object(text: str) -> dict:
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        try:
            parsed, _ = decoder.raw_decode(raw[match.start() :])
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_scene(
    scene: dict, index: int, cursor: float, duration: float
) -> tuple[dict, float]:
    start = _coerce_float(scene.get("start"), cursor)
    end = _coerce_float(scene.get("end"), start + duration)
    if end <= start:
        end = start + duration
    title = str(scene.get("title") or f"场景 {index + 1}").strip()
    subtitle = str(scene.get("subtitle") or scene.get("visual") or title).strip()
    narration = str(
        scene.get("narration") or scene.get("voiceover") or subtitle
    ).strip()
    elements = scene.get("elements") if isinstance(scene.get("elements"), list) else []
    if not elements:
        elements = _default_scene_elements(index, title, subtitle)
    animation_steps = scene.get("animation_steps")
    if not isinstance(animation_steps, list) or not animation_steps:
        animation_steps = DEFAULT_ANIMATION_STEPS
    else:
        animation_steps = [
            str(item).strip() for item in animation_steps if str(item).strip()
        ] or DEFAULT_ANIMATION_STEPS
    normalized = {
        "scene_id": str(
            scene.get("scene_id") or scene.get("id") or f"scene_{index + 1}"
        ),
        "start": round(start, 3),
        "end": round(end, 3),
        "title": title,
        "subtitle": subtitle,
        "narration": narration,
        "visual_type": str(scene.get("visual_type") or "concept_diagram"),
        "elements": elements,
        "animation_steps": animation_steps,
    }
    return normalized, end


def _ensure_animation_spec(spec: dict, state: LearningState, reason: str = "") -> dict:
    if not isinstance(spec, dict) or not spec:
        return _fallback_animation_spec(state, reason or "animation spec was empty")

    title = str(spec.get("title") or _topic_title(state)).strip()
    duration_seconds = min(
        90.0, max(1.0, _coerce_float(spec.get("duration_seconds"), 70.0))
    )
    resolution = (
        spec.get("resolution") if isinstance(spec.get("resolution"), dict) else {}
    )
    style = spec.get("style") if isinstance(spec.get("style"), dict) else {}
    raw_scenes = spec.get("scenes") if isinstance(spec.get("scenes"), list) else []
    if len(raw_scenes) < 3:
        return _fallback_animation_spec(
            state, reason or "animation spec had fewer than three scenes"
        )

    raw_scenes = raw_scenes[:8]
    scene_budget = duration_seconds / max(1, len(raw_scenes))
    scenes: list[dict] = []
    cursor = 0.0
    for index, scene in enumerate(raw_scenes):
        if not isinstance(scene, dict):
            continue
        normalized_scene, cursor = _normalize_scene(scene, index, cursor, scene_budget)
        if normalized_scene["start"] >= duration_seconds:
            break
        normalized_scene["end"] = min(float(normalized_scene["end"]), duration_seconds)
        scenes.append(normalized_scene)
    if len(scenes) < 3:
        return _fallback_animation_spec(
            state, reason or "animation spec had invalid scenes"
        )

    return {
        "title": title,
        "duration_seconds": round(duration_seconds, 3),
        "resolution": {
            "width": _coerce_int(resolution.get("width"), 1280),
            "height": _coerce_int(resolution.get("height"), 720),
        },
        "style": {
            "theme": str(style.get("theme") or "clean academic"),
            "background": str(style.get("background") or "#f8fafc"),
            "font": str(style.get("font") or "Microsoft YaHei, Arial, sans-serif"),
        },
        "scenes": scenes,
    }


def _topic_terms(state: dict) -> list[str]:
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


def _is_topic_relevant(spec: dict, state: dict) -> bool:
    text = json.dumps(spec, ensure_ascii=False).lower()
    subject = str(state.get("primary_subject") or "").strip().lower()
    if subject == "python":
        return "python" in text or "类" in text or "对象" in text or "面向对象" in text
    terms = _topic_terms(state)
    return True if not terms else any(term.lower() in text for term in terms)


def _local_check_video_animation(spec: dict, state: dict) -> dict:
    failed_reasons: list[str] = []
    if not isinstance(spec, dict):
        return {
            "passed": False,
            "failed_reasons": ["animation_spec is not a dict"],
            "scene_count": 0,
            "topic_relevant": False,
        }
    title = str(spec.get("title") or "").strip()
    duration = _coerce_float(spec.get("duration_seconds"), 0.0)
    scenes = spec.get("scenes") if isinstance(spec.get("scenes"), list) else []
    if not title:
        failed_reasons.append("missing title")
    if duration <= 0 or duration > 90:
        failed_reasons.append("duration_seconds must be > 0 and <= 90")
    if len(scenes) < 3:
        failed_reasons.append("scenes count must be >= 3")

    previous_end = -1.0
    required_scene_fields = ("start", "end", "title", "subtitle", "narration")
    for index, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            failed_reasons.append(f"scene {index + 1} is not a dict")
            continue
        missing = [
            field for field in required_scene_fields if scene.get(field) in {None, ""}
        ]
        if missing:
            failed_reasons.append(
                f"scene {index + 1} missing fields: {', '.join(missing)}"
            )
        start = _coerce_float(scene.get("start"), -1.0)
        end = _coerce_float(scene.get("end"), -1.0)
        if start < previous_end:
            failed_reasons.append(f"scene {index + 1} overlaps previous scene")
        if end <= start:
            failed_reasons.append(f"scene {index + 1} end must be greater than start")
        previous_end = max(previous_end, end)
    if scenes and previous_end > duration:
        failed_reasons.append("last scene end exceeds duration_seconds")

    topic_relevant = _is_topic_relevant(spec, state)
    if not topic_relevant:
        failed_reasons.append("animation spec is not relevant to user topic")

    return {
        "passed": not failed_reasons,
        "failed_reasons": failed_reasons,
        "scene_count": len(scenes),
        "duration_seconds": duration,
        "topic_relevant": topic_relevant,
    }


@traced_node
async def video_animation_planner(state: LearningState) -> dict:
    draft_spec = _fallback_animation_spec(state, "planner fallback seed")
    try:
        response = await invoke_plain_llm_fail_fast(
            node_name="video_animation_planner",
            llm_node="video_animation",
            messages=[
                SystemMessage(
                    content="You are a teaching animation planner. Return JSON only."
                ),
                HumanMessage(content=_planner_prompt(state)),
            ],
            state=state,
            temperature=get_setting("video_animation.temperature", 0.2),
        )
        parsed = _parse_json_object(response)
        if parsed:
            draft_spec = _ensure_animation_spec(
                parsed, state, "planner LLM spec was invalid"
            )
    except Exception as exc:
        logger.warning("video_animation_planner fallback used: %s", exc)

    emit_a3_trace(
        logger,
        "video_animation_planner",
        {
            "has_video_script_markdown": bool(state.get("video_script_markdown")),
            "scene_count": len(draft_spec.get("scenes") or []),
            "duration_seconds": draft_spec.get("duration_seconds", 0),
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "video_animation_spec": draft_spec,
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
    draft_spec = state.get("video_animation_spec") or _fallback_animation_spec(
        state, "agent received empty draft"
    )
    round_no = int(state.get("video_animation_round", 0) or 0) + 1
    fallback_used = False
    try:
        response = await invoke_plain_llm_fail_fast(
            node_name="video_animation_agent",
            llm_node="video_animation",
            messages=[
                SystemMessage(
                    content="You create JSON animation specs for deterministic HTML/MP4 rendering."
                ),
                HumanMessage(content=_agent_prompt(state, draft_spec)),
            ],
            state=state,
            temperature=get_setting("video_animation.temperature", 0.2),
        )
        parsed = _parse_json_object(response)
        spec = (
            _ensure_animation_spec(
                parsed, state, "agent LLM returned invalid animation spec"
            )
            if parsed
            else {}
        )
        if not spec:
            raise ValueError("agent returned no JSON object")
    except Exception as exc:
        fallback_used = True
        logger.warning("video_animation_agent fallback used: %s", exc)
        spec = _ensure_animation_spec(draft_spec, state, f"{type(exc).__name__}: {exc}")

    emit_a3_trace(
        logger,
        "video_animation_agent",
        {
            "round": round_no,
            "fallback_used": fallback_used,
            "scene_count": len(spec.get("scenes") or []),
            "duration_seconds": spec.get("duration_seconds", 0),
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "video_animation_spec": spec,
        "video_animation_round": round_no,
        "video_animation_review_verdict": "",
        "video_animation_review_reason": "",
    }


@traced_node
async def video_animation_reviewer(state: LearningState) -> dict:
    spec = state.get("video_animation_spec") or {}
    local_check = _local_check_video_animation(spec, state)
    verdict = "approve" if local_check["passed"] else "reject"
    reason = (
        "local check passed"
        if local_check["passed"]
        else "; ".join(local_check["failed_reasons"])
    )

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
    if state.get("video_animation_review_verdict") == "approve":
        return "output"
    current_round = int(state.get("video_animation_round", 0) or 0)
    if current_round < 2:
        return "rewrite"
    return "output"


@traced_node
async def video_animation_rewrite(state: LearningState) -> dict:
    spec = _ensure_animation_spec(
        state.get("video_animation_spec") or {},
        state,
        state.get("video_animation_review_reason", "") or "reviewer requested rewrite",
    )
    return {
        "video_animation_spec": spec,
        "video_animation_revision_notes": state.get(
            "video_animation_review_reason", ""
        ),
    }


def _read_artifact_html(artifact: dict) -> str:
    artifact_id = str(artifact.get("artifact_id") or "").strip()
    filename = str(artifact.get("html_filename") or "").strip()
    if not artifact_id or not filename:
        return ""
    root = get_video_animation_artifact_dir()
    path = (root / artifact_id / filename).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return ""
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


@traced_node
async def video_animation_output(state: LearningState) -> dict:
    spec = _ensure_animation_spec(
        state.get("video_animation_spec") or {}, state, "output received invalid spec"
    )
    title = str(spec.get("title") or _topic_title(state)).strip()
    render_log = ""
    html = ""
    render_mode = (
        "test"
        if str(state.get("video_animation_render_mode") or "").strip().lower() == "test"
        else "production"
    )
    try:
        artifact = await create_video_animation_artifact_async(
            animation_spec=spec,
            title=title,
            fps=12,
            width=1280,
            height=720,
            max_duration_seconds=int(
                get_setting("video_animation.max_duration_seconds", 90) or 90
            ),
            render_mode=render_mode,
        )
        render_log = str(artifact.get("render_log") or "")
        html = _read_artifact_html(artifact)
    except Exception as exc:
        logger.exception("video_animation_output failed to create artifact")
        render_log = f"{type(exc).__name__}: {exc}"
        artifact = {
            "title": title,
            "artifact_id": "",
            "html_filename": "",
            "json_filename": "",
            "srt_filename": "",
            "mp4_filename": "",
            "html_url": "",
            "json_url": "",
            "srt_url": "",
            "mp4_url": "",
            "render_mode": render_mode,
            "render_label": "5秒测试视频"
            if render_mode == "test"
            else "正式教学动画视频",
            "is_preview_video": render_mode == "test",
            "video_valid_for_teaching": False,
            "duration_seconds": spec.get("duration_seconds", 0),
            "full_duration_seconds": spec.get("duration_seconds", 0),
            "render_duration_seconds": 0,
            "fps": 12,
            "width": 1280,
            "height": 720,
            "frame_count": 0,
            "ffmpeg_path": "",
            "playwright_available": False,
            "html_available": False,
            "json_available": False,
            "srt_available": False,
            "mp4_available": False,
            "mp4_exists": False,
            "mp4_file_size": 0,
            "render_success": False,
            "render_log": render_log,
        }

    content = (
        f"# {title}\n\n"
        "已生成教学动画资源。\n\n"
        f"- HTML 预览：{artifact.get('html_url') or '未生成'}\n"
        f"- JSON 规格：{artifact.get('json_url') or '未生成'}\n"
        f"- 字幕 SRT：{artifact.get('srt_url') or '未生成'}\n"
        f"- MP4 视频：{artifact.get('mp4_url') if artifact.get('render_success') else 'MP4 渲染未成功，可先使用 HTML 预览'}\n"
    )
    emit_a3_trace(
        logger,
        "video_animation_output",
        {
            "title": title,
            "artifact_id": artifact.get("artifact_id", ""),
            "render_success": bool(artifact.get("render_success")),
            "render_mode": artifact.get("render_mode", ""),
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
            "mp4_available": bool(artifact.get("mp4_available")),
            "html_url": artifact.get("html_url", ""),
            "mp4_url": artifact.get("mp4_url", ""),
            "render_log_chars": len(render_log),
            "emits_ai_message": True,
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "video_animation_artifact": artifact,
        "video_animation_spec": spec,
        "video_animation_html": html,
        "video_animation_render_log": render_log,
        "messages": [AIMessage(content=content)],
    }
