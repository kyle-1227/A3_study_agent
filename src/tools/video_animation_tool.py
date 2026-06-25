"""Artifact helpers for rendered teaching animations."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from src.tools.document_tool import _safe_filename_stem

DEFAULT_ARTIFACT_ROOT = Path(__file__).resolve().parents[2] / "artifacts"
DEFAULT_VIDEO_ANIMATION_ARTIFACT_DIR = DEFAULT_ARTIFACT_ROOT / "video-animations"
VIDEO_ANIMATION_URL_PREFIX = "/artifacts/video-animations"
TEST_RENDER_SECONDS = 5
TEST_RENDER_FPS = 12
PRODUCTION_RENDER_SECONDS = 30
PRODUCTION_RENDER_FPS = 24
DEFAULT_ANIMATION_STEPS = [
    "fade_in",
    "move",
    "highlight",
    "arrow_draw",
    "code_highlight",
    "fade_out",
]


def get_video_animation_artifact_dir() -> Path:
    """Return the directory used for generated animation artifacts."""
    root = Path(
        os.getenv(
            "VIDEO_ANIMATION_ARTIFACT_DIR", str(DEFAULT_VIDEO_ANIMATION_ARTIFACT_DIR)
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def create_video_animation_artifact(
    animation_spec: dict,
    title: str,
    srt_text: str | None = None,
    fps: int = 12,
    width: int = 1280,
    height: int = 720,
    max_duration_seconds: int = 90,
    render_mode: str = "production",
) -> dict:
    """Synchronous compatibility wrapper for non-async scripts."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            create_video_animation_artifact_async(
                animation_spec=animation_spec,
                title=title,
                srt_text=srt_text,
                fps=fps,
                width=width,
                height=height,
                max_duration_seconds=max_duration_seconds,
                render_mode=render_mode,
            )
        )
    raise RuntimeError(
        "create_video_animation_artifact cannot run inside an active asyncio loop; "
        "await create_video_animation_artifact_async instead."
    )


async def create_video_animation_artifact_async(
    animation_spec: dict,
    title: str,
    srt_text: str | None = None,
    fps: int = 12,
    width: int = 1280,
    height: int = 720,
    max_duration_seconds: int = 90,
    render_mode: str = "production",
) -> dict:
    """Create HTML/JSON/SRT artifacts and render an MP4 when local tools exist."""
    normalized_render_mode = (
        "test" if str(render_mode).strip().lower() == "test" else "production"
    )
    normalized_fps = (
        TEST_RENDER_FPS if normalized_render_mode == "test" else PRODUCTION_RENDER_FPS
    )
    normalized_width = 1280
    normalized_height = 720
    max_duration = _clamp_int(max_duration_seconds, default=90, minimum=1, maximum=600)
    safe_title = str(title or "").strip() or str(
        (animation_spec or {}).get("title") or "teaching-animation"
    )
    filename_stem = _safe_filename_stem(safe_title, default="teaching-animation")

    artifact_id = uuid.uuid4().hex
    root = get_video_animation_artifact_dir()
    artifact_dir = _safe_child_dir(root, artifact_id)
    frames_dir = _safe_child_dir(artifact_dir, "frames")
    result_path = _safe_child_file(artifact_dir, "render_result.json")

    html_filename = f"{filename_stem}.html"
    json_filename = f"{filename_stem}.json"
    srt_filename = f"{filename_stem}.srt"
    mp4_filename = f"{filename_stem}.mp4"

    html_path = _safe_child_file(artifact_dir, html_filename)
    json_path = _safe_child_file(artifact_dir, json_filename)
    srt_path = _safe_child_file(artifact_dir, srt_filename)
    mp4_path = _safe_child_file(artifact_dir, mp4_filename)

    normalized_spec = _normalize_animation_spec(
        animation_spec=animation_spec or {},
        title=safe_title,
        fps=normalized_fps,
        width=normalized_width,
        height=normalized_height,
        max_duration_seconds=max_duration,
    )
    full_duration_seconds = max(1, int(float(normalized_spec["duration_seconds"])))
    if (
        normalized_render_mode == "production"
        and full_duration_seconds < PRODUCTION_RENDER_SECONDS
    ):
        full_duration_seconds = PRODUCTION_RENDER_SECONDS
        normalized_spec["duration_seconds"] = full_duration_seconds
    duration_seconds = normalized_spec["duration_seconds"]
    html = _render_animation_html(normalized_spec)
    srt = _build_srt(normalized_spec["scenes"], srt_text)

    html_path.write_text(html, encoding="utf-8")
    json_path.write_text(
        json.dumps(normalized_spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    srt_path.write_text(srt.rstrip() + "\n", encoding="utf-8")

    if normalized_render_mode == "test":
        render_duration_seconds = TEST_RENDER_SECONDS
    else:
        render_duration_seconds = min(full_duration_seconds, PRODUCTION_RENDER_SECONDS)
    is_preview_video = normalized_render_mode == "test"
    video_valid_for_teaching = (
        normalized_render_mode == "production"
        and render_duration_seconds >= PRODUCTION_RENDER_SECONDS
    )
    render_result = await render_html_animation_to_mp4_async(
        html_path=html_path,
        mp4_path=mp4_path,
        frames_dir=frames_dir,
        result_path=result_path,
        fps=normalized_fps,
        width=normalized_width,
        height=normalized_height,
        render_duration_seconds=render_duration_seconds,
    )
    mp4_exists = mp4_path.is_file()
    mp4_file_size = mp4_path.stat().st_size if mp4_exists else 0
    render_success = bool(
        render_result["render_success"] and mp4_exists and mp4_file_size > 0
    )
    mp4_available = render_success
    render_log = str(render_result["render_log"])
    html_available = html_path.is_file()
    json_available = json_path.is_file()
    srt_available = srt_path.is_file()

    return {
        "artifact_id": artifact_id,
        "title": safe_title,
        "html_filename": html_filename,
        "json_filename": json_filename,
        "srt_filename": srt_filename,
        "mp4_filename": mp4_filename if mp4_available else "",
        "html_url": f"{VIDEO_ANIMATION_URL_PREFIX}/{artifact_id}/{html_filename}",
        "json_url": f"{VIDEO_ANIMATION_URL_PREFIX}/{artifact_id}/{json_filename}",
        "srt_url": f"{VIDEO_ANIMATION_URL_PREFIX}/{artifact_id}/{srt_filename}",
        "mp4_url": f"{VIDEO_ANIMATION_URL_PREFIX}/{artifact_id}/{mp4_filename}"
        if mp4_available
        else "",
        "render_mode": normalized_render_mode,
        "render_label": "5秒测试视频" if is_preview_video else "正式教学动画视频",
        "is_preview_video": is_preview_video,
        "video_valid_for_teaching": video_valid_for_teaching and render_success,
        "duration_seconds": duration_seconds,
        "full_duration_seconds": full_duration_seconds,
        "render_duration_seconds": render_duration_seconds,
        "fps": normalized_fps,
        "width": normalized_width,
        "height": normalized_height,
        "frame_count": int(render_result["frame_count"]),
        "ffmpeg_path": str(render_result["ffmpeg_path"]),
        "playwright_available": bool(render_result["playwright_available"]),
        "html_available": html_available,
        "json_available": json_available,
        "srt_available": srt_available,
        "mp4_available": mp4_available,
        "mp4_exists": mp4_exists,
        "mp4_file_size": mp4_file_size,
        "render_success": render_success,
        "render_log": render_log,
    }


def _safe_child_dir(parent: Path, name: str) -> Path:
    child = (parent / name).resolve()
    child.relative_to(parent.resolve())
    child.mkdir(parents=True, exist_ok=True)
    return child


def _safe_child_file(parent: Path, filename: str) -> Path:
    path = (parent / filename).resolve()
    path.relative_to(parent.resolve())
    return path


def _clamp_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(minimum, min(maximum, parsed))


def _normalize_animation_spec(
    *,
    animation_spec: dict,
    title: str,
    fps: int,
    width: int,
    height: int,
    max_duration_seconds: int,
) -> dict:
    raw_scenes = (
        animation_spec.get("scenes") if isinstance(animation_spec, dict) else []
    )
    scenes = [
        _normalize_scene(item, index)
        for index, item in enumerate(raw_scenes or [])
        if isinstance(item, dict)
    ]
    if not scenes:
        scenes = _fallback_scenes(animation_spec, title)

    requested_duration = _float_or_none(animation_spec.get("duration_seconds"))
    has_explicit_timing = any(
        scene.get("requested_start") is not None
        or scene.get("requested_end") is not None
        for scene in scenes
    )
    if (
        requested_duration is not None
        and requested_duration > 0
        and not has_explicit_timing
    ):
        duration_budget = min(float(max_duration_seconds), requested_duration)
        per_scene = duration_budget / max(1, len(scenes))
        for scene in scenes:
            scene["duration_seconds"] = per_scene

    normalized_scenes: list[dict] = []
    cursor = 0.0
    for scene in scenes:
        requested_start = _float_or_none(scene.pop("requested_start", None))
        requested_end = _float_or_none(scene.pop("requested_end", None))
        start = (
            requested_start
            if requested_start is not None and requested_start >= 0
            else cursor
        )
        if start >= max_duration_seconds:
            break
        if requested_end is not None and requested_end > start:
            duration = requested_end - start
        else:
            duration = max(1.0, float(scene.get("duration_seconds") or 6.0))
        duration = max(1.0, duration)
        duration = min(duration, max_duration_seconds - start)
        normalized = dict(scene)
        normalized["start"] = round(start, 3)
        normalized["end"] = round(start + duration, 3)
        normalized["duration_seconds"] = round(duration, 3)
        normalized_scenes.append(normalized)
        cursor = start + duration

    if not normalized_scenes:
        normalized_scenes = _fallback_scenes(animation_spec, title)[:1]
        normalized_scenes[0]["start"] = 0.0
        normalized_scenes[0]["end"] = float(min(max_duration_seconds, 8))
        normalized_scenes[0]["duration_seconds"] = normalized_scenes[0]["end"]

    last_scene_end = max(float(scene.get("end") or 0.0) for scene in normalized_scenes)
    duration_seconds = (
        requested_duration
        if requested_duration and requested_duration > 0
        else last_scene_end
    )
    duration_seconds = round(
        min(float(max_duration_seconds), max(1.0, duration_seconds)), 3
    )
    style = _normalize_style(
        animation_spec.get("style") if isinstance(animation_spec, dict) else {}
    )
    return {
        "title": title,
        "topic": str(
            animation_spec.get("topic") or animation_spec.get("subject") or title
        ),
        "style": style,
        "theme": style["theme"],
        "background": style["background"],
        "font": style["font"],
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
        "scenes": normalized_scenes,
        "source_spec": animation_spec,
    }


def _normalize_scene(scene: dict, index: int) -> dict:
    title = str(scene.get("title") or scene.get("name") or f"Scene {index + 1}").strip()
    subtitle = str(scene.get("subtitle") or "").strip()
    narration = str(
        scene.get("narration") or scene.get("voiceover") or scene.get("text") or title
    ).strip()
    visual = str(
        scene.get("visual")
        or scene.get("visual_description")
        or scene.get("description")
        or subtitle
        or ""
    ).strip()
    bullets = (
        scene.get("bullets") or scene.get("key_points") or scene.get("points") or []
    )
    if not isinstance(bullets, list):
        bullets = [str(bullets)]
    if subtitle and not bullets:
        bullets = [subtitle]
    duration = _float_or_none(
        scene.get("duration_seconds") or scene.get("duration") or scene.get("seconds")
    )
    start = _float_or_none(scene.get("start"))
    end = _float_or_none(scene.get("end"))
    if duration is None and start is not None and end is not None and end > start:
        duration = end - start
    animation_steps = scene.get("animation_steps")
    if not isinstance(animation_steps, list) or not animation_steps:
        animation_steps = DEFAULT_ANIMATION_STEPS
    else:
        animation_steps = [
            str(item).strip() for item in animation_steps if str(item).strip()
        ] or DEFAULT_ANIMATION_STEPS
    return {
        "scene_id": str(
            scene.get("scene_id") or scene.get("id") or f"scene_{index + 1}"
        ),
        "title": title,
        "subtitle": subtitle,
        "narration": narration,
        "visual": visual or narration,
        "visual_type": str(scene.get("visual_type") or ""),
        "elements": _normalize_elements(scene.get("elements") or []),
        "animation_steps": animation_steps,
        "bullets": [str(item).strip() for item in bullets if str(item).strip()][:5],
        "duration_seconds": duration or 6.0,
        "accent": str(scene.get("accent") or ""),
        "requested_start": start,
        "requested_end": end,
    }


def _normalize_elements(elements: Any) -> list[dict]:
    if not isinstance(elements, list):
        return []
    normalized: list[dict] = []
    for item in elements[:30]:
        if not isinstance(item, dict):
            continue
        element_type = str(item.get("type") or "").strip().lower()
        if element_type not in {"box", "arrow", "text", "circle"}:
            continue
        normalized_item: dict[str, Any] = {"type": element_type}
        if element_type in {"box", "text", "circle"}:
            normalized_item.update(
                {
                    "text": str(item.get("text") or item.get("label") or "").strip(),
                    "x": _clamp_float(
                        item.get("x"), default=80.0, minimum=0.0, maximum=1200.0
                    ),
                    "y": _clamp_float(
                        item.get("y"), default=80.0, minimum=0.0, maximum=650.0
                    ),
                    "width": _clamp_float(
                        item.get("width"), default=220.0, minimum=40.0, maximum=520.0
                    ),
                    "height": _clamp_float(
                        item.get("height"), default=72.0, minimum=30.0, maximum=240.0
                    ),
                }
            )
        elif element_type == "arrow":
            normalized_item.update(
                {
                    "from": str(item.get("from") or item.get("source") or "").strip(),
                    "to": str(item.get("to") or item.get("target") or "").strip(),
                    "text": str(item.get("text") or item.get("label") or "").strip(),
                }
            )
        normalized.append(normalized_item)
    return normalized


def _normalize_style(style: Any) -> dict:
    if isinstance(style, dict):
        return {
            "theme": str(style.get("theme") or "clean academic"),
            "background": _safe_css_color(str(style.get("background") or "#f8fafc")),
            "font": _safe_css_font(
                str(style.get("font") or "Microsoft YaHei, Arial, sans-serif")
            ),
        }
    text = str(style or "").strip()
    return {
        "theme": text or "clean academic",
        "background": "#f8fafc",
        "font": "Microsoft YaHei, Arial, sans-serif",
    }


def _clamp_float(
    value: Any, *, default: float, minimum: float, maximum: float
) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    return max(minimum, min(maximum, parsed))


def _safe_css_color(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("#") and len(stripped) in {4, 7}:
        return (
            stripped
            if all(ch in "#0123456789abcdefABCDEF" for ch in stripped)
            else "#f8fafc"
        )
    if stripped.lower() in {"white", "black", "transparent"}:
        return stripped.lower()
    return "#f8fafc"


def _safe_css_font(value: str) -> str:
    cleaned = "".join(ch for ch in value if ch not in "{};<>")
    return cleaned.strip()[:160] or "Microsoft YaHei, Arial, sans-serif"


def _fallback_scenes(animation_spec: dict, title: str) -> list[dict]:
    topic = str(animation_spec.get("topic") or animation_spec.get("subject") or title)
    concepts = animation_spec.get("concepts") or animation_spec.get("keypoints") or []
    if not isinstance(concepts, list):
        concepts = [str(concepts)]
    concepts = [str(item).strip() for item in concepts if str(item).strip()]
    if not concepts:
        concepts = ["core idea", "visual example", "summary"]
    return [
        {
            "title": f"{topic}: overview",
            "narration": f"Introduce {topic} with a simple visual overview.",
            "visual": "Title card with topic keywords and a moving focus marker.",
            "bullets": concepts[:3],
            "duration_seconds": 8.0,
            "accent": "#2f6f4e",
        },
        {
            "title": "Key concept animation",
            "narration": "Break the concept into parts and show how they connect.",
            "visual": "Cards move into a connected diagram.",
            "bullets": concepts[:5],
            "duration_seconds": 10.0,
            "accent": "#2767a8",
        },
        {
            "title": "Practice and recap",
            "narration": "End with a quick recap and one self-check question.",
            "visual": "Checklist appears with progress animation.",
            "bullets": ["recap", "self-check", "next step"],
            "duration_seconds": 8.0,
            "accent": "#8a5a20",
        },
    ]


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _render_animation_html(spec: dict) -> str:
    data_json = json.dumps(spec, ensure_ascii=False).replace("<", "\\u003c")
    background = _safe_css_color(str(spec.get("background") or "#f8fafc"))
    font = _safe_css_font(str(spec.get("font") or "Microsoft YaHei, Arial, sans-serif"))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_escape_html(str(spec["title"]))}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: {background};
      --ink: #17231b;
      --muted: #526256;
      --accent: #2f6f4e;
      --panel: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: var(--bg);
      color: var(--ink);
      font-family: {font};
    }}
    .stage {{
      position: relative;
      width: {int(spec["width"])}px;
      height: {int(spec["height"])}px;
      overflow: hidden;
      background:
        radial-gradient(circle at 14% 18%, rgba(47,111,78,.16), transparent 26%),
        linear-gradient(135deg, var(--bg) 0%, #eef5ef 100%);
    }}
    .content {{
      position: absolute;
      inset: 54px 70px;
      display: grid;
      grid-template-columns: 1fr 420px;
      gap: 42px;
      align-items: center;
    }}
    .kicker {{ color: var(--accent); font-size: 28px; font-weight: 700; letter-spacing: .02em; }}
    h1 {{ margin: 18px 0 22px; font-size: 58px; line-height: 1.08; max-width: 760px; }}
    .narration {{ color: var(--muted); font-size: 30px; line-height: 1.48; max-width: 760px; }}
    .bullets {{ margin-top: 28px; display: grid; gap: 14px; }}
    .bullet {{
      width: fit-content;
      max-width: 760px;
      padding: 12px 18px;
      border-radius: 14px;
      background: rgba(255,255,255,.72);
      border: 1px solid rgba(23,35,27,.12);
      font-size: 26px;
    }}
    .visual {{
      position: relative;
      height: 520px;
      border-radius: 26px;
      background: rgba(255,255,255,.76);
      border: 1px solid rgba(23,35,27,.12);
      box-shadow: 0 26px 80px rgba(20,40,24,.12);
      overflow: hidden;
    }}
    .orb {{
      position: absolute;
      width: 180px;
      height: 180px;
      border-radius: 999px;
      background: var(--accent);
      opacity: .16;
      transform: translate(calc(var(--p) * 170px), calc(var(--p) * 90px));
      left: 28px;
      top: 32px;
    }}
    .card {{
      position: absolute;
      left: 52px;
      right: 52px;
      min-height: 86px;
      padding: 20px 22px;
      border-radius: 18px;
      background: var(--panel);
      border: 1px solid rgba(23,35,27,.13);
      box-shadow: 0 18px 50px rgba(20,40,24,.10);
      font-size: 25px;
      transform:
        translateX(calc((var(--i) - 1) * var(--move-p) * 28px))
        translateY(calc((1 - var(--fade-p)) * 18px));
      opacity: calc(var(--fade-p) * (1 - var(--fade-out-p) * .45));
    }}
    .card.one {{ top: 120px; --i: 1; }}
    .card.two {{ top: 230px; --i: 2; }}
    .card.three {{ top: 340px; --i: 3; }}
    .element-layer {{
      position: absolute;
      inset: 0;
      pointer-events: none;
    }}
    .spec-box, .spec-text, .spec-circle {{
      position: absolute;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 12px 16px;
      border: 2px solid var(--accent);
      border-radius: 16px;
      background: rgba(255,255,255,.92);
      box-shadow: 0 14px 42px rgba(20,40,24,.12);
      color: var(--ink);
      font-weight: 700;
      text-align: center;
      transform:
        translateY(calc((1 - var(--move-p)) * 18px))
        scale(calc(.96 + var(--highlight-p) * .04));
      opacity: calc((.2 + var(--fade-p) * .8) * (1 - var(--fade-out-p) * .5));
    }}
    .spec-box.highlight, .spec-circle.highlight {{
      background: color-mix(in srgb, var(--accent) 16%, white);
      box-shadow: 0 18px 58px rgba(47,111,78,.24);
    }}
    .code-highlight {{
      outline: 4px solid color-mix(in srgb, var(--accent) 45%, white);
      outline-offset: 3px;
    }}
    .spec-text {{
      border-color: transparent;
      background: transparent;
      box-shadow: none;
      justify-content: flex-start;
    }}
    .spec-circle {{
      border-radius: 999px;
    }}
    .spec-arrow-layer {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      overflow: visible;
    }}
    .progress {{
      position: absolute;
      left: 70px;
      right: 70px;
      bottom: 38px;
      height: 10px;
      border-radius: 99px;
      background: rgba(23,35,27,.12);
      overflow: hidden;
    }}
    .bar {{ height: 100%; width: calc(var(--global-p) * 100%); background: var(--accent); }}
  </style>
</head>
<body>
  <main class="stage" id="stage">
    <section class="content">
      <div>
        <div class="kicker" id="sceneIndex"></div>
        <h1 id="title"></h1>
        <div class="narration" id="narration"></div>
        <div class="bullets" id="bullets"></div>
      </div>
      <div class="visual" id="visual">
        <div class="orb"></div>
        <div class="card one" id="cardOne"></div>
        <div class="card two" id="cardTwo"></div>
        <div class="card three" id="cardThree"></div>
        <div class="element-layer" id="elementLayer"></div>
      </div>
    </section>
    <div class="progress"><div class="bar"></div></div>
  </main>
  <script>
    const DATA = {data_json};
    const root = document.documentElement;
    const titleEl = document.getElementById("title");
    const narrationEl = document.getElementById("narration");
    const sceneIndexEl = document.getElementById("sceneIndex");
    const bulletsEl = document.getElementById("bullets");
    const visualEl = document.getElementById("visual");
    const cardOne = document.getElementById("cardOne");
    const cardTwo = document.getElementById("cardTwo");
    const cardThree = document.getElementById("cardThree");
    const elementLayer = document.getElementById("elementLayer");

    function clamp(value, min, max) {{
      return Math.max(min, Math.min(max, value));
    }}

    function sceneAt(t) {{
      const scenes = DATA.scenes || [];
      return scenes.find((scene) => t >= scene.start && t < scene.end) || scenes[scenes.length - 1] || {{}};
    }}

    function stepProgress(p, start, end) {{
      return clamp((p - start) / Math.max(.001, end - start), 0, 1);
    }}

    window.renderAt = function renderAt(t) {{
      const duration = DATA.duration_seconds || 1;
      const time = clamp(Number(t) || 0, 0, duration);
      const scene = sceneAt(time);
      const sceneDuration = Math.max(.001, (scene.end || duration) - (scene.start || 0));
      const p = clamp((time - (scene.start || 0)) / sceneDuration, 0, 1);
      const globalP = clamp(time / duration, 0, 1);
      const sceneNumber = (DATA.scenes || []).indexOf(scene) + 1;
      const fadeP = stepProgress(p, 0, .16);
      const moveP = stepProgress(p, .08, .42);
      const highlightP = stepProgress(p, .28, .62);
      const arrowP = stepProgress(p, .34, .78);
      const codeP = stepProgress(p, .48, .72);
      const fadeOutP = stepProgress(p, .86, 1);

      root.style.setProperty("--p", p.toFixed(4));
      root.style.setProperty("--global-p", globalP.toFixed(4));
      root.style.setProperty("--fade-p", fadeP.toFixed(4));
      root.style.setProperty("--move-p", moveP.toFixed(4));
      root.style.setProperty("--highlight-p", highlightP.toFixed(4));
      root.style.setProperty("--arrow-p", arrowP.toFixed(4));
      root.style.setProperty("--code-p", codeP.toFixed(4));
      root.style.setProperty("--fade-out-p", fadeOutP.toFixed(4));
      root.style.setProperty("--accent", scene.accent || "#2f6f4e");
      sceneIndexEl.textContent = `Scene ${{sceneNumber || 1}} / ${{(DATA.scenes || []).length || 1}}`;
      titleEl.textContent = scene.title || DATA.title || "Teaching animation";
      narrationEl.textContent = scene.narration || scene.visual || "";
      bulletsEl.replaceChildren(...(scene.bullets || []).slice(0, 5).map((text) => {{
        const item = document.createElement("div");
        item.className = "bullet";
        item.textContent = text;
        return item;
      }}));
      const cards = (scene.bullets && scene.bullets.length ? scene.bullets : [scene.visual, scene.narration, DATA.topic]).filter(Boolean);
      cardOne.textContent = cards[0] || "Concept";
      cardTwo.textContent = cards[1] || "Example";
      cardThree.textContent = cards[2] || "Summary";
      renderSpecElements(scene, {{ p, fadeP, moveP, highlightP, arrowP, codeP, fadeOutP }});
      visualEl.style.transform = `scale(${{1 + moveP * 0.02 - fadeOutP * 0.01}})`;
    }};

    function renderSpecElements(scene, progress) {{
      const elements = Array.isArray(scene.elements) ? scene.elements : [];
      elementLayer.replaceChildren();
      if (!elements.length) return;

      const boxes = new Map();
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("class", "spec-arrow-layer");
      svg.setAttribute("viewBox", "0 0 420 520");
      const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
      const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
      marker.setAttribute("id", "arrowhead");
      marker.setAttribute("markerWidth", "10");
      marker.setAttribute("markerHeight", "7");
      marker.setAttribute("refX", "9");
      marker.setAttribute("refY", "3.5");
      marker.setAttribute("orient", "auto");
      const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      polygon.setAttribute("points", "0 0, 10 3.5, 0 7");
      polygon.setAttribute("fill", getComputedStyle(root).getPropertyValue("--accent") || "#2f6f4e");
      marker.appendChild(polygon);
      defs.appendChild(marker);
      svg.appendChild(defs);

      for (const element of elements) {{
        if (!["box", "text", "circle"].includes(element.type)) continue;
        const node = document.createElement("div");
        node.className = element.type === "box" ? "spec-box" : element.type === "circle" ? "spec-circle" : "spec-text";
        if (progress.highlightP > .25 && element.type !== "text") node.classList.add("highlight");
        if (progress.codeP > .5 && /class|def|self|__init__|\\(|\\)/i.test(element.text || "")) node.classList.add("code-highlight");
        node.textContent = element.text || "";
        node.style.left = `${{(Number(element.x) || 0) * 0.31 + (1 - progress.moveP) * 18}}px`;
        node.style.top = `${{(Number(element.y) || 0) * 0.62 + Math.sin(progress.p * Math.PI) * 8}}px`;
        node.style.width = `${{Math.max(60, (Number(element.width) || 220) * 0.62)}}px`;
        node.style.height = `${{Math.max(34, (Number(element.height) || 72) * 0.62)}}px`;
        elementLayer.appendChild(node);
        if (element.text) {{
          boxes.set(element.text, {{
            x: (Number(element.x) || 0) * 0.31,
            y: (Number(element.y) || 0) * 0.62,
            w: Math.max(60, (Number(element.width) || 220) * 0.62),
            h: Math.max(34, (Number(element.height) || 72) * 0.62),
          }});
        }}
      }}

      for (const element of elements) {{
        if (element.type !== "arrow") continue;
        const from = boxes.get(element.from);
        const to = boxes.get(element.to);
        if (!from || !to) continue;
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        const progressX = from.x + from.w + (to.x - (from.x + from.w)) * progress.arrowP;
        const progressY = from.y + from.h / 2 + (to.y + to.h / 2 - (from.y + from.h / 2)) * progress.arrowP;
        line.setAttribute("x1", String(from.x + from.w));
        line.setAttribute("y1", String(from.y + from.h / 2));
        line.setAttribute("x2", String(progressX));
        line.setAttribute("y2", String(progressY));
        line.setAttribute("stroke", getComputedStyle(root).getPropertyValue("--accent") || "#2f6f4e");
        line.setAttribute("stroke-width", "4");
        line.setAttribute("stroke-linecap", "round");
        line.setAttribute("marker-end", "url(#arrowhead)");
        line.setAttribute("opacity", String(.15 + progress.arrowP * .85));
        svg.appendChild(line);
      }}
      elementLayer.prepend(svg);
    }}

    let start = performance.now();
    function animate(now) {{
      const t = ((now - start) / 1000) % (DATA.duration_seconds || 1);
      window.renderAt(t);
      requestAnimationFrame(animate);
    }}
    window.renderAt(0);
    requestAnimationFrame(animate);
  </script>
</body>
</html>
"""


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _build_srt(scenes: list[dict], srt_text: str | None) -> str:
    if str(srt_text or "").strip():
        return str(srt_text).strip()
    blocks: list[str] = []
    for index, scene in enumerate(scenes, 1):
        start = _format_srt_time(float(scene.get("start") or 0.0))
        end = _format_srt_time(
            float(scene.get("end") or scene.get("duration_seconds") or 1.0)
        )
        text = str(
            scene.get("narration") or scene.get("title") or "Please view the animation."
        ).strip()
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
    return (
        "\n\n".join(blocks)
        or "1\n00:00:00,000 --> 00:00:05,000\nPlease view the animation HTML."
    )


def _format_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    sec = total_seconds % 60
    total_minutes = total_seconds // 60
    minute = total_minutes % 60
    hour = total_minutes // 60
    return f"{hour:02d}:{minute:02d}:{sec:02d},{ms:03d}"


async def render_html_animation_to_mp4_async(
    *,
    html_path: Path,
    mp4_path: Path,
    frames_dir: Path,
    result_path: Path,
    fps: int,
    width: int,
    height: int,
    render_duration_seconds: int,
) -> dict[str, Any]:
    """Render an animation HTML file to MP4 in an isolated worker process."""
    render_duration_seconds = max(
        1, min(int(float(render_duration_seconds or TEST_RENDER_SECONDS)), 60)
    )
    frame_count = max(1, render_duration_seconds * fps)
    started_at = time.monotonic()
    ffmpeg_path = shutil.which("ffmpeg") or ""
    playwright_available = False
    try:
        import playwright.sync_api  # noqa: F401
    except Exception as exc:
        playwright_import_error = f"Playwright import failed: {exc}"
    else:
        playwright_available = True
        playwright_import_error = ""

    if not ffmpeg_path:
        return _render_result(
            render_success=False,
            render_log="ffmpeg not found. Please install ffmpeg and add ffmpeg\\bin to PATH.",
            ffmpeg_path=ffmpeg_path,
            playwright_available=playwright_available,
            frame_count=frame_count,
            render_duration_seconds=render_duration_seconds,
            fps=fps,
            started_at=started_at,
            mp4_path=mp4_path,
        )
    if not playwright_available:
        return _render_result(
            render_success=False,
            render_log=playwright_import_error,
            ffmpeg_path=ffmpeg_path,
            playwright_available=False,
            frame_count=frame_count,
            render_duration_seconds=render_duration_seconds,
            fps=fps,
            started_at=started_at,
            mp4_path=mp4_path,
        )

    project_root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "-m",
        "src.tools.render_animation_worker",
        "--html-path",
        str(html_path),
        "--frames-dir",
        str(frames_dir),
        "--mp4-path",
        str(mp4_path),
        "--ffmpeg-path",
        ffmpeg_path,
        "--fps",
        str(fps),
        "--width",
        str(width),
        "--height",
        str(height),
        "--duration",
        str(render_duration_seconds),
        "--result-path",
        str(result_path),
    ]
    try:
        worker_timeout_seconds = max(180, min(900, frame_count + 120))
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=worker_timeout_seconds,
            cwd=str(project_root),
        )
    except Exception as exc:
        return _render_result(
            render_success=False,
            render_log=f"Animation worker execution failed: {type(exc).__name__}: {exc}",
            ffmpeg_path=ffmpeg_path,
            playwright_available=playwright_available,
            frame_count=frame_count,
            render_duration_seconds=render_duration_seconds,
            fps=fps,
            started_at=started_at,
            mp4_path=mp4_path,
        )

    worker_result = _read_worker_result(result_path)
    if not worker_result:
        worker_log = "\n".join(
            part
            for part in [completed.stdout.strip(), completed.stderr.strip()]
            if part
        ).strip()
        return _render_result(
            render_success=False,
            render_log=(
                "Animation worker did not write render_result.json."
                f"\nexit_code={completed.returncode}\n{worker_log}"
            )[:6000],
            ffmpeg_path=ffmpeg_path,
            playwright_available=playwright_available,
            frame_count=frame_count,
            render_duration_seconds=render_duration_seconds,
            fps=fps,
            started_at=started_at,
            mp4_path=mp4_path,
        )

    worker_success = bool(worker_result.get("render_success"))
    mp4_exists = mp4_path.is_file()
    mp4_file_size = mp4_path.stat().st_size if mp4_exists else 0
    render_success = bool(worker_success and mp4_exists and mp4_file_size > 0)
    return _render_result(
        render_success=render_success,
        render_log=str(worker_result.get("render_log") or "")[:6000],
        ffmpeg_path=ffmpeg_path,
        playwright_available=playwright_available,
        frame_count=int(worker_result.get("frame_count") or frame_count),
        render_duration_seconds=render_duration_seconds,
        fps=fps,
        started_at=started_at,
        mp4_path=mp4_path,
    )


def _read_worker_result(result_path: Path) -> dict[str, Any]:
    if not result_path.is_file():
        return {}
    try:
        parsed = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _render_result(
    *,
    render_success: bool,
    render_log: str,
    ffmpeg_path: str,
    playwright_available: bool,
    frame_count: int,
    render_duration_seconds: int,
    fps: int,
    started_at: float,
    mp4_path: Path,
) -> dict[str, Any]:
    mp4_exists = mp4_path.is_file()
    mp4_file_size = mp4_path.stat().st_size if mp4_exists else 0
    return {
        "render_success": render_success,
        "render_log": render_log,
        "ffmpeg_path": ffmpeg_path,
        "playwright_available": playwright_available,
        "frame_count": frame_count,
        "render_duration_seconds": render_duration_seconds,
        "fps": fps,
        "mp4_exists": mp4_exists,
        "mp4_file_size": mp4_file_size,
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
    }
