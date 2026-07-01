"""Worker process for rendering teaching animation HTML into MP4."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass


def _write_result(result_path: Path, payload: dict) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sample_frame_paths(frames_dir: Path, max_samples: int = 60) -> list[Path]:
    frames = sorted(frames_dir.glob("*.png"))
    if len(frames) <= max_samples:
        return frames
    step = max(1, len(frames) // max_samples)
    sampled = frames[::step][:max_samples]
    return sampled if sampled[-1] == frames[-1] else sampled[:-1] + [frames[-1]]


def _image_rms_difference(left_path: Path, right_path: Path) -> float:
    with Image.open(left_path) as left, Image.open(right_path) as right:
        left_small = left.convert("RGB").resize((160, 90))
        right_small = right.convert("RGB").resize((160, 90))
        diff = ImageChops.difference(left_small, right_small)
        stat = ImageStat.Stat(diff)
        return sum(value * value for value in stat.rms) ** 0.5


def _frames_are_nearly_static(frames_dir: Path) -> bool:
    sampled = _sample_frame_paths(frames_dir)
    if len(sampled) < 3:
        return True
    static_pairs = 0
    total_pairs = 0
    for previous, current in zip(sampled, sampled[1:]):
        total_pairs += 1
        try:
            rms = _image_rms_difference(previous, current)
        except Exception:
            previous_hash = hashlib.sha256(previous.read_bytes()).hexdigest()
            current_hash = hashlib.sha256(current.read_bytes()).hexdigest()
            rms = 0.0 if previous_hash == current_hash else 255.0
        if rms < 1.0:
            static_pairs += 1
    return bool(total_pairs and static_pairs / total_pairs >= 0.8)


def _render(args: argparse.Namespace) -> dict:
    started_at = time.monotonic()
    html_path = Path(args.html_path).resolve()
    frames_dir = Path(args.frames_dir).resolve()
    mp4_path = Path(args.mp4_path).resolve()
    ffmpeg_path = str(args.ffmpeg_path)
    fps = int(args.fps)
    width = int(args.width)
    height = int(args.height)
    duration = max(1, int(args.duration))
    frame_count = duration * fps
    frames_dir.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": width, "height": height})
            page.set_default_timeout(120_000)
            page.goto(html_path.as_uri(), wait_until="load")
            has_render_at = page.evaluate("typeof window.renderAt === 'function'")
            if not has_render_at:
                raise RuntimeError("window.renderAt is not defined")
            for frame_index in range(frame_count):
                timestamp = frame_index / fps
                page.evaluate("(t) => window.renderAt(t)", timestamp)
                page.screenshot(
                    path=str(frames_dir / f"{frame_index + 1:06d}.png"),
                    timeout=120_000,
                )
        finally:
            browser.close()

    if _frames_are_nearly_static(frames_dir):
        raise RuntimeError("Frames are nearly identical; animation appears static.")

    result = subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "%06d.png"),
            "-vf",
            f"scale={width}:{height},format=yuv420p",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(mp4_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    ffmpeg_log = "\n".join(
        part for part in [result.stdout.strip(), result.stderr.strip()] if part
    ).strip()
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed with exit code {result.returncode}.\n{ffmpeg_log}"
        )

    mp4_exists = mp4_path.exists()
    mp4_file_size = mp4_path.stat().st_size if mp4_exists else 0
    if not mp4_exists or mp4_file_size <= 0:
        raise RuntimeError("ffmpeg completed but MP4 file was not created or is empty.")

    return {
        "render_success": True,
        "mp4_exists": True,
        "mp4_file_size": mp4_file_size,
        "frame_count": frame_count,
        "render_log": f"Rendered {frame_count} frames and created MP4 successfully.\n{ffmpeg_log}"[
            :6000
        ],
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render animation HTML to MP4.")
    parser.add_argument("--html-path", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--mp4-path", required=True)
    parser.add_argument("--ffmpeg-path", required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--duration", type=int, required=True)
    parser.add_argument("--result-path", required=True)
    args = parser.parse_args()

    result_path = Path(args.result_path).resolve()
    try:
        payload = _render(args)
    except Exception:
        mp4_path = Path(args.mp4_path).resolve()
        mp4_exists = mp4_path.exists()
        payload = {
            "render_success": False,
            "mp4_exists": mp4_exists,
            "mp4_file_size": mp4_path.stat().st_size if mp4_exists else 0,
            "frame_count": max(1, int(args.duration) * int(args.fps)),
            "render_log": traceback.format_exc()[:6000],
        }
        _write_result(result_path, payload)
        return 1

    _write_result(result_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
