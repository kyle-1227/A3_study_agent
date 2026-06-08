from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_QUERY = "请生成一份 Python 函数 参数 返回值 作用域的分层练习题"
REPORT_PATH = PROJECT_ROOT / "reports" / "sse_bubble_output_compare.txt"


def _api_base_url() -> str:
    return (
        os.getenv("NEXT_PUBLIC_API_URL")
        or os.getenv("API_BASE_URL")
        or "http://localhost:8000"
    ).rstrip("/")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _preview(text: str, limit: int = 300) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized[:limit] + ("..." if len(normalized) > limit else "")


def _event_summary(event: dict[str, Any]) -> dict[str, Any]:
    event_type = event.get("type")
    summary: dict[str, Any] = {"type": event_type}

    if event_type == "token":
        content = str(event.get("content") or "")
        summary.update({"chars": len(content), "preview": _preview(content, 120)})
    elif event_type == "text":
        content = str(event.get("content") or "")
        summary.update(
            {
                "node": event.get("node", ""),
                "chars": len(content),
                "preview": _preview(content, 120),
            }
        )
    elif event_type == "resource_final":
        answer = str(event.get("answer") or "")
        exercise_items = event.get("exercise_items") or []
        mindmap = event.get("mindmap") or {}
        summary.update(
            {
                "resource_type": event.get("resource_type", ""),
                "answer_chars": len(answer),
                "has_mindmap": bool(mindmap),
                "exercise_items_count": len(exercise_items) if isinstance(exercise_items, list) else 0,
                "preview": _preview(answer, 120),
            }
        )
    elif event_type == "mindmap_result":
        tree = event.get("tree") or {}
        summary.update(
            {
                "title": event.get("title", ""),
                "has_tree": bool(tree),
                "xmind_url": event.get("xmind_url", ""),
            }
        )
    elif event_type == "node_event":
        summary.update(
            {
                "node": event.get("node", ""),
                "status": event.get("status", ""),
                "duration_ms": event.get("duration_ms"),
                "error": event.get("error", ""),
            }
        )
    elif event_type == "error":
        summary["message"] = str(event.get("message") or "")
    elif event_type == "thread_id":
        summary["thread_id"] = event.get("thread_id", "")

    return summary


def _parse_sse_response(response) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    data_lines: list[str] = []

    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                raw_data = "\n".join(data_lines)
                data_lines.clear()
                try:
                    events.append(json.loads(raw_data))
                except json.JSONDecodeError:
                    events.append({"type": "parse_error", "raw": raw_data})
            continue

        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines:
        raw_data = "\n".join(data_lines)
        try:
            events.append(json.loads(raw_data))
        except json.JSONDecodeError:
            events.append({"type": "parse_error", "raw": raw_data})

    return events


def _fetch_stream_events(api_base_url: str, query: str, timeout: int) -> tuple[list[dict[str, Any]], float]:
    payload = json.dumps({"query": query}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{api_base_url}/stream",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "text/event-stream",
        },
    )

    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        events = _parse_sse_response(response)
    return events, time.monotonic() - start


def _simulate_frontend_bubble(events: list[dict[str, Any]]) -> dict[str, Any]:
    bubble = ""
    token_accumulator = ""
    last_text = ""
    last_resource_answer = ""
    last_mindmap: dict[str, Any] | None = None
    first_error = ""
    thread_id = ""

    for event in events:
        event_type = event.get("type")
        if event_type == "thread_id":
            thread_id = str(event.get("thread_id") or "")
        elif event_type == "token":
            content = str(event.get("content") or "")
            token_accumulator += content
            bubble += content
        elif event_type == "text":
            last_text = str(event.get("content") or "")
            bubble = last_text
        elif event_type == "mindmap_result":
            last_mindmap = {
                "title": event.get("title", ""),
                "has_tree": bool(event.get("tree")),
                "xmind_url": event.get("xmind_url", ""),
            }
        elif event_type == "resource_final":
            answer = str(event.get("answer") or "")
            if answer:
                last_resource_answer = answer
                bubble = answer
            if event.get("resource_type") == "mindmap" and event.get("mindmap"):
                mindmap = event.get("mindmap") or {}
                last_mindmap = {
                    "title": mindmap.get("title", ""),
                    "has_tree": bool(mindmap.get("tree")),
                    "xmind_url": mindmap.get("xmind_url", ""),
                }
        elif event_type == "error" and not first_error:
            first_error = str(event.get("message") or "")

    backend_final = last_resource_answer or last_text or token_accumulator
    return {
        "thread_id": thread_id,
        "backend_final": backend_final,
        "frontend_bubble": bubble,
        "token_accumulator": token_accumulator,
        "last_text": last_text,
        "last_resource_answer": last_resource_answer,
        "last_mindmap": last_mindmap or {},
        "first_error": first_error,
    }


def _first_diff_index(left: str, right: str) -> int:
    for idx, (left_char, right_char) in enumerate(zip(left, right)):
        if left_char != right_char:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return -1


def _comparison(backend: str, frontend: str) -> dict[str, Any]:
    diff_index = _first_diff_index(backend, frontend)
    window = 300
    start = max(diff_index - window, 0) if diff_index >= 0 else 0
    end = diff_index + window if diff_index >= 0 else 0

    missing_suffix = ""
    if frontend and backend.startswith(frontend) and len(frontend) < len(backend):
        missing_suffix = backend[len(frontend) : len(frontend) + 800]

    return {
        "match": backend == frontend,
        "backend_len": len(backend),
        "frontend_len": len(frontend),
        "backend_sha256": _sha256(backend),
        "frontend_sha256": _sha256(frontend),
        "first_diff_index": diff_index,
        "backend_diff_preview": backend[start:end] if diff_index >= 0 else "",
        "frontend_diff_preview": frontend[start:end] if diff_index >= 0 else "",
        "missing_suffix_preview": _preview(missing_suffix, 800),
    }


def _build_report(
    *,
    query: str,
    api_base_url: str,
    events: list[dict[str, Any]],
    elapsed_seconds: float,
    simulated: dict[str, Any],
) -> str:
    event_counts = Counter(str(event.get("type") or "unknown") for event in events)
    backend = simulated["backend_final"]
    frontend = simulated["frontend_bubble"]
    comparison = _comparison(backend, frontend)
    resource_events = [event for event in events if event.get("type") == "resource_final"]

    lines: list[str] = []
    lines.append("# SSE Bubble Output Compare Report")
    lines.append("")
    lines.append("## QUERY")
    lines.append(query)
    lines.append("")
    lines.append("## API_BASE_URL")
    lines.append(api_base_url)
    lines.append("")
    lines.append("## THREAD_ID")
    lines.append(simulated.get("thread_id") or "")
    lines.append("")
    lines.append("## EVENT SUMMARY")
    lines.append(f"elapsed_seconds: {elapsed_seconds:.2f}")
    lines.append(f"event_count: {len(events)}")
    lines.append(f"event_type_counts: {dict(event_counts)}")
    if simulated.get("first_error"):
        lines.append(f"first_error: {simulated['first_error']}")
    lines.append("")
    lines.append("## COMPARISON")
    for key, value in comparison.items():
        if isinstance(value, str) and "\n" in value:
            lines.append(f"{key}:")
            lines.append(value)
        else:
            lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("## RAW RESOURCE_FINAL SUMMARY")
    if resource_events:
        for idx, event in enumerate(resource_events, 1):
            lines.append(f"resource_final[{idx}]:")
            lines.append(json.dumps(_event_summary(event), ensure_ascii=False, indent=2, default=str))
    else:
        lines.append("No resource_final event received.")
    lines.append("")
    lines.append("## EVENT TAIL")
    for event in events[-20:]:
        lines.append(json.dumps(_event_summary(event), ensure_ascii=False, default=str))
    lines.append("")
    lines.append("## BACKEND FINAL ANSWER")
    lines.append(backend)
    lines.append("")
    lines.append("## FRONTEND SIMULATED BUBBLE")
    lines.append(frontend)
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare backend final SSE answer with simulated frontend bubble content."
    )
    parser.add_argument("query", nargs="*", help="Question to send to /stream.")
    parser.add_argument("--api-base-url", default=_api_base_url(), help="Backend API base URL.")
    parser.add_argument("--timeout", type=int, default=600, help="HTTP timeout seconds.")
    parser.add_argument("--report-path", default=str(REPORT_PATH), help="Output txt report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    query = " ".join(args.query).strip() or DEFAULT_QUERY
    report_path = Path(args.report_path)

    try:
        events, elapsed_seconds = _fetch_stream_events(args.api_base_url, query, args.timeout)
    except urllib.error.URLError as exc:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        message = (
            "# SSE Bubble Output Compare Report\n\n"
            "## ERROR\n"
            f"Failed to connect to {args.api_base_url}/stream\n"
            f"{type(exc).__name__}: {exc}\n"
        )
        report_path.write_text(message, encoding="utf-8")
        print(message)
        print(f"[ERROR] Report written to {report_path}")
        return 1

    simulated = _simulate_frontend_bubble(events)
    report = _build_report(
        query=query,
        api_base_url=args.api_base_url,
        events=events,
        elapsed_seconds=elapsed_seconds,
        simulated=simulated,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    comparison = _comparison(simulated["backend_final"], simulated["frontend_bubble"])
    print(f"query: {query}")
    print(f"api_base_url: {args.api_base_url}")
    print(f"event_count: {len(events)}")
    print(f"backend_len: {comparison['backend_len']}")
    print(f"frontend_len: {comparison['frontend_len']}")
    print(f"match: {str(comparison['match']).lower()}")
    print(f"[OK] Full comparison report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
