"""Run the bounded six-case production canary through the real web page."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from playwright.async_api import BrowserContext, Page, async_playwright
from pydantic import ValidationError

from src.evaluation.evidence_rollout.contracts import (
    EvidenceEvaluationDatasetContentV2,
    canonical_sha256,
)
from src.graph.resource_final_v3 import validate_resource_final_v3
from src.schemas import (
    LearningGuidanceCatalogV1,
    OnboardResultV2,
    ThreadStatusResponse,
)
from src.streaming.contracts import AgentStreamEventV2


_AUTHORITATIVE_TERMINAL_TYPES = frozenset(
    {
        "qa_final",
        "resource_final",
        "recommendation_final",
        "assessment_final",
        "interrupt",
        "stopped",
        "stream_error",
    }
)
_OUTCOME_TERMINAL_TYPES = frozenset(_AUTHORITATIVE_TERMINAL_TYPES - {"interrupt"})
_STREAM_CAPTURE_SCRIPT = r"""
(() => {
  const originalFetch = window.fetch.bind(window);
  window.__a3ProductionCanary = { active: 0, streams: [], errors: [] };
  window.fetch = async (...args) => {
    const response = await originalFetch(...args);
    try {
      const requestUrl = typeof args[0] === "string" ? args[0] : args[0].url;
      const parsed = new URL(requestUrl, window.location.href);
      const contentType = response.headers.get("content-type") || "";
      const isLifecycleStream =
        parsed.pathname === "/stream" ||
        parsed.pathname === "/resume" ||
        /^\/threads\/[^/]+\/continue$/.test(parsed.pathname);
      if (isLifecycleStream && contentType.includes("text/event-stream")) {
        const state = window.__a3ProductionCanary;
        const captured = { url: parsed.pathname, events: [] };
        state.streams.push(captured);
        state.active += 1;
        response.clone().text().then((body) => {
          for (const frame of body.split(/\r?\n\r?\n/)) {
            const dataLines = frame
              .split(/\r?\n/)
              .filter((line) => line.startsWith("data:"))
              .map((line) => line.slice(5).trimStart());
            if (!dataLines.length) continue;
            try {
              captured.events.push(JSON.parse(dataLines.join("\n")));
            } catch (error) {
              state.errors.push(error instanceof Error ? error.name : "SSEParseError");
            }
          }
        }).catch((error) => {
          state.errors.push(error instanceof Error ? error.name : "SSEReadError");
        }).finally(() => {
          state.active -= 1;
        });
      }
    } catch (error) {
      window.__a3ProductionCanary.errors.push(
        error instanceof Error ? error.name : "FetchCaptureError"
      );
    }
    return response;
  };
})();
"""


class ProductionCanaryError(RuntimeError):
    """The live browser canary violated a production contract."""


def _required_url(value: str, *, field_name: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must be an absolute HTTP(S) URL")
    return value.rstrip("/")


def _contained_file(root: Path, value: Path, *, field_name: str) -> Path:
    path = value if value.is_absolute() else root / value
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(root):
        raise ValueError(f"{field_name} must be a file inside project_root")
    return resolved


def _contained_output(root: Path, value: Path) -> Path:
    path = value if value.is_absolute() else root / value
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError("output_dir must remain inside project_root")
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise FileExistsError("output_dir must be absent or an empty directory")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _load_dataset(path: Path) -> EvidenceEvaluationDatasetContentV2:
    return EvidenceEvaluationDatasetContentV2.model_validate_json(
        path.read_text(encoding="utf-8"),
        strict=True,
    )


def _validate_stream_events(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        raise ProductionCanaryError("browser captured no SSE events")
    streams: dict[str, list[dict[str, Any]]] = {}
    stream_order: list[str] = []
    for event in events:
        required = {
            "schema_version",
            "type",
            "stream_id",
            "event_id",
            "sequence",
            "request_id",
            "thread_id",
            "created_at",
            "data",
        }
        if set(event) != required or event["schema_version"] != "agent_stream_v2":
            raise ProductionCanaryError("SSE envelope violates agent_stream_v2")
        try:
            validated = AgentStreamEventV2.model_validate(event)
        except (TypeError, ValueError, ValidationError) as exc:
            raise ProductionCanaryError(
                "SSE envelope violates agent_stream_v2"
            ) from exc
        stream_id = validated.stream_id
        if stream_id not in streams:
            stream_order.append(stream_id)
        streams.setdefault(stream_id, []).append(event)

    stream_terminals: list[dict[str, Any]] = []
    for stream_id in stream_order:
        stream_events = streams[stream_id]
        sequences = [item["sequence"] for item in stream_events]
        if sequences != list(range(1, len(stream_events) + 1)):
            raise ProductionCanaryError("SSE sequence is not contiguous")
        if (
            stream_events[0]["type"] != "stream_start"
            or sum(item["type"] == "stream_start" for item in stream_events) != 1
        ):
            raise ProductionCanaryError("SSE stream does not begin with stream_start")
        identities = {(item["request_id"], item["thread_id"]) for item in stream_events}
        if len(identities) != 1:
            raise ProductionCanaryError("SSE identity changes within one stream")
        terminals = [
            item
            for item in stream_events
            if item["type"] in _AUTHORITATIVE_TERMINAL_TYPES
        ]
        if len(terminals) != 1:
            raise ProductionCanaryError(
                "each stream requires exactly one authoritative terminal"
            )
        terminal = terminals[0]
        if (
            len(stream_events) < 3
            or stream_events[-2] is not terminal
            or stream_events[-1]["type"] != "stream_done"
        ):
            raise ProductionCanaryError(
                "authoritative terminal must be followed only by stream_done"
            )
        done = stream_events[-1]
        if (
            done["request_id"] != terminal["request_id"]
            or done["thread_id"] != terminal["thread_id"]
        ):
            raise ProductionCanaryError(
                "stream_done identity differs from authoritative terminal"
            )
        stream_terminals.append(terminal)

    terminal = stream_terminals[-1]
    if terminal["type"] != "resource_final":
        raise ProductionCanaryError(
            f"resource canary terminated with {terminal['type']}"
        )
    thread_ids = {item["thread_id"] for item in stream_terminals}
    if len(thread_ids) != 1:
        raise ProductionCanaryError("resume stream changed the canary thread identity")
    return {
        "stream_count": len(streams),
        "event_count": len(events),
        "interrupt_count": sum(
            item["type"] == "interrupt" for item in stream_terminals
        ),
        "terminal_type": terminal["type"],
        "stream_id": terminal["stream_id"],
        "initial_request_id": streams[stream_order[0]][0]["request_id"],
        "request_id": terminal["request_id"],
        "thread_id": terminal["thread_id"],
        "terminal_data": terminal["data"],
    }


def _safe_terminal_projection(
    terminal_data: dict[str, Any],
    *,
    expected_resource_types: Sequence[str],
    expected_request_id: str,
    expected_thread_id: str,
) -> dict[str, Any]:
    try:
        resource_final = validate_resource_final_v3(terminal_data)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ProductionCanaryError(
            "resource_final payload violates resource_final_v3"
        ) from exc
    if (
        resource_final.request_id != expected_request_id
        or resource_final.thread_id != expected_thread_id
    ):
        raise ProductionCanaryError(
            "resource_final identity differs from its SSE envelope"
        )
    observed = [
        {"resource_type": item.kind, "status": item.status}
        for item in resource_final.resources
    ]
    blocked = [
        {"resource_type": item.resource_type, "status": item.status}
        for item in resource_final.blocked_resources
    ]
    observed_types = {item["resource_type"] for item in (*observed, *blocked)}
    missing = set(expected_resource_types) - observed_types
    if missing:
        raise ProductionCanaryError(
            "resource_final omitted expected resource types: "
            + ", ".join(sorted(missing))
        )
    return {
        "schema_version": resource_final.schema_version,
        "resource_final_id": resource_final.resource_final_id,
        "payload_hash": resource_final.payload_hash,
        "request_id": resource_final.request_id,
        "thread_id": resource_final.thread_id,
        "terminal_status": resource_final.terminal_status,
        "resources": observed,
        "blocked_resources": blocked,
    }


def _parse_sse_text(body: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    normalized = body.replace("\r\n", "\n")
    for frame in normalized.split("\n\n"):
        data_lines = [
            line[5:].lstrip() for line in frame.split("\n") if line.startswith("data:")
        ]
        if data_lines:
            value = json.loads("\n".join(data_lines))
            if not isinstance(value, dict):
                raise ProductionCanaryError("replayed SSE data must be an object")
            events.append(value)
    return events


def _artifact_paths(terminal_data: dict[str, Any]) -> tuple[str, ...]:
    paths: list[str] = []
    resources = terminal_data.get("resources")
    if not isinstance(resources, list):
        return ()
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        refs = resource.get("artifact_refs")
        if not isinstance(refs, dict):
            continue
        for value in refs.values():
            if isinstance(value, str) and value.startswith("/artifacts/"):
                paths.append(value)
    return tuple(dict.fromkeys(paths))


def _refresh_projection_mode(
    terminal: dict[str, Any],
    artifact_paths: Sequence[str],
) -> str:
    if terminal["resources"]:
        if not artifact_paths:
            raise ProductionCanaryError(
                "ready refresh recovery requires a local artifact projection"
            )
        return "artifact_download"
    if terminal["blocked_resources"]:
        return "blocked_status"
    raise ProductionCanaryError(
        "refresh recovery requires a ready or blocked resource projection"
    )


async def _verify_refreshed_ui(
    *,
    page: Page,
    thread_id: str,
    terminal: dict[str, Any],
    artifact_paths: Sequence[str],
    timeout_seconds: float,
) -> dict[str, Any]:
    await page.reload(wait_until="networkidle")
    await page.locator("main textarea").wait_for()
    restored_thread_id = await page.evaluate(
        "() => localStorage.getItem('a3_current_thread_id')"
    )
    if restored_thread_id != thread_id:
        raise ProductionCanaryError(
            "browser refresh lost the completed thread identity"
        )

    mode = _refresh_projection_mode(terminal, artifact_paths)
    timeout_ms = timeout_seconds * 1_000
    if mode == "artifact_download":
        links = page.locator("main a[download]")
        await links.first.wait_for(timeout=timeout_ms)
        hrefs = await links.evaluate_all(
            "elements => elements.map((element) => element.href)"
        )
        rendered_paths = {
            urlparse(href).path for href in hrefs if isinstance(href, str)
        }
        matching_paths = set(artifact_paths) & rendered_paths
        if not matching_paths:
            raise ProductionCanaryError(
                "browser refresh did not restore a referenced artifact card"
            )
        return {
            "mode": mode,
            "matching_artifact_link_count": len(matching_paths),
        }

    stored_projection = await page.evaluate(
        """({ restoredThreadId }) => {
          const raw = localStorage.getItem(`a3_messages:${restoredThreadId}`);
          if (!raw) return false;
          try {
            const messages = JSON.parse(raw);
            return Array.isArray(messages) && messages.some((message) =>
              message &&
              message.role === "assistant" &&
              typeof message.resourceFinalDedupeKey === "string" &&
              message.resourceFinalDedupeKey.length > 0 &&
              message.resourceStatus &&
              typeof message.resourceStatus.state === "string"
            );
          } catch (_) {
            return false;
          }
        }""",
        {"restoredThreadId": thread_id},
    )
    if not stored_projection:
        raise ProductionCanaryError(
            "browser refresh lost the blocked resource status projection"
        )
    status_panels = page.locator("main .chat-scroll-area button.w-full.text-left")
    await status_panels.first.wait_for(timeout=timeout_ms)
    return {
        "mode": mode,
        "stored_resource_projection": True,
        "visible_status_panel_count": await status_panels.count(),
    }


async def _verify_replay(
    *,
    client: httpx.AsyncClient,
    backend_url: str,
    events: Sequence[dict[str, Any]],
    stream_id: str,
) -> dict[str, Any]:
    stream_events = [item for item in events if item["stream_id"] == stream_id]
    if len(stream_events) < 3:
        raise ProductionCanaryError(
            "completed stream is too short for replay verification"
        )
    after = stream_events[len(stream_events) // 2 - 1]
    response = await client.get(
        f"{backend_url}/streams/{stream_id}",
        headers={"Last-Event-ID": after["event_id"]},
    )
    response.raise_for_status()
    replayed = _parse_sse_text(response.text)
    expected = stream_events[after["sequence"] :]
    for event in replayed:
        try:
            AgentStreamEventV2.model_validate(event)
        except (TypeError, ValueError, ValidationError) as exc:
            raise ProductionCanaryError(
                "replayed SSE envelope violates agent_stream_v2"
            ) from exc
    if replayed != expected:
        raise ProductionCanaryError("Last-Event-ID replay differs from journal tail")
    return {
        "after_sequence": after["sequence"],
        "replayed_count": len(replayed),
        "tail_matches": True,
    }


async def _verify_downloads(
    *,
    client: httpx.AsyncClient,
    backend_url: str,
    artifact_paths: Sequence[str],
) -> dict[str, Any]:
    verified = 0
    attachment_count = 0
    downloaded_bytes = 0
    for artifact_path in artifact_paths:
        parsed = urlparse(artifact_path)
        if parsed.scheme or parsed.netloc or not parsed.path.startswith("/artifacts/"):
            raise ProductionCanaryError(
                "artifact reference is not a local download path"
            )
        async with client.stream("GET", f"{backend_url}{artifact_path}") as response:
            response.raise_for_status()
            if (
                "attachment"
                not in response.headers.get("content-disposition", "").lower()
            ):
                raise ProductionCanaryError(
                    "artifact response is missing attachment disposition"
                )
            artifact_bytes = 0
            async for chunk in response.aiter_bytes():
                artifact_bytes += len(chunk)
            if artifact_bytes < 1:
                raise ProductionCanaryError("artifact download returned an empty body")
            downloaded_bytes += artifact_bytes
            attachment_count += 1
            verified += 1
    return {
        "referenced_count": len(artifact_paths),
        "verified_count": verified,
        "attachment_header_count": attachment_count,
        "downloaded_bytes": downloaded_bytes,
    }


async def _provision_user(
    *,
    backend_url: str,
    dataset: EvidenceEvaluationDatasetContentV2,
    timeout_seconds: float,
) -> tuple[str, str]:
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.get(f"{backend_url}/learning-guidance/catalog")
        response.raise_for_status()
        catalog = LearningGuidanceCatalogV1.model_validate(response.json(), strict=True)
        topic_owners = {
            topic.topic_id: subject.subject_id
            for subject in catalog.subjects
            for topic in subject.topics
        }
        target_pairs = tuple(
            dict.fromkeys(
                (target.subject, target.topic_id)
                for case in dataset.cases
                for target in case.targets
            )
        )
        for subject, topic_id in target_pairs:
            if topic_owners.get(topic_id) != subject:
                raise ProductionCanaryError(
                    "canary target topic is absent from the served catalog"
                )
        user_id = f"canary-{uuid4()}"
        request_id = str(uuid4())
        payload = {
            "schema_version": "onboard_v2",
            "profile": {
                "schema_version": "learning_guidance_profile_write_request_v1",
                "request_id": request_id,
                "user_id": user_id,
                "skills": [
                    {
                        "subject": subject,
                        "topic_id": topic_id,
                        "level": 0.5,
                        "confidence": 0.8,
                    }
                    for subject, topic_id in target_pairs
                ],
                "goals": [
                    {
                        "subject": subject,
                        "topic_id": topic_id,
                        "goal": f"Production canary for {topic_id}",
                        "importance": 0.8,
                        "progress": 0.0,
                    }
                    for subject, topic_id in target_pairs
                ],
                "preferences": [
                    {
                        "subject": subject,
                        "topic_id": topic_id,
                        "dimension": "prefer_step_by_step",
                        "strength": 0.7,
                    }
                    for subject, topic_id in target_pairs
                ],
            },
            "nickname": "production-canary",
            "grade": "university",
            "dislikes": ["fabricated evidence"],
        }
        onboard = await client.post(f"{backend_url}/onboard", json=payload)
        onboard.raise_for_status()
        result = OnboardResultV2.model_validate(onboard.json(), strict=True)
        if result.user_id != user_id or result.request_id != request_id:
            raise ProductionCanaryError("onboarding result identity mismatch")
        return user_id, "production-canary"


async def _install_browser_identity(
    context: BrowserContext,
    *,
    user_id: str,
    nickname: str,
) -> None:
    script = f"""
(() => {{
  try {{
    localStorage.setItem("a3_user_id", {json.dumps(user_id)});
    localStorage.setItem("a3_nickname", {json.dumps(nickname)});
    localStorage.setItem("a3_onboarding_completed", "true");
  }} catch (_) {{}}
}})();
"""
    await context.add_init_script(script=script)
    await context.add_init_script(script=_STREAM_CAPTURE_SCRIPT)


async def _resume_visible_interrupt(page: Page) -> bool:
    profile_dialog = page.get_by_role("dialog").filter(has_text="请补充必要学习信息")
    if await profile_dialog.count() and await profile_dialog.first.is_visible():
        textareas = profile_dialog.first.locator("textarea")
        for index in range(await textareas.count()):
            await textareas.nth(index).fill(
                "Use the current course evidence and a concise step-by-step plan."
            )
        await profile_dialog.first.get_by_role("button", name="继续生成").click()
        return True
    memory_dialog = page.get_by_role("dialog").filter(has_text="是否结合历史学习记录")
    if await memory_dialog.count() and await memory_dialog.first.is_visible():
        await memory_dialog.first.get_by_role("button", name="只看当前问题").click()
        return True
    return False


async def _captured_events(page: Page) -> tuple[list[dict[str, Any]], int, list[str]]:
    snapshot = await page.evaluate(
        """() => ({
          active: window.__a3ProductionCanary?.active ?? 0,
          errors: window.__a3ProductionCanary?.errors ?? [],
          streams: window.__a3ProductionCanary?.streams ?? [],
        })"""
    )
    events = [event for stream in snapshot["streams"] for event in stream["events"]]
    return events, int(snapshot["active"]), list(snapshot["errors"])


async def _run_case(
    *,
    page: Page,
    backend_url: str,
    case: Any,
    output_dir: Path,
    timeout_seconds: float,
    user_id: str,
    verify_recovery: bool,
) -> dict[str, Any]:
    await page.evaluate(
        """() => { window.__a3ProductionCanary = { active: 0, streams: [], errors: [] }; }"""
    )
    input_box = page.get_by_placeholder("输入你的问题...")
    await input_box.fill(case.query)
    await page.get_by_title("Send").click()

    deadline = time.monotonic() + timeout_seconds
    observed_stream = False
    while time.monotonic() < deadline:
        events, active, capture_errors = await _captured_events(page)
        if capture_errors:
            raise ProductionCanaryError(
                "browser SSE capture failed: " + ", ".join(capture_errors)
            )
        observed_stream = observed_stream or bool(events) or active > 0
        if observed_stream and active == 0:
            outcome_terminal_count = sum(
                event.get("type") in _OUTCOME_TERMINAL_TYPES for event in events
            )
            if outcome_terminal_count:
                break
            if await _resume_visible_interrupt(page):
                await page.wait_for_timeout(250)
        await page.wait_for_timeout(250)
    else:
        raise TimeoutError(f"canary case timed out: {case.case_id}")

    events, active, capture_errors = await _captured_events(page)
    if active or capture_errors:
        raise ProductionCanaryError("SSE capture did not settle cleanly")
    stream_summary = _validate_stream_events(events)
    terminal_data = stream_summary.pop("terminal_data")
    terminal = _safe_terminal_projection(
        terminal_data,
        expected_resource_types=case.resource_types,
        expected_request_id=stream_summary["request_id"],
        expected_thread_id=stream_summary["thread_id"],
    )
    thread_id = stream_summary["thread_id"]
    artifact_paths = _artifact_paths(terminal_data)

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        status = await client.get(f"{backend_url}/threads/{thread_id}/status")
        status.raise_for_status()
        try:
            status_payload = ThreadStatusResponse.model_validate_json(
                status.content,
                strict=True,
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise ProductionCanaryError(
                "thread status violates ThreadStatusResponse"
            ) from exc
        if (
            status_payload.thread_id != thread_id
            or status_payload.schema_version != "run_control_v1"
            or status_payload.run_status != "completed"
            or status_payload.resume_available
        ):
            raise ProductionCanaryError(
                "thread status is not a completed run_control_v1 checkpoint"
            )
        replay = await _verify_replay(
            client=client,
            backend_url=backend_url,
            events=events,
            stream_id=stream_summary["stream_id"],
        )
        downloads = await _verify_downloads(
            client=client,
            backend_url=backend_url,
            artifact_paths=artifact_paths,
        )
        conflict_status: int | None = None
        if verify_recovery:
            async with client.stream(
                "POST",
                f"{backend_url}/stream",
                json={
                    "query": case.query + " conflicting replay",
                    "request_id": stream_summary["initial_request_id"],
                    "thread_id": thread_id,
                    "user_id": user_id,
                },
            ) as conflict:
                conflict_status = conflict.status_code
            if conflict_status != 409:
                raise ProductionCanaryError(
                    "request-id drift did not return an explicit 409 conflict"
                )

    screenshot_path = output_dir / f"{case.case_id}.png"
    await page.screenshot(path=str(screenshot_path), full_page=True)
    refresh_screenshot: str | None = None
    refresh_projection: dict[str, Any] | None = None
    if verify_recovery:
        refresh_projection = await _verify_refreshed_ui(
            page=page,
            thread_id=thread_id,
            terminal=terminal,
            artifact_paths=artifact_paths,
            timeout_seconds=timeout_seconds,
        )
        refresh_path = output_dir / f"{case.case_id}.refresh.png"
        await page.screenshot(path=str(refresh_path), full_page=True)
        refresh_screenshot = refresh_path.name
    return {
        "case_id": case.case_id,
        "subjects": list(case.subjects),
        "resource_types": list(case.resource_types),
        "initial_evidence_state": case.initial_evidence.state,
        "stream": stream_summary,
        "terminal": terminal,
        "replay": replay,
        "downloads": downloads,
        "conflict_status": conflict_status,
        "status": {
            "schema_version": status_payload.schema_version,
            "run_status": status_payload.run_status,
            "resume_available": status_payload.resume_available,
        },
        "screenshot": screenshot_path.name,
        "refresh_screenshot": refresh_screenshot,
        "refresh_projection": refresh_projection,
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.resolve(strict=True)
    if not project_root.is_dir():
        raise ValueError("project_root must be an existing directory")
    dataset_path = _contained_file(
        project_root,
        args.dataset,
        field_name="dataset",
    )
    output_dir = _contained_output(project_root, args.output_dir)
    frontend_url = _required_url(args.frontend_url, field_name="frontend_url")
    backend_url = _required_url(args.backend_url, field_name="backend_url")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    dataset = _load_dataset(dataset_path)
    if len(dataset.cases) != 6:
        raise ProductionCanaryError(
            "production smoke dataset must contain exactly 6 cases"
        )

    user_id, nickname = await _provision_user(
        backend_url=backend_url,
        dataset=dataset,
        timeout_seconds=args.timeout_seconds,
    )
    results: list[dict[str, Any]] = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=args.headless)
        context = await browser.new_context(viewport={"width": 1440, "height": 1000})
        await _install_browser_identity(context, user_id=user_id, nickname=nickname)
        page = await context.new_page()
        await page.goto(frontend_url, wait_until="networkidle")
        await page.get_by_placeholder("输入你的问题...").wait_for()
        try:
            for case_index, case in enumerate(dataset.cases):
                if case_index:
                    await page.get_by_role("button", name="发起新对话").click()
                    await page.get_by_placeholder("输入你的问题...").wait_for()
                results.append(
                    await _run_case(
                        page=page,
                        backend_url=backend_url,
                        case=case,
                        output_dir=output_dir,
                        timeout_seconds=args.timeout_seconds,
                        user_id=user_id,
                        verify_recovery=case_index == len(dataset.cases) - 1,
                    )
                )
        finally:
            await context.close()
            await browser.close()
    verified_downloads = sum(
        result["downloads"]["verified_count"] for result in results
    )
    if verified_downloads < 1:
        raise ProductionCanaryError(
            "six-case canary did not verify any local artifact download"
        )

    report = {
        "schema_version": "production_browser_canary_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_id": dataset.dataset_id,
        "dataset_content_fingerprint": canonical_sha256(
            dataset.model_dump(mode="json")
        ),
        "smoke_authoring_only": True,
        "case_count": len(results),
        "all_cases_completed": len(results) == len(dataset.cases),
        "verified_download_count": verified_downloads,
        "cases": results,
    }
    (output_dir / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frontend-url", required=True)
    parser.add_argument("--backend-url", required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    browser_mode = parser.add_mutually_exclusive_group(required=True)
    browser_mode.add_argument("--headless", action="store_true")
    browser_mode.add_argument("--headed", action="store_false", dest="headless")
    return parser.parse_args()


def main() -> None:
    report = asyncio.run(_run(_parse_args()))
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "case_count": report["case_count"],
                "all_cases_completed": report["all_cases_completed"],
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
