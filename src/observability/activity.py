"""Safe activity construction and bounded idempotent timeline reduction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from src.context_engineering.workspace import sanitize_workspace_text, utc_now_iso
from src.observability.contracts import (
    ACTIVITY_EVENT_SCHEMA_VERSION,
    ActivityEvent,
    ActivityKind,
    ActivityStatus,
)
from src.observability.node_registry import get_node_runtime_metadata

ACTIVITY_TIMELINE_ITEM_LIMIT = 200
ACTIVITY_TIMELINE_CHAR_LIMIT = 96_000

_SAFE_DETAIL_FIELDS = {
    "error_type",
    "finish_reason",
    "interrupt_type",
    "item_count",
    "manifest_id",
    "max_retries",
    "output_mode",
    "report_id",
    "resource_id",
    "resource_type",
    "retry_count",
    "status_code",
    "trace_call_id",
    "trace_seq",
    "warning_level",
}

_RESOURCE_TRACE_STATUS: dict[str, ActivityStatus] = {
    "start": "running",
    "success": "completed",
    "failed": "failed",
    "interrupted": "interrupted",
}


def stable_activity_id(
    *,
    thread_id: str,
    request_id: str,
    kind: str,
    activity_key: str,
) -> str:
    identity = {
        "thread_id": _required_text(thread_id, "thread_id", 120),
        "request_id": _required_text(request_id, "request_id", 120),
        "kind": _required_text(kind, "kind", 80),
        "activity_key": _required_text(activity_key, "activity_key", 240),
    }
    digest = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:32]
    return f"activity:v1:{digest}"


def build_activity_event(
    *,
    thread_id: str,
    request_id: str,
    sequence: int,
    kind: ActivityKind,
    status: ActivityStatus,
    activity_key: str,
    title: str,
    summary: str = "",
    node: str = "",
    parent: str = "",
    tool: str = "",
    model: str = "",
    duration_ms: int | None = None,
    safe_details: Mapping[str, Any] | None = None,
    now: str | None = None,
) -> ActivityEvent:
    timestamp = now or utc_now_iso()
    terminal = status in {"completed", "interrupted", "failed", "skipped"}
    return ActivityEvent(
        schema_version=ACTIVITY_EVENT_SCHEMA_VERSION,
        activity_id=stable_activity_id(
            thread_id=thread_id,
            request_id=request_id,
            kind=kind,
            activity_key=activity_key,
        ),
        sequence=sequence,
        thread_id=thread_id,
        request_id=request_id,
        kind=kind,
        status=status,
        node=_safe_text(node, 160),
        parent=_safe_text(parent, 160),
        title=_required_text(title, "title", 200),
        summary=_safe_text(summary, 320),
        tool=_safe_text(tool, 120),
        model=_safe_text(model, 160),
        started_at=timestamp,
        updated_at=timestamp,
        completed_at=timestamp if terminal else "",
        duration_ms=duration_ms,
        safe_details=_safe_details(safe_details),
    )


def build_node_activity_event(
    *,
    thread_id: str,
    request_id: str,
    sequence: int,
    node_id: str,
    status: ActivityStatus,
    duration_ms: int | None = None,
    error_type: str = "",
    now: str | None = None,
) -> ActivityEvent:
    metadata = get_node_runtime_metadata(node_id)
    if metadata is None:
        raise ValueError(f"node activity metadata missing for {node_id}")
    kind = _activity_kind_for_role(metadata.role)
    title = (
        metadata.activity_running
        if status in {"queued", "running", "retrying", "waiting"}
        else metadata.activity_completed
    )
    return build_activity_event(
        thread_id=thread_id,
        request_id=request_id,
        sequence=sequence,
        kind=kind,
        status=status,
        activity_key=f"node:{node_id}",
        title=title,
        summary=metadata.description,
        node=node_id,
        parent=metadata.parent,
        duration_ms=duration_ms,
        safe_details={"error_type": error_type} if error_type else {},
        now=now,
    )


def activity_from_trace_event(
    event: Mapping[str, Any],
    *,
    thread_id: str,
    request_id: str,
    sequence: int,
    now: str | None = None,
) -> ActivityEvent | None:
    """Map selected A3_TRACE families into the normalized activity contract."""
    stage = _safe_text(event.get("stage"), 160)
    if not stage:
        return None
    node = _safe_text(event.get("node_name"), 160)
    trace_call_id = _safe_text(event.get("trace_call_id"), 120)
    common_details = {
        key: event.get(key) for key in _SAFE_DETAIL_FIELDS if key in event
    }

    if stage in {"resource_subnode.start", "resource_subnode.end"}:
        subnode = _required_text(event.get("subnode"), "subnode", 160)
        raw_status = _safe_text(event.get("status"), 40)
        default_status: ActivityStatus = (
            "running" if stage.endswith(".start") else "completed"
        )
        resource_status = _RESOURCE_TRACE_STATUS.get(raw_status, default_status)
        metadata = get_node_runtime_metadata(subnode)
        title = (
            metadata.activity_running
            if metadata is not None and resource_status == "running"
            else metadata.activity_completed
            if metadata is not None
            else subnode.replace("_", " ")
        )
        kind = (
            _activity_kind_for_role(metadata.role) if metadata is not None else "node"
        )
        elapsed = event.get("elapsed_ms")
        duration_ms = (
            elapsed
            if isinstance(elapsed, int) and not isinstance(elapsed, bool)
            else None
        )
        return build_activity_event(
            thread_id=thread_id,
            request_id=request_id,
            sequence=sequence,
            kind=kind,
            status=resource_status,
            activity_key=(
                f"resource_subnode:{_safe_text(event.get('resource_type'), 80)}:{subnode}"
            ),
            title=title,
            summary=metadata.description
            if metadata is not None
            else "Resource workflow step",
            node=subnode,
            parent="resource_worker",
            duration_ms=duration_ms,
            safe_details=common_details,
            now=now,
        )

    if stage in {
        "provider_transport_retry_attempt",
        "provider_transport_error",
        "final_failure_after_retries",
    }:
        retry_status: ActivityStatus = (
            "failed" if stage == "final_failure_after_retries" else "retrying"
        )
        key = trace_call_id or f"{node}:{_safe_text(event.get('llm_node'), 120)}"
        return build_activity_event(
            thread_id=thread_id,
            request_id=request_id,
            sequence=sequence,
            kind="retry",
            status=retry_status,
            activity_key=f"provider_retry:{key}",
            title="Provider retry"
            if retry_status == "retrying"
            else "Provider call failed",
            summary="Provider transport retry lifecycle",
            node=node,
            model=_safe_text(event.get("model"), 160),
            safe_details=common_details,
            now=now,
        )

    if stage in {
        "structured_llm_output",
        "plain_llm_output",
        "llm_provider.invoke_guarded",
    }:
        success = event.get("success")
        llm_status: ActivityStatus = "failed" if success is False else "completed"
        key = (
            trace_call_id or f"{node}:{_safe_text(event.get('llm_node'), 120)}:{stage}"
        )
        return build_activity_event(
            thread_id=thread_id,
            request_id=request_id,
            sequence=sequence,
            kind="llm",
            status=llm_status,
            activity_key=f"llm:{key}",
            title="Model call completed"
            if llm_status == "completed"
            else "Model call failed",
            summary="Guarded provider invocation",
            node=node,
            model=_safe_text(event.get("model"), 160),
            safe_details=common_details,
            now=now,
        )

    if stage == "resource_artifacts.indexed":
        return build_activity_event(
            thread_id=thread_id,
            request_id=request_id,
            sequence=sequence,
            kind="artifact",
            status="completed",
            activity_key=f"artifact_index:{node or 'resource_bundle'}",
            title="Resource artifacts indexed",
            summary="Stable artifact summaries were indexed for thread continuity",
            node=node,
            safe_details=common_details,
            now=now,
        )

    if _is_context_stage(stage):
        context_status: ActivityStatus = (
            "failed"
            if stage.endswith("failed") or stage.endswith("error")
            else "completed"
        )
        key = trace_call_id or f"{node}:{stage}"
        return build_activity_event(
            thread_id=thread_id,
            request_id=request_id,
            sequence=sequence,
            kind="context",
            status=context_status,
            activity_key=f"context:{key}:{stage}",
            title="Context observation updated"
            if context_status == "completed"
            else "Context observation failed",
            summary=stage,
            node=node,
            safe_details=common_details,
            now=now,
        )
    return None


def merge_activity_timeline(
    existing: list[dict[str, Any]] | None,
    update: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Idempotently merge activities, preserving start time and deterministic bounds."""
    merged: dict[str, ActivityEvent] = {}
    for raw in [*(existing or []), *(update or [])]:
        if not isinstance(raw, Mapping):
            continue
        try:
            candidate = ActivityEvent.model_validate(raw)
        except ValidationError:
            continue
        prior = merged.get(candidate.activity_id)
        if prior is not None:
            prior_order = (prior.sequence, prior.updated_at, _status_rank(prior.status))
            candidate_order = (
                candidate.sequence,
                candidate.updated_at,
                _status_rank(candidate.status),
            )
            if candidate_order < prior_order:
                continue
            payload = prior.model_dump(mode="json")
            payload.update(candidate.model_dump(mode="json"))
            payload["started_at"] = min(prior.started_at, candidate.started_at)
            payload["sequence"] = max(prior.sequence, candidate.sequence)
            candidate = ActivityEvent.model_validate(payload)
        merged[candidate.activity_id] = candidate
    ordered = sorted(
        merged.values(),
        key=lambda item: (item.sequence, item.activity_id),
    )[-ACTIVITY_TIMELINE_ITEM_LIMIT:]
    bounded: list[dict[str, Any]] = []
    total_chars = 2
    for item in reversed(ordered):
        payload = item.model_dump(mode="json")
        item_chars = len(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        if bounded and total_chars + item_chars > ACTIVITY_TIMELINE_CHAR_LIMIT:
            continue
        if not bounded and item_chars > ACTIVITY_TIMELINE_CHAR_LIMIT:
            continue
        bounded.append(payload)
        total_chars += item_chars
    return list(reversed(bounded))


def next_activity_sequence(timeline: list[dict[str, Any]] | None) -> int:
    maximum = 0
    for item in timeline or []:
        if not isinstance(item, Mapping):
            continue
        value = item.get("sequence")
        if isinstance(value, int) and not isinstance(value, bool):
            maximum = max(maximum, value)
    return maximum + 1


def activity_timeline_status(timeline: list[dict[str, Any]] | None) -> dict[str, Any]:
    bounded = merge_activity_timeline([], timeline or [])
    return {
        "activity_timeline_count": len(bounded),
        "activity_timeline_last_sequence": next_activity_sequence(bounded) - 1,
    }


def _activity_kind_for_role(role: str) -> ActivityKind:
    if role == "retrieval":
        return "retrieval"
    if role in {"judge", "reviewer", "consensus"}:
        return "review"
    if role == "interrupt":
        return "interrupt"
    return "node"


def _status_rank(status: ActivityStatus) -> int:
    return {
        "queued": 0,
        "running": 1,
        "retrying": 2,
        "waiting": 3,
        "skipped": 4,
        "completed": 5,
        "interrupted": 6,
        "failed": 7,
    }[status]


def _is_context_stage(stage: str) -> bool:
    return stage.startswith(
        (
            "context_",
            "llm_input_manifest.",
            "task_workspace.",
            "workspace_context.",
            "background_context_window.",
        )
    )


def _safe_details(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in sorted(_SAFE_DETAIL_FIELDS):
        if key not in value:
            continue
        item = value[key]
        if isinstance(item, bool | int | float) or item is None:
            result[key] = item
        elif isinstance(item, str):
            result[key] = _safe_text(item, 180)
    return result


def _safe_text(value: object, max_chars: int) -> str:
    return sanitize_workspace_text(value, max_chars=max_chars, fallback="")


def _required_text(value: object, field: str, max_chars: int) -> str:
    text = _safe_text(value, max_chars)
    if not text:
        raise ValueError(f"{field} is required")
    return text
