"""Activity event contract, adapter, and bounded reducer tests."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.observability.activity import (
    ACTIVITY_TIMELINE_CHAR_LIMIT,
    ACTIVITY_TIMELINE_ITEM_LIMIT,
    activity_from_trace_event,
    build_activity_event,
    build_node_activity_event,
    merge_activity_timeline,
    next_activity_sequence,
    stable_activity_id,
)
from src.observability.contracts import ActivityEvent

NOW = "2026-07-10T00:00:00+00:00"
LATER = "2026-07-10T00:00:01+00:00"


def test_activity_id_is_stable_sha256_identity():
    first = stable_activity_id(
        thread_id="t1",
        request_id="r1",
        kind="node",
        activity_key="node:supervisor",
    )
    second = stable_activity_id(
        thread_id="t1",
        request_id="r1",
        kind="node",
        activity_key="node:supervisor",
    )

    assert first == second
    assert first.startswith("activity:v1:")
    assert len(first.removeprefix("activity:v1:")) == 32


def test_node_start_and_end_merge_by_activity_id():
    started = build_node_activity_event(
        thread_id="t1",
        request_id="r1",
        sequence=1,
        node_id="supervisor",
        status="running",
        now=NOW,
    )
    completed = build_node_activity_event(
        thread_id="t1",
        request_id="r1",
        sequence=2,
        node_id="supervisor",
        status="completed",
        duration_ms=120,
        now=LATER,
    )

    timeline = merge_activity_timeline(
        [started.model_dump(mode="json")],
        [completed.model_dump(mode="json")],
    )

    assert len(timeline) == 1
    assert timeline[0]["status"] == "completed"
    assert timeline[0]["started_at"] == NOW
    assert timeline[0]["updated_at"] == LATER
    assert timeline[0]["duration_ms"] == 120


def test_provider_retry_updates_same_logical_activity():
    base = {
        "node_name": "mindmap_agent",
        "llm_node": "mindmap",
        "trace_call_id": "call-1",
        "retry_count": 1,
    }
    first = activity_from_trace_event(
        {"stage": "provider_transport_retry_attempt", **base},
        thread_id="t1",
        request_id="r1",
        sequence=1,
        now=NOW,
    )
    failed = activity_from_trace_event(
        {"stage": "final_failure_after_retries", **base},
        thread_id="t1",
        request_id="r1",
        sequence=2,
        now=LATER,
    )
    assert first is not None and failed is not None

    timeline = merge_activity_timeline(
        [first.model_dump(mode="json")],
        [failed.model_dump(mode="json")],
    )

    assert first.activity_id == failed.activity_id
    assert len(timeline) == 1
    assert timeline[0]["status"] == "failed"

    replayed = merge_activity_timeline(
        timeline,
        [first.model_dump(mode="json")],
    )
    assert replayed == timeline


def test_trace_adapter_whitelists_details_and_excludes_raw_fields():
    event = activity_from_trace_event(
        {
            "stage": "context_usage_report",
            "node_name": "mindmap_agent",
            "report_id": "context_usage:v1:abc",
            "warning_level": "ok",
            "raw_prompt": "secret prompt",
            "raw_output": "secret output",
        },
        thread_id="t1",
        request_id="r1",
        sequence=1,
        now=NOW,
    )
    assert event is not None
    serialized = event.model_dump_json()

    assert "context_usage:v1:abc" in serialized
    assert "secret prompt" not in serialized
    assert "secret output" not in serialized


def test_timeline_is_bounded_by_count_and_characters():
    events = [
        build_activity_event(
            thread_id="t1",
            request_id=f"r{index}",
            sequence=index + 1,
            kind="stream",
            status="completed",
            activity_key=f"stream:{index}",
            title="Completed request",
            summary="x" * 300,
            now=NOW,
        ).model_dump(mode="json")
        for index in range(ACTIVITY_TIMELINE_ITEM_LIMIT + 50)
    ]

    timeline = merge_activity_timeline([], events)
    serialized = json.dumps(timeline, ensure_ascii=False)

    assert len(timeline) <= ACTIVITY_TIMELINE_ITEM_LIMIT
    assert len(serialized) <= ACTIVITY_TIMELINE_CHAR_LIMIT
    assert timeline[-1]["sequence"] == ACTIVITY_TIMELINE_ITEM_LIMIT + 50
    assert next_activity_sequence(timeline) == ACTIVITY_TIMELINE_ITEM_LIMIT + 51


def test_corrupt_timeline_entries_degrade_gracefully():
    valid = build_node_activity_event(
        thread_id="t1",
        request_id="r1",
        sequence=1,
        node_id="supervisor",
        status="running",
        now=NOW,
    ).model_dump(mode="json")

    assert merge_activity_timeline([{"activity_id": "broken"}], [valid]) == [valid]


def test_activity_contract_forbids_extra_fields_and_naive_timestamps():
    valid = build_node_activity_event(
        thread_id="t1",
        request_id="r1",
        sequence=1,
        node_id="supervisor",
        status="running",
        now=NOW,
    ).model_dump(mode="json")
    valid["unexpected"] = True
    with pytest.raises(ValidationError):
        ActivityEvent.model_validate(valid)

    valid.pop("unexpected")
    valid["started_at"] = "2026-07-10T00:00:00"
    with pytest.raises(ValidationError, match="timezone-aware"):
        ActivityEvent.model_validate(valid)
