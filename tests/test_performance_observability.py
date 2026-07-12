from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.observability.performance_contracts import (
    PERFORMANCE_EVENT_SCHEMA_VERSION,
    PerformanceSpanEventV1,
)
from src.observability.performance_runtime import (
    PerformanceRecorder,
    build_performance_report,
    performance_report_trace_payload,
    performance_request_recorder,
    performance_span,
    stable_performance_id,
)
from src.tracing import traced_llm_call, traced_node, traced_retrieval, traced_search


def _event(
    *,
    label: str,
    operation_type: str,
    parent: str,
    start: int,
    end: int,
) -> PerformanceSpanEventV1:
    trace_id = stable_performance_id(
        "trace", {"request_id": "request-1", "thread_id": "thread-1"}
    )
    span_id = stable_performance_id("span", {"label": label})
    now = datetime.now(timezone.utc).isoformat()
    return PerformanceSpanEventV1(
        schema_version=PERFORMANCE_EVENT_SCHEMA_VERSION,
        stage="performance.span.completed",
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent,
        operation_id=stable_performance_id("operation", {"label": label}),
        request_id="request-1",
        thread_id="thread-1",
        operation_type=operation_type,
        operation_name=f"test.{label}",
        status="completed",
        started_at=now,
        ended_at=now,
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        duration_ns=end - start,
        error_type="",
        attributes={},
    )


def test_performance_event_rejects_non_whitelisted_content_fields():
    root = _event(
        label="request",
        operation_type="request",
        parent="",
        start=0,
        end=10,
    )
    payload = root.model_dump(mode="python")
    payload["attributes"] = {"query": "private user query"}

    with pytest.raises(ValidationError, match="not allowed"):
        PerformanceSpanEventV1.model_validate(payload)


def test_runtime_recorder_builds_request_node_and_llm_hierarchy():
    with performance_request_recorder(
        request_id="request-1",
        thread_id="thread-1",
        max_spans=32,
    ) as recorder:
        with performance_span("node", "node.qa_agent", parent_policy="request"):
            with performance_span("llm", "llm.qa_agent"):
                pass

    report = build_performance_report(recorder)

    assert report.coverage.root_span_count == 1
    assert report.coverage.invalid_parent_count == 0
    assert report.critical_path_status == "complete"
    assert [item.operation_type for item in report.span_metrics] == [
        "request",
        "node",
        "llm",
    ]


def test_summary_separates_accumulated_union_exclusive_and_critical_path():
    recorder = PerformanceRecorder(
        request_id="request-1",
        thread_id="thread-1",
        max_spans=32,
    )
    root = _event(
        label="request",
        operation_type="request",
        parent="",
        start=0,
        end=100_000_000,
    )
    node_one = _event(
        label="node_one",
        operation_type="node",
        parent=root.span_id,
        start=10_000_000,
        end=60_000_000,
    )
    node_two = _event(
        label="node_two",
        operation_type="node",
        parent=root.span_id,
        start=20_000_000,
        end=80_000_000,
    )
    llm = _event(
        label="llm",
        operation_type="llm",
        parent=node_one.span_id,
        start=20_000_000,
        end=50_000_000,
    )
    database = _event(
        label="database",
        operation_type="database",
        parent=root.span_id,
        start=80_000_000,
        end=95_000_000,
    )
    checkpoint = _event(
        label="checkpoint",
        operation_type="checkpoint",
        parent=database.span_id,
        start=82_000_000,
        end=90_000_000,
    )
    recorder.trace_id = root.trace_id
    recorder.root_span_id = root.span_id
    recorder.events = [llm, node_two, checkpoint, database, node_one, root]

    report = build_performance_report(recorder)
    node_summary = next(
        item for item in report.operation_summaries if item.operation_type == "node"
    )

    assert report.request_inclusive_ms == 100.0
    assert report.request_exclusive_ms == 15.0
    assert node_summary.accumulated_ms == 110.0
    assert node_summary.wall_clock_union_ms == 70.0
    assert report.database_total_ms == 15.0
    assert report.checkpoint_ms == 8.0
    assert report.database_non_checkpoint_ms == 7.0
    assert report.critical_path_status == "complete"
    assert report.critical_path_ms == 90.0


def test_invalid_hierarchy_marks_critical_path_incomplete():
    recorder = PerformanceRecorder(
        request_id="request-1",
        thread_id="thread-1",
        max_spans=8,
    )
    root = _event(
        label="request",
        operation_type="request",
        parent="",
        start=0,
        end=100,
    )
    invalid_checkpoint = _event(
        label="checkpoint",
        operation_type="checkpoint",
        parent=root.span_id,
        start=10,
        end=20,
    )
    recorder.trace_id = root.trace_id
    recorder.root_span_id = root.span_id
    recorder.events = [invalid_checkpoint, root]

    report = build_performance_report(recorder)

    assert report.coverage.invalid_parent_count == 1
    assert report.critical_path_status == "incomplete"
    assert report.critical_path_ms is None
    assert report.critical_path_span_ids == []


def test_trace_projection_contains_only_counts_and_aggregate_metrics():
    with performance_request_recorder(
        request_id="request-1",
        thread_id="thread-1",
        max_spans=8,
    ) as recorder:
        with performance_span("node", "node.qa_agent", parent_policy="request"):
            pass

    payload = performance_report_trace_payload(build_performance_report(recorder))
    serialized = str(payload)

    assert "span_metrics" not in payload
    assert "operation_summaries" not in payload
    assert "private" not in serialized
    assert payload["span_count"] == 2


@pytest.mark.anyio
async def test_existing_tracing_boundaries_emit_content_free_child_spans():
    @traced_node
    async def measured_node(state: dict) -> dict:
        with traced_llm_call(model_name="configured-model", node_name="measured_node"):
            pass
        with traced_retrieval("private retrieval query", subject="private subject"):
            pass
        with traced_search("private web query"):
            pass
        return {"ok": True}

    with performance_request_recorder(
        request_id="request-1",
        thread_id="thread-1",
        max_spans=32,
    ) as recorder:
        assert await measured_node({"request_id": "request-1"}) == {"ok": True}

    by_type = {event.operation_type: event for event in recorder.events}
    node = by_type["node"]
    assert by_type["llm"].parent_span_id == node.span_id
    assert by_type["retrieval"].parent_span_id == node.span_id
    assert by_type["search"].parent_span_id == node.span_id
    assert "private" not in str([event.model_dump() for event in recorder.events])
