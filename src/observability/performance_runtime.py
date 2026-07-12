"""Monotonic request span collection and hierarchy-aware performance summaries."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter_ns
from typing import Any, Literal

from src.context_engineering.workspace import sanitize_workspace_text
from src.observability.performance_contracts import (
    PERFORMANCE_EVENT_SCHEMA_VERSION,
    PERFORMANCE_REPORT_SCHEMA_VERSION,
    FrontendSampleStatus,
    PerformanceCoverageV1,
    PerformanceOperationSummaryV1,
    PerformanceOperationType,
    PerformanceRequestReportV1,
    PerformanceSpanEventV1,
    PerformanceSpanMetricV1,
    PerformanceSpanStatus,
)

ParentPolicy = Literal["current", "request"]

_CURRENT_RECORDER: ContextVar[PerformanceRecorder | None] = ContextVar(
    "a3_performance_recorder",
    default=None,
)
_CURRENT_SPAN_ID: ContextVar[str] = ContextVar("a3_performance_span_id", default="")

_OPERATION_ORDER: tuple[PerformanceOperationType, ...] = (
    "request",
    "node",
    "llm",
    "search",
    "retrieval",
    "render",
    "database",
    "checkpoint",
)
_ALLOWED_PARENT_TYPES: dict[
    PerformanceOperationType, frozenset[PerformanceOperationType]
] = {
    "request": frozenset(),
    "node": frozenset({"request"}),
    "llm": frozenset({"node"}),
    "search": frozenset({"node"}),
    "retrieval": frozenset({"node"}),
    "render": frozenset({"node"}),
    "database": frozenset({"request", "node"}),
    "checkpoint": frozenset({"database"}),
}
_SAFE_ATTRIBUTE_KEYS = frozenset(
    {
        "attempt_count",
        "backend_type",
        "item_count",
        "node_name",
        "renderable_count",
        "resource_type",
        "result_count",
        "retry_count",
        "status_code",
    }
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_performance_id(prefix: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            dict(payload),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return f"{prefix}:v1:{digest}"


@dataclass(slots=True)
class _OpenSpan:
    trace_id: str
    span_id: str
    parent_span_id: str
    operation_id: str
    operation_type: PerformanceOperationType
    operation_name: str
    started_at: str
    started_monotonic_ns: int
    attributes: dict[str, bool | int | float | str] = field(default_factory=dict)


class PerformanceRecorder:
    """Bounded recorder for one request trace."""

    def __init__(
        self,
        *,
        request_id: str,
        thread_id: str,
        max_spans: int,
    ) -> None:
        self.request_id = _required_text(request_id, "request_id", 120)
        self.thread_id = _required_text(thread_id, "thread_id", 120)
        self.max_spans = max_spans
        self.trace_id = stable_performance_id(
            "trace",
            {"request_id": self.request_id, "thread_id": self.thread_id},
        )
        self.root_span_id = ""
        self.events: list[PerformanceSpanEventV1] = []
        self.dropped_span_count = 0
        self._sequence = 0
        self._open_spans: dict[str, _OpenSpan] = {}

    def open_span(
        self,
        *,
        operation_type: PerformanceOperationType,
        operation_name: str,
        parent_policy: ParentPolicy = "current",
        attributes: Mapping[str, Any] | None = None,
    ) -> _OpenSpan | None:
        if len(self.events) + len(self._open_spans) >= self.max_spans:
            self.dropped_span_count += 1
            return None
        self._sequence += 1
        safe_name = _operation_name(operation_name)
        if operation_type == "request":
            parent_span_id = ""
        elif parent_policy == "request":
            parent_span_id = self.root_span_id
        else:
            parent_span_id = _CURRENT_SPAN_ID.get() or self.root_span_id
        identity = {
            "trace_id": self.trace_id,
            "sequence": self._sequence,
            "operation_type": operation_type,
            "operation_name": safe_name,
            "parent_span_id": parent_span_id,
        }
        span = _OpenSpan(
            trace_id=self.trace_id,
            span_id=stable_performance_id("span", identity),
            parent_span_id=parent_span_id,
            operation_id=stable_performance_id(
                "operation",
                {
                    "trace_id": self.trace_id,
                    "sequence": self._sequence,
                    "operation_type": operation_type,
                    "operation_name": safe_name,
                },
            ),
            operation_type=operation_type,
            operation_name=safe_name,
            started_at=utc_now_iso(),
            started_monotonic_ns=perf_counter_ns(),
            attributes=_safe_attributes(attributes),
        )
        if operation_type == "request":
            if self.root_span_id:
                raise RuntimeError("performance recorder already has a request root")
            self.root_span_id = span.span_id
        elif not parent_span_id:
            raise RuntimeError("performance child span requires an active request root")
        self._open_spans[span.span_id] = span
        return span

    def close_span(
        self,
        span: _OpenSpan,
        *,
        status: PerformanceSpanStatus,
        error_type: str = "",
    ) -> PerformanceSpanEventV1:
        current = self._open_spans.pop(span.span_id, None)
        if current is None:
            raise RuntimeError("performance span is not open")
        ended_ns = perf_counter_ns()
        event = PerformanceSpanEventV1(
            schema_version=PERFORMANCE_EVENT_SCHEMA_VERSION,
            stage="performance.span.completed",
            trace_id=span.trace_id,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id,
            operation_id=span.operation_id,
            request_id=self.request_id,
            thread_id=self.thread_id,
            operation_type=span.operation_type,
            operation_name=span.operation_name,
            status=status,
            started_at=span.started_at,
            ended_at=utc_now_iso(),
            started_monotonic_ns=span.started_monotonic_ns,
            ended_monotonic_ns=ended_ns,
            duration_ns=ended_ns - span.started_monotonic_ns,
            error_type=_error_type(error_type),
            attributes=span.attributes,
        )
        self.events.append(event)
        return event

    def operation_type_for_span(self, span_id: str) -> PerformanceOperationType | None:
        open_span = self._open_spans.get(span_id)
        if open_span is not None:
            return open_span.operation_type
        for event in reversed(self.events):
            if event.span_id == span_id:
                return event.operation_type
        return None


@contextmanager
def performance_span(
    operation_type: PerformanceOperationType,
    operation_name: str,
    *,
    parent_policy: ParentPolicy = "current",
    attributes: Mapping[str, Any] | None = None,
    coalesce_same_type: bool = False,
) -> Iterator[_OpenSpan | None]:
    """Record one child span when a request recorder is active."""

    recorder = _CURRENT_RECORDER.get()
    if recorder is None:
        yield None
        return
    current_span_id = _CURRENT_SPAN_ID.get()
    if (
        coalesce_same_type
        and current_span_id
        and recorder.operation_type_for_span(current_span_id) == operation_type
    ):
        yield None
        return
    span = recorder.open_span(
        operation_type=operation_type,
        operation_name=operation_name,
        parent_policy=parent_policy,
        attributes=attributes,
    )
    if span is None:
        yield None
        return
    token = _CURRENT_SPAN_ID.set(span.span_id)
    status: PerformanceSpanStatus = "completed"
    error_type = ""
    try:
        yield span
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
        error_type = type(exc).__name__
        raise
    finally:
        _CURRENT_SPAN_ID.reset(token)
        recorder.close_span(span, status=status, error_type=error_type)


@contextmanager
def performance_request_recorder(
    *,
    request_id: str,
    thread_id: str,
    max_spans: int,
) -> Iterator[PerformanceRecorder]:
    """Install one request recorder and root span in the current async context."""

    recorder = PerformanceRecorder(
        request_id=request_id,
        thread_id=thread_id,
        max_spans=max_spans,
    )
    recorder_token = _CURRENT_RECORDER.set(recorder)
    root = recorder.open_span(
        operation_type="request",
        operation_name="request.stream",
    )
    if root is None:
        _CURRENT_RECORDER.reset(recorder_token)
        raise RuntimeError("performance request root could not be recorded")
    span_token = _CURRENT_SPAN_ID.set(root.span_id)
    status: PerformanceSpanStatus = "completed"
    error_type = ""
    try:
        yield recorder
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
        error_type = type(exc).__name__
        raise
    finally:
        _CURRENT_SPAN_ID.reset(span_token)
        recorder.close_span(root, status=status, error_type=error_type)
        _CURRENT_RECORDER.reset(recorder_token)


def current_performance_recorder() -> PerformanceRecorder | None:
    return _CURRENT_RECORDER.get()


def build_performance_report(
    recorder: PerformanceRecorder,
    *,
    frontend_sample_status: FrontendSampleStatus = "not_requested",
    frontend_milestone_count: int = 0,
) -> PerformanceRequestReportV1:
    """Build a deterministic report from completed spans without mutating them."""

    events = list(recorder.events)
    if not events:
        raise ValueError("performance report requires completed spans")
    by_id = {event.span_id: event for event in events}
    roots = [event for event in events if event.operation_type == "request"]
    children: dict[str, list[PerformanceSpanEventV1]] = defaultdict(list)
    orphan_count = 0
    invalid_parent_count = 0
    out_of_bounds_count = 0
    for event in events:
        if event.operation_type == "request":
            continue
        parent = by_id.get(event.parent_span_id)
        if parent is None:
            orphan_count += 1
            continue
        children[parent.span_id].append(event)
        if parent.operation_type not in _ALLOWED_PARENT_TYPES[event.operation_type]:
            invalid_parent_count += 1
        if (
            event.started_monotonic_ns < parent.started_monotonic_ns
            or event.ended_monotonic_ns > parent.ended_monotonic_ns
        ):
            out_of_bounds_count += 1

    exclusive_by_id: dict[str, int] = {}
    for event in events:
        child_intervals = [
            (
                max(child.started_monotonic_ns, event.started_monotonic_ns),
                min(child.ended_monotonic_ns, event.ended_monotonic_ns),
            )
            for child in children.get(event.span_id, [])
        ]
        child_union = _interval_union_ns(child_intervals)
        exclusive_by_id[event.span_id] = max(event.duration_ns - child_union, 0)

    span_metrics = [
        PerformanceSpanMetricV1(
            span_id=event.span_id,
            parent_span_id=event.parent_span_id,
            operation_type=event.operation_type,
            operation_name=event.operation_name,
            status=event.status,
            inclusive_ms=_ms(event.duration_ns),
            exclusive_ms=_ms(exclusive_by_id[event.span_id]),
        )
        for event in sorted(
            events, key=lambda item: (item.started_monotonic_ns, item.span_id)
        )
    ]
    operation_summaries = [
        _operation_summary(operation_type, events, exclusive_by_id)
        for operation_type in _OPERATION_ORDER
        if any(event.operation_type == operation_type for event in events)
    ]

    database_intervals = _intervals_for_type(events, "database")
    checkpoint_intervals = _intervals_for_type(events, "checkpoint")
    database_total_ns = _interval_union_ns(database_intervals)
    checkpoint_ns = _interval_union_ns(checkpoint_intervals)
    checkpoint_inside_database_ns = _interval_intersection_union_ns(
        database_intervals,
        checkpoint_intervals,
    )

    complete_hierarchy = (
        len(roots) == 1
        and orphan_count == 0
        and invalid_parent_count == 0
        and out_of_bounds_count == 0
        and recorder.dropped_span_count == 0
    )
    critical_path_ns: int | None = None
    critical_path_ids: list[str] = []
    if complete_hierarchy:
        critical_path_ns, critical_path_ids = _critical_path(
            roots[0],
            children=children,
            exclusive_by_id=exclusive_by_id,
        )

    root = (
        roots[0] if roots else min(events, key=lambda item: item.started_monotonic_ns)
    )
    valid_span_count = max(
        len(events) - orphan_count - invalid_parent_count - out_of_bounds_count,
        0,
    )
    coverage_ratio = round(valid_span_count / len(events), 6) if events else 0.0
    return PerformanceRequestReportV1(
        schema_version=PERFORMANCE_REPORT_SCHEMA_VERSION,
        trace_id=recorder.trace_id,
        request_id=recorder.request_id,
        thread_id=recorder.thread_id,
        status=root.status,
        started_at=root.started_at,
        ended_at=root.ended_at,
        request_inclusive_ms=_ms(root.duration_ns),
        request_exclusive_ms=_ms(exclusive_by_id.get(root.span_id, root.duration_ns)),
        span_count=len(events),
        span_metrics=span_metrics,
        operation_summaries=operation_summaries,
        database_total_ms=_ms(database_total_ns),
        checkpoint_ms=_ms(checkpoint_ns),
        database_non_checkpoint_ms=_ms(
            max(database_total_ns - checkpoint_inside_database_ns, 0)
        ),
        critical_path_status="complete" if complete_hierarchy else "incomplete",
        critical_path_ms=_ms(critical_path_ns)
        if critical_path_ns is not None
        else None,
        critical_path_span_ids=critical_path_ids,
        coverage=PerformanceCoverageV1(
            root_span_present=bool(roots),
            root_span_count=len(roots),
            orphan_span_count=orphan_count,
            invalid_parent_count=invalid_parent_count,
            out_of_bounds_child_count=out_of_bounds_count,
            dropped_span_count=recorder.dropped_span_count,
            span_coverage_ratio=coverage_ratio,
        ),
        frontend_sample_status=frontend_sample_status,
        frontend_milestone_count=frontend_milestone_count,
    )


def performance_report_trace_payload(
    report: PerformanceRequestReportV1,
) -> dict[str, Any]:
    """Return a bounded A3_TRACE projection, excluding individual spans."""

    slowest = max(
        report.operation_summaries,
        key=lambda item: item.wall_clock_union_ms,
        default=None,
    )
    return {
        "schema_version": report.schema_version,
        "trace_id": report.trace_id,
        "request_id": report.request_id,
        "thread_id": report.thread_id,
        "status": report.status,
        "request_inclusive_ms": report.request_inclusive_ms,
        "request_exclusive_ms": report.request_exclusive_ms,
        "span_count": report.span_count,
        "span_coverage_ratio": report.coverage.span_coverage_ratio,
        "critical_path_status": report.critical_path_status,
        "critical_path_ms": report.critical_path_ms,
        "database_total_ms": report.database_total_ms,
        "checkpoint_ms": report.checkpoint_ms,
        "database_non_checkpoint_ms": report.database_non_checkpoint_ms,
        "slowest_operation_type": slowest.operation_type if slowest else "",
        "slowest_operation_wall_clock_ms": (
            slowest.wall_clock_union_ms if slowest else 0.0
        ),
        "frontend_sample_status": report.frontend_sample_status,
        "frontend_milestone_count": report.frontend_milestone_count,
        "dropped_span_count": report.coverage.dropped_span_count,
    }


def _operation_summary(
    operation_type: PerformanceOperationType,
    events: Sequence[PerformanceSpanEventV1],
    exclusive_by_id: Mapping[str, int],
) -> PerformanceOperationSummaryV1:
    selected = [event for event in events if event.operation_type == operation_type]
    accumulated_ns = sum(event.duration_ns for event in selected)
    exclusive_ns = sum(exclusive_by_id[event.span_id] for event in selected)
    union_ns = _interval_union_ns(
        [(event.started_monotonic_ns, event.ended_monotonic_ns) for event in selected]
    )
    return PerformanceOperationSummaryV1(
        operation_type=operation_type,
        span_count=len(selected),
        accumulated_ms=_ms(accumulated_ns),
        exclusive_accumulated_ms=_ms(exclusive_ns),
        wall_clock_union_ms=_ms(union_ns),
        max_span_inclusive_ms=_ms(
            max((event.duration_ns for event in selected), default=0)
        ),
    )


def _critical_path(
    event: PerformanceSpanEventV1,
    *,
    children: Mapping[str, list[PerformanceSpanEventV1]],
    exclusive_by_id: Mapping[str, int],
) -> tuple[int, list[str]]:
    child_paths = [
        (
            child.started_monotonic_ns,
            child.ended_monotonic_ns,
            *_critical_path(child, children=children, exclusive_by_id=exclusive_by_id),
        )
        for child in children.get(event.span_id, [])
    ]
    selected_weight, selected_ids = _weighted_non_overlapping_paths(child_paths)
    return exclusive_by_id[event.span_id] + selected_weight, [
        event.span_id,
        *selected_ids,
    ]


def _weighted_non_overlapping_paths(
    paths: Sequence[tuple[int, int, int, list[str]]],
) -> tuple[int, list[str]]:
    if not paths:
        return 0, []
    ordered = sorted(paths, key=lambda item: (item[1], item[0], item[3][0]))
    previous: list[int] = []
    for index, item in enumerate(ordered):
        prior = -1
        for candidate in range(index - 1, -1, -1):
            if ordered[candidate][1] <= item[0]:
                prior = candidate
                break
        previous.append(prior)
    weights = [0] * (len(ordered) + 1)
    selections: list[list[str]] = [[] for _ in range(len(ordered) + 1)]
    for index, item in enumerate(ordered, start=1):
        include_index = previous[index - 1] + 1
        include_weight = item[2] + weights[include_index]
        exclude_weight = weights[index - 1]
        if include_weight > exclude_weight:
            weights[index] = include_weight
            selections[index] = [*selections[include_index], *item[3]]
        else:
            weights[index] = exclude_weight
            selections[index] = list(selections[index - 1])
    return weights[-1], selections[-1]


def _interval_union_ns(intervals: Sequence[tuple[int, int]]) -> int:
    normalized = sorted((start, end) for start, end in intervals if end > start)
    if not normalized:
        return 0
    total = 0
    current_start, current_end = normalized[0]
    for start, end in normalized[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    return total + current_end - current_start


def _interval_intersection_union_ns(
    left: Sequence[tuple[int, int]],
    right: Sequence[tuple[int, int]],
) -> int:
    intersections = [
        (max(left_start, right_start), min(left_end, right_end))
        for left_start, left_end in left
        for right_start, right_end in right
        if min(left_end, right_end) > max(left_start, right_start)
    ]
    return _interval_union_ns(intersections)


def _intervals_for_type(
    events: Sequence[PerformanceSpanEventV1],
    operation_type: PerformanceOperationType,
) -> list[tuple[int, int]]:
    return [
        (event.started_monotonic_ns, event.ended_monotonic_ns)
        for event in events
        if event.operation_type == operation_type
    ]


def _safe_attributes(
    attributes: Mapping[str, Any] | None,
) -> dict[str, bool | int | float | str]:
    if not isinstance(attributes, Mapping):
        return {}
    safe: dict[str, bool | int | float | str] = {}
    for key, value in attributes.items():
        if key not in _SAFE_ATTRIBUTE_KEYS or len(safe) >= 12:
            continue
        if isinstance(value, bool | int | float):
            safe[key] = value
        elif isinstance(value, str):
            text = sanitize_workspace_text(value, max_chars=120, fallback="")
            if text:
                safe[key] = text
    return safe


def _required_text(value: Any, field_name: str, max_chars: int) -> str:
    text = sanitize_workspace_text(value, max_chars=max_chars, fallback="")
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _operation_name(value: Any) -> str:
    text = sanitize_workspace_text(value, max_chars=120, fallback="").lower()
    normalized = "".join(
        character if character.isalnum() or character in "_.:-" else "_"
        for character in text
    ).strip("_.:-")
    if not normalized or not normalized[0].isalpha():
        raise ValueError("performance operation name is invalid")
    return normalized[:120]


def _error_type(value: Any) -> str:
    text = sanitize_workspace_text(value, max_chars=80, fallback="")
    return "".join(
        character for character in text if character.isalnum() or character in "_."
    )[:80]


def _ms(value_ns: int | None) -> float:
    return round(max(int(value_ns or 0), 0) / 1_000_000, 3)


__all__ = [
    "PerformanceRecorder",
    "build_performance_report",
    "current_performance_recorder",
    "performance_report_trace_payload",
    "performance_request_recorder",
    "performance_span",
    "stable_performance_id",
    "utc_now_iso",
]
