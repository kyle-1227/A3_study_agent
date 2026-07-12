"""Strict, content-free contracts for request performance observation."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

PERFORMANCE_EVENT_SCHEMA_VERSION = "performance_event_v1"
PERFORMANCE_REPORT_SCHEMA_VERSION = "performance_report_v1"
FRONTEND_PERFORMANCE_SCHEMA_VERSION = "frontend_performance_v1"

PerformanceOperationType = Literal[
    "request",
    "node",
    "llm",
    "search",
    "retrieval",
    "render",
    "database",
    "checkpoint",
]
PerformanceSpanStatus = Literal["completed", "failed", "cancelled"]
CriticalPathStatus = Literal["complete", "incomplete"]
FrontendMilestoneName = Literal[
    "submit_to_stream_context",
    "submit_to_first_event",
    "submit_to_first_token",
    "submit_to_resource_final",
    "submit_to_done",
    "submit_to_interrupt",
    "submit_to_error",
]
FrontendTerminalStatus = Literal["completed", "failed", "interrupted"]
FrontendSampleStatus = Literal["not_requested", "pending", "accepted", "incomplete"]

SafeId = Annotated[
    str,
    Field(pattern=r"^(?:trace|span|operation):v1:[a-f0-9]{64}$", max_length=80),
]
SafeOperationName = Annotated[
    str,
    Field(pattern=r"^[a-z][a-z0-9_.:-]{0,119}$", max_length=120),
]

_SAFE_ERROR_TYPE = re.compile(r"^[A-Za-z][A-Za-z0-9_.]{0,79}$")
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


def _validate_aware_utc_iso(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("timestamp is required")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    offset = parsed.utcoffset()
    if parsed.tzinfo is None or offset is None:
        raise ValueError("timestamp must be timezone-aware")
    if offset.total_seconds() != 0:
        raise ValueError("timestamp must use UTC")
    return parsed.isoformat()


class PerformanceSpanEventV1(BaseModel):
    """One completed monotonic span with no business content."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["performance_event_v1"]
    stage: Literal["performance.span.completed"]
    trace_id: SafeId
    span_id: SafeId
    parent_span_id: SafeId | Literal[""]
    operation_id: SafeId
    request_id: str = Field(min_length=1, max_length=120)
    thread_id: str = Field(min_length=1, max_length=120)
    operation_type: PerformanceOperationType
    operation_name: SafeOperationName
    status: PerformanceSpanStatus
    started_at: str
    ended_at: str
    started_monotonic_ns: int = Field(ge=0)
    ended_monotonic_ns: int = Field(ge=0)
    duration_ns: int = Field(ge=0)
    error_type: str = Field(default="", max_length=80)
    attributes: dict[str, bool | int | float | str] = Field(default_factory=dict)

    _started_at_utc = field_validator("started_at")(_validate_aware_utc_iso)
    _ended_at_utc = field_validator("ended_at")(_validate_aware_utc_iso)

    @field_validator("error_type")
    @classmethod
    def validate_error_type(cls, value: str) -> str:
        if value and not _SAFE_ERROR_TYPE.fullmatch(value):
            raise ValueError("error_type is not a safe class name")
        return value

    @field_validator("attributes")
    @classmethod
    def validate_attributes(
        cls,
        value: dict[str, bool | int | float | str],
    ) -> dict[str, bool | int | float | str]:
        if len(value) > 12:
            raise ValueError("performance attributes exceed item bound")
        for key, item in value.items():
            if key not in _SAFE_ATTRIBUTE_KEYS:
                raise ValueError(f"performance attribute is not allowed: {key}")
            if isinstance(item, str) and len(item) > 120:
                raise ValueError("performance attribute value exceeds character bound")
        return value

    @model_validator(mode="after")
    def validate_interval(self) -> "PerformanceSpanEventV1":
        if self.ended_monotonic_ns < self.started_monotonic_ns:
            raise ValueError("span monotonic interval is reversed")
        if self.duration_ns != self.ended_monotonic_ns - self.started_monotonic_ns:
            raise ValueError("span duration does not reconcile")
        if self.operation_type == "request" and self.parent_span_id:
            raise ValueError("request span cannot have a parent")
        if self.operation_type != "request" and not self.parent_span_id:
            raise ValueError("non-request span requires a parent")
        return self


class PerformanceFrontendBatchEventV1(BaseModel):
    """Safe acknowledgement that one browser milestone batch was accepted."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["performance_event_v1"]
    stage: Literal["performance.frontend.batch.accepted"]
    trace_id: SafeId
    span_id: SafeId
    parent_span_id: SafeId | Literal[""]
    operation_id: SafeId
    request_id: str = Field(min_length=1, max_length=120)
    thread_id: str = Field(min_length=1, max_length=120)
    operation_type: Literal["request"]
    operation_name: Literal["frontend.milestones"]
    status: Literal["completed"]
    occurred_at: str
    monotonic_clock_source: Literal["browser.performance_now"]
    milestone_count: int = Field(ge=1, le=16)

    _occurred_at_utc = field_validator("occurred_at")(_validate_aware_utc_iso)


PerformanceEventV1: TypeAlias = Annotated[
    PerformanceSpanEventV1 | PerformanceFrontendBatchEventV1,
    Field(discriminator="stage"),
]
PERFORMANCE_EVENT_ADAPTER = TypeAdapter(PerformanceEventV1)


class PerformanceSpanMetricV1(BaseModel):
    """Per-span inclusive and exclusive measurements."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    span_id: SafeId
    parent_span_id: SafeId | Literal[""]
    operation_type: PerformanceOperationType
    operation_name: SafeOperationName
    status: PerformanceSpanStatus
    inclusive_ms: float = Field(ge=0)
    exclusive_ms: float = Field(ge=0)


class PerformanceOperationSummaryV1(BaseModel):
    """Aggregate measurements without double-counting wall-clock overlap."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    operation_type: PerformanceOperationType
    span_count: int = Field(ge=0)
    accumulated_ms: float = Field(ge=0)
    exclusive_accumulated_ms: float = Field(ge=0)
    wall_clock_union_ms: float = Field(ge=0)
    max_span_inclusive_ms: float = Field(ge=0)


class PerformanceCoverageV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    root_span_present: bool
    root_span_count: int = Field(ge=0)
    orphan_span_count: int = Field(ge=0)
    invalid_parent_count: int = Field(ge=0)
    out_of_bounds_child_count: int = Field(ge=0)
    dropped_span_count: int = Field(ge=0)
    span_coverage_ratio: float = Field(ge=0, le=1)


class PerformanceRequestReportV1(BaseModel):
    """One request report with explicit timing semantics."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["performance_report_v1"]
    trace_id: SafeId
    request_id: str = Field(min_length=1, max_length=120)
    thread_id: str = Field(min_length=1, max_length=120)
    status: PerformanceSpanStatus
    started_at: str
    ended_at: str
    request_inclusive_ms: float = Field(ge=0)
    request_exclusive_ms: float = Field(ge=0)
    span_count: int = Field(ge=1)
    span_metrics: list[PerformanceSpanMetricV1]
    operation_summaries: list[PerformanceOperationSummaryV1]
    database_total_ms: float = Field(ge=0)
    checkpoint_ms: float = Field(ge=0)
    database_non_checkpoint_ms: float = Field(ge=0)
    critical_path_status: CriticalPathStatus
    critical_path_ms: float | None = Field(default=None, ge=0)
    critical_path_span_ids: list[SafeId]
    coverage: PerformanceCoverageV1
    frontend_sample_status: FrontendSampleStatus
    frontend_milestone_count: int = Field(ge=0, le=16)

    _started_at_utc = field_validator("started_at")(_validate_aware_utc_iso)
    _ended_at_utc = field_validator("ended_at")(_validate_aware_utc_iso)

    @model_validator(mode="after")
    def validate_critical_path(self) -> "PerformanceRequestReportV1":
        if self.critical_path_status == "complete":
            if self.critical_path_ms is None or not self.critical_path_span_ids:
                raise ValueError(
                    "complete critical path requires duration and span ids"
                )
        elif self.critical_path_ms is not None or self.critical_path_span_ids:
            raise ValueError("incomplete critical path cannot claim a duration or path")
        if self.database_non_checkpoint_ms > self.database_total_ms + 0.001:
            raise ValueError("database non-checkpoint time exceeds database total")
        return self


class FrontendPerformanceMilestoneV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    name: FrontendMilestoneName
    duration_ms: float = Field(ge=0, le=86_400_000)
    count: int | None = Field(default=None, ge=0, le=1_000_000)
    status: FrontendTerminalStatus | None = None


class FrontendPerformanceBatchV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["frontend_performance_v1"]
    request_id: str = Field(min_length=1, max_length=120)
    thread_id: str = Field(min_length=1, max_length=120)
    trace_id: SafeId
    milestones: list[FrontendPerformanceMilestoneV1] = Field(
        min_length=1, max_length=16
    )

    @model_validator(mode="after")
    def validate_unique_milestones(self) -> "FrontendPerformanceBatchV1":
        names = [item.name for item in self.milestones]
        if len(names) != len(set(names)):
            raise ValueError("frontend milestones must be unique")
        terminal = [
            item
            for item in self.milestones
            if item.name in {"submit_to_done", "submit_to_interrupt", "submit_to_error"}
        ]
        if len(terminal) != 1:
            raise ValueError("frontend batch requires exactly one terminal milestone")
        return self


def validate_performance_event(event: PerformanceEventV1 | dict) -> PerformanceEventV1:
    if isinstance(event, BaseModel):
        return PERFORMANCE_EVENT_ADAPTER.validate_python(
            event.model_dump(mode="python")
        )
    return PERFORMANCE_EVENT_ADAPTER.validate_python(event)


__all__ = [
    "FRONTEND_PERFORMANCE_SCHEMA_VERSION",
    "PERFORMANCE_EVENT_SCHEMA_VERSION",
    "PERFORMANCE_REPORT_SCHEMA_VERSION",
    "CriticalPathStatus",
    "FrontendPerformanceBatchV1",
    "FrontendPerformanceMilestoneV1",
    "FrontendSampleStatus",
    "PerformanceCoverageV1",
    "PerformanceEventV1",
    "PerformanceFrontendBatchEventV1",
    "PerformanceOperationSummaryV1",
    "PerformanceOperationType",
    "PerformanceRequestReportV1",
    "PerformanceSpanEventV1",
    "PerformanceSpanMetricV1",
    "PerformanceSpanStatus",
    "validate_performance_event",
]
