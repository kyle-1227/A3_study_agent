"""Strict content-free trace contract for evidence orchestration."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from src.observability.a3_trace import emit_a3_trace
from src.streaming.evidence_progress import (
    build_evidence_progress,
    evidence_progress_sink_active,
    publish_evidence_progress,
)

EVIDENCE_TRACE_SCHEMA_VERSION = "evidence_orchestration_trace_v1"
EVIDENCE_TRACE_ENV_FLAG = "LOG_EVIDENCE_ORCHESTRATION_TRACE"

EvidenceSource = Literal["local", "web"]
Sha256Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
SafeCode = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.-]{0,79}$")]
SafeResourceType = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")]
SafeErrorType = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.]{0,79}$")]
BoundedCount = Annotated[int, Field(ge=0, le=100_000)]
RoundIndex = Annotated[int, Field(ge=0, le=100)]
LatencyMs = Annotated[int, Field(ge=0, le=86_400_000)]

_TRACE_CONTEXT_FIELDS = frozenset({"request_id", "session_id", "thread_id"})
_SENSITIVE_VALUE_PATTERN = re.compile(
    r"(?:https?://|www\.|bearer\s+|sk-(?:or-v1-)?[A-Za-z0-9_-]{8,})",
    flags=re.IGNORECASE,
)


class _EvidenceTraceBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["evidence_orchestration_trace_v1"]

    @model_validator(mode="after")
    def reject_sensitive_values(self) -> "_EvidenceTraceBase":
        for field_name, value in self.model_dump(mode="python").items():
            if isinstance(value, str) and _SENSITIVE_VALUE_PATTERN.search(value):
                raise ValueError(f"{field_name} contains a forbidden trace value")
        return self


class EvidencePlanAcceptedTrace(_EvidenceTraceBase):
    stage: Literal["evidence_orchestration.plan.accepted"]
    orchestration_fingerprint: Sha256Digest
    profile_fingerprint: Sha256Digest
    requirement_count: BoundedCount
    resource_count: BoundedCount
    subject_count: BoundedCount
    budget_max_rounds: BoundedCount
    budget_max_tasks: BoundedCount


class EvidenceRoundStartedTrace(_EvidenceTraceBase):
    stage: Literal["evidence_orchestration.round.started"]
    round_index: RoundIndex
    task_count: BoundedCount
    local_task_count: BoundedCount
    web_task_count: BoundedCount
    budget_used_tasks: BoundedCount
    budget_remaining_tasks: BoundedCount

    @model_validator(mode="after")
    def validate_task_partition(self) -> "EvidenceRoundStartedTrace":
        if self.local_task_count + self.web_task_count != self.task_count:
            raise ValueError("source task counts must equal task_count")
        return self


class EvidenceSourceCompletedTrace(_EvidenceTraceBase):
    stage: Literal["evidence_orchestration.source.completed"]
    round_index: RoundIndex
    source: EvidenceSource
    status: Literal["completed"]
    task_count: BoundedCount
    query_batch_fingerprint: Sha256Digest
    candidate_count: BoundedCount
    latency_ms: LatencyMs


class EvidenceSourceEmptyTrace(_EvidenceTraceBase):
    stage: Literal["evidence_orchestration.source.empty"]
    round_index: RoundIndex
    source: EvidenceSource
    status: Literal["empty"]
    task_count: BoundedCount
    query_batch_fingerprint: Sha256Digest
    latency_ms: LatencyMs
    reason_code: SafeCode


class EvidenceSourceFailedTrace(_EvidenceTraceBase):
    stage: Literal["evidence_orchestration.source.failed"]
    round_index: RoundIndex
    source: EvidenceSource
    status: Literal["failed"]
    task_count: BoundedCount
    query_batch_fingerprint: Sha256Digest
    latency_ms: LatencyMs
    reason_code: SafeCode
    error_type: SafeErrorType


class EvidenceRoundMergedTrace(_EvidenceTraceBase):
    stage: Literal["evidence_orchestration.round.merged"]
    round_index: RoundIndex
    local_candidate_count: BoundedCount
    web_candidate_count: BoundedCount
    deduplicated_count: BoundedCount
    ledger_count: BoundedCount
    ledger_fingerprint: Sha256Digest


class EvidenceCoverageJudgedTrace(_EvidenceTraceBase):
    stage: Literal["evidence_orchestration.coverage.judged"]
    round_index: RoundIndex
    requirement_count: BoundedCount
    complete_count: BoundedCount
    partial_count: BoundedCount
    missing_count: BoundedCount
    accepted_evidence_count: BoundedCount
    coverage_fingerprint: Sha256Digest

    @model_validator(mode="after")
    def validate_coverage_partition(self) -> "EvidenceCoverageJudgedTrace":
        total = self.complete_count + self.partial_count + self.missing_count
        if total != self.requirement_count:
            raise ValueError("coverage counts must equal requirement_count")
        return self


class EvidenceProgressEvaluatedTrace(_EvidenceTraceBase):
    stage: Literal["evidence_orchestration.progress.evaluated"]
    round_index: RoundIndex
    previous_complete_count: BoundedCount
    current_complete_count: BoundedCount
    previous_partial_count: BoundedCount
    current_partial_count: BoundedCount
    previous_missing_count: BoundedCount
    current_missing_count: BoundedCount
    new_accepted_evidence_count: BoundedCount
    progressed: bool
    consecutive_no_progress_rounds: BoundedCount

    @model_validator(mode="after")
    def validate_progress_signal(self) -> "EvidenceProgressEvaluatedTrace":
        measurable_progress = (
            self.current_missing_count < self.previous_missing_count
            or self.current_complete_count > self.previous_complete_count
            or self.new_accepted_evidence_count > 0
        )
        if self.progressed is not measurable_progress:
            raise ValueError("progressed must reflect measurable coverage progress")
        return self


class EvidenceRouteDecidedTrace(_EvidenceTraceBase):
    stage: Literal["evidence_orchestration.route.decided"]
    round_index: RoundIndex
    status: Literal["repair", "terminal"]
    reason_code: SafeCode
    next_local_task_count: BoundedCount
    next_web_task_count: BoundedCount
    budget_remaining_rounds: BoundedCount
    budget_remaining_tasks: BoundedCount

    @model_validator(mode="after")
    def validate_route_tasks(self) -> "EvidenceRouteDecidedTrace":
        next_task_count = self.next_local_task_count + self.next_web_task_count
        if self.status == "terminal" and next_task_count != 0:
            raise ValueError("terminal route cannot schedule search tasks")
        if self.status == "repair" and next_task_count == 0:
            raise ValueError("repair route requires at least one search task")
        return self


class EvidenceResourceAssignedTrace(_EvidenceTraceBase):
    stage: Literal["evidence_orchestration.resource.assigned"]
    round_index: RoundIndex
    resource_type: SafeResourceType
    status: Literal["ready", "blocked"]
    requirement_count: BoundedCount
    assigned_evidence_count: BoundedCount
    missing_requirement_count: BoundedCount
    assignment_fingerprint: Sha256Digest
    assignment_contract_version: Literal["resource_evidence_assignment_v1"]

    @model_validator(mode="after")
    def validate_readiness(self) -> "EvidenceResourceAssignedTrace":
        if self.status == "ready" and self.missing_requirement_count != 0:
            raise ValueError("ready resource cannot have missing requirements")
        if self.status == "blocked" and self.missing_requirement_count == 0:
            raise ValueError("blocked resource requires a missing requirement")
        if self.missing_requirement_count > self.requirement_count:
            raise ValueError("missing requirements cannot exceed requirement_count")
        return self


class EvidenceTerminalTrace(_EvidenceTraceBase):
    stage: Literal["evidence_orchestration.terminal"]
    orchestration_fingerprint: Sha256Digest
    status: Literal[
        "sufficient",
        "partial_resources_ready",
        "insufficient_max_rounds",
        "insufficient_no_progress",
        "insufficient_empty_sources",
        "blocked_insufficient_evidence",
    ]
    rounds_completed: BoundedCount
    ready_resource_count: BoundedCount
    blocked_resource_count: BoundedCount
    total_search_tasks: BoundedCount
    ledger_count: BoundedCount
    reason_code: SafeCode


class EvidenceFailedTrace(_EvidenceTraceBase):
    stage: Literal["evidence_orchestration.failed"]
    status: Literal["failed"]
    round_index: RoundIndex
    source: Literal["orchestration", "local", "web", "judge", "assignment"]
    error_type: SafeErrorType
    reason_code: SafeCode
    budget_used_tasks: BoundedCount
    budget_remaining_tasks: BoundedCount


EvidenceTraceEvent: TypeAlias = Annotated[
    EvidencePlanAcceptedTrace
    | EvidenceRoundStartedTrace
    | EvidenceSourceCompletedTrace
    | EvidenceSourceEmptyTrace
    | EvidenceSourceFailedTrace
    | EvidenceRoundMergedTrace
    | EvidenceCoverageJudgedTrace
    | EvidenceProgressEvaluatedTrace
    | EvidenceRouteDecidedTrace
    | EvidenceResourceAssignedTrace
    | EvidenceTerminalTrace
    | EvidenceFailedTrace,
    Field(discriminator="stage"),
]

_EVIDENCE_TRACE_ADAPTER: TypeAdapter[EvidenceTraceEvent] = TypeAdapter(
    EvidenceTraceEvent
)
EVIDENCE_TRACE_STAGES = frozenset(
    {
        "evidence_orchestration.plan.accepted",
        "evidence_orchestration.round.started",
        "evidence_orchestration.source.completed",
        "evidence_orchestration.source.empty",
        "evidence_orchestration.source.failed",
        "evidence_orchestration.round.merged",
        "evidence_orchestration.coverage.judged",
        "evidence_orchestration.progress.evaluated",
        "evidence_orchestration.route.decided",
        "evidence_orchestration.resource.assigned",
        "evidence_orchestration.terminal",
        "evidence_orchestration.failed",
    }
)


def is_evidence_trace_stage(stage: object) -> bool:
    """Return whether ``stage`` belongs to the strict evidence trace family."""
    return isinstance(stage, str) and stage in EVIDENCE_TRACE_STAGES


def validate_evidence_trace_event(
    event: EvidenceTraceEvent | Mapping[str, object],
    *,
    emitted: bool = False,
) -> EvidenceTraceEvent:
    """Validate an input event, optionally allowing emitter-owned trace IDs."""
    if isinstance(event, BaseModel):
        raw: dict[str, object] = event.model_dump(mode="python")
    else:
        raw = dict(event)
    if emitted:
        for key in _TRACE_CONTEXT_FIELDS:
            raw.pop(key, None)
    return _EVIDENCE_TRACE_ADAPTER.validate_python(raw)


def emit_evidence_trace(
    logger: logging.Logger,
    event: EvidenceTraceEvent | Mapping[str, object],
    *,
    state: dict | None = None,
) -> None:
    """Validate and emit one evidence trace event before any sink side effect."""
    validated = validate_evidence_trace_event(event)
    payload = validated.model_dump(mode="json")
    if evidence_progress_sink_active():
        if state is None:
            raise ValueError(
                "streaming evidence progress requires graph state identity"
            )
        request_id = state.get("request_id")
        thread_id = state.get("thread_id")
        if not request_id or not thread_id:
            raise ValueError(
                "streaming evidence progress requires request_id and thread_id"
            )
        publish_evidence_progress(
            build_evidence_progress(
                payload,
                request_id=str(request_id),
                thread_id=str(thread_id),
            )
        )
    stage = str(payload.pop("stage"))
    emit_a3_trace(
        logger,
        stage,
        payload,
        state=state,
        env_flag=EVIDENCE_TRACE_ENV_FLAG,
        level="info",
    )


__all__ = [
    "EVIDENCE_TRACE_ENV_FLAG",
    "EVIDENCE_TRACE_SCHEMA_VERSION",
    "EVIDENCE_TRACE_STAGES",
    "EvidenceCoverageJudgedTrace",
    "EvidenceFailedTrace",
    "EvidencePlanAcceptedTrace",
    "EvidenceProgressEvaluatedTrace",
    "EvidenceResourceAssignedTrace",
    "EvidenceRoundMergedTrace",
    "EvidenceRoundStartedTrace",
    "EvidenceRouteDecidedTrace",
    "EvidenceSourceCompletedTrace",
    "EvidenceSourceEmptyTrace",
    "EvidenceSourceFailedTrace",
    "EvidenceTerminalTrace",
    "EvidenceTraceEvent",
    "emit_evidence_trace",
    "is_evidence_trace_stage",
    "validate_evidence_trace_event",
]
