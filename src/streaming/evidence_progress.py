"""Strict, content-free public progress contract for evidence orchestration."""

from __future__ import annotations

import hashlib
import json
import re
from contextvars import ContextVar, Token
from typing import Annotated, Callable, Literal, Mapping, TypeAlias
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

EVIDENCE_PROGRESS_SCHEMA_VERSION: Literal["evidence_progress_v1"] = (
    "evidence_progress_v1"
)

BoundedCount = Annotated[int, Field(ge=0, le=100_000)]
LatencyMs = Annotated[int, Field(ge=0, le=86_400_000)]
RoundIndex = Annotated[int, Field(ge=0, le=100)]
SafeCode = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.-]{0,79}$")]
SafeErrorType = Annotated[
    str,
    Field(pattern=r"^[A-Za-z][A-Za-z0-9_.]{0,79}$"),
]
SafeResourceType = Annotated[
    str,
    Field(pattern=r"^[a-z][a-z0-9_]{0,39}$"),
]
Sha256Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]

EvidenceProgressPhaseStatus: TypeAlias = Literal["running", "completed", "failed"]
EvidenceSource: TypeAlias = Literal["local", "web"]

_SENSITIVE_VALUE_PATTERN = re.compile(
    r"(?:https?://|www\.|bearer\s+|sk-(?:or-v1-)?[A-Za-z0-9_-]{8,}|"
    r"(?:authorization|api[_-]?key|cookie|x-api-key)\s*[:=])",
    flags=re.IGNORECASE,
)


class _ProgressDetailsBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    @model_validator(mode="after")
    def reject_sensitive_values(self) -> "_ProgressDetailsBase":
        for field_name, value in self.model_dump(mode="python").items():
            if isinstance(value, str) and _SENSITIVE_VALUE_PATTERN.search(value):
                raise ValueError(f"{field_name} contains a forbidden progress value")
        return self


class EvidencePlanAcceptedDetails(_ProgressDetailsBase):
    stage: Literal["evidence_orchestration.plan.accepted"]
    requirement_count: BoundedCount
    resource_count: BoundedCount
    subject_count: BoundedCount
    budget_max_rounds: BoundedCount
    budget_max_tasks: BoundedCount


class EvidenceRoundStartedDetails(_ProgressDetailsBase):
    stage: Literal["evidence_orchestration.round.started"]
    round_index: RoundIndex
    task_count: BoundedCount
    local_task_count: BoundedCount
    web_task_count: BoundedCount
    budget_used_tasks: BoundedCount
    budget_remaining_tasks: BoundedCount

    @model_validator(mode="after")
    def validate_task_partition(self) -> "EvidenceRoundStartedDetails":
        if self.local_task_count + self.web_task_count != self.task_count:
            raise ValueError("source task counts must equal task_count")
        return self


class EvidenceSourceCompletedDetails(_ProgressDetailsBase):
    stage: Literal["evidence_orchestration.source.completed"]
    round_index: RoundIndex
    source: EvidenceSource
    status: Literal["completed"]
    task_count: BoundedCount
    candidate_count: BoundedCount
    latency_ms: LatencyMs


class EvidenceSourceEmptyDetails(_ProgressDetailsBase):
    stage: Literal["evidence_orchestration.source.empty"]
    round_index: RoundIndex
    source: EvidenceSource
    status: Literal["empty"]
    task_count: BoundedCount
    latency_ms: LatencyMs
    reason_code: SafeCode


class EvidenceSourceFailedDetails(_ProgressDetailsBase):
    stage: Literal["evidence_orchestration.source.failed"]
    round_index: RoundIndex
    source: EvidenceSource
    status: Literal["failed"]
    task_count: BoundedCount
    latency_ms: LatencyMs
    reason_code: SafeCode
    error_type: SafeErrorType


class EvidenceRoundMergedDetails(_ProgressDetailsBase):
    stage: Literal["evidence_orchestration.round.merged"]
    round_index: RoundIndex
    local_candidate_count: BoundedCount
    web_candidate_count: BoundedCount
    deduplicated_count: BoundedCount
    ledger_count: BoundedCount


class EvidenceCoverageJudgedDetails(_ProgressDetailsBase):
    stage: Literal["evidence_orchestration.coverage.judged"]
    round_index: RoundIndex
    requirement_count: BoundedCount
    complete_count: BoundedCount
    partial_count: BoundedCount
    missing_count: BoundedCount
    accepted_evidence_count: BoundedCount

    @model_validator(mode="after")
    def validate_coverage_partition(self) -> "EvidenceCoverageJudgedDetails":
        if (
            self.complete_count + self.partial_count + self.missing_count
            != self.requirement_count
        ):
            raise ValueError("coverage counts must equal requirement_count")
        return self


class EvidenceProgressEvaluatedDetails(_ProgressDetailsBase):
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
    def validate_progress_signal(self) -> "EvidenceProgressEvaluatedDetails":
        measurable = (
            self.current_missing_count < self.previous_missing_count
            or self.current_complete_count > self.previous_complete_count
            or self.new_accepted_evidence_count > 0
        )
        if self.progressed is not measurable:
            raise ValueError("progressed must reflect measurable coverage progress")
        return self


class EvidenceRouteDecidedDetails(_ProgressDetailsBase):
    stage: Literal["evidence_orchestration.route.decided"]
    round_index: RoundIndex
    status: Literal["repair", "terminal"]
    reason_code: SafeCode
    next_local_task_count: BoundedCount
    next_web_task_count: BoundedCount
    budget_remaining_rounds: BoundedCount
    budget_remaining_tasks: BoundedCount

    @model_validator(mode="after")
    def validate_route(self) -> "EvidenceRouteDecidedDetails":
        task_count = self.next_local_task_count + self.next_web_task_count
        if self.status == "terminal" and task_count != 0:
            raise ValueError("terminal route cannot schedule search tasks")
        if self.status == "repair" and task_count == 0:
            raise ValueError("repair route requires at least one search task")
        return self


class EvidenceResourceAssignedDetails(_ProgressDetailsBase):
    stage: Literal["evidence_orchestration.resource.assigned"]
    round_index: RoundIndex
    resource_type: SafeResourceType
    status: Literal["ready", "blocked"]
    requirement_count: BoundedCount
    assigned_evidence_count: BoundedCount
    missing_requirement_count: BoundedCount

    @model_validator(mode="after")
    def validate_readiness(self) -> "EvidenceResourceAssignedDetails":
        if self.status == "ready" and self.missing_requirement_count != 0:
            raise ValueError("ready resource cannot have missing requirements")
        if self.status == "blocked" and self.missing_requirement_count == 0:
            raise ValueError("blocked resource requires a missing requirement")
        if self.missing_requirement_count > self.requirement_count:
            raise ValueError("missing requirements cannot exceed requirement_count")
        return self


class EvidenceTerminalDetails(_ProgressDetailsBase):
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


class EvidenceFailedDetails(_ProgressDetailsBase):
    stage: Literal["evidence_orchestration.failed"]
    status: Literal["failed"]
    round_index: RoundIndex
    source: Literal["orchestration", "local", "web", "judge", "assignment"]
    error_type: SafeErrorType
    reason_code: SafeCode
    budget_used_tasks: BoundedCount
    budget_remaining_tasks: BoundedCount


EvidenceProgressDetails: TypeAlias = Annotated[
    EvidencePlanAcceptedDetails
    | EvidenceRoundStartedDetails
    | EvidenceSourceCompletedDetails
    | EvidenceSourceEmptyDetails
    | EvidenceSourceFailedDetails
    | EvidenceRoundMergedDetails
    | EvidenceCoverageJudgedDetails
    | EvidenceProgressEvaluatedDetails
    | EvidenceRouteDecidedDetails
    | EvidenceResourceAssignedDetails
    | EvidenceTerminalDetails
    | EvidenceFailedDetails,
    Field(discriminator="stage"),
]

_DETAILS_ADAPTER: TypeAdapter[EvidenceProgressDetails] = TypeAdapter(
    EvidenceProgressDetails
)


class EvidenceProgressV1(BaseModel):
    """One replay-safe, UI-safe evidence lifecycle update."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["evidence_progress_v1"] = EVIDENCE_PROGRESS_SCHEMA_VERSION
    progress_id: str = Field(pattern=r"^evidence-progress:v1:[a-f0-9]{64}$")
    request_id: str = Field(min_length=1, max_length=160)
    thread_id: str = Field(min_length=1, max_length=160)
    lifecycle_key: str = Field(pattern=r"^[a-z0-9:._-]{1,160}$")
    phase_status: EvidenceProgressPhaseStatus
    details: EvidenceProgressDetails

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        try:
            parsed = UUID(value)
        except ValueError as exc:
            raise ValueError("request_id must be a UUID") from exc
        if str(parsed) != value:
            raise ValueError("request_id must use canonical UUID text")
        return value

    @field_validator("thread_id")
    @classmethod
    def validate_thread_id(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("thread_id must be stripped")
        return value

    @model_validator(mode="after")
    def validate_derived_identity(self) -> "EvidenceProgressV1":
        expected_key = _lifecycle_key(self.details)
        if self.lifecycle_key != expected_key:
            raise ValueError("lifecycle_key does not match evidence details")
        expected_status = _phase_status(self.details)
        if self.phase_status != expected_status:
            raise ValueError("phase_status does not match evidence stage")
        expected_id = _progress_id(
            request_id=self.request_id,
            thread_id=self.thread_id,
            lifecycle_key=expected_key,
        )
        if self.progress_id != expected_id:
            raise ValueError("progress_id does not match evidence identity")
        return self


EvidenceProgressSink = Callable[[EvidenceProgressV1], None]
_EVIDENCE_PROGRESS_SINK: ContextVar[EvidenceProgressSink | None] = ContextVar(
    "evidence_progress_sink",
    default=None,
)


def set_evidence_progress_sink(
    sink: EvidenceProgressSink,
) -> Token[EvidenceProgressSink | None]:
    if not callable(sink):
        raise TypeError("evidence progress sink must be callable")
    return _EVIDENCE_PROGRESS_SINK.set(sink)


def reset_evidence_progress_sink(token: Token[EvidenceProgressSink | None]) -> None:
    _EVIDENCE_PROGRESS_SINK.reset(token)


def publish_evidence_progress(progress: EvidenceProgressV1) -> None:
    """Publish to an installed request sink without swallowing sink failures."""

    validated = EvidenceProgressV1.model_validate(progress)
    sink = _EVIDENCE_PROGRESS_SINK.get()
    if sink is not None:
        sink(validated)


def evidence_progress_sink_active() -> bool:
    """Return whether this execution explicitly requested public progress."""

    return _EVIDENCE_PROGRESS_SINK.get() is not None


def build_evidence_progress(
    trace_event: Mapping[str, object],
    *,
    request_id: str,
    thread_id: str,
) -> EvidenceProgressV1:
    """Project one already-strict trace event into the smaller public contract."""

    stage = trace_event.get("stage")
    if not isinstance(stage, str):
        raise ValueError("evidence trace stage is required")
    detail_fields = _PUBLIC_FIELDS_BY_STAGE.get(stage)
    if detail_fields is None:
        raise ValueError("unsupported evidence progress stage")
    details = _DETAILS_ADAPTER.validate_python(
        {field: trace_event[field] for field in detail_fields}
    )
    lifecycle_key = _lifecycle_key(details)
    return EvidenceProgressV1(
        progress_id=_progress_id(
            request_id=request_id,
            thread_id=thread_id,
            lifecycle_key=lifecycle_key,
        ),
        request_id=request_id,
        thread_id=thread_id,
        lifecycle_key=lifecycle_key,
        phase_status=_phase_status(details),
        details=details,
    )


def _lifecycle_key(details: EvidenceProgressDetails) -> str:
    if isinstance(details, EvidencePlanAcceptedDetails):
        return "plan"
    if isinstance(details, (EvidenceRoundStartedDetails, EvidenceRoundMergedDetails)):
        return f"round:{details.round_index}"
    if isinstance(
        details,
        (
            EvidenceSourceCompletedDetails,
            EvidenceSourceEmptyDetails,
            EvidenceSourceFailedDetails,
        ),
    ):
        return f"round:{details.round_index}:source:{details.source}"
    if isinstance(details, EvidenceCoverageJudgedDetails):
        return f"round:{details.round_index}:coverage"
    if isinstance(details, EvidenceProgressEvaluatedDetails):
        return f"round:{details.round_index}:progress"
    if isinstance(details, EvidenceRouteDecidedDetails):
        return f"round:{details.round_index}:route"
    if isinstance(details, EvidenceResourceAssignedDetails):
        return f"resource:{details.resource_type}"
    if isinstance(details, (EvidenceTerminalDetails, EvidenceFailedDetails)):
        return "terminal"
    raise ValueError("unsupported evidence progress lifecycle")


def _phase_status(details: EvidenceProgressDetails) -> EvidenceProgressPhaseStatus:
    if isinstance(details, EvidenceRoundStartedDetails):
        return "running"
    if isinstance(details, (EvidenceSourceFailedDetails, EvidenceFailedDetails)):
        return "failed"
    return "completed"


def _progress_id(*, request_id: str, thread_id: str, lifecycle_key: str) -> str:
    payload = json.dumps(
        {
            "lifecycle_key": lifecycle_key,
            "request_id": request_id,
            "schema_version": EVIDENCE_PROGRESS_SCHEMA_VERSION,
            "thread_id": thread_id,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"evidence-progress:v1:{hashlib.sha256(payload).hexdigest()}"


_PUBLIC_FIELDS_BY_STAGE: dict[str, tuple[str, ...]] = {
    "evidence_orchestration.plan.accepted": (
        "stage",
        "requirement_count",
        "resource_count",
        "subject_count",
        "budget_max_rounds",
        "budget_max_tasks",
    ),
    "evidence_orchestration.round.started": (
        "stage",
        "round_index",
        "task_count",
        "local_task_count",
        "web_task_count",
        "budget_used_tasks",
        "budget_remaining_tasks",
    ),
    "evidence_orchestration.source.completed": (
        "stage",
        "round_index",
        "source",
        "status",
        "task_count",
        "candidate_count",
        "latency_ms",
    ),
    "evidence_orchestration.source.empty": (
        "stage",
        "round_index",
        "source",
        "status",
        "task_count",
        "latency_ms",
        "reason_code",
    ),
    "evidence_orchestration.source.failed": (
        "stage",
        "round_index",
        "source",
        "status",
        "task_count",
        "latency_ms",
        "reason_code",
        "error_type",
    ),
    "evidence_orchestration.round.merged": (
        "stage",
        "round_index",
        "local_candidate_count",
        "web_candidate_count",
        "deduplicated_count",
        "ledger_count",
    ),
    "evidence_orchestration.coverage.judged": (
        "stage",
        "round_index",
        "requirement_count",
        "complete_count",
        "partial_count",
        "missing_count",
        "accepted_evidence_count",
    ),
    "evidence_orchestration.progress.evaluated": (
        "stage",
        "round_index",
        "previous_complete_count",
        "current_complete_count",
        "previous_partial_count",
        "current_partial_count",
        "previous_missing_count",
        "current_missing_count",
        "new_accepted_evidence_count",
        "progressed",
        "consecutive_no_progress_rounds",
    ),
    "evidence_orchestration.route.decided": (
        "stage",
        "round_index",
        "status",
        "reason_code",
        "next_local_task_count",
        "next_web_task_count",
        "budget_remaining_rounds",
        "budget_remaining_tasks",
    ),
    "evidence_orchestration.resource.assigned": (
        "stage",
        "round_index",
        "resource_type",
        "status",
        "requirement_count",
        "assigned_evidence_count",
        "missing_requirement_count",
    ),
    "evidence_orchestration.terminal": (
        "stage",
        "orchestration_fingerprint",
        "status",
        "rounds_completed",
        "ready_resource_count",
        "blocked_resource_count",
        "total_search_tasks",
        "ledger_count",
        "reason_code",
    ),
    "evidence_orchestration.failed": (
        "stage",
        "status",
        "round_index",
        "source",
        "error_type",
        "reason_code",
        "budget_used_tasks",
        "budget_remaining_tasks",
    ),
}


__all__ = [
    "EVIDENCE_PROGRESS_SCHEMA_VERSION",
    "EvidenceProgressDetails",
    "EvidenceProgressPhaseStatus",
    "EvidenceProgressV1",
    "build_evidence_progress",
    "evidence_progress_sink_active",
    "publish_evidence_progress",
    "reset_evidence_progress_sink",
    "set_evidence_progress_sink",
]
