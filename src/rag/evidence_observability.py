"""Content-free shadow and health contracts for evidence orchestration."""

from __future__ import annotations

import hashlib
import math
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TerminalStatus = Literal[
    "sufficient",
    "partial_resources_ready",
    "insufficient_max_rounds",
    "insufficient_no_progress",
    "insufficient_empty_sources",
    "blocked_insufficient_evidence",
]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class EvidenceShadowSummary(_StrictFrozenModel):
    """Content-free comparable result for one graph variant."""

    schema_version: Literal["evidence_shadow_summary_v1"]
    terminal_status: TerminalStatus
    requirement_count: int = Field(ge=0)
    complete_count: int = Field(ge=0)
    partial_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    round_count: int = Field(ge=1)
    search_task_count: int = Field(ge=0)
    ready_resource_count: int = Field(ge=0)
    blocked_resource_count: int = Field(ge=0)
    ledger_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_partitions(self) -> Self:
        if (
            self.complete_count + self.partial_count + self.missing_count
            != self.requirement_count
        ):
            raise ValueError("coverage counts must partition requirement_count")
        if self.ready_resource_count + self.blocked_resource_count <= 0:
            raise ValueError("shadow summary requires at least one requested resource")
        return self


class EvidenceShadowRecord(_StrictFrozenModel):
    """Versioned shadow outcome; candidate failure never becomes primary output."""

    schema_version: Literal["evidence_shadow_record_v1"]
    request_id_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_graph_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_bundle_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_summary: EvidenceShadowSummary
    candidate_status: Literal["ok", "failed"]
    candidate_summary: EvidenceShadowSummary | None
    candidate_failure_type: str | None
    primary_latency_ms: float = Field(ge=0.0)
    candidate_latency_ms: float = Field(ge=0.0)

    @field_validator("primary_latency_ms", "candidate_latency_ms")
    @classmethod
    def validate_finite_latency(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("shadow latency must be finite")
        return value

    @model_validator(mode="after")
    def validate_candidate_outcome(self) -> Self:
        if self.candidate_status == "failed":
            if self.candidate_summary is not None or not self.candidate_failure_type:
                raise ValueError("failed candidate requires only a failure type")
        elif self.candidate_summary is None or self.candidate_failure_type is not None:
            raise ValueError("successful candidate requires only a validated summary")
        return self


class EvidenceOrchestrationHealthEvent(_StrictFrozenModel):
    """Safe operational event with counts, hashes, and reason codes only."""

    schema_version: Literal["evidence_orchestration_health_v1"]
    request_id_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_id: str = Field(min_length=1)
    candidate_bundle_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    route_kind: Literal["candidate", "shadow"]
    status: Literal["ok", "empty", "failed"]
    terminal_status: TerminalStatus | None
    failure_reason_code: str | None
    round_count: int = Field(ge=0)
    search_task_count: int = Field(ge=0)
    ready_resource_count: int = Field(ge=0)
    blocked_resource_count: int = Field(ge=0)
    ledger_count: int = Field(ge=0)
    total_latency_ms: float = Field(ge=0.0)
    candidate_failure_policy: Literal["fail_fast"]

    @field_validator("total_latency_ms")
    @classmethod
    def validate_finite_latency(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("health latency must be finite")
        return value

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.status == "failed":
            if not self.failure_reason_code or self.terminal_status is not None:
                raise ValueError(
                    "failed health event requires only a failure reason code"
                )
        elif self.failure_reason_code is not None or self.terminal_status is None:
            raise ValueError(
                "successful/empty health event requires only a terminal status"
            )
        return self


def request_id_hash(request_id: str) -> str:
    """Hash one nonblank request identity for shadow and health records."""

    if not request_id or request_id != request_id.strip():
        raise ValueError("request_id must be nonblank and stripped")
    return hashlib.sha256(request_id.encode("utf-8")).hexdigest()


__all__ = [
    "EvidenceOrchestrationHealthEvent",
    "EvidenceShadowRecord",
    "EvidenceShadowSummary",
    "TerminalStatus",
    "request_id_hash",
]
