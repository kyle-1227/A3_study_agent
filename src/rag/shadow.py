"""Baseline-served shadow execution with observable candidate outcomes."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
from time import perf_counter_ns
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PrimaryT = TypeVar("PrimaryT")
CandidateT = TypeVar("CandidateT")


class ShadowExecutionError(RuntimeError):
    """Shadow configuration or primary served execution failed."""


class ShadowComparable(BaseModel):
    """Content-free output summary safe for shadow comparison/telemetry."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    status: Literal["ok", "empty"]
    evidence_count: int = Field(ge=0)
    parent_count: int = Field(ge=0)
    context_tokens: int = Field(ge=0)


class ShadowExecutionRecord(BaseModel):
    """Sanitized record; candidate exceptions never masquerade as success."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["rag_shadow_record_v1"]
    request_id_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    subject: str = Field(min_length=1)
    primary_generation_id: str = Field(min_length=1)
    candidate_generation_id: str = Field(min_length=1)
    primary_summary: ShadowComparable
    candidate_status: Literal["ok", "empty", "failed"]
    candidate_summary: ShadowComparable | None
    candidate_failure_type: str | None
    primary_latency_ms: float = Field(ge=0.0)
    candidate_latency_ms: float = Field(ge=0.0)
    evidence_count_delta: int | None
    parent_count_delta: int | None
    context_token_delta: int | None

    @field_validator("primary_latency_ms", "candidate_latency_ms")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("shadow latencies must be finite")
        return value

    @model_validator(mode="after")
    def _candidate_outcome_contract(self) -> ShadowExecutionRecord:
        deltas = (
            self.evidence_count_delta,
            self.parent_count_delta,
            self.context_token_delta,
        )
        if self.candidate_status == "failed":
            if self.candidate_summary is not None or not self.candidate_failure_type:
                raise ValueError("failed shadow candidate requires only a failure type")
            if any(value is not None for value in deltas):
                raise ValueError("failed shadow candidate cannot carry metric deltas")
        else:
            if (
                self.candidate_summary is None
                or self.candidate_summary.status != self.candidate_status
                or self.candidate_failure_type is not None
                or any(value is None for value in deltas)
            ):
                raise ValueError("successful shadow candidate outcome is inconsistent")
        return self


class ShadowExecutionResult(Generic[PrimaryT]):
    """Served primary output plus a separate candidate comparison record."""

    def __init__(
        self, *, served_output: PrimaryT, record: ShadowExecutionRecord
    ) -> None:
        self.served_output = served_output
        self.record = record


def _elapsed_ms(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000.0


def execute_shadow(
    *,
    request_id: str,
    subject: str,
    primary_generation_id: str,
    candidate_generation_id: str,
    primary_call: Callable[[], PrimaryT],
    candidate_call: Callable[[], CandidateT],
    primary_summarizer: Callable[[PrimaryT], ShadowComparable],
    candidate_summarizer: Callable[[CandidateT], ShadowComparable],
) -> ShadowExecutionResult[PrimaryT]:
    """Execute both routes concurrently while serving only the predetermined primary."""

    if any(
        not value.strip()
        for value in (
            request_id,
            subject,
            primary_generation_id,
            candidate_generation_id,
        )
    ):
        raise ShadowExecutionError("shadow identity fields must be nonblank")

    def timed_primary() -> tuple[PrimaryT, float]:
        start = perf_counter_ns()
        value = primary_call()
        return value, _elapsed_ms(start)

    def timed_candidate() -> tuple[CandidateT | None, float, Exception | None]:
        start = perf_counter_ns()
        try:
            value = candidate_call()
        except Exception as exc:
            return None, _elapsed_ms(start), exc
        return value, _elapsed_ms(start), None

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-shadow") as executor:
        primary_future = executor.submit(timed_primary)
        candidate_future = executor.submit(timed_candidate)
        try:
            primary_output, primary_latency = primary_future.result()
        except Exception as exc:
            raise ShadowExecutionError("primary served execution failed") from exc
        candidate_output, candidate_latency, candidate_error = candidate_future.result()

    primary_summary = primary_summarizer(primary_output)
    candidate_summary: ShadowComparable | None = None
    failure_type: str | None = None
    evidence_delta: int | None
    parent_delta: int | None
    token_delta: int | None
    if candidate_error is None:
        if candidate_output is None:
            raise ShadowExecutionError("candidate returned an impossible null outcome")
        candidate_summary = candidate_summarizer(candidate_output)
        candidate_status: Literal["ok", "empty", "failed"] = candidate_summary.status
        evidence_delta = (
            candidate_summary.evidence_count - primary_summary.evidence_count
        )
        parent_delta = candidate_summary.parent_count - primary_summary.parent_count
        token_delta = candidate_summary.context_tokens - primary_summary.context_tokens
    else:
        candidate_status = "failed"
        failure_type = type(candidate_error).__name__[:128]
        evidence_delta = parent_delta = token_delta = None

    record = ShadowExecutionRecord(
        schema_version="rag_shadow_record_v1",
        request_id_hash=hashlib.sha256(request_id.encode("utf-8")).hexdigest(),
        subject=subject,
        primary_generation_id=primary_generation_id,
        candidate_generation_id=candidate_generation_id,
        primary_summary=primary_summary,
        candidate_status=candidate_status,
        candidate_summary=candidate_summary,
        candidate_failure_type=failure_type,
        primary_latency_ms=primary_latency,
        candidate_latency_ms=candidate_latency,
        evidence_count_delta=evidence_delta,
        parent_count_delta=parent_delta,
        context_token_delta=token_delta,
    )
    return ShadowExecutionResult(served_output=primary_output, record=record)


__all__ = [
    "ShadowComparable",
    "ShadowExecutionError",
    "ShadowExecutionRecord",
    "ShadowExecutionResult",
    "execute_shadow",
]
