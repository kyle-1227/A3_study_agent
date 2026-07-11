"""Content-free RAG health metrics and fingerprint-drift invalidation."""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class GenerationValidationFingerprint(_StrictFrozenModel):
    schema_version: Literal["generation_validation_fingerprint_v1"]
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    subject_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_dataset_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class ValidationReuseDecision(_StrictFrozenModel):
    schema_version: Literal["validation_reuse_decision_v1"]
    reusable: bool
    invalidation_codes: tuple[str, ...]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.reusable == bool(self.invalidation_codes):
            raise ValueError("validation reuse flag conflicts with invalidation codes")
        if self.invalidation_codes != tuple(sorted(set(self.invalidation_codes))):
            raise ValueError("validation invalidation codes must be sorted and unique")
        return self


class RetrievalHealthEvent(_StrictFrozenModel):
    """Bounded metric event with hashes/counts/reason codes and no source body."""

    schema_version: Literal["retrieval_health_event_v1"]
    request_id_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_id: str = Field(min_length=1)
    retrieval_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    subject: str = Field(min_length=1)
    route_kind: Literal["primary", "candidate", "shadow"]
    status: Literal["ok", "empty", "failed"]
    failure_reason_code: str | None
    vector_ms: float = Field(ge=0.0)
    bm25_ms: float = Field(ge=0.0)
    reranker_ms: float = Field(ge=0.0)
    aggregate_ms: float = Field(ge=0.0)
    hydrate_ms: float = Field(ge=0.0)
    total_ms: float = Field(ge=0.0)
    child_hit_count: int = Field(ge=0)
    parent_hit_count: int = Field(ge=0)
    context_tokens: int = Field(ge=0)
    judge_keep_count: int = Field(ge=0)
    orphan_child_count: int = Field(ge=0)
    generation_mismatch_count: int = Field(ge=0)
    parent_hydration_failure_count: int = Field(ge=0)

    @field_validator(
        "vector_ms",
        "bm25_ms",
        "reranker_ms",
        "aggregate_ms",
        "hydrate_ms",
        "total_ms",
    )
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("health event timings must be finite")
        return value

    @model_validator(mode="after")
    def _failure_contract(self) -> Self:
        if self.status == "failed" and not self.failure_reason_code:
            raise ValueError("failed health events require a reason code")
        if self.status != "failed" and self.failure_reason_code is not None:
            raise ValueError(
                "successful/empty health events cannot carry failure reason"
            )
        return self


def assess_validation_reuse(
    *,
    validated: GenerationValidationFingerprint,
    current: GenerationValidationFingerprint,
) -> ValidationReuseDecision:
    fields_to_codes = {
        "source_fingerprint": "source_changed",
        "subject_fingerprint": "subject_inventory_changed",
        "policy_fingerprint": "policy_changed",
        "embedding_fingerprint": "embedding_changed",
        "benchmark_dataset_fingerprint": "benchmark_dataset_changed",
    }
    codes = tuple(
        sorted(
            code
            for field, code in fields_to_codes.items()
            if getattr(validated, field) != getattr(current, field)
        )
    )
    return ValidationReuseDecision(
        schema_version="validation_reuse_decision_v1",
        reusable=not codes,
        invalidation_codes=codes,
    )


__all__ = [
    "GenerationValidationFingerprint",
    "RetrievalHealthEvent",
    "ValidationReuseDecision",
    "assess_validation_reuse",
]
