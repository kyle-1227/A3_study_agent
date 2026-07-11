"""Strict shadow, canary, and explicit rollback configuration for RAG."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, field_validator, model_validator

from src.config._rag_config import (
    NonBlankStr,
    NonBlankStrTuple,
    StrictRagConfigModel,
    load_strict_rag_yaml,
)


NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
AllocationPercent = Annotated[float, Field(ge=0.0, le=100.0)]

RolloutStageId = Literal[
    "shadow",
    "internal",
    "single_subject",
    "multi_subject",
    "expand_1",
    "expand_2",
    "full",
]

_STAGE_ORDER: tuple[RolloutStageId, ...] = (
    "shadow",
    "internal",
    "single_subject",
    "multi_subject",
    "expand_1",
    "expand_2",
    "full",
)


def _freeze_sequence(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


class RolloutStageConfig(StrictRagConfigModel):
    stage_id: RolloutStageId
    candidate_allocation_percent: AllocationPercent
    min_evaluable_requests: NonNegativeInt
    min_observation_hours: NonNegativeFloat
    eligible_subjects: NonBlankStrTuple
    internal_only: bool

    @field_validator("eligible_subjects")
    @classmethod
    def _validate_eligible_subjects(
        cls,
        subjects: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(subjects) != len(set(subjects)):
            raise ValueError("eligible_subjects must not contain duplicates")
        return subjects


class RolloutStopConditions(StrictRagConfigModel):
    max_candidate_error_rate: UnitFloat
    max_p95_latency_baseline_ratio: Annotated[float, Field(ge=1.0)]
    max_context_token_baseline_ratio: Annotated[float, Field(ge=1.0)]
    max_recall_at_5_absolute_regression: UnitFloat
    max_answer_correctness_absolute_regression: UnitFloat
    max_citation_support_absolute_regression: UnitFloat
    max_hallucination_absolute_increase: UnitFloat
    max_integrity_failures: Literal[0]
    max_generation_mismatches: Literal[0]
    max_parent_hydration_failures: Literal[0]


class RagRolloutConfig(StrictRagConfigModel):
    """Complete rollout policy with no request-level failure switching."""

    schema_version: NonBlankStr
    activation_enabled: bool
    shadow_enabled: bool
    primary_subjects: NonBlankStrTuple
    request_hash_algorithm: Literal["sha256_v1"]
    candidate_failure_policy: Literal["fail_fast"]
    rollback_mode: Literal["explicit_registry_activation"]
    benchmark_eligibility_required: Literal[True]
    stages: Annotated[
        tuple[RolloutStageConfig, ...],
        BeforeValidator(_freeze_sequence),
        Field(min_length=1),
    ]
    stop_conditions: RolloutStopConditions

    @field_validator("primary_subjects")
    @classmethod
    def _validate_primary_subjects(cls, subjects: tuple[str, ...]) -> tuple[str, ...]:
        if not subjects:
            raise ValueError("primary_subjects must not be empty")
        if len(subjects) != len(set(subjects)):
            raise ValueError("primary_subjects must not contain duplicates")
        for subject in subjects:
            if subject != subject.casefold():
                raise ValueError("primary_subjects must already be case-folded")
            if (
                subject.startswith("_")
                or subject.endswith("_")
                or "__" in subject
                or not all(
                    character.isalnum() or character == "_" for character in subject
                )
            ):
                raise ValueError("primary_subjects must contain normalized identifiers")
        return subjects

    @model_validator(mode="after")
    def _validate_rollout_sequence(self) -> "RagRolloutConfig":
        stage_ids = tuple(stage.stage_id for stage in self.stages)
        if stage_ids != _STAGE_ORDER:
            raise ValueError(
                "stages must contain the complete ordered rollout sequence exactly once"
            )
        allocations = tuple(stage.candidate_allocation_percent for stage in self.stages)
        if allocations[0] != 0.0:
            raise ValueError("shadow candidate allocation must be 0 percent")
        if allocations[-1] != 100.0:
            raise ValueError("full candidate allocation must be 100 percent")
        external_allocations = allocations[2:]
        if external_allocations != tuple(sorted(external_allocations)):
            raise ValueError(
                "external candidate allocations must be monotonically increasing"
            )

        primary = set(self.primary_subjects)
        for stage in self.stages:
            unknown = set(stage.eligible_subjects) - primary
            if unknown:
                raise ValueError("stage eligible_subjects must be primary subjects")
        if set(self.stages[-1].eligible_subjects) != primary:
            raise ValueError("full stage must include every primary subject")
        return self


def load_rag_rollout_config(config_path: Path) -> RagRolloutConfig:
    """Load a required RAG rollout YAML file."""
    return load_strict_rag_yaml(config_path, RagRolloutConfig)


__all__ = [
    "RagRolloutConfig",
    "RolloutStageConfig",
    "RolloutStopConditions",
    "load_rag_rollout_config",
]
