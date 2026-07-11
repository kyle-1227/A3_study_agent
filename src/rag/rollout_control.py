"""Deterministic, observable rollout routing for immutable RAG generations.

This module never retries a candidate request against the primary generation.
It only selects a generation before request execution and evaluates explicit
control-plane gates after observations have been collected.
"""

from __future__ import annotations

import hashlib
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RolloutControlError(RuntimeError):
    """Raised when a rollout request or observation is inconsistent."""


class RolloutStage(BaseModel):
    """One fully configured rollout stage without implicit thresholds."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    stage_id: str = Field(min_length=1)
    activation_enabled: bool
    candidate_fraction: float = Field(ge=0.0, le=1.0)
    eligible_subjects: tuple[str, ...]
    minimum_evaluable_requests: int = Field(ge=0)
    minimum_observation_seconds: int = Field(gt=0)
    maximum_candidate_error_rate: float = Field(ge=0.0, le=1.0)
    maximum_recall_regression: float = Field(ge=0.0, le=1.0)
    maximum_p95_latency_ratio: float = Field(ge=1.0)
    maximum_context_token_ratio: float = Field(ge=1.0)

    @model_validator(mode="after")
    def _validate_subjects(self) -> RolloutStage:
        if len(set(self.eligible_subjects)) != len(self.eligible_subjects):
            raise ValueError("eligible_subjects must be unique")
        if self.candidate_fraction > 0 and not self.eligible_subjects:
            raise ValueError("candidate traffic requires eligible subjects")
        return self


class GenerationRoute(BaseModel):
    """Auditable generation selection made before request execution."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    stage_id: str
    request_id_hash: str
    subject: str
    generation_id: str
    route_kind: Literal["primary", "candidate"]


class RolloutObservation(BaseModel):
    """Aggregate observation supplied to a control-plane gate."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    stage_id: str
    evaluable_requests: int = Field(ge=0)
    observation_seconds: int = Field(ge=0)
    candidate_error_rate: float = Field(ge=0.0, le=1.0)
    worst_subject_recall_delta: float = Field(ge=-1.0, le=1.0)
    p95_latency_ratio: float = Field(ge=0.0)
    context_token_ratio: float = Field(ge=0.0)
    orphan_child_count: int = Field(ge=0)
    parent_hydration_failure_count: int = Field(ge=0)
    generation_mismatch_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_finite_metrics(self) -> RolloutObservation:
        for value in (
            self.candidate_error_rate,
            self.worst_subject_recall_delta,
            self.p95_latency_ratio,
            self.context_token_ratio,
        ):
            if not math.isfinite(value):
                raise ValueError("rollout metrics must be finite")
        return self


class RolloutDecision(BaseModel):
    """Structured control-plane decision; never a request-level fallback."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    stage_id: str
    action: Literal["hold", "continue", "rollback_required"]
    reason_codes: tuple[str, ...]


def route_generation(
    *,
    request_id: str,
    subject: str,
    primary_generation_id: str,
    candidate_generation_id: str | None,
    stage: RolloutStage,
) -> GenerationRoute:
    """Select one generation deterministically without failure-time rerouting."""

    if not request_id:
        raise RolloutControlError("request_id must not be blank")
    if not primary_generation_id:
        raise RolloutControlError("primary_generation_id must not be blank")

    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    candidate_eligible = (
        stage.activation_enabled
        and subject in stage.eligible_subjects
        and stage.candidate_fraction > 0
    )
    if candidate_eligible and candidate_generation_id is None:
        raise RolloutControlError("candidate generation is required by rollout stage")

    bucket = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    use_candidate = candidate_eligible and bucket < stage.candidate_fraction
    generation_id = candidate_generation_id if use_candidate else primary_generation_id
    if generation_id is None:
        raise RolloutControlError("selected generation is missing")
    return GenerationRoute(
        stage_id=stage.stage_id,
        request_id_hash=digest,
        subject=subject,
        generation_id=generation_id,
        route_kind="candidate" if use_candidate else "primary",
    )


def evaluate_rollout_stage(
    *, stage: RolloutStage, observation: RolloutObservation
) -> RolloutDecision:
    """Evaluate configured rollout gates without lowering any threshold."""

    if observation.stage_id != stage.stage_id:
        raise RolloutControlError("observation stage_id does not match rollout stage")
    if not stage.activation_enabled:
        return RolloutDecision(
            stage_id=stage.stage_id,
            action="hold",
            reason_codes=("activation_disabled",),
        )

    insufficient: list[str] = []
    if observation.evaluable_requests < stage.minimum_evaluable_requests:
        insufficient.append("insufficient_evaluable_requests")
    if observation.observation_seconds < stage.minimum_observation_seconds:
        insufficient.append("insufficient_observation_time")
    if insufficient:
        return RolloutDecision(
            stage_id=stage.stage_id,
            action="hold",
            reason_codes=tuple(insufficient),
        )

    failures: list[str] = []
    if observation.candidate_error_rate > stage.maximum_candidate_error_rate:
        failures.append("candidate_error_rate_exceeded")
    if observation.worst_subject_recall_delta < -stage.maximum_recall_regression:
        failures.append("subject_recall_regression_exceeded")
    if observation.p95_latency_ratio > stage.maximum_p95_latency_ratio:
        failures.append("p95_latency_ratio_exceeded")
    if observation.context_token_ratio > stage.maximum_context_token_ratio:
        failures.append("context_token_ratio_exceeded")
    if observation.orphan_child_count:
        failures.append("orphan_children_detected")
    if observation.parent_hydration_failure_count:
        failures.append("parent_hydration_failure_detected")
    if observation.generation_mismatch_count:
        failures.append("generation_mismatch_detected")

    return RolloutDecision(
        stage_id=stage.stage_id,
        action="rollback_required" if failures else "continue",
        reason_codes=tuple(failures),
    )


def rollout_stage_from_config(
    *,
    rollout_config: object,
    stage_id: str,
) -> RolloutStage:
    """Convert one strict rollout YAML stage without inventing thresholds."""

    from src.config.rag_rollout_config import RagRolloutConfig

    if not isinstance(rollout_config, RagRolloutConfig):
        raise TypeError("rollout_config must be a validated RagRolloutConfig")
    matches = tuple(
        stage for stage in rollout_config.stages if stage.stage_id == stage_id
    )
    if len(matches) != 1:
        raise RolloutControlError(
            "requested rollout stage is not configured exactly once"
        )
    configured = matches[0]
    stops = rollout_config.stop_conditions
    return RolloutStage(
        stage_id=configured.stage_id,
        activation_enabled=rollout_config.activation_enabled,
        candidate_fraction=configured.candidate_allocation_percent / 100.0,
        eligible_subjects=configured.eligible_subjects,
        minimum_evaluable_requests=configured.min_evaluable_requests,
        minimum_observation_seconds=round(configured.min_observation_hours * 3600),
        maximum_candidate_error_rate=stops.max_candidate_error_rate,
        maximum_recall_regression=stops.max_recall_at_5_absolute_regression,
        maximum_p95_latency_ratio=stops.max_p95_latency_baseline_ratio,
        maximum_context_token_ratio=stops.max_context_token_baseline_ratio,
    )
