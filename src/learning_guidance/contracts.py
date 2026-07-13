"""Strict contracts for learner paths and resource recommendations.

These contracts deliberately contain no inferred identities, neutral scores, or
schema aliases.  Every score-bearing recommendation must carry the exact profile
and history evidence used by its engine.
"""

from __future__ import annotations

from datetime import datetime
import math
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from src.resource_contracts import ResourceType


RecommendationMode = Literal[
    "automatic_after_generation",
    "explicit_request",
]
LearnerPathUnavailableReason = Literal[
    "missing_user_id",
    "missing_subject",
    "profile_unavailable",
    "history_unavailable",
]
RecommendationUnavailableReason = Literal[
    "missing_user_id",
    "missing_subject",
    "profile_unavailable",
    "history_unavailable",
    "generated_resources_unavailable",
]


def _validate_identifier_tuple(
    values: tuple[str, ...],
    *,
    field_name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not allow_empty and not values:
        raise ValueError(f"{field_name} must not be empty")
    if any(not value.strip() or value != value.strip() for value in values):
        raise ValueError(f"{field_name} must contain normalized non-blank strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must contain unique values")
    return values


def _validate_aware_datetime(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


class _StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    @field_validator("*", mode="before", check_fields=False)
    @classmethod
    def reject_unnormalized_direct_text(cls, value: object) -> object:
        if isinstance(value, str) and (not value.strip() or value != value.strip()):
            raise ValueError("text fields must be normalized and non-blank")
        return value


class LearnerSkillSignalV1(_StrictContract):
    signal_id: str = Field(min_length=1, max_length=160)
    subject: str = Field(min_length=1, max_length=120)
    topic_id: str = Field(min_length=1, max_length=160)
    level: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class LearnerGoalSignalV1(_StrictContract):
    signal_id: str = Field(min_length=1, max_length=160)
    subject: str = Field(min_length=1, max_length=120)
    goal: str = Field(min_length=1, max_length=500)
    importance: float = Field(ge=0.0, le=1.0)
    progress: float = Field(ge=0.0, le=1.0)


class LearnerPreferenceSignalV1(_StrictContract):
    signal_id: str = Field(min_length=1, max_length=160)
    dimension: Literal[
        "prefer_examples",
        "prefer_visual",
        "prefer_step_by_step",
        "prefer_concise",
        "prefer_theory",
        "prefer_practice",
        "prefer_analogy",
    ]
    strength: float = Field(ge=0.0, le=1.0)


class LearnerProfileSnapshotV1(_StrictContract):
    schema_version: Literal["learner_profile_snapshot_v1"]
    user_id: str = Field(min_length=1, max_length=160)
    skills: tuple[LearnerSkillSignalV1, ...] = Field(min_length=1, max_length=200)
    goals: tuple[LearnerGoalSignalV1, ...] = Field(min_length=1, max_length=50)
    preferences: tuple[LearnerPreferenceSignalV1, ...] = Field(
        min_length=1,
        max_length=20,
    )

    @model_validator(mode="after")
    def validate_signal_inventory(self) -> "LearnerProfileSnapshotV1":
        signal_ids = (
            tuple(signal.signal_id for signal in self.skills)
            + tuple(signal.signal_id for signal in self.goals)
            + tuple(signal.signal_id for signal in self.preferences)
        )
        _validate_identifier_tuple(
            signal_ids,
            field_name="profile signal ids",
            allow_empty=False,
        )
        skill_slots = tuple((signal.subject, signal.topic_id) for signal in self.skills)
        if len(skill_slots) != len(set(skill_slots)):
            raise ValueError("profile skills must not repeat a subject/topic slot")
        dimensions = tuple(signal.dimension for signal in self.preferences)
        if len(dimensions) != len(set(dimensions)):
            raise ValueError("profile preferences must not repeat a dimension")
        return self

    def supports_subject(self, subject: str) -> bool:
        has_skill = any(signal.subject == subject for signal in self.skills)
        has_goal = any(signal.subject == subject for signal in self.goals)
        return has_skill and has_goal


class LearnerHistoryEventV1(_StrictContract):
    history_id: str = Field(min_length=1, max_length=160)
    subject: str = Field(min_length=1, max_length=120)
    topic_id: str = Field(min_length=1, max_length=160)
    event_type: Literal[
        "practice",
        "assessment",
        "resource_completion",
        "study_session",
    ]
    observed_at: datetime
    outcome_score: float | None = Field(ge=0.0, le=1.0)

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        return _validate_aware_datetime(value, field_name="observed_at")


class LearnerHistorySnapshotV1(_StrictContract):
    schema_version: Literal["learner_history_snapshot_v1"]
    user_id: str = Field(min_length=1, max_length=160)
    subject: str = Field(min_length=1, max_length=120)
    events: tuple[LearnerHistoryEventV1, ...] = Field(
        min_length=1,
        max_length=500,
    )

    @model_validator(mode="after")
    def validate_history_binding(self) -> "LearnerHistorySnapshotV1":
        event_ids = tuple(event.history_id for event in self.events)
        _validate_identifier_tuple(
            event_ids,
            field_name="history event ids",
            allow_empty=False,
        )
        if any(event.subject != self.subject for event in self.events):
            raise ValueError("history events must match the snapshot subject")
        return self

    def history_ids(self) -> frozenset[str]:
        return frozenset(event.history_id for event in self.events)


class LearnerPathEngineRequestV1(_StrictContract):
    schema_version: Literal["learner_path_engine_request_v1"]
    request_id: str = Field(min_length=1, max_length=160)
    user_id: str = Field(min_length=1, max_length=160)
    subject: str = Field(min_length=1, max_length=120)
    profile: LearnerProfileSnapshotV1
    history: LearnerHistorySnapshotV1

    @model_validator(mode="after")
    def validate_input_binding(self) -> "LearnerPathEngineRequestV1":
        if self.profile.user_id != self.user_id:
            raise ValueError("profile user_id must match the planner request")
        if self.history.user_id != self.user_id:
            raise ValueError("history user_id must match the planner request")
        if self.history.subject != self.subject:
            raise ValueError("history subject must match the planner request")
        if not self.profile.supports_subject(self.subject):
            raise ValueError(
                "profile lacks explicit skill and goal signals for subject"
            )
        return self


class LearnerPathStepV1(_StrictContract):
    step_id: str = Field(min_length=1, max_length=160)
    position: int = Field(ge=1)
    topic_id: str = Field(min_length=1, max_length=160)
    subject: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=300)
    status: Literal["ready", "blocked", "reinforce", "repeat", "skip"]
    estimated_hours: float = Field(gt=0.0)
    reason: str = Field(min_length=1, max_length=1000)
    recommended_resource_types: tuple[ResourceType, ...]
    profile_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=50)
    history_ids: tuple[str, ...] = Field(min_length=1, max_length=100)

    @field_validator(
        "recommended_resource_types",
        "profile_signal_ids",
        "history_ids",
    )
    @classmethod
    def validate_identifier_collections(
        cls,
        value: tuple[str, ...],
        info: ValidationInfo,
    ) -> tuple[str, ...]:
        return _validate_identifier_tuple(
            value,
            field_name=info.field_name,
            allow_empty=info.field_name == "recommended_resource_types",
        )


class LearnerPathPlanV1(_StrictContract):
    schema_version: Literal["learner_path_plan_v1"]
    user_id: str = Field(min_length=1, max_length=160)
    subject: str = Field(min_length=1, max_length=120)
    generated_at: datetime
    steps: tuple[LearnerPathStepV1, ...] = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=2000)

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: datetime) -> datetime:
        return _validate_aware_datetime(value, field_name="generated_at")

    @model_validator(mode="after")
    def validate_plan(self) -> "LearnerPathPlanV1":
        step_ids = tuple(step.step_id for step in self.steps)
        _validate_identifier_tuple(
            step_ids,
            field_name="learning path step ids",
            allow_empty=False,
        )
        positions = tuple(step.position for step in self.steps)
        if positions != tuple(range(1, len(self.steps) + 1)):
            raise ValueError("learning path positions must be contiguous and ordered")
        if any(step.subject != self.subject for step in self.steps):
            raise ValueError("learning path steps must match the plan subject")
        return self


class RecommendationResourceContextV1(_StrictContract):
    resource_id: str = Field(min_length=1, max_length=200)
    resource_type: ResourceType
    subject: str = Field(min_length=1, max_length=120)


class ResourceRecommendationEngineRequestV1(_StrictContract):
    schema_version: Literal["resource_recommendation_engine_request_v1"]
    request_id: str = Field(min_length=1, max_length=160)
    mode: RecommendationMode
    user_id: str = Field(min_length=1, max_length=160)
    subject: str = Field(min_length=1, max_length=120)
    profile: LearnerProfileSnapshotV1
    history: LearnerHistorySnapshotV1
    generated_resources: tuple[RecommendationResourceContextV1, ...]

    @model_validator(mode="after")
    def validate_input_binding(self) -> "ResourceRecommendationEngineRequestV1":
        if self.profile.user_id != self.user_id:
            raise ValueError("profile user_id must match recommendation request")
        if self.history.user_id != self.user_id:
            raise ValueError("history user_id must match recommendation request")
        if self.history.subject != self.subject:
            raise ValueError("history subject must match recommendation request")
        if not self.profile.supports_subject(self.subject):
            raise ValueError(
                "profile lacks explicit skill and goal signals for subject"
            )
        resource_ids = tuple(item.resource_id for item in self.generated_resources)
        _validate_identifier_tuple(
            resource_ids,
            field_name="generated resource ids",
            allow_empty=self.mode == "explicit_request",
        )
        if any(item.subject != self.subject for item in self.generated_resources):
            raise ValueError("generated resources must match recommendation subject")
        if self.mode == "explicit_request" and self.generated_resources:
            raise ValueError(
                "explicit recommendation requests cannot inject generated resources"
            )
        return self


class RecommendationScoreWeightsV1(_StrictContract):
    weakness: float = Field(ge=0.0, le=1.0)
    forgetting: float = Field(ge=0.0, le=1.0)
    preference: float = Field(ge=0.0, le=1.0)
    goal: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_total_weight(self) -> "RecommendationScoreWeightsV1":
        total = self.weakness + self.forgetting + self.preference + self.goal
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("recommendation score weights must sum to 1")
        return self


class RecommendationScoreFactorsV1(_StrictContract):
    weakness: float = Field(ge=0.0, le=1.0)
    forgetting: float = Field(ge=0.0, le=1.0)
    preference: float = Field(ge=0.0, le=1.0)
    goal: float = Field(ge=0.0, le=1.0)
    combined: float = Field(ge=0.0, le=1.0)
    weights: RecommendationScoreWeightsV1

    @model_validator(mode="after")
    def validate_combined_score(self) -> "RecommendationScoreFactorsV1":
        expected = (
            self.weakness * self.weights.weakness
            + self.forgetting * self.weights.forgetting
            + self.preference * self.weights.preference
            + self.goal * self.weights.goal
        )
        if not math.isclose(self.combined, expected, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("combined recommendation score must match its factors")
        return self


class ResourceRecommendationItemV1(_StrictContract):
    recommendation_id: str = Field(min_length=1, max_length=200)
    resource_id: str = Field(min_length=1, max_length=200)
    resource_type: ResourceType
    subject: str = Field(min_length=1, max_length=120)
    topic_id: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=300)
    rank: int = Field(ge=1)
    score_factors: RecommendationScoreFactorsV1
    reason: str = Field(min_length=1, max_length=1000)
    profile_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=50)
    history_ids: tuple[str, ...] = Field(min_length=1, max_length=100)
    source_resource_ids: tuple[str, ...] = Field(max_length=50)

    @field_validator("profile_signal_ids", "history_ids", "source_resource_ids")
    @classmethod
    def validate_reference_collections(
        cls,
        value: tuple[str, ...],
        info: ValidationInfo,
    ) -> tuple[str, ...]:
        return _validate_identifier_tuple(
            value,
            field_name=info.field_name,
            allow_empty=info.field_name == "source_resource_ids",
        )


class ResourceRecommendationBatchV1(_StrictContract):
    schema_version: Literal["resource_recommendation_batch_v1"]
    mode: RecommendationMode
    user_id: str = Field(min_length=1, max_length=160)
    subject: str = Field(min_length=1, max_length=120)
    generated_at: datetime
    items: tuple[ResourceRecommendationItemV1, ...] = Field(
        min_length=1,
        max_length=50,
    )
    summary: str = Field(min_length=1, max_length=2000)

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: datetime) -> datetime:
        return _validate_aware_datetime(value, field_name="generated_at")

    @model_validator(mode="after")
    def validate_batch(self) -> "ResourceRecommendationBatchV1":
        recommendation_ids = tuple(item.recommendation_id for item in self.items)
        _validate_identifier_tuple(
            recommendation_ids,
            field_name="recommendation ids",
            allow_empty=False,
        )
        ranks = tuple(item.rank for item in self.items)
        if ranks != tuple(range(1, len(self.items) + 1)):
            raise ValueError("recommendation ranks must be contiguous and ordered")
        if any(item.subject != self.subject for item in self.items):
            raise ValueError("recommendation items must match the batch subject")
        if self.mode == "automatic_after_generation" and any(
            not item.source_resource_ids for item in self.items
        ):
            raise ValueError(
                "automatic recommendations must reference generated resources"
            )
        if self.mode == "explicit_request" and any(
            item.source_resource_ids for item in self.items
        ):
            raise ValueError(
                "explicit recommendations cannot reference generated resources"
            )
        return self


class LearnerPathPlannerOutputV1(_StrictContract):
    schema_version: Literal["learner_path_planner_output_v1"]
    request_id: str = Field(min_length=1, max_length=160)
    status: Literal["available", "unavailable"]
    unavailable_reason: LearnerPathUnavailableReason | None
    user_id: str | None = Field(max_length=160)
    subject: str | None = Field(max_length=120)
    plan: LearnerPathPlanV1 | None

    @model_validator(mode="after")
    def validate_status_payload(self) -> "LearnerPathPlannerOutputV1":
        if self.status == "available":
            if self.unavailable_reason is not None or self.plan is None:
                raise ValueError("available planner output requires only a plan")
            if self.user_id is None or self.subject is None:
                raise ValueError("available planner output requires user and subject")
            if self.plan.user_id != self.user_id or self.plan.subject != self.subject:
                raise ValueError("available plan identity must match the node output")
            return self
        if self.unavailable_reason is None or self.plan is not None:
            raise ValueError("unavailable planner output requires only a reason")
        if self.unavailable_reason == "missing_user_id" and self.user_id is not None:
            raise ValueError("missing_user_id output cannot contain a user_id")
        if self.unavailable_reason == "missing_subject" and self.subject is not None:
            raise ValueError("missing_subject output cannot contain a subject")
        return self


class ResourceRecommendationOutputV1(_StrictContract):
    schema_version: Literal["resource_recommendation_output_v1"]
    request_id: str = Field(min_length=1, max_length=160)
    mode: RecommendationMode
    status: Literal["available", "unavailable"]
    unavailable_reason: RecommendationUnavailableReason | None
    user_id: str | None = Field(max_length=160)
    subject: str | None = Field(max_length=120)
    batch: ResourceRecommendationBatchV1 | None

    @model_validator(mode="after")
    def validate_status_payload(self) -> "ResourceRecommendationOutputV1":
        if self.status == "available":
            if self.unavailable_reason is not None or self.batch is None:
                raise ValueError(
                    "available recommendation output requires only a batch"
                )
            if self.user_id is None or self.subject is None:
                raise ValueError("available recommendation output requires identity")
            if (
                self.batch.user_id != self.user_id
                or self.batch.subject != self.subject
                or self.batch.mode != self.mode
            ):
                raise ValueError("recommendation batch binding must match node output")
            return self
        if self.unavailable_reason is None or self.batch is not None:
            raise ValueError("unavailable recommendation output requires only a reason")
        if self.unavailable_reason == "missing_user_id" and self.user_id is not None:
            raise ValueError("missing_user_id output cannot contain a user_id")
        if self.unavailable_reason == "missing_subject" and self.subject is not None:
            raise ValueError("missing_subject output cannot contain a subject")
        return self


__all__ = [
    "LearnerGoalSignalV1",
    "LearnerHistoryEventV1",
    "LearnerHistorySnapshotV1",
    "LearnerPathEngineRequestV1",
    "LearnerPathPlanV1",
    "LearnerPathPlannerOutputV1",
    "LearnerPathStepV1",
    "LearnerPathUnavailableReason",
    "LearnerPreferenceSignalV1",
    "LearnerProfileSnapshotV1",
    "LearnerSkillSignalV1",
    "RecommendationMode",
    "RecommendationResourceContextV1",
    "RecommendationScoreFactorsV1",
    "RecommendationScoreWeightsV1",
    "RecommendationUnavailableReason",
    "ResourceRecommendationBatchV1",
    "ResourceRecommendationEngineRequestV1",
    "ResourceRecommendationItemV1",
    "ResourceRecommendationOutputV1",
]
