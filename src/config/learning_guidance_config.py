"""Strict production policy for learning-guidance adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BeforeValidator, Field, field_validator, model_validator

from src.config._rag_config import (
    ConfigPath,
    StrictRagConfigModel,
    load_strict_rag_yaml,
)
from src.resource_contracts import RESOURCE_TYPE_ORDER, ResourceType


PreferenceDimension: TypeAlias = Literal[
    "prefer_examples",
    "prefer_visual",
    "prefer_step_by_step",
    "prefer_concise",
    "prefer_theory",
    "prefer_practice",
    "prefer_analogy",
]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveInt = Annotated[int, Field(gt=0)]


def _freeze_sequence(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


class LearnerPathPolicyV1(StrictRagConfigModel):
    max_steps: PositiveInt
    mastery_level: UnitFloat
    mastery_confidence: UnitFloat
    reinforce_level: UnitFloat
    repeat_outcome_threshold: UnitFloat
    recent_failure_window_days: PositiveInt

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "LearnerPathPolicyV1":
        if not self.reinforce_level < self.mastery_level:
            raise ValueError("reinforce_level must be below mastery_level")
        return self


class RecommendationWeightsV1(StrictRagConfigModel):
    weakness: UnitFloat
    forgetting: UnitFloat
    preference: UnitFloat
    goal: UnitFloat

    @model_validator(mode="after")
    def validate_total(self) -> "RecommendationWeightsV1":
        total = self.weakness + self.forgetting + self.preference + self.goal
        if abs(total - 1.0) > 1e-9:
            raise ValueError("recommendation weights must sum to one")
        return self


class ResourcePreferenceBindingV1(StrictRagConfigModel):
    resource_type: ResourceType
    preference_dimension: PreferenceDimension


class ResourceRecommendationPolicyV1(StrictRagConfigModel):
    top_n: PositiveInt
    min_combined_score: UnitFloat
    forgetting_horizon_days: PositiveInt
    weights: RecommendationWeightsV1
    resource_preferences: Annotated[
        tuple[ResourcePreferenceBindingV1, ...],
        BeforeValidator(_freeze_sequence),
        Field(min_length=len(RESOURCE_TYPE_ORDER), max_length=len(RESOURCE_TYPE_ORDER)),
    ]

    @field_validator("resource_preferences")
    @classmethod
    def validate_resource_inventory(
        cls,
        bindings: tuple[ResourcePreferenceBindingV1, ...],
    ) -> tuple[ResourcePreferenceBindingV1, ...]:
        if tuple(item.resource_type for item in bindings) != RESOURCE_TYPE_ORDER:
            raise ValueError(
                "resource_preferences must contain every canonical resource type "
                "exactly once in canonical order"
            )
        return bindings

    def preference_for(self, resource_type: ResourceType) -> PreferenceDimension:
        for binding in self.resource_preferences:
            if binding.resource_type == resource_type:
                return binding.preference_dimension
        raise ValueError(f"missing resource preference binding: {resource_type}")


class LearningGuidanceConfigV1(StrictRagConfigModel):
    schema_version: Literal["learning_guidance_config_v1"]
    adapter_version: Literal["learning_guidance_adapters_v1"]
    knowledge_graph_path: ConfigPath
    provider_projection_max_steps: PositiveInt
    provider_projection_max_chars: PositiveInt
    history_limit: PositiveInt
    path_policy: LearnerPathPolicyV1
    recommendation_policy: ResourceRecommendationPolicyV1

    @model_validator(mode="after")
    def validate_cross_policy_limits(self) -> "LearningGuidanceConfigV1":
        if self.path_policy.max_steps > self.provider_projection_max_steps:
            raise ValueError(
                "path max_steps must fit the provider projection step limit"
            )
        return self

    @property
    def policy_fingerprint(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_learning_guidance_config(
    config_path: Path,
) -> LearningGuidanceConfigV1:
    return load_strict_rag_yaml(config_path, LearningGuidanceConfigV1)


__all__ = [
    "LearnerPathPolicyV1",
    "LearningGuidanceConfigV1",
    "PreferenceDimension",
    "RecommendationWeightsV1",
    "ResourcePreferenceBindingV1",
    "ResourceRecommendationPolicyV1",
    "load_learning_guidance_config",
]
