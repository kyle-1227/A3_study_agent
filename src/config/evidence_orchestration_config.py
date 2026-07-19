"""Strict configuration for resource-aware evidence orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, field_validator, model_validator

from src.config._rag_config import (
    NonBlankStr,
    StrictRagConfigModel,
    load_strict_rag_yaml,
)
from src.resource_contracts import RESOURCE_TYPE_ORDER, ResourceType


PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
NonNegativeInt = Annotated[int, Field(ge=0)]
UnitFloat = Annotated[float, Field(gt=0.0, le=1.0)]

EvidenceCriticality = Literal["required", "supporting"]
EvidenceNeedScope = Literal["per_subject"]
EvidenceSourcePolicy = Literal[
    "local_only",
    "web_only",
    "local_then_web_on_gap",
    "local_and_web",
]
RetrievalPriority = Literal["high", "medium", "low"]

CANONICAL_RESOURCE_TYPES = RESOURCE_TYPE_ORDER


def _freeze_sequence(value: object) -> object:
    """Freeze YAML sequences before strict item validation."""
    if isinstance(value, list):
        return tuple(value)
    return value


def _validate_normalized_identifier(value: str, field_name: str) -> str:
    if value != value.casefold():
        raise ValueError(f"{field_name} must already be case-folded")
    if (
        value.startswith("_")
        or value.endswith("_")
        or "__" in value
        or not all(character.isalnum() or character == "_" for character in value)
    ):
        raise ValueError(f"{field_name} must be a normalized identifier")
    return value


class RetrievalPriorityWeights(StrictRagConfigModel):
    """Explicit adapter weights for typed task priorities."""

    high: UnitFloat
    medium: UnitFloat
    low: UnitFloat

    @model_validator(mode="after")
    def validate_order(self) -> "RetrievalPriorityWeights":
        if not self.high > self.medium > self.low:
            raise ValueError("priority weights must be strictly high > medium > low")
        return self

    def weight_for(self, priority: RetrievalPriority) -> float:
        """Resolve one already-validated priority without a business-node mapping."""

        return {"high": self.high, "medium": self.medium, "low": self.low}[priority]


class EvidenceJudgePartitionReaskConfig(StrictRagConfigModel):
    """Explicit bounded recovery policy for the Evidence Judge."""

    schema_version: Literal["evidence_judge_partition_reask_v1"]
    strategy: Literal["resource_subject_partition_v1"]
    max_partition_calls: PositiveInt
    incomplete_partition_policy: Literal["block_resource"]


class EvidenceFallbackTriggerPolicy(StrictRagConfigModel):
    """Explicit terminal-reason policy for bounded fallback delivery."""

    schema_version: Literal["evidence_fallback_trigger_v1"]
    supplement_round_budget_exhausted: Literal["fallback_if_evidence_eligible"]
    no_measurable_coverage_progress: Literal["fallback_at_configured_threshold"]


class EvidenceFallbackDeliveryConfig(StrictRagConfigModel):
    """Explicit, bounded delivery policy for evidence-limited resources."""

    schema_version: Literal["evidence_fallback_delivery_v2"]
    trigger_policy: EvidenceFallbackTriggerPolicy
    eligible_resource_types: Annotated[
        tuple[ResourceType, ...],
        BeforeValidator(_freeze_sequence),
        Field(min_length=1),
    ]
    minimum_accepted_evidence_per_resource: PositiveInt
    max_resource_generation_attempts: Literal[1]
    max_delivery_seconds: PositiveFloat
    additional_retrieval_task_budget: Literal[0]
    runtime_identity_policy: Literal["inherit_normal_resource_runtime"]
    evidence_binding_policy: Literal["accepted_bound_only"]
    validation_policy: Literal["normal_pydantic_and_business_validators"]
    no_evidence_policy: Literal["block_resource"]
    validation_failure_policy: Literal["block_resource"]
    terminal_status: Literal["partial_success"]

    @model_validator(mode="after")
    def _validate_complete_resource_coverage(self) -> "EvidenceFallbackDeliveryConfig":
        if self.eligible_resource_types != CANONICAL_RESOURCE_TYPES:
            raise ValueError(
                "fallback delivery must cover every canonical resource exactly once "
                "in canonical order"
            )
        return self


class EvidenceOrchestrationConfig(StrictRagConfigModel):
    """Bounded orchestration policy with explicit failure behavior."""

    schema_version: Literal["evidence_orchestration_config_v3"]
    max_supplement_rounds: NonNegativeInt
    max_search_tasks_per_round: PositiveInt
    max_total_search_tasks: PositiveInt
    max_concurrent_tasks: PositiveInt
    max_results_per_task: PositiveInt
    max_requirements_per_request: PositiveInt
    max_ledger_entries: PositiveInt
    max_evidence_per_requirement: PositiveInt
    max_consecutive_no_progress_rounds: PositiveInt
    judge_partition_reask: EvidenceJudgePartitionReaskConfig
    fallback_delivery: EvidenceFallbackDeliveryConfig
    web_timeout_seconds: PositiveFloat
    required_task_priority: RetrievalPriority
    supporting_task_priority: RetrievalPriority
    retrieval_priority_weights: RetrievalPriorityWeights
    query_fingerprint_algorithm: Literal["sha256_v1"]
    evidence_dedupe_policy: Literal["exact_identity_v1"]
    generation_policy: Literal["per_resource_ready"]
    candidate_failure_policy: Literal["fail_fast"]
    source_error_policy: Literal["fail_fast"]

    @model_validator(mode="after")
    def _validate_budgets(self) -> "EvidenceOrchestrationConfig":
        configured_rounds = self.max_supplement_rounds + 1
        round_capacity = configured_rounds * self.max_search_tasks_per_round
        if self.max_total_search_tasks > round_capacity:
            raise ValueError(
                "max_total_search_tasks must fit the configured initial and "
                "supplement rounds"
            )
        if self.max_concurrent_tasks > self.max_search_tasks_per_round:
            raise ValueError(
                "max_concurrent_tasks must not exceed max_search_tasks_per_round"
            )
        maximum_results = self.max_total_search_tasks * self.max_results_per_task
        if self.max_ledger_entries < maximum_results:
            raise ValueError(
                "max_ledger_entries must hold the maximum configured retrieval results"
            )
        if self.max_evidence_per_requirement > self.max_ledger_entries:
            raise ValueError(
                "max_evidence_per_requirement must not exceed max_ledger_entries"
            )
        if self.required_task_priority == self.supporting_task_priority:
            raise ValueError(
                "required and supporting task priorities must be explicitly distinct"
            )
        return self


class ResourceEvidenceNeed(StrictRagConfigModel):
    """One explicit evidence need used to compile resource requirements."""

    need_id: NonBlankStr
    evidence_kind: NonBlankStr
    scope: EvidenceNeedScope
    criticality: EvidenceCriticality
    source_policy: EvidenceSourcePolicy
    acceptance_criteria: NonBlankStr

    @field_validator("need_id", "evidence_kind")
    @classmethod
    def _validate_identifiers(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "identifier")
        return _validate_normalized_identifier(value, field_name)


class ResourceEvidenceProfile(StrictRagConfigModel):
    """Evidence needs for one canonical generated resource."""

    resource_type: ResourceType
    needs: Annotated[
        tuple[ResourceEvidenceNeed, ...],
        BeforeValidator(_freeze_sequence),
        Field(min_length=1),
    ]

    @field_validator("needs")
    @classmethod
    def _validate_unique_needs(
        cls,
        needs: tuple[ResourceEvidenceNeed, ...],
    ) -> tuple[ResourceEvidenceNeed, ...]:
        need_ids = tuple(need.need_id for need in needs)
        if len(need_ids) != len(set(need_ids)):
            raise ValueError("profile need_id values must be unique")
        return needs


class ResourceEvidenceProfilesConfig(StrictRagConfigModel):
    """Complete evidence profile inventory for every canonical resource."""

    schema_version: Literal["resource_evidence_profiles_v1"]
    profiles: Annotated[
        tuple[ResourceEvidenceProfile, ...],
        BeforeValidator(_freeze_sequence),
        Field(min_length=1),
    ]

    @model_validator(mode="after")
    def _validate_complete_inventory(self) -> "ResourceEvidenceProfilesConfig":
        resource_types = tuple(profile.resource_type for profile in self.profiles)
        if resource_types != CANONICAL_RESOURCE_TYPES:
            raise ValueError(
                "profiles must contain every canonical resource exactly once "
                "in canonical order"
            )
        return self

    def profile_for(self, resource_type: ResourceType) -> ResourceEvidenceProfile:
        """Return the exact configured profile without alias or fallback lookup."""
        for profile in self.profiles:
            if profile.resource_type == resource_type:
                return profile
        raise ValueError(f"missing canonical resource profile: {resource_type}")


def load_evidence_orchestration_config(
    config_path: Path,
) -> EvidenceOrchestrationConfig:
    """Load the required orchestration YAML file without fallback behavior."""
    return load_strict_rag_yaml(config_path, EvidenceOrchestrationConfig)


def load_resource_evidence_profiles(
    config_path: Path,
) -> ResourceEvidenceProfilesConfig:
    """Load the required resource evidence profile YAML file."""
    return load_strict_rag_yaml(config_path, ResourceEvidenceProfilesConfig)


__all__ = [
    "CANONICAL_RESOURCE_TYPES",
    "EvidenceCriticality",
    "EvidenceFallbackDeliveryConfig",
    "EvidenceFallbackTriggerPolicy",
    "EvidenceJudgePartitionReaskConfig",
    "EvidenceNeedScope",
    "EvidenceOrchestrationConfig",
    "EvidenceSourcePolicy",
    "ResourceEvidenceNeed",
    "ResourceEvidenceProfile",
    "ResourceEvidenceProfilesConfig",
    "ResourceType",
    "RetrievalPriority",
    "RetrievalPriorityWeights",
    "load_evidence_orchestration_config",
    "load_resource_evidence_profiles",
]
