"""Strict content-bearing inputs and content-free rollout evaluation outputs."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from src.config._rag_config import StrictRagConfigModel, load_strict_rag_yaml
from src.rag.parent_child.evidence_evaluation import (
    EvidenceActivationDecision,
    EvidenceEvaluationCaseResult,
    Variant,
)
from src.resource_contracts import ResourceType


Sha256Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
ComponentStatus = Literal["ok", "not_required", "failed"]
EvidenceSource = Literal["parent_child", "web"]
ExecutionMode = Literal["hermetic", "live"]
DecisionStatus = Literal["pass", "fail", "blocked"]

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
_CANONICAL_VARIANTS: tuple[tuple[Variant, bool, bool], ...] = (
    ("P0", False, False),
    ("PG", True, False),
    ("PR", False, True),
    ("PGR", True, True),
)


class _StrictFrozenModel(StrictRagConfigModel):
    pass


def _identifier(value: str, *, field_name: str) -> str:
    if value != value.strip() or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a normalized identifier")
    return value


def _normalized_subject(value: str) -> str:
    if value != value.casefold() or value != value.strip():
        raise ValueError("subject must already be case-folded and stripped")
    if (
        not value
        or value.startswith("_")
        or value.endswith("_")
        or "__" in value
        or not all(character.isalnum() or character == "_" for character in value)
    ):
        raise ValueError("subject must be a normalized identifier")
    return value


def canonical_sha256(payload: object) -> str:
    """Hash one JSON-compatible payload with a single canonical encoding."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_fingerprint(model: BaseModel) -> str:
    """Fingerprint one already-validated model without accepting mappings."""

    if not isinstance(model, BaseModel):
        raise TypeError("model must be a validated Pydantic model")
    return canonical_sha256(model.model_dump(mode="json"))


def query_fingerprint(query: str) -> str:
    """Bind execution to local query text without exposing it in reports."""

    if not isinstance(query, str) or not query or query != query.strip():
        raise ValueError("query must be a non-blank stripped string")
    return canonical_sha256(
        {"algorithm": "evidence_evaluation_query_v2", "query": query}
    )


class EvidenceVariantDefinitionV2(_StrictFrozenModel):
    variant: Variant
    resource_planning_enabled: bool
    bounded_repair_enabled: bool


class EvidenceLiveAdapterIdentityV2(_StrictFrozenModel):
    """Auditable identity exposed by one concrete live variant adapter."""

    schema_version: Literal["evidence_live_adapter_identity_v2"]
    variant: Variant
    resource_planning_enabled: bool
    bounded_repair_enabled: bool
    adapter_fingerprint: Sha256Digest
    dataset_id: str
    dataset_fingerprint: Sha256Digest
    knowledge_graph_data_version: str
    knowledge_graph_artifact_fingerprint: Sha256Digest
    declared_cases: list["EvidenceCaseBindingIdentityV2"] = Field(min_length=1)

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, value: str) -> str:
        return _identifier(value, field_name="dataset_id")

    @field_validator("knowledge_graph_data_version")
    @classmethod
    def validate_knowledge_graph_data_version(cls, value: str) -> str:
        return _identifier(value, field_name="knowledge_graph_data_version")

    @model_validator(mode="after")
    def validate_declared_cases(self) -> Self:
        case_ids = tuple(item.case_id for item in self.declared_cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("declared_cases must not repeat case_id")
        return self


class EvidenceRolloutExecutionConfigV2(_StrictFrozenModel):
    """Execution semantics only; benchmark thresholds remain in their source config."""

    schema_version: Literal["evidence_rollout_execution_config_v2"]
    variants: list[EvidenceVariantDefinitionV2] = Field(min_length=4, max_length=4)
    human_semantic_review_required: Literal[True]
    human_review_protocol_fingerprint: Sha256Digest
    candidate_failure_policy: Literal["fail_fast"]
    report_policy: Literal["content_free_v2"]
    max_case_count: PositiveInt

    @model_validator(mode="after")
    def validate_canonical_factorial_semantics(self) -> Self:
        actual = tuple(
            (
                item.variant,
                item.resource_planning_enabled,
                item.bounded_repair_enabled,
            )
            for item in self.variants
        )
        if actual != _CANONICAL_VARIANTS:
            raise ValueError(
                "variants must exactly preserve canonical P0, PG, PR, PGR semantics"
            )
        return self

    def definition_for(self, variant: Variant) -> EvidenceVariantDefinitionV2:
        matches = tuple(item for item in self.variants if item.variant == variant)
        if len(matches) != 1:
            raise ValueError("variant must be configured exactly once")
        return matches[0]


def load_evidence_rollout_execution_config(
    path: Path,
) -> EvidenceRolloutExecutionConfigV2:
    return load_strict_rag_yaml(path, EvidenceRolloutExecutionConfigV2)


class EvidenceResourceSubjectTargetV2(_StrictFrozenModel):
    """Human-authored resource/subject assignment and expected source routes."""

    schema_version: Literal["evidence_resource_subject_target_v2"]
    target_id: str
    subject: str
    resource_type: ResourceType
    topic_id: str
    catalog_resource_ids: list[str] = Field(min_length=1)
    required_sources: list[EvidenceSource] = Field(min_length=1, max_length=2)
    target_fingerprint: Sha256Digest

    @field_validator("target_id", "topic_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", None)
        if not isinstance(field_name, str):
            raise ValueError("target identifier field is unavailable")
        return _identifier(value, field_name=field_name)

    @field_validator("subject")
    @classmethod
    def validate_subject(cls, value: str) -> str:
        return _normalized_subject(value)

    @field_validator("required_sources")
    @classmethod
    def validate_required_sources(
        cls,
        values: list[EvidenceSource],
    ) -> list[EvidenceSource]:
        if len(values) != len(set(values)):
            raise ValueError("required_sources must not contain duplicates")
        return values

    @field_validator("catalog_resource_ids")
    @classmethod
    def validate_catalog_resource_ids(cls, values: list[str]) -> list[str]:
        for value in values:
            _identifier(value, field_name="catalog_resource_id")
        if len(values) != len(set(values)):
            raise ValueError("catalog_resource_ids must not contain duplicates")
        return values

    @model_validator(mode="after")
    def validate_target_fingerprint(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"target_fingerprint"})
        if self.target_fingerprint != canonical_sha256(payload):
            raise ValueError("target_fingerprint does not match target content")
        return self


class EvidenceInitialEvidenceIdentityV2(_StrictFrozenModel):
    """Scenario identity only; it does not prove evidence content or sufficiency."""

    schema_version: Literal["evidence_initial_evidence_identity_v2"]
    state: Literal["sufficient", "insufficient"]
    fixture_id: str
    fixture_fingerprint: Sha256Digest
    source_inventory: list[EvidenceSource] = Field(max_length=2)

    @field_validator("fixture_id")
    @classmethod
    def validate_fixture_id(cls, value: str) -> str:
        return _identifier(value, field_name="fixture_id")

    @field_validator("source_inventory")
    @classmethod
    def validate_source_inventory(
        cls,
        values: list[EvidenceSource],
    ) -> list[EvidenceSource]:
        if len(values) != len(set(values)):
            raise ValueError("source_inventory must not contain duplicates")
        return values

    @model_validator(mode="after")
    def validate_sufficient_inventory(self) -> Self:
        if self.state == "sufficient" and not self.source_inventory:
            raise ValueError("sufficient initial evidence requires a source inventory")
        payload = self.model_dump(mode="json", exclude={"fixture_fingerprint"})
        if self.fixture_fingerprint != canonical_sha256(payload):
            raise ValueError(
                "fixture_fingerprint does not match initial-evidence identity"
            )
        return self


class EvidenceRequirementGoldV2(_StrictFrozenModel):
    """One human-authored criterion used to score evidence coverage."""

    schema_version: Literal["evidence_requirement_gold_v2"]
    requirement_id: str
    target_id: str
    criterion: str = Field(min_length=1, max_length=2_000)
    weight: PositiveFloat

    @field_validator("requirement_id", "target_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", None)
        if not isinstance(field_name, str):
            raise ValueError("requirement identifier field is unavailable")
        return _identifier(value, field_name=field_name)

    @field_validator("criterion")
    @classmethod
    def validate_criterion(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("criterion must already be stripped")
        return value


class EvidenceEvaluationCaseSpecV2(_StrictFrozenModel):
    """One local content-bearing request; query text never enters public reports."""

    schema_version: Literal["evidence_evaluation_case_spec_v2"]
    case_id: str
    query: str = Field(min_length=1, max_length=8_000)
    subjects: list[str] = Field(min_length=1, max_length=20)
    resource_types: list[ResourceType] = Field(min_length=1, max_length=7)
    initial_evidence: EvidenceInitialEvidenceIdentityV2
    targets: list[EvidenceResourceSubjectTargetV2] = Field(min_length=1)
    requirements: list[EvidenceRequirementGoldV2] = Field(min_length=1)
    case_fingerprint: Sha256Digest

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        return _identifier(value, field_name="case_id")

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("query must already be stripped")
        return value

    @field_validator("subjects")
    @classmethod
    def validate_subjects(cls, values: list[str]) -> list[str]:
        normalized = [_normalized_subject(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("subjects must not contain duplicates")
        return values

    @field_validator("resource_types")
    @classmethod
    def validate_resource_types(cls, values: list[ResourceType]) -> list[ResourceType]:
        if len(values) != len(set(values)):
            raise ValueError("resource_types must not contain duplicates")
        return values

    @model_validator(mode="after")
    def validate_gold_inventory(self) -> Self:
        target_ids = tuple(target.target_id for target in self.targets)
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("targets must not repeat target_id")
        target_pairs = tuple(
            (target.subject, target.resource_type) for target in self.targets
        )
        if len(target_pairs) != len(set(target_pairs)):
            raise ValueError("targets must not repeat subject/resource pairs")
        if {target.subject for target in self.targets} != set(self.subjects):
            raise ValueError("targets must cover exactly the declared subjects")
        if {target.resource_type for target in self.targets} != set(
            self.resource_types
        ):
            raise ValueError("targets must cover exactly the declared resource types")

        requirement_ids = tuple(
            requirement.requirement_id for requirement in self.requirements
        )
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("requirements must not repeat requirement_id")
        known_targets = set(target_ids)
        referenced_targets = {
            requirement.target_id for requirement in self.requirements
        }
        if not referenced_targets.issubset(known_targets):
            raise ValueError("requirements must reference declared targets")
        if referenced_targets != known_targets:
            raise ValueError("every target must have at least one gold requirement")
        payload = self.model_dump(mode="json", exclude={"case_fingerprint"})
        if self.case_fingerprint != canonical_sha256(payload):
            raise ValueError("case_fingerprint does not match case content")
        return self


class EvidenceEvaluationDatasetContentV2(_StrictFrozenModel):
    schema_version: Literal["evidence_evaluation_dataset_v2"]
    dataset_id: str
    knowledge_graph_data_version: str
    knowledge_graph_artifact_fingerprint: Sha256Digest
    cases: list[EvidenceEvaluationCaseSpecV2] = Field(min_length=1)

    @field_validator("dataset_id", "knowledge_graph_data_version")
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", None)
        if not isinstance(field_name, str):
            raise ValueError("dataset identifier field is unavailable")
        return _identifier(value, field_name=field_name)

    @model_validator(mode="after")
    def validate_case_inventory(self) -> Self:
        case_ids = tuple(item.case_id for item in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("cases must not repeat case_id")
        has_simple = any(
            len(item.subjects) == 1 and len(item.resource_types) == 1
            for item in self.cases
        )
        has_multi = any(
            len(item.subjects) > 1 or len(item.resource_types) > 1
            for item in self.cases
        )
        has_initially_sufficient = any(
            item.initial_evidence.state == "sufficient" for item in self.cases
        )
        if not has_simple or not has_multi or not has_initially_sufficient:
            raise ValueError(
                "dataset must include simple, multi, and initially sufficient cases"
            )
        return self


class EvidenceEvaluationDatasetV2(EvidenceEvaluationDatasetContentV2):
    dataset_fingerprint: Sha256Digest

    @model_validator(mode="after")
    def validate_dataset_fingerprint(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"dataset_fingerprint"})
        if self.dataset_fingerprint != canonical_sha256(payload):
            raise ValueError("dataset_fingerprint does not match dataset content")
        return self


def seal_evidence_evaluation_dataset(
    content: EvidenceEvaluationDatasetContentV2,
) -> EvidenceEvaluationDatasetV2:
    """Seal one validated authoring document with its canonical fingerprint."""

    if not isinstance(content, EvidenceEvaluationDatasetContentV2):
        raise TypeError("content must be EvidenceEvaluationDatasetContentV2")
    payload = content.model_dump(mode="json")
    return EvidenceEvaluationDatasetV2(
        **content.model_dump(mode="python"),
        dataset_fingerprint=canonical_sha256(payload),
    )


class EvidenceTargetBindingIdentityV2(_StrictFrozenModel):
    schema_version: Literal["evidence_target_binding_identity_v2"]
    target_id: str
    target_fingerprint: Sha256Digest
    topic_id: str
    catalog_resource_ids: list[str] = Field(min_length=1)

    @field_validator("target_id", "topic_id", "catalog_resource_ids")
    @classmethod
    def validate_identifiers(cls, value: object, info: object) -> object:
        field_name = getattr(info, "field_name", None)
        if field_name == "catalog_resource_ids":
            if not isinstance(value, list):
                raise TypeError("catalog_resource_ids must be a list")
            for item in value:
                _identifier(item, field_name="catalog_resource_id")
            if len(value) != len(set(value)):
                raise ValueError("catalog_resource_ids must not contain duplicates")
            return value
        if not isinstance(value, str) or not isinstance(field_name, str):
            raise ValueError("target binding identifier field is unavailable")
        return _identifier(value, field_name=field_name)


class EvidenceCaseBindingIdentityV2(_StrictFrozenModel):
    schema_version: Literal["evidence_case_binding_identity_v2"]
    case_id: str
    case_fingerprint: Sha256Digest
    initial_evidence: EvidenceInitialEvidenceIdentityV2
    targets: list[EvidenceTargetBindingIdentityV2] = Field(min_length=1)

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        return _identifier(value, field_name="case_id")

    @model_validator(mode="after")
    def validate_target_inventory(self) -> Self:
        target_ids = tuple(item.target_id for item in self.targets)
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("case binding targets must not repeat target_id")
        return self


def case_binding_identity(
    case: EvidenceEvaluationCaseSpecV2,
) -> EvidenceCaseBindingIdentityV2:
    """Project one validated case into the shared execution identity contract."""

    if not isinstance(case, EvidenceEvaluationCaseSpecV2):
        raise TypeError("case must be EvidenceEvaluationCaseSpecV2")
    return EvidenceCaseBindingIdentityV2(
        schema_version="evidence_case_binding_identity_v2",
        case_id=case.case_id,
        case_fingerprint=case.case_fingerprint,
        initial_evidence=case.initial_evidence,
        targets=[
            EvidenceTargetBindingIdentityV2(
                schema_version="evidence_target_binding_identity_v2",
                target_id=target.target_id,
                target_fingerprint=target.target_fingerprint,
                topic_id=target.topic_id,
                catalog_resource_ids=list(target.catalog_resource_ids),
            )
            for target in case.targets
        ],
    )


def dataset_case_bindings(
    dataset: EvidenceEvaluationDatasetV2,
) -> list[EvidenceCaseBindingIdentityV2]:
    """Project the exact ordered case/target inventory of a sealed dataset."""

    if not isinstance(dataset, EvidenceEvaluationDatasetV2):
        raise TypeError("dataset must be EvidenceEvaluationDatasetV2")
    return [case_binding_identity(case) for case in dataset.cases]


def case_binding_inventory_fingerprint(
    bindings: list[EvidenceCaseBindingIdentityV2],
) -> str:
    """Fingerprint a validated ordered binding inventory without normalization."""

    if not isinstance(bindings, list) or not bindings:
        raise TypeError("bindings must be a non-empty list")
    if any(not isinstance(item, EvidenceCaseBindingIdentityV2) for item in bindings):
        raise TypeError("bindings must contain EvidenceCaseBindingIdentityV2 values")
    return canonical_sha256(
        {
            "schema_version": "evidence_case_binding_inventory_v2",
            "cases": [item.model_dump(mode="json") for item in bindings],
        }
    )


class EvidenceEvaluationRuntimeBindingV2(_StrictFrozenModel):
    schema_version: Literal["evidence_evaluation_runtime_binding_v2"]
    run_id: str
    execution_mode: ExecutionMode
    dataset_id: str
    dataset_fingerprint: Sha256Digest
    knowledge_graph_data_version: str
    knowledge_graph_artifact_fingerprint: Sha256Digest
    case_bindings: list[EvidenceCaseBindingIdentityV2] = Field(min_length=1)
    execution_config_fingerprint: Sha256Digest
    benchmark_config_fingerprint: Sha256Digest
    rollout_config_fingerprint: Sha256Digest
    runtime_fingerprint: Sha256Digest
    generation_id: str
    generation_manifest_fingerprint: Sha256Digest
    executor_fingerprint: Sha256Digest

    @field_validator(
        "run_id",
        "dataset_id",
        "knowledge_graph_data_version",
        "generation_id",
    )
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", None)
        if not isinstance(field_name, str):
            raise ValueError("binding identifier field is unavailable")
        return _identifier(value, field_name=field_name)

    @model_validator(mode="after")
    def validate_case_bindings(self) -> Self:
        case_ids = tuple(item.case_id for item in self.case_bindings)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case_bindings must not repeat case_id")
        return self


class EvidenceVariantObservationV2(_StrictFrozenModel):
    """Content-free measurement emitted by one real or hermetic variant run."""

    schema_version: Literal["evidence_variant_observation_v2"]
    case_id: str
    variant: Variant
    query_fingerprint: Sha256Digest
    dataset_fingerprint: Sha256Digest
    knowledge_graph_data_version: str
    knowledge_graph_artifact_fingerprint: Sha256Digest
    case_binding: EvidenceCaseBindingIdentityV2
    execution_config_fingerprint: Sha256Digest
    benchmark_config_fingerprint: Sha256Digest
    rollout_config_fingerprint: Sha256Digest
    runtime_fingerprint: Sha256Digest
    generation_id: str
    generation_manifest_fingerprint: Sha256Digest
    executor_fingerprint: Sha256Digest
    variant_definition_fingerprint: Sha256Digest
    output_fingerprint: Sha256Digest
    provider_status: ComponentStatus
    parent_child_status: ComponentStatus
    web_status: ComponentStatus
    bounded: bool
    forced_stop_marked_sufficient: bool
    silent_resource_omission: bool
    silent_subject_omission: bool
    repeated_query_count: NonNegativeInt
    weighted_covered: float = Field(ge=0.0, allow_inf_nan=False)
    weighted_total: PositiveFloat
    required_gap_count: NonNegativeInt
    selected_evidence_count: NonNegativeInt
    correct_evidence_count: NonNegativeInt
    premature_stop: bool
    over_search: bool
    source_route_true_positive: NonNegativeInt
    source_route_false_positive: NonNegativeInt
    source_route_false_negative: NonNegativeInt
    expected_resource_subject_count: PositiveInt
    assigned_resource_subject_count: NonNegativeInt
    correct_resource_subject_count: NonNegativeInt
    retrieval_cost_units: PositiveFloat
    latency_ms: PositiveFloat

    @field_validator("case_id", "knowledge_graph_data_version", "generation_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", None)
        if not isinstance(field_name, str):
            raise ValueError("observation identifier field is unavailable")
        return _identifier(value, field_name=field_name)

    @model_validator(mode="after")
    def validate_measurement_counts(self) -> Self:
        if self.weighted_covered > self.weighted_total:
            raise ValueError("weighted_covered must not exceed weighted_total")
        if self.correct_evidence_count > self.selected_evidence_count:
            raise ValueError("correct_evidence_count must not exceed selected evidence")
        if self.correct_resource_subject_count > min(
            self.expected_resource_subject_count,
            self.assigned_resource_subject_count,
        ):
            raise ValueError(
                "correct resource-subject count exceeds expected or assigned count"
            )
        if self.source_route_true_positive + self.source_route_false_negative <= 0:
            raise ValueError("source routing requires at least one expected route")
        for value in (
            self.weighted_covered,
            self.weighted_total,
            self.retrieval_cost_units,
            self.latency_ms,
        ):
            if not math.isfinite(float(value)):
                raise ValueError("observation metrics must be finite")
        return self


class EvidenceVariantAttemptV2(_StrictFrozenModel):
    schema_version: Literal["evidence_variant_attempt_v2"]
    case_id: str
    variant: Variant
    status: Literal["success", "failed", "blocked"]
    observation: EvidenceVariantObservationV2 | None
    failure_reason_code: str | None
    failure_type: str | None

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        return _identifier(value, field_name="case_id")

    @field_validator("failure_reason_code", "failure_type")
    @classmethod
    def validate_optional_safe_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _identifier(value, field_name="failure metadata")

    @model_validator(mode="after")
    def validate_attempt_shape(self) -> Self:
        if self.status == "success":
            if (
                self.observation is None
                or self.failure_reason_code is not None
                or self.failure_type is not None
            ):
                raise ValueError("successful attempt requires only an observation")
            if (
                self.observation.case_id != self.case_id
                or self.observation.variant != self.variant
            ):
                raise ValueError("attempt identity must match its observation")
        elif (
            self.observation is not None
            or self.failure_reason_code is None
            or self.failure_type is None
        ):
            raise ValueError(
                "failed/blocked attempt requires only safe failure metadata"
            )
        return self


class EvidenceVariantAttemptBatchContentV2(_StrictFrozenModel):
    schema_version: Literal["evidence_variant_attempt_batch_v2"]
    execution_mode: Literal["hermetic"]
    executor_fingerprint: Sha256Digest
    attempts: list[EvidenceVariantAttemptV2] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_slots(self) -> Self:
        slots = tuple((item.case_id, item.variant) for item in self.attempts)
        if len(slots) != len(set(slots)):
            raise ValueError("attempt batch must not repeat case/variant slots")
        return self


class EvidenceVariantAttemptBatchV2(EvidenceVariantAttemptBatchContentV2):
    bundle_fingerprint: Sha256Digest

    @model_validator(mode="after")
    def validate_bundle_fingerprint(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"bundle_fingerprint"})
        if self.bundle_fingerprint != canonical_sha256(payload):
            raise ValueError("bundle_fingerprint does not match attempt batch")
        return self


def seal_evidence_variant_attempt_batch(
    content: EvidenceVariantAttemptBatchContentV2,
) -> EvidenceVariantAttemptBatchV2:
    """Seal one validated hermetic attempt batch."""

    if not isinstance(content, EvidenceVariantAttemptBatchContentV2):
        raise TypeError("content must be EvidenceVariantAttemptBatchContentV2")
    payload = content.model_dump(mode="json")
    return EvidenceVariantAttemptBatchV2(
        **content.model_dump(mode="python"),
        bundle_fingerprint=canonical_sha256(payload),
    )


class HumanSemanticReviewV2(_StrictFrozenModel):
    schema_version: Literal["human_semantic_review_v2"]
    case_id: str
    variant: Variant
    output_fingerprint: Sha256Digest
    reviewer_identity_hash: Sha256Digest
    reviewed_at: str = Field(min_length=1, max_length=64)
    assessment_source: Literal["human"]
    supported_claim_count: NonNegativeInt
    claim_count: PositiveInt
    ungrounded_fact_count: NonNegativeInt
    fact_count: PositiveInt

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        return _identifier(value, field_name="case_id")

    @field_validator("reviewed_at")
    @classmethod
    def validate_reviewed_at(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            raise ValueError("reviewed_at must be canonical ISO 8601") from None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("reviewed_at must be timezone-aware")
        if parsed.isoformat() != value:
            raise ValueError("reviewed_at must be canonical ISO 8601")
        return value

    @model_validator(mode="after")
    def validate_review_counts(self) -> Self:
        if self.supported_claim_count > self.claim_count:
            raise ValueError("supported_claim_count must not exceed claim_count")
        if self.ungrounded_fact_count > self.fact_count:
            raise ValueError("ungrounded_fact_count must not exceed fact_count")
        return self


class HumanSemanticReviewBatchContentV2(_StrictFrozenModel):
    schema_version: Literal["human_semantic_review_batch_v2"]
    dataset_fingerprint: Sha256Digest
    runtime_fingerprint: Sha256Digest
    generation_id: str
    generation_manifest_fingerprint: Sha256Digest
    review_protocol_fingerprint: Sha256Digest
    reviews: list[HumanSemanticReviewV2]

    @field_validator("generation_id")
    @classmethod
    def validate_generation_id(cls, value: str) -> str:
        return _identifier(value, field_name="generation_id")

    @model_validator(mode="after")
    def validate_unique_review_slots(self) -> Self:
        slots = tuple((item.case_id, item.variant) for item in self.reviews)
        if len(slots) != len(set(slots)):
            raise ValueError("reviews must not repeat case/variant slots")
        return self


class HumanSemanticReviewBatchV2(HumanSemanticReviewBatchContentV2):
    review_bundle_fingerprint: Sha256Digest

    @model_validator(mode="after")
    def validate_review_bundle_fingerprint(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"review_bundle_fingerprint"})
        if self.review_bundle_fingerprint != canonical_sha256(payload):
            raise ValueError("review_bundle_fingerprint does not match review batch")
        return self


def seal_human_semantic_review_batch(
    content: HumanSemanticReviewBatchContentV2,
) -> HumanSemanticReviewBatchV2:
    """Seal one complete, validated human semantic-review bundle."""

    if not isinstance(content, HumanSemanticReviewBatchContentV2):
        raise TypeError("content must be HumanSemanticReviewBatchContentV2")
    payload = content.model_dump(mode="json")
    return HumanSemanticReviewBatchV2(
        **content.model_dump(mode="python"),
        review_bundle_fingerprint=canonical_sha256(payload),
    )


class EvidenceExecutionRecordV2(_StrictFrozenModel):
    """Safe record: no query, URL, evidence body, or provider body fields exist."""

    schema_version: Literal["evidence_execution_record_v2"]
    case_id: str
    variant: Variant
    status: Literal["success", "failed", "blocked", "not_executed"]
    output_fingerprint: Sha256Digest | None
    failure_reason_code: str | None
    failure_type: str | None

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        return _identifier(value, field_name="case_id")

    @field_validator("failure_reason_code", "failure_type")
    @classmethod
    def validate_optional_safe_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _identifier(value, field_name="failure metadata")

    @model_validator(mode="after")
    def validate_record_shape(self) -> Self:
        if self.status == "success":
            if self.output_fingerprint is None or any(
                value is not None
                for value in (self.failure_reason_code, self.failure_type)
            ):
                raise ValueError("successful record requires only output_fingerprint")
        elif self.output_fingerprint is not None or any(
            value is None for value in (self.failure_reason_code, self.failure_type)
        ):
            raise ValueError("non-success record requires only safe failure metadata")
        return self


class EvidenceRolloutDecisionContentV2(_StrictFrozenModel):
    schema_version: Literal["evidence_rollout_activation_decision_v2"]
    run_id: str
    execution_mode: ExecutionMode
    status: DecisionStatus
    benchmark_eligible: bool
    activation_allowed: bool
    rollout_activation_enabled: bool
    reason_codes: list[str]
    expected_execution_count: PositiveInt
    successful_execution_count: NonNegativeInt
    reviewed_execution_count: NonNegativeInt
    variant_matrix_complete: bool
    dataset_id: str
    dataset_fingerprint: Sha256Digest
    knowledge_graph_data_version: str
    knowledge_graph_artifact_fingerprint: Sha256Digest
    case_binding_inventory_fingerprint: Sha256Digest
    execution_config_fingerprint: Sha256Digest
    benchmark_config_fingerprint: Sha256Digest
    rollout_config_fingerprint: Sha256Digest
    runtime_fingerprint: Sha256Digest
    generation_id: str
    generation_manifest_fingerprint: Sha256Digest
    executor_fingerprint: Sha256Digest
    review_protocol_fingerprint: Sha256Digest
    review_bundle_fingerprint: Sha256Digest
    activation_decision: EvidenceActivationDecision | None
    case_results: list[EvidenceEvaluationCaseResult]
    execution_records: list[EvidenceExecutionRecordV2]

    @field_validator(
        "run_id",
        "dataset_id",
        "knowledge_graph_data_version",
        "generation_id",
    )
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", None)
        if not isinstance(field_name, str):
            raise ValueError("decision identifier field is unavailable")
        return _identifier(value, field_name=field_name)

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("reason_codes must not contain duplicates")
        for value in values:
            _identifier(value, field_name="reason_code")
        return values

    @model_validator(mode="after")
    def validate_activation_truth(self) -> Self:
        if len(self.execution_records) != self.expected_execution_count:
            raise ValueError("execution records must cover every expected slot")
        record_slots = tuple(
            (record.case_id, record.variant) for record in self.execution_records
        )
        if len(record_slots) != len(set(record_slots)):
            raise ValueError("execution records must not repeat case/variant slots")
        recorded_success_count = sum(
            record.status == "success" for record in self.execution_records
        )
        if recorded_success_count != self.successful_execution_count:
            raise ValueError("successful execution count must match records")
        result_slots = tuple(
            (result.case_id, result.variant) for result in self.case_results
        )
        if len(result_slots) != len(set(result_slots)):
            raise ValueError("case results must not repeat case/variant slots")
        if len(result_slots) != self.reviewed_execution_count:
            raise ValueError("reviewed execution count must match case results")
        successful_slots = {
            (record.case_id, record.variant)
            for record in self.execution_records
            if record.status == "success"
        }
        if not set(result_slots).issubset(successful_slots):
            raise ValueError("case results must belong to successful executions")
        counts_complete = (
            self.successful_execution_count == self.expected_execution_count
            and self.reviewed_execution_count == self.expected_execution_count
            and len(self.case_results) == self.expected_execution_count
        )
        if self.variant_matrix_complete != counts_complete:
            raise ValueError("variant_matrix_complete does not match execution counts")
        if self.activation_allowed != (self.status == "pass"):
            raise ValueError("activation_allowed must be true only for pass")
        if self.execution_mode == "hermetic":
            if (
                self.status != "blocked"
                or "non_live_execution" not in self.reason_codes
            ):
                raise ValueError("hermetic decisions must be explicitly blocked")
        if (
            self.status == "blocked"
            and not self.rollout_activation_enabled
            and "rollout_activation_disabled" not in self.reason_codes
        ):
            raise ValueError("disabled rollout must be explicit in blocked decisions")
        if self.status == "pass":
            if (
                self.execution_mode != "live"
                or not self.rollout_activation_enabled
                or not self.benchmark_eligible
                or not self.variant_matrix_complete
                or self.reason_codes
                or self.activation_decision is None
                or not self.activation_decision.eligible
            ):
                raise ValueError(
                    "pass requires complete live benchmark and rollout gate"
                )
        elif not self.reason_codes:
            raise ValueError("fail/blocked decision requires reason_codes")
        if self.status == "fail":
            if (
                self.execution_mode != "live"
                or not self.variant_matrix_complete
                or self.activation_decision is None
                or self.activation_decision.eligible
                or self.benchmark_eligible
            ):
                raise ValueError("fail requires a complete live threshold failure")
        if self.benchmark_eligible != (
            self.activation_decision is not None and self.activation_decision.eligible
        ):
            raise ValueError("benchmark_eligible must match activation_decision")
        return self


class EvidenceRolloutDecisionV2(EvidenceRolloutDecisionContentV2):
    decision_fingerprint: Sha256Digest

    @model_validator(mode="after")
    def validate_decision_fingerprint(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"decision_fingerprint"})
        if self.decision_fingerprint != canonical_sha256(payload):
            raise ValueError("decision_fingerprint does not match decision content")
        return self


__all__ = [
    "DecisionStatus",
    "EvidenceCaseBindingIdentityV2",
    "EvidenceEvaluationCaseSpecV2",
    "EvidenceEvaluationDatasetContentV2",
    "EvidenceEvaluationDatasetV2",
    "EvidenceEvaluationRuntimeBindingV2",
    "EvidenceInitialEvidenceIdentityV2",
    "EvidenceRequirementGoldV2",
    "EvidenceResourceSubjectTargetV2",
    "EvidenceTargetBindingIdentityV2",
    "EvidenceExecutionRecordV2",
    "EvidenceLiveAdapterIdentityV2",
    "EvidenceRolloutDecisionContentV2",
    "EvidenceRolloutDecisionV2",
    "EvidenceRolloutExecutionConfigV2",
    "EvidenceSource",
    "EvidenceVariantAttemptBatchContentV2",
    "EvidenceVariantAttemptBatchV2",
    "EvidenceVariantAttemptV2",
    "EvidenceVariantDefinitionV2",
    "EvidenceVariantObservationV2",
    "ExecutionMode",
    "HumanSemanticReviewBatchContentV2",
    "HumanSemanticReviewBatchV2",
    "HumanSemanticReviewV2",
    "case_binding_identity",
    "case_binding_inventory_fingerprint",
    "canonical_sha256",
    "dataset_case_bindings",
    "load_evidence_rollout_execution_config",
    "model_fingerprint",
    "query_fingerprint",
    "seal_evidence_evaluation_dataset",
    "seal_evidence_variant_attempt_batch",
    "seal_human_semantic_review_batch",
]
