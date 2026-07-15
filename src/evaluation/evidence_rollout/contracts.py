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
        {"algorithm": "evidence_evaluation_query_v1", "query": query}
    )


class EvidenceVariantDefinitionV1(_StrictFrozenModel):
    variant: Variant
    resource_planning_enabled: bool
    bounded_repair_enabled: bool


class EvidenceLiveAdapterIdentityV1(_StrictFrozenModel):
    """Auditable identity exposed by one concrete live variant adapter."""

    schema_version: Literal["evidence_live_adapter_identity_v1"]
    variant: Variant
    resource_planning_enabled: bool
    bounded_repair_enabled: bool
    adapter_fingerprint: Sha256Digest
    declared_case_ids: list[str] = Field(min_length=1)

    @field_validator("declared_case_ids")
    @classmethod
    def validate_case_ids(cls, values: list[str]) -> list[str]:
        for value in values:
            _identifier(value, field_name="declared_case_id")
        if len(values) != len(set(values)):
            raise ValueError("declared_case_ids must not contain duplicates")
        return values


class EvidenceRolloutExecutionConfigV1(_StrictFrozenModel):
    """Execution semantics only; benchmark thresholds remain in their source config."""

    schema_version: Literal["evidence_rollout_execution_config_v1"]
    variants: list[EvidenceVariantDefinitionV1] = Field(min_length=4, max_length=4)
    human_semantic_review_required: Literal[True]
    human_review_protocol_fingerprint: Sha256Digest
    candidate_failure_policy: Literal["fail_fast"]
    report_policy: Literal["content_free_v1"]
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

    def definition_for(self, variant: Variant) -> EvidenceVariantDefinitionV1:
        matches = tuple(item for item in self.variants if item.variant == variant)
        if len(matches) != 1:
            raise ValueError("variant must be configured exactly once")
        return matches[0]


def load_evidence_rollout_execution_config(
    path: Path,
) -> EvidenceRolloutExecutionConfigV1:
    return load_strict_rag_yaml(path, EvidenceRolloutExecutionConfigV1)


class EvidenceResourceSubjectTargetV1(_StrictFrozenModel):
    """Human-authored resource/subject assignment and expected source routes."""

    schema_version: Literal["evidence_resource_subject_target_v1"]
    target_id: str
    subject: str
    resource_type: ResourceType
    required_sources: list[EvidenceSource] = Field(min_length=1, max_length=2)

    @field_validator("target_id")
    @classmethod
    def validate_target_id(cls, value: str) -> str:
        return _identifier(value, field_name="target_id")

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


class EvidenceRequirementGoldV1(_StrictFrozenModel):
    """One human-authored criterion used to score evidence coverage."""

    schema_version: Literal["evidence_requirement_gold_v1"]
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


class EvidenceEvaluationCaseSpecV1(_StrictFrozenModel):
    """One local content-bearing request; query text never enters public reports."""

    schema_version: Literal["evidence_evaluation_case_spec_v1"]
    case_id: str
    query: str = Field(min_length=1, max_length=8_000)
    subjects: list[str] = Field(min_length=1, max_length=20)
    resource_types: list[ResourceType] = Field(min_length=1, max_length=7)
    initial_evidence_sufficient: bool
    targets: list[EvidenceResourceSubjectTargetV1] = Field(min_length=1)
    requirements: list[EvidenceRequirementGoldV1] = Field(min_length=1)

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
        return self


class EvidenceEvaluationDatasetContentV1(_StrictFrozenModel):
    schema_version: Literal["evidence_evaluation_dataset_v1"]
    dataset_id: str
    cases: list[EvidenceEvaluationCaseSpecV1] = Field(min_length=1)

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, value: str) -> str:
        return _identifier(value, field_name="dataset_id")

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
            item.initial_evidence_sufficient for item in self.cases
        )
        if not has_simple or not has_multi or not has_initially_sufficient:
            raise ValueError(
                "dataset must include simple, multi, and initially sufficient cases"
            )
        return self


class EvidenceEvaluationDatasetV1(EvidenceEvaluationDatasetContentV1):
    dataset_fingerprint: Sha256Digest

    @model_validator(mode="after")
    def validate_dataset_fingerprint(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"dataset_fingerprint"})
        if self.dataset_fingerprint != canonical_sha256(payload):
            raise ValueError("dataset_fingerprint does not match dataset content")
        return self


def seal_evidence_evaluation_dataset(
    content: EvidenceEvaluationDatasetContentV1,
) -> EvidenceEvaluationDatasetV1:
    """Seal one validated authoring document with its canonical fingerprint."""

    if not isinstance(content, EvidenceEvaluationDatasetContentV1):
        raise TypeError("content must be EvidenceEvaluationDatasetContentV1")
    payload = content.model_dump(mode="json")
    return EvidenceEvaluationDatasetV1(
        **content.model_dump(mode="python"),
        dataset_fingerprint=canonical_sha256(payload),
    )


class EvidenceEvaluationRuntimeBindingV1(_StrictFrozenModel):
    schema_version: Literal["evidence_evaluation_runtime_binding_v1"]
    run_id: str
    execution_mode: ExecutionMode
    dataset_id: str
    dataset_fingerprint: Sha256Digest
    execution_config_fingerprint: Sha256Digest
    benchmark_config_fingerprint: Sha256Digest
    rollout_config_fingerprint: Sha256Digest
    runtime_fingerprint: Sha256Digest
    generation_id: str
    generation_manifest_fingerprint: Sha256Digest
    executor_fingerprint: Sha256Digest

    @field_validator("run_id", "dataset_id", "generation_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", None)
        if not isinstance(field_name, str):
            raise ValueError("binding identifier field is unavailable")
        return _identifier(value, field_name=field_name)


class EvidenceVariantObservationV1(_StrictFrozenModel):
    """Content-free measurement emitted by one real or hermetic variant run."""

    schema_version: Literal["evidence_variant_observation_v1"]
    case_id: str
    variant: Variant
    query_fingerprint: Sha256Digest
    dataset_fingerprint: Sha256Digest
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

    @field_validator("case_id", "generation_id")
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


class EvidenceVariantAttemptV1(_StrictFrozenModel):
    schema_version: Literal["evidence_variant_attempt_v1"]
    case_id: str
    variant: Variant
    status: Literal["success", "failed", "blocked"]
    observation: EvidenceVariantObservationV1 | None
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


class EvidenceVariantAttemptBatchContentV1(_StrictFrozenModel):
    schema_version: Literal["evidence_variant_attempt_batch_v1"]
    execution_mode: Literal["hermetic"]
    executor_fingerprint: Sha256Digest
    attempts: list[EvidenceVariantAttemptV1] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_slots(self) -> Self:
        slots = tuple((item.case_id, item.variant) for item in self.attempts)
        if len(slots) != len(set(slots)):
            raise ValueError("attempt batch must not repeat case/variant slots")
        return self


class EvidenceVariantAttemptBatchV1(EvidenceVariantAttemptBatchContentV1):
    bundle_fingerprint: Sha256Digest

    @model_validator(mode="after")
    def validate_bundle_fingerprint(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"bundle_fingerprint"})
        if self.bundle_fingerprint != canonical_sha256(payload):
            raise ValueError("bundle_fingerprint does not match attempt batch")
        return self


def seal_evidence_variant_attempt_batch(
    content: EvidenceVariantAttemptBatchContentV1,
) -> EvidenceVariantAttemptBatchV1:
    """Seal one validated hermetic attempt batch."""

    if not isinstance(content, EvidenceVariantAttemptBatchContentV1):
        raise TypeError("content must be EvidenceVariantAttemptBatchContentV1")
    payload = content.model_dump(mode="json")
    return EvidenceVariantAttemptBatchV1(
        **content.model_dump(mode="python"),
        bundle_fingerprint=canonical_sha256(payload),
    )


class HumanSemanticReviewV1(_StrictFrozenModel):
    schema_version: Literal["human_semantic_review_v1"]
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


class HumanSemanticReviewBatchContentV1(_StrictFrozenModel):
    schema_version: Literal["human_semantic_review_batch_v1"]
    dataset_fingerprint: Sha256Digest
    runtime_fingerprint: Sha256Digest
    generation_id: str
    generation_manifest_fingerprint: Sha256Digest
    review_protocol_fingerprint: Sha256Digest
    reviews: list[HumanSemanticReviewV1]

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


class HumanSemanticReviewBatchV1(HumanSemanticReviewBatchContentV1):
    review_bundle_fingerprint: Sha256Digest

    @model_validator(mode="after")
    def validate_review_bundle_fingerprint(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"review_bundle_fingerprint"})
        if self.review_bundle_fingerprint != canonical_sha256(payload):
            raise ValueError("review_bundle_fingerprint does not match review batch")
        return self


def seal_human_semantic_review_batch(
    content: HumanSemanticReviewBatchContentV1,
) -> HumanSemanticReviewBatchV1:
    """Seal one complete, validated human semantic-review bundle."""

    if not isinstance(content, HumanSemanticReviewBatchContentV1):
        raise TypeError("content must be HumanSemanticReviewBatchContentV1")
    payload = content.model_dump(mode="json")
    return HumanSemanticReviewBatchV1(
        **content.model_dump(mode="python"),
        review_bundle_fingerprint=canonical_sha256(payload),
    )


class EvidenceExecutionRecordV1(_StrictFrozenModel):
    """Safe record: no query, URL, evidence body, or provider body fields exist."""

    schema_version: Literal["evidence_execution_record_v1"]
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


class EvidenceRolloutDecisionContentV1(_StrictFrozenModel):
    schema_version: Literal["evidence_rollout_activation_decision_v1"]
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
    execution_records: list[EvidenceExecutionRecordV1]

    @field_validator("run_id", "dataset_id", "generation_id")
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


class EvidenceRolloutDecisionV1(EvidenceRolloutDecisionContentV1):
    decision_fingerprint: Sha256Digest

    @model_validator(mode="after")
    def validate_decision_fingerprint(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"decision_fingerprint"})
        if self.decision_fingerprint != canonical_sha256(payload):
            raise ValueError("decision_fingerprint does not match decision content")
        return self


__all__ = [
    "DecisionStatus",
    "EvidenceEvaluationCaseSpecV1",
    "EvidenceEvaluationDatasetContentV1",
    "EvidenceEvaluationDatasetV1",
    "EvidenceEvaluationRuntimeBindingV1",
    "EvidenceRequirementGoldV1",
    "EvidenceResourceSubjectTargetV1",
    "EvidenceExecutionRecordV1",
    "EvidenceLiveAdapterIdentityV1",
    "EvidenceRolloutDecisionContentV1",
    "EvidenceRolloutDecisionV1",
    "EvidenceRolloutExecutionConfigV1",
    "EvidenceVariantAttemptBatchContentV1",
    "EvidenceVariantAttemptBatchV1",
    "EvidenceVariantAttemptV1",
    "EvidenceVariantDefinitionV1",
    "EvidenceVariantObservationV1",
    "ExecutionMode",
    "HumanSemanticReviewBatchContentV1",
    "HumanSemanticReviewBatchV1",
    "HumanSemanticReviewV1",
    "canonical_sha256",
    "load_evidence_rollout_execution_config",
    "model_fingerprint",
    "query_fingerprint",
    "seal_evidence_evaluation_dataset",
    "seal_evidence_variant_attempt_batch",
    "seal_human_semantic_review_batch",
]
