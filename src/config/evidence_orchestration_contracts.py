"""Strict, provider-neutral contracts for evidence orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import AbstractSet, Annotated, Literal, TypeAlias

from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    field_validator,
    model_validator,
)

from src.config._rag_config import NonBlankStr, StrictRagConfigModel
from src.config.evidence_orchestration_config import (
    CANONICAL_RESOURCE_TYPES,
    EvidenceCriticality,
    EvidenceNeedScope,
    EvidenceOrchestrationConfig,
    EvidenceSourcePolicy,
    ResourceEvidenceNeed,
    ResourceEvidenceProfilesConfig,
    ResourceType,
)


NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
Sha256Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

EvidenceSourceType = Literal["local_rag", "web"]
RetrievalPriority = Literal["high", "medium", "low"]
CoverageState = Literal["complete", "partial", "missing"]
ReadinessState = Literal["ready", "blocked_insufficient_evidence"]
RESOURCE_EVIDENCE_CONTRACT_VERSION = "resource_evidence_assignment_v1"


def _freeze_sequence(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


def _explicit_query(value: str) -> str:
    if value and (not value.strip() or value != value.strip()):
        raise ValueError(
            "query must be empty or contain no leading or trailing whitespace"
        )
    return value


ExplicitQuery = Annotated[str, AfterValidator(_explicit_query)]
NonBlankStrTuple = Annotated[
    tuple[NonBlankStr, ...],
    BeforeValidator(_freeze_sequence),
]


def _canonical_json_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _validate_topic_identifier(value: str) -> str:
    if value != value.casefold():
        raise ValueError("topic_id must already be case-folded")
    if (
        not value
        or not value[0].isalnum()
        or not value[-1].isalnum()
        or any(
            not (character.isascii() and (character.isalnum() or character in "._:-"))
            for character in value
        )
    ):
        raise ValueError("topic_id must be a normalized identifier")
    return value


def _unique_preserving_order(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


class EvidenceOrchestrationContractError(ValueError):
    """Base class for fail-fast orchestration business validation errors."""

    def __init__(self, *, code: str, reason: str) -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"{code}: {reason}")


class EvidenceRequirementValidationError(EvidenceOrchestrationContractError):
    """The requirement inventory does not match the explicit request/profile."""


class RetrievalTaskValidationError(EvidenceOrchestrationContractError):
    """A retrieval task is not legal for its bound requirement."""


class DuplicateRetrievalSignatureError(RetrievalTaskValidationError):
    """A retrieval signature repeats within or across rounds."""


class EvidenceBudgetExceededError(RetrievalTaskValidationError):
    """A retrieval or evidence bound from configuration was exceeded."""


class EvidenceLedgerValidationError(EvidenceOrchestrationContractError):
    """An evidence ledger entry has an invalid binding or identity."""


class RequirementCoverageValidationError(EvidenceOrchestrationContractError):
    """Coverage output is incomplete or references invalid evidence."""


class ResourceEvidenceAssignmentError(EvidenceOrchestrationContractError):
    """Resource readiness or evidence assignment is inconsistent."""


class EvidenceRequirementDraft(StrictRagConfigModel):
    """LLM-produced requirement draft, validated against a configured profile."""

    resource_type: ResourceType
    subject: NonBlankStr
    topic_id: NonBlankStr
    profile_need_id: NonBlankStr
    evidence_kind: NonBlankStr
    scope: EvidenceNeedScope
    criticality: EvidenceCriticality
    source_policy: EvidenceSourcePolicy
    acceptance_criteria: NonBlankStr
    query_intent: NonBlankStr

    @field_validator("subject", "profile_need_id", "evidence_kind")
    @classmethod
    def _validate_identifiers(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "identifier")
        return _validate_normalized_identifier(value, field_name)

    @field_validator("topic_id")
    @classmethod
    def _validate_topic_id(cls, value: str) -> str:
        return _validate_topic_identifier(value)


class EvidenceRequirementDraftBatch(StrictRagConfigModel):
    """Versioned structured-output wrapper for requirement drafts."""

    schema_version: Literal["evidence_requirement_draft_batch_v1"]
    requirements: list[EvidenceRequirementDraft] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unique_profile_slots(self) -> "EvidenceRequirementDraftBatch":
        keys = tuple(
            (
                item.resource_type,
                item.subject,
                item.topic_id,
                item.profile_need_id,
            )
            for item in self.requirements
        )
        if len(keys) != len(set(keys)):
            raise ValueError(
                "requirement drafts must not repeat a resource/profile slot"
            )
        return self


def make_evidence_requirement_id(draft: EvidenceRequirementDraft) -> str:
    """Create a deterministic identifier from the complete validated draft."""
    digest = _canonical_json_digest(
        {
            "algorithm": "evidence_requirement_id_v1",
            "resource_type": draft.resource_type,
            "subject": draft.subject,
            "topic_id": draft.topic_id,
            "profile_need_id": draft.profile_need_id,
            "evidence_kind": draft.evidence_kind,
            "scope": draft.scope,
            "criticality": draft.criticality,
            "source_policy": draft.source_policy,
            "acceptance_criteria": draft.acceptance_criteria,
            "query_intent": draft.query_intent,
        }
    )
    return f"requirement_{digest}"


class EvidenceRequirement(StrictRagConfigModel):
    """Compiled evidence requirement with a verified deterministic identity."""

    requirement_id: NonBlankStr
    resource_type: ResourceType
    subject: NonBlankStr
    topic_id: NonBlankStr
    profile_need_id: NonBlankStr
    evidence_kind: NonBlankStr
    scope: EvidenceNeedScope
    criticality: EvidenceCriticality
    source_policy: EvidenceSourcePolicy
    acceptance_criteria: NonBlankStr
    query_intent: NonBlankStr

    @field_validator("subject", "profile_need_id", "evidence_kind")
    @classmethod
    def _validate_identifiers(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "identifier")
        return _validate_normalized_identifier(value, field_name)

    @field_validator("topic_id")
    @classmethod
    def _validate_topic_id(cls, value: str) -> str:
        return _validate_topic_identifier(value)

    @model_validator(mode="after")
    def _validate_requirement_id(self) -> "EvidenceRequirement":
        draft = EvidenceRequirementDraft(
            resource_type=self.resource_type,
            subject=self.subject,
            topic_id=self.topic_id,
            profile_need_id=self.profile_need_id,
            evidence_kind=self.evidence_kind,
            scope=self.scope,
            criticality=self.criticality,
            source_policy=self.source_policy,
            acceptance_criteria=self.acceptance_criteria,
            query_intent=self.query_intent,
        )
        if self.requirement_id != make_evidence_requirement_id(draft):
            raise ValueError("requirement_id does not match the deterministic payload")
        return self


def compile_evidence_requirement(
    draft: EvidenceRequirementDraft,
) -> EvidenceRequirement:
    """Compile one validated draft without unchecked construction."""
    return EvidenceRequirement(
        requirement_id=make_evidence_requirement_id(draft),
        resource_type=draft.resource_type,
        subject=draft.subject,
        topic_id=draft.topic_id,
        profile_need_id=draft.profile_need_id,
        evidence_kind=draft.evidence_kind,
        scope=draft.scope,
        criticality=draft.criticality,
        source_policy=draft.source_policy,
        acceptance_criteria=draft.acceptance_criteria,
        query_intent=draft.query_intent,
    )


def compile_evidence_requirement_batch(
    batch: EvidenceRequirementDraftBatch,
) -> tuple[EvidenceRequirement, ...]:
    """Compile every draft in stable batch order."""
    return tuple(compile_evidence_requirement(draft) for draft in batch.requirements)


def make_query_fingerprint(query: str) -> str:
    """Fingerprint a non-blank exact query without normalizing its content."""
    if not query or not query.strip() or query != query.strip():
        raise ValueError("query must be non-blank without surrounding whitespace")
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def make_retrieval_signature(
    *,
    requirement_id: str,
    source_type: EvidenceSourceType,
    subject: str,
    query_fingerprint: str,
) -> str:
    """Identify an exact requirement/source/query attempt across all rounds."""
    return _canonical_json_digest(
        {
            "algorithm": "retrieval_signature_v1",
            "requirement_id": requirement_id,
            "source_type": source_type,
            "subject": subject,
            "query_fingerprint": query_fingerprint,
        }
    )


def make_retrieval_task_id(*, retrieval_signature: str, round_index: int) -> str:
    """Create a deterministic round-specific task identifier."""
    digest = _canonical_json_digest(
        {
            "algorithm": "retrieval_task_id_v1",
            "retrieval_signature": retrieval_signature,
            "round_index": round_index,
        }
    )
    return f"retrieval_task_{digest}"


class RetrievalTask(StrictRagConfigModel):
    """One bounded retrieval action bound to an existing requirement."""

    task_id: NonBlankStr
    requirement_id: NonBlankStr
    resource_type: ResourceType
    source_type: EvidenceSourceType
    subject: NonBlankStr
    query: NonBlankStr
    purpose: NonBlankStr
    priority: RetrievalPriority
    round_index: NonNegativeInt
    result_limit: PositiveInt
    query_fingerprint: Sha256Digest
    retrieval_signature: Sha256Digest

    @field_validator("subject")
    @classmethod
    def _validate_subject(cls, value: str) -> str:
        return _validate_normalized_identifier(value, "subject")

    @model_validator(mode="after")
    def _validate_computed_identity(self) -> "RetrievalTask":
        expected_query_fingerprint = make_query_fingerprint(self.query)
        if self.query_fingerprint != expected_query_fingerprint:
            raise ValueError("query_fingerprint does not match query")
        expected_signature = make_retrieval_signature(
            requirement_id=self.requirement_id,
            source_type=self.source_type,
            subject=self.subject,
            query_fingerprint=self.query_fingerprint,
        )
        if self.retrieval_signature != expected_signature:
            raise ValueError("retrieval_signature does not match task fields")
        expected_task_id = make_retrieval_task_id(
            retrieval_signature=self.retrieval_signature,
            round_index=self.round_index,
        )
        if self.task_id != expected_task_id:
            raise ValueError("task_id does not match task fields")
        return self


def build_retrieval_task(
    *,
    requirement: EvidenceRequirement,
    source_type: EvidenceSourceType,
    query: str,
    purpose: str,
    priority: RetrievalPriority,
    round_index: int,
    result_limit: int,
) -> RetrievalTask:
    """Build one retrieval task with verified deterministic identity fields."""
    query_fingerprint = make_query_fingerprint(query)
    signature = make_retrieval_signature(
        requirement_id=requirement.requirement_id,
        source_type=source_type,
        subject=requirement.subject,
        query_fingerprint=query_fingerprint,
    )
    return RetrievalTask(
        task_id=make_retrieval_task_id(
            retrieval_signature=signature,
            round_index=round_index,
        ),
        requirement_id=requirement.requirement_id,
        resource_type=requirement.resource_type,
        source_type=source_type,
        subject=requirement.subject,
        query=query,
        purpose=purpose,
        priority=priority,
        round_index=round_index,
        result_limit=result_limit,
        query_fingerprint=query_fingerprint,
        retrieval_signature=signature,
    )


def make_evidence_id(
    *,
    requirement_id: str,
    source_type: EvidenceSourceType,
    source_identity_fingerprint: str,
    content_fingerprint: str,
) -> str:
    """Create an exact, requirement-bound evidence identity."""
    digest = _canonical_json_digest(
        {
            "algorithm": "evidence_identity_v1",
            "requirement_id": requirement_id,
            "source_type": source_type,
            "source_identity_fingerprint": source_identity_fingerprint,
            "content_fingerprint": content_fingerprint,
        }
    )
    return f"evidence_{digest}"


class EvidenceLedgerEntry(StrictRagConfigModel):
    """Content-free reference to one accepted or rejected retrieval candidate."""

    round_index: NonNegativeInt
    task_id: NonBlankStr
    requirement_id: NonBlankStr
    evidence_id: NonBlankStr
    resource_type: ResourceType
    subject: NonBlankStr
    source_type: EvidenceSourceType
    candidate_ref: NonBlankStr
    candidate_snapshot_fingerprint: Sha256Digest
    source_identity_fingerprint: Sha256Digest
    content_fingerprint: Sha256Digest
    accepted: bool
    rejection_reason_code: str

    @field_validator("subject")
    @classmethod
    def _validate_subject(cls, value: str) -> str:
        return _validate_normalized_identifier(value, "subject")

    @field_validator("rejection_reason_code")
    @classmethod
    def _validate_explicit_reason(cls, value: str) -> str:
        if value and (not value.strip() or value != value.strip()):
            raise ValueError(
                "rejection_reason_code must be empty or a normalized value"
            )
        return value

    @model_validator(mode="after")
    def _validate_entry(self) -> "EvidenceLedgerEntry":
        if self.accepted and self.rejection_reason_code:
            raise ValueError("accepted evidence must use an empty rejection reason")
        if not self.accepted and not self.rejection_reason_code:
            raise ValueError("rejected evidence must include a rejection reason code")
        expected_id = make_evidence_id(
            requirement_id=self.requirement_id,
            source_type=self.source_type,
            source_identity_fingerprint=self.source_identity_fingerprint,
            content_fingerprint=self.content_fingerprint,
        )
        if self.evidence_id != expected_id:
            raise ValueError("evidence_id does not match the exact evidence identity")
        return self


class RequirementCoverage(StrictRagConfigModel):
    """Judge output for exactly one requirement in one retrieval round."""

    requirement_id: NonBlankStr
    resource_type: ResourceType
    subject: NonBlankStr
    round_index: NonNegativeInt
    coverage_state: CoverageState
    evidence_ids: list[NonBlankStr]
    confidence: UnitFloat
    reason: NonBlankStr
    suggested_local_query: ExplicitQuery
    suggested_web_query: ExplicitQuery

    @field_validator("subject")
    @classmethod
    def _validate_subject(cls, value: str) -> str:
        return _validate_normalized_identifier(value, "subject")

    @field_validator("evidence_ids")
    @classmethod
    def _validate_unique_evidence_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("evidence_ids must not contain duplicates")
        return values

    @model_validator(mode="after")
    def _validate_coverage_shape(self) -> "RequirementCoverage":
        _validate_requirement_coverage_shape(
            coverage_state=self.coverage_state,
            evidence_ids=self.evidence_ids,
            suggested_local_query=self.suggested_local_query,
            suggested_web_query=self.suggested_web_query,
        )
        return self


class RequirementCoverageBatch(StrictRagConfigModel):
    """Versioned judge batch containing one row per requirement."""

    schema_version: Literal["requirement_coverage_batch_v1"]
    round_index: NonNegativeInt
    coverages: list[RequirementCoverage] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_batch(self) -> "RequirementCoverageBatch":
        _validate_requirement_coverage_batch(
            round_index=self.round_index,
            coverages=self.coverages,
        )
        return self


def _validate_requirement_coverage_shape(
    *,
    coverage_state: CoverageState,
    evidence_ids: Sequence[str],
    suggested_local_query: str,
    suggested_web_query: str,
) -> None:
    has_local_query = bool(suggested_local_query)
    has_web_query = bool(suggested_web_query)
    if coverage_state == "complete":
        if not evidence_ids:
            raise ValueError("complete coverage must reference evidence")
        if has_local_query or has_web_query:
            raise ValueError(
                "complete coverage must explicitly use empty suggested queries"
            )
    elif coverage_state == "partial":
        if not evidence_ids:
            raise ValueError("partial coverage must reference evidence")
        if not has_local_query and not has_web_query:
            raise ValueError("partial coverage must suggest a gap query")
    else:
        if evidence_ids:
            raise ValueError("missing coverage must not reference evidence")
        if not has_local_query and not has_web_query:
            raise ValueError("missing coverage must suggest a gap query")


def _validate_requirement_coverage_batch(
    *,
    round_index: int,
    coverages: Sequence[RequirementCoverage | "CompiledRequirementCoverage"],
) -> None:
    requirement_ids = tuple(item.requirement_id for item in coverages)
    if len(requirement_ids) != len(set(requirement_ids)):
        raise ValueError("coverage batch must contain each requirement at most once")
    if any(item.round_index != round_index for item in coverages):
        raise ValueError("coverage rows must match the batch round_index")


class CompiledRequirementCoverage(StrictRagConfigModel):
    """Immutable internal projection of one validated Provider coverage row."""

    requirement_id: NonBlankStr
    resource_type: ResourceType
    subject: NonBlankStr
    round_index: NonNegativeInt
    coverage_state: CoverageState
    evidence_ids: NonBlankStrTuple
    confidence: UnitFloat
    reason: NonBlankStr
    suggested_local_query: ExplicitQuery
    suggested_web_query: ExplicitQuery

    @field_validator("subject")
    @classmethod
    def _validate_subject(cls, value: str) -> str:
        return _validate_normalized_identifier(value, "subject")

    @field_validator("evidence_ids")
    @classmethod
    def _validate_unique_evidence_ids(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("evidence_ids must not contain duplicates")
        return values

    @model_validator(mode="after")
    def _validate_coverage_shape(self) -> "CompiledRequirementCoverage":
        _validate_requirement_coverage_shape(
            coverage_state=self.coverage_state,
            evidence_ids=self.evidence_ids,
            suggested_local_query=self.suggested_local_query,
            suggested_web_query=self.suggested_web_query,
        )
        return self


CompiledRequirementCoverageTuple: TypeAlias = Annotated[
    tuple[CompiledRequirementCoverage, ...],
    BeforeValidator(_freeze_sequence),
    Field(min_length=1),
]


class CompiledRequirementCoverageBatch(StrictRagConfigModel):
    """Frozen internal coverage batch persisted in graph state and replayed."""

    schema_version: Literal["requirement_coverage_batch_v1"]
    round_index: NonNegativeInt
    coverages: CompiledRequirementCoverageTuple

    @model_validator(mode="after")
    def _validate_batch(self) -> "CompiledRequirementCoverageBatch":
        _validate_requirement_coverage_batch(
            round_index=self.round_index,
            coverages=self.coverages,
        )
        return self


def compile_requirement_coverage_batch(
    batch: RequirementCoverageBatch,
) -> CompiledRequirementCoverageBatch:
    """Explicitly project JSON-native Provider lists into immutable tuples."""

    if not isinstance(batch, RequirementCoverageBatch):
        raise TypeError("batch must be RequirementCoverageBatch")
    return CompiledRequirementCoverageBatch(
        schema_version="requirement_coverage_batch_v1",
        round_index=batch.round_index,
        coverages=tuple(
            CompiledRequirementCoverage(
                requirement_id=item.requirement_id,
                resource_type=item.resource_type,
                subject=item.subject,
                round_index=item.round_index,
                coverage_state=item.coverage_state,
                evidence_ids=tuple(item.evidence_ids),
                confidence=item.confidence,
                reason=item.reason,
                suggested_local_query=item.suggested_local_query,
                suggested_web_query=item.suggested_web_query,
            )
            for item in batch.coverages
        ),
    )


def make_repair_plan_signature(
    *,
    round_index: int,
    target_requirement_ids: Sequence[str],
    tasks: Sequence[RetrievalTask],
) -> str:
    """Fingerprint the exact ordered repair plan."""
    return _canonical_json_digest(
        {
            "algorithm": "evidence_repair_plan_v1",
            "round_index": round_index,
            "target_requirement_ids": list(target_requirement_ids),
            "retrieval_signatures": [task.retrieval_signature for task in tasks],
        }
    )


class EvidenceRepairPlan(StrictRagConfigModel):
    """A bounded next-round plan containing only targeted repair tasks."""

    round_index: PositiveInt
    target_requirement_ids: NonBlankStrTuple
    tasks: Annotated[
        tuple[RetrievalTask, ...],
        BeforeValidator(_freeze_sequence),
        Field(min_length=1),
    ]
    reason: NonBlankStr
    plan_signature: Sha256Digest

    @model_validator(mode="after")
    def _validate_plan(self) -> "EvidenceRepairPlan":
        if len(self.target_requirement_ids) != len(set(self.target_requirement_ids)):
            raise ValueError("target_requirement_ids must not contain duplicates")
        if any(task.round_index != self.round_index for task in self.tasks):
            raise ValueError("repair tasks must match the repair round_index")
        task_requirement_ids = {task.requirement_id for task in self.tasks}
        if task_requirement_ids != set(self.target_requirement_ids):
            raise ValueError("repair targets must exactly match task requirements")
        expected = make_repair_plan_signature(
            round_index=self.round_index,
            target_requirement_ids=self.target_requirement_ids,
            tasks=self.tasks,
        )
        if self.plan_signature != expected:
            raise ValueError("plan_signature does not match the repair plan")
        return self


def make_resource_assignment_fingerprint(
    *,
    resource_type: ResourceType,
    subjects: Sequence[str],
    topic_ids: Sequence[str],
    requirement_ids: Sequence[str],
    evidence_ids: Sequence[str],
) -> str:
    """Fingerprint the exact evidence bundle assigned to one resource."""
    return _canonical_json_digest(
        {
            "algorithm": "resource_evidence_assignment_v1",
            "resource_type": resource_type,
            "subjects": list(subjects),
            "topic_ids": list(topic_ids),
            "requirement_ids": list(requirement_ids),
            "evidence_ids": list(evidence_ids),
        }
    )


class ResourceEvidenceAssignment(StrictRagConfigModel):
    """Approved evidence references assigned to one ready resource."""

    resource_type: ResourceType
    subjects: NonBlankStrTuple
    topic_ids: NonBlankStrTuple
    requirement_ids: NonBlankStrTuple
    evidence_ids: NonBlankStrTuple
    assignment_fingerprint: Sha256Digest

    @model_validator(mode="after")
    def _validate_assignment(self) -> "ResourceEvidenceAssignment":
        for field_name in (
            "subjects",
            "topic_ids",
            "requirement_ids",
            "evidence_ids",
        ):
            values = getattr(self, field_name)
            if not values:
                raise ValueError(f"{field_name} must not be empty")
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicates")
        if len(self.subjects) != len(self.topic_ids):
            raise ValueError(
                "subjects and topic_ids must form an ordered one-to-one binding"
            )
        expected = make_resource_assignment_fingerprint(
            resource_type=self.resource_type,
            subjects=self.subjects,
            topic_ids=self.topic_ids,
            requirement_ids=self.requirement_ids,
            evidence_ids=self.evidence_ids,
        )
        if self.assignment_fingerprint != expected:
            raise ValueError("assignment_fingerprint does not match the assignment")
        return self


class ResourceReadiness(StrictRagConfigModel):
    """Code-derived readiness for one requested resource."""

    resource_type: ResourceType
    readiness_state: ReadinessState
    required_requirement_ids: NonBlankStrTuple
    complete_requirement_ids: NonBlankStrTuple
    blocked_requirement_ids: NonBlankStrTuple
    evidence_ids: NonBlankStrTuple
    reason_code: str

    @model_validator(mode="after")
    def _validate_readiness(self) -> "ResourceReadiness":
        for field_name in (
            "required_requirement_ids",
            "complete_requirement_ids",
            "blocked_requirement_ids",
            "evidence_ids",
        ):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicates")
        if not self.required_requirement_ids:
            raise ValueError("resource readiness requires at least one required need")
        required = set(self.required_requirement_ids)
        complete = set(self.complete_requirement_ids)
        blocked = set(self.blocked_requirement_ids)
        if complete | blocked != required or complete & blocked:
            raise ValueError(
                "complete and blocked requirement ids must partition required ids"
            )
        if self.readiness_state == "ready":
            if blocked or self.reason_code:
                raise ValueError(
                    "ready resource must have no blocked ids or reason code"
                )
        else:
            if not blocked or not self.reason_code:
                raise ValueError(
                    "blocked resource must include blocked ids and a reason code"
                )
        return self


def validate_requirement_inventory(
    *,
    requested_resource_types: Sequence[ResourceType],
    requested_subjects: Sequence[str],
    canonical_subjects: AbstractSet[str],
    requirements: Sequence[EvidenceRequirement],
    profiles: ResourceEvidenceProfilesConfig,
    config: EvidenceOrchestrationConfig,
) -> None:
    """Require exact request/resource/profile coverage with no silent omission."""
    resources = tuple(requested_resource_types)
    subjects = tuple(requested_subjects)
    if not resources or len(resources) != len(set(resources)):
        raise EvidenceRequirementValidationError(
            code="invalid_requested_resources",
            reason="requested resources must be non-empty and unique",
        )
    if any(resource not in CANONICAL_RESOURCE_TYPES for resource in resources):
        raise EvidenceRequirementValidationError(
            code="noncanonical_resource",
            reason="requested resource is not canonical",
        )
    if not subjects or len(subjects) != len(set(subjects)):
        raise EvidenceRequirementValidationError(
            code="invalid_requested_subjects",
            reason="requested subjects must be non-empty and unique",
        )
    if any(subject not in canonical_subjects for subject in subjects):
        raise EvidenceRequirementValidationError(
            code="noncanonical_subject",
            reason="requested subject is not canonical",
        )
    if len(requirements) > config.max_requirements_per_request:
        raise EvidenceBudgetExceededError(
            code="requirement_budget_exceeded",
            reason="compiled requirements exceed max_requirements_per_request",
        )

    expected: dict[tuple[str, str, str], ResourceEvidenceNeed] = {}
    for resource in resources:
        profile = profiles.profile_for(resource)
        for subject in subjects:
            for need in profile.needs:
                expected[(resource, subject, need.need_id)] = need

    actual: dict[tuple[str, str, str], EvidenceRequirement] = {}
    topic_by_resource_subject: dict[tuple[str, str], str] = {}
    for requirement in requirements:
        requirement_slot = (
            requirement.resource_type,
            requirement.subject,
            requirement.profile_need_id,
        )
        if requirement_slot in actual:
            raise EvidenceRequirementValidationError(
                code="duplicate_requirement_slot",
                reason="multiple requirements target one resource/profile slot",
            )
        actual[requirement_slot] = requirement
        resource_subject = (requirement.resource_type, requirement.subject)
        bound_topic = topic_by_resource_subject.get(resource_subject)
        if bound_topic is not None and bound_topic != requirement.topic_id:
            raise EvidenceRequirementValidationError(
                code="multiple_target_topics_for_resource_subject",
                reason=("one generated resource must use one target topic per subject"),
            )
        topic_by_resource_subject[resource_subject] = requirement.topic_id
    if set(actual) != set(expected):
        raise EvidenceRequirementValidationError(
            code="requirement_inventory_mismatch",
            reason="requirements must exactly cover requested resource profile needs",
        )

    for actual_slot, requirement in actual.items():
        need = expected[actual_slot]
        if (
            requirement.evidence_kind != need.evidence_kind
            or requirement.scope != need.scope
            or requirement.criticality != need.criticality
            or requirement.source_policy != need.source_policy
            or requirement.acceptance_criteria != need.acceptance_criteria
        ):
            raise EvidenceRequirementValidationError(
                code="profile_contract_mismatch",
                reason="requirement metadata must exactly match its configured profile need",
            )


def validate_retrieval_tasks(
    *,
    tasks: Sequence[RetrievalTask],
    requirements: Sequence[EvidenceRequirement],
    config: EvidenceOrchestrationConfig,
    round_index: int,
    existing_total_search_tasks: int,
    prior_retrieval_signatures: AbstractSet[str],
    local_then_web_gap_requirement_ids: AbstractSet[str],
) -> None:
    """Validate task bindings, source policies, signatures, and budgets."""
    if round_index < 0 or round_index > config.max_supplement_rounds:
        raise EvidenceBudgetExceededError(
            code="round_budget_exceeded",
            reason="round_index exceeds configured supplement rounds",
        )
    if existing_total_search_tasks < 0:
        raise RetrievalTaskValidationError(
            code="invalid_existing_task_count",
            reason="existing_total_search_tasks must be non-negative",
        )
    if len(tasks) > config.max_search_tasks_per_round:
        raise EvidenceBudgetExceededError(
            code="round_task_budget_exceeded",
            reason="tasks exceed max_search_tasks_per_round",
        )
    if existing_total_search_tasks + len(tasks) > config.max_total_search_tasks:
        raise EvidenceBudgetExceededError(
            code="total_task_budget_exceeded",
            reason="tasks exceed max_total_search_tasks",
        )

    requirement_by_id = {item.requirement_id: item for item in requirements}
    if len(requirement_by_id) != len(requirements):
        raise RetrievalTaskValidationError(
            code="duplicate_requirement_id",
            reason="requirements must have unique deterministic ids",
        )
    current_signatures: set[str] = set()
    for task in tasks:
        requirement = requirement_by_id.get(task.requirement_id)
        if requirement is None:
            raise RetrievalTaskValidationError(
                code="unknown_requirement_id",
                reason="retrieval task must bind an existing requirement",
            )
        if task.round_index != round_index:
            raise RetrievalTaskValidationError(
                code="task_round_mismatch",
                reason="retrieval task round_index does not match the round",
            )
        if (
            task.resource_type != requirement.resource_type
            or task.subject != requirement.subject
        ):
            raise RetrievalTaskValidationError(
                code="task_requirement_mismatch",
                reason="task resource and subject must match its requirement",
            )
        if task.result_limit > config.max_results_per_task:
            raise EvidenceBudgetExceededError(
                code="task_result_budget_exceeded",
                reason="task result_limit exceeds max_results_per_task",
            )
        if (
            requirement.source_policy == "local_only"
            and task.source_type != "local_rag"
        ):
            raise RetrievalTaskValidationError(
                code="illegal_source_for_requirement",
                reason="local_only requirement cannot use web retrieval",
            )
        if requirement.source_policy == "web_only" and task.source_type != "web":
            raise RetrievalTaskValidationError(
                code="illegal_source_for_requirement",
                reason="web_only requirement cannot use local retrieval",
            )
        if (
            requirement.source_policy == "local_then_web_on_gap"
            and task.source_type == "web"
            and task.requirement_id not in local_then_web_gap_requirement_ids
        ):
            raise RetrievalTaskValidationError(
                code="web_before_local_gap",
                reason="local_then_web_on_gap requires an explicit local gap decision",
            )
        if (
            task.retrieval_signature in prior_retrieval_signatures
            or task.retrieval_signature in current_signatures
        ):
            raise DuplicateRetrievalSignatureError(
                code="duplicate_retrieval_signature",
                reason="retrieval signature must not repeat across rounds",
            )
        current_signatures.add(task.retrieval_signature)


def validate_evidence_ledger(
    *,
    entries: Sequence[EvidenceLedgerEntry],
    tasks: Sequence[RetrievalTask],
    requirements: Sequence[EvidenceRequirement],
    config: EvidenceOrchestrationConfig,
) -> None:
    """Validate cumulative ledger identity, binding, and configured bounds."""
    if len(entries) > config.max_ledger_entries:
        raise EvidenceBudgetExceededError(
            code="ledger_budget_exceeded",
            reason="ledger entries exceed max_ledger_entries",
        )
    task_by_id = {task.task_id: task for task in tasks}
    requirement_by_id = {item.requirement_id: item for item in requirements}
    if len(task_by_id) != len(tasks):
        raise EvidenceLedgerValidationError(
            code="duplicate_task_id",
            reason="retrieval tasks must have unique task ids",
        )
    evidence_ids: set[str] = set()
    accepted_counts: dict[str, int] = {}
    for entry in entries:
        if entry.evidence_id in evidence_ids:
            raise EvidenceLedgerValidationError(
                code="duplicate_evidence_identity",
                reason="exact evidence identity must not repeat in the ledger",
            )
        evidence_ids.add(entry.evidence_id)
        task = task_by_id.get(entry.task_id)
        requirement = requirement_by_id.get(entry.requirement_id)
        if task is None or requirement is None:
            raise EvidenceLedgerValidationError(
                code="unknown_ledger_binding",
                reason="ledger entry must bind known task and requirement ids",
            )
        if task.requirement_id != entry.requirement_id:
            raise EvidenceLedgerValidationError(
                code="task_requirement_binding_mismatch",
                reason="ledger task and requirement binding must agree",
            )
        if (
            entry.round_index != task.round_index
            or entry.resource_type != requirement.resource_type
            or entry.subject != requirement.subject
            or entry.source_type != task.source_type
        ):
            raise EvidenceLedgerValidationError(
                code="ledger_metadata_mismatch",
                reason="ledger metadata must match its task and requirement",
            )
        if entry.accepted:
            accepted_counts[entry.requirement_id] = (
                accepted_counts.get(entry.requirement_id, 0) + 1
            )
    if any(
        count > config.max_evidence_per_requirement
        for count in accepted_counts.values()
    ):
        raise EvidenceBudgetExceededError(
            code="requirement_evidence_budget_exceeded",
            reason="accepted evidence exceeds max_evidence_per_requirement",
        )


def validate_requirement_coverage(
    *,
    batch: CompiledRequirementCoverageBatch,
    requirements: Sequence[EvidenceRequirement],
    entries: Sequence[EvidenceLedgerEntry],
) -> None:
    """Require exactly one policy-valid coverage row per requirement."""
    if not isinstance(batch, CompiledRequirementCoverageBatch):
        raise TypeError("batch must be CompiledRequirementCoverageBatch")
    requirement_by_id = {item.requirement_id: item for item in requirements}
    coverage_by_id = {item.requirement_id: item for item in batch.coverages}
    if set(coverage_by_id) != set(requirement_by_id):
        raise RequirementCoverageValidationError(
            code="coverage_inventory_mismatch",
            reason="coverage batch must contain exactly one row per requirement",
        )
    accepted_by_id = {entry.evidence_id: entry for entry in entries if entry.accepted}
    evidence_ref_violations: list[tuple[str, int, int]] = []
    query_shape_violations: list[tuple[str, str, str, str, str]] = []
    for requirement_id, coverage in coverage_by_id.items():
        requirement = requirement_by_id[requirement_id]
        if (
            coverage.resource_type != requirement.resource_type
            or coverage.subject != requirement.subject
        ):
            raise RequirementCoverageValidationError(
                code="coverage_requirement_mismatch",
                reason="coverage resource and subject must match its requirement",
            )
        unknown_ref_count = 0
        cross_requirement_ref_count = 0
        for evidence_id in coverage.evidence_ids:
            entry = accepted_by_id.get(evidence_id)
            if entry is None:
                unknown_ref_count += 1
            elif entry.requirement_id != requirement_id:
                cross_requirement_ref_count += 1
        if unknown_ref_count or cross_requirement_ref_count:
            evidence_ref_violations.append(
                (
                    requirement_id,
                    unknown_ref_count,
                    cross_requirement_ref_count,
                )
            )
        has_local = bool(coverage.suggested_local_query)
        has_web = bool(coverage.suggested_web_query)
        if coverage.coverage_state == "complete":
            continue
        actual_shape = (
            "both"
            if has_local and has_web
            else "local_only"
            if has_local
            else "web_only"
            if has_web
            else "none"
        )
        violation_code = ""
        required_shape = ""
        if requirement.source_policy == "local_only" and (not has_local or has_web):
            violation_code = "invalid_local_only_gap_query"
            required_shape = "local_only"
        if requirement.source_policy == "web_only" and (has_local or not has_web):
            violation_code = "invalid_web_only_gap_query"
            required_shape = "web_only"
        if requirement.source_policy == "local_and_web" and (
            not has_local or not has_web
        ):
            violation_code = "invalid_dual_source_gap_query"
            required_shape = "both"
        if requirement.source_policy == "local_then_web_on_gap" and (
            not has_local and not has_web
        ):
            violation_code = "invalid_staged_source_gap_query"
            required_shape = "eligible_next_source_only"
        if violation_code:
            query_shape_violations.append(
                (
                    requirement_id,
                    requirement.source_policy,
                    actual_shape,
                    required_shape,
                    violation_code,
                )
            )
    ordered_query_shape_violations = sorted(query_shape_violations)
    query_shape_details = "; ".join(
        (
            f"requirement_id={requirement_id},"
            f"source_policy={source_policy},actual_shape={actual_shape},"
            f"required_shape={required_shape}"
        )
        for requirement_id, source_policy, actual_shape, required_shape, _code in (
            ordered_query_shape_violations
        )
    )
    if evidence_ref_violations:
        evidence_ref_details = "; ".join(
            (
                f"requirement_id={requirement_id},"
                f"unknown_ref_count={unknown_ref_count},"
                f"cross_requirement_ref_count={cross_requirement_ref_count}"
            )
            for requirement_id, unknown_ref_count, cross_requirement_ref_count in sorted(
                evidence_ref_violations
            )
        )
        reason = (
            "coverage rows may reference only accepted evidence bound to the same "
            f"requirement; violations=[{evidence_ref_details}]"
        )
        if query_shape_details:
            reason += f"; query_shape_violations=[{query_shape_details}]"
        raise RequirementCoverageValidationError(
            code="invalid_coverage_evidence_ref",
            reason=reason,
        )
    if ordered_query_shape_violations:
        violation_codes = {item[4] for item in ordered_query_shape_violations}
        error_code = (
            next(iter(violation_codes))
            if len(violation_codes) == 1
            else "invalid_source_gap_query_matrix"
        )
        raise RequirementCoverageValidationError(
            code=error_code,
            reason=(
                "coverage gap query shapes violate their source policies; "
                f"violations=[{query_shape_details}]"
            ),
        )


def derive_resource_readiness(
    *,
    requested_resource_types: Sequence[ResourceType],
    requirements: Sequence[EvidenceRequirement],
    batch: CompiledRequirementCoverageBatch,
) -> tuple[ResourceReadiness, ...]:
    """Derive readiness in code; LLM coverage cannot declare global readiness."""
    if not isinstance(batch, CompiledRequirementCoverageBatch):
        raise TypeError("batch must be CompiledRequirementCoverageBatch")
    coverage_by_id = {item.requirement_id: item for item in batch.coverages}
    readiness_rows: list[ResourceReadiness] = []
    for raw_resource_type in requested_resource_types:
        if raw_resource_type not in CANONICAL_RESOURCE_TYPES:
            raise ResourceEvidenceAssignmentError(
                code="noncanonical_resource",
                reason="readiness can only be derived for canonical resources",
            )
        resource_type = raw_resource_type
        resource_requirements = tuple(
            item for item in requirements if item.resource_type == resource_type
        )
        required_ids = tuple(
            item.requirement_id
            for item in resource_requirements
            if item.criticality == "required"
        )
        if not required_ids:
            raise ResourceEvidenceAssignmentError(
                code="missing_required_resource_need",
                reason="resource has no compiled required evidence need",
            )
        complete_ids = tuple(
            requirement_id
            for requirement_id in required_ids
            if coverage_by_id.get(requirement_id) is not None
            and coverage_by_id[requirement_id].coverage_state == "complete"
        )
        blocked_ids = tuple(
            requirement_id
            for requirement_id in required_ids
            if requirement_id not in set(complete_ids)
        )
        evidence_ids = _unique_preserving_order(
            tuple(
                evidence_id
                for requirement in resource_requirements
                if requirement.requirement_id in coverage_by_id
                for evidence_id in coverage_by_id[
                    requirement.requirement_id
                ].evidence_ids
            )
        )
        is_ready = not blocked_ids
        readiness_rows.append(
            ResourceReadiness(
                resource_type=resource_type,
                readiness_state=(
                    "ready" if is_ready else "blocked_insufficient_evidence"
                ),
                required_requirement_ids=required_ids,
                complete_requirement_ids=complete_ids,
                blocked_requirement_ids=blocked_ids,
                evidence_ids=evidence_ids,
                reason_code="" if is_ready else "required_evidence_incomplete",
            )
        )
    return tuple(readiness_rows)


def derive_resource_evidence_assignments(
    *,
    readiness: Sequence[ResourceReadiness],
    requirements: Sequence[EvidenceRequirement],
    batch: CompiledRequirementCoverageBatch,
    entries: Sequence[EvidenceLedgerEntry],
) -> tuple[ResourceEvidenceAssignment, ...]:
    """Assign accepted evidence only to resources whose required needs are ready."""
    if not isinstance(batch, CompiledRequirementCoverageBatch):
        raise TypeError("batch must be CompiledRequirementCoverageBatch")
    accepted_by_id = {entry.evidence_id: entry for entry in entries if entry.accepted}
    coverage_by_id = {item.requirement_id: item for item in batch.coverages}
    assignments: list[ResourceEvidenceAssignment] = []
    for row in readiness:
        if row.readiness_state != "ready":
            continue
        resource_requirements = tuple(
            item for item in requirements if item.resource_type == row.resource_type
        )
        subjects = _unique_preserving_order(
            tuple(item.subject for item in resource_requirements)
        )
        topic_ids = _unique_preserving_order(
            tuple(item.topic_id for item in resource_requirements)
        )
        requirement_ids = tuple(
            item.requirement_id
            for item in resource_requirements
            if coverage_by_id[item.requirement_id].evidence_ids
        )
        evidence_ids = _unique_preserving_order(
            tuple(
                evidence_id
                for requirement_id in requirement_ids
                for evidence_id in coverage_by_id[requirement_id].evidence_ids
            )
        )
        if not evidence_ids:
            raise ResourceEvidenceAssignmentError(
                code="ready_resource_without_evidence",
                reason="ready resource must have accepted assigned evidence",
            )
        for evidence_id in evidence_ids:
            entry = accepted_by_id.get(evidence_id)
            if entry is None or entry.resource_type != row.resource_type:
                raise ResourceEvidenceAssignmentError(
                    code="invalid_assignment_evidence_ref",
                    reason="assignment may use only accepted evidence for its resource",
                )
        fingerprint = make_resource_assignment_fingerprint(
            resource_type=row.resource_type,
            subjects=subjects,
            topic_ids=topic_ids,
            requirement_ids=requirement_ids,
            evidence_ids=evidence_ids,
        )
        assignments.append(
            ResourceEvidenceAssignment(
                resource_type=row.resource_type,
                subjects=subjects,
                topic_ids=topic_ids,
                requirement_ids=requirement_ids,
                evidence_ids=evidence_ids,
                assignment_fingerprint=fingerprint,
            )
        )
    return tuple(assignments)


__all__ = [
    "CompiledRequirementCoverage",
    "CompiledRequirementCoverageBatch",
    "CoverageState",
    "DuplicateRetrievalSignatureError",
    "EvidenceBudgetExceededError",
    "EvidenceLedgerEntry",
    "EvidenceLedgerValidationError",
    "EvidenceOrchestrationContractError",
    "EvidenceRepairPlan",
    "EvidenceRequirement",
    "EvidenceRequirementDraft",
    "EvidenceRequirementDraftBatch",
    "EvidenceRequirementValidationError",
    "EvidenceSourceType",
    "RequirementCoverage",
    "RequirementCoverageBatch",
    "RequirementCoverageValidationError",
    "RESOURCE_EVIDENCE_CONTRACT_VERSION",
    "ResourceEvidenceAssignment",
    "ResourceEvidenceAssignmentError",
    "ResourceReadiness",
    "RetrievalPriority",
    "RetrievalTask",
    "RetrievalTaskValidationError",
    "build_retrieval_task",
    "compile_requirement_coverage_batch",
    "compile_evidence_requirement",
    "compile_evidence_requirement_batch",
    "derive_resource_evidence_assignments",
    "derive_resource_readiness",
    "make_evidence_id",
    "make_evidence_requirement_id",
    "make_query_fingerprint",
    "make_repair_plan_signature",
    "make_resource_assignment_fingerprint",
    "make_retrieval_signature",
    "make_retrieval_task_id",
    "validate_evidence_ledger",
    "validate_requirement_coverage",
    "validate_requirement_inventory",
    "validate_retrieval_tasks",
]
