"""Strict sidecar contracts for human-directed GoldDataset authoring.

These artifacts record draft-writing authorization and evaluation intent.  They
are deliberately separate from :class:`GoldDataset`: neither a checkpoint nor
an evaluation target is accepted by the benchmark/readiness Gold loader.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.rag.gold_dataset import (
    GoldDatasetAuthoringError,
    GoldDatasetDraft,
    GoldSourceInspection,
)
from src.rag.parent_child._storage_io import (
    model_json_bytes,
    sha256_bytes,
)
from src.rag.parent_child.evaluation import GoldDataset, GoldQuery
from src.rag.parent_child.project_paths import (
    ProjectPathError,
    atomic_write_project_bytes,
    resolve_project_path,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CAPABILITY_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")

GoldAuthoringOperation = Literal["replace_existing", "add_new"]
GoldAuthoringMethod = Literal["human_directed_ai_assisted"]
GoldAuthoringDecision = Literal["approved_for_draft_write"]
GoldAuthorizationScope = Literal["draft_write_only"]
GoldProductFunction = Literal[
    "academic_question_answering",
    "personalized_resource_generation",
]
GoldGraphRoute = Literal["academic", "personalized_resource"]
GoldResourceType = Literal[
    "review_doc",
    "mindmap",
    "quiz",
    "code_practice",
    "video_script",
    "video_animation",
    "study_plan",
]
GoldRagStage = Literal[
    "query_rewrite",
    "subject_scoped_vector_retrieval",
    "bm25_retrieval",
    "weighted_rrf",
    "reranker",
    "evidence_judge",
    "parent_hydration",
    "page_citation",
]


class GoldAuthoringContractError(GoldDatasetAuthoringError):
    """A separately valid Gold authoring artifact violates a cross-contract rule."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _validate_identifier(value: str, *, field_name: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty and already stripped")
    if any(character in value for character in ("/", "\\", "\x00")):
        raise ValueError(f"{field_name} must not contain path separators or NUL")
    return value


def _validate_sha256(value: str, *, field_name: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def gold_query_sha256(query: GoldQuery) -> str:
    """Return the canonical identity of one strictly validated proposal."""

    return sha256_bytes(model_json_bytes(query))


class GoldEvidenceSliceDigest(_StrictFrozenModel):
    """Content identity for one approved policy-independent evidence slice."""

    schema_version: Literal["gold_evidence_slice_digest_v1"]
    gold_span_id: str
    cleaned_slice_sha256: str

    @field_validator("gold_span_id")
    @classmethod
    def _validate_gold_span_id(_cls, value: str) -> str:
        return _validate_identifier(value, field_name="gold_span_id")

    @field_validator("cleaned_slice_sha256")
    @classmethod
    def _validate_cleaned_slice_sha256(_cls, value: str) -> str:
        return _validate_sha256(value, field_name="cleaned_slice_sha256")


class GoldAuthoringApproval(_StrictFrozenModel):
    """One explicit human authorization to write an AI-assisted draft proposal."""

    schema_version: Literal["gold_authoring_approval_v1"]
    approval_id: str
    decision_sequence: int = Field(ge=1)
    decision_recorded_at_utc: datetime
    operation: GoldAuthoringOperation
    selected_option_id: str
    authoring_method: GoldAuthoringMethod
    decision: GoldAuthoringDecision
    authorization_scope: GoldAuthorizationScope
    proposal: GoldQuery
    proposal_sha256: str
    evidence_slice_digests: tuple[GoldEvidenceSliceDigest, ...] = Field(min_length=1)

    @field_validator("approval_id", "selected_option_id")
    @classmethod
    def _validate_ids(_cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "identifier")
        return _validate_identifier(value, field_name=field_name)

    @field_validator("proposal_sha256")
    @classmethod
    def _validate_proposal_sha256(_cls, value: str) -> str:
        return _validate_sha256(value, field_name="proposal_sha256")

    @field_validator("decision_recorded_at_utc")
    @classmethod
    def _validate_decision_recorded_at_utc(_cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("decision_recorded_at_utc must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def _validate_approval(self) -> GoldAuthoringApproval:
        if self.proposal.dataset_kind != "human_gold":
            raise ValueError("authoring proposals must use dataset_kind=human_gold")
        if self.proposal.eligible_for_rollout:
            raise ValueError("draft-writing approval must never be rollout eligible")
        if self.proposal_sha256 != gold_query_sha256(self.proposal):
            raise ValueError("proposal_sha256 does not match the canonical proposal")
        digest_ids = tuple(item.gold_span_id for item in self.evidence_slice_digests)
        span_ids = tuple(item.gold_span_id for item in self.proposal.gold_spans)
        if len(digest_ids) != len(set(digest_ids)):
            raise ValueError(
                "evidence_slice_digests must use unique gold_span_id values"
            )
        if set(digest_ids) != set(span_ids):
            raise ValueError("evidence slice digest IDs must equal proposal span IDs")
        return self


class GoldAuthoringCheckpoint(_StrictFrozenModel):
    """SHA-bound resumable state for draft authoring, never semantic approval."""

    schema_version: Literal["gold_authoring_checkpoint_v1"]
    checkpoint_id: str
    checkpoint_sequence: int = Field(ge=1)
    previous_checkpoint_sha256: str | None
    base_dataset_id: str
    draft_dataset_id: str
    base_gold_dataset_sha256: str
    draft_gold_dataset_sha256: str
    index_config_sha256: str
    source_groups_sha256: str
    workflow_state: Literal["authoring_in_progress"]
    completed_semantic_reviewer_count: int = Field(ge=0)
    required_semantic_reviewer_count: int = Field(ge=2)
    second_reviewer_status: Literal["missing"]
    evaluation_eligible: bool
    approvals: tuple[GoldAuthoringApproval, ...] = Field(min_length=1)
    unresolved_authoring_query_ids: tuple[str, ...]

    @field_validator("checkpoint_id", "base_dataset_id", "draft_dataset_id")
    @classmethod
    def _validate_ids(_cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "identifier")
        return _validate_identifier(value, field_name=field_name)

    @field_validator(
        "base_gold_dataset_sha256",
        "draft_gold_dataset_sha256",
        "index_config_sha256",
        "source_groups_sha256",
    )
    @classmethod
    def _validate_required_sha256(_cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "sha256")
        return _validate_sha256(value, field_name=field_name)

    @field_validator("previous_checkpoint_sha256")
    @classmethod
    def _validate_previous_sha256(_cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_sha256(value, field_name="previous_checkpoint_sha256")

    @field_validator("unresolved_authoring_query_ids")
    @classmethod
    def _validate_unresolved_ids(_cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for query_id in value:
            _validate_identifier(query_id, field_name="unresolved_authoring_query_id")
        if len(value) != len(set(value)):
            raise ValueError("unresolved authoring query IDs must be unique")
        if value != tuple(sorted(value)):
            raise ValueError("unresolved authoring query IDs must be sorted")
        return value

    @model_validator(mode="after")
    def _validate_checkpoint(self) -> GoldAuthoringCheckpoint:
        if (self.checkpoint_sequence == 1) != (self.previous_checkpoint_sha256 is None):
            raise ValueError(
                "only checkpoint sequence 1 may omit previous_checkpoint_sha256"
            )
        approval_ids = tuple(item.approval_id for item in self.approvals)
        query_ids = tuple(item.proposal.query_id for item in self.approvals)
        sequences = tuple(item.decision_sequence for item in self.approvals)
        if len(approval_ids) != len(set(approval_ids)):
            raise ValueError("checkpoint approval IDs must be unique")
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("checkpoint proposal query IDs must be unique")
        if sequences != tuple(range(1, len(sequences) + 1)):
            raise ValueError("approval decision_sequence values must be contiguous")
        if set(query_ids) & set(self.unresolved_authoring_query_ids):
            raise ValueError("approved and unresolved query IDs must be disjoint")
        if (
            self.completed_semantic_reviewer_count
            >= self.required_semantic_reviewer_count
        ):
            raise ValueError(
                "authoring checkpoint cannot claim completed semantic review"
            )
        if self.evaluation_eligible:
            raise ValueError("authoring checkpoint is never evaluation eligible")
        return self


class EvaluationTarget(_StrictFrozenModel):
    """What product behavior one approved draft question is intended to test."""

    schema_version: Literal["gold_evaluation_target_v1"]
    query_id: str
    subject: str
    proposal_sha256: str
    product_function: GoldProductFunction
    expected_graph_route: GoldGraphRoute
    tested_capability_ids: tuple[str, ...] = Field(min_length=1)
    rag_stages: tuple[GoldRagStage, ...] = Field(min_length=1)
    success_criteria: tuple[str, ...] = Field(min_length=1)
    resource_type: GoldResourceType | None

    @field_validator("query_id", "subject")
    @classmethod
    def _validate_ids(_cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "identifier")
        return _validate_identifier(value, field_name=field_name)

    @field_validator("proposal_sha256")
    @classmethod
    def _validate_proposal_sha256(_cls, value: str) -> str:
        return _validate_sha256(value, field_name="proposal_sha256")

    @field_validator("tested_capability_ids")
    @classmethod
    def _validate_capabilities(_cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(_CAPABILITY_ID_RE.fullmatch(item) is None for item in value):
            raise ValueError("tested capability IDs must be normalized identifiers")
        if len(value) != len(set(value)):
            raise ValueError("tested capability IDs must be unique")
        if value != tuple(sorted(value)):
            raise ValueError("tested capability IDs must be sorted")
        return value

    @field_validator("rag_stages")
    @classmethod
    def _validate_rag_stages(
        _cls, value: tuple[GoldRagStage, ...]
    ) -> tuple[GoldRagStage, ...]:
        if len(value) != len(set(value)):
            raise ValueError("RAG stages must be unique")
        return value

    @field_validator("success_criteria")
    @classmethod
    def _validate_success_criteria(_cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item or item != item.strip() for item in value):
            raise ValueError("success criteria must be non-empty and stripped")
        if len(value) != len(set(value)):
            raise ValueError("success criteria must be unique")
        return value

    @model_validator(mode="after")
    def _validate_product_route(self) -> EvaluationTarget:
        if self.product_function == "academic_question_answering":
            if (
                self.expected_graph_route != "academic"
                or self.resource_type is not None
            ):
                raise ValueError(
                    "academic question answering requires academic route and no resource"
                )
        elif (
            self.expected_graph_route != "personalized_resource"
            or self.resource_type is None
        ):
            raise ValueError(
                "personalized resource generation requires its route and resource type"
            )
        return self


class EvaluationTargetArtifact(_StrictFrozenModel):
    """Evaluation targets bound to exactly one authoring checkpoint."""

    schema_version: Literal["gold_evaluation_target_artifact_v1"]
    dataset_id: str
    authoring_checkpoint_sha256: str
    evaluation_eligible: bool
    targets: tuple[EvaluationTarget, ...] = Field(min_length=1)

    @field_validator("dataset_id")
    @classmethod
    def _validate_dataset_id(_cls, value: str) -> str:
        return _validate_identifier(value, field_name="dataset_id")

    @field_validator("authoring_checkpoint_sha256")
    @classmethod
    def _validate_checkpoint_sha256(_cls, value: str) -> str:
        return _validate_sha256(value, field_name="authoring_checkpoint_sha256")

    @model_validator(mode="after")
    def _validate_targets(self) -> EvaluationTargetArtifact:
        query_ids = tuple(target.query_id for target in self.targets)
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("evaluation target query IDs must be unique")
        if query_ids != tuple(sorted(query_ids)):
            raise ValueError("evaluation targets must be sorted by query_id")
        if self.evaluation_eligible:
            raise ValueError(
                "authoring evaluation targets are never evaluation eligible"
            )
        return self


GoldAuthoringArtifact = GoldAuthoringCheckpoint | EvaluationTargetArtifact


def apply_gold_authoring_approvals(
    *,
    base_dataset: GoldDataset,
    draft_dataset_id: str,
    approvals: tuple[GoldAuthoringApproval, ...],
) -> GoldDatasetDraft:
    """Apply explicitly approved proposals without mutating the source dataset."""

    approval_by_id = {item.proposal.query_id: item for item in approvals}
    base_by_id = {item.query_id: item for item in base_dataset.queries}
    for approval in approvals:
        query_id = approval.proposal.query_id
        original = base_by_id.get(query_id)
        if approval.operation == "replace_existing":
            if original is None:
                raise GoldAuthoringContractError(
                    "replacement proposal query_id is absent from base Gold"
                )
            if original.subject != approval.proposal.subject:
                raise GoldAuthoringContractError(
                    "replacement proposal cannot change the Gold subject"
                )
        elif original is not None:
            raise GoldAuthoringContractError(
                "add_new proposal query_id already exists in base Gold"
            )
    queries: list[GoldQuery] = []
    for item in base_dataset.queries:
        matching_approval = approval_by_id.get(item.query_id)
        queries.append(
            matching_approval.proposal if matching_approval is not None else item
        )
    additions = sorted(
        (item.proposal for item in approvals if item.operation == "add_new"),
        key=lambda item: item.query_id,
    )
    queries.extend(additions)
    return GoldDatasetDraft(
        schema_version="gold_dataset_draft_v1",
        dataset_id=draft_dataset_id,
        queries=tuple(queries),
    )


def validate_gold_authoring_checkpoint(
    *,
    checkpoint: GoldAuthoringCheckpoint,
    base_dataset: GoldDataset,
    draft_dataset: GoldDatasetDraft,
) -> None:
    """Validate identity and exact draft projection across authoring artifacts."""

    if checkpoint.base_dataset_id != base_dataset.dataset_id:
        raise GoldAuthoringContractError("checkpoint base dataset_id mismatch")
    if checkpoint.draft_dataset_id != draft_dataset.dataset_id:
        raise GoldAuthoringContractError("checkpoint draft dataset_id mismatch")
    if checkpoint.base_gold_dataset_sha256 != sha256_bytes(
        model_json_bytes(base_dataset)
    ):
        raise GoldAuthoringContractError("checkpoint base Gold SHA-256 mismatch")
    if checkpoint.draft_gold_dataset_sha256 != sha256_bytes(
        model_json_bytes(draft_dataset)
    ):
        raise GoldAuthoringContractError("checkpoint draft Gold SHA-256 mismatch")
    expected = apply_gold_authoring_approvals(
        base_dataset=base_dataset,
        draft_dataset_id=draft_dataset.dataset_id,
        approvals=checkpoint.approvals,
    )
    if expected != draft_dataset:
        raise GoldAuthoringContractError(
            "draft dataset differs from approved proposals"
        )


def validate_gold_authoring_evidence(
    *,
    approval: GoldAuthoringApproval,
    inspections: Mapping[str, GoldSourceInspection],
) -> None:
    """Recompute approved cleaned-slice hashes from page-aware inspections."""

    digests = {
        item.gold_span_id: item.cleaned_slice_sha256
        for item in approval.evidence_slice_digests
    }
    for span in approval.proposal.gold_spans:
        inspection = inspections.get(span.source_relpath)
        if inspection is None:
            raise GoldAuthoringContractError(
                "approved evidence has no page-aware source inspection"
            )
        if span.doc_id != inspection.doc_id:
            raise GoldAuthoringContractError("approved evidence doc_id mismatch")
        if approval.proposal.subject != inspection.subject:
            raise GoldAuthoringContractError("approved evidence subject mismatch")
        if span.pagination_kind != inspection.pagination_kind:
            raise GoldAuthoringContractError(
                "approved evidence pagination_kind mismatch"
            )
        if span.end_char > len(inspection.cleaned_content):
            raise GoldAuthoringContractError("approved evidence exceeds cleaned source")
        content = inspection.cleaned_content[span.start_char : span.end_char]
        if not content.strip():
            raise GoldAuthoringContractError("approved evidence slice is empty")
        if sha256_bytes(content.encode("utf-8")) != digests[span.gold_span_id]:
            raise GoldAuthoringContractError("approved evidence slice SHA-256 mismatch")
        matching_pages = tuple(
            page.page_number
            for page in inspection.pages
            if page.start_char < span.end_char and span.start_char < page.end_char
        )
        if not matching_pages or (
            min(matching_pages),
            max(matching_pages),
        ) != (span.page_start, span.page_end):
            raise GoldAuthoringContractError("approved evidence page range mismatch")
        matching_sections = tuple(
            section.section_path
            for section in inspection.sections
            if section.start_char <= span.start_char
            and span.end_char <= section.end_char
        )
        if matching_sections != (span.section_path,):
            raise GoldAuthoringContractError("approved evidence section path mismatch")


def validate_evaluation_target_artifact(
    *,
    checkpoint: GoldAuthoringCheckpoint,
    artifact: EvaluationTargetArtifact,
) -> None:
    """Require exact proposal identity parity between targets and approvals."""

    if artifact.dataset_id != checkpoint.draft_dataset_id:
        raise GoldAuthoringContractError("evaluation target dataset_id mismatch")
    if artifact.authoring_checkpoint_sha256 != sha256_bytes(
        model_json_bytes(checkpoint)
    ):
        raise GoldAuthoringContractError("evaluation target checkpoint SHA mismatch")
    approvals = {item.proposal.query_id: item for item in checkpoint.approvals}
    targets = {item.query_id: item for item in artifact.targets}
    if set(approvals) != set(targets):
        raise GoldAuthoringContractError(
            "evaluation targets must exactly cover checkpoint approvals"
        )
    for query_id, approval in approvals.items():
        target = targets[query_id]
        if target.subject != approval.proposal.subject:
            raise GoldAuthoringContractError("evaluation target subject mismatch")
        if target.proposal_sha256 != approval.proposal_sha256:
            raise GoldAuthoringContractError("evaluation target proposal SHA mismatch")


def load_gold_authoring_checkpoint(path: Path) -> GoldAuthoringCheckpoint:
    """Strictly load one checkpoint without schema repair or fallback."""

    try:
        return GoldAuthoringCheckpoint.model_validate_json(path.read_text("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise GoldAuthoringContractError(
            f"unable to load Gold authoring checkpoint: {type(exc).__name__}"
        ) from exc


def load_evaluation_target_artifact(path: Path) -> EvaluationTargetArtifact:
    """Strictly load one evaluation target sidecar."""

    try:
        return EvaluationTargetArtifact.model_validate_json(path.read_text("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise GoldAuthoringContractError(
            f"unable to load evaluation target artifact: {type(exc).__name__}"
        ) from exc


def write_gold_authoring_artifact(
    *,
    project_root: Path,
    output_path: Path,
    artifact: GoldAuthoringArtifact,
    overwrite: bool,
) -> Path:
    """Atomically persist a validated sidecar below the project root."""

    try:
        root = resolve_project_path(project_root, project_root, must_exist=True)
        output = resolve_project_path(root, output_path, must_exist=False)
        return atomic_write_project_bytes(
            root,
            output,
            model_json_bytes(artifact),
            overwrite=overwrite,
        )
    except ProjectPathError as exc:
        raise GoldAuthoringContractError(str(exc)) from exc


__all__ = [
    "EvaluationTarget",
    "EvaluationTargetArtifact",
    "GoldAuthoringApproval",
    "GoldAuthoringCheckpoint",
    "GoldAuthoringContractError",
    "GoldEvidenceSliceDigest",
    "apply_gold_authoring_approvals",
    "gold_query_sha256",
    "load_evaluation_target_artifact",
    "load_gold_authoring_checkpoint",
    "validate_evaluation_target_artifact",
    "validate_gold_authoring_checkpoint",
    "validate_gold_authoring_evidence",
    "write_gold_authoring_artifact",
]
