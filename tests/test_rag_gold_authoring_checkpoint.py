"""Focused contracts for resumable human-directed Gold authoring."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.rag.gold_authoring import (
    EvaluationTarget,
    EvaluationTargetArtifact,
    GoldAuthoringApproval,
    GoldAuthoringCheckpoint,
    GoldAuthoringContractError,
    GoldEvidenceSliceDigest,
    apply_gold_authoring_approvals,
    gold_query_sha256,
    load_evaluation_target_artifact,
    load_gold_authoring_checkpoint,
    validate_evaluation_target_artifact,
    validate_gold_authoring_checkpoint,
    validate_gold_authoring_evidence,
    write_gold_authoring_artifact,
)
from src.rag.gold_dataset import (
    GoldDatasetDraft,
    GoldInspectionPage,
    GoldInspectionSection,
    GoldSourceInspection,
    load_gold_dataset,
)
from src.rag.parent_child._storage_io import (
    model_json_bytes,
    sha256_bytes,
)
from src.rag.parent_child.evaluation import GoldDataset, GoldEvidenceSpan, GoldQuery


_DOC_ID = "doc_" + "a" * 40
_CONTENT = (
    "Separable equations\n"
    "dy/dx is not a fraction; it is a mnemonic. "
    "The chain rule with y=y(x) and dy=y'(x)dx gives the same integral."
)
_START = _CONTENT.index("dy/dx")
_END = len(_CONTENT)


def _query(*, subject: str = "math", eligible: bool = False) -> GoldQuery:
    return GoldQuery(
        schema_version="gold_query_v1",
        query_id="math-q023",
        subject=subject,
        query=(
            "Why is treating dy/dx as a fraction only a mnemonic, and how does "
            "the chain rule justify separation of variables?"
        ),
        dataset_kind="human_gold",
        eligible_for_rollout=eligible,
        gold_spans=(
            GoldEvidenceSpan(
                schema_version="gold_evidence_span_v1",
                gold_span_id="gold_math_023",
                source_group_id="calculus_clp2",
                source_relpath="math/clp2.pdf",
                doc_id=_DOC_ID,
                pagination_kind="physical",
                page_start=1,
                page_end=1,
                start_char=_START,
                end_char=_END,
                section_path=("SEPARABLE DIFFERENTIAL EQUATIONS",),
                relevance_grade=3,
            ),
        ),
    )


def _approval(*, proposal: GoldQuery | None = None) -> GoldAuthoringApproval:
    selected = proposal or _query()
    content = _CONTENT[_START:_END]
    return GoldAuthoringApproval(
        schema_version="gold_authoring_approval_v1",
        approval_id="approval_math_q023_a_v1",
        decision_sequence=1,
        decision_recorded_at_utc=datetime(2026, 7, 15, tzinfo=timezone.utc),
        operation="replace_existing",
        selected_option_id="q023-A",
        authoring_method="human_directed_ai_assisted",
        decision="approved_for_draft_write",
        authorization_scope="draft_write_only",
        proposal=selected,
        proposal_sha256=gold_query_sha256(selected),
        evidence_slice_digests=(
            GoldEvidenceSliceDigest(
                schema_version="gold_evidence_slice_digest_v1",
                gold_span_id="gold_math_023",
                cleaned_slice_sha256=sha256_bytes(content.encode("utf-8")),
            ),
        ),
    )


def _base_dataset() -> GoldDataset:
    original = _query(eligible=True)
    original_payload = original.model_dump(mode="python")
    original_payload["query"] = "What conditions are required for a limit?"
    return GoldDataset(
        schema_version="gold_dataset_v1",
        dataset_id="local_gold_v2",
        queries=(GoldQuery.model_validate(original_payload),),
    )


def _checkpoint() -> tuple[GoldAuthoringCheckpoint, GoldDatasetDraft]:
    base = _base_dataset()
    approval = _approval()
    draft = apply_gold_authoring_approvals(
        base_dataset=base,
        draft_dataset_id="local_gold_v3_draft",
        approvals=(approval,),
    )
    checkpoint = GoldAuthoringCheckpoint(
        schema_version="gold_authoring_checkpoint_v1",
        checkpoint_id="gold_v3_checkpoint_001",
        checkpoint_sequence=1,
        previous_checkpoint_sha256=None,
        base_dataset_id=base.dataset_id,
        draft_dataset_id=draft.dataset_id,
        base_gold_dataset_sha256=sha256_bytes(model_json_bytes(base)),
        draft_gold_dataset_sha256=sha256_bytes(model_json_bytes(draft)),
        index_config_sha256="b" * 64,
        source_groups_sha256="c" * 64,
        workflow_state="authoring_in_progress",
        completed_semantic_reviewer_count=0,
        required_semantic_reviewer_count=2,
        second_reviewer_status="missing",
        evaluation_eligible=False,
        approvals=(approval,),
        unresolved_authoring_query_ids=("computer-q101",),
    )
    return checkpoint, draft


def _inspection() -> GoldSourceInspection:
    return GoldSourceInspection(
        schema_version="gold_source_inspection_v1",
        source_relpath="math/clp2.pdf",
        doc_id=_DOC_ID,
        subject="math",
        pagination_kind="physical",
        pages=(
            GoldInspectionPage(
                page_number=1,
                start_char=0,
                content_end_char=len(_CONTENT),
                end_char=len(_CONTENT),
                cleaned_text=_CONTENT,
            ),
        ),
        sections=(
            GoldInspectionSection(
                start_char=0,
                end_char=len(_CONTENT),
                section_path=("SEPARABLE DIFFERENTIAL EQUATIONS",),
            ),
        ),
        cleaned_content=_CONTENT,
    )


def _target_artifact(checkpoint: GoldAuthoringCheckpoint) -> EvaluationTargetArtifact:
    approval = checkpoint.approvals[0]
    return EvaluationTargetArtifact(
        schema_version="gold_evaluation_target_artifact_v1",
        dataset_id=checkpoint.draft_dataset_id,
        authoring_checkpoint_sha256=sha256_bytes(model_json_bytes(checkpoint)),
        evaluation_eligible=False,
        targets=(
            EvaluationTarget(
                schema_version="gold_evaluation_target_v1",
                query_id=approval.proposal.query_id,
                subject=approval.proposal.subject,
                proposal_sha256=approval.proposal_sha256,
                product_function="academic_question_answering",
                expected_graph_route="academic",
                tested_capability_ids=(
                    "cross_line_formula_retrieval",
                    "mathematical_derivation_synthesis",
                ),
                rag_stages=(
                    "query_rewrite",
                    "subject_scoped_vector_retrieval",
                    "bm25_retrieval",
                    "weighted_rrf",
                    "reranker",
                    "evidence_judge",
                    "parent_hydration",
                    "page_citation",
                ),
                success_criteria=(
                    "Explain why dy/dx is not an ordinary fraction.",
                    "Use the chain rule or substitution and cite the evidence page.",
                ),
                resource_type=None,
            ),
        ),
    )


def test_checkpoint_evidence_targets_and_safe_writes_round_trip(tmp_path: Path) -> None:
    checkpoint, draft = _checkpoint()
    base = _base_dataset()
    target_artifact = _target_artifact(checkpoint)

    validate_gold_authoring_checkpoint(
        checkpoint=checkpoint,
        base_dataset=base,
        draft_dataset=draft,
    )
    validate_gold_authoring_evidence(
        approval=checkpoint.approvals[0],
        inspections={"math/clp2.pdf": _inspection()},
    )
    validate_evaluation_target_artifact(
        checkpoint=checkpoint,
        artifact=target_artifact,
    )

    checkpoint_path = write_gold_authoring_artifact(
        project_root=tmp_path,
        output_path=Path("reports/checkpoint.json"),
        artifact=checkpoint,
        overwrite=False,
    )
    target_path = write_gold_authoring_artifact(
        project_root=tmp_path,
        output_path=Path("reports/target.json"),
        artifact=target_artifact,
        overwrite=False,
    )
    assert load_gold_authoring_checkpoint(checkpoint_path) == checkpoint
    assert load_evaluation_target_artifact(target_path) == target_artifact
    with pytest.raises(Exception, match="final GoldDataset contract"):
        load_gold_dataset(checkpoint_path)
    with pytest.raises(FileExistsError):
        write_gold_authoring_artifact(
            project_root=tmp_path,
            output_path=Path("reports/checkpoint.json"),
            artifact=checkpoint,
            overwrite=False,
        )


def test_approval_rejects_rollout_eligibility_and_digest_tampering() -> None:
    rollout_query = _query(eligible=True)
    with pytest.raises(ValidationError, match="never be rollout eligible"):
        _approval(proposal=rollout_query)

    approval = _approval()
    payload = approval.model_dump(mode="python")
    payload["proposal_sha256"] = "d" * 64
    with pytest.raises(ValidationError, match="proposal_sha256"):
        GoldAuthoringApproval.model_validate(payload)

    digest_payload = approval.model_dump(mode="python")
    digest_payload["evidence_slice_digests"][0]["cleaned_slice_sha256"] = "e" * 64
    tampered = GoldAuthoringApproval.model_validate(digest_payload)
    with pytest.raises(GoldAuthoringContractError, match="slice SHA-256"):
        validate_gold_authoring_evidence(
            approval=tampered,
            inspections={"math/clp2.pdf": _inspection()},
        )


def test_replacement_cannot_change_subject_or_target_identity() -> None:
    subject_changed = _approval(proposal=_query(subject="computer"))
    with pytest.raises(GoldAuthoringContractError, match="cannot change"):
        apply_gold_authoring_approvals(
            base_dataset=_base_dataset(),
            draft_dataset_id="local_gold_v3_draft",
            approvals=(subject_changed,),
        )

    checkpoint, _ = _checkpoint()
    target = _target_artifact(checkpoint)
    payload = target.model_dump(mode="python")
    payload["targets"][0]["proposal_sha256"] = "f" * 64
    mismatched = EvaluationTargetArtifact.model_validate(payload)
    with pytest.raises(GoldAuthoringContractError, match="proposal SHA"):
        validate_evaluation_target_artifact(
            checkpoint=checkpoint,
            artifact=mismatched,
        )


def test_checkpoint_rejects_eligibility_missing_previous_and_extra_fields() -> None:
    checkpoint, _ = _checkpoint()
    eligible = checkpoint.model_dump(mode="python")
    eligible["evaluation_eligible"] = True
    with pytest.raises(ValidationError, match="never evaluation eligible"):
        GoldAuthoringCheckpoint.model_validate(eligible)

    missing_previous = checkpoint.model_dump(mode="python")
    missing_previous["checkpoint_sequence"] = 2
    with pytest.raises(ValidationError, match="previous_checkpoint_sha256"):
        GoldAuthoringCheckpoint.model_validate(missing_previous)

    extra = checkpoint.model_dump(mode="python")
    extra["reviewer_02_approved"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GoldAuthoringCheckpoint.model_validate(extra)


def test_evaluation_target_requires_function_route_resource_consistency() -> None:
    checkpoint, _ = _checkpoint()
    target = _target_artifact(checkpoint).targets[0]
    payload = target.model_dump(mode="python")
    payload["expected_graph_route"] = "personalized_resource"
    with pytest.raises(ValidationError, match="academic route"):
        EvaluationTarget.model_validate(payload)

    duplicate = target.model_dump(mode="python")
    duplicate["tested_capability_ids"] = (
        "cross_line_formula_retrieval",
        "cross_line_formula_retrieval",
    )
    with pytest.raises(ValidationError, match="must be unique"):
        EvaluationTarget.model_validate(duplicate)
