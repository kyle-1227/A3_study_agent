"""Strict schema and fingerprint tests for evidence rollout evaluation."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from src.evaluation.evidence_rollout.contracts import (
    EvidenceEvaluationDatasetContentV1,
    EvidenceEvaluationDatasetV1,
    EvidenceVariantAttemptBatchContentV1,
    EvidenceVariantAttemptBatchV1,
    EvidenceVariantAttemptV1,
    HumanSemanticReviewBatchContentV1,
    HumanSemanticReviewBatchV1,
    HumanSemanticReviewV1,
    canonical_sha256,
    query_fingerprint,
)


def _dataset_payload() -> dict[str, object]:
    return {
        "schema_version": "evidence_evaluation_dataset_v1",
        "dataset_id": "hermetic_contract_suite",
        "cases": [
            {
                "schema_version": "evidence_evaluation_case_spec_v1",
                "case_id": "simple_case",
                "query": "Explain a Python iterator.",
                "subjects": ["python"],
                "resource_types": ["review_doc"],
                "initial_evidence_sufficient": False,
                "targets": [
                    {
                        "schema_version": "evidence_resource_subject_target_v1",
                        "target_id": "simple_python_doc",
                        "subject": "python",
                        "resource_type": "review_doc",
                        "required_sources": ["parent_child"],
                    }
                ],
                "requirements": [
                    {
                        "schema_version": "evidence_requirement_gold_v1",
                        "requirement_id": "simple_iterator_definition",
                        "target_id": "simple_python_doc",
                        "criterion": "Defines iterator behavior accurately.",
                        "weight": 1.0,
                    }
                ],
            },
            {
                "schema_version": "evidence_evaluation_case_spec_v1",
                "case_id": "multi_case",
                "query": "Compare a model pipeline with its data platform.",
                "subjects": ["machine_learning", "big_data"],
                "resource_types": ["review_doc", "quiz"],
                "initial_evidence_sufficient": True,
                "targets": [
                    {
                        "schema_version": "evidence_resource_subject_target_v1",
                        "target_id": "multi_ml_doc",
                        "subject": "machine_learning",
                        "resource_type": "review_doc",
                        "required_sources": ["parent_child"],
                    },
                    {
                        "schema_version": "evidence_resource_subject_target_v1",
                        "target_id": "multi_data_quiz",
                        "subject": "big_data",
                        "resource_type": "quiz",
                        "required_sources": ["parent_child", "web"],
                    },
                ],
                "requirements": [
                    {
                        "schema_version": "evidence_requirement_gold_v1",
                        "requirement_id": "multi_model_pipeline",
                        "target_id": "multi_ml_doc",
                        "criterion": "Explains the model pipeline dependency.",
                        "weight": 1.0,
                    },
                    {
                        "schema_version": "evidence_requirement_gold_v1",
                        "requirement_id": "multi_data_platform",
                        "target_id": "multi_data_quiz",
                        "criterion": "Tests the data platform relationship.",
                        "weight": 1.0,
                    },
                ],
            },
        ],
    }


def _signed_dataset(
    payload: dict[str, object] | None = None,
) -> EvidenceEvaluationDatasetV1:
    content = EvidenceEvaluationDatasetContentV1.model_validate(
        _dataset_payload() if payload is None else payload
    )
    return EvidenceEvaluationDatasetV1(
        **content.model_dump(mode="python"),
        dataset_fingerprint=canonical_sha256(content.model_dump(mode="json")),
    )


def test_dataset_requires_exact_schema_and_precomputed_content_fingerprint() -> None:
    dataset = _signed_dataset()

    assert dataset.dataset_fingerprint == canonical_sha256(
        dataset.model_dump(mode="json", exclude={"dataset_fingerprint"})
    )

    invalid_fingerprint = dataset.model_dump(mode="python")
    invalid_fingerprint["dataset_fingerprint"] = "0" * 64
    with pytest.raises(ValidationError, match="dataset_fingerprint"):
        EvidenceEvaluationDatasetV1.model_validate(invalid_fingerprint)

    extra_field = dataset.model_dump(mode="python")
    extra_field["query_alias"] = "must not be accepted"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvidenceEvaluationDatasetV1.model_validate(extra_field)


def test_dataset_does_not_coerce_tuple_or_normalize_subjects() -> None:
    tuple_payload = _dataset_payload()
    tuple_payload["cases"][0]["subjects"] = ("python",)  # type: ignore[index]
    with pytest.raises(ValidationError, match="valid list"):
        EvidenceEvaluationDatasetContentV1.model_validate(tuple_payload)

    noncanonical_payload = _dataset_payload()
    noncanonical_payload["cases"][0]["subjects"] = [  # type: ignore[index]
        "Python"
    ]
    with pytest.raises(ValidationError, match="case-folded"):
        EvidenceEvaluationDatasetContentV1.model_validate(noncanonical_payload)


def test_dataset_fingerprint_binds_every_content_change() -> None:
    original = _signed_dataset()
    changed_payload = deepcopy(_dataset_payload())
    changed_payload["cases"][0]["query"] = "Explain a Python generator."  # type: ignore[index]
    changed_content = EvidenceEvaluationDatasetContentV1.model_validate(changed_payload)

    stale_signature_payload = changed_content.model_dump(mode="python")
    stale_signature_payload["dataset_fingerprint"] = original.dataset_fingerprint
    with pytest.raises(ValidationError, match="dataset_fingerprint"):
        EvidenceEvaluationDatasetV1.model_validate(stale_signature_payload)

    assert query_fingerprint("Explain a Python iterator.") != query_fingerprint(
        "Explain a Python generator."
    )


def test_attempt_batch_rejects_duplicate_slots_and_stale_fingerprint() -> None:
    attempt = EvidenceVariantAttemptV1(
        schema_version="evidence_variant_attempt_v1",
        case_id="simple_case",
        variant="P0",
        status="blocked",
        observation=None,
        failure_reason_code="fixture_blocked",
        failure_type="HermeticFixtureError",
    )
    with pytest.raises(ValidationError, match="must not repeat"):
        EvidenceVariantAttemptBatchContentV1(
            schema_version="evidence_variant_attempt_batch_v1",
            execution_mode="hermetic",
            executor_fingerprint="1" * 64,
            attempts=[attempt, attempt],
        )

    content = EvidenceVariantAttemptBatchContentV1(
        schema_version="evidence_variant_attempt_batch_v1",
        execution_mode="hermetic",
        executor_fingerprint="1" * 64,
        attempts=[attempt],
    )
    with pytest.raises(ValidationError, match="bundle_fingerprint"):
        EvidenceVariantAttemptBatchV1(
            **content.model_dump(mode="python"),
            bundle_fingerprint="2" * 64,
        )


def test_attempt_shape_cannot_turn_failure_into_empty_success() -> None:
    with pytest.raises(ValidationError, match="requires only an observation"):
        EvidenceVariantAttemptV1(
            schema_version="evidence_variant_attempt_v1",
            case_id="simple_case",
            variant="P0",
            status="success",
            observation=None,
            failure_reason_code=None,
            failure_type=None,
        )

    payload = {
        "schema_version": "evidence_variant_attempt_v1",
        "case_id": "simple_case",
        "variant": "P0",
        "status": "failed",
        "observation": None,
        "failure_reason_code": "fixture_failed",
        "failure_type": "HermeticFixtureError",
        "raw_provider_body": "must not enter the contract",
    }
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvidenceVariantAttemptV1.model_validate(payload)


def test_human_review_requires_canonical_time_and_signed_inventory() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        HumanSemanticReviewV1(
            schema_version="human_semantic_review_v1",
            case_id="simple_case",
            variant="P0",
            output_fingerprint="3" * 64,
            reviewer_identity_hash="4" * 64,
            reviewed_at="2026-07-15T10:00:00",
            assessment_source="human",
            supported_claim_count=1,
            claim_count=1,
            ungrounded_fact_count=0,
            fact_count=1,
        )

    content = HumanSemanticReviewBatchContentV1(
        schema_version="human_semantic_review_batch_v1",
        dataset_fingerprint="5" * 64,
        runtime_fingerprint="6" * 64,
        generation_id="generation_1",
        generation_manifest_fingerprint="8" * 64,
        review_protocol_fingerprint="9" * 64,
        reviews=[],
    )
    with pytest.raises(ValidationError, match="review_bundle_fingerprint"):
        HumanSemanticReviewBatchV1(
            **content.model_dump(mode="python"),
            review_bundle_fingerprint="7" * 64,
        )
