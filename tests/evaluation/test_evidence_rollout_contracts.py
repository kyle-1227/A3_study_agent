"""Strict schema and fingerprint tests for evidence rollout evaluation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.evaluation.evidence_rollout.contracts import (
    EvidenceEvaluationDatasetContentV2,
    EvidenceEvaluationDatasetV2,
    EvidenceVariantAttemptBatchContentV2,
    EvidenceVariantAttemptBatchV2,
    EvidenceVariantAttemptV2,
    HumanSemanticReviewBatchContentV2,
    HumanSemanticReviewBatchV2,
    HumanSemanticReviewV2,
    canonical_sha256,
    query_fingerprint,
)
from src.learning_guidance.knowledge_graph import load_knowledge_graph


KG = load_knowledge_graph(
    Path(__file__).resolve().parents[2]
    / "config"
    / "learning_guidance"
    / "knowledge_graph_v1.yaml"
)


def _fingerprint_target(payload: dict[str, object]) -> dict[str, object]:
    payload["target_fingerprint"] = canonical_sha256(payload)
    return payload


def _fingerprint_initial(
    *,
    case_id: str,
    state: str,
    source_inventory: list[str],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "evidence_initial_evidence_identity_v2",
        "state": state,
        "fixture_id": f"{case_id}_initial",
        "source_inventory": source_inventory,
    }
    payload["fixture_fingerprint"] = canonical_sha256(payload)
    return payload


def _fingerprint_case(payload: dict[str, object]) -> dict[str, object]:
    payload["case_fingerprint"] = canonical_sha256(payload)
    return payload


def _dataset_payload() -> dict[str, object]:
    return {
        "schema_version": "evidence_evaluation_dataset_v2",
        "dataset_id": "hermetic_contract_suite",
        "knowledge_graph_data_version": KG.data_version,
        "knowledge_graph_artifact_fingerprint": KG.artifact_fingerprint,
        "cases": [
            _fingerprint_case(
                {
                    "schema_version": "evidence_evaluation_case_spec_v2",
                    "case_id": "simple_case",
                    "query": "Explain a Python iterator.",
                    "subjects": ["python"],
                    "resource_types": ["review_doc"],
                    "initial_evidence": _fingerprint_initial(
                        case_id="simple_case",
                        state="insufficient",
                        source_inventory=[],
                    ),
                    "targets": [
                        _fingerprint_target(
                            {
                                "schema_version": "evidence_resource_subject_target_v2",
                                "target_id": "simple_python_doc",
                                "subject": "python",
                                "resource_type": "review_doc",
                                "topic_id": "python.fundamentals",
                                "catalog_resource_ids": ["python_basics_real_python"],
                                "required_sources": ["parent_child"],
                            }
                        )
                    ],
                    "requirements": [
                        {
                            "schema_version": "evidence_requirement_gold_v2",
                            "requirement_id": "simple_iterator_definition",
                            "target_id": "simple_python_doc",
                            "criterion": "Defines iterator behavior accurately.",
                            "weight": 1.0,
                        }
                    ],
                }
            ),
            _fingerprint_case(
                {
                    "schema_version": "evidence_evaluation_case_spec_v2",
                    "case_id": "multi_case",
                    "query": "Compare a model pipeline with its data platform.",
                    "subjects": ["machine_learning", "big_data"],
                    "resource_types": ["review_doc", "quiz"],
                    "initial_evidence": _fingerprint_initial(
                        case_id="multi_case",
                        state="sufficient",
                        source_inventory=["parent_child", "web"],
                    ),
                    "targets": [
                        _fingerprint_target(
                            {
                                "schema_version": "evidence_resource_subject_target_v2",
                                "target_id": "multi_ml_doc",
                                "subject": "machine_learning",
                                "resource_type": "review_doc",
                                "topic_id": "machine_learning.classical_methods",
                                "catalog_resource_ids": [
                                    "machine_learning_zhou_zhihua"
                                ],
                                "required_sources": ["parent_child"],
                            }
                        ),
                        _fingerprint_target(
                            {
                                "schema_version": "evidence_resource_subject_target_v2",
                                "target_id": "multi_data_quiz",
                                "subject": "big_data",
                                "resource_type": "quiz",
                                "topic_id": "big_data.data_engineering",
                                "catalog_resource_ids": [
                                    "big_data_data_engineering_zoomcamp"
                                ],
                                "required_sources": ["parent_child", "web"],
                            }
                        ),
                    ],
                    "requirements": [
                        {
                            "schema_version": "evidence_requirement_gold_v2",
                            "requirement_id": "multi_model_pipeline",
                            "target_id": "multi_ml_doc",
                            "criterion": "Explains the model pipeline dependency.",
                            "weight": 1.0,
                        },
                        {
                            "schema_version": "evidence_requirement_gold_v2",
                            "requirement_id": "multi_data_platform",
                            "target_id": "multi_data_quiz",
                            "criterion": "Tests the data platform relationship.",
                            "weight": 1.0,
                        },
                    ],
                }
            ),
        ],
    }


def _signed_dataset(
    payload: dict[str, object] | None = None,
) -> EvidenceEvaluationDatasetV2:
    content = EvidenceEvaluationDatasetContentV2.model_validate(
        _dataset_payload() if payload is None else payload
    )
    return EvidenceEvaluationDatasetV2(
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
        EvidenceEvaluationDatasetV2.model_validate(invalid_fingerprint)

    extra_field = dataset.model_dump(mode="python")
    extra_field["query_alias"] = "must not be accepted"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvidenceEvaluationDatasetV2.model_validate(extra_field)


def test_dataset_does_not_coerce_tuple_or_normalize_subjects() -> None:
    tuple_payload = _dataset_payload()
    tuple_payload["cases"][0]["subjects"] = ("python",)  # type: ignore[index]
    with pytest.raises(ValidationError, match="valid list"):
        EvidenceEvaluationDatasetContentV2.model_validate(tuple_payload)

    noncanonical_payload = _dataset_payload()
    noncanonical_payload["cases"][0]["subjects"] = [  # type: ignore[index]
        "Python"
    ]
    with pytest.raises(ValidationError, match="case-folded"):
        EvidenceEvaluationDatasetContentV2.model_validate(noncanonical_payload)

    tuple_catalog = _dataset_payload()
    tuple_catalog["cases"][0]["targets"][0]["catalog_resource_ids"] = (  # type: ignore[index]
        "python_basics_real_python",
    )
    with pytest.raises(ValidationError, match="valid list"):
        EvidenceEvaluationDatasetContentV2.model_validate(tuple_catalog)


def test_v1_and_legacy_initial_evidence_payloads_are_rejected() -> None:
    legacy_schema = _dataset_payload()
    legacy_schema["schema_version"] = "evidence_evaluation_dataset_v1"
    with pytest.raises(ValidationError):
        EvidenceEvaluationDatasetContentV2.model_validate(legacy_schema)

    legacy_boolean = _dataset_payload()
    first_case = legacy_boolean["cases"][0]  # type: ignore[index]
    first_case.pop("initial_evidence")
    first_case["initial_evidence_sufficient"] = True
    with pytest.raises(ValidationError, match="initial_evidence"):
        EvidenceEvaluationDatasetContentV2.model_validate(legacy_boolean)


def test_target_case_and_initial_identity_fingerprints_fail_closed() -> None:
    stale_target = _dataset_payload()
    stale_target["cases"][0]["targets"][0]["required_sources"] = [  # type: ignore[index]
        "web"
    ]
    with pytest.raises(ValidationError, match="target_fingerprint"):
        EvidenceEvaluationDatasetContentV2.model_validate(stale_target)

    stale_case = _dataset_payload()
    stale_case["cases"][0]["query"] = "Changed without resigning."  # type: ignore[index]
    with pytest.raises(ValidationError, match="case_fingerprint"):
        EvidenceEvaluationDatasetContentV2.model_validate(stale_case)

    stale_initial = _dataset_payload()
    stale_initial["cases"][0]["initial_evidence"]["source_inventory"] = [  # type: ignore[index]
        "parent_child"
    ]
    with pytest.raises(ValidationError, match="fixture_fingerprint"):
        EvidenceEvaluationDatasetContentV2.model_validate(stale_initial)


def test_target_binding_fields_are_required_without_aliases_or_empty_inventory() -> (
    None
):
    missing_topic = _dataset_payload()
    missing_topic["cases"][0]["targets"][0].pop("topic_id")  # type: ignore[index]
    with pytest.raises(ValidationError, match="topic_id"):
        EvidenceEvaluationDatasetContentV2.model_validate(missing_topic)

    alias_resources = _dataset_payload()
    target = alias_resources["cases"][0]["targets"][0]  # type: ignore[index]
    target["resource_ids"] = target.pop("catalog_resource_ids")
    with pytest.raises(ValidationError, match="catalog_resource_ids"):
        EvidenceEvaluationDatasetContentV2.model_validate(alias_resources)

    empty_resources = _dataset_payload()
    empty_resources["cases"][0]["targets"][0]["catalog_resource_ids"] = []  # type: ignore[index]
    with pytest.raises(ValidationError, match="at least 1 item"):
        EvidenceEvaluationDatasetContentV2.model_validate(empty_resources)

    sufficient_without_source = _dataset_payload()
    first = sufficient_without_source["cases"][0]  # type: ignore[index]
    first["initial_evidence"] = _fingerprint_initial(
        case_id="simple_case",
        state="sufficient",
        source_inventory=[],
    )
    with pytest.raises(ValidationError, match="requires a source inventory"):
        EvidenceEvaluationDatasetContentV2.model_validate(sufficient_without_source)


def test_dataset_fingerprint_binds_every_content_change() -> None:
    original = _signed_dataset()
    changed_payload = deepcopy(_dataset_payload())
    changed_payload["cases"][0]["query"] = "Explain a Python generator."  # type: ignore[index]
    changed_case = changed_payload["cases"][0]  # type: ignore[index]
    changed_case.pop("case_fingerprint")
    changed_case["case_fingerprint"] = canonical_sha256(changed_case)
    changed_content = EvidenceEvaluationDatasetContentV2.model_validate(changed_payload)

    stale_signature_payload = changed_content.model_dump(mode="python")
    stale_signature_payload["dataset_fingerprint"] = original.dataset_fingerprint
    with pytest.raises(ValidationError, match="dataset_fingerprint"):
        EvidenceEvaluationDatasetV2.model_validate(stale_signature_payload)

    assert query_fingerprint("Explain a Python iterator.") != query_fingerprint(
        "Explain a Python generator."
    )


def test_attempt_batch_rejects_duplicate_slots_and_stale_fingerprint() -> None:
    attempt = EvidenceVariantAttemptV2(
        schema_version="evidence_variant_attempt_v2",
        case_id="simple_case",
        variant="P0",
        status="blocked",
        observation=None,
        failure_reason_code="fixture_blocked",
        failure_type="HermeticFixtureError",
    )
    with pytest.raises(ValidationError, match="must not repeat"):
        EvidenceVariantAttemptBatchContentV2(
            schema_version="evidence_variant_attempt_batch_v2",
            execution_mode="hermetic",
            executor_fingerprint="1" * 64,
            attempts=[attempt, attempt],
        )

    content = EvidenceVariantAttemptBatchContentV2(
        schema_version="evidence_variant_attempt_batch_v2",
        execution_mode="hermetic",
        executor_fingerprint="1" * 64,
        attempts=[attempt],
    )
    with pytest.raises(ValidationError, match="bundle_fingerprint"):
        EvidenceVariantAttemptBatchV2(
            **content.model_dump(mode="python"),
            bundle_fingerprint="2" * 64,
        )


def test_attempt_shape_cannot_turn_failure_into_empty_success() -> None:
    with pytest.raises(ValidationError, match="requires only an observation"):
        EvidenceVariantAttemptV2(
            schema_version="evidence_variant_attempt_v2",
            case_id="simple_case",
            variant="P0",
            status="success",
            observation=None,
            failure_reason_code=None,
            failure_type=None,
        )

    payload = {
        "schema_version": "evidence_variant_attempt_v2",
        "case_id": "simple_case",
        "variant": "P0",
        "status": "failed",
        "observation": None,
        "failure_reason_code": "fixture_failed",
        "failure_type": "HermeticFixtureError",
        "raw_provider_body": "must not enter the contract",
    }
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvidenceVariantAttemptV2.model_validate(payload)


def test_human_review_requires_canonical_time_and_signed_inventory() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        HumanSemanticReviewV2(
            schema_version="human_semantic_review_v2",
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

    content = HumanSemanticReviewBatchContentV2(
        schema_version="human_semantic_review_batch_v2",
        dataset_fingerprint="5" * 64,
        runtime_fingerprint="6" * 64,
        generation_id="generation_1",
        generation_manifest_fingerprint="8" * 64,
        review_protocol_fingerprint="9" * 64,
        reviews=[],
    )
    with pytest.raises(ValidationError, match="review_bundle_fingerprint"):
        HumanSemanticReviewBatchV2(
            **content.model_dump(mode="python"),
            review_bundle_fingerprint="7" * 64,
        )
