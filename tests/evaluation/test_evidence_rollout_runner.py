"""Hermetic, fail-closed tests for the evidence rollout runner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn

import pytest

from src.config.evidence_benchmark_config import (
    EvidenceBenchmarkConfig,
    load_evidence_benchmark_config,
)
from src.config.rag_rollout_config import RagRolloutConfig, load_rag_rollout_config
from src.evaluation.evidence_rollout.contracts import (
    EvidenceEvaluationCaseSpecV2,
    EvidenceEvaluationDatasetContentV2,
    EvidenceEvaluationDatasetV2,
    EvidenceEvaluationRuntimeBindingV2,
    EvidenceSource,
    EvidenceLiveAdapterIdentityV2,
    EvidenceRolloutExecutionConfigV2,
    EvidenceVariantAttemptBatchContentV2,
    EvidenceVariantAttemptBatchV2,
    EvidenceVariantAttemptV2,
    EvidenceVariantDefinitionV2,
    EvidenceVariantObservationV2,
    HumanSemanticReviewBatchContentV2,
    HumanSemanticReviewBatchV2,
    HumanSemanticReviewV2,
    case_binding_identity,
    canonical_sha256,
    dataset_case_bindings,
    model_fingerprint,
    query_fingerprint,
)
from src.learning_guidance.knowledge_graph import (
    KnowledgeGraphV1,
    load_knowledge_graph,
)
from src.evaluation.evidence_rollout.runner import (
    LiveEvidenceVariantExecutor,
    SealedAttemptVariantExecutor,
    run_evidence_rollout_evaluation,
)
from src.rag.parent_child.evidence_evaluation import (
    EvidenceEvaluationError,
    Variant,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG_PATH = ROOT / "config" / "rag" / "evidence_benchmark.yaml"
ROLLOUT_CONFIG_PATH = ROOT / "config" / "rag" / "rollout.yaml"
KNOWLEDGE_GRAPH_PATH = ROOT / "config" / "learning_guidance" / "knowledge_graph_v1.yaml"
KNOWLEDGE_GRAPH = load_knowledge_graph(KNOWLEDGE_GRAPH_PATH)
EXECUTOR_FINGERPRINT = "b" * 64
RUNTIME_FINGERPRINT = "a" * 64
REVIEWER_IDENTITY_HASH = "d" * 64
GENERATION_MANIFEST_FINGERPRINT = "c" * 64
REVIEW_PROTOCOL_FINGERPRINT = "9" * 64

_VARIANT_FACTORS: tuple[tuple[Variant, bool, bool], ...] = (
    ("P0", False, False),
    ("PG", True, False),
    ("PR", False, True),
    ("PGR", True, True),
)

_MEASUREMENTS: dict[tuple[str, Variant], dict[str, float | int]] = {
    ("simple_case", "P0"): {
        "coverage": 0.80,
        "gaps": 1,
        "precision": 0.94,
        "claim_support": 0.75,
        "ungrounded": 0.10,
        "cost": 1.0,
        "latency": 100.0,
    },
    ("simple_case", "PG"): {
        "coverage": 0.82,
        "gaps": 1,
        "precision": 0.94,
        "claim_support": 0.77,
        "ungrounded": 0.09,
        "cost": 1.05,
        "latency": 105.0,
    },
    ("simple_case", "PR"): {
        "coverage": 0.83,
        "gaps": 0,
        "precision": 0.93,
        "claim_support": 0.79,
        "ungrounded": 0.08,
        "cost": 1.08,
        "latency": 110.0,
    },
    ("simple_case", "PGR"): {
        "coverage": 0.85,
        "gaps": 0,
        "precision": 0.93,
        "claim_support": 0.84,
        "ungrounded": 0.07,
        "cost": 1.10,
        "latency": 120.0,
    },
    ("multi_case", "P0"): {
        "coverage": 0.60,
        "gaps": 2,
        "precision": 0.94,
        "claim_support": 0.65,
        "ungrounded": 0.20,
        "cost": 2.0,
        "latency": 200.0,
    },
    ("multi_case", "PG"): {
        "coverage": 0.67,
        "gaps": 1,
        "precision": 0.93,
        "claim_support": 0.70,
        "ungrounded": 0.17,
        "cost": 2.2,
        "latency": 215.0,
    },
    ("multi_case", "PR"): {
        "coverage": 0.70,
        "gaps": 1,
        "precision": 0.93,
        "claim_support": 0.72,
        "ungrounded": 0.15,
        "cost": 2.5,
        "latency": 225.0,
    },
    ("multi_case", "PGR"): {
        "coverage": 0.75,
        "gaps": 1,
        "precision": 0.93,
        "claim_support": 0.78,
        "ungrounded": 0.12,
        "cost": 2.8,
        "latency": 240.0,
    },
}


@dataclass(frozen=True)
class _Scenario:
    dataset: EvidenceEvaluationDatasetV2
    knowledge_graph: KnowledgeGraphV1
    execution_config: EvidenceRolloutExecutionConfigV2
    benchmark_config: EvidenceBenchmarkConfig
    rollout_config: RagRolloutConfig
    binding: EvidenceEvaluationRuntimeBindingV2
    batch: EvidenceVariantAttemptBatchV2
    reviews: HumanSemanticReviewBatchV2


def _execution_config() -> EvidenceRolloutExecutionConfigV2:
    return EvidenceRolloutExecutionConfigV2(
        schema_version="evidence_rollout_execution_config_v2",
        variants=[
            EvidenceVariantDefinitionV2(
                variant=variant,
                resource_planning_enabled=planning,
                bounded_repair_enabled=repair,
            )
            for variant, planning, repair in _VARIANT_FACTORS
        ],
        human_semantic_review_required=True,
        human_review_protocol_fingerprint=REVIEW_PROTOCOL_FINGERPRINT,
        candidate_failure_policy="fail_fast",
        report_policy="content_free_v2",
        max_case_count=10,
    )


def _target(
    *,
    target_id: str,
    subject: str,
    resource_type: str,
    topic_id: str,
    catalog_resource_ids: list[str],
    required_sources: list[EvidenceSource],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "evidence_resource_subject_target_v2",
        "target_id": target_id,
        "subject": subject,
        "resource_type": resource_type,
        "topic_id": topic_id,
        "catalog_resource_ids": catalog_resource_ids,
        "required_sources": required_sources,
    }
    payload["target_fingerprint"] = canonical_sha256(payload)
    return payload


def _initial_evidence(
    *,
    case_id: str,
    state: Literal["sufficient", "insufficient"],
    source_inventory: list[EvidenceSource],
) -> dict[str, object]:
    fixture_id = f"{case_id}_initial_evidence"
    descriptor = {
        "schema_version": "evidence_initial_evidence_identity_v2",
        "fixture_id": fixture_id,
        "state": state,
        "source_inventory": source_inventory,
    }
    return {
        "schema_version": "evidence_initial_evidence_identity_v2",
        "state": state,
        "fixture_id": fixture_id,
        "fixture_fingerprint": canonical_sha256(descriptor),
        "source_inventory": source_inventory,
    }


def _case(
    *,
    case_id: str,
    query: str,
    subjects: list[str],
    resource_types: list[str],
    initial_state: Literal["sufficient", "insufficient"],
    initial_sources: list[EvidenceSource],
    targets: list[dict[str, object]],
    requirements: list[dict[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "evidence_evaluation_case_spec_v2",
        "case_id": case_id,
        "query": query,
        "subjects": subjects,
        "resource_types": resource_types,
        "initial_evidence": _initial_evidence(
            case_id=case_id,
            state=initial_state,
            source_inventory=initial_sources,
        ),
        "targets": targets,
        "requirements": requirements,
    }
    payload["case_fingerprint"] = canonical_sha256(payload)
    return payload


def _dataset(
    *, simple_query: str = "Explain a Python iterator."
) -> EvidenceEvaluationDatasetV2:
    cases = [
        _case(
            case_id="simple_case",
            query=simple_query,
            subjects=["python"],
            resource_types=["review_doc"],
            initial_state="sufficient",
            initial_sources=["parent_child"],
            targets=[
                _target(
                    target_id="simple_python_doc",
                    subject="python",
                    resource_type="review_doc",
                    topic_id="python.fundamentals",
                    catalog_resource_ids=["python_basics_real_python"],
                    required_sources=["parent_child"],
                )
            ],
            requirements=[
                {
                    "schema_version": "evidence_requirement_gold_v2",
                    "requirement_id": "simple_iterator_definition",
                    "target_id": "simple_python_doc",
                    "criterion": "Defines iterator behavior accurately.",
                    "weight": 1.0,
                }
            ],
        ),
        _case(
            case_id="multi_case",
            query="Compare a model pipeline with its data platform.",
            subjects=["machine_learning", "big_data"],
            resource_types=["review_doc", "quiz"],
            initial_state="insufficient",
            initial_sources=[],
            targets=[
                _target(
                    target_id="multi_ml_doc",
                    subject="machine_learning",
                    resource_type="review_doc",
                    topic_id="machine_learning.classical_methods",
                    catalog_resource_ids=["machine_learning_zhou_zhihua"],
                    required_sources=["parent_child"],
                ),
                _target(
                    target_id="multi_data_quiz",
                    subject="big_data",
                    resource_type="quiz",
                    topic_id="big_data.data_engineering",
                    catalog_resource_ids=["big_data_data_engineering_zoomcamp"],
                    required_sources=["parent_child", "web"],
                ),
            ],
            requirements=[
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
        ),
    ]
    content = EvidenceEvaluationDatasetContentV2(
        schema_version="evidence_evaluation_dataset_v2",
        dataset_id="hermetic_runner_suite",
        knowledge_graph_data_version=KNOWLEDGE_GRAPH.data_version,
        knowledge_graph_artifact_fingerprint=KNOWLEDGE_GRAPH.artifact_fingerprint,
        cases=cases,
    )
    return EvidenceEvaluationDatasetV2(
        **content.model_dump(mode="python"),
        dataset_fingerprint=canonical_sha256(content.model_dump(mode="json")),
    )


def _binding(
    *,
    dataset: EvidenceEvaluationDatasetV2,
    execution_config: EvidenceRolloutExecutionConfigV2,
    benchmark_config: EvidenceBenchmarkConfig,
    rollout_config: RagRolloutConfig,
    executor_fingerprint: str = EXECUTOR_FINGERPRINT,
) -> EvidenceEvaluationRuntimeBindingV2:
    return EvidenceEvaluationRuntimeBindingV2(
        schema_version="evidence_evaluation_runtime_binding_v2",
        run_id="hermetic_run_1",
        execution_mode="hermetic",
        dataset_id=dataset.dataset_id,
        dataset_fingerprint=dataset.dataset_fingerprint,
        knowledge_graph_data_version=dataset.knowledge_graph_data_version,
        knowledge_graph_artifact_fingerprint=(
            dataset.knowledge_graph_artifact_fingerprint
        ),
        case_bindings=dataset_case_bindings(dataset),
        execution_config_fingerprint=model_fingerprint(execution_config),
        benchmark_config_fingerprint=model_fingerprint(benchmark_config),
        rollout_config_fingerprint=model_fingerprint(rollout_config),
        runtime_fingerprint=RUNTIME_FINGERPRINT,
        generation_id="hermetic_generation_1",
        generation_manifest_fingerprint=GENERATION_MANIFEST_FINGERPRINT,
        executor_fingerprint=executor_fingerprint,
    )


def _observation(
    *,
    case: EvidenceEvaluationCaseSpecV2,
    definition: EvidenceVariantDefinitionV2,
    binding: EvidenceEvaluationRuntimeBindingV2,
    **overrides: object,
) -> EvidenceVariantObservationV2:
    measurement = _MEASUREMENTS[(case.case_id, definition.variant)]
    requirement_weight_total = sum(item.weight for item in case.requirements)
    expected_route_count = sum(len(target.required_sources) for target in case.targets)
    web_required = any("web" in target.required_sources for target in case.targets)
    values: dict[str, object] = {
        "schema_version": "evidence_variant_observation_v2",
        "case_id": case.case_id,
        "variant": definition.variant,
        "query_fingerprint": query_fingerprint(case.query),
        "dataset_fingerprint": binding.dataset_fingerprint,
        "knowledge_graph_data_version": binding.knowledge_graph_data_version,
        "knowledge_graph_artifact_fingerprint": (
            binding.knowledge_graph_artifact_fingerprint
        ),
        "case_binding": case_binding_identity(case),
        "execution_config_fingerprint": binding.execution_config_fingerprint,
        "benchmark_config_fingerprint": binding.benchmark_config_fingerprint,
        "rollout_config_fingerprint": binding.rollout_config_fingerprint,
        "runtime_fingerprint": binding.runtime_fingerprint,
        "generation_id": binding.generation_id,
        "generation_manifest_fingerprint": (binding.generation_manifest_fingerprint),
        "executor_fingerprint": binding.executor_fingerprint,
        "variant_definition_fingerprint": model_fingerprint(definition),
        "output_fingerprint": canonical_sha256(
            {
                "case_id": case.case_id,
                "variant": definition.variant,
                "sealed": True,
            }
        ),
        "provider_status": "ok",
        "parent_child_status": "ok",
        "web_status": "ok" if web_required else "not_required",
        "bounded": True,
        "forced_stop_marked_sufficient": False,
        "silent_resource_omission": False,
        "silent_subject_omission": False,
        "repeated_query_count": 0,
        "weighted_covered": (float(measurement["coverage"]) * requirement_weight_total),
        "weighted_total": requirement_weight_total,
        "required_gap_count": measurement["gaps"],
        "selected_evidence_count": 100,
        "correct_evidence_count": int(float(measurement["precision"]) * 100),
        "premature_stop": False,
        "over_search": False,
        "source_route_true_positive": expected_route_count,
        "source_route_false_positive": 0,
        "source_route_false_negative": 0,
        "expected_resource_subject_count": len(case.targets),
        "assigned_resource_subject_count": len(case.targets),
        "correct_resource_subject_count": len(case.targets),
        "retrieval_cost_units": measurement["cost"],
        "latency_ms": measurement["latency"],
    }
    values.update(overrides)
    return EvidenceVariantObservationV2.model_validate(values)


def _signed_batch(
    *,
    attempts: list[EvidenceVariantAttemptV2],
    execution_mode: str = "hermetic",
    executor_fingerprint: str = EXECUTOR_FINGERPRINT,
) -> EvidenceVariantAttemptBatchV2:
    content = EvidenceVariantAttemptBatchContentV2(
        schema_version="evidence_variant_attempt_batch_v2",
        execution_mode=execution_mode,
        executor_fingerprint=executor_fingerprint,
        attempts=attempts,
    )
    return EvidenceVariantAttemptBatchV2(
        **content.model_dump(mode="python"),
        bundle_fingerprint=canonical_sha256(content.model_dump(mode="json")),
    )


def _signed_reviews(
    *,
    dataset: EvidenceEvaluationDatasetV2,
    binding: EvidenceEvaluationRuntimeBindingV2,
    attempts: list[EvidenceVariantAttemptV2],
) -> HumanSemanticReviewBatchV2:
    reviews: list[HumanSemanticReviewV2] = []
    for attempt in attempts:
        observation = attempt.observation
        if observation is None:
            raise AssertionError("review fixture requires successful observations")
        measurement = _MEASUREMENTS[(attempt.case_id, attempt.variant)]
        reviews.append(
            HumanSemanticReviewV2(
                schema_version="human_semantic_review_v2",
                case_id=attempt.case_id,
                variant=attempt.variant,
                output_fingerprint=observation.output_fingerprint,
                reviewer_identity_hash=REVIEWER_IDENTITY_HASH,
                reviewed_at="2026-07-15T10:00:00+00:00",
                assessment_source="human",
                supported_claim_count=int(float(measurement["claim_support"]) * 100),
                claim_count=100,
                ungrounded_fact_count=int(float(measurement["ungrounded"]) * 100),
                fact_count=100,
            )
        )
    content = HumanSemanticReviewBatchContentV2(
        schema_version="human_semantic_review_batch_v2",
        dataset_fingerprint=dataset.dataset_fingerprint,
        runtime_fingerprint=binding.runtime_fingerprint,
        generation_id=binding.generation_id,
        generation_manifest_fingerprint=binding.generation_manifest_fingerprint,
        review_protocol_fingerprint=REVIEW_PROTOCOL_FINGERPRINT,
        reviews=reviews,
    )
    return HumanSemanticReviewBatchV2(
        **content.model_dump(mode="python"),
        review_bundle_fingerprint=canonical_sha256(content.model_dump(mode="json")),
    )


def _scenario(*, simple_query: str = "Explain a Python iterator.") -> _Scenario:
    dataset = _dataset(simple_query=simple_query)
    execution_config = _execution_config()
    benchmark_config = load_evidence_benchmark_config(BENCHMARK_CONFIG_PATH)
    rollout_config = load_rag_rollout_config(ROLLOUT_CONFIG_PATH)
    binding = _binding(
        dataset=dataset,
        execution_config=execution_config,
        benchmark_config=benchmark_config,
        rollout_config=rollout_config,
    )
    attempts = [
        EvidenceVariantAttemptV2(
            schema_version="evidence_variant_attempt_v2",
            case_id=case.case_id,
            variant=definition.variant,
            status="success",
            observation=_observation(
                case=case,
                definition=definition,
                binding=binding,
            ),
            failure_reason_code=None,
            failure_type=None,
        )
        for case in dataset.cases
        for definition in execution_config.variants
    ]
    return _Scenario(
        dataset=dataset,
        knowledge_graph=KNOWLEDGE_GRAPH,
        execution_config=execution_config,
        benchmark_config=benchmark_config,
        rollout_config=rollout_config,
        binding=binding,
        batch=_signed_batch(attempts=attempts),
        reviews=_signed_reviews(
            dataset=dataset,
            binding=binding,
            attempts=attempts,
        ),
    )


def _resign_reviews(
    scenario: _Scenario,
    reviews: list[HumanSemanticReviewV2],
) -> HumanSemanticReviewBatchV2:
    content = HumanSemanticReviewBatchContentV2(
        schema_version="human_semantic_review_batch_v2",
        dataset_fingerprint=scenario.dataset.dataset_fingerprint,
        runtime_fingerprint=scenario.binding.runtime_fingerprint,
        generation_id=scenario.binding.generation_id,
        generation_manifest_fingerprint=(
            scenario.binding.generation_manifest_fingerprint
        ),
        review_protocol_fingerprint=REVIEW_PROTOCOL_FINGERPRINT,
        reviews=reviews,
    )
    return HumanSemanticReviewBatchV2(
        **content.model_dump(mode="python"),
        review_bundle_fingerprint=canonical_sha256(content.model_dump(mode="json")),
    )


async def _run(
    scenario: _Scenario,
    *,
    dataset: EvidenceEvaluationDatasetV2 | None = None,
    knowledge_graph: KnowledgeGraphV1 | None = None,
    executor: object | None = None,
    binding: EvidenceEvaluationRuntimeBindingV2 | None = None,
    reviews: HumanSemanticReviewBatchV2 | None = None,
):
    return await run_evidence_rollout_evaluation(
        dataset=scenario.dataset if dataset is None else dataset,
        knowledge_graph=(
            scenario.knowledge_graph if knowledge_graph is None else knowledge_graph
        ),
        execution_config=scenario.execution_config,
        benchmark_config=scenario.benchmark_config,
        rollout_config=scenario.rollout_config,
        binding=scenario.binding if binding is None else binding,
        reviews=scenario.reviews if reviews is None else reviews,
        executor=(
            SealedAttemptVariantExecutor(scenario.batch)
            if executor is None
            else executor
        ),
    )


def _resign_dataset_values(values: dict[str, object]) -> EvidenceEvaluationDatasetV2:
    values.pop("dataset_fingerprint", None)
    content = EvidenceEvaluationDatasetContentV2.model_validate(values)
    return EvidenceEvaluationDatasetV2(
        **content.model_dump(mode="python"),
        dataset_fingerprint=canonical_sha256(content.model_dump(mode="json")),
    )


def _mutate_first_target(
    dataset: EvidenceEvaluationDatasetV2,
    **changes: object,
) -> EvidenceEvaluationDatasetV2:
    values = dataset.model_dump(mode="python")
    case = values["cases"][0]
    target = case["targets"][0]
    target.update(changes)
    target.pop("target_fingerprint")
    target["target_fingerprint"] = canonical_sha256(target)
    case.pop("case_fingerprint")
    case["case_fingerprint"] = canonical_sha256(case)
    return _resign_dataset_values(values)


def _replace_first_observation(
    scenario: _Scenario,
    **overrides: object,
) -> EvidenceVariantAttemptBatchV2:
    attempts = list(scenario.batch.attempts)
    first = attempts[0]
    if first.observation is None:
        raise AssertionError("fixture first attempt must be successful")
    values = first.observation.model_dump(mode="python")
    values.update(overrides)
    observation = EvidenceVariantObservationV2.model_validate(values)
    attempts[0] = EvidenceVariantAttemptV2(
        schema_version="evidence_variant_attempt_v2",
        case_id=first.case_id,
        variant=first.variant,
        status="success",
        observation=observation,
        failure_reason_code=None,
        failure_type=None,
    )
    return _signed_batch(attempts=attempts)


class _RaisingExecutor:
    def __init__(
        self,
        *,
        declared_slots: frozenset[tuple[str, Variant]],
        failure: Callable[[], NoReturn],
    ) -> None:
        self._declared_slots = declared_slots
        self._failure = failure

    @property
    def execution_mode(self) -> str:
        return "hermetic"

    @property
    def executor_fingerprint(self) -> str:
        return EXECUTOR_FINGERPRINT

    @property
    def declared_slots(self) -> frozenset[tuple[str, Variant]]:
        return self._declared_slots

    async def execute(self, **_: object) -> EvidenceVariantAttemptV2:
        self._failure()


class _NeverCalledLiveAdapter:
    def __init__(
        self,
        *,
        identity: EvidenceLiveAdapterIdentityV2,
    ) -> None:
        self._identity = identity

    @property
    def identity(self) -> EvidenceLiveAdapterIdentityV2:
        return self._identity

    async def execute(self, **_: object) -> EvidenceVariantAttemptV2:
        raise AssertionError("incomplete live adapter inventory must not execute")


class _RebindingFixtureLiveAdapter:
    def __init__(
        self,
        *,
        identity: EvidenceLiveAdapterIdentityV2,
        attempts: dict[str, EvidenceVariantAttemptV2],
    ) -> None:
        self._identity = identity
        self._attempts = attempts

    @property
    def identity(self) -> EvidenceLiveAdapterIdentityV2:
        return self._identity

    async def execute(
        self,
        *,
        case: EvidenceEvaluationCaseSpecV2,
        binding: EvidenceEvaluationRuntimeBindingV2,
    ) -> EvidenceVariantAttemptV2:
        source = self._attempts[case.case_id]
        if source.observation is None:
            raise AssertionError("fixture adapter requires a successful observation")
        values = source.observation.model_dump(mode="python")
        values["executor_fingerprint"] = binding.executor_fingerprint
        observation = EvidenceVariantObservationV2.model_validate(values)
        return EvidenceVariantAttemptV2(
            schema_version="evidence_variant_attempt_v2",
            case_id=case.case_id,
            variant=self._identity.variant,
            status="success",
            observation=observation,
            failure_reason_code=None,
            failure_type=None,
        )


def _schema_failure() -> NoReturn:
    EvidenceVariantAttemptV2.model_validate(
        {
            "schema_version": "invalid",
            "raw_provider_body": "SECRET_PROVIDER_BODY",
        }
    )
    raise AssertionError("invalid attempt unexpectedly passed validation")


def _business_failure() -> NoReturn:
    raise EvidenceEvaluationError(
        code="private_business_code",
        reason="SECRET_EVIDENCE_BODY https://private.example.invalid",
    )


def _unexpected_failure() -> NoReturn:
    raise RuntimeError(
        "SECRET_QUERY https://private.example.invalid SECRET_PROVIDER_BODY"
    )


async def test_complete_four_variant_hermetic_run_is_never_activation_allowed() -> None:
    scenario = _scenario()

    decision = await _run(scenario)

    assert decision.status == "blocked"
    assert decision.reason_codes == [
        "non_live_execution",
        "rollout_activation_disabled",
    ]
    assert decision.benchmark_eligible is True
    assert decision.activation_allowed is False
    assert decision.variant_matrix_complete is True
    assert decision.expected_execution_count == 8
    assert decision.successful_execution_count == 8
    assert decision.reviewed_execution_count == 8
    assert len(decision.execution_records) == 8
    assert all(record.status == "success" for record in decision.execution_records)


def test_sealed_attempt_bundle_cannot_claim_live_execution() -> None:
    scenario = _scenario()

    with pytest.raises(ValueError, match="hermetic"):
        live_content = EvidenceVariantAttemptBatchContentV2(
            schema_version="evidence_variant_attempt_batch_v2",
            execution_mode="live",
            executor_fingerprint=scenario.batch.executor_fingerprint,
            attempts=list(scenario.batch.attempts),
        )
        live_batch = EvidenceVariantAttemptBatchV2(
            **live_content.model_dump(mode="python"),
            bundle_fingerprint=canonical_sha256(live_content.model_dump(mode="json")),
        )
        SealedAttemptVariantExecutor(live_batch)


async def test_missing_human_review_blocks_and_fail_fast_covers_remaining_slots() -> (
    None
):
    scenario = _scenario()
    reviews = _resign_reviews(scenario, list(scenario.reviews.reviews[1:]))

    decision = await _run(scenario, reviews=reviews)

    assert decision.status == "blocked"
    assert decision.reason_codes == [
        "human_semantic_review_inventory_mismatch",
        "non_live_execution",
        "rollout_activation_disabled",
    ]
    assert decision.successful_execution_count == 0
    assert decision.reviewed_execution_count == 0
    assert len(decision.execution_records) == decision.expected_execution_count
    assert all(
        record.status == "not_executed"
        and record.failure_reason_code == "evaluation_binding_invalid"
        for record in decision.execution_records
    )


async def test_failed_attempt_blocks_and_fail_fast_covers_every_remaining_slot() -> (
    None
):
    scenario = _scenario()
    attempts = list(scenario.batch.attempts)
    first = attempts[0]
    attempts[0] = EvidenceVariantAttemptV2(
        schema_version="evidence_variant_attempt_v2",
        case_id=first.case_id,
        variant=first.variant,
        status="failed",
        observation=None,
        failure_reason_code="hermetic_attempt_failed",
        failure_type="HermeticAttemptError",
    )
    executor = SealedAttemptVariantExecutor(_signed_batch(attempts=attempts))

    decision = await _run(scenario, executor=executor)

    assert decision.status == "blocked"
    assert "variant_execution_incomplete" in decision.reason_codes
    assert "factorial_execution_incomplete" in decision.reason_codes
    assert decision.successful_execution_count == 0
    assert len(decision.execution_records) == decision.expected_execution_count
    assert decision.execution_records[0].status == "failed"
    assert (
        decision.execution_records[0].failure_reason_code == "hermetic_attempt_failed"
    )
    assert all(
        record.status == "not_executed"
        and record.failure_reason_code == "not_executed_after_fail_fast"
        for record in decision.execution_records[1:]
    )


async def test_runtime_binding_mismatch_blocks_before_any_execution() -> None:
    scenario = _scenario()
    mismatched_binding = _binding(
        dataset=scenario.dataset,
        execution_config=scenario.execution_config,
        benchmark_config=scenario.benchmark_config,
        rollout_config=scenario.rollout_config,
        executor_fingerprint="f" * 64,
    )

    decision = await _run(scenario, binding=mismatched_binding)

    assert decision.status == "blocked"
    assert "executor_fingerprint_mismatch" in decision.reason_codes
    assert decision.successful_execution_count == 0
    assert all(
        record.status == "not_executed"
        and record.failure_reason_code == "evaluation_binding_invalid"
        for record in decision.execution_records
    )


@pytest.mark.parametrize(
    ("dataset_factory", "reason_code"),
    [
        (
            lambda dataset: _resign_dataset_values(
                {
                    **dataset.model_dump(mode="python"),
                    "knowledge_graph_data_version": "wrong-data-version",
                }
            ),
            "knowledge_graph_data_version_mismatch",
        ),
        (
            lambda dataset: _mutate_first_target(
                dataset,
                topic_id="unknown.topic",
            ),
            "target_topic_unknown",
        ),
        (
            lambda dataset: _mutate_first_target(
                dataset,
                topic_id="big_data.data_engineering",
                catalog_resource_ids=["big_data_data_engineering_zoomcamp"],
            ),
            "target_topic_subject_mismatch",
        ),
        (
            lambda dataset: _mutate_first_target(
                dataset,
                catalog_resource_ids=["unknown_resource"],
            ),
            "target_catalog_resource_unknown",
        ),
        (
            lambda dataset: _mutate_first_target(
                dataset,
                catalog_resource_ids=[
                    "python_for_everybody",
                    "python_basics_real_python",
                ],
            ),
            "target_catalog_resource_order_mismatch",
        ),
    ],
)
async def test_knowledge_graph_binding_drift_blocks_before_execution(
    dataset_factory,
    reason_code: str,
) -> None:
    scenario = _scenario()
    drifted = dataset_factory(scenario.dataset)

    decision = await _run(scenario, dataset=drifted)

    assert decision.status == "blocked"
    assert reason_code in decision.reason_codes
    assert decision.successful_execution_count == 0
    assert all(record.status == "not_executed" for record in decision.execution_records)


async def test_runtime_and_observation_case_target_identity_drift_are_blocked() -> None:
    scenario = _scenario()
    binding_values = scenario.binding.model_dump(mode="python")
    binding_values["case_bindings"] = list(reversed(binding_values["case_bindings"]))
    drifted_binding = EvidenceEvaluationRuntimeBindingV2.model_validate(binding_values)

    binding_decision = await _run(scenario, binding=drifted_binding)

    assert "binding_case_target_inventory_mismatch" in binding_decision.reason_codes
    assert binding_decision.successful_execution_count == 0

    observation_batch = _replace_first_observation(
        scenario,
        case_binding=case_binding_identity(scenario.dataset.cases[1]),
    )
    observation_decision = await _run(
        scenario,
        executor=SealedAttemptVariantExecutor(observation_batch),
    )

    assert "observation_case_binding_mismatch" in observation_decision.reason_codes
    assert observation_decision.execution_records[0].status == "blocked"


async def test_missing_pg_and_pr_live_adapters_form_machine_readable_blocker() -> None:
    scenario = _scenario()
    declared_cases = dataset_case_bindings(scenario.dataset)

    def adapter_for(variant: Variant) -> _NeverCalledLiveAdapter:
        definition = scenario.execution_config.definition_for(variant)
        return _NeverCalledLiveAdapter(
            identity=EvidenceLiveAdapterIdentityV2(
                schema_version="evidence_live_adapter_identity_v2",
                variant=variant,
                resource_planning_enabled=(definition.resource_planning_enabled),
                bounded_repair_enabled=definition.bounded_repair_enabled,
                adapter_fingerprint=canonical_sha256(
                    {"adapter": variant, "fixture": True}
                ),
                dataset_id=scenario.dataset.dataset_id,
                dataset_fingerprint=scenario.dataset.dataset_fingerprint,
                knowledge_graph_data_version=(
                    scenario.dataset.knowledge_graph_data_version
                ),
                knowledge_graph_artifact_fingerprint=(
                    scenario.dataset.knowledge_graph_artifact_fingerprint
                ),
                declared_cases=declared_cases,
            )
        )

    executor = LiveEvidenceVariantExecutor(
        dataset=scenario.dataset,
        knowledge_graph=scenario.knowledge_graph,
        execution_config=scenario.execution_config,
        adapters=[adapter_for("P0"), adapter_for("PGR")],
    )
    binding_values = scenario.binding.model_dump(mode="python")
    binding_values["execution_mode"] = "live"
    binding_values["executor_fingerprint"] = executor.executor_fingerprint
    binding = EvidenceEvaluationRuntimeBindingV2.model_validate(binding_values)

    decision = await _run(scenario, executor=executor, binding=binding)

    assert decision.status == "blocked"
    assert "executor_variant_inventory_mismatch" in decision.reason_codes
    assert "live_variant_adapter_missing_pg" in decision.reason_codes
    assert "live_variant_adapter_missing_pr" in decision.reason_codes
    assert "rollout_activation_disabled" in decision.reason_codes
    assert decision.successful_execution_count == 0
    assert all(record.status == "not_executed" for record in decision.execution_records)


async def test_complete_live_protocol_is_blocked_while_rollout_is_disabled() -> None:
    scenario = _scenario()
    declared_cases = dataset_case_bindings(scenario.dataset)
    adapters = []
    for definition in scenario.execution_config.variants:
        variant_attempts = {
            attempt.case_id: attempt
            for attempt in scenario.batch.attempts
            if attempt.variant == definition.variant
        }
        adapters.append(
            _RebindingFixtureLiveAdapter(
                identity=EvidenceLiveAdapterIdentityV2(
                    schema_version="evidence_live_adapter_identity_v2",
                    variant=definition.variant,
                    resource_planning_enabled=(definition.resource_planning_enabled),
                    bounded_repair_enabled=definition.bounded_repair_enabled,
                    adapter_fingerprint=canonical_sha256(
                        {
                            "adapter": definition.variant,
                            "fixture": "live_protocol_only",
                        }
                    ),
                    dataset_id=scenario.dataset.dataset_id,
                    dataset_fingerprint=scenario.dataset.dataset_fingerprint,
                    knowledge_graph_data_version=(
                        scenario.dataset.knowledge_graph_data_version
                    ),
                    knowledge_graph_artifact_fingerprint=(
                        scenario.dataset.knowledge_graph_artifact_fingerprint
                    ),
                    declared_cases=declared_cases,
                ),
                attempts=variant_attempts,
            )
        )
    executor = LiveEvidenceVariantExecutor(
        dataset=scenario.dataset,
        knowledge_graph=scenario.knowledge_graph,
        execution_config=scenario.execution_config,
        adapters=adapters,
    )
    binding_values = scenario.binding.model_dump(mode="python")
    binding_values["execution_mode"] = "live"
    binding_values["executor_fingerprint"] = executor.executor_fingerprint
    binding = EvidenceEvaluationRuntimeBindingV2.model_validate(binding_values)

    decision = await _run(scenario, executor=executor, binding=binding)

    assert decision.status == "blocked"
    assert decision.reason_codes == ["rollout_activation_disabled"]
    assert decision.benchmark_eligible is True
    assert decision.activation_allowed is False
    assert decision.variant_matrix_complete is True


@pytest.mark.parametrize(
    ("status_field", "reason_code"),
    [
        ("provider_status", "provider_execution_failed"),
        ("parent_child_status", "parent_child_execution_failed"),
        ("web_status", "web_execution_failed"),
    ],
)
async def test_component_failure_is_blocked_without_executing_remaining_slots(
    status_field: str,
    reason_code: str,
) -> None:
    scenario = _scenario()
    executor = SealedAttemptVariantExecutor(
        _replace_first_observation(scenario, **{status_field: "failed"})
    )

    decision = await _run(scenario, executor=executor)

    assert decision.status == "blocked"
    assert reason_code in decision.reason_codes
    assert "factorial_execution_incomplete" in decision.reason_codes
    assert decision.execution_records[0].status == "blocked"
    assert decision.execution_records[0].failure_reason_code == reason_code
    assert all(
        record.status == "not_executed" for record in decision.execution_records[1:]
    )


@pytest.mark.parametrize(
    ("overrides", "reason_code"),
    [
        (
            {"weighted_total": 2.0},
            "observation_requirement_weight_total_mismatch",
        ),
        (
            {"required_gap_count": 2},
            "observation_required_gap_count_invalid",
        ),
        (
            {"expected_resource_subject_count": 2},
            "observation_resource_subject_inventory_mismatch",
        ),
        (
            {"source_route_true_positive": 2},
            "observation_source_route_inventory_mismatch",
        ),
    ],
)
async def test_observation_must_bind_authored_gold_inventory(
    overrides: dict[str, object],
    reason_code: str,
) -> None:
    scenario = _scenario()
    executor = SealedAttemptVariantExecutor(
        _replace_first_observation(scenario, **overrides)
    )

    decision = await _run(scenario, executor=executor)

    assert decision.status == "blocked"
    assert reason_code in decision.reason_codes
    assert decision.execution_records[0].status == "blocked"
    assert decision.execution_records[0].failure_reason_code == reason_code
    assert all(
        record.status == "not_executed" for record in decision.execution_records[1:]
    )


@pytest.mark.parametrize(
    ("failure", "reason_code"),
    [
        (_schema_failure, "schema_validation_failed"),
        (_business_failure, "business_validation_failed"),
    ],
)
async def test_schema_and_business_validation_failures_are_blocked(
    failure: Callable[[], NoReturn],
    reason_code: str,
) -> None:
    scenario = _scenario()
    executor = _RaisingExecutor(
        declared_slots=frozenset(
            (attempt.case_id, attempt.variant) for attempt in scenario.batch.attempts
        ),
        failure=failure,
    )

    decision = await _run(scenario, executor=executor)

    assert decision.status == "blocked"
    assert "variant_execution_incomplete" in decision.reason_codes
    assert decision.execution_records[0].status == "failed"
    assert decision.execution_records[0].failure_reason_code == reason_code
    assert all(
        record.status == "not_executed" for record in decision.execution_records[1:]
    )


async def test_human_review_output_binding_mismatch_is_blocked() -> None:
    scenario = _scenario()
    reviews = list(scenario.reviews.reviews)
    first = reviews[0]
    values = first.model_dump(mode="python")
    values["output_fingerprint"] = "e" * 64
    reviews[0] = HumanSemanticReviewV2.model_validate(values)

    decision = await _run(scenario, reviews=_resign_reviews(scenario, reviews))

    assert decision.status == "blocked"
    assert "human_semantic_review_invalid" in decision.reason_codes
    assert "factorial_execution_incomplete" in decision.reason_codes
    assert decision.reviewed_execution_count == 0
    assert decision.execution_records[0].status == "success"
    assert all(
        record.status == "not_executed" for record in decision.execution_records[1:]
    )


async def test_decision_never_leaks_query_url_evidence_or_provider_body() -> None:
    sensitive_query = (
        "SECRET_QUERY https://private.example.invalid "
        "SECRET_EVIDENCE_BODY SECRET_PROVIDER_BODY api_key=forbidden"
    )
    scenario = _scenario(simple_query=sensitive_query)
    executor = _RaisingExecutor(
        declared_slots=frozenset(
            (attempt.case_id, attempt.variant) for attempt in scenario.batch.attempts
        ),
        failure=_unexpected_failure,
    )

    decision = await _run(scenario, executor=executor)
    serialized = decision.model_dump_json().casefold()

    assert decision.status == "blocked"
    assert decision.execution_records[0].failure_type == "RuntimeError"
    for forbidden in (
        sensitive_query.casefold(),
        "secret_query",
        "https://",
        "secret_evidence_body",
        "secret_provider_body",
        "api_key",
        "authorization",
        "raw_provider_body",
    ):
        assert forbidden not in serialized
