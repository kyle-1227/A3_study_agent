"""Fail-closed orchestration for the canonical P0/PG/PR/PGR matrix."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Protocol

from pydantic import ValidationError

from src.config.evidence_benchmark_config import EvidenceBenchmarkConfig
from src.config.rag_rollout_config import RagRolloutConfig
from src.evaluation.evidence_rollout.contracts import (
    DecisionStatus,
    EvidenceEvaluationCaseSpecV1,
    EvidenceEvaluationDatasetV1,
    EvidenceEvaluationRuntimeBindingV1,
    EvidenceExecutionRecordV1,
    EvidenceLiveAdapterIdentityV1,
    EvidenceRolloutDecisionContentV1,
    EvidenceRolloutDecisionV1,
    EvidenceRolloutExecutionConfigV1,
    EvidenceVariantAttemptBatchV1,
    EvidenceVariantAttemptV1,
    EvidenceVariantDefinitionV1,
    EvidenceVariantObservationV1,
    ExecutionMode,
    HumanSemanticReviewBatchV1,
    HumanSemanticReviewV1,
    canonical_sha256,
    model_fingerprint,
    query_fingerprint,
)
from src.rag.parent_child.evidence_evaluation import (
    EvidenceActivationDecision,
    EvidenceEvaluationCaseResult,
    EvidenceEvaluationError,
    Variant,
    evaluate_evidence_activation,
)


Slot = tuple[str, Variant]


class EvidenceVariantExecutor(Protocol):
    """Explicit execution boundary; implementations may not invent missing slots."""

    @property
    def execution_mode(self) -> ExecutionMode: ...

    @property
    def executor_fingerprint(self) -> str: ...

    @property
    def declared_slots(self) -> frozenset[Slot]: ...

    async def execute(
        self,
        *,
        case: EvidenceEvaluationCaseSpecV1,
        definition: EvidenceVariantDefinitionV1,
        binding: EvidenceEvaluationRuntimeBindingV1,
    ) -> EvidenceVariantAttemptV1: ...


class LiveEvidenceVariantAdapter(Protocol):
    """One concrete live implementation of exactly one factorial variant."""

    @property
    def identity(self) -> EvidenceLiveAdapterIdentityV1: ...

    async def execute(
        self,
        *,
        case: EvidenceEvaluationCaseSpecV1,
        binding: EvidenceEvaluationRuntimeBindingV1,
    ) -> EvidenceVariantAttemptV1: ...


class LiveEvidenceVariantExecutor:
    """Compose four explicit live adapters without aliases or missing-slot repair."""

    def __init__(
        self,
        *,
        dataset: EvidenceEvaluationDatasetV1,
        execution_config: EvidenceRolloutExecutionConfigV1,
        adapters: Sequence[LiveEvidenceVariantAdapter],
    ) -> None:
        if not isinstance(dataset, EvidenceEvaluationDatasetV1):
            raise TypeError("dataset must be EvidenceEvaluationDatasetV1")
        if not isinstance(execution_config, EvidenceRolloutExecutionConfigV1):
            raise TypeError("execution_config must be EvidenceRolloutExecutionConfigV1")
        expected_case_ids = [case.case_id for case in dataset.cases]
        by_variant: dict[Variant, LiveEvidenceVariantAdapter] = {}
        identity_by_variant: dict[Variant, EvidenceLiveAdapterIdentityV1] = {}
        for adapter in adapters:
            identity = adapter.identity
            if not isinstance(identity, EvidenceLiveAdapterIdentityV1):
                raise TypeError(
                    "live adapter identity must be EvidenceLiveAdapterIdentityV1"
                )
            definition = execution_config.definition_for(identity.variant)
            if (
                identity.variant != definition.variant
                or identity.resource_planning_enabled
                != definition.resource_planning_enabled
                or identity.bounded_repair_enabled != definition.bounded_repair_enabled
            ):
                raise EvidenceEvaluationError(
                    code="live_variant_adapter_semantics_mismatch",
                    reason="live adapter identity differs from canonical variant config",
                )
            if identity.declared_case_ids != expected_case_ids:
                raise EvidenceEvaluationError(
                    code="live_variant_adapter_case_inventory_mismatch",
                    reason="live adapter must declare the complete ordered case inventory",
                )
            if identity.variant in by_variant:
                raise EvidenceEvaluationError(
                    code="duplicate_live_variant_adapter",
                    reason="each live variant may have exactly one adapter",
                )
            by_variant[identity.variant] = adapter
            identity_by_variant[identity.variant] = identity

        identities = [
            identity_by_variant[definition.variant]
            for definition in execution_config.variants
            if definition.variant in identity_by_variant
        ]
        self._missing_variants = tuple(
            definition.variant
            for definition in execution_config.variants
            if definition.variant not in by_variant
        )

        self._dataset_fingerprint = dataset.dataset_fingerprint
        self._execution_config_fingerprint = model_fingerprint(execution_config)
        self._adapters = by_variant
        self._declared_slots = frozenset(
            (case_id, identity.variant)
            for identity in identities
            for case_id in identity.declared_case_ids
        )
        self._executor_fingerprint = canonical_sha256(
            {
                "schema_version": "live_evidence_variant_executor_v1",
                "dataset_fingerprint": self._dataset_fingerprint,
                "execution_config_fingerprint": self._execution_config_fingerprint,
                "adapters": [
                    identity.model_dump(mode="json") for identity in identities
                ],
            }
        )

    @property
    def execution_mode(self) -> ExecutionMode:
        return "live"

    @property
    def executor_fingerprint(self) -> str:
        return self._executor_fingerprint

    @property
    def declared_slots(self) -> frozenset[Slot]:
        return self._declared_slots

    @property
    def missing_variants(self) -> tuple[Variant, ...]:
        return self._missing_variants

    async def execute(
        self,
        *,
        case: EvidenceEvaluationCaseSpecV1,
        definition: EvidenceVariantDefinitionV1,
        binding: EvidenceEvaluationRuntimeBindingV1,
    ) -> EvidenceVariantAttemptV1:
        if binding.execution_mode != "live":
            raise EvidenceEvaluationError(
                code="live_executor_mode_mismatch",
                reason="live executor requires an explicitly live runtime binding",
            )
        if binding.dataset_fingerprint != self._dataset_fingerprint:
            raise EvidenceEvaluationError(
                code="live_executor_dataset_mismatch",
                reason="runtime binding differs from the live executor dataset",
            )
        if binding.execution_config_fingerprint != self._execution_config_fingerprint:
            raise EvidenceEvaluationError(
                code="live_executor_config_mismatch",
                reason="runtime binding differs from the live executor config",
            )
        adapter = self._adapters.get(definition.variant)
        if adapter is None:
            raise EvidenceEvaluationError(
                code="live_variant_adapter_unavailable",
                reason="requested live variant has no explicit adapter",
            )
        return await adapter.execute(case=case, binding=binding)


class SealedAttemptVariantExecutor:
    """Replay a fingerprinted attempt bundle without upgrading it to live proof."""

    def __init__(self, batch: EvidenceVariantAttemptBatchV1) -> None:
        if not isinstance(batch, EvidenceVariantAttemptBatchV1):
            raise TypeError("batch must be EvidenceVariantAttemptBatchV1")
        if batch.execution_mode != "hermetic":
            raise ValueError("sealed attempt bundles are hermetic-only")
        self._batch = batch
        self._attempts = {
            (attempt.case_id, attempt.variant): attempt for attempt in batch.attempts
        }

    @property
    def execution_mode(self) -> ExecutionMode:
        return "hermetic"

    @property
    def executor_fingerprint(self) -> str:
        return self._batch.executor_fingerprint

    @property
    def declared_slots(self) -> frozenset[Slot]:
        return frozenset(self._attempts)

    async def execute(
        self,
        *,
        case: EvidenceEvaluationCaseSpecV1,
        definition: EvidenceVariantDefinitionV1,
        binding: EvidenceEvaluationRuntimeBindingV1,
    ) -> EvidenceVariantAttemptV1:
        del binding
        attempt = self._attempts.get((case.case_id, definition.variant))
        if attempt is None:
            return EvidenceVariantAttemptV1(
                schema_version="evidence_variant_attempt_v1",
                case_id=case.case_id,
                variant=definition.variant,
                status="blocked",
                observation=None,
                failure_reason_code="missing_variant_attempt",
                failure_type="AttemptInventoryError",
            )
        return attempt


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _safe_exception_type(error: BaseException) -> str:
    value = type(error).__name__
    if not value or not value.replace("_", "").isalnum():
        return "UnexpectedException"
    return value[:200]


def _not_executed_record(
    *,
    case_id: str,
    variant: Variant,
    reason_code: str,
) -> EvidenceExecutionRecordV1:
    return EvidenceExecutionRecordV1(
        schema_version="evidence_execution_record_v1",
        case_id=case_id,
        variant=variant,
        status="not_executed",
        output_fingerprint=None,
        failure_reason_code=reason_code,
        failure_type="FailClosedExecutionStop",
    )


def _attempt_record(attempt: EvidenceVariantAttemptV1) -> EvidenceExecutionRecordV1:
    if attempt.status == "success":
        if attempt.observation is None:
            raise AssertionError("validated successful attempt requires observation")
        return EvidenceExecutionRecordV1(
            schema_version="evidence_execution_record_v1",
            case_id=attempt.case_id,
            variant=attempt.variant,
            status="success",
            output_fingerprint=attempt.observation.output_fingerprint,
            failure_reason_code=None,
            failure_type=None,
        )
    return EvidenceExecutionRecordV1(
        schema_version="evidence_execution_record_v1",
        case_id=attempt.case_id,
        variant=attempt.variant,
        status=attempt.status,
        output_fingerprint=None,
        failure_reason_code=attempt.failure_reason_code,
        failure_type=attempt.failure_type,
    )


def _blocked_records(
    slots: Sequence[Slot],
    *,
    reason_code: str,
) -> list[EvidenceExecutionRecordV1]:
    return [
        _not_executed_record(
            case_id=case_id,
            variant=variant,
            reason_code=reason_code,
        )
        for case_id, variant in slots
    ]


def _sign_decision(
    content: EvidenceRolloutDecisionContentV1,
) -> EvidenceRolloutDecisionV1:
    payload = content.model_dump(mode="json")
    return EvidenceRolloutDecisionV1(
        **content.model_dump(mode="python"),
        decision_fingerprint=canonical_sha256(payload),
    )


def _build_decision(
    *,
    binding: EvidenceEvaluationRuntimeBindingV1,
    reviews: HumanSemanticReviewBatchV1,
    status: DecisionStatus,
    benchmark_eligible: bool,
    rollout_activation_enabled: bool,
    reason_codes: Sequence[str],
    expected_execution_count: int,
    successful_execution_count: int,
    reviewed_execution_count: int,
    activation_decision: EvidenceActivationDecision | None,
    case_results: Sequence[EvidenceEvaluationCaseResult],
    records: Sequence[EvidenceExecutionRecordV1],
) -> EvidenceRolloutDecisionV1:
    effective_reason_codes = list(reason_codes)
    if status == "blocked" and binding.execution_mode != "live":
        effective_reason_codes.append("non_live_execution")
    if status == "blocked" and not rollout_activation_enabled:
        effective_reason_codes.append("rollout_activation_disabled")
    complete = (
        successful_execution_count == expected_execution_count
        and reviewed_execution_count == expected_execution_count
        and len(case_results) == expected_execution_count
    )
    content = EvidenceRolloutDecisionContentV1(
        schema_version="evidence_rollout_activation_decision_v1",
        run_id=binding.run_id,
        execution_mode=binding.execution_mode,
        status=status,
        benchmark_eligible=benchmark_eligible,
        activation_allowed=status == "pass",
        rollout_activation_enabled=rollout_activation_enabled,
        reason_codes=_unique(effective_reason_codes),
        expected_execution_count=expected_execution_count,
        successful_execution_count=successful_execution_count,
        reviewed_execution_count=reviewed_execution_count,
        variant_matrix_complete=complete,
        dataset_id=binding.dataset_id,
        dataset_fingerprint=binding.dataset_fingerprint,
        execution_config_fingerprint=binding.execution_config_fingerprint,
        benchmark_config_fingerprint=binding.benchmark_config_fingerprint,
        rollout_config_fingerprint=binding.rollout_config_fingerprint,
        runtime_fingerprint=binding.runtime_fingerprint,
        generation_id=binding.generation_id,
        generation_manifest_fingerprint=(binding.generation_manifest_fingerprint),
        executor_fingerprint=binding.executor_fingerprint,
        review_protocol_fingerprint=reviews.review_protocol_fingerprint,
        review_bundle_fingerprint=reviews.review_bundle_fingerprint,
        activation_decision=activation_decision,
        case_results=list(case_results),
        execution_records=list(records),
    )
    return _sign_decision(content)


def _binding_failure_reasons(
    *,
    dataset: EvidenceEvaluationDatasetV1,
    execution_config: EvidenceRolloutExecutionConfigV1,
    benchmark_config: EvidenceBenchmarkConfig,
    rollout_config: RagRolloutConfig,
    binding: EvidenceEvaluationRuntimeBindingV1,
    reviews: HumanSemanticReviewBatchV1,
    executor: EvidenceVariantExecutor,
    expected_slots: frozenset[Slot],
) -> list[str]:
    reasons: list[str] = []
    if len(dataset.cases) > execution_config.max_case_count:
        reasons.append("dataset_case_limit_exceeded")
    if binding.dataset_id != dataset.dataset_id:
        reasons.append("dataset_id_mismatch")
    if binding.dataset_fingerprint != dataset.dataset_fingerprint:
        reasons.append("dataset_fingerprint_mismatch")
    if binding.execution_config_fingerprint != model_fingerprint(execution_config):
        reasons.append("execution_config_fingerprint_mismatch")
    if binding.benchmark_config_fingerprint != model_fingerprint(benchmark_config):
        reasons.append("benchmark_config_fingerprint_mismatch")
    if binding.rollout_config_fingerprint != model_fingerprint(rollout_config):
        reasons.append("rollout_config_fingerprint_mismatch")
    if tuple(benchmark_config.required_variants) != tuple(
        item.variant for item in execution_config.variants
    ):
        reasons.append("benchmark_variant_contract_mismatch")
    if binding.execution_mode != executor.execution_mode:
        reasons.append("executor_mode_mismatch")
    if binding.executor_fingerprint != executor.executor_fingerprint:
        reasons.append("executor_fingerprint_mismatch")
    if executor.declared_slots != expected_slots:
        reasons.append("executor_variant_inventory_mismatch")
    if isinstance(executor, LiveEvidenceVariantExecutor):
        reasons.extend(
            f"live_variant_adapter_missing_{variant.casefold()}"
            for variant in executor.missing_variants
        )
    if reviews.dataset_fingerprint != dataset.dataset_fingerprint:
        reasons.append("review_dataset_fingerprint_mismatch")
    if reviews.runtime_fingerprint != binding.runtime_fingerprint:
        reasons.append("review_runtime_fingerprint_mismatch")
    if reviews.generation_id != binding.generation_id:
        reasons.append("review_generation_mismatch")
    if (
        reviews.generation_manifest_fingerprint
        != binding.generation_manifest_fingerprint
    ):
        reasons.append("review_generation_manifest_mismatch")
    if (
        reviews.review_protocol_fingerprint
        != execution_config.human_review_protocol_fingerprint
    ):
        reasons.append("review_protocol_fingerprint_mismatch")
    review_slots = frozenset(
        (review.case_id, review.variant) for review in reviews.reviews
    )
    if review_slots != expected_slots:
        reasons.append("human_semantic_review_inventory_mismatch")
    return _unique(reasons)


def _observation_binding_reasons(
    *,
    observation: EvidenceVariantObservationV1,
    case: EvidenceEvaluationCaseSpecV1,
    definition: EvidenceVariantDefinitionV1,
    binding: EvidenceEvaluationRuntimeBindingV1,
) -> list[str]:
    expected = {
        "case_id": case.case_id,
        "variant": definition.variant,
        "query_fingerprint": query_fingerprint(case.query),
        "dataset_fingerprint": binding.dataset_fingerprint,
        "execution_config_fingerprint": binding.execution_config_fingerprint,
        "benchmark_config_fingerprint": binding.benchmark_config_fingerprint,
        "rollout_config_fingerprint": binding.rollout_config_fingerprint,
        "runtime_fingerprint": binding.runtime_fingerprint,
        "generation_id": binding.generation_id,
        "generation_manifest_fingerprint": (binding.generation_manifest_fingerprint),
        "executor_fingerprint": binding.executor_fingerprint,
        "variant_definition_fingerprint": model_fingerprint(definition),
    }
    reasons = [
        f"observation_{field_name}_mismatch"
        for field_name, expected_value in expected.items()
        if getattr(observation, field_name) != expected_value
    ]
    if observation.provider_status != "ok":
        reasons.append("provider_execution_failed")
    if observation.parent_child_status != "ok":
        reasons.append("parent_child_execution_failed")
    if observation.web_status == "failed":
        reasons.append("web_execution_failed")
    expected_weight = sum(requirement.weight for requirement in case.requirements)
    if not math.isclose(
        observation.weighted_total,
        expected_weight,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        reasons.append("observation_requirement_weight_total_mismatch")
    if observation.required_gap_count > len(case.requirements):
        reasons.append("observation_required_gap_count_invalid")
    if observation.expected_resource_subject_count != len(case.targets):
        reasons.append("observation_resource_subject_inventory_mismatch")
    expected_route_count = sum(len(target.required_sources) for target in case.targets)
    if (
        observation.source_route_true_positive + observation.source_route_false_negative
        != expected_route_count
    ):
        reasons.append("observation_source_route_inventory_mismatch")
    return reasons


def _case_result(
    *,
    case: EvidenceEvaluationCaseSpecV1,
    observation: EvidenceVariantObservationV1,
    review: HumanSemanticReviewV1,
) -> EvidenceEvaluationCaseResult:
    if review.output_fingerprint != observation.output_fingerprint:
        raise EvidenceEvaluationError(
            code="human_review_output_mismatch",
            reason="human review must bind the exact variant output fingerprint",
        )
    source_denominator = (
        2 * observation.source_route_true_positive
        + observation.source_route_false_positive
        + observation.source_route_false_negative
    )
    source_routing_f1 = 2 * observation.source_route_true_positive / source_denominator
    evidence_precision = (
        observation.correct_evidence_count / observation.selected_evidence_count
        if observation.selected_evidence_count
        else 0.0
    )
    assignment_precision = (
        observation.correct_resource_subject_count
        / observation.assigned_resource_subject_count
        if observation.assigned_resource_subject_count
        else 0.0
    )
    return EvidenceEvaluationCaseResult(
        schema_version="evidence_evaluation_case_v1",
        case_id=case.case_id,
        variant=observation.variant,
        subject_count=len(case.subjects),
        resource_count=len(case.resource_types),
        initial_evidence_sufficient=case.initial_evidence_sufficient,
        bounded=observation.bounded,
        forced_stop_marked_sufficient=observation.forced_stop_marked_sufficient,
        silent_resource_omission=observation.silent_resource_omission,
        silent_subject_omission=observation.silent_subject_omission,
        repeated_query_count=observation.repeated_query_count,
        weighted_coverage=(observation.weighted_covered / observation.weighted_total),
        required_gap_count=observation.required_gap_count,
        evidence_precision=evidence_precision,
        premature_stop=observation.premature_stop,
        over_search=observation.over_search,
        source_routing_f1=source_routing_f1,
        resource_subject_recall=(
            observation.correct_resource_subject_count
            / observation.expected_resource_subject_count
        ),
        assignment_precision=assignment_precision,
        claim_support_rate=review.supported_claim_count / review.claim_count,
        ungrounded_fact_rate=review.ungrounded_fact_count / review.fact_count,
        retrieval_cost_units=observation.retrieval_cost_units,
        latency_ms=observation.latency_ms,
    )


async def run_evidence_rollout_evaluation(
    *,
    dataset: EvidenceEvaluationDatasetV1,
    execution_config: EvidenceRolloutExecutionConfigV1,
    benchmark_config: EvidenceBenchmarkConfig,
    rollout_config: RagRolloutConfig,
    binding: EvidenceEvaluationRuntimeBindingV1,
    reviews: HumanSemanticReviewBatchV1,
    executor: EvidenceVariantExecutor,
) -> EvidenceRolloutDecisionV1:
    """Execute every canonical slot, bind human reviews, then evaluate activation."""

    for value, expected_type, field_name in (
        (dataset, EvidenceEvaluationDatasetV1, "dataset"),
        (execution_config, EvidenceRolloutExecutionConfigV1, "execution_config"),
        (benchmark_config, EvidenceBenchmarkConfig, "benchmark_config"),
        (rollout_config, RagRolloutConfig, "rollout_config"),
        (binding, EvidenceEvaluationRuntimeBindingV1, "binding"),
        (reviews, HumanSemanticReviewBatchV1, "reviews"),
    ):
        if not isinstance(value, expected_type):
            raise TypeError(
                f"{field_name} must be a validated {expected_type.__name__}"
            )

    ordered_slots: list[Slot] = [
        (case.case_id, definition.variant)
        for case in dataset.cases
        for definition in execution_config.variants
    ]
    expected_slots = frozenset(ordered_slots)
    binding_reasons = _binding_failure_reasons(
        dataset=dataset,
        execution_config=execution_config,
        benchmark_config=benchmark_config,
        rollout_config=rollout_config,
        binding=binding,
        reviews=reviews,
        executor=executor,
        expected_slots=expected_slots,
    )
    if binding_reasons:
        return _build_decision(
            binding=binding,
            reviews=reviews,
            status="blocked",
            benchmark_eligible=False,
            rollout_activation_enabled=rollout_config.activation_enabled,
            reason_codes=binding_reasons,
            expected_execution_count=len(ordered_slots),
            successful_execution_count=0,
            reviewed_execution_count=0,
            activation_decision=None,
            case_results=(),
            records=_blocked_records(
                ordered_slots,
                reason_code="evaluation_binding_invalid",
            ),
        )

    review_by_slot = {
        (review.case_id, review.variant): review for review in reviews.reviews
    }
    case_results: list[EvidenceEvaluationCaseResult] = []
    records: list[EvidenceExecutionRecordV1] = []
    successful_count = 0
    reviewed_count = 0
    global_reasons: list[str] = []
    stop = False

    for case in dataset.cases:
        for definition in execution_config.variants:
            if stop:
                records.append(
                    _not_executed_record(
                        case_id=case.case_id,
                        variant=definition.variant,
                        reason_code="not_executed_after_fail_fast",
                    )
                )
                continue
            try:
                attempt = await executor.execute(
                    case=case,
                    definition=definition,
                    binding=binding,
                )
                if not isinstance(attempt, EvidenceVariantAttemptV1):
                    raise TypeError(
                        "variant executor must return EvidenceVariantAttemptV1"
                    )
            except ValidationError as error:
                attempt = EvidenceVariantAttemptV1(
                    schema_version="evidence_variant_attempt_v1",
                    case_id=case.case_id,
                    variant=definition.variant,
                    status="failed",
                    observation=None,
                    failure_reason_code="schema_validation_failed",
                    failure_type=_safe_exception_type(error),
                )
            except EvidenceEvaluationError as error:
                attempt = EvidenceVariantAttemptV1(
                    schema_version="evidence_variant_attempt_v1",
                    case_id=case.case_id,
                    variant=definition.variant,
                    status="failed",
                    observation=None,
                    failure_reason_code="business_validation_failed",
                    failure_type=_safe_exception_type(error),
                )
            except Exception as error:
                attempt = EvidenceVariantAttemptV1(
                    schema_version="evidence_variant_attempt_v1",
                    case_id=case.case_id,
                    variant=definition.variant,
                    status="failed",
                    observation=None,
                    failure_reason_code="variant_execution_exception",
                    failure_type=_safe_exception_type(error),
                )

            if attempt.case_id != case.case_id or attempt.variant != definition.variant:
                attempt = EvidenceVariantAttemptV1(
                    schema_version="evidence_variant_attempt_v1",
                    case_id=case.case_id,
                    variant=definition.variant,
                    status="blocked",
                    observation=None,
                    failure_reason_code="attempt_identity_mismatch",
                    failure_type="ExecutionBindingError",
                )
            records.append(_attempt_record(attempt))
            if attempt.status != "success":
                global_reasons.append("variant_execution_incomplete")
                stop = True
                continue

            observation = attempt.observation
            if observation is None:
                raise AssertionError("validated successful attempt has no observation")
            observation_reasons = _observation_binding_reasons(
                observation=observation,
                case=case,
                definition=definition,
                binding=binding,
            )
            if observation_reasons:
                records[-1] = EvidenceExecutionRecordV1(
                    schema_version="evidence_execution_record_v1",
                    case_id=case.case_id,
                    variant=definition.variant,
                    status="blocked",
                    output_fingerprint=None,
                    failure_reason_code=observation_reasons[0],
                    failure_type="ObservationBindingError",
                )
                global_reasons.extend(observation_reasons)
                stop = True
                continue

            successful_count += 1
            review = review_by_slot.get((case.case_id, definition.variant))
            if review is None:
                global_reasons.append("human_semantic_review_incomplete")
                stop = True
                continue
            try:
                case_results.append(
                    _case_result(
                        case=case,
                        observation=observation,
                        review=review,
                    )
                )
            except (EvidenceEvaluationError, ValidationError):
                global_reasons.append("human_semantic_review_invalid")
                stop = True
                continue
            reviewed_count += 1

    if (
        len(review_by_slot) != len(expected_slots)
        or set(review_by_slot) != expected_slots
    ):
        global_reasons.append("human_semantic_review_inventory_mismatch")

    matrix_complete = (
        successful_count == len(ordered_slots)
        and reviewed_count == len(ordered_slots)
        and len(case_results) == len(ordered_slots)
    )
    if not matrix_complete:
        global_reasons.append("factorial_execution_incomplete")
        return _build_decision(
            binding=binding,
            reviews=reviews,
            status="blocked",
            benchmark_eligible=False,
            rollout_activation_enabled=rollout_config.activation_enabled,
            reason_codes=global_reasons,
            expected_execution_count=len(ordered_slots),
            successful_execution_count=successful_count,
            reviewed_execution_count=reviewed_count,
            activation_decision=None,
            case_results=case_results,
            records=records,
        )

    try:
        activation = evaluate_evidence_activation(
            results=tuple(case_results),
            config=benchmark_config,
        )
    except (EvidenceEvaluationError, ValidationError):
        return _build_decision(
            binding=binding,
            reviews=reviews,
            status="blocked",
            benchmark_eligible=False,
            rollout_activation_enabled=rollout_config.activation_enabled,
            reason_codes=("activation_evaluation_failed",),
            expected_execution_count=len(ordered_slots),
            successful_execution_count=successful_count,
            reviewed_execution_count=reviewed_count,
            activation_decision=None,
            case_results=case_results,
            records=records,
        )

    if binding.execution_mode != "live":
        return _build_decision(
            binding=binding,
            reviews=reviews,
            status="blocked",
            benchmark_eligible=activation.eligible,
            rollout_activation_enabled=rollout_config.activation_enabled,
            reason_codes=("non_live_execution", *activation.reason_codes),
            expected_execution_count=len(ordered_slots),
            successful_execution_count=successful_count,
            reviewed_execution_count=reviewed_count,
            activation_decision=activation,
            case_results=case_results,
            records=records,
        )
    if not activation.eligible:
        return _build_decision(
            binding=binding,
            reviews=reviews,
            status="fail",
            benchmark_eligible=False,
            rollout_activation_enabled=rollout_config.activation_enabled,
            reason_codes=activation.reason_codes,
            expected_execution_count=len(ordered_slots),
            successful_execution_count=successful_count,
            reviewed_execution_count=reviewed_count,
            activation_decision=activation,
            case_results=case_results,
            records=records,
        )
    if not rollout_config.activation_enabled:
        return _build_decision(
            binding=binding,
            reviews=reviews,
            status="blocked",
            benchmark_eligible=True,
            rollout_activation_enabled=False,
            reason_codes=("rollout_activation_disabled",),
            expected_execution_count=len(ordered_slots),
            successful_execution_count=successful_count,
            reviewed_execution_count=reviewed_count,
            activation_decision=activation,
            case_results=case_results,
            records=records,
        )
    return _build_decision(
        binding=binding,
        reviews=reviews,
        status="pass",
        benchmark_eligible=True,
        rollout_activation_enabled=True,
        reason_codes=(),
        expected_execution_count=len(ordered_slots),
        successful_execution_count=successful_count,
        reviewed_execution_count=reviewed_count,
        activation_decision=activation,
        case_results=case_results,
        records=records,
    )


__all__ = [
    "EvidenceVariantExecutor",
    "LiveEvidenceVariantAdapter",
    "LiveEvidenceVariantExecutor",
    "SealedAttemptVariantExecutor",
    "run_evidence_rollout_evaluation",
]
