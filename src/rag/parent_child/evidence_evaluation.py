"""Paired 2x2 evaluation and strict activation gate for evidence orchestration."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Annotated, Literal

from pydantic import Field, model_validator

from src.config._rag_config import NonBlankStr, StrictRagConfigModel
from src.config.evidence_benchmark_config import EvidenceBenchmarkConfig

Variant = Literal["P0", "PG", "PR", "PGR"]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]


class EvidenceEvaluationError(ValueError):
    """The factorial evaluation dataset violates a required invariant."""

    def __init__(self, *, code: str, reason: str) -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"{code}: {reason}")


class EvidenceEvaluationCaseResult(StrictRagConfigModel):
    """One content-free result for one case and one factorial variant."""

    schema_version: Literal["evidence_evaluation_case_v1"]
    case_id: NonBlankStr
    variant: Variant
    subject_count: PositiveInt
    resource_count: PositiveInt
    initial_evidence_sufficient: bool
    bounded: bool
    forced_stop_marked_sufficient: bool
    silent_resource_omission: bool
    silent_subject_omission: bool
    repeated_query_count: NonNegativeInt
    weighted_coverage: UnitFloat
    required_gap_count: NonNegativeInt
    evidence_precision: UnitFloat
    premature_stop: bool
    over_search: bool
    source_routing_f1: UnitFloat
    resource_subject_recall: UnitFloat
    assignment_precision: UnitFloat
    claim_support_rate: UnitFloat
    ungrounded_fact_rate: UnitFloat
    retrieval_cost_units: PositiveFloat
    latency_ms: PositiveFloat


class EvidenceActivationMetrics(StrictRagConfigModel):
    """Deterministic PGR-versus-P0 metrics plus 2x2 effect decomposition."""

    schema_version: Literal["evidence_activation_metrics_v1"]
    case_count: PositiveInt
    simple_case_count: PositiveInt
    multi_case_count: PositiveInt
    initial_sufficient_case_count: PositiveInt
    bounded_rate: UnitFloat
    forced_stop_marked_sufficient_count: NonNegativeInt
    silent_resource_omission_count: NonNegativeInt
    silent_subject_omission_count: NonNegativeInt
    repeated_query_count: NonNegativeInt
    planning_only_coverage_lift: float
    repair_only_coverage_lift: float
    joint_coverage_lift: float
    factorial_interaction: float
    overall_weighted_coverage_lift: float
    multi_weighted_coverage_lift: float
    required_gap_reduction: UnitFloat
    evidence_precision_loss: float
    simple_case_coverage_regression: float
    premature_stop_rate: UnitFloat
    over_search_rate: UnitFloat
    source_routing_f1: UnitFloat
    resource_subject_recall: UnitFloat
    assignment_precision: UnitFloat
    claim_support_lift: float
    ungrounded_fact_reduction: UnitFloat
    average_retrieval_cost_ratio: PositiveFloat
    initial_sufficient_cost_ratio: PositiveFloat
    p95_latency_ratio: PositiveFloat


class EvidenceActivationDecision(StrictRagConfigModel):
    """Fail-closed activation result; no request-time fallback is authorized."""

    schema_version: Literal["evidence_activation_decision_v1"]
    eligible: bool
    reason_codes: tuple[NonBlankStr, ...]
    metrics: EvidenceActivationMetrics
    candidate_failure_policy: Literal["fail_fast"]

    @model_validator(mode="after")
    def validate_reason_contract(self) -> "EvidenceActivationDecision":
        if self.eligible and self.reason_codes:
            raise ValueError("eligible decision must not contain failure reasons")
        if not self.eligible and not self.reason_codes:
            raise ValueError("ineligible decision requires failure reasons")
        return self


def _mean(values: list[float], *, label: str) -> float:
    if not values:
        raise EvidenceEvaluationError(
            code="empty_metric_partition",
            reason=f"evaluation partition is empty: {label}",
        )
    return sum(values) / len(values)


def _p95(values: list[float]) -> float:
    if not values:
        raise EvidenceEvaluationError(
            code="empty_latency_partition",
            reason="p95 latency requires at least one case",
        )
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _reduction(baseline: float, candidate: float) -> float:
    if baseline == 0.0:
        return 1.0 if candidate == 0.0 else 0.0
    return max(0.0, min(1.0, 1.0 - (candidate / baseline)))


def _validate_factorial_results(
    results: tuple[EvidenceEvaluationCaseResult, ...],
    config: EvidenceBenchmarkConfig,
) -> dict[str, dict[Variant, EvidenceEvaluationCaseResult]]:
    if not results:
        raise EvidenceEvaluationError(
            code="empty_evaluation_dataset",
            reason="activation evaluation requires cases",
        )
    grouped: dict[str, dict[Variant, EvidenceEvaluationCaseResult]] = defaultdict(dict)
    for result in results:
        variants = grouped[result.case_id]
        if result.variant in variants:
            raise EvidenceEvaluationError(
                code="duplicate_case_variant",
                reason="each case may contain one result per variant",
            )
        variants[result.variant] = result
    expected = set(config.required_variants)
    for case_id, variants in grouped.items():
        if set(variants) != expected:
            raise EvidenceEvaluationError(
                code="factorial_variant_mismatch",
                reason=f"case {case_id} must contain P0, PG, PR, and PGR",
            )
        shape = {
            (
                result.subject_count,
                result.resource_count,
                result.initial_evidence_sufficient,
            )
            for result in variants.values()
        }
        if len(shape) != 1:
            raise EvidenceEvaluationError(
                code="case_shape_mismatch",
                reason="case shape and initial sufficiency must be variant-invariant",
            )
    return dict(grouped)


def evaluate_evidence_activation(
    *,
    results: tuple[EvidenceEvaluationCaseResult, ...],
    config: EvidenceBenchmarkConfig,
) -> EvidenceActivationDecision:
    """Evaluate PGR strictly against P0 while retaining PG/PR interaction metrics."""

    grouped = _validate_factorial_results(results, config)
    rows = list(grouped.values())
    simple_rows = [
        row
        for row in rows
        if row["P0"].subject_count == 1 and row["P0"].resource_count == 1
    ]
    multi_rows = [
        row
        for row in rows
        if row["P0"].subject_count > 1 or row["P0"].resource_count > 1
    ]
    initial_rows = [row for row in rows if row["P0"].initial_evidence_sufficient]
    if not simple_rows or not multi_rows or not initial_rows:
        raise EvidenceEvaluationError(
            code="required_case_partition_missing",
            reason="dataset must include simple, multi, and initially sufficient cases",
        )

    def mean_metric(variant: Variant, field_name: str, subset=rows) -> float:
        return _mean(
            [float(getattr(row[variant], field_name)) for row in subset],
            label=f"{variant}.{field_name}",
        )

    p0_coverage = mean_metric("P0", "weighted_coverage")
    pg_coverage = mean_metric("PG", "weighted_coverage")
    pr_coverage = mean_metric("PR", "weighted_coverage")
    pgr_coverage = mean_metric("PGR", "weighted_coverage")
    p0_gap = mean_metric("P0", "required_gap_count")
    pgr_gap = mean_metric("PGR", "required_gap_count")
    p0_ungrounded = mean_metric("P0", "ungrounded_fact_rate")
    pgr_ungrounded = mean_metric("PGR", "ungrounded_fact_rate")
    p0_cost = mean_metric("P0", "retrieval_cost_units")
    pgr_cost = mean_metric("PGR", "retrieval_cost_units")
    initial_p0_cost = mean_metric("P0", "retrieval_cost_units", initial_rows)
    initial_pgr_cost = mean_metric("PGR", "retrieval_cost_units", initial_rows)
    p0_p95 = _p95([row["P0"].latency_ms for row in rows])
    pgr_p95 = _p95([row["PGR"].latency_ms for row in rows])

    candidate_rows = [row["PGR"] for row in rows]
    metrics = EvidenceActivationMetrics(
        schema_version="evidence_activation_metrics_v1",
        case_count=len(rows),
        simple_case_count=len(simple_rows),
        multi_case_count=len(multi_rows),
        initial_sufficient_case_count=len(initial_rows),
        bounded_rate=_mean(
            [float(item.bounded) for item in candidate_rows],
            label="PGR.bounded",
        ),
        forced_stop_marked_sufficient_count=sum(
            item.forced_stop_marked_sufficient for item in candidate_rows
        ),
        silent_resource_omission_count=sum(
            item.silent_resource_omission for item in candidate_rows
        ),
        silent_subject_omission_count=sum(
            item.silent_subject_omission for item in candidate_rows
        ),
        repeated_query_count=sum(item.repeated_query_count for item in candidate_rows),
        planning_only_coverage_lift=pg_coverage - p0_coverage,
        repair_only_coverage_lift=pr_coverage - p0_coverage,
        joint_coverage_lift=pgr_coverage - p0_coverage,
        factorial_interaction=(pgr_coverage - pg_coverage - pr_coverage + p0_coverage),
        overall_weighted_coverage_lift=pgr_coverage - p0_coverage,
        multi_weighted_coverage_lift=(
            mean_metric("PGR", "weighted_coverage", multi_rows)
            - mean_metric("P0", "weighted_coverage", multi_rows)
        ),
        required_gap_reduction=_reduction(p0_gap, pgr_gap),
        evidence_precision_loss=(
            mean_metric("P0", "evidence_precision")
            - mean_metric("PGR", "evidence_precision")
        ),
        simple_case_coverage_regression=max(
            0.0,
            mean_metric("P0", "weighted_coverage", simple_rows)
            - mean_metric("PGR", "weighted_coverage", simple_rows),
        ),
        premature_stop_rate=_mean(
            [float(item.premature_stop) for item in candidate_rows],
            label="PGR.premature_stop",
        ),
        over_search_rate=_mean(
            [float(item.over_search) for item in candidate_rows],
            label="PGR.over_search",
        ),
        source_routing_f1=mean_metric("PGR", "source_routing_f1"),
        resource_subject_recall=mean_metric("PGR", "resource_subject_recall"),
        assignment_precision=mean_metric("PGR", "assignment_precision"),
        claim_support_lift=(
            mean_metric("PGR", "claim_support_rate")
            - mean_metric("P0", "claim_support_rate")
        ),
        ungrounded_fact_reduction=_reduction(p0_ungrounded, pgr_ungrounded),
        average_retrieval_cost_ratio=pgr_cost / p0_cost,
        initial_sufficient_cost_ratio=initial_pgr_cost / initial_p0_cost,
        p95_latency_ratio=pgr_p95 / p0_p95,
    )

    reasons: list[str] = []

    def require(condition: bool, code: str) -> None:
        if not condition:
            reasons.append(code)

    require(metrics.bounded_rate == 1.0, "unbounded_execution")
    require(
        metrics.forced_stop_marked_sufficient_count == 0,
        "forced_stop_marked_sufficient",
    )
    require(
        metrics.silent_resource_omission_count == 0,
        "silent_resource_omission",
    )
    require(
        metrics.silent_subject_omission_count == 0,
        "silent_subject_omission",
    )
    require(metrics.repeated_query_count == 0, "repeated_query")
    require(
        metrics.overall_weighted_coverage_lift
        >= config.min_overall_weighted_coverage_lift,
        "overall_coverage_lift_below_gate",
    )
    require(
        metrics.multi_weighted_coverage_lift >= config.min_multi_weighted_coverage_lift,
        "multi_coverage_lift_below_gate",
    )
    require(
        pgr_gap <= p0_gap * (1.0 - config.min_required_gap_reduction),
        "required_gap_reduction_below_gate",
    )
    require(
        metrics.evidence_precision_loss <= config.max_evidence_precision_loss,
        "evidence_precision_loss_above_gate",
    )
    require(
        metrics.simple_case_coverage_regression
        <= config.max_simple_case_coverage_regression,
        "simple_case_regression_above_gate",
    )
    require(
        metrics.premature_stop_rate <= config.max_premature_stop_rate,
        "premature_stop_rate_above_gate",
    )
    require(
        metrics.over_search_rate <= config.max_over_search_rate,
        "over_search_rate_above_gate",
    )
    require(
        metrics.source_routing_f1 >= config.min_source_routing_f1,
        "source_routing_f1_below_gate",
    )
    require(
        metrics.resource_subject_recall >= config.min_resource_subject_recall,
        "resource_subject_recall_below_gate",
    )
    require(
        metrics.assignment_precision >= config.min_assignment_precision,
        "assignment_precision_below_gate",
    )
    require(
        metrics.claim_support_lift >= config.min_claim_support_lift,
        "claim_support_lift_below_gate",
    )
    require(
        pgr_ungrounded <= p0_ungrounded * (1.0 - config.min_ungrounded_fact_reduction),
        "ungrounded_fact_reduction_below_gate",
    )
    require(
        metrics.average_retrieval_cost_ratio <= config.max_average_retrieval_cost_ratio,
        "average_cost_ratio_above_gate",
    )
    require(
        metrics.initial_sufficient_cost_ratio
        <= config.max_initial_sufficient_cost_ratio,
        "initial_sufficient_cost_ratio_above_gate",
    )
    require(
        metrics.p95_latency_ratio <= config.max_p95_latency_ratio,
        "p95_latency_ratio_above_gate",
    )
    return EvidenceActivationDecision(
        schema_version="evidence_activation_decision_v1",
        eligible=not reasons,
        reason_codes=tuple(reasons),
        metrics=metrics,
        candidate_failure_policy=config.candidate_failure_policy,
    )


__all__ = [
    "EvidenceActivationDecision",
    "EvidenceActivationMetrics",
    "EvidenceEvaluationCaseResult",
    "EvidenceEvaluationError",
    "Variant",
    "evaluate_evidence_activation",
]
