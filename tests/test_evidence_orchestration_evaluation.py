"""2x2 factorial evaluation and activation-gate tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.evidence_benchmark_config import (
    load_evidence_benchmark_config,
)
from src.rag.parent_child.evidence_evaluation import (
    EvidenceEvaluationCaseResult,
    EvidenceEvaluationError,
    evaluate_evidence_activation,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "rag" / "evidence_benchmark.yaml"


def _result(
    *,
    case_id: str,
    variant: str,
    subject_count: int,
    resource_count: int,
    initial_sufficient: bool,
    coverage: float,
    gaps: int,
    precision: float,
    claim_support: float,
    ungrounded: float,
    cost: float,
    latency: float,
    repeated_queries: int = 0,
):
    return EvidenceEvaluationCaseResult(
        schema_version="evidence_evaluation_case_v1",
        case_id=case_id,
        variant=variant,
        subject_count=subject_count,
        resource_count=resource_count,
        initial_evidence_sufficient=initial_sufficient,
        bounded=True,
        forced_stop_marked_sufficient=False,
        silent_resource_omission=False,
        silent_subject_omission=False,
        repeated_query_count=repeated_queries,
        weighted_coverage=coverage,
        required_gap_count=gaps,
        evidence_precision=precision,
        premature_stop=False,
        over_search=False,
        source_routing_f1=0.90,
        resource_subject_recall=0.98,
        assignment_precision=0.95,
        claim_support_rate=claim_support,
        ungrounded_fact_rate=ungrounded,
        retrieval_cost_units=cost,
        latency_ms=latency,
    )


def _passing_results():
    return (
        _result(
            case_id="simple",
            variant="P0",
            subject_count=1,
            resource_count=1,
            initial_sufficient=True,
            coverage=0.80,
            gaps=2,
            precision=0.94,
            claim_support=0.75,
            ungrounded=0.10,
            cost=1.0,
            latency=100.0,
        ),
        _result(
            case_id="simple",
            variant="PG",
            subject_count=1,
            resource_count=1,
            initial_sufficient=True,
            coverage=0.82,
            gaps=2,
            precision=0.94,
            claim_support=0.77,
            ungrounded=0.09,
            cost=1.05,
            latency=105.0,
        ),
        _result(
            case_id="simple",
            variant="PR",
            subject_count=1,
            resource_count=1,
            initial_sufficient=True,
            coverage=0.83,
            gaps=1,
            precision=0.93,
            claim_support=0.79,
            ungrounded=0.08,
            cost=1.08,
            latency=110.0,
        ),
        _result(
            case_id="simple",
            variant="PGR",
            subject_count=1,
            resource_count=1,
            initial_sufficient=True,
            coverage=0.85,
            gaps=1,
            precision=0.93,
            claim_support=0.84,
            ungrounded=0.07,
            cost=1.10,
            latency=120.0,
        ),
        _result(
            case_id="multi",
            variant="P0",
            subject_count=2,
            resource_count=2,
            initial_sufficient=False,
            coverage=0.60,
            gaps=4,
            precision=0.94,
            claim_support=0.65,
            ungrounded=0.20,
            cost=2.0,
            latency=200.0,
        ),
        _result(
            case_id="multi",
            variant="PG",
            subject_count=2,
            resource_count=2,
            initial_sufficient=False,
            coverage=0.67,
            gaps=3,
            precision=0.93,
            claim_support=0.70,
            ungrounded=0.17,
            cost=2.2,
            latency=215.0,
        ),
        _result(
            case_id="multi",
            variant="PR",
            subject_count=2,
            resource_count=2,
            initial_sufficient=False,
            coverage=0.70,
            gaps=3,
            precision=0.93,
            claim_support=0.72,
            ungrounded=0.15,
            cost=2.5,
            latency=225.0,
        ),
        _result(
            case_id="multi",
            variant="PGR",
            subject_count=2,
            resource_count=2,
            initial_sufficient=False,
            coverage=0.75,
            gaps=2,
            precision=0.93,
            claim_support=0.78,
            ungrounded=0.12,
            cost=2.8,
            latency=240.0,
        ),
    )


def test_factorial_gate_accepts_joint_candidate_only_when_all_thresholds_pass():
    config = load_evidence_benchmark_config(CONFIG_PATH)
    decision = evaluate_evidence_activation(
        results=_passing_results(),
        config=config,
    )

    assert decision.eligible is True
    assert decision.reason_codes == ()
    assert decision.metrics.case_count == 2
    assert decision.metrics.multi_case_count == 1
    assert decision.metrics.overall_weighted_coverage_lift >= 0.08
    assert decision.metrics.multi_weighted_coverage_lift >= 0.10
    assert decision.metrics.average_retrieval_cost_ratio <= 1.50
    assert decision.candidate_failure_policy == "fail_fast"


def test_factorial_gate_rejects_exact_query_repeat_without_fallback():
    config = load_evidence_benchmark_config(CONFIG_PATH)
    results = list(_passing_results())
    repeated = results[-1].model_copy(update={"repeated_query_count": 1})
    results[-1] = EvidenceEvaluationCaseResult.model_validate(
        repeated.model_dump(mode="python")
    )

    decision = evaluate_evidence_activation(
        results=tuple(results),
        config=config,
    )

    assert decision.eligible is False
    assert "repeated_query" in decision.reason_codes


def test_factorial_gate_rejects_incomplete_variant_matrix():
    config = load_evidence_benchmark_config(CONFIG_PATH)
    results = tuple(
        item
        for item in _passing_results()
        if not (item.case_id == "multi" and item.variant == "PR")
    )

    with pytest.raises(
        EvidenceEvaluationError,
        match="factorial_variant_mismatch",
    ):
        evaluate_evidence_activation(results=results, config=config)
