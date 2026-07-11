from __future__ import annotations

from pathlib import Path

from src.config.rag_benchmark_config import load_rag_benchmark_config
from src.rag.parent_child.evaluation import (
    BootstrapConfig,
    PairedComparisonReport,
    PairedMetricComparison,
    SubjectPairedComparison,
)
from src.rag.parent_child.evaluation_gate import (
    EndToEndQualityOutcome,
    OperationalBenchmarkOutcome,
    evaluate_activation_eligibility,
)


def _metric(
    name: str,
    k: int | None,
    *,
    baseline: float,
    candidate: float,
    lower: float,
    upper: float,
) -> PairedMetricComparison:
    return PairedMetricComparison(
        schema_version="paired_metric_comparison_v1",
        metric_name=name,
        k=k,
        sample_count=100,
        baseline_mean=baseline,
        candidate_mean=candidate,
        mean_delta=candidate - baseline,
        confidence=0.95,
        ci_lower=lower,
        ci_upper=upper,
    )


def _comparison(
    *,
    recall_baseline: float = 0.50,
    recall_candidate: float = 0.56,
    recall_lower: float = 0.01,
    recall_upper: float = 0.10,
) -> PairedComparisonReport:
    recall = _metric(
        "evidence_recall_at_k",
        5,
        baseline=recall_baseline,
        candidate=recall_candidate,
        lower=recall_lower,
        upper=recall_upper,
    )
    subjects = tuple(
        SubjectPairedComparison(
            schema_version="subject_paired_comparison_v1",
            subject=subject,
            metrics=(recall,),
        )
        for subject in (
            "big_data",
            "computer",
            "machine_learning",
            "math",
            "python",
        )
    )
    return PairedComparisonReport(
        schema_version="paired_comparison_report_v1",
        dataset_id="human-gold-v1",
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        eligible_queries_only=True,
        paired_query_count=100,
        bootstrap=BootstrapConfig(
            schema_version="paired_bootstrap_config_v1",
            iterations=10000,
            seed=20260710,
            confidence=0.95,
        ),
        global_metrics=(
            recall,
            _metric(
                "mrr",
                None,
                baseline=0.40,
                candidate=0.44,
                lower=0.01,
                upper=0.07,
            ),
            _metric(
                "noise_at_k",
                5,
                baseline=0.20,
                candidate=0.21,
                lower=-0.01,
                upper=0.02,
            ),
        ),
        subjects=subjects,
        rollout_data_gate_passed=True,
        production_recommendation_blocked=False,
        blocker_codes=(),
    )


def _operational(**overrides: object) -> OperationalBenchmarkOutcome:
    values: dict[str, object] = {
        "schema_version": "operational_benchmark_outcome_v2",
        "dataset_id": "human-gold-v1",
        "gold_dataset_sha256": "a" * 64,
        "baseline_run_id": "baseline",
        "candidate_run_id": "candidate",
        "candidate_generation_id": "candidate-generation",
        "embedding_fingerprint": "b" * 64,
        "baseline_artifact_manifest_sha256": "c" * 64,
        "candidate_artifact_manifest_sha256": "d" * 64,
        "query_count": 100,
        "baseline_p50_latency_ms": 80.0,
        "baseline_p95_latency_ms": 100.0,
        "candidate_p50_latency_ms": 100.0,
        "candidate_p95_latency_ms": 120.0,
        "baseline_error_count": 0,
        "candidate_error_count": 0,
        "baseline_error_rate": 0.0,
        "candidate_error_rate": 0.0,
        "baseline_context_tokens_total": 10000,
        "candidate_context_tokens_total": 12000,
        "baseline_context_tokens_mean": 100.0,
        "candidate_context_tokens_mean": 120.0,
        "baseline_context_tokens_p95": 100.0,
        "candidate_context_tokens_p95": 120.0,
        "parent_context_token_ratio": 1.2,
        "parent_hydration_attempt_count": 0,
        "parent_hydration_success_count": 0,
        "orphan_child_count": 0,
        "parent_hydration_failure_count": 0,
        "generation_mismatch_count": 0,
    }
    values.update(overrides)
    return OperationalBenchmarkOutcome.model_validate(values)


def _end_to_end() -> EndToEndQualityOutcome:
    return EndToEndQualityOutcome(
        schema_version="end_to_end_quality_outcome_v2",
        dataset_id="human-gold-v1",
        gold_dataset_sha256="a" * 64,
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        answer_model_fingerprint="b" * 64,
        assessment_protocol_sha256="c" * 64,
        assessment_source="human",
        scored_query_count=100,
        baseline_answer_correctness=0.80,
        candidate_answer_correctness=0.80,
        baseline_citation_support=0.85,
        candidate_citation_support=0.86,
        baseline_hallucination_rate=0.05,
        candidate_hallucination_rate=0.05,
        baseline_context_tokens_total=10000,
        candidate_context_tokens_total=11000,
        baseline_context_tokens_mean=100.0,
        candidate_context_tokens_mean=110.0,
    )


def test_all_retrieval_operational_and_answer_gates_are_required() -> None:
    benchmark = load_rag_benchmark_config(Path("config/rag/benchmark.yaml"))

    eligible = evaluate_activation_eligibility(
        benchmark=benchmark,
        comparison=_comparison(),
        operational=_operational(),
        end_to_end=_end_to_end(),
        functional_tests_passed=True,
    )
    assert eligible.activation_eligible is True
    assert eligible.blocker_codes == ()

    blocked = evaluate_activation_eligibility(
        benchmark=benchmark,
        comparison=_comparison(),
        operational=_operational(orphan_child_count=1),
        end_to_end=_end_to_end(),
        functional_tests_passed=True,
    )
    assert blocked.activation_eligible is False
    assert blocked.blocker_codes == ("orphan_child_detected",)


def test_perfect_baseline_recall_is_eligible_when_candidate_remains_perfect() -> None:
    benchmark = load_rag_benchmark_config(Path("config/rag/benchmark.yaml"))

    outcome = evaluate_activation_eligibility(
        benchmark=benchmark,
        comparison=_comparison(
            recall_baseline=1.0,
            recall_candidate=1.0,
            recall_lower=0.0,
            recall_upper=0.0,
        ),
        operational=_operational(),
        end_to_end=_end_to_end(),
        functional_tests_passed=True,
    )

    assert outcome.activation_eligible is True
    assert outcome.blocker_codes == ()
