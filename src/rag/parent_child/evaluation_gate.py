"""Offline activation eligibility gates kept separate from metric computation."""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config.rag_benchmark_config import RagBenchmarkConfig
from src.rag.parent_child.evaluation import (
    EvaluationReport,
    PairedComparisonReport,
    PairedMetricComparison,
)


class EvaluationEligibilityError(RuntimeError):
    """Required benchmark metrics or subjects are absent/inconsistent."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class OperationalBenchmarkOutcome(_StrictFrozenModel):
    """Content-free operational evidence bound to one paired retrieval run."""

    schema_version: Literal["operational_benchmark_outcome_v2"]
    dataset_id: str
    gold_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_run_id: str
    candidate_run_id: str
    candidate_generation_id: str
    embedding_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_artifact_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_artifact_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_count: int = Field(gt=0)
    baseline_p50_latency_ms: float = Field(ge=0.0)
    baseline_p95_latency_ms: float = Field(gt=0.0)
    candidate_p50_latency_ms: float = Field(ge=0.0)
    candidate_p95_latency_ms: float = Field(gt=0.0)
    baseline_error_count: int = Field(ge=0)
    candidate_error_count: int = Field(ge=0)
    baseline_error_rate: float = Field(ge=0.0, le=1.0)
    candidate_error_rate: float = Field(ge=0.0, le=1.0)
    baseline_context_tokens_total: int = Field(ge=0)
    candidate_context_tokens_total: int = Field(ge=0)
    baseline_context_tokens_mean: float = Field(ge=0.0)
    candidate_context_tokens_mean: float = Field(ge=0.0)
    baseline_context_tokens_p95: float = Field(ge=0.0)
    candidate_context_tokens_p95: float = Field(ge=0.0)
    parent_context_token_ratio: float = Field(ge=0.0)
    parent_hydration_attempt_count: int = Field(ge=0)
    parent_hydration_success_count: int = Field(ge=0)
    orphan_child_count: int = Field(ge=0)
    parent_hydration_failure_count: int = Field(ge=0)
    generation_mismatch_count: int = Field(ge=0)

    @field_validator(
        "baseline_p50_latency_ms",
        "baseline_p95_latency_ms",
        "candidate_p50_latency_ms",
        "candidate_p95_latency_ms",
        "baseline_error_rate",
        "candidate_error_rate",
        "baseline_context_tokens_mean",
        "candidate_context_tokens_mean",
        "baseline_context_tokens_p95",
        "candidate_context_tokens_p95",
        "parent_context_token_ratio",
    )
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("operational benchmark metrics must be finite")
        return value

    @model_validator(mode="after")
    def _binding_and_count_invariants(self) -> Self:
        if not self.dataset_id or self.dataset_id != self.dataset_id.strip():
            raise ValueError("dataset_id must be non-empty and already stripped")
        if not self.baseline_run_id or not self.candidate_run_id:
            raise ValueError("baseline and candidate run IDs are required")
        if self.baseline_run_id == self.candidate_run_id:
            raise ValueError("baseline_run_id and candidate_run_id must differ")
        if not self.candidate_generation_id:
            raise ValueError("candidate_generation_id is required")
        if self.baseline_p95_latency_ms < self.baseline_p50_latency_ms:
            raise ValueError("baseline p95 latency must be at least p50")
        if self.candidate_p95_latency_ms < self.candidate_p50_latency_ms:
            raise ValueError("candidate p95 latency must be at least p50")
        if not math.isclose(
            self.baseline_error_rate,
            self.baseline_error_count / self.query_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("baseline error_rate must equal error_count/query_count")
        if not math.isclose(
            self.candidate_error_rate,
            self.candidate_error_count / self.query_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("candidate error_rate must equal error_count/query_count")
        baseline_mean = self.baseline_context_tokens_total / self.query_count
        candidate_mean = self.candidate_context_tokens_total / self.query_count
        if not math.isclose(
            self.baseline_context_tokens_mean,
            baseline_mean,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("baseline context-token mean must equal total/query_count")
        if not math.isclose(
            self.candidate_context_tokens_mean,
            candidate_mean,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "candidate context-token mean must equal total/query_count"
            )
        if baseline_mean <= 0.0:
            raise ValueError("baseline context-token mean must be positive")
        if not math.isclose(
            self.parent_context_token_ratio,
            candidate_mean / baseline_mean,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("parent context-token ratio must match means")
        if (
            self.parent_hydration_success_count + self.parent_hydration_failure_count
            != self.parent_hydration_attempt_count
        ):
            raise ValueError("parent hydration counts must sum to attempts")
        return self


class EndToEndQualityOutcome(_StrictFrozenModel):
    """Human-scored final-answer quality bound to one retrieval comparison."""

    schema_version: Literal["end_to_end_quality_outcome_v2"]
    dataset_id: str
    gold_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_run_id: str
    candidate_run_id: str
    answer_model_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    assessment_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assessment_source: Literal["human"]
    scored_query_count: int = Field(gt=0)
    baseline_answer_correctness: float = Field(ge=0.0, le=1.0)
    candidate_answer_correctness: float = Field(ge=0.0, le=1.0)
    baseline_citation_support: float = Field(ge=0.0, le=1.0)
    candidate_citation_support: float = Field(ge=0.0, le=1.0)
    baseline_hallucination_rate: float = Field(ge=0.0, le=1.0)
    candidate_hallucination_rate: float = Field(ge=0.0, le=1.0)
    baseline_context_tokens_total: int = Field(ge=0)
    candidate_context_tokens_total: int = Field(ge=0)
    baseline_context_tokens_mean: float = Field(ge=0.0)
    candidate_context_tokens_mean: float = Field(ge=0.0)

    @field_validator("dataset_id", "baseline_run_id", "candidate_run_id")
    @classmethod
    def _identifiers(cls, value: str, info: object) -> str:
        field_name = str(getattr(info, "field_name", "identifier"))
        if not value or value != value.strip():
            raise ValueError(f"{field_name} must be non-empty and already stripped")
        if any(character in value for character in ("/", "\\", "\x00")):
            raise ValueError(f"{field_name} must not contain path separators or NUL")
        return value

    @field_validator(
        "baseline_answer_correctness",
        "candidate_answer_correctness",
        "baseline_citation_support",
        "candidate_citation_support",
        "baseline_hallucination_rate",
        "candidate_hallucination_rate",
        "baseline_context_tokens_mean",
        "candidate_context_tokens_mean",
    )
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("end-to-end quality metrics must be finite")
        return value

    @model_validator(mode="after")
    def _binding_and_token_invariants(self) -> Self:
        if self.baseline_run_id == self.candidate_run_id:
            raise ValueError("baseline_run_id and candidate_run_id must differ")
        expected_baseline_mean = (
            self.baseline_context_tokens_total / self.scored_query_count
        )
        expected_candidate_mean = (
            self.candidate_context_tokens_total / self.scored_query_count
        )
        if not math.isclose(
            self.baseline_context_tokens_mean,
            expected_baseline_mean,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "baseline_context_tokens_mean must equal total/scored_query_count"
            )
        if not math.isclose(
            self.candidate_context_tokens_mean,
            expected_candidate_mean,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "candidate_context_tokens_mean must equal total/scored_query_count"
            )
        return self


class ActivationEligibilityReport(_StrictFrozenModel):
    schema_version: Literal["activation_eligibility_report_v1"]
    functional_tests_passed: bool
    data_gate_passed: bool
    retrieval_gate_passed: bool
    operational_gate_passed: bool
    end_to_end_gate_passed: bool
    activation_eligible: bool
    production_recommendation_blocked: bool
    blocker_codes: tuple[str, ...]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        gates = (
            self.functional_tests_passed,
            self.data_gate_passed,
            self.retrieval_gate_passed,
            self.operational_gate_passed,
            self.end_to_end_gate_passed,
        )
        expected = all(gates)
        if self.activation_eligible != expected:
            raise ValueError("activation eligibility conflicts with gate states")
        if self.production_recommendation_blocked == expected:
            raise ValueError(
                "production blocked flag must invert activation eligibility"
            )
        if expected == bool(self.blocker_codes):
            raise ValueError("eligibility blocker codes conflict with gate states")
        if self.blocker_codes != tuple(sorted(set(self.blocker_codes))):
            raise ValueError("eligibility blocker codes must be sorted and unique")
        return self


class CandidateValidationArtifact(_StrictFrozenModel):
    """Complete offline comparison and eligibility decision for one fixed dataset."""

    schema_version: Literal["candidate_validation_artifact_v2"]
    benchmark_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gold_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_report: EvaluationReport
    candidate_report: EvaluationReport
    comparison: PairedComparisonReport
    operational: OperationalBenchmarkOutcome
    end_to_end: EndToEndQualityOutcome
    eligibility: ActivationEligibilityReport


def _metric(
    values: tuple[PairedMetricComparison, ...],
    *,
    name: str,
    k: int | None,
) -> PairedMetricComparison:
    matches = tuple(
        value for value in values if value.metric_name == name and value.k == k
    )
    if len(matches) != 1:
        raise EvaluationEligibilityError(
            f"required paired metric is absent or duplicated: {name}@{k}"
        )
    return matches[0]


def evaluate_activation_eligibility(
    *,
    benchmark: RagBenchmarkConfig,
    comparison: PairedComparisonReport,
    operational: OperationalBenchmarkOutcome,
    end_to_end: EndToEndQualityOutcome,
    functional_tests_passed: bool,
) -> ActivationEligibilityReport:
    """Apply every configured offline/operational/final-answer production gate."""

    blockers: list[str] = []
    if not functional_tests_passed:
        blockers.append("functional_tests_failed")
    data_gate_passed = (
        comparison.rollout_data_gate_passed
        and comparison.eligible_queries_only
        and comparison.paired_query_count >= benchmark.min_global_gold_queries
    )
    if not data_gate_passed:
        blockers.append("rollout_data_gate_failed")
        blockers.extend(comparison.blocker_codes)

    gates = benchmark.gates
    recall = _metric(
        comparison.global_metrics,
        name="evidence_recall_at_k",
        k=5,
    )
    mrr = _metric(comparison.global_metrics, name="mrr", k=None)
    noise = _metric(comparison.global_metrics, name="noise_at_k", k=5)
    retrieval_blockers: list[str] = []
    if recall.baseline_mean >= gates.high_baseline_recall_threshold:
        baseline_error = 1.0 - recall.baseline_mean
        error_reduction = (
            1.0
            if baseline_error == 0.0
            else (recall.candidate_mean - recall.baseline_mean) / baseline_error
        )
        if (
            recall.candidate_mean < recall.baseline_mean
            or error_reduction < gates.high_baseline_relative_error_reduction
            or recall.ci_lower < -gates.high_baseline_noninferiority_margin
        ):
            retrieval_blockers.append("high_baseline_recall_gate_failed")
    elif (
        recall.mean_delta < gates.recall_at_5_min_absolute_gain
        or recall.ci_lower < gates.recall_at_5_ci_lower_bound_min
    ):
        retrieval_blockers.append("recall_at_5_gain_gate_failed")
    if (
        mrr.mean_delta < gates.mrr_min_absolute_gain
        or mrr.ci_lower < gates.mrr_ci_lower_bound_min
    ):
        retrieval_blockers.append("mrr_gain_gate_failed")
    if (
        noise.mean_delta > gates.noise_at_5_max_absolute_increase
        or noise.ci_upper > gates.noise_at_5_ci_upper_bound_max
    ):
        retrieval_blockers.append("noise_at_5_gate_failed")

    subject_by_id = {subject.subject: subject for subject in comparison.subjects}
    if set(subject_by_id) != set(benchmark.primary_subjects):
        retrieval_blockers.append("primary_subject_comparison_set_mismatch")
    else:
        for subject in benchmark.primary_subjects:
            subject_recall = _metric(
                subject_by_id[subject].metrics,
                name="evidence_recall_at_k",
                k=5,
            )
            if subject_recall.ci_lower < gates.per_subject_recall_ci_lower_bound_min:
                retrieval_blockers.append(f"{subject}:recall_regression_gate_failed")
    retrieval_gate_passed = not retrieval_blockers
    blockers.extend(retrieval_blockers)

    latency_ratio = (
        operational.candidate_p95_latency_ms / operational.baseline_p95_latency_ms
    )
    operational_blockers: list[str] = []
    if (
        latency_ratio > gates.p95_latency_max_baseline_ratio
        or operational.candidate_p95_latency_ms > gates.p95_latency_absolute_budget_ms
    ):
        operational_blockers.append("p95_latency_gate_failed")
    if operational.parent_context_token_ratio > gates.parent_context_max_baseline_ratio:
        operational_blockers.append("parent_context_token_gate_failed")
    if operational.baseline_error_count or operational.candidate_error_count:
        operational_blockers.append("benchmark_retrieval_error_detected")
    if operational.orphan_child_count:
        operational_blockers.append("orphan_child_detected")
    if operational.parent_hydration_failure_count:
        operational_blockers.append("parent_hydration_failure_detected")
    if operational.generation_mismatch_count:
        operational_blockers.append("generation_mismatch_detected")
    operational_gate_passed = not operational_blockers
    blockers.extend(operational_blockers)

    end_to_end_blockers: list[str] = []
    if (
        end_to_end.candidate_answer_correctness
        < end_to_end.baseline_answer_correctness
        - gates.answer_correctness_noninferiority_margin
    ):
        end_to_end_blockers.append("answer_correctness_gate_failed")
    if (
        end_to_end.candidate_citation_support
        < end_to_end.baseline_citation_support
        - gates.citation_support_noninferiority_margin
    ):
        end_to_end_blockers.append("citation_support_gate_failed")
    if (
        end_to_end.candidate_hallucination_rate
        > end_to_end.baseline_hallucination_rate
        + gates.hallucination_max_absolute_increase
    ):
        end_to_end_blockers.append("hallucination_gate_failed")
    end_to_end_gate_passed = not end_to_end_blockers
    blockers.extend(end_to_end_blockers)

    activation_eligible = all(
        (
            functional_tests_passed,
            data_gate_passed,
            retrieval_gate_passed,
            operational_gate_passed,
            end_to_end_gate_passed,
        )
    )
    return ActivationEligibilityReport(
        schema_version="activation_eligibility_report_v1",
        functional_tests_passed=functional_tests_passed,
        data_gate_passed=data_gate_passed,
        retrieval_gate_passed=retrieval_gate_passed,
        operational_gate_passed=operational_gate_passed,
        end_to_end_gate_passed=end_to_end_gate_passed,
        activation_eligible=activation_eligible,
        production_recommendation_blocked=not activation_eligible,
        blocker_codes=tuple(sorted(set(blockers))),
    )


__all__ = [
    "ActivationEligibilityReport",
    "CandidateValidationArtifact",
    "EndToEndQualityOutcome",
    "EvaluationEligibilityError",
    "OperationalBenchmarkOutcome",
    "evaluate_activation_eligibility",
]
