from __future__ import annotations

import pytest

from scripts.validate_parent_child_candidate import _validate_run_bindings
from src.rag.parent_child.evaluation import (
    GoldDataset,
    GoldEvidenceSpan,
    GoldQuery,
    QueryRetrievalResult,
    RetrievalEvaluationInput,
)
from src.rag.parent_child.evaluation_gate import (
    EndToEndQualityOutcome,
    OperationalBenchmarkOutcome,
)


_DIGEST = "a" * 64


def _dataset() -> GoldDataset:
    span = GoldEvidenceSpan(
        schema_version="gold_evidence_span_v1",
        gold_span_id="gold_validation",
        source_group_id="math-source-a",
        source_relpath="math/source.txt",
        doc_id="doc_" + "b" * 40,
        pagination_kind="logical",
        page_start=1,
        page_end=1,
        start_char=0,
        end_char=5,
        section_path=("Limits",),
        relevance_grade=3,
    )
    query = GoldQuery(
        schema_version="gold_query_v1",
        query_id="validation-query",
        subject="math",
        query="What is a limit?",
        dataset_kind="human_gold",
        eligible_for_rollout=True,
        gold_spans=(span,),
    )
    return GoldDataset(
        schema_version="gold_dataset_v1",
        dataset_id="validation-gold",
        queries=(query,),
    )


def _retrieval_inputs(
    dataset: GoldDataset,
) -> tuple[RetrievalEvaluationInput, RetrievalEvaluationInput]:
    result = QueryRetrievalResult(
        schema_version="query_retrieval_result_v1",
        query_id=dataset.queries[0].query_id,
        subject="math",
        hits=(),
    )
    baseline = RetrievalEvaluationInput(
        schema_version="retrieval_evaluation_input_v2",
        run_id="baseline-run",
        dataset_id=dataset.dataset_id,
        gold_dataset_sha256=_DIGEST,
        embedding_fingerprint="b" * 64,
        retrieval_fingerprint="c" * 64,
        implementation_kind="flat_baseline",
        artifact_manifest_sha256="d" * 64,
        generation_id=None,
        parent_aware=False,
        results=(result,),
    )
    candidate = RetrievalEvaluationInput(
        schema_version="retrieval_evaluation_input_v2",
        run_id="candidate-run",
        dataset_id=dataset.dataset_id,
        gold_dataset_sha256=_DIGEST,
        embedding_fingerprint="b" * 64,
        retrieval_fingerprint="e" * 64,
        implementation_kind="parent_child_candidate",
        artifact_manifest_sha256="f" * 64,
        generation_id="candidate-generation",
        parent_aware=True,
        results=(result,),
    )
    return baseline, candidate


def _operational(
    baseline: RetrievalEvaluationInput,
    candidate: RetrievalEvaluationInput,
) -> OperationalBenchmarkOutcome:
    return OperationalBenchmarkOutcome(
        schema_version="operational_benchmark_outcome_v2",
        dataset_id=baseline.dataset_id,
        gold_dataset_sha256=_DIGEST,
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        candidate_generation_id="candidate-generation",
        embedding_fingerprint=baseline.embedding_fingerprint,
        baseline_artifact_manifest_sha256=baseline.artifact_manifest_sha256,
        candidate_artifact_manifest_sha256=candidate.artifact_manifest_sha256,
        query_count=1,
        baseline_p50_latency_ms=1.0,
        baseline_p95_latency_ms=1.0,
        candidate_p50_latency_ms=1.0,
        candidate_p95_latency_ms=1.0,
        baseline_error_count=0,
        candidate_error_count=0,
        baseline_error_rate=0.0,
        candidate_error_rate=0.0,
        baseline_context_tokens_total=1,
        candidate_context_tokens_total=1,
        baseline_context_tokens_mean=1.0,
        candidate_context_tokens_mean=1.0,
        baseline_context_tokens_p95=1.0,
        candidate_context_tokens_p95=1.0,
        parent_context_token_ratio=1.0,
        parent_hydration_attempt_count=0,
        parent_hydration_success_count=0,
        orphan_child_count=0,
        parent_hydration_failure_count=0,
        generation_mismatch_count=0,
    )


def _end_to_end(
    baseline: RetrievalEvaluationInput,
    candidate: RetrievalEvaluationInput,
) -> EndToEndQualityOutcome:
    return EndToEndQualityOutcome(
        schema_version="end_to_end_quality_outcome_v2",
        dataset_id=baseline.dataset_id,
        gold_dataset_sha256=_DIGEST,
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        answer_model_fingerprint="1" * 64,
        assessment_protocol_sha256="2" * 64,
        assessment_source="human",
        scored_query_count=1,
        baseline_answer_correctness=1.0,
        candidate_answer_correctness=1.0,
        baseline_citation_support=1.0,
        candidate_citation_support=1.0,
        baseline_hallucination_rate=0.0,
        candidate_hallucination_rate=0.0,
        baseline_context_tokens_total=1,
        candidate_context_tokens_total=1,
        baseline_context_tokens_mean=1.0,
        candidate_context_tokens_mean=1.0,
    )


def test_validation_rejects_operational_generation_mixed_with_candidate_input() -> None:
    dataset = _dataset()
    baseline, candidate = _retrieval_inputs(dataset)
    operational = _operational(baseline, candidate)
    end_to_end = _end_to_end(baseline, candidate)

    _validate_run_bindings(
        dataset=dataset,
        gold_dataset_sha256=_DIGEST,
        baseline_input=baseline,
        candidate_input=candidate,
        operational=operational,
        end_to_end=end_to_end,
    )

    mismatched = operational.model_copy(
        update={"candidate_generation_id": "another-generation"}
    )
    with pytest.raises(ValueError, match="operational outcome"):
        _validate_run_bindings(
            dataset=dataset,
            gold_dataset_sha256=_DIGEST,
            baseline_input=baseline,
            candidate_input=candidate,
            operational=mismatched,
            end_to_end=end_to_end,
        )
