"""Safe paired benchmark projection for flat and parent-child retrieval runs.

The module deliberately produces validator inputs rather than computing formal
retrieval metrics.  Query and context bodies remain in memory; persisted
diagnostics contain only identifiers, bounded counters, timings, and reason
codes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.rag.parent_child.evaluation import (
    GoldDataset,
    GoldQuery,
    QueryRetrievalResult,
    RetrievalEvaluationInput,
    RetrievedEvidenceHit,
)
from src.rag.parent_child.flat_baseline import FlatBaselineRetrievalResult
from src.rag.parent_child.retrieval import HybridRetrievalResult


_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class BenchmarkExecutionError(RuntimeError):
    """A required benchmark arm failed; no partial result is a success."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class BenchmarkRunBinding(_StrictFrozenModel):
    """Identity that binds one retrieval arm to exactly one GoldDataset."""

    schema_version: Literal["benchmark_run_binding_v1"]
    run_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    gold_dataset_sha256: str = Field(pattern=_SHA256_PATTERN)
    embedding_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    retrieval_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    implementation_kind: Literal["flat_baseline", "parent_child_candidate"]
    artifact_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    generation_id: str | None

    @model_validator(mode="after")
    def _validate_implementation_contract(self) -> BenchmarkRunBinding:
        if self.implementation_kind == "flat_baseline":
            if self.generation_id is not None:
                raise ValueError("flat baseline binding must not carry a generation_id")
        elif not self.generation_id:
            raise ValueError("candidate binding requires an explicit generation_id")
        return self


class RetrievalLatencySummary(_StrictFrozenModel):
    """Safe aggregate timing distribution for one required retrieval stage."""

    schema_version: Literal["retrieval_latency_summary_v1"]
    p50_ms: float = Field(ge=0.0)
    p95_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _validate_finite(self) -> RetrievalLatencySummary:
        if not math.isfinite(self.p50_ms) or not math.isfinite(self.p95_ms):
            raise ValueError("latency summaries must be finite")
        if self.p95_ms < self.p50_ms:
            raise ValueError("p95 latency must be at least p50 latency")
        return self


class ArmOperationalSummary(_StrictFrozenModel):
    """Content-free operational aggregate for one benchmark arm."""

    schema_version: Literal["benchmark_arm_operational_summary_v1"]
    query_count: int = Field(gt=0)
    error_count: int = Field(ge=0)
    error_rate: float = Field(ge=0.0, le=1.0)
    vector: RetrievalLatencySummary
    bm25: RetrievalLatencySummary
    reranker: RetrievalLatencySummary
    hydrate: RetrievalLatencySummary
    total: RetrievalLatencySummary
    context_tokens_total: int = Field(ge=0)
    context_tokens_mean: float = Field(ge=0.0)
    context_tokens_p95: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _validate_error_rate(self) -> ArmOperationalSummary:
        expected = self.error_count / self.query_count
        if not math.isclose(self.error_rate, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("benchmark arm error_rate conflicts with counts")
        if not math.isfinite(self.context_tokens_mean) or not math.isfinite(
            self.context_tokens_p95
        ):
            raise ValueError("context-token summaries must be finite")
        return self


class BenchmarkOperationalDetails(_StrictFrozenModel):
    """Exact benchmark binding and operational facts for validation input."""

    schema_version: Literal["benchmark_operational_details_v1"]
    dataset_id: str = Field(min_length=1)
    gold_dataset_sha256: str = Field(pattern=_SHA256_PATTERN)
    baseline_run_id: str = Field(min_length=1)
    candidate_run_id: str = Field(min_length=1)
    candidate_generation_id: str = Field(min_length=1)
    embedding_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    baseline: ArmOperationalSummary
    candidate: ArmOperationalSummary
    parent_context_token_ratio: float = Field(ge=0.0)
    orphan_child_count: int = Field(ge=0)
    parent_hydration_failure_count: int = Field(ge=0)
    generation_mismatch_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_operational_success(self) -> BenchmarkOperationalDetails:
        if self.baseline.error_count or self.candidate.error_count:
            raise ValueError(
                "successful benchmark details cannot contain retrieval errors"
            )
        if not math.isfinite(self.parent_context_token_ratio):
            raise ValueError("parent context ratio must be finite")
        return self


class QueryBenchmarkDiagnostic(_StrictFrozenModel):
    """Persistable query-level diagnostic with no question or context body."""

    schema_version: Literal["benchmark_query_diagnostic_v1"]
    query_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    arm: Literal["baseline", "candidate"]
    status: Literal["ok", "empty"]
    vector_ms: float = Field(ge=0.0)
    bm25_ms: float = Field(ge=0.0)
    reranker_ms: float = Field(ge=0.0)
    hydrate_ms: float = Field(ge=0.0)
    total_ms: float = Field(ge=0.0)
    returned_hit_count: int = Field(ge=0)
    context_tokens: int = Field(ge=0)
    hydrated_parent_count: int = Field(ge=0)
    hydration_state: Literal["not_applicable", "empty", "hydrated"]

    @model_validator(mode="after")
    def _validate_finite_timings(self) -> QueryBenchmarkDiagnostic:
        values = (
            self.vector_ms,
            self.bm25_ms,
            self.reranker_ms,
            self.hydrate_ms,
            self.total_ms,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("diagnostic timings must be finite")
        return self


class PairedBenchmarkExecution(_StrictFrozenModel):
    """All success artifacts assembled in memory before the CLI writes them."""

    schema_version: Literal["paired_benchmark_execution_v1"]
    baseline_input: RetrievalEvaluationInput
    candidate_input: RetrievalEvaluationInput
    operational: BenchmarkOperationalDetails
    diagnostics: tuple[QueryBenchmarkDiagnostic, ...]

    @model_validator(mode="after")
    def _validate_bindings(self) -> PairedBenchmarkExecution:
        baseline = self.baseline_input
        candidate = self.candidate_input
        if baseline.implementation_kind != "flat_baseline":
            raise ValueError("baseline input must be a flat baseline")
        if candidate.implementation_kind != "parent_child_candidate":
            raise ValueError("candidate input must be parent-child")
        identity_fields = ("dataset_id", "gold_dataset_sha256", "embedding_fingerprint")
        if any(
            getattr(baseline, field) != getattr(candidate, field)
            for field in identity_fields
        ):
            raise ValueError(
                "benchmark arms do not share exact dataset/embedding identity"
            )
        if (
            self.operational.dataset_id != baseline.dataset_id
            or self.operational.gold_dataset_sha256 != baseline.gold_dataset_sha256
            or self.operational.baseline_run_id != baseline.run_id
            or self.operational.candidate_run_id != candidate.run_id
            or self.operational.candidate_generation_id != candidate.generation_id
            or self.operational.embedding_fingerprint != baseline.embedding_fingerprint
        ):
            raise ValueError(
                "operational benchmark binding differs from retrieval inputs"
            )
        expected_diagnostics = len(baseline.results) + len(candidate.results)
        if len(self.diagnostics) != expected_diagnostics:
            raise ValueError(
                "benchmark diagnostics must contain one record per query and arm"
            )
        return self


@dataclass(frozen=True, slots=True)
class _ArmObservation:
    result: QueryRetrievalResult
    diagnostic: QueryBenchmarkDiagnostic


def _percentile(values: tuple[float, ...], probability: float) -> float:
    if not values:
        raise BenchmarkExecutionError(
            "cannot summarize an empty benchmark distribution"
        )
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be in [0, 1]")
    ordered = tuple(sorted(values))
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _latency_summary(values: tuple[float, ...]) -> RetrievalLatencySummary:
    return RetrievalLatencySummary(
        schema_version="retrieval_latency_summary_v1",
        p50_ms=_percentile(values, 0.5),
        p95_ms=_percentile(values, 0.95),
    )


def _project_section_path(
    section_path: tuple[str, ...],
) -> tuple[str, ...] | None:
    """Map the explicit domain empty path into the evaluation no-section state."""

    return section_path if section_path else None


def _project_baseline(
    *,
    query: GoldQuery,
    result: FlatBaselineRetrievalResult,
    token_counter: Callable[[str], int],
) -> _ArmObservation:
    hits = tuple(
        RetrievedEvidenceHit(
            schema_version="retrieved_evidence_hit_v1",
            hit_id=hit.document.metadata.chunk_id,
            rank=hit.rank,
            parent_id=None,
            parent_rank=None,
            doc_id=hit.document.metadata.doc_id,
            source_relpath=hit.document.metadata.source_relpath,
            pagination_kind=hit.document.metadata.pagination_kind,
            page_start=hit.document.metadata.page_start,
            page_end=hit.document.metadata.page_end,
            start_char=hit.document.metadata.start_char,
            end_char=hit.document.metadata.end_char,
            section_path=_project_section_path(hit.document.metadata.section_path),
        )
        for hit in result.hits
    )
    context_tokens = sum(token_counter(hit.document.content) for hit in result.hits)
    return _ArmObservation(
        result=QueryRetrievalResult(
            schema_version="query_retrieval_result_v1",
            query_id=query.query_id,
            subject=query.subject,
            hits=hits,
        ),
        diagnostic=QueryBenchmarkDiagnostic(
            schema_version="benchmark_query_diagnostic_v1",
            query_id=query.query_id,
            subject=query.subject,
            arm="baseline",
            status="ok" if hits else "empty",
            vector_ms=result.vector_ms,
            bm25_ms=result.bm25_ms,
            reranker_ms=result.reranker_ms,
            hydrate_ms=0.0,
            total_ms=result.total_ms,
            returned_hit_count=len(hits),
            context_tokens=context_tokens,
            hydrated_parent_count=0,
            hydration_state="not_applicable",
        ),
    )


def _project_candidate(
    *,
    query: GoldQuery,
    result: HybridRetrievalResult,
    token_counter: Callable[[str], int],
) -> _ArmObservation:
    parent_ranks = {parent.parent_id: parent.rank for parent in result.ranked_parents}
    hits = tuple(
        RetrievedEvidenceHit(
            schema_version="retrieved_evidence_hit_v1",
            hit_id=hit.document.metadata.child_id,
            rank=hit.final_rank,
            parent_id=hit.document.metadata.parent_id,
            parent_rank=parent_ranks[hit.document.metadata.parent_id],
            doc_id=hit.document.metadata.doc_id,
            source_relpath=hit.document.metadata.source_relpath,
            pagination_kind=hit.document.metadata.pagination_kind,
            page_start=hit.document.metadata.page_start,
            page_end=hit.document.metadata.page_end,
            start_char=hit.document.metadata.start_char,
            end_char=hit.document.metadata.end_char,
            section_path=_project_section_path(hit.document.metadata.section_path),
        )
        for hit in result.ranked_children
    )
    if result.status == "ok" and len(parent_ranks) != len(result.hydrated_parents):
        raise BenchmarkExecutionError("candidate parent hydration cardinality mismatch")
    context_tokens = 0
    for context in result.hydrated_parents:
        context_tokens += token_counter(context.heading)
        context_tokens += sum(
            token_counter(window.content) for window in context.windows
        )
    return _ArmObservation(
        result=QueryRetrievalResult(
            schema_version="query_retrieval_result_v1",
            query_id=query.query_id,
            subject=query.subject,
            hits=hits,
        ),
        diagnostic=QueryBenchmarkDiagnostic(
            schema_version="benchmark_query_diagnostic_v1",
            query_id=query.query_id,
            subject=query.subject,
            arm="candidate",
            status=result.status,
            vector_ms=result.timings.vector_ms,
            bm25_ms=result.timings.bm25_ms,
            reranker_ms=result.timings.reranker_ms,
            hydrate_ms=result.timings.hydrate_ms,
            total_ms=result.timings.total_ms,
            returned_hit_count=len(hits),
            context_tokens=context_tokens,
            hydrated_parent_count=len(result.hydrated_parents),
            hydration_state="hydrated" if result.status == "ok" else "empty",
        ),
    )


def _arm_summary(*, observations: tuple[_ArmObservation, ...]) -> ArmOperationalSummary:
    diagnostics = tuple(item.diagnostic for item in observations)

    def values(field: str) -> tuple[float, ...]:
        return tuple(float(getattr(item, field)) for item in diagnostics)

    context = tuple(float(item.context_tokens) for item in diagnostics)
    return ArmOperationalSummary(
        schema_version="benchmark_arm_operational_summary_v1",
        query_count=len(diagnostics),
        error_count=0,
        error_rate=0.0,
        vector=_latency_summary(values("vector_ms")),
        bm25=_latency_summary(values("bm25_ms")),
        reranker=_latency_summary(values("reranker_ms")),
        hydrate=_latency_summary(values("hydrate_ms")),
        total=_latency_summary(values("total_ms")),
        context_tokens_total=int(sum(context)),
        context_tokens_mean=sum(context) / len(context),
        context_tokens_p95=_percentile(context, 0.95),
    )


def run_paired_benchmark(
    *,
    dataset: GoldDataset,
    baseline_binding: BenchmarkRunBinding,
    candidate_binding: BenchmarkRunBinding,
    baseline_retrieve: Callable[[GoldQuery], FlatBaselineRetrievalResult],
    candidate_retrieve: Callable[[GoldQuery], HybridRetrievalResult],
    token_counter: Callable[[str], int],
) -> PairedBenchmarkExecution:
    """Run both arms on one fixed dataset or raise before any success is written."""

    if (
        baseline_binding.dataset_id != dataset.dataset_id
        or candidate_binding.dataset_id != dataset.dataset_id
    ):
        raise BenchmarkExecutionError(
            "benchmark binding dataset_id differs from GoldDataset"
        )
    if baseline_binding.implementation_kind != "flat_baseline":
        raise BenchmarkExecutionError("baseline binding must select flat_baseline")
    if candidate_binding.implementation_kind != "parent_child_candidate":
        raise BenchmarkExecutionError(
            "candidate binding must select parent_child_candidate"
        )
    if (
        baseline_binding.gold_dataset_sha256 != candidate_binding.gold_dataset_sha256
        or baseline_binding.embedding_fingerprint
        != candidate_binding.embedding_fingerprint
    ):
        raise BenchmarkExecutionError(
            "benchmark arms do not share Gold/embedding identity"
        )
    baseline_observations: list[_ArmObservation] = []
    candidate_observations: list[_ArmObservation] = []
    for query in sorted(dataset.queries, key=lambda item: item.query_id):
        try:
            baseline = baseline_retrieve(query)
        except Exception as exc:
            raise BenchmarkExecutionError(
                f"baseline retrieval failed for query_id={query.query_id}"
            ) from exc
        baseline_observations.append(
            _project_baseline(query=query, result=baseline, token_counter=token_counter)
        )
        try:
            candidate = candidate_retrieve(query)
        except Exception as exc:
            raise BenchmarkExecutionError(
                f"candidate retrieval failed for query_id={query.query_id}"
            ) from exc
        if candidate.request.generation_id != candidate_binding.generation_id:
            raise BenchmarkExecutionError(
                "candidate result generation_id differs from binding"
            )
        if candidate.retrieval_fingerprint != candidate_binding.retrieval_fingerprint:
            raise BenchmarkExecutionError(
                "candidate retrieval fingerprint differs from binding"
            )
        candidate_observations.append(
            _project_candidate(
                query=query, result=candidate, token_counter=token_counter
            )
        )
    baseline_tuple = tuple(baseline_observations)
    candidate_tuple = tuple(candidate_observations)
    baseline_summary = _arm_summary(observations=baseline_tuple)
    candidate_summary = _arm_summary(observations=candidate_tuple)
    if baseline_summary.context_tokens_mean <= 0.0:
        raise BenchmarkExecutionError("baseline context-token mean must be positive")
    operational = BenchmarkOperationalDetails(
        schema_version="benchmark_operational_details_v1",
        dataset_id=dataset.dataset_id,
        gold_dataset_sha256=baseline_binding.gold_dataset_sha256,
        baseline_run_id=baseline_binding.run_id,
        candidate_run_id=candidate_binding.run_id,
        candidate_generation_id=candidate_binding.generation_id,
        embedding_fingerprint=baseline_binding.embedding_fingerprint,
        baseline=baseline_summary,
        candidate=candidate_summary,
        parent_context_token_ratio=(
            candidate_summary.context_tokens_mean / baseline_summary.context_tokens_mean
        ),
        orphan_child_count=0,
        parent_hydration_failure_count=0,
        generation_mismatch_count=0,
    )
    baseline_input = RetrievalEvaluationInput(
        schema_version="retrieval_evaluation_input_v2",
        run_id=baseline_binding.run_id,
        dataset_id=dataset.dataset_id,
        gold_dataset_sha256=baseline_binding.gold_dataset_sha256,
        embedding_fingerprint=baseline_binding.embedding_fingerprint,
        retrieval_fingerprint=baseline_binding.retrieval_fingerprint,
        implementation_kind="flat_baseline",
        artifact_manifest_sha256=baseline_binding.artifact_manifest_sha256,
        generation_id=None,
        parent_aware=False,
        results=tuple(item.result for item in baseline_tuple),
    )
    candidate_input = RetrievalEvaluationInput(
        schema_version="retrieval_evaluation_input_v2",
        run_id=candidate_binding.run_id,
        dataset_id=dataset.dataset_id,
        gold_dataset_sha256=candidate_binding.gold_dataset_sha256,
        embedding_fingerprint=candidate_binding.embedding_fingerprint,
        retrieval_fingerprint=candidate_binding.retrieval_fingerprint,
        implementation_kind="parent_child_candidate",
        artifact_manifest_sha256=candidate_binding.artifact_manifest_sha256,
        generation_id=candidate_binding.generation_id,
        parent_aware=True,
        results=tuple(item.result for item in candidate_tuple),
    )
    return PairedBenchmarkExecution(
        schema_version="paired_benchmark_execution_v1",
        baseline_input=baseline_input,
        candidate_input=candidate_input,
        operational=operational,
        diagnostics=tuple(
            item.diagnostic for item in (*baseline_tuple, *candidate_tuple)
        ),
    )


__all__ = [
    "ArmOperationalSummary",
    "BenchmarkExecutionError",
    "BenchmarkOperationalDetails",
    "BenchmarkRunBinding",
    "PairedBenchmarkExecution",
    "QueryBenchmarkDiagnostic",
    "RetrievalLatencySummary",
    "run_paired_benchmark",
]
