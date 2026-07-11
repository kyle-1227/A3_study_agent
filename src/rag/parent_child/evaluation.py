"""Strict, retrieval-agnostic evaluation for parent-child RAG runs.

Gold evidence is expressed only in source-document coordinates.  The module
consumes a small projection of retrieval output and never calls or reimplements
the production retriever.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import PurePosixPath
import random
import re
from statistics import fmean
from typing import Hashable, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DatasetKind = Literal["synthetic_smoke", "human_gold", "historical_annotated"]
_RecallItemT = TypeVar("_RecallItemT", bound=Hashable)
MetricName = Literal[
    "evidence_recall_at_k",
    "mrr",
    "parent_recall_at_k",
    "source_recall_at_k",
    "section_recall_at_k",
    "noise_at_k",
]

_DOC_ID_RE = re.compile(r"^doc_[0-9a-f]{40}$")
_GOLD_SPAN_ID_RE = re.compile(r"^gold_[A-Za-z0-9._-]+$")


class EvaluationContractError(ValueError):
    """Raised when separately valid evaluation artifacts disagree."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _validate_identifier(value: str, *, field_name: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty and already stripped")
    if any(character in value for character in ("/", "\\", "\x00")):
        raise ValueError(f"{field_name} must not contain path separators or NUL")
    return value


def _validate_subject(value: str) -> str:
    _validate_identifier(value, field_name="subject")
    if value != value.casefold():
        raise ValueError("subject must already be case-folded")
    if (
        value.startswith("_")
        or value.endswith("_")
        or "__" in value
        or not all(character.isalnum() or character == "_" for character in value)
    ):
        raise ValueError("subject must be a normalized identifier")
    return value


def _validate_source_relpath(value: str) -> str:
    if not value or value != value.strip() or "\\" in value or "\x00" in value:
        raise ValueError(
            "source_relpath must be non-empty, stripped, POSIX, and contain no NUL"
        )
    parts = value.split("/")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or parts[0].endswith(":")
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError("source_relpath must be a contained POSIX relative path")
    return path.as_posix()


def _validate_section_path(
    value: tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not value:
        raise ValueError("section_path must be None or a non-empty tuple")
    for item in value:
        if not item or item != item.strip():
            raise ValueError("section_path items must be non-empty and stripped")
    return value


class GoldEvidenceSpan(_StrictFrozenModel):
    """Policy-independent relevant evidence in cleaned-source coordinates."""

    schema_version: Literal["gold_evidence_span_v1"]
    gold_span_id: str
    source_group_id: str
    source_relpath: str
    doc_id: str
    pagination_kind: Literal["physical", "logical"]
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    section_path: tuple[str, ...] | None
    relevance_grade: int = Field(ge=1, le=3)

    @field_validator("gold_span_id")
    @classmethod
    def validate_gold_span_id(cls, value: str) -> str:
        if _GOLD_SPAN_ID_RE.fullmatch(value) is None:
            raise ValueError("gold_span_id must use the gold_ namespace")
        return value

    @field_validator("source_group_id")
    @classmethod
    def validate_source_group_id(cls, value: str) -> str:
        return _validate_identifier(value, field_name="source_group_id")

    @field_validator("source_relpath")
    @classmethod
    def validate_source_relpath(cls, value: str) -> str:
        return _validate_source_relpath(value)

    @field_validator("doc_id")
    @classmethod
    def validate_doc_id(cls, value: str) -> str:
        if _DOC_ID_RE.fullmatch(value) is None:
            raise ValueError("doc_id must use the policy-independent doc_ SHA1 form")
        return value

    @field_validator("section_path")
    @classmethod
    def validate_section_path(
        cls, value: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        return _validate_section_path(value)

    @model_validator(mode="after")
    def validate_span(self) -> GoldEvidenceSpan:
        if self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start")
        if self.end_char <= self.start_char:
            raise ValueError("gold cleaned-character span must be non-empty")
        return self


class GoldQuery(_StrictFrozenModel):
    """One versioned query with one or more source-coordinate gold spans."""

    schema_version: Literal["gold_query_v1"]
    query_id: str
    subject: str
    query: str
    dataset_kind: DatasetKind
    eligible_for_rollout: bool
    gold_spans: tuple[GoldEvidenceSpan, ...] = Field(min_length=1)

    @field_validator("query_id")
    @classmethod
    def validate_query_id(cls, value: str) -> str:
        return _validate_identifier(value, field_name="query_id")

    @field_validator("subject")
    @classmethod
    def validate_subject(cls, value: str) -> str:
        return _validate_subject(value)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("query must be non-empty and already stripped")
        return value

    @model_validator(mode="after")
    def validate_gold_query(self) -> GoldQuery:
        if self.dataset_kind == "synthetic_smoke" and self.eligible_for_rollout:
            raise ValueError("synthetic_smoke queries are never rollout eligible")
        span_ids = tuple(span.gold_span_id for span in self.gold_spans)
        if len(span_ids) != len(set(span_ids)):
            raise ValueError(
                "gold_spans must not contain duplicate gold_span_id values"
            )
        coordinates = tuple(
            (
                span.doc_id,
                span.start_char,
                span.end_char,
                span.relevance_grade,
            )
            for span in self.gold_spans
        )
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("gold_spans must not duplicate source coordinates")
        return self


class GoldDataset(_StrictFrozenModel):
    """A fixed policy-independent query set shared by all compared runs."""

    schema_version: Literal["gold_dataset_v1"]
    dataset_id: str
    queries: tuple[GoldQuery, ...] = Field(min_length=1)

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, value: str) -> str:
        return _validate_identifier(value, field_name="dataset_id")

    @model_validator(mode="after")
    def validate_unique_queries(self) -> GoldDataset:
        query_ids = tuple(query.query_id for query in self.queries)
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("queries must not contain duplicate query_id values")
        return self


class RetrievedEvidenceHit(_StrictFrozenModel):
    """Evaluation projection of one ranked production retrieval hit."""

    schema_version: Literal["retrieved_evidence_hit_v1"]
    hit_id: str
    rank: int = Field(ge=1)
    parent_id: str | None
    parent_rank: int | None
    doc_id: str
    source_relpath: str
    pagination_kind: Literal["physical", "logical"]
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    section_path: tuple[str, ...] | None

    @field_validator("hit_id")
    @classmethod
    def validate_hit_id(cls, value: str) -> str:
        return _validate_identifier(value, field_name="hit_id")

    @field_validator("parent_id")
    @classmethod
    def validate_parent_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_identifier(value, field_name="parent_id")

    @field_validator("doc_id")
    @classmethod
    def validate_doc_id(cls, value: str) -> str:
        if _DOC_ID_RE.fullmatch(value) is None:
            raise ValueError("doc_id must use the doc_ SHA1 form")
        return value

    @field_validator("source_relpath")
    @classmethod
    def validate_source_relpath(cls, value: str) -> str:
        return _validate_source_relpath(value)

    @field_validator("section_path")
    @classmethod
    def validate_section_path(
        cls, value: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        return _validate_section_path(value)

    @model_validator(mode="after")
    def validate_hit(self) -> RetrievedEvidenceHit:
        if (self.parent_id is None) != (self.parent_rank is None):
            raise ValueError(
                "parent_id and parent_rank must either both exist or both be None"
            )
        if self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start")
        if self.end_char <= self.start_char:
            raise ValueError("retrieved cleaned-character span must be non-empty")
        return self


class QueryRetrievalResult(_StrictFrozenModel):
    """Ordered retrieval hits for exactly one gold query."""

    schema_version: Literal["query_retrieval_result_v1"]
    query_id: str
    subject: str
    hits: tuple[RetrievedEvidenceHit, ...]

    @field_validator("query_id")
    @classmethod
    def validate_query_id(cls, value: str) -> str:
        return _validate_identifier(value, field_name="query_id")

    @field_validator("subject")
    @classmethod
    def validate_subject(cls, value: str) -> str:
        return _validate_subject(value)

    @model_validator(mode="after")
    def validate_hit_order(self) -> QueryRetrievalResult:
        expected_ranks = tuple(range(1, len(self.hits) + 1))
        if tuple(hit.rank for hit in self.hits) != expected_ranks:
            raise ValueError("hit ranks must be contiguous, 1-based, and tuple-ordered")
        hit_ids = tuple(hit.hit_id for hit in self.hits)
        if len(hit_ids) != len(set(hit_ids)):
            raise ValueError("hits must not contain duplicate hit_id values")

        parent_ranks: dict[str, int] = {}
        rank_owners: dict[int, str] = {}
        for hit in self.hits:
            if hit.parent_id is None or hit.parent_rank is None:
                continue
            existing_rank = parent_ranks.setdefault(hit.parent_id, hit.parent_rank)
            if existing_rank != hit.parent_rank:
                raise ValueError("one parent_id must have exactly one parent_rank")
            existing_owner = rank_owners.setdefault(hit.parent_rank, hit.parent_id)
            if existing_owner != hit.parent_id:
                raise ValueError("one parent_rank must identify exactly one parent_id")
        if parent_ranks:
            expected_parent_ranks = set(range(1, len(parent_ranks) + 1))
            if set(parent_ranks.values()) != expected_parent_ranks:
                raise ValueError("parent ranks must be contiguous and 1-based")
        return self


class RetrievalEvaluationInput(_StrictFrozenModel):
    """Complete retrieval projection for one implementation under test."""

    schema_version: Literal["retrieval_evaluation_input_v2"]
    run_id: str
    dataset_id: str
    gold_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieval_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_kind: Literal["flat_baseline", "parent_child_candidate"]
    artifact_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_id: str | None
    parent_aware: bool
    results: tuple[QueryRetrievalResult, ...] = Field(min_length=1)

    @field_validator("run_id", "dataset_id")
    @classmethod
    def validate_ids(cls, value: str, info: object) -> str:
        return _validate_identifier(
            value, field_name=str(getattr(info, "field_name", "identifier"))
        )

    @model_validator(mode="after")
    def validate_unique_results(self) -> RetrievalEvaluationInput:
        query_ids = tuple(result.query_id for result in self.results)
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("results must not contain duplicate query_id values")
        if self.implementation_kind == "flat_baseline":
            if self.parent_aware or self.generation_id is not None:
                raise ValueError(
                    "flat_baseline retrieval inputs must not be parent-aware or "
                    "carry a generation_id"
                )
        elif not self.parent_aware or not self.generation_id:
            raise ValueError(
                "parent_child_candidate retrieval inputs require parent-aware "
                "results and a generation_id"
            )
        return self


class EvaluationGateConfig(_StrictFrozenModel):
    """Minimum real-gold inventory required before rollout can be considered."""

    schema_version: Literal["evaluation_gate_config_v1"]
    primary_subjects: tuple[str, ...] = Field(min_length=1)
    min_global_rollout_queries: int = Field(gt=0)
    min_subject_rollout_queries: int = Field(gt=0)
    min_independent_sources_per_subject: int = Field(gt=0)

    @field_validator("primary_subjects")
    @classmethod
    def validate_primary_subjects(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_validate_subject(subject) for subject in value)
        if len(normalized) != len(set(normalized)):
            raise ValueError("primary_subjects must not contain duplicates")
        return normalized


class MetricAtK(_StrictFrozenModel):
    schema_version: Literal["metric_at_k_v1"]
    k: int = Field(gt=0)
    value: float | None

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
            raise ValueError("metric value must be finite and between zero and one")
        return value


class QueryMetrics(_StrictFrozenModel):
    schema_version: Literal["query_metrics_v1"]
    query_id: str
    subject: str
    dataset_kind: DatasetKind
    eligible_for_rollout: bool
    evidence_recall: tuple[MetricAtK, ...]
    parent_recall: tuple[MetricAtK, ...]
    source_recall: tuple[MetricAtK, ...]
    section_recall: tuple[MetricAtK, ...]
    noise: tuple[MetricAtK, ...]
    mrr: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_metric_k_sets(self) -> QueryMetrics:
        for field_name in (
            "evidence_recall",
            "parent_recall",
            "source_recall",
            "section_recall",
            "noise",
        ):
            values = getattr(self, field_name)
            ks = tuple(metric.k for metric in values)
            if tuple(sorted(set(ks))) != ks:
                raise ValueError(f"{field_name} k values must be sorted and unique")
        return self


class AggregateMetricAtK(_StrictFrozenModel):
    schema_version: Literal["aggregate_metric_at_k_v1"]
    k: int = Field(gt=0)
    value: float | None
    defined_query_count: int = Field(ge=0)

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
            raise ValueError("aggregate metric value must be finite and in [0,1]")
        return value

    @model_validator(mode="after")
    def validate_defined_count(self) -> AggregateMetricAtK:
        if (self.value is None) != (self.defined_query_count == 0):
            raise ValueError("value is None exactly when defined_query_count is zero")
        return self


class AggregateMetrics(_StrictFrozenModel):
    schema_version: Literal["aggregate_metrics_v1"]
    query_count: int = Field(ge=0)
    evidence_recall: tuple[AggregateMetricAtK, ...]
    parent_recall: tuple[AggregateMetricAtK, ...]
    source_recall: tuple[AggregateMetricAtK, ...]
    section_recall: tuple[AggregateMetricAtK, ...]
    noise: tuple[AggregateMetricAtK, ...]
    mrr: float | None
    mrr_defined_query_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_mrr(self) -> AggregateMetrics:
        if self.mrr is not None and (
            not math.isfinite(self.mrr) or not 0.0 <= self.mrr <= 1.0
        ):
            raise ValueError("aggregate MRR must be finite and in [0,1]")
        if (self.mrr is None) != (self.mrr_defined_query_count == 0):
            raise ValueError("mrr is None exactly when its defined count is zero")
        if self.mrr_defined_query_count > self.query_count:
            raise ValueError("MRR defined count must not exceed query_count")
        return self


class SubjectMetrics(_StrictFrozenModel):
    schema_version: Literal["subject_metrics_v1"]
    subject: str
    all_queries: AggregateMetrics
    rollout_eligible_queries: AggregateMetrics


class SubjectDataGate(_StrictFrozenModel):
    schema_version: Literal["subject_data_gate_v1"]
    subject: str
    rollout_query_count: int = Field(ge=0)
    independent_source_count: int = Field(ge=0)
    passed: bool
    blocker_codes: tuple[str, ...]


class DatasetRolloutGate(_StrictFrozenModel):
    schema_version: Literal["dataset_rollout_gate_v1"]
    passed: bool
    production_recommendation_blocked: bool
    rollout_query_count: int = Field(ge=0)
    subject_gates: tuple[SubjectDataGate, ...]
    blocker_codes: tuple[str, ...]

    @model_validator(mode="after")
    def validate_gate_state(self) -> DatasetRolloutGate:
        if self.production_recommendation_blocked != self.passed:
            return self
        raise ValueError(
            "production_recommendation_blocked must be the inverse of passed"
        )


class EvaluationReport(_StrictFrozenModel):
    """Complete per-query and aggregate metrics for one retrieval run."""

    schema_version: Literal["parent_child_evaluation_report_v1"]
    run_id: str
    dataset_id: str
    parent_aware: bool
    top_ks: tuple[int, ...]
    parent_top_ks: tuple[int, ...]
    per_query: tuple[QueryMetrics, ...]
    global_metrics: AggregateMetrics
    rollout_eligible_global_metrics: AggregateMetrics
    subjects: tuple[SubjectMetrics, ...]
    rollout_data_gate: DatasetRolloutGate


class BootstrapConfig(_StrictFrozenModel):
    schema_version: Literal["paired_bootstrap_config_v1"]
    iterations: int = Field(gt=0)
    seed: int = Field(ge=0)
    confidence: float = Field(gt=0.0, lt=1.0)


class ComparisonMetricSpec(_StrictFrozenModel):
    schema_version: Literal["comparison_metric_spec_v1"]
    metric_name: MetricName
    k: int | None

    @model_validator(mode="after")
    def validate_k(self) -> ComparisonMetricSpec:
        if self.metric_name == "mrr":
            if self.k is not None:
                raise ValueError("MRR comparison must not specify k")
        elif self.k is None or self.k <= 0:
            raise ValueError("at-k comparison metrics require a positive k")
        return self


class PairedMetricComparison(_StrictFrozenModel):
    schema_version: Literal["paired_metric_comparison_v1"]
    metric_name: MetricName
    k: int | None
    sample_count: int = Field(gt=0)
    baseline_mean: float
    candidate_mean: float
    mean_delta: float
    confidence: float = Field(gt=0.0, lt=1.0)
    ci_lower: float
    ci_upper: float

    @model_validator(mode="after")
    def validate_finite_values(self) -> PairedMetricComparison:
        values = (
            self.baseline_mean,
            self.candidate_mean,
            self.mean_delta,
            self.ci_lower,
            self.ci_upper,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("paired comparison values must be finite")
        if not 0.0 <= self.baseline_mean <= 1.0:
            raise ValueError("baseline_mean must be in [0,1]")
        if not 0.0 <= self.candidate_mean <= 1.0:
            raise ValueError("candidate_mean must be in [0,1]")
        if self.ci_lower > self.ci_upper:
            raise ValueError("ci_lower must not exceed ci_upper")
        return self


class SubjectPairedComparison(_StrictFrozenModel):
    schema_version: Literal["subject_paired_comparison_v1"]
    subject: str
    metrics: tuple[PairedMetricComparison, ...]


class PairedComparisonReport(_StrictFrozenModel):
    schema_version: Literal["paired_comparison_report_v1"]
    dataset_id: str
    baseline_run_id: str
    candidate_run_id: str
    eligible_queries_only: bool
    paired_query_count: int = Field(gt=0)
    bootstrap: BootstrapConfig
    global_metrics: tuple[PairedMetricComparison, ...]
    subjects: tuple[SubjectPairedComparison, ...]
    rollout_data_gate_passed: bool
    production_recommendation_blocked: bool
    blocker_codes: tuple[str, ...]

    @model_validator(mode="after")
    def validate_rollout_state(self) -> PairedComparisonReport:
        if self.production_recommendation_blocked != self.rollout_data_gate_passed:
            return self
        raise ValueError(
            "production_recommendation_blocked must invert rollout_data_gate_passed"
        )


def _validate_ks(values: tuple[int, ...], *, field_name: str) -> tuple[int, ...]:
    if not values:
        raise EvaluationContractError(f"{field_name} must not be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise EvaluationContractError(f"{field_name} must contain positive integers")
    if tuple(sorted(set(values))) != values:
        raise EvaluationContractError(f"{field_name} must be sorted and unique")
    return values


def _span_matches(hit: RetrievedEvidenceHit, gold: GoldEvidenceSpan) -> bool:
    return (
        hit.doc_id == gold.doc_id
        and hit.source_relpath == gold.source_relpath
        and hit.pagination_kind == gold.pagination_kind
        and hit.page_start <= gold.page_end
        and gold.page_start <= hit.page_end
        and hit.start_char < gold.end_char
        and gold.start_char < hit.end_char
    )


def _recall(matched: set[_RecallItemT], expected: set[_RecallItemT]) -> float:
    if not expected:
        raise EvaluationContractError("recall requires at least one expected item")
    return len(matched & expected) / len(expected)


def _metric_at_k(k: int, value: float | None) -> MetricAtK:
    return MetricAtK(schema_version="metric_at_k_v1", k=k, value=value)


def _evaluate_query(
    query: GoldQuery,
    result: QueryRetrievalResult,
    *,
    parent_aware: bool,
    top_ks: tuple[int, ...],
    parent_top_ks: tuple[int, ...],
) -> QueryMetrics:
    gold_by_id = {span.gold_span_id: span for span in query.gold_spans}
    expected_gold_ids = set(gold_by_id)
    evidence_values: list[MetricAtK] = []
    source_values: list[MetricAtK] = []
    section_values: list[MetricAtK] = []
    noise_values: list[MetricAtK] = []

    expected_sources = {span.source_relpath for span in query.gold_spans}
    expected_sections = {
        (span.source_relpath, span.section_path)
        for span in query.gold_spans
        if span.section_path is not None
    }
    for k in top_ks:
        hits = result.hits[:k]
        matched_gold = {
            gold_id
            for hit in hits
            for gold_id, gold in gold_by_id.items()
            if _span_matches(hit, gold)
        }
        evidence_values.append(
            _metric_at_k(k, _recall(matched_gold, expected_gold_ids))
        )
        retrieved_sources = {hit.source_relpath for hit in hits}
        source_values.append(
            _metric_at_k(k, _recall(retrieved_sources, expected_sources))
        )
        if expected_sections:
            retrieved_sections = {
                (hit.source_relpath, hit.section_path)
                for hit in hits
                if hit.section_path is not None
            }
            section_value: float | None = _recall(retrieved_sections, expected_sections)
        else:
            section_value = None
        section_values.append(_metric_at_k(k, section_value))
        if hits:
            irrelevant_count = sum(
                1
                for hit in hits
                if not any(_span_matches(hit, gold) for gold in query.gold_spans)
            )
            noise_value: float | None = irrelevant_count / len(hits)
        else:
            noise_value = None
        noise_values.append(_metric_at_k(k, noise_value))

    parent_values: list[MetricAtK] = []
    if parent_aware:
        for k in parent_top_ks:
            selected_parent_ids = {
                hit.parent_id
                for hit in result.hits
                if hit.parent_rank is not None and hit.parent_rank <= k
            }
            matched_gold = {
                gold_id
                for hit in result.hits
                if hit.parent_id in selected_parent_ids
                for gold_id, gold in gold_by_id.items()
                if _span_matches(hit, gold)
            }
            parent_values.append(
                _metric_at_k(k, _recall(matched_gold, expected_gold_ids))
            )

    first_relevant_rank = next(
        (
            hit.rank
            for hit in result.hits
            if any(_span_matches(hit, gold) for gold in query.gold_spans)
        ),
        None,
    )
    mrr = 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank
    return QueryMetrics(
        schema_version="query_metrics_v1",
        query_id=query.query_id,
        subject=query.subject,
        dataset_kind=query.dataset_kind,
        eligible_for_rollout=query.eligible_for_rollout,
        evidence_recall=tuple(evidence_values),
        parent_recall=tuple(parent_values),
        source_recall=tuple(source_values),
        section_recall=tuple(section_values),
        noise=tuple(noise_values),
        mrr=mrr,
    )


def _mean_or_none(values: list[float]) -> float | None:
    return None if not values else fmean(values)


def _aggregate_at_k(
    query_metrics: tuple[QueryMetrics, ...], field_name: str, ks: tuple[int, ...]
) -> tuple[AggregateMetricAtK, ...]:
    output: list[AggregateMetricAtK] = []
    for k in ks:
        values: list[float] = []
        for query_metric in query_metrics:
            metrics: tuple[MetricAtK, ...] = getattr(query_metric, field_name)
            by_k = {metric.k: metric.value for metric in metrics}
            if k not in by_k:
                continue
            value = by_k[k]
            if value is not None:
                values.append(value)
        output.append(
            AggregateMetricAtK(
                schema_version="aggregate_metric_at_k_v1",
                k=k,
                value=_mean_or_none(values),
                defined_query_count=len(values),
            )
        )
    return tuple(output)


def _aggregate_metrics(
    query_metrics: tuple[QueryMetrics, ...],
    *,
    top_ks: tuple[int, ...],
    parent_top_ks: tuple[int, ...],
) -> AggregateMetrics:
    mrr_values = [metric.mrr for metric in query_metrics]
    return AggregateMetrics(
        schema_version="aggregate_metrics_v1",
        query_count=len(query_metrics),
        evidence_recall=_aggregate_at_k(query_metrics, "evidence_recall", top_ks),
        parent_recall=_aggregate_at_k(query_metrics, "parent_recall", parent_top_ks),
        source_recall=_aggregate_at_k(query_metrics, "source_recall", top_ks),
        section_recall=_aggregate_at_k(query_metrics, "section_recall", top_ks),
        noise=_aggregate_at_k(query_metrics, "noise", top_ks),
        mrr=_mean_or_none(mrr_values),
        mrr_defined_query_count=len(mrr_values),
    )


def _build_rollout_gate(
    dataset: GoldDataset, gate_config: EvaluationGateConfig
) -> DatasetRolloutGate:
    eligible_queries = tuple(
        query for query in dataset.queries if query.eligible_for_rollout
    )
    blocker_codes: list[str] = []
    if len(eligible_queries) < gate_config.min_global_rollout_queries:
        blocker_codes.append("global_rollout_gold_below_minimum")

    subject_gates: list[SubjectDataGate] = []
    for subject in gate_config.primary_subjects:
        subject_queries = tuple(
            query for query in eligible_queries if query.subject == subject
        )
        source_groups = {
            span.source_group_id
            for query in subject_queries
            for span in query.gold_spans
        }
        subject_blockers: list[str] = []
        if len(subject_queries) < gate_config.min_subject_rollout_queries:
            subject_blockers.append("rollout_gold_below_minimum")
        if len(source_groups) < gate_config.min_independent_sources_per_subject:
            subject_blockers.append("independent_sources_below_minimum")
        for code in subject_blockers:
            blocker_codes.append(f"{subject}:{code}")
        subject_gates.append(
            SubjectDataGate(
                schema_version="subject_data_gate_v1",
                subject=subject,
                rollout_query_count=len(subject_queries),
                independent_source_count=len(source_groups),
                passed=not subject_blockers,
                blocker_codes=tuple(subject_blockers),
            )
        )

    passed = not blocker_codes
    return DatasetRolloutGate(
        schema_version="dataset_rollout_gate_v1",
        passed=passed,
        production_recommendation_blocked=not passed,
        rollout_query_count=len(eligible_queries),
        subject_gates=tuple(subject_gates),
        blocker_codes=tuple(blocker_codes),
    )


def evaluate_retrieval_run(
    dataset: GoldDataset,
    retrieval_input: RetrievalEvaluationInput,
    *,
    top_ks: tuple[int, ...],
    parent_top_ks: tuple[int, ...],
    gate_config: EvaluationGateConfig,
) -> EvaluationReport:
    """Evaluate one run against exact source-coordinate gold evidence."""

    validated_top_ks = _validate_ks(top_ks, field_name="top_ks")
    if retrieval_input.parent_aware:
        validated_parent_top_ks = _validate_ks(
            parent_top_ks, field_name="parent_top_ks"
        )
    elif parent_top_ks:
        raise EvaluationContractError(
            "parent_top_ks must be empty for a non-parent-aware run"
        )
    else:
        validated_parent_top_ks = ()

    if retrieval_input.dataset_id != dataset.dataset_id:
        raise EvaluationContractError("retrieval input dataset_id does not match gold")
    gold_by_id = {query.query_id: query for query in dataset.queries}
    result_by_id = {result.query_id: result for result in retrieval_input.results}
    if set(gold_by_id) != set(result_by_id):
        missing = sorted(set(gold_by_id) - set(result_by_id))
        extra = sorted(set(result_by_id) - set(gold_by_id))
        raise EvaluationContractError(
            f"retrieval query set mismatch; missing={missing!r}, extra={extra!r}"
        )

    per_query: list[QueryMetrics] = []
    for gold_query in dataset.queries:
        result = result_by_id[gold_query.query_id]
        if result.subject != gold_query.subject:
            raise EvaluationContractError(
                f"subject mismatch for query_id={gold_query.query_id!r}"
            )
        for hit in result.hits:
            has_parent = hit.parent_id is not None and hit.parent_rank is not None
            if has_parent != retrieval_input.parent_aware:
                raise EvaluationContractError(
                    "hit parent fields must exactly match run parent_aware contract"
                )
        per_query.append(
            _evaluate_query(
                gold_query,
                result,
                parent_aware=retrieval_input.parent_aware,
                top_ks=validated_top_ks,
                parent_top_ks=validated_parent_top_ks,
            )
        )

    per_query_tuple = tuple(per_query)
    rollout_metrics = tuple(
        metric for metric in per_query_tuple if metric.eligible_for_rollout
    )
    subjects: list[SubjectMetrics] = []
    for subject in sorted({query.subject for query in dataset.queries}):
        all_subject_metrics = tuple(
            metric for metric in per_query_tuple if metric.subject == subject
        )
        rollout_subject_metrics = tuple(
            metric for metric in all_subject_metrics if metric.eligible_for_rollout
        )
        subjects.append(
            SubjectMetrics(
                schema_version="subject_metrics_v1",
                subject=subject,
                all_queries=_aggregate_metrics(
                    all_subject_metrics,
                    top_ks=validated_top_ks,
                    parent_top_ks=validated_parent_top_ks,
                ),
                rollout_eligible_queries=_aggregate_metrics(
                    rollout_subject_metrics,
                    top_ks=validated_top_ks,
                    parent_top_ks=validated_parent_top_ks,
                ),
            )
        )

    return EvaluationReport(
        schema_version="parent_child_evaluation_report_v1",
        run_id=retrieval_input.run_id,
        dataset_id=dataset.dataset_id,
        parent_aware=retrieval_input.parent_aware,
        top_ks=validated_top_ks,
        parent_top_ks=validated_parent_top_ks,
        per_query=per_query_tuple,
        global_metrics=_aggregate_metrics(
            per_query_tuple,
            top_ks=validated_top_ks,
            parent_top_ks=validated_parent_top_ks,
        ),
        rollout_eligible_global_metrics=_aggregate_metrics(
            rollout_metrics,
            top_ks=validated_top_ks,
            parent_top_ks=validated_parent_top_ks,
        ),
        subjects=tuple(subjects),
        rollout_data_gate=_build_rollout_gate(dataset, gate_config),
    )


def _lookup_metric(metric: QueryMetrics, spec: ComparisonMetricSpec) -> float | None:
    if spec.metric_name == "mrr":
        return metric.mrr
    field_name = {
        "evidence_recall_at_k": "evidence_recall",
        "parent_recall_at_k": "parent_recall",
        "source_recall_at_k": "source_recall",
        "section_recall_at_k": "section_recall",
        "noise_at_k": "noise",
    }[spec.metric_name]
    for value in getattr(metric, field_name):
        if value.k == spec.k:
            return value.value
    raise EvaluationContractError(
        f"metric {spec.metric_name!r} at k={spec.k!r} is absent from report"
    )


def _derived_seed(
    config: BootstrapConfig,
    spec: ComparisonMetricSpec,
    *,
    scope: str,
) -> int:
    payload = f"{config.seed}|{spec.metric_name}|{spec.k}|{scope}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise EvaluationContractError("percentile requires at least one value")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] + fraction * (
        sorted_values[upper] - sorted_values[lower]
    )


def _bootstrap_ci(
    paired_values: tuple[tuple[str, float, float], ...],
    *,
    config: BootstrapConfig,
    spec: ComparisonMetricSpec,
    scope: str,
) -> tuple[float, float]:
    by_subject: dict[str, list[float]] = defaultdict(list)
    for subject, baseline, candidate in paired_values:
        by_subject[subject].append(candidate - baseline)
    rng = random.Random(_derived_seed(config, spec, scope=scope))
    bootstrap_means: list[float] = []
    for _ in range(config.iterations):
        sampled: list[float] = []
        for subject in sorted(by_subject):
            subject_values = by_subject[subject]
            sampled.extend(
                subject_values[rng.randrange(len(subject_values))]
                for _ in range(len(subject_values))
            )
        bootstrap_means.append(fmean(sampled))
    bootstrap_means.sort()
    tail = (1.0 - config.confidence) / 2.0
    return (
        _percentile(bootstrap_means, tail),
        _percentile(bootstrap_means, 1.0 - tail),
    )


def _compare_metric(
    paired_values: tuple[tuple[str, float, float], ...],
    *,
    spec: ComparisonMetricSpec,
    bootstrap: BootstrapConfig,
    scope: str,
) -> PairedMetricComparison:
    if not paired_values:
        raise EvaluationContractError(
            f"metric {spec.metric_name!r} at k={spec.k!r} has no paired values"
        )
    baseline_values = [baseline for _, baseline, _ in paired_values]
    candidate_values = [candidate for _, _, candidate in paired_values]
    baseline_mean = fmean(baseline_values)
    candidate_mean = fmean(candidate_values)
    ci_lower, ci_upper = _bootstrap_ci(
        paired_values,
        config=bootstrap,
        spec=spec,
        scope=scope,
    )
    return PairedMetricComparison(
        schema_version="paired_metric_comparison_v1",
        metric_name=spec.metric_name,
        k=spec.k,
        sample_count=len(paired_values),
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        mean_delta=candidate_mean - baseline_mean,
        confidence=bootstrap.confidence,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def compare_paired_reports(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
    *,
    metric_specs: tuple[ComparisonMetricSpec, ...],
    bootstrap: BootstrapConfig,
    eligible_queries_only: bool,
) -> PairedComparisonReport:
    """Compare identical queries using subject-stratified paired bootstrap."""

    if baseline.dataset_id != candidate.dataset_id:
        raise EvaluationContractError("paired reports must use the same dataset_id")
    if baseline.run_id == candidate.run_id:
        raise EvaluationContractError("baseline and candidate run_id must differ")
    if not metric_specs:
        raise EvaluationContractError("metric_specs must not be empty")
    spec_keys = tuple((spec.metric_name, spec.k) for spec in metric_specs)
    if len(spec_keys) != len(set(spec_keys)):
        raise EvaluationContractError("metric_specs must not contain duplicates")

    baseline_by_id = {metric.query_id: metric for metric in baseline.per_query}
    candidate_by_id = {metric.query_id: metric for metric in candidate.per_query}
    if set(baseline_by_id) != set(candidate_by_id):
        raise EvaluationContractError("paired reports must contain identical query IDs")

    selected_ids: list[str] = []
    for query_id in sorted(baseline_by_id):
        baseline_metric = baseline_by_id[query_id]
        candidate_metric = candidate_by_id[query_id]
        if (
            baseline_metric.subject != candidate_metric.subject
            or baseline_metric.dataset_kind != candidate_metric.dataset_kind
            or baseline_metric.eligible_for_rollout
            != candidate_metric.eligible_for_rollout
        ):
            raise EvaluationContractError(
                f"paired query contract mismatch for query_id={query_id!r}"
            )
        if eligible_queries_only and not baseline_metric.eligible_for_rollout:
            continue
        selected_ids.append(query_id)
    if not selected_ids:
        raise EvaluationContractError("no queries remain for paired comparison")

    global_comparisons: list[PairedMetricComparison] = []
    subject_values: dict[str, list[PairedMetricComparison]] = defaultdict(list)
    for spec in metric_specs:
        paired: list[tuple[str, float, float]] = []
        by_subject: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
        for query_id in selected_ids:
            baseline_metric = baseline_by_id[query_id]
            candidate_metric = candidate_by_id[query_id]
            baseline_value = _lookup_metric(baseline_metric, spec)
            candidate_value = _lookup_metric(candidate_metric, spec)
            if baseline_value is None or candidate_value is None:
                continue
            item = (baseline_metric.subject, baseline_value, candidate_value)
            paired.append(item)
            by_subject[baseline_metric.subject].append(item)
        global_comparisons.append(
            _compare_metric(
                tuple(paired),
                spec=spec,
                bootstrap=bootstrap,
                scope="global",
            )
        )
        for subject, values in sorted(by_subject.items()):
            subject_values[subject].append(
                _compare_metric(
                    tuple(values),
                    spec=spec,
                    bootstrap=bootstrap,
                    scope=f"subject:{subject}",
                )
            )

    gate = candidate.rollout_data_gate
    return PairedComparisonReport(
        schema_version="paired_comparison_report_v1",
        dataset_id=baseline.dataset_id,
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        eligible_queries_only=eligible_queries_only,
        paired_query_count=len(selected_ids),
        bootstrap=bootstrap,
        global_metrics=tuple(global_comparisons),
        subjects=tuple(
            SubjectPairedComparison(
                schema_version="subject_paired_comparison_v1",
                subject=subject,
                metrics=tuple(values),
            )
            for subject, values in sorted(subject_values.items())
        ),
        rollout_data_gate_passed=gate.passed and eligible_queries_only,
        production_recommendation_blocked=not (gate.passed and eligible_queries_only),
        blocker_codes=gate.blocker_codes,
    )


__all__ = [
    "AggregateMetricAtK",
    "AggregateMetrics",
    "BootstrapConfig",
    "ComparisonMetricSpec",
    "DatasetRolloutGate",
    "EvaluationContractError",
    "EvaluationGateConfig",
    "EvaluationReport",
    "GoldDataset",
    "GoldEvidenceSpan",
    "GoldQuery",
    "MetricAtK",
    "PairedComparisonReport",
    "PairedMetricComparison",
    "QueryMetrics",
    "QueryRetrievalResult",
    "RetrievalEvaluationInput",
    "RetrievedEvidenceHit",
    "SubjectDataGate",
    "SubjectMetrics",
    "SubjectPairedComparison",
    "compare_paired_reports",
    "evaluate_retrieval_run",
]
