"""Safe stage attribution for strict parent-child retrieval regressions.

The module consumes policy-independent Gold coordinates and the body-free trace
emitted by the production retriever.  It never receives provider payloads and
never serializes query or evidence text.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.rag.parent_child.evaluation import GoldEvidenceSpan, GoldQuery
from src.rag.parent_child.retrieval import (
    HybridRetrievalPolicy,
    HybridRetrievalDiagnosticTrace,
    RetrievalDiagnosticChildCoordinate,
    RetrievalDiagnosticHydrationCoordinate,
    compute_retrieval_fingerprint,
)


RegressionStageOutcome = Literal[
    "hydrated_match",
    "child_retrieval_miss",
    "fusion_cutoff",
    "reranker_demotion",
    "parent_aggregation",
    "source_cap",
    "unique_parent_cap",
    "hydration_omission",
    "window_omission",
]


class RegressionDiagnosticError(ValueError):
    """Raised when Gold and trace identities cannot be diagnosed safely."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class GoldSpanStageAttribution(_StrictFrozenModel):
    """One Gold span projected through every body-free retrieval stage."""

    schema_version: Literal["gold_span_stage_attribution_v1"]
    gold_span_id: str = Field(min_length=1)
    source_group_id: str = Field(min_length=1)
    source_relpath: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    pagination_kind: Literal["physical", "logical"]
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    section_path: tuple[str, ...] | None
    relevance_grade: int = Field(ge=1, le=3)
    outcome: RegressionStageOutcome
    vector_match_ranks: tuple[int, ...]
    bm25_match_ranks: tuple[int, ...]
    fusion_match_ranks: tuple[int, ...]
    reranker_match_ranks: tuple[int, ...]
    pre_cap_parent_ranks: tuple[int, ...]
    source_cap_parent_ranks: tuple[int, ...]
    unique_parent_cap_parent_ranks: tuple[int, ...]
    selected_parent_ranks: tuple[int, ...]
    hydrated_parent_ranks: tuple[int, ...]
    matched_window_parent_ranks: tuple[int, ...]
    reranker_rank_worsened: bool

    @model_validator(mode="after")
    def validate_attribution(self) -> Self:
        for ranks in (
            self.vector_match_ranks,
            self.bm25_match_ranks,
            self.fusion_match_ranks,
            self.reranker_match_ranks,
            self.pre_cap_parent_ranks,
            self.source_cap_parent_ranks,
            self.unique_parent_cap_parent_ranks,
            self.selected_parent_ranks,
            self.hydrated_parent_ranks,
            self.matched_window_parent_ranks,
        ):
            if any(rank <= 0 for rank in ranks) or tuple(sorted(set(ranks))) != ranks:
                raise ValueError("diagnostic rank tuples must be sorted and unique")
        if self.end_char <= self.start_char or self.page_end < self.page_start:
            raise ValueError("Gold diagnostic coordinates must be non-empty")
        if self.outcome == "hydrated_match" and not self.matched_window_parent_ranks:
            raise ValueError("hydrated_match requires a matching hydration window")
        if self.outcome != "hydrated_match" and self.matched_window_parent_ranks:
            raise ValueError("failed attribution cannot carry a matching window")
        return self


class QueryRegressionDiagnostic(_StrictFrozenModel):
    """Safe query identity and per-span stage attribution, without query text."""

    schema_version: Literal["query_regression_diagnostic_v1"]
    query_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    trace: HybridRetrievalDiagnosticTrace
    gold_spans: tuple[GoldSpanStageAttribution, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_query_binding(self) -> Self:
        if self.trace.subject != self.subject:
            raise ValueError("Gold query and diagnostic trace subject differ")
        if self.trace.request_id != self.query_id:
            raise ValueError("Gold query and diagnostic trace request identity differ")
        span_ids = tuple(span.gold_span_id for span in self.gold_spans)
        if len(span_ids) != len(set(span_ids)):
            raise ValueError("query diagnostics require unique Gold span IDs")
        return self


class RegressionStageCount(_StrictFrozenModel):
    """Stable aggregate count for one explicit diagnostic outcome."""

    schema_version: Literal["regression_stage_count_v1"]
    outcome: RegressionStageOutcome
    count: int = Field(ge=0)


class RegressionQuerySubset(_StrictFrozenModel):
    """Safe resumable selection bound to one canonical GoldDataset digest."""

    schema_version: Literal["regression_query_subset_v1"]
    gold_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_query_ids(self) -> Self:
        if any(
            not query_id or query_id != query_id.strip() for query_id in self.query_ids
        ):
            raise ValueError("subset query IDs must be non-empty and already stripped")
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("subset query IDs must be unique")
        return self


class ParentChildRegressionReport(_StrictFrozenModel):
    """Canonical safe output for one exact-generation diagnostic run."""

    schema_version: Literal["parent_child_regression_report_v1"]
    dataset_id: str = Field(min_length=1)
    gold_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_id: str = Field(min_length=1)
    generation_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieval_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieval_policy: HybridRetrievalPolicy
    query_ids: tuple[str, ...] = Field(min_length=1)
    stage_counts: tuple[RegressionStageCount, ...] = Field(min_length=1)
    diagnostics: tuple[QueryRegressionDiagnostic, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_report_binding(self) -> Self:
        diagnostic_ids = tuple(item.query_id for item in self.diagnostics)
        if diagnostic_ids != self.query_ids:
            raise ValueError("report query IDs must preserve diagnostic order")
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("report query IDs must be unique")
        if any(
            item.trace.generation_id != self.generation_id
            or item.trace.retrieval_fingerprint != self.retrieval_fingerprint
            for item in self.diagnostics
        ):
            raise ValueError("report diagnostics differ from bound generation identity")
        if self.retrieval_policy.generation_manifest_sha256 != (
            self.generation_manifest_sha256
        ):
            raise ValueError("report policy differs from the generation manifest")
        if compute_retrieval_fingerprint(self.retrieval_policy) != (
            self.retrieval_fingerprint
        ):
            raise ValueError("report retrieval policy fingerprint is inconsistent")
        expected = Counter(
            span.outcome
            for diagnostic in self.diagnostics
            for span in diagnostic.gold_spans
        )
        actual = {item.outcome: item.count for item in self.stage_counts}
        if len(actual) != len(self.stage_counts) or actual != expected:
            raise ValueError("stage counts do not match per-span diagnostics")
        return self


def _overlaps(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return max(start_a, start_b) < min(end_a, end_b)


def _child_matches(
    child: RetrievalDiagnosticChildCoordinate,
    gold: GoldEvidenceSpan,
) -> bool:
    return (
        child.doc_id == gold.doc_id
        and child.source_relpath == gold.source_relpath
        and child.pagination_kind == gold.pagination_kind
        and _overlaps(
            child.page_start, child.page_end + 1, gold.page_start, gold.page_end + 1
        )
        and _overlaps(child.start_char, child.end_char, gold.start_char, gold.end_char)
    )


def _window_matches(
    parent: RetrievalDiagnosticHydrationCoordinate,
    gold: GoldEvidenceSpan,
) -> bool:
    if (
        parent.doc_id != gold.doc_id
        or parent.source_relpath != gold.source_relpath
        or parent.pagination_kind != gold.pagination_kind
        or not _overlaps(
            parent.page_start,
            parent.page_end + 1,
            gold.page_start,
            gold.page_end + 1,
        )
    ):
        return False
    return any(
        _overlaps(window.start_char, window.end_char, gold.start_char, gold.end_char)
        for window in parent.windows
    )


def _sorted_ranks(values: list[int]) -> tuple[int, ...]:
    return tuple(sorted(set(values)))


def _attribution(
    gold: GoldEvidenceSpan,
    trace: HybridRetrievalDiagnosticTrace,
) -> GoldSpanStageAttribution:
    matching_children = tuple(
        child for child in trace.children if _child_matches(child, gold)
    )
    matching_child_ids = {child.child_id for child in matching_children}
    matching_parent_ids = {child.parent_id for child in matching_children}
    parent_coordinates = tuple(
        parent for parent in trace.parents if parent.parent_id in matching_parent_ids
    )
    supporting_parents = tuple(
        parent
        for parent in parent_coordinates
        if matching_child_ids.intersection(parent.supporting_child_ids)
    )
    hydrated_target_parents = tuple(
        parent
        for parent in trace.hydrated_parents
        if parent.parent_id in matching_parent_ids
    )
    matching_windows = tuple(
        parent for parent in trace.hydrated_parents if _window_matches(parent, gold)
    )

    fusion_ranks = _sorted_ranks([child.fusion_rank for child in matching_children])
    reranker_ranks = _sorted_ranks(
        [
            child.reranker_rank
            for child in matching_children
            if child.reranker_rank is not None
        ]
    )
    rank_worsened = bool(
        fusion_ranks and reranker_ranks and min(reranker_ranks) > min(fusion_ranks)
    )

    if matching_windows:
        outcome: RegressionStageOutcome = "hydrated_match"
    elif not matching_children:
        outcome = "child_retrieval_miss"
    elif not reranker_ranks:
        outcome = "fusion_cutoff"
    elif not supporting_parents:
        outcome = "reranker_demotion" if rank_worsened else "parent_aggregation"
    else:
        best_supporting_parent = min(
            supporting_parents,
            key=lambda parent: (parent.pre_cap_rank, parent.parent_id),
        )
        if best_supporting_parent.selection_outcome == "source_cap":
            outcome = "source_cap"
        elif best_supporting_parent.selection_outcome == "unique_parent_cap":
            outcome = "unique_parent_cap"
        elif not hydrated_target_parents:
            outcome = "hydration_omission"
        else:
            outcome = "window_omission"

    return GoldSpanStageAttribution(
        schema_version="gold_span_stage_attribution_v1",
        gold_span_id=gold.gold_span_id,
        source_group_id=gold.source_group_id,
        source_relpath=gold.source_relpath,
        doc_id=gold.doc_id,
        pagination_kind=gold.pagination_kind,
        page_start=gold.page_start,
        page_end=gold.page_end,
        start_char=gold.start_char,
        end_char=gold.end_char,
        section_path=gold.section_path,
        relevance_grade=gold.relevance_grade,
        outcome=outcome,
        vector_match_ranks=_sorted_ranks(
            [
                child.vector_rank
                for child in matching_children
                if child.vector_rank is not None
            ]
        ),
        bm25_match_ranks=_sorted_ranks(
            [
                child.bm25_rank
                for child in matching_children
                if child.bm25_rank is not None
            ]
        ),
        fusion_match_ranks=fusion_ranks,
        reranker_match_ranks=reranker_ranks,
        pre_cap_parent_ranks=_sorted_ranks(
            [parent.pre_cap_rank for parent in parent_coordinates]
        ),
        source_cap_parent_ranks=_sorted_ranks(
            [
                parent.pre_cap_rank
                for parent in parent_coordinates
                if parent.selection_outcome == "source_cap"
            ]
        ),
        unique_parent_cap_parent_ranks=_sorted_ranks(
            [
                parent.pre_cap_rank
                for parent in parent_coordinates
                if parent.selection_outcome == "unique_parent_cap"
            ]
        ),
        selected_parent_ranks=_sorted_ranks(
            [
                parent.selected_rank
                for parent in parent_coordinates
                if parent.selected_rank is not None
            ]
        ),
        hydrated_parent_ranks=_sorted_ranks(
            [parent.selected_rank for parent in hydrated_target_parents]
        ),
        matched_window_parent_ranks=_sorted_ranks(
            [parent.selected_rank for parent in matching_windows]
        ),
        reranker_rank_worsened=rank_worsened,
    )


def diagnose_gold_query(
    *,
    query: GoldQuery,
    trace: HybridRetrievalDiagnosticTrace,
) -> QueryRegressionDiagnostic:
    """Attribute every Gold span without copying the query into diagnostics."""

    if query.subject != trace.subject:
        raise RegressionDiagnosticError("Gold query and retrieval subject differ")
    return QueryRegressionDiagnostic(
        schema_version="query_regression_diagnostic_v1",
        query_id=query.query_id,
        subject=query.subject,
        trace=trace,
        gold_spans=tuple(_attribution(span, trace) for span in query.gold_spans),
    )


def build_regression_report(
    *,
    dataset_id: str,
    gold_dataset_sha256: str,
    generation_id: str,
    generation_manifest_sha256: str,
    retrieval_policy: HybridRetrievalPolicy,
    diagnostics: tuple[QueryRegressionDiagnostic, ...],
) -> ParentChildRegressionReport:
    """Build and cross-validate one immutable body-free diagnostic report."""

    if not diagnostics:
        raise RegressionDiagnosticError("at least one query diagnostic is required")
    counts = Counter(
        span.outcome for diagnostic in diagnostics for span in diagnostic.gold_spans
    )
    stage_counts = tuple(
        RegressionStageCount(
            schema_version="regression_stage_count_v1",
            outcome=outcome,
            count=counts[outcome],
        )
        for outcome in sorted(counts)
    )
    return ParentChildRegressionReport(
        schema_version="parent_child_regression_report_v1",
        dataset_id=dataset_id,
        gold_dataset_sha256=gold_dataset_sha256,
        generation_id=generation_id,
        generation_manifest_sha256=generation_manifest_sha256,
        retrieval_fingerprint=compute_retrieval_fingerprint(retrieval_policy),
        retrieval_policy=retrieval_policy,
        query_ids=tuple(item.query_id for item in diagnostics),
        stage_counts=stage_counts,
        diagnostics=diagnostics,
    )


__all__ = [
    "GoldSpanStageAttribution",
    "ParentChildRegressionReport",
    "QueryRegressionDiagnostic",
    "RegressionDiagnosticError",
    "RegressionQuerySubset",
    "RegressionStageCount",
    "RegressionStageOutcome",
    "build_regression_report",
    "diagnose_gold_query",
]
