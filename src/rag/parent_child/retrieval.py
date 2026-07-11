"""Strict hybrid child retrieval with authoritative parent hydration.

The candidate path is intentionally isolated from the legacy retriever.  Both
configured child-search channels are required, every provider payload is
validated, and no exception activates a substitute channel or result.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import math
from time import perf_counter_ns
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic import model_validator

from src.rag.parent_child._storage_io import model_json_bytes, sha256_bytes
from src.rag.parent_child.models import ChildDocument, ParentRecord


class HybridRetrievalError(RuntimeError):
    """Base class for explicit candidate-retrieval failures."""


class RetrievalChannelError(HybridRetrievalError):
    """Raised when a required vector, BM25, or reranker call fails."""


class RetrievalProtocolError(HybridRetrievalError):
    """Raised when a required provider violates its declared response contract."""


class RetrievalInvariantError(HybridRetrievalError):
    """Raised when cross-artifact child or parent invariants do not hold."""


class ParentHydrationError(HybridRetrievalError):
    """Raised when authoritative parent hydration fails or returns invalid data."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class HybridRetrievalPolicy(_StrictFrozenModel):
    """Complete runtime-affecting policy for one strict hybrid retrieval."""

    schema_version: Literal["hybrid_retrieval_policy_v1"]
    generation_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    bm25_tokenizer_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    reranker_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    vector_top_k: int = Field(gt=0)
    bm25_top_k: int = Field(gt=0)
    vector_rrf_weight: float = Field(gt=0)
    bm25_rrf_weight: float = Field(gt=0)
    rrf_k: int = Field(gt=0)
    reranker_top_n: int = Field(gt=0)
    unique_parent_top_k: int = Field(gt=0)
    max_children_per_parent: int = Field(gt=0)
    max_parents_per_source: int = Field(gt=0)
    parent_support_lambda: float = Field(ge=0, le=1)
    full_parent_max_chars: int = Field(gt=0)
    hit_window_chars_per_side: int = Field(ge=0)
    multi_subject_per_subject_top_k: int = Field(gt=0)
    multi_subject_max_parents: int = Field(gt=0)
    subject_coverage_quota: int = Field(gt=0)

    @field_validator(
        "vector_rrf_weight",
        "bm25_rrf_weight",
        "parent_support_lambda",
    )
    @classmethod
    def validate_finite_float(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("retrieval policy floating-point values must be finite")
        return value

    @model_validator(mode="after")
    def validate_cross_field_constraints(self) -> Self:
        candidate_limit = self.vector_top_k + self.bm25_top_k
        if self.reranker_top_n > candidate_limit:
            raise ValueError("reranker_top_n must not exceed vector_top_k + bm25_top_k")
        if self.unique_parent_top_k > self.reranker_top_n:
            raise ValueError("unique_parent_top_k must not exceed reranker_top_n")
        if self.multi_subject_per_subject_top_k > self.multi_subject_max_parents:
            raise ValueError(
                "multi_subject_per_subject_top_k must not exceed max parents"
            )
        return self


class HybridRetrievalRequest(_StrictFrozenModel):
    """One exact-subject query bound to one immutable generation."""

    schema_version: Literal["hybrid_retrieval_request_v1"]
    request_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)

    @field_validator("request_id", "query", "subject", "generation_id")
    @classmethod
    def validate_stripped_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("retrieval request text fields must already be stripped")
        return value


class ChildSearchCandidate(_StrictFrozenModel):
    """Validated child payload returned by a required search channel."""

    schema_version: Literal["child_search_candidate_v1"]
    document: ChildDocument
    raw_score: float

    @field_validator("raw_score")
    @classmethod
    def validate_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("search-channel scores must be finite")
        return value


class RerankCandidate(_StrictFrozenModel):
    """Exact child identity and text passed to the strict reranker."""

    schema_version: Literal["rerank_candidate_v1"]
    child_id: str = Field(min_length=1)
    content: str = Field(min_length=1)


class RerankScore(_StrictFrozenModel):
    """Required reranker score for exactly one submitted child."""

    schema_version: Literal["rerank_score_v1"]
    child_id: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)

    @field_validator("score")
    @classmethod
    def validate_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("reranker scores must be finite")
        return value


class ChildEvidenceHit(_StrictFrozenModel):
    """One fully provenance-preserving, reranked child hit."""

    schema_version: Literal["child_evidence_hit_v1"]
    final_rank: int = Field(gt=0)
    document: ChildDocument
    vector_rank: int | None
    bm25_rank: int | None
    vector_raw_score: float | None
    bm25_raw_score: float | None
    rrf_score: float = Field(gt=0)
    rerank_score: float = Field(ge=0, le=1)

    @field_validator(
        "vector_raw_score",
        "bm25_raw_score",
        "rrf_score",
        "rerank_score",
    )
    @classmethod
    def validate_finite_scores(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("child hit scores must be finite")
        return value

    @model_validator(mode="after")
    def validate_channel_provenance(self) -> Self:
        if (self.vector_rank is None) != (self.vector_raw_score is None):
            raise ValueError(
                "vector rank and score must either both exist or both be null"
            )
        if (self.bm25_rank is None) != (self.bm25_raw_score is None):
            raise ValueError(
                "BM25 rank and score must either both exist or both be null"
            )
        if self.vector_rank is None and self.bm25_rank is None:
            raise ValueError("a child hit must originate from at least one channel")
        if self.vector_rank is not None and self.vector_rank <= 0:
            raise ValueError("vector rank must be positive")
        if self.bm25_rank is not None and self.bm25_rank <= 0:
            raise ValueError("BM25 rank must be positive")
        return self


class ParentAggregate(_StrictFrozenModel):
    """Selected parent score and its bounded supporting child set."""

    schema_version: Literal["parent_aggregate_v1"]
    rank: int = Field(gt=0)
    parent_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    source_relpath: str = Field(min_length=1)
    parent_score: float = Field(ge=0, le=1)
    best_child_rank: int = Field(gt=0)
    supporting_child_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("parent_score")
    @classmethod
    def validate_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("parent score must be finite")
        return value

    @model_validator(mode="after")
    def validate_unique_children(self) -> Self:
        if len(self.supporting_child_ids) != len(set(self.supporting_child_ids)):
            raise ValueError("supporting child IDs must be unique")
        return self


class ParentContextWindow(_StrictFrozenModel):
    """One exact, parent-relative half-open context window."""

    schema_version: Literal["parent_context_window_v1"]
    start_in_parent: int = Field(ge=0)
    end_in_parent: int = Field(gt=0)
    content: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_window_length(self) -> Self:
        if self.end_in_parent <= self.start_in_parent:
            raise ValueError("parent context windows must be non-empty")
        if len(self.content) != self.end_in_parent - self.start_in_parent:
            raise ValueError("parent context window content length is inconsistent")
        return self


class HydratedParentContext(_StrictFrozenModel):
    """Authoritative parent and exact expansion supporting selected children."""

    schema_version: Literal["hydrated_parent_context_v1"]
    rank: int = Field(gt=0)
    parent: ParentRecord
    supporting_child_ids: tuple[str, ...] = Field(min_length=1)
    expansion_mode: Literal["full_parent", "hit_window"]
    heading: str
    windows: tuple[ParentContextWindow, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_hydrated_context(self) -> Self:
        if len(self.supporting_child_ids) != len(set(self.supporting_child_ids)):
            raise ValueError("hydrated supporting child IDs must be unique")
        cursor = -1
        for window in self.windows:
            if window.start_in_parent < cursor:
                raise ValueError("parent context windows must be ordered and disjoint")
            if window.end_in_parent > self.parent.parent_chars:
                raise ValueError("parent context window exceeds parent content")
            if (
                self.parent.content[window.start_in_parent : window.end_in_parent]
                != window.content
            ):
                raise ValueError("parent context window is not an exact parent slice")
            cursor = window.end_in_parent
        if self.expansion_mode == "full_parent":
            if len(self.windows) != 1:
                raise ValueError(
                    "full-parent expansion must contain exactly one window"
                )
            only = self.windows[0]
            if (
                only.start_in_parent != 0
                or only.end_in_parent != self.parent.parent_chars
            ):
                raise ValueError("full-parent expansion must cover the complete parent")
        return self


class RetrievalTimings(_StrictFrozenModel):
    """Monotonic elapsed milliseconds for each required retrieval stage."""

    schema_version: Literal["retrieval_timings_v1"]
    vector_ms: float = Field(ge=0)
    bm25_ms: float = Field(ge=0)
    reranker_ms: float = Field(ge=0)
    hydrate_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)

    @field_validator("vector_ms", "bm25_ms", "reranker_ms", "hydrate_ms", "total_ms")
    @classmethod
    def validate_finite_timing(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("retrieval timings must be finite")
        return value


class HybridRetrievalResult(_StrictFrozenModel):
    """Strict retrieval outcome; zero hits are data, while failures raise."""

    schema_version: Literal["hybrid_retrieval_result_v1"]
    status: Literal["ok", "empty"]
    request: HybridRetrievalRequest
    retrieval_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    ranked_children: tuple[ChildEvidenceHit, ...]
    ranked_parents: tuple[ParentAggregate, ...]
    hydrated_parents: tuple[HydratedParentContext, ...]
    timings: RetrievalTimings

    @model_validator(mode="after")
    def validate_result_contract(self) -> Self:
        if self.status == "empty":
            if self.ranked_children or self.ranked_parents or self.hydrated_parents:
                raise ValueError("empty retrieval result cannot contain hits")
            return self
        if not self.ranked_children or not self.ranked_parents:
            raise ValueError("successful retrieval requires child and parent hits")
        if tuple(hit.final_rank for hit in self.ranked_children) != tuple(
            range(1, len(self.ranked_children) + 1)
        ):
            raise ValueError("child final ranks must be contiguous and 1-based")
        if tuple(parent.rank for parent in self.ranked_parents) != tuple(
            range(1, len(self.ranked_parents) + 1)
        ):
            raise ValueError("parent ranks must be contiguous and 1-based")
        if tuple(item.rank for item in self.hydrated_parents) != tuple(
            range(1, len(self.hydrated_parents) + 1)
        ):
            raise ValueError("hydrated parent ranks must be contiguous and 1-based")
        if tuple(parent.parent_id for parent in self.ranked_parents) != tuple(
            item.parent.parent_id for item in self.hydrated_parents
        ):
            raise ValueError("ranked and hydrated parent identities must match")
        return self


class HybridChildRetrievalResult(_StrictFrozenModel):
    """Judge-safe child/parent-reference result containing no parent body text."""

    schema_version: Literal["hybrid_child_retrieval_result_v1"]
    status: Literal["ok", "empty"]
    request: HybridRetrievalRequest
    retrieval_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    ranked_children: tuple[ChildEvidenceHit, ...]
    ranked_parents: tuple[ParentAggregate, ...]
    timings: RetrievalTimings

    @model_validator(mode="after")
    def validate_child_result(self) -> Self:
        if self.status == "empty":
            if self.ranked_children or self.ranked_parents:
                raise ValueError("empty child retrieval result cannot contain hits")
            return self
        if not self.ranked_children or not self.ranked_parents:
            raise ValueError("successful child retrieval requires children and parents")
        if tuple(hit.final_rank for hit in self.ranked_children) != tuple(
            range(1, len(self.ranked_children) + 1)
        ):
            raise ValueError("child result ranks must be contiguous and 1-based")
        if tuple(parent.rank for parent in self.ranked_parents) != tuple(
            range(1, len(self.ranked_parents) + 1)
        ):
            raise ValueError("parent reference ranks must be contiguous and 1-based")
        return self


class WeightedHybridBranch(_StrictFrozenModel):
    """One weighted query/subject branch in cross-branch fusion."""

    schema_version: Literal["weighted_hybrid_branch_v1"]
    branch_id: str = Field(min_length=1)
    weight: float = Field(gt=0)
    request: HybridRetrievalRequest

    @field_validator("branch_id")
    @classmethod
    def validate_branch_id(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("branch_id must already be stripped")
        return value

    @field_validator("weight")
    @classmethod
    def validate_finite_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("branch weight must be finite")
        return value


class MultiBranchHybridRequest(_StrictFrozenModel):
    """Weighted branches bound to one generation for parent-level fusion."""

    schema_version: Literal["multi_branch_hybrid_request_v1"]
    request_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    branches: tuple[WeightedHybridBranch, ...] = Field(min_length=1)
    cross_branch_rrf_k: int = Field(gt=0)
    parent_top_k: int = Field(gt=0)

    @field_validator("request_id", "generation_id")
    @classmethod
    def validate_stripped_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("multi-branch request text must already be stripped")
        return value

    @model_validator(mode="after")
    def validate_branches(self) -> Self:
        branch_ids = tuple(branch.branch_id for branch in self.branches)
        if len(branch_ids) != len(set(branch_ids)):
            raise ValueError("multi-branch request branch IDs must be unique")
        if any(
            branch.request.generation_id != self.generation_id
            for branch in self.branches
        ):
            raise ValueError("every branch must use the multi-branch generation")
        return self


class BranchParentProvenance(_StrictFrozenModel):
    """Parent rank contribution from one explicit query branch."""

    schema_version: Literal["branch_parent_provenance_v1"]
    branch_id: str = Field(min_length=1)
    branch_parent_rank: int = Field(gt=0)
    branch_weight: float = Field(gt=0)

    @field_validator("branch_weight")
    @classmethod
    def validate_finite_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("branch provenance weight must be finite")
        return value


class CrossBranchParentHit(_StrictFrozenModel):
    """Deterministically fused parent identity across query/subject branches."""

    schema_version: Literal["cross_branch_parent_hit_v1"]
    rank: int = Field(gt=0)
    parent_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    source_relpath: str = Field(min_length=1)
    cross_branch_rrf_score: float = Field(gt=0)
    best_branch_parent_rank: int = Field(gt=0)
    provenance: tuple[BranchParentProvenance, ...] = Field(min_length=1)

    @field_validator("cross_branch_rrf_score")
    @classmethod
    def validate_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("cross-branch RRF score must be finite")
        return value

    @model_validator(mode="after")
    def validate_unique_provenance(self) -> Self:
        branch_ids = tuple(item.branch_id for item in self.provenance)
        if len(branch_ids) != len(set(branch_ids)):
            raise ValueError("cross-branch provenance must contain unique branches")
        return self


class MultiBranchHybridResult(_StrictFrozenModel):
    """Branch-local strict results plus weighted parent-level cross-RRF."""

    schema_version: Literal["multi_branch_hybrid_result_v1"]
    status: Literal["ok", "empty"]
    request: MultiBranchHybridRequest
    branch_results: tuple[HybridRetrievalResult, ...]
    ranked_parents: tuple[CrossBranchParentHit, ...]

    @model_validator(mode="after")
    def validate_multi_branch_result(self) -> Self:
        expected_requests = tuple(branch.request for branch in self.request.branches)
        actual_requests = tuple(result.request for result in self.branch_results)
        if actual_requests != expected_requests:
            raise ValueError("branch results must preserve exact request order")
        if self.status == "empty" and self.ranked_parents:
            raise ValueError("empty multi-branch result cannot contain parents")
        if self.status == "ok" and not self.ranked_parents:
            raise ValueError("successful multi-branch result requires parents")
        if tuple(parent.rank for parent in self.ranked_parents) != tuple(
            range(1, len(self.ranked_parents) + 1)
        ):
            raise ValueError("cross-branch parent ranks must be contiguous and 1-based")
        return self


class MultiBranchHybridChildResult(_StrictFrozenModel):
    """Judge-safe branch results and cross-branch parent references."""

    schema_version: Literal["multi_branch_hybrid_child_result_v1"]
    status: Literal["ok", "empty"]
    request: MultiBranchHybridRequest
    branch_results: tuple[HybridChildRetrievalResult, ...]
    ranked_parents: tuple[CrossBranchParentHit, ...]

    @model_validator(mode="after")
    def validate_multi_branch_child_result(self) -> Self:
        expected_requests = tuple(branch.request for branch in self.request.branches)
        actual_requests = tuple(result.request for result in self.branch_results)
        if actual_requests != expected_requests:
            raise ValueError("child branch results must preserve exact request order")
        if self.status == "empty" and self.ranked_parents:
            raise ValueError("empty multi-branch child result cannot contain parents")
        if self.status == "ok" and not self.ranked_parents:
            raise ValueError("successful multi-branch child result requires parents")
        if tuple(parent.rank for parent in self.ranked_parents) != tuple(
            range(1, len(self.ranked_parents) + 1)
        ):
            raise ValueError("child cross-branch parent ranks must be contiguous")
        return self


class ChildSearchChannel(Protocol):
    """Injected exact-subject child search boundary."""

    def search(
        self,
        *,
        query: str,
        subject: str,
        generation_id: str,
        top_k: int,
    ) -> Sequence[ChildSearchCandidate]: ...


class ChildReranker(Protocol):
    """Injected strict reranker boundary."""

    def rerank(
        self,
        *,
        query: str,
        candidates: tuple[RerankCandidate, ...],
    ) -> Sequence[RerankScore]: ...


class ParentHydrator(Protocol):
    """Injected authoritative batch-parent store boundary."""

    def get_many(self, parent_ids: Sequence[str]) -> tuple[ParentRecord, ...]: ...


@dataclass(slots=True)
class _MergedCandidate:
    document: ChildDocument
    vector_rank: int | None
    bm25_rank: int | None
    vector_raw_score: float | None
    bm25_raw_score: float | None
    rrf_score: float
    fused_rank: int


@dataclass(frozen=True, slots=True)
class _UnrankedParent:
    parent_id: str
    subject: str
    source_relpath: str
    parent_score: float
    best_child_rank: int
    supporting_child_ids: tuple[str, ...]


@dataclass(slots=True)
class _CrossBranchParent:
    parent_id: str
    subject: str
    source_relpath: str
    score: float
    best_rank: int
    provenance: list[BranchParentProvenance]


def _fuse_cross_branch_parents(
    *,
    policy: HybridRetrievalPolicy,
    request: MultiBranchHybridRequest,
    branch_results: Sequence[HybridRetrievalResult | HybridChildRetrievalResult],
) -> tuple[CrossBranchParentHit, ...]:
    if request.parent_top_k > policy.multi_subject_max_parents:
        raise RetrievalInvariantError(
            "multi-branch parent_top_k exceeds configured maximum"
        )
    merged: dict[str, _CrossBranchParent] = {}
    for branch, result in zip(request.branches, branch_results, strict=True):
        for parent in result.ranked_parents[: policy.multi_subject_per_subject_top_k]:
            contribution = branch.weight / (request.cross_branch_rrf_k + parent.rank)
            provenance = BranchParentProvenance(
                schema_version="branch_parent_provenance_v1",
                branch_id=branch.branch_id,
                branch_parent_rank=parent.rank,
                branch_weight=branch.weight,
            )
            existing = merged.get(parent.parent_id)
            if existing is None:
                merged[parent.parent_id] = _CrossBranchParent(
                    parent_id=parent.parent_id,
                    subject=parent.subject,
                    source_relpath=parent.source_relpath,
                    score=contribution,
                    best_rank=parent.rank,
                    provenance=[provenance],
                )
                continue
            if (
                existing.subject != parent.subject
                or existing.source_relpath != parent.source_relpath
            ):
                raise RetrievalInvariantError(
                    "branches disagree on subject or source for a parent ID"
                )
            existing.score += contribution
            existing.best_rank = min(existing.best_rank, parent.rank)
            existing.provenance.append(provenance)

    ordered = sorted(
        merged.values(),
        key=lambda item: (-item.score, item.best_rank, item.parent_id),
    )
    selected: list[_CrossBranchParent] = []
    selected_ids: set[str] = set()
    source_counts: dict[str, int] = defaultdict(int)
    requested_subjects = tuple(
        dict.fromkeys(branch.request.subject for branch in request.branches)
    )
    for subject in requested_subjects:
        subject_selected = 0
        for cross_parent in ordered:
            if cross_parent.subject != subject:
                continue
            if cross_parent.parent_id in selected_ids:
                continue
            if (
                source_counts[cross_parent.source_relpath]
                >= policy.max_parents_per_source
            ):
                continue
            selected.append(cross_parent)
            selected_ids.add(cross_parent.parent_id)
            source_counts[cross_parent.source_relpath] += 1
            subject_selected += 1
            if (
                subject_selected == policy.subject_coverage_quota
                or len(selected) == request.parent_top_k
            ):
                break
        if len(selected) == request.parent_top_k:
            break
    for cross_parent in ordered:
        if cross_parent.parent_id in selected_ids:
            continue
        if source_counts[cross_parent.source_relpath] >= policy.max_parents_per_source:
            continue
        selected.append(cross_parent)
        selected_ids.add(cross_parent.parent_id)
        source_counts[cross_parent.source_relpath] += 1
        if len(selected) == request.parent_top_k:
            break

    return tuple(
        CrossBranchParentHit(
            schema_version="cross_branch_parent_hit_v1",
            rank=rank,
            parent_id=parent.parent_id,
            subject=parent.subject,
            source_relpath=parent.source_relpath,
            cross_branch_rrf_score=parent.score,
            best_branch_parent_rank=parent.best_rank,
            provenance=tuple(parent.provenance),
        )
        for rank, parent in enumerate(selected, start=1)
    )


def compute_retrieval_fingerprint(policy: HybridRetrievalPolicy) -> str:
    """Fingerprint every ranking and expansion input in the strict policy."""

    return sha256_bytes(model_json_bytes(policy))


def weighted_rrf_score(
    *,
    vector_rank: int | None,
    bm25_rank: int | None,
    vector_weight: float,
    bm25_weight: float,
    rrf_k: int,
) -> float:
    """Compute the configured 1-based weighted reciprocal-rank score."""

    if vector_rank is None and bm25_rank is None:
        raise ValueError("weighted RRF requires at least one channel rank")
    if vector_rank is not None and vector_rank <= 0:
        raise ValueError("vector_rank must be positive")
    if bm25_rank is not None and bm25_rank <= 0:
        raise ValueError("bm25_rank must be positive")
    if not math.isfinite(vector_weight) or vector_weight <= 0:
        raise ValueError("vector_weight must be positive and finite")
    if not math.isfinite(bm25_weight) or bm25_weight <= 0:
        raise ValueError("bm25_weight must be positive and finite")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    score = 0.0
    if vector_rank is not None:
        score += vector_weight / (rrf_k + vector_rank)
    if bm25_rank is not None:
        score += bm25_weight / (rrf_k + bm25_rank)
    return score


def aggregate_parent_score(
    child_scores: Sequence[float],
    *,
    support_lambda: float,
) -> float:
    """Aggregate ordered reranker scores with bounded secondary support."""

    scores = tuple(child_scores)
    if not scores:
        raise ValueError("parent aggregation requires at least one child score")
    if not math.isfinite(support_lambda) or not 0 <= support_lambda <= 1:
        raise ValueError("support_lambda must be finite and within [0, 1]")
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("child scores must be finite and within [0, 1]")
    strongest = scores[0]
    residual_product = math.prod(1.0 - score for score in scores[1:])
    return strongest + support_lambda * (1.0 - strongest) * (1.0 - residual_product)


def _elapsed_ms(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000.0


def _validate_search_results(
    raw_results: object,
    *,
    channel_name: Literal["vector", "bm25"],
    request: HybridRetrievalRequest,
    top_k: int,
) -> tuple[ChildSearchCandidate, ...]:
    if isinstance(raw_results, (str, bytes, bytearray)) or not isinstance(
        raw_results, Sequence
    ):
        raise RetrievalProtocolError(
            f"{channel_name} channel must return a sequence of child candidates"
        )
    if len(raw_results) > top_k:
        raise RetrievalProtocolError(
            f"{channel_name} channel returned more than its configured top_k"
        )

    validated: list[ChildSearchCandidate] = []
    child_ids: set[str] = set()
    for index, raw_result in enumerate(raw_results):
        try:
            candidate = ChildSearchCandidate.model_validate(raw_result)
        except ValidationError as exc:
            raise RetrievalProtocolError(
                f"{channel_name} candidate at index {index} is invalid"
            ) from exc
        metadata = candidate.document.metadata
        if metadata.subject != request.subject:
            raise RetrievalInvariantError(
                f"{channel_name} candidate subject does not match the request"
            )
        if metadata.generation_id != request.generation_id:
            raise RetrievalInvariantError(
                f"{channel_name} candidate generation does not match the request"
            )
        if metadata.child_id in child_ids:
            raise RetrievalProtocolError(
                f"{channel_name} channel returned a duplicate child ID"
            )
        child_ids.add(metadata.child_id)
        validated.append(candidate)
    return tuple(validated)


def _validate_reranker_results(
    raw_results: object,
    *,
    submitted_child_ids: tuple[str, ...],
) -> dict[str, float]:
    if isinstance(raw_results, (str, bytes, bytearray)) or not isinstance(
        raw_results, Sequence
    ):
        raise RetrievalProtocolError("reranker must return a sequence of child scores")
    scores: dict[str, float] = {}
    for index, raw_result in enumerate(raw_results):
        try:
            result = RerankScore.model_validate(raw_result)
        except ValidationError as exc:
            raise RetrievalProtocolError(
                f"reranker result at index {index} is invalid"
            ) from exc
        if result.child_id in scores:
            raise RetrievalProtocolError("reranker returned a duplicate child ID")
        scores[result.child_id] = result.score

    submitted = set(submitted_child_ids)
    returned = set(scores)
    if returned != submitted:
        missing_count = len(submitted - returned)
        unknown_count = len(returned - submitted)
        raise RetrievalProtocolError(
            "reranker result identity set mismatch: "
            f"missing={missing_count}, unknown={unknown_count}"
        )
    return scores


def _merge_child_channels(
    vector_results: tuple[ChildSearchCandidate, ...],
    bm25_results: tuple[ChildSearchCandidate, ...],
    policy: HybridRetrievalPolicy,
) -> tuple[_MergedCandidate, ...]:
    merged: dict[str, _MergedCandidate] = {}
    for rank, candidate in enumerate(vector_results, start=1):
        child_id = candidate.document.metadata.child_id
        merged[child_id] = _MergedCandidate(
            document=candidate.document,
            vector_rank=rank,
            bm25_rank=None,
            vector_raw_score=candidate.raw_score,
            bm25_raw_score=None,
            rrf_score=0.0,
            fused_rank=0,
        )
    for rank, candidate in enumerate(bm25_results, start=1):
        child_id = candidate.document.metadata.child_id
        existing = merged.get(child_id)
        if existing is None:
            merged[child_id] = _MergedCandidate(
                document=candidate.document,
                vector_rank=None,
                bm25_rank=rank,
                vector_raw_score=None,
                bm25_raw_score=candidate.raw_score,
                rrf_score=0.0,
                fused_rank=0,
            )
            continue
        if existing.document != candidate.document:
            raise RetrievalInvariantError(
                "vector and BM25 payloads disagree for the same child ID"
            )
        existing.bm25_rank = rank
        existing.bm25_raw_score = candidate.raw_score

    for merged_candidate in merged.values():
        merged_candidate.rrf_score = weighted_rrf_score(
            vector_rank=merged_candidate.vector_rank,
            bm25_rank=merged_candidate.bm25_rank,
            vector_weight=policy.vector_rrf_weight,
            bm25_weight=policy.bm25_rrf_weight,
            rrf_k=policy.rrf_k,
        )
    ranked = sorted(
        merged.values(),
        key=lambda item: (
            -item.rrf_score,
            min(
                rank for rank in (item.vector_rank, item.bm25_rank) if rank is not None
            ),
            item.document.metadata.child_id,
        ),
    )
    for rank, ranked_candidate in enumerate(ranked, start=1):
        ranked_candidate.fused_rank = rank
    return tuple(ranked)


def _build_child_hits(
    fused: tuple[_MergedCandidate, ...],
    reranker_scores: dict[str, float],
) -> tuple[ChildEvidenceHit, ...]:
    ranked = sorted(
        fused,
        key=lambda item: (
            -reranker_scores[item.document.metadata.child_id],
            item.fused_rank,
            item.document.metadata.child_id,
        ),
    )
    return tuple(
        ChildEvidenceHit(
            schema_version="child_evidence_hit_v1",
            final_rank=final_rank,
            document=item.document,
            vector_rank=item.vector_rank,
            bm25_rank=item.bm25_rank,
            vector_raw_score=item.vector_raw_score,
            bm25_raw_score=item.bm25_raw_score,
            rrf_score=item.rrf_score,
            rerank_score=reranker_scores[item.document.metadata.child_id],
        )
        for final_rank, item in enumerate(ranked, start=1)
    )


def _select_parent_aggregates(
    hits: tuple[ChildEvidenceHit, ...],
    policy: HybridRetrievalPolicy,
) -> tuple[ParentAggregate, ...]:
    grouped: dict[str, list[ChildEvidenceHit]] = defaultdict(list)
    for hit in hits:
        grouped[hit.document.metadata.parent_id].append(hit)

    unranked: list[_UnrankedParent] = []
    for parent_id, parent_hits in grouped.items():
        selected_hits = tuple(parent_hits[: policy.max_children_per_parent])
        first_metadata = selected_hits[0].document.metadata
        for hit in selected_hits[1:]:
            metadata = hit.document.metadata
            if (
                metadata.subject != first_metadata.subject
                or metadata.source_relpath != first_metadata.source_relpath
            ):
                raise RetrievalInvariantError(
                    "children sharing a parent ID disagree on subject or source"
                )
        unranked.append(
            _UnrankedParent(
                parent_id=parent_id,
                subject=first_metadata.subject,
                source_relpath=first_metadata.source_relpath,
                parent_score=aggregate_parent_score(
                    tuple(hit.rerank_score for hit in selected_hits),
                    support_lambda=policy.parent_support_lambda,
                ),
                best_child_rank=selected_hits[0].final_rank,
                supporting_child_ids=tuple(
                    hit.document.metadata.child_id for hit in selected_hits
                ),
            )
        )

    ordered = sorted(
        unranked,
        key=lambda item: (
            -item.parent_score,
            item.best_child_rank,
            item.parent_id,
        ),
    )
    selected: list[_UnrankedParent] = []
    source_counts: dict[str, int] = defaultdict(int)
    for parent in ordered:
        if source_counts[parent.source_relpath] >= policy.max_parents_per_source:
            continue
        selected.append(parent)
        source_counts[parent.source_relpath] += 1
        if len(selected) == policy.unique_parent_top_k:
            break

    return tuple(
        ParentAggregate(
            schema_version="parent_aggregate_v1",
            rank=rank,
            parent_id=item.parent_id,
            subject=item.subject,
            source_relpath=item.source_relpath,
            parent_score=item.parent_score,
            best_child_rank=item.best_child_rank,
            supporting_child_ids=item.supporting_child_ids,
        )
        for rank, item in enumerate(selected, start=1)
    )


def _validate_parent_against_hits(
    parent: ParentRecord,
    aggregate: ParentAggregate,
    hits_by_id: dict[str, ChildEvidenceHit],
    request: HybridRetrievalRequest,
) -> tuple[ChildEvidenceHit, ...]:
    if parent.parent_id != aggregate.parent_id:
        raise RetrievalInvariantError("hydrated parent ID does not match its aggregate")
    if parent.generation_id != request.generation_id:
        raise RetrievalInvariantError("hydrated parent generation mismatch")
    if parent.subject != request.subject or parent.subject != aggregate.subject:
        raise RetrievalInvariantError("hydrated parent subject mismatch")
    if parent.source_relpath != aggregate.source_relpath:
        raise RetrievalInvariantError("hydrated parent source mismatch")

    supporting_hits = tuple(
        hits_by_id[child_id] for child_id in aggregate.supporting_child_ids
    )
    for hit in supporting_hits:
        child = hit.document
        metadata = child.metadata
        identity_fields = (
            (metadata.parent_id, parent.parent_id),
            (metadata.doc_id, parent.doc_id),
            (metadata.subject, parent.subject),
            (metadata.generation_id, parent.generation_id),
            (metadata.policy_id, parent.policy_id),
            (metadata.source_file, parent.source_file),
            (metadata.source_relpath, parent.source_relpath),
            (metadata.source_file_sha1, parent.source_file_sha1),
            (metadata.doc_type, parent.doc_type),
            (metadata.section_id, parent.section_id),
            (metadata.section_title, parent.section_title),
            (metadata.section_path, parent.section_path),
            (metadata.pagination_kind, parent.pagination_kind),
        )
        if any(
            child_value != parent_value for child_value, parent_value in identity_fields
        ):
            raise RetrievalInvariantError(
                "hydrated parent provenance does not match a supporting child"
            )
        if (
            not parent.page_start
            <= metadata.page_start
            <= metadata.page_end
            <= parent.page_end
        ):
            raise RetrievalInvariantError(
                "supporting child page range exceeds its hydrated parent"
            )
        if metadata.start_char != parent.start_char + metadata.child_start_in_parent:
            raise RetrievalInvariantError(
                "supporting child absolute start offset is inconsistent"
            )
        if metadata.end_char != parent.start_char + metadata.child_end_in_parent:
            raise RetrievalInvariantError(
                "supporting child absolute end offset is inconsistent"
            )
        if metadata.child_end_in_parent > parent.parent_chars:
            raise RetrievalInvariantError("supporting child exceeds hydrated parent")
        if (
            parent.content[
                metadata.child_start_in_parent : metadata.child_end_in_parent
            ]
            != child.content
        ):
            raise RetrievalInvariantError(
                "supporting child content is not the exact hydrated-parent slice"
            )
    return supporting_hits


def _merge_intervals(
    intervals: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    ordered = sorted(intervals)
    if not ordered:
        raise ValueError("at least one interval is required")
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _expand_parent(
    parent: ParentRecord,
    aggregate: ParentAggregate,
    supporting_hits: tuple[ChildEvidenceHit, ...],
    policy: HybridRetrievalPolicy,
) -> HydratedParentContext:
    windows: tuple[ParentContextWindow, ...]
    if parent.parent_chars <= policy.full_parent_max_chars:
        windows = (
            ParentContextWindow(
                schema_version="parent_context_window_v1",
                start_in_parent=0,
                end_in_parent=parent.parent_chars,
                content=parent.content,
            ),
        )
        mode: Literal["full_parent", "hit_window"] = "full_parent"
    else:
        intervals = tuple(
            (
                max(
                    0,
                    hit.document.metadata.child_start_in_parent
                    - policy.hit_window_chars_per_side,
                ),
                min(
                    parent.parent_chars,
                    hit.document.metadata.child_end_in_parent
                    + policy.hit_window_chars_per_side,
                ),
            )
            for hit in supporting_hits
        )
        windows = tuple(
            ParentContextWindow(
                schema_version="parent_context_window_v1",
                start_in_parent=start,
                end_in_parent=end,
                content=parent.content[start:end],
            )
            for start, end in _merge_intervals(intervals)
        )
        mode = "hit_window"
    return HydratedParentContext(
        schema_version="hydrated_parent_context_v1",
        rank=aggregate.rank,
        parent=parent,
        supporting_child_ids=aggregate.supporting_child_ids,
        expansion_mode=mode,
        heading=parent.section_title,
        windows=windows,
    )


class ParentChildHybridRetriever:
    """Strict candidate retriever with no request-time baseline fallback."""

    def __init__(
        self,
        *,
        policy: HybridRetrievalPolicy,
        vector_search: ChildSearchChannel,
        bm25_search: ChildSearchChannel,
        reranker: ChildReranker,
        parent_hydrator: ParentHydrator,
    ) -> None:
        self._policy = policy
        self._vector_search = vector_search
        self._bm25_search = bm25_search
        self._reranker = reranker
        self._parent_hydrator = parent_hydrator
        self._retrieval_fingerprint = compute_retrieval_fingerprint(policy)

    def retrieve_children(
        self,
        request: HybridRetrievalRequest,
    ) -> HybridChildRetrievalResult:
        """Run precision retrieval for Judge input without hydrating parent bodies."""

        total_start = perf_counter_ns()
        vector_start = perf_counter_ns()
        try:
            raw_vector = self._vector_search.search(
                query=request.query,
                subject=request.subject,
                generation_id=request.generation_id,
                top_k=self._policy.vector_top_k,
            )
        except Exception as exc:
            raise RetrievalChannelError(
                "required vector channel failed: " + type(exc).__name__
            ) from exc
        vector_ms = _elapsed_ms(vector_start)
        vector_results = _validate_search_results(
            raw_vector,
            channel_name="vector",
            request=request,
            top_k=self._policy.vector_top_k,
        )

        bm25_start = perf_counter_ns()
        try:
            raw_bm25 = self._bm25_search.search(
                query=request.query,
                subject=request.subject,
                generation_id=request.generation_id,
                top_k=self._policy.bm25_top_k,
            )
        except Exception as exc:
            raise RetrievalChannelError(
                "required BM25 channel failed: " + type(exc).__name__
            ) from exc
        bm25_ms = _elapsed_ms(bm25_start)
        bm25_results = _validate_search_results(
            raw_bm25,
            channel_name="bm25",
            request=request,
            top_k=self._policy.bm25_top_k,
        )

        fused_all = _merge_child_channels(vector_results, bm25_results, self._policy)
        fused = fused_all[: self._policy.reranker_top_n]
        if not fused:
            return HybridChildRetrievalResult(
                schema_version="hybrid_child_retrieval_result_v1",
                status="empty",
                request=request,
                retrieval_fingerprint=self._retrieval_fingerprint,
                ranked_children=(),
                ranked_parents=(),
                timings=RetrievalTimings(
                    schema_version="retrieval_timings_v1",
                    vector_ms=vector_ms,
                    bm25_ms=bm25_ms,
                    reranker_ms=0.0,
                    hydrate_ms=0.0,
                    total_ms=_elapsed_ms(total_start),
                ),
            )

        rerank_candidates = tuple(
            RerankCandidate(
                schema_version="rerank_candidate_v1",
                child_id=item.document.metadata.child_id,
                content=item.document.content,
            )
            for item in fused
        )
        reranker_start = perf_counter_ns()
        try:
            raw_reranker = self._reranker.rerank(
                query=request.query,
                candidates=rerank_candidates,
            )
        except Exception as exc:
            raise RetrievalChannelError(
                "required reranker failed: " + type(exc).__name__
            ) from exc
        reranker_ms = _elapsed_ms(reranker_start)
        reranker_scores = _validate_reranker_results(
            raw_reranker,
            submitted_child_ids=tuple(item.child_id for item in rerank_candidates),
        )
        child_hits = _build_child_hits(fused, reranker_scores)
        parent_aggregates = _select_parent_aggregates(child_hits, self._policy)
        if not parent_aggregates:
            raise RetrievalInvariantError(
                "non-empty reranked children produced no selectable parents"
            )

        return HybridChildRetrievalResult(
            schema_version="hybrid_child_retrieval_result_v1",
            status="ok",
            request=request,
            retrieval_fingerprint=self._retrieval_fingerprint,
            ranked_children=child_hits,
            ranked_parents=parent_aggregates,
            timings=RetrievalTimings(
                schema_version="retrieval_timings_v1",
                vector_ms=vector_ms,
                bm25_ms=bm25_ms,
                reranker_ms=reranker_ms,
                hydrate_ms=0.0,
                total_ms=_elapsed_ms(total_start),
            ),
        )

    def hydrate_kept_parents(
        self,
        result: HybridChildRetrievalResult,
        kept_child_ids: Sequence[str],
    ) -> tuple[HydratedParentContext, ...]:
        """Hydrate only Judge-kept children, merging children by selected parent."""

        kept = tuple(kept_child_ids)
        if len(kept) != len(set(kept)) or any(not child_id for child_id in kept):
            raise ParentHydrationError("kept child IDs must be unique and non-empty")
        if result.status == "empty":
            if kept:
                raise ParentHydrationError(
                    "empty retrieval cannot hydrate kept children"
                )
            return ()
        hits_by_id = {
            hit.document.metadata.child_id: hit for hit in result.ranked_children
        }
        unknown = set(kept) - set(hits_by_id)
        if unknown:
            raise ParentHydrationError("Judge returned unknown child evidence IDs")
        selected_parent_ids = {
            aggregate.parent_id for aggregate in result.ranked_parents
        }
        if any(
            hits_by_id[child_id].document.metadata.parent_id not in selected_parent_ids
            for child_id in kept
        ):
            raise ParentHydrationError(
                "Judge kept a child whose parent was not selected for hydration"
            )

        kept_set = set(kept)
        selected_aggregates: list[ParentAggregate] = []
        for aggregate in result.ranked_parents:
            supporting = tuple(
                child_id
                for child_id in aggregate.supporting_child_ids
                if child_id in kept_set
            )
            if not supporting:
                continue
            scores = tuple(hits_by_id[child_id].rerank_score for child_id in supporting)
            selected_aggregates.append(
                ParentAggregate(
                    schema_version="parent_aggregate_v1",
                    rank=len(selected_aggregates) + 1,
                    parent_id=aggregate.parent_id,
                    subject=aggregate.subject,
                    source_relpath=aggregate.source_relpath,
                    parent_score=aggregate_parent_score(
                        scores,
                        support_lambda=self._policy.parent_support_lambda,
                    ),
                    best_child_rank=min(
                        hits_by_id[child_id].final_rank for child_id in supporting
                    ),
                    supporting_child_ids=supporting,
                )
            )
        if not selected_aggregates:
            return ()

        parent_ids = tuple(parent.parent_id for parent in selected_aggregates)
        try:
            raw_parents = self._parent_hydrator.get_many(parent_ids)
        except Exception as exc:
            raise ParentHydrationError(
                "authoritative parent hydration failed: " + type(exc).__name__
            ) from exc
        if not isinstance(raw_parents, tuple):
            raise ParentHydrationError("parent hydrator must return a tuple")

        validated_parents: list[ParentRecord] = []
        for index, raw_parent in enumerate(raw_parents):
            try:
                parent = ParentRecord.model_validate(raw_parent)
            except ValidationError as exc:
                raise ParentHydrationError(
                    f"hydrated parent at index {index} is invalid"
                ) from exc
            validated_parents.append(parent)
        if tuple(parent.parent_id for parent in validated_parents) != parent_ids:
            raise ParentHydrationError(
                "parent hydrator did not return the exact requested order and ID set"
            )

        hydrated_contexts: list[HydratedParentContext] = []
        for aggregate, parent in zip(
            selected_aggregates, validated_parents, strict=True
        ):
            supporting_hits = _validate_parent_against_hits(
                parent,
                aggregate,
                hits_by_id,
                result.request,
            )
            hydrated_contexts.append(
                _expand_parent(parent, aggregate, supporting_hits, self._policy)
            )

        return tuple(hydrated_contexts)

    def hydrate_kept_multi(
        self,
        result: MultiBranchHybridChildResult,
        kept_child_ids: Sequence[str],
    ) -> tuple[HydratedParentContext, ...]:
        """Hydrate Judge-kept parents after cross-branch child-only retrieval."""

        kept = tuple(kept_child_ids)
        if len(kept) != len(set(kept)) or any(not child_id for child_id in kept):
            raise ParentHydrationError("kept child IDs must be unique and non-empty")
        if result.status == "empty":
            if kept:
                raise ParentHydrationError(
                    "empty multi-branch retrieval cannot hydrate kept children"
                )
            return ()

        hits_by_id: dict[str, ChildEvidenceHit] = {}
        for branch_result in result.branch_results:
            for hit in branch_result.ranked_children:
                child_id = hit.document.metadata.child_id
                existing = hits_by_id.get(child_id)
                if existing is not None and existing.document != hit.document:
                    raise ParentHydrationError(
                        "branches returned conflicting content for one child ID"
                    )
                if existing is None or (
                    hit.rerank_score,
                    -hit.final_rank,
                    hit.rrf_score,
                ) > (
                    existing.rerank_score,
                    -existing.final_rank,
                    existing.rrf_score,
                ):
                    hits_by_id[child_id] = hit

        unknown = set(kept) - set(hits_by_id)
        if unknown:
            raise ParentHydrationError("Judge returned unknown child evidence IDs")
        selected_parent_ids = {parent.parent_id for parent in result.ranked_parents}
        eligible_child_ids = {
            child_id
            for branch_result in result.branch_results
            for aggregate in branch_result.ranked_parents
            if aggregate.parent_id in selected_parent_ids
            for child_id in aggregate.supporting_child_ids
        }
        ineligible = set(kept) - eligible_child_ids
        if ineligible:
            raise ParentHydrationError(
                "Judge kept children outside selected parent support"
            )
        if any(
            hits_by_id[child_id].document.metadata.parent_id not in selected_parent_ids
            for child_id in kept
        ):
            raise ParentHydrationError(
                "Judge kept a child whose cross-branch parent was not selected"
            )

        kept_set = set(kept)
        selected_aggregates: list[ParentAggregate] = []
        request_by_subject = {
            branch.request.subject: branch.request for branch in result.request.branches
        }
        requests_for_aggregates: list[HybridRetrievalRequest] = []
        for cross_parent in result.ranked_parents:
            supporting = tuple(
                sorted(
                    (
                        child_id
                        for child_id in kept_set
                        if hits_by_id[child_id].document.metadata.parent_id
                        == cross_parent.parent_id
                    ),
                    key=lambda child_id: (
                        hits_by_id[child_id].final_rank,
                        child_id,
                    ),
                )
            )
            if not supporting:
                continue
            branch_request = request_by_subject.get(cross_parent.subject)
            if branch_request is None:
                raise ParentHydrationError(
                    "selected parent subject has no corresponding branch request"
                )
            selected_aggregates.append(
                ParentAggregate(
                    schema_version="parent_aggregate_v1",
                    rank=len(selected_aggregates) + 1,
                    parent_id=cross_parent.parent_id,
                    subject=cross_parent.subject,
                    source_relpath=cross_parent.source_relpath,
                    parent_score=aggregate_parent_score(
                        tuple(
                            hits_by_id[child_id].rerank_score for child_id in supporting
                        ),
                        support_lambda=self._policy.parent_support_lambda,
                    ),
                    best_child_rank=min(
                        hits_by_id[child_id].final_rank for child_id in supporting
                    ),
                    supporting_child_ids=supporting,
                )
            )
            requests_for_aggregates.append(branch_request)
        if not selected_aggregates:
            return ()

        parent_ids = tuple(parent.parent_id for parent in selected_aggregates)
        try:
            raw_parents = self._parent_hydrator.get_many(parent_ids)
        except Exception as exc:
            raise ParentHydrationError(
                "authoritative parent hydration failed: " + type(exc).__name__
            ) from exc
        if not isinstance(raw_parents, tuple):
            raise ParentHydrationError("parent hydrator must return a tuple")
        try:
            parents = tuple(ParentRecord.model_validate(item) for item in raw_parents)
        except ValidationError as exc:
            raise ParentHydrationError(
                "hydrated multi-branch parent is invalid"
            ) from exc
        if tuple(parent.parent_id for parent in parents) != parent_ids:
            raise ParentHydrationError(
                "parent hydrator did not return the exact requested order and ID set"
            )

        hydrated: list[HydratedParentContext] = []
        for aggregate, parent, branch_request in zip(
            selected_aggregates,
            parents,
            requests_for_aggregates,
            strict=True,
        ):
            supporting_hits = _validate_parent_against_hits(
                parent,
                aggregate,
                hits_by_id,
                branch_request,
            )
            hydrated.append(
                _expand_parent(parent, aggregate, supporting_hits, self._policy)
            )
        return tuple(hydrated)

    def retrieve(self, request: HybridRetrievalRequest) -> HybridRetrievalResult:
        """Compatibility full retrieval; production Judge flow uses two stages."""

        child_result = self.retrieve_children(request)
        if child_result.status == "empty":
            return HybridRetrievalResult(
                schema_version="hybrid_retrieval_result_v1",
                status="empty",
                request=request,
                retrieval_fingerprint=child_result.retrieval_fingerprint,
                ranked_children=(),
                ranked_parents=(),
                hydrated_parents=(),
                timings=child_result.timings,
            )
        kept_child_ids = tuple(
            child_id
            for parent in child_result.ranked_parents
            for child_id in parent.supporting_child_ids
        )
        hydrate_start = perf_counter_ns()
        hydrated_contexts = self.hydrate_kept_parents(
            child_result,
            kept_child_ids,
        )
        hydrate_ms = _elapsed_ms(hydrate_start)
        return HybridRetrievalResult(
            schema_version="hybrid_retrieval_result_v1",
            status="ok",
            request=request,
            retrieval_fingerprint=self._retrieval_fingerprint,
            ranked_children=child_result.ranked_children,
            ranked_parents=child_result.ranked_parents,
            hydrated_parents=hydrated_contexts,
            timings=RetrievalTimings(
                schema_version="retrieval_timings_v1",
                vector_ms=child_result.timings.vector_ms,
                bm25_ms=child_result.timings.bm25_ms,
                reranker_ms=child_result.timings.reranker_ms,
                hydrate_ms=hydrate_ms,
                total_ms=child_result.timings.total_ms + hydrate_ms,
            ),
        )

    def retrieve_children_multi(
        self,
        request: MultiBranchHybridRequest,
    ) -> MultiBranchHybridChildResult:
        """Retrieve and fuse branch parents without reading parent bodies."""

        branch_results = tuple(
            self.retrieve_children(branch.request) for branch in request.branches
        )
        ranked = _fuse_cross_branch_parents(
            policy=self._policy,
            request=request,
            branch_results=branch_results,
        )
        return MultiBranchHybridChildResult(
            schema_version="multi_branch_hybrid_child_result_v1",
            status="ok" if ranked else "empty",
            request=request,
            branch_results=branch_results,
            ranked_parents=ranked,
        )

    def retrieve_multi(
        self,
        request: MultiBranchHybridRequest,
    ) -> MultiBranchHybridResult:
        """Retrieve every branch strictly, then fuse parent ranks with weighted RRF."""

        branch_results = tuple(
            self.retrieve(branch.request) for branch in request.branches
        )
        ranked = _fuse_cross_branch_parents(
            policy=self._policy,
            request=request,
            branch_results=branch_results,
        )
        return MultiBranchHybridResult(
            schema_version="multi_branch_hybrid_result_v1",
            status="ok" if ranked else "empty",
            request=request,
            branch_results=branch_results,
            ranked_parents=ranked,
        )
