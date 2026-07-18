"""Judge-safe child references and post-Judge parent context handoff contracts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.rag.parent_child.retrieval import (
    ChildEvidenceHit,
    HybridChildRetrievalResult,
    HydratedParentContext,
    MultiBranchHybridChildResult,
)


class EvidenceHandoffError(RuntimeError):
    """Judge decisions or context handoff violate child-parent identity."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class LocalEvidenceRef(_StrictFrozenModel):
    """Local Judge candidate containing child text only and no parent body."""

    schema_version: Literal["local_evidence_ref_v1"]
    evidence_id: str = Field(min_length=1)
    child_id: str = Field(min_length=1)
    parent_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    source_relpath: str = Field(min_length=1)
    section_id: str = Field(min_length=1)
    section_title: str
    section_path: tuple[str, ...]
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    child_start_in_parent: int = Field(ge=0)
    child_end_in_parent: int = Field(gt=0)
    child_chars: int = Field(gt=0)
    final_rank: int = Field(gt=0)
    vector_rank: int | None
    bm25_rank: int | None
    rrf_score: float = Field(gt=0)
    rerank_score: float = Field(ge=0.0, le=1.0)
    content_preview: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        if self.evidence_id != self.child_id:
            raise ValueError("local evidence_id must equal child_id")
        if self.page_end < self.page_start:
            raise ValueError("local evidence page range is invalid")
        if self.end_char <= self.start_char:
            raise ValueError("local evidence absolute span is invalid")
        if self.child_end_in_parent <= self.child_start_in_parent:
            raise ValueError("local evidence parent-relative span is invalid")
        if self.child_chars != self.end_char - self.start_char:
            raise ValueError("local evidence child length conflicts with absolute span")
        if self.child_chars != (self.child_end_in_parent - self.child_start_in_parent):
            raise ValueError(
                "local evidence child length conflicts with parent-relative span"
            )
        if len(self.content_preview) > self.child_chars:
            raise ValueError("child preview cannot exceed child content length")
        return self

    def graph_candidate_payload(
        self,
        *,
        provider: str,
        role: str,
        purpose: str,
        branch_status: str | None,
    ) -> dict[str, object]:
        """Return an explicit payload for the later Graph adapter boundary."""

        if any(not value.strip() for value in (provider, role, purpose)):
            raise EvidenceHandoffError("Graph candidate labels must be nonblank")
        if branch_status is not None and not branch_status.strip():
            raise EvidenceHandoffError("Graph branch status must be null or nonblank")
        return {
            "evidence_id": self.evidence_id,
            "source_type": "local_rag",
            "provider": provider,
            "subject": self.subject,
            "role": role,
            "purpose": purpose,
            "title": self.section_title or self.source_file,
            "source": self.source_relpath,
            "url": "",
            "content_preview": self.content_preview,
            "raw_vector_score": None,
            "raw_vector_score_source": None,
            "raw_vector_score_direction": None,
            "rerank_score": self.rerank_score,
            "branch_status": branch_status,
            "branch_status_score_source": (
                "parent_child_hybrid_v1" if branch_status is not None else None
            ),
            "tavily_score": None,
            "tavily_query": None,
            "metadata": {
                "schema_version": self.schema_version,
                "child_id": self.child_id,
                "parent_id": self.parent_id,
                "generation_id": self.generation_id,
                "policy_id": self.policy_id,
                "source_relpath": self.source_relpath,
                "section_id": self.section_id,
                "section_path": list(self.section_path),
                "page_start": self.page_start,
                "page_end": self.page_end,
                "start_char": self.start_char,
                "end_char": self.end_char,
                "child_start_in_parent": self.child_start_in_parent,
                "child_end_in_parent": self.child_end_in_parent,
                "vector_rank": self.vector_rank,
                "bm25_rank": self.bm25_rank,
                "rrf_score": self.rrf_score,
            },
        }


class JudgeKeepDecision(_StrictFrozenModel):
    schema_version: Literal["judge_keep_decision_v1"]
    evidence_id: str = Field(min_length=1)
    keep: bool


class ParentContextEvidenceItem(_StrictFrozenModel):
    """Post-Judge CE item retaining exact supporting child and window provenance."""

    schema_version: Literal["parent_context_evidence_item_v1"]
    parent_id: str
    generation_id: str
    policy_id: str
    subject: str
    source_relpath: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    supporting_child_ids: tuple[str, ...] = Field(min_length=1)
    expansion_mode: Literal["full_parent", "hit_window"]
    window_spans: tuple[tuple[int, int], ...] = Field(min_length=1)
    content: str = Field(min_length=1)


def _local_ref_from_hit(
    hit: ChildEvidenceHit,
    *,
    final_rank: int,
    preview_max_chars: int,
) -> LocalEvidenceRef:
    metadata = hit.document.metadata
    return LocalEvidenceRef(
        schema_version="local_evidence_ref_v1",
        evidence_id=metadata.child_id,
        child_id=metadata.child_id,
        parent_id=metadata.parent_id,
        generation_id=metadata.generation_id,
        policy_id=metadata.policy_id,
        subject=metadata.subject,
        source_file=metadata.source_file,
        source_relpath=metadata.source_relpath,
        section_id=metadata.section_id,
        section_title=metadata.section_title,
        section_path=metadata.section_path,
        page_start=metadata.page_start,
        page_end=metadata.page_end,
        start_char=metadata.start_char,
        end_char=metadata.end_char,
        child_start_in_parent=metadata.child_start_in_parent,
        child_end_in_parent=metadata.child_end_in_parent,
        child_chars=metadata.child_chars,
        final_rank=final_rank,
        vector_rank=hit.vector_rank,
        bm25_rank=hit.bm25_rank,
        rrf_score=hit.rrf_score,
        rerank_score=hit.rerank_score,
        content_preview=hit.document.content[:preview_max_chars],
    )


def build_local_evidence_refs(
    result: HybridChildRetrievalResult,
    *,
    preview_max_chars: int,
) -> tuple[LocalEvidenceRef, ...]:
    """Create Judge candidates only for children supporting selected parents."""

    if preview_max_chars <= 0:
        raise ValueError("preview_max_chars must be positive")
    if result.status == "empty":
        return ()
    selected_parent_ids = {parent.parent_id for parent in result.ranked_parents}
    refs = [
        _local_ref_from_hit(
            hit,
            final_rank=hit.final_rank,
            preview_max_chars=preview_max_chars,
        )
        for hit in result.ranked_children
        if hit.document.metadata.parent_id in selected_parent_ids
    ]
    evidence_ids = tuple(ref.evidence_id for ref in refs)
    if len(evidence_ids) != len(set(evidence_ids)):
        raise EvidenceHandoffError("local Judge evidence IDs are not unique")
    return tuple(refs)


def build_multi_local_evidence_refs(
    result: MultiBranchHybridChildResult,
    *,
    preview_max_chars: int,
) -> tuple[LocalEvidenceRef, ...]:
    """Create unique child previews ordered by fused parent rank then child quality."""

    if preview_max_chars <= 0:
        raise ValueError("preview_max_chars must be positive")
    if result.status == "empty":
        return ()
    selected_parent_ids = {parent.parent_id for parent in result.ranked_parents}
    eligible_child_ids: dict[str, set[str]] = {
        parent_id: set() for parent_id in selected_parent_ids
    }
    for branch_result in result.branch_results:
        for aggregate in branch_result.ranked_parents:
            if aggregate.parent_id in selected_parent_ids:
                eligible_child_ids[aggregate.parent_id].update(
                    aggregate.supporting_child_ids
                )
    if any(not child_ids for child_ids in eligible_child_ids.values()):
        raise EvidenceHandoffError(
            "a fused parent has no branch-selected supporting children"
        )

    hits_by_parent: dict[str, dict[str, ChildEvidenceHit]] = {}
    for branch_result in result.branch_results:
        for hit in branch_result.ranked_children:
            metadata = hit.document.metadata
            if metadata.child_id not in eligible_child_ids.get(
                metadata.parent_id, set()
            ):
                continue
            parent_hits = hits_by_parent.setdefault(metadata.parent_id, {})
            existing = parent_hits.get(metadata.child_id)
            if existing is not None and existing.document != hit.document:
                raise EvidenceHandoffError(
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
                parent_hits[metadata.child_id] = hit

    ordered_hits: list[ChildEvidenceHit] = []
    for parent in result.ranked_parents:
        parent_hits = hits_by_parent.get(parent.parent_id, {})
        missing = eligible_child_ids[parent.parent_id] - set(parent_hits)
        if missing:
            raise EvidenceHandoffError(
                "branch-selected supporting children are absent from child hits"
            )
        ordered_hits.extend(
            sorted(
                parent_hits.values(),
                key=lambda hit: (
                    -hit.rerank_score,
                    hit.final_rank,
                    hit.document.metadata.child_id,
                ),
            )
        )
    refs = tuple(
        _local_ref_from_hit(
            hit,
            final_rank=rank,
            preview_max_chars=preview_max_chars,
        )
        for rank, hit in enumerate(ordered_hits, start=1)
    )
    evidence_ids = tuple(ref.evidence_id for ref in refs)
    if len(evidence_ids) != len(set(evidence_ids)):
        raise EvidenceHandoffError("multi-branch Judge evidence IDs are not unique")
    return refs


def kept_child_ids_from_decisions(
    refs: Sequence[LocalEvidenceRef],
    decisions: Sequence[JudgeKeepDecision],
) -> tuple[str, ...]:
    """Validate a complete Judge identity set and preserve retrieval rank order."""

    ref_ids = tuple(ref.evidence_id for ref in refs)
    decision_by_id: dict[str, JudgeKeepDecision] = {}
    for decision in decisions:
        if decision.evidence_id in decision_by_id:
            raise EvidenceHandoffError("Judge returned duplicate evidence IDs")
        decision_by_id[decision.evidence_id] = decision
    if set(decision_by_id) != set(ref_ids):
        raise EvidenceHandoffError(
            "Judge decision identity set differs from candidates"
        )
    return tuple(
        evidence_id for evidence_id in ref_ids if decision_by_id[evidence_id].keep
    )


def parent_context_items(
    contexts: Sequence[HydratedParentContext],
) -> tuple[ParentContextEvidenceItem, ...]:
    """Convert post-Judge hydration to CE items without start-only truncation."""

    items: list[ParentContextEvidenceItem] = []
    for context in contexts:
        window_content = "\n\n".join(window.content for window in context.windows)
        content = (
            f"{context.heading}\n\n{window_content}"
            if context.heading
            else window_content
        )
        parent = context.parent
        items.append(
            ParentContextEvidenceItem(
                schema_version="parent_context_evidence_item_v1",
                parent_id=parent.parent_id,
                generation_id=parent.generation_id,
                policy_id=parent.policy_id,
                subject=parent.subject,
                source_relpath=parent.source_relpath,
                page_start=parent.page_start,
                page_end=parent.page_end,
                supporting_child_ids=context.supporting_child_ids,
                expansion_mode=context.expansion_mode,
                window_spans=tuple(
                    (window.start_in_parent, window.end_in_parent)
                    for window in context.windows
                ),
                content=content,
            )
        )
    return tuple(items)


__all__ = [
    "EvidenceHandoffError",
    "JudgeKeepDecision",
    "LocalEvidenceRef",
    "ParentContextEvidenceItem",
    "build_local_evidence_refs",
    "build_multi_local_evidence_refs",
    "kept_child_ids_from_decisions",
    "parent_context_items",
]
