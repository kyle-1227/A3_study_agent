"""Explicit candidate Graph nodes for child Judge and post-Judge parent hydration."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.graph.academic import build_retrieval_branches_for_parent_child
from src.graph.evidence import EvidenceCandidate, EvidenceJudgeItem, EvidenceJudgeOutput
from src.graph.parent_child_handoff import (
    judge_items_to_keep_decisions,
    local_refs_to_evidence_candidates,
)
from src.graph.state import CONTEXT_CLEAR, LearningState
from src.rag.parent_child.handoff import (
    LocalEvidenceRef,
    ParentContextEvidenceItem,
    build_multi_local_evidence_refs,
    kept_child_ids_from_decisions,
    parent_context_items,
)
from src.rag.parent_child.models import ChildDocument
from src.rag.parent_child.retrieval import (
    HydratedParentContext,
    HybridRetrievalRequest,
    MultiBranchHybridChildResult,
    MultiBranchHybridRequest,
    WeightedHybridBranch,
    compute_retrieval_fingerprint,
)
from src.rag.parent_child.runtime_loader import LoadedGenerationRuntime


class ParentChildGraphContractError(RuntimeError):
    """Candidate Graph state or injected retrieval runtime is inconsistent."""


class ParentChildGraphRetriever(Protocol):
    def retrieve_children_multi(
        self,
        request: MultiBranchHybridRequest,
    ) -> MultiBranchHybridChildResult: ...

    def hydrate_kept_multi(
        self,
        result: MultiBranchHybridChildResult,
        kept_child_ids: Sequence[str],
    ) -> tuple[HydratedParentContext, ...]: ...


@dataclass(frozen=True, slots=True)
class ParentChildGraphRuntime:
    """Generation-pinned dependencies required by the explicit candidate graph."""

    generation_id: str
    available_subjects: tuple[str, ...]
    retriever: ParentChildGraphRetriever
    retrieval_fingerprint: str
    cross_branch_rrf_k: int
    parent_top_k: int
    preview_max_chars: int

    def __post_init__(self) -> None:
        if not self.generation_id or self.generation_id != self.generation_id.strip():
            raise ValueError("generation_id must be nonblank and stripped")
        if not self.available_subjects or self.available_subjects != tuple(
            sorted(set(self.available_subjects))
        ):
            raise ValueError("available_subjects must be non-empty, sorted, and unique")
        if len(self.retrieval_fingerprint) != 64 or any(
            char not in "0123456789abcdef" for char in self.retrieval_fingerprint
        ):
            raise ValueError("retrieval_fingerprint must be lowercase SHA256")
        if (
            isinstance(self.cross_branch_rrf_k, bool)
            or self.cross_branch_rrf_k <= 0
            or isinstance(self.parent_top_k, bool)
            or self.parent_top_k <= 0
            or isinstance(self.preview_max_chars, bool)
            or self.preview_max_chars <= 0
        ):
            raise ValueError(
                "candidate graph retrieval limits must be positive integers"
            )

    @property
    def graph_handoff_fingerprint(self) -> str:
        """Fingerprint retrieval plus cross-branch and Judge handoff controls."""

        payload = json.dumps(
            {
                "cross_branch_rrf_k": self.cross_branch_rrf_k,
                "judge_preview_max_chars": self.preview_max_chars,
                "parent_top_k": self.parent_top_k,
                "retrieval_fingerprint": self.retrieval_fingerprint,
                "schema_version": "parent_child_graph_handoff_policy_v1",
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class ParentChildRetrievalBranch(BaseModel):
    """Strict candidate-only projection of one normalized academic branch."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    subject: str = Field(min_length=1)
    role: str = Field(min_length=1)
    local_retrieval_query: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    priority: float = Field(gt=0)
    priority_explicit: bool

    @field_validator("subject", "role", "local_retrieval_query", "purpose")
    @classmethod
    def validate_stripped_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("candidate branch text must already be stripped")
        return value

    @field_validator("priority")
    @classmethod
    def validate_finite_priority(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("candidate branch priority must be finite")
        return value

    @field_validator("priority_explicit")
    @classmethod
    def validate_explicit_priority(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("candidate branch priority must be explicitly supplied")
        return value


def parent_child_graph_runtime_from_loaded(
    *,
    loaded: LoadedGenerationRuntime,
) -> ParentChildGraphRuntime:
    """Bind one verified loaded generation to the strict Graph handoff limits."""

    if not isinstance(loaded, LoadedGenerationRuntime):
        raise TypeError("loaded must be a verified LoadedGenerationRuntime")
    if loaded.generation_id != loaded.resources.generation_id:
        raise ParentChildGraphContractError("loaded runtime generation is inconsistent")
    return ParentChildGraphRuntime(
        generation_id=loaded.generation_id,
        available_subjects=loaded.available_subjects,
        retriever=loaded.retriever(),
        retrieval_fingerprint=compute_retrieval_fingerprint(loaded.retrieval_policy),
        cross_branch_rrf_k=loaded.cross_branch_rrf_k,
        parent_top_k=loaded.retrieval_policy.multi_subject_max_parents,
        preview_max_chars=loaded.judge_preview_max_chars,
    )


def _required_state_text(state: LearningState, key: str) -> str:
    value = state.get(key)
    if not isinstance(value, str) or not value or value != value.strip():
        raise ParentChildGraphContractError(
            f"state.{key} must be nonblank and stripped"
        )
    return value


def _strict_branch(branch: object, *, branch_index: int) -> ParentChildRetrievalBranch:
    if not isinstance(branch, dict):
        raise ParentChildGraphContractError(
            f"retrieval branch {branch_index} must be a mapping"
        )
    try:
        return ParentChildRetrievalBranch.model_validate(
            {
                "subject": branch.get("subject"),
                "role": branch.get("role"),
                "local_retrieval_query": branch.get("local_retrieval_query"),
                "purpose": branch.get("purpose"),
                "priority": branch.get("priority"),
                "priority_explicit": branch.get("_parent_child_priority_explicit"),
            }
        )
    except ValidationError as exc:
        raise ParentChildGraphContractError(
            f"retrieval branch {branch_index} violates the candidate contract"
        ) from exc


def _child_branch_labels(
    result: MultiBranchHybridChildResult,
    labels: dict[str, tuple[str, str]],
) -> dict[str, tuple[str, str]]:
    """Select role/purpose from the best branch that supports each child."""

    choices: dict[str, list[tuple[tuple[int, int, float, str], str]]] = {}
    for branch, branch_result in zip(
        result.request.branches,
        result.branch_results,
        strict=True,
    ):
        hit_rank = {
            hit.document.metadata.child_id: hit.final_rank
            for hit in branch_result.ranked_children
        }
        for aggregate in branch_result.ranked_parents:
            for child_id in aggregate.supporting_child_ids:
                child_rank = hit_rank.get(child_id)
                if child_rank is None:
                    raise ParentChildGraphContractError(
                        "branch parent support references an absent child hit"
                    )
                choices.setdefault(child_id, []).append(
                    (
                        (
                            child_rank,
                            aggregate.rank,
                            -branch.weight,
                            branch.branch_id,
                        ),
                        branch.branch_id,
                    )
                )
    selected: dict[str, tuple[str, str]] = {}
    for child_id, candidates in choices.items():
        branch_id = min(candidates, key=lambda item: item[0])[1]
        label = labels.get(branch_id)
        if label is None:
            raise ParentChildGraphContractError(
                "supporting child references an unknown retrieval branch"
            )
        selected[child_id] = label
    return selected


def _candidate_originals(
    result: MultiBranchHybridChildResult,
    refs: tuple[LocalEvidenceRef, ...],
) -> dict[str, dict]:
    children: dict[str, ChildDocument] = {}
    for branch_result in result.branch_results:
        for hit in branch_result.ranked_children:
            child_id = hit.document.metadata.child_id
            existing = children.get(child_id)
            if existing is not None and existing != hit.document:
                raise ParentChildGraphContractError(
                    "branches returned conflicting child documents"
                )
            children[child_id] = hit.document
    originals: dict[str, dict] = {}
    for ref in refs:
        child = children.get(ref.child_id)
        if child is None:
            raise ParentChildGraphContractError(
                "Judge ref has no retrieved child document"
            )
        content = child.content
        originals[ref.evidence_id] = {
            "type": "rag",
            "source_type": "local_rag",
            "provider": "chroma_parent_child",
            "source": ref.source_relpath,
            "content": content,
            "page_content": content,
            "retrieval_subject": ref.subject,
            "subject": ref.subject,
            "child_id": ref.child_id,
            "parent_id": ref.parent_id,
            "generation_id": ref.generation_id,
            "policy_id": ref.policy_id,
            "page_start": ref.page_start,
            "page_end": ref.page_end,
            "start_char": ref.start_char,
            "end_char": ref.end_char,
            "metadata": ref.model_dump(mode="json"),
        }
    return originals


def make_parent_child_rag_node(
    runtime: ParentChildGraphRuntime,
) -> Callable[[LearningState], Awaitable[dict]]:
    """Create a child-only retrieval node bound to one immutable generation."""

    async def parent_child_rag_retrieve(state: LearningState) -> dict:
        request_id = _required_state_text(state, "request_id")
        branches, _branch_debug = build_retrieval_branches_for_parent_child(state)
        if not branches:
            raise ParentChildGraphContractError(
                "candidate retrieval requires at least one normalized branch"
            )

        weighted: list[WeightedHybridBranch] = []
        branch_labels: dict[str, tuple[str, str]] = {}
        for index, raw_branch in enumerate(branches):
            branch = _strict_branch(raw_branch, branch_index=index)
            if branch.subject not in runtime.available_subjects:
                raise ParentChildGraphContractError(
                    f"retrieval branch subject is unavailable: {branch.subject!r}"
                )
            branch_id = f"branch-{index}:{branch.subject}:{branch.role}"
            branch_labels[branch_id] = (branch.role, branch.purpose)
            weighted.append(
                WeightedHybridBranch(
                    schema_version="weighted_hybrid_branch_v1",
                    branch_id=branch_id,
                    weight=branch.priority,
                    request=HybridRetrievalRequest(
                        schema_version="hybrid_retrieval_request_v1",
                        request_id=f"{request_id}:branch-{index}",
                        query=branch.local_retrieval_query,
                        subject=branch.subject,
                        generation_id=runtime.generation_id,
                    ),
                )
            )
        request = MultiBranchHybridRequest(
            schema_version="multi_branch_hybrid_request_v1",
            request_id=request_id,
            generation_id=runtime.generation_id,
            branches=tuple(weighted),
            cross_branch_rrf_k=runtime.cross_branch_rrf_k,
            parent_top_k=runtime.parent_top_k,
        )
        result = await asyncio.to_thread(
            runtime.retriever.retrieve_children_multi, request
        )
        if result.request != request:
            raise ParentChildGraphContractError(
                "candidate retriever returned a different multi-branch request"
            )
        if any(
            branch_result.retrieval_fingerprint != runtime.retrieval_fingerprint
            for branch_result in result.branch_results
        ):
            raise ParentChildGraphContractError(
                "candidate retriever returned an unexpected retrieval fingerprint"
            )
        refs = build_multi_local_evidence_refs(
            result,
            preview_max_chars=runtime.preview_max_chars,
        )
        child_labels = _child_branch_labels(result, branch_labels)
        candidates: list[EvidenceCandidate] = []
        for ref in refs:
            label = child_labels.get(ref.child_id)
            if label is None:
                raise ParentChildGraphContractError(
                    "Judge ref is absent from branch-selected child support"
                )
            role, purpose = label
            candidates.extend(
                local_refs_to_evidence_candidates(
                    (ref,),
                    provider="chroma_parent_child",
                    role=role,
                    purpose=purpose,
                    branch_status=None,
                )
            )
        candidate_ids = tuple(candidate.evidence_id for candidate in candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ParentChildGraphContractError(
                "candidate graph produced duplicate child evidence IDs"
            )
        return {
            "local_evidence_candidates": [
                candidate.model_dump(mode="json") for candidate in candidates
            ],
            "local_evidence_originals": _candidate_originals(result, refs),
            "retrieval_branch_mode": "parent_child_hybrid",
            "parent_child_retrieval_result": result.model_dump(mode="json"),
            "parent_child_local_refs": [ref.model_dump(mode="json") for ref in refs],
            "parent_child_generation_id": runtime.generation_id,
            "parent_child_retrieval_fingerprint": (runtime.graph_handoff_fingerprint),
            "parent_child_hydration": {},
        }

    return parent_child_rag_retrieve


def _parent_context_doc(
    *,
    item: ParentContextEvidenceItem,
    judge_by_id: dict[str, EvidenceJudgeItem],
    candidate_by_id: dict[str, EvidenceCandidate],
    ref_by_id: dict[str, LocalEvidenceRef],
    retrieval_fingerprint: str,
) -> dict:
    supporting_ids = item.supporting_child_ids
    try:
        judged = [judge_by_id[child_id] for child_id in supporting_ids]
        supporting_refs = [ref_by_id[child_id] for child_id in supporting_ids]
    except KeyError as exc:
        raise ParentChildGraphContractError(
            "hydrated parent support is absent from Judge child provenance"
        ) from exc
    best = max(judged, key=lambda value: (value.evidence_score, value.evidence_id))
    candidate = candidate_by_id.get(best.evidence_id)
    if candidate is None:
        raise ParentChildGraphContractError(
            "hydrated parent support is absent from Judge candidates"
        )
    return {
        "type": "rag",
        "source_type": "local_rag",
        "provider": "chroma_parent_child",
        "evidence_id": f"parent:{item.parent_id}",
        "parent_id": item.parent_id,
        "generation_id": item.generation_id,
        "policy_id": item.policy_id,
        "retrieval_fingerprint": retrieval_fingerprint,
        "subject": item.subject,
        "retrieval_subject": item.subject,
        "role": candidate.role,
        "retrieval_role": candidate.role,
        "purpose": candidate.purpose,
        "retrieval_purpose": candidate.purpose,
        "title": candidate.title,
        "source": item.source_relpath,
        "source_relpath": item.source_relpath,
        "page_start": item.page_start,
        "page_end": item.page_end,
        "supporting_child_ids": list(supporting_ids),
        "supporting_children": [
            {
                "child_id": ref.child_id,
                "page_start": ref.page_start,
                "page_end": ref.page_end,
                "start_char": ref.start_char,
                "end_char": ref.end_char,
                "child_start_in_parent": ref.child_start_in_parent,
                "child_end_in_parent": ref.child_end_in_parent,
            }
            for ref in supporting_refs
        ],
        "expansion_mode": item.expansion_mode,
        "window_spans": [list(span) for span in item.window_spans],
        "content": item.content,
        "judge_keep": True,
        "judge_quality": best.final_quality,
        "judge_relevance": best.relevance,
        "judge_authority": best.authority,
        "judge_usefulness": best.usefulness,
        "judge_risk": best.risk,
        "evidence_score": best.evidence_score,
        "relevance_score": best.evidence_score,
        "score": best.evidence_score,
        "score_source": "evidence_item_grader_parent_expansion",
        "score_scale": "0-1",
        "score_type": "task_relevance",
        "score_reason": best.score_reason,
        "evidence_type": best.evidence_type,
        "use_case": best.use_case,
        "coverage_contribution": best.coverage_contribution,
        "judge_reason": best.reason,
        "branch_status_score_source": "parent_child_hybrid_v1",
    }


def make_parent_child_hydration_node(
    runtime: ParentChildGraphRuntime,
) -> Callable[[LearningState], Awaitable[dict]]:
    """Create a post-Judge node that replaces local child context with parents."""

    async def hydrate_parent_child_context(state: LearningState) -> dict:
        if state.get("parent_child_generation_id") != runtime.generation_id:
            raise ParentChildGraphContractError(
                "candidate generation changed in-flight"
            )
        if (
            state.get("parent_child_retrieval_fingerprint")
            != runtime.graph_handoff_fingerprint
        ):
            raise ParentChildGraphContractError(
                "candidate Graph handoff policy changed in-flight"
            )
        result = MultiBranchHybridChildResult.model_validate_json(
            json.dumps(
                state.get("parent_child_retrieval_result"),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
        if (
            result.request.generation_id != runtime.generation_id
            or result.request.request_id != _required_state_text(state, "request_id")
        ):
            raise ParentChildGraphContractError(
                "stored child retrieval request differs from the active request"
            )
        if any(
            branch_result.retrieval_fingerprint != runtime.retrieval_fingerprint
            for branch_result in result.branch_results
        ):
            raise ParentChildGraphContractError(
                "stored child retrieval fingerprint differs from the runtime"
            )
        refs = tuple(
            LocalEvidenceRef.model_validate_json(
                json.dumps(
                    item,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            )
            for item in (state.get("parent_child_local_refs") or [])
        )
        parsed = EvidenceJudgeOutput.model_validate(state.get("evidence_judge_output"))
        candidates = tuple(
            EvidenceCandidate.model_validate(item)
            for item in (state.get("evidence_candidates") or [])
        )
        ref_by_id = {ref.evidence_id: ref for ref in refs}
        if len(ref_by_id) != len(refs) or any(
            ref.generation_id != runtime.generation_id for ref in refs
        ):
            raise ParentChildGraphContractError(
                "stored child references have duplicate or mismatched identities"
            )
        local_candidates = tuple(
            candidate
            for candidate in candidates
            if candidate.source_type == "local_rag"
            and (candidate.metadata or {}).get("schema_version")
            == "local_evidence_ref_v1"
        )
        local_ids = tuple(candidate.evidence_id for candidate in local_candidates)
        if len(local_ids) != len(set(local_ids)) or not set(local_ids).issubset(
            ref_by_id
        ):
            raise ParentChildGraphContractError(
                "judged local candidate identities differ from strict child refs"
            )
        selected_refs = tuple(ref_by_id[evidence_id] for evidence_id in local_ids)
        all_decisions = judge_items_to_keep_decisions(parsed.judged_evidence)
        decision_by_id = {decision.evidence_id: decision for decision in all_decisions}
        if not set(local_ids).issubset(decision_by_id):
            raise ParentChildGraphContractError(
                "Evidence Judge omitted local child decisions"
            )
        selected_decisions = tuple(
            decision_by_id[evidence_id] for evidence_id in local_ids
        )
        kept_ids = kept_child_ids_from_decisions(selected_refs, selected_decisions)
        contexts = await asyncio.to_thread(
            runtime.retriever.hydrate_kept_multi,
            result,
            kept_ids,
        )
        returned_support = {
            child_id
            for context in contexts
            for child_id in context.supporting_child_ids
        }
        if returned_support != set(kept_ids):
            raise ParentChildGraphContractError(
                "hydrated parent support differs from Judge-kept child IDs"
            )
        parent_items = parent_context_items(contexts)
        judge_by_id = {item.evidence_id: item for item in parsed.judged_evidence}
        candidate_by_id = {candidate.evidence_id: candidate for candidate in candidates}
        parent_docs = [
            _parent_context_doc(
                item=item,
                judge_by_id=judge_by_id,
                candidate_by_id=candidate_by_id,
                ref_by_id=ref_by_id,
                retrieval_fingerprint=runtime.graph_handoff_fingerprint,
            )
            for item in parent_items
        ]
        existing_context = state.get("context") or []
        if not isinstance(existing_context, list) or any(
            not isinstance(item, dict) for item in existing_context
        ):
            raise ParentChildGraphContractError(
                "judged context must be a list of mappings"
            )
        web_context = [
            item for item in existing_context if item.get("source_type") != "local_rag"
        ]
        combined = sorted(
            [*parent_docs, *web_context],
            key=lambda item: (
                -float(item.get("evidence_score") or 0.0),
                str(item.get("source_type") or ""),
                str(item.get("evidence_id") or item.get("source") or ""),
            ),
        )
        return {
            "context": [*CONTEXT_CLEAR, *combined],
            "graded_evidence": combined,
            "parent_child_hydration": {
                "schema_version": "parent_child_graph_hydration_v1",
                "generation_id": runtime.generation_id,
                "retrieval_fingerprint": runtime.graph_handoff_fingerprint,
                "parent_count": len(parent_docs),
                "supporting_child_count": len(kept_ids),
                "parent_ids": [context.parent.parent_id for context in contexts],
            },
        }

    return hydrate_parent_child_context


__all__ = [
    "ParentChildGraphContractError",
    "ParentChildGraphRetriever",
    "ParentChildGraphRuntime",
    "ParentChildRetrievalBranch",
    "make_parent_child_hydration_node",
    "make_parent_child_rag_node",
    "parent_child_graph_runtime_from_loaded",
]
