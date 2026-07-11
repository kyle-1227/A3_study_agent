"""Narrow Graph adapter for Judge-safe parent-child RAG evidence references."""

from __future__ import annotations

from collections.abc import Sequence

from src.graph.evidence import EvidenceCandidate, EvidenceJudgeItem
from src.rag.parent_child.handoff import (
    JudgeKeepDecision,
    LocalEvidenceRef,
)


def local_refs_to_evidence_candidates(
    refs: Sequence[LocalEvidenceRef],
    *,
    provider: str,
    role: str,
    purpose: str,
    branch_status: str | None,
) -> tuple[EvidenceCandidate, ...]:
    """Validate Graph candidates while keeping parent content out of the prompt path."""

    candidates = tuple(
        EvidenceCandidate.model_validate(
            ref.graph_candidate_payload(
                provider=provider,
                role=role,
                purpose=purpose,
                branch_status=branch_status,
            )
        )
        for ref in refs
    )
    evidence_ids = tuple(candidate.evidence_id for candidate in candidates)
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("Graph evidence candidate IDs must be unique")
    return candidates


def judge_items_to_keep_decisions(
    items: Sequence[EvidenceJudgeItem],
) -> tuple[JudgeKeepDecision, ...]:
    """Project the unchanged Judge schema to the strict RAG hydration decision."""

    decisions = tuple(
        JudgeKeepDecision(
            schema_version="judge_keep_decision_v1",
            evidence_id=item.evidence_id,
            keep=item.keep,
        )
        for item in items
    )
    evidence_ids = tuple(decision.evidence_id for decision in decisions)
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("Judge evidence IDs must be unique")
    return decisions


__all__ = [
    "judge_items_to_keep_decisions",
    "local_refs_to_evidence_candidates",
]
