"""Graph adapter for strict Parent--Child evidence provenance."""

from __future__ import annotations

from collections.abc import Sequence

from src.graph.evidence import EvidenceCandidate, EvidenceJudgeItem
from src.rag.parent_child.handoff import JudgeKeepDecision, LocalEvidenceRef


def local_refs_to_evidence_candidates(
    refs: Sequence[LocalEvidenceRef],
    *,
    provider: str,
    role: str,
    purpose: str,
    branch_status: str | None,
    primary_revision: int,
    primary_config_fingerprint: str,
) -> tuple[EvidenceCandidate, ...]:
    """Validate child evidence and attach the active primary identity."""

    if isinstance(primary_revision, bool) or primary_revision < 1:
        raise ValueError("primary_revision must be positive")
    if len(primary_config_fingerprint) != 64 or any(
        char not in "0123456789abcdef" for char in primary_config_fingerprint
    ):
        raise ValueError("primary_config_fingerprint must be lowercase SHA256")
    candidates = []
    for ref in refs:
        payload = ref.graph_candidate_payload(
            provider=provider,
            role=role,
            purpose=purpose,
            branch_status=branch_status,
        )
        raw_metadata = payload.get("metadata")
        if not isinstance(raw_metadata, dict) or any(
            not isinstance(key, str) for key in raw_metadata
        ):
            raise ValueError("Graph evidence metadata must be a string-keyed mapping")
        metadata = {
            key: value for key, value in raw_metadata.items() if isinstance(key, str)
        }
        metadata["primary_revision"] = primary_revision
        metadata["primary_config_fingerprint"] = primary_config_fingerprint
        payload["metadata"] = metadata
        candidates.append(EvidenceCandidate.model_validate(payload))
    evidence_ids = tuple(candidate.evidence_id for candidate in candidates)
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("Graph evidence candidate IDs must be unique")
    return tuple(candidates)


def judge_items_to_keep_decisions(
    items: Sequence[EvidenceJudgeItem],
) -> tuple[JudgeKeepDecision, ...]:
    """Project the unchanged Judge schema to strict hydration decisions."""

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
        raise ValueError("Judge evidence candidate IDs must be unique")
    return decisions


__all__ = ["judge_items_to_keep_decisions", "local_refs_to_evidence_candidates"]
