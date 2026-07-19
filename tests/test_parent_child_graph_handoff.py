from __future__ import annotations

from src.graph.evidence import EvidenceJudgeItem
from src.graph.parent_child_handoff import (
    judge_items_to_keep_decisions,
    local_refs_to_evidence_candidates,
)
from src.rag.parent_child.handoff import LocalEvidenceRef


def _ref() -> LocalEvidenceRef:
    return LocalEvidenceRef(
        schema_version="local_evidence_ref_v1",
        evidence_id="child-a",
        child_id="child-a",
        parent_id="parent-a",
        generation_id="gen-a",
        policy_id="a" * 64,
        subject="math",
        source_file="notes.md",
        source_relpath="math/notes.md",
        section_id="section-a",
        section_title="Limits",
        section_path=("Limits",),
        page_start=1,
        page_end=1,
        start_char=10,
        end_char=20,
        child_start_in_parent=0,
        child_end_in_parent=10,
        child_chars=10,
        final_rank=1,
        vector_rank=1,
        bm25_rank=None,
        rrf_score=0.1,
        rerank_score=0.9,
        content_preview="0123456789",
    )


def test_graph_adapter_keeps_parent_body_out_and_preserves_child_identity() -> None:
    candidate = local_refs_to_evidence_candidates(
        (_ref(),),
        provider="parent_child_hybrid",
        role="core_evidence",
        purpose="answer",
        branch_status="ok",
        primary_revision=1,
        primary_config_fingerprint="e" * 64,
    )[0]

    assert candidate.evidence_id == "child-a"
    assert candidate.content_preview == "0123456789"
    assert candidate.metadata["parent_id"] == "parent-a"
    assert candidate.metadata["primary_revision"] == 1
    assert candidate.metadata["primary_config_fingerprint"] == "e" * 64
    assert "content" not in candidate.metadata

    item = EvidenceJudgeItem(
        evidence_id="child-a",
        keep=True,
        final_quality="high",
        relevance="high",
        authority="high",
        usefulness="high",
        risk="low",
        evidence_score=0.9,
        score_reason="Directly relevant local evidence.",
        evidence_type="local_textbook_chunk",
        use_case="core_evidence",
        coverage_contribution="Defines the requested limit concept.",
        reason="Keep for the final explanation.",
    )
    decision = judge_items_to_keep_decisions((item,))[0]
    assert decision.evidence_id == "child-a"
    assert decision.keep is True
