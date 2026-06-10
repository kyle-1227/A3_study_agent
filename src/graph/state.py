"""TutorState: the shared state object that flows through all nodes in the LangGraph, acting as the single source of truth for the system."""

from __future__ import annotations

from typing import Annotated, Literal

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# Sentinel value: returning this from a node signals "clear all context"
CONTEXT_CLEAR: list[dict] = [{"__clear__": True}]


def context_reducer(existing: list[dict], update: list[dict]) -> list[dict]:
    """Merge context lists from fan-out branches.

    Returning CONTEXT_CLEAR resets context to empty (used on retry path).
    Normal updates are appended (same as operator.add).
    """
    if update and update[0].get("__clear__"):
        return []
    return existing + update


class TutorState(TypedDict):
    messages: Annotated[list, add_messages]                             # Chat history
    request_id: str                                                      # Per-request trace identifier
    session_id: str                                                      # Session identifier for trace grouping
    thread_id: str                                                       # LangGraph thread identifier
    intent: Literal["academic", "planning", "emotional", "unknown"]    # User intent
    subject: str                                                        # The topic being discussed
    subject_candidates: list[str]                                       # Ordered available-subject candidates
    keypoints: list[str]                                                # Key points
    requested_resource_type: str                                        # Requested resource type, e.g. mindmap
    needs_mindmap: bool                                                 # Route to mindmap collaboration chain when true
    mindmap_outline: str                                                # Planner-produced knowledge structure blueprint
    mindmap_tree: dict                                                  # Reviewed JSON tree draft
    mindmap_artifact: dict                                              # Generated mindmap tree and artifact metadata
    mindmap_review_verdict: str                                         # "approve" / "reject"
    mindmap_review_reason: str                                          # Reviewer reasoning
    mindmap_revision_notes: str                                         # Feedback for mindmap_agent regeneration
    mindmap_round: int                                                  # Mindmap generation/review round
    exercise_outline: str                                               # Planner-produced exercise blueprint
    exercise_items: list[dict]                                          # Reviewed exercise item drafts
    exercise_artifact: dict                                             # Generated exercise metadata/content
    exercise_review_verdict: str                                        # "approve" / "reject"
    exercise_review_reason: str                                         # Exercise reviewer reasoning
    exercise_revision_notes: str                                        # Feedback for exercise_agent regeneration
    exercise_round: int                                                 # Exercise generation/review round
    review_doc_outline: str                                             # Planner-produced review document blueprint
    review_doc_markdown: str                                            # Reviewed Markdown review document draft
    review_doc_artifact: dict                                           # Generated review document content and artifact metadata
    review_doc_review_verdict: str                                      # "approve" / "reject"
    review_doc_review_reason: str                                       # Review document reviewer reasoning
    review_doc_revision_notes: str                                      # Feedback for review_doc_agent regeneration
    review_doc_round: int                                               # Review document generation/review round
    context: Annotated[list[dict], context_reducer]                    # Merged retrieval context (fan-in)
    search_results: list[dict]                                          # Planner search results
    retrieval_plan: list[dict]                                          # Multi-subject retrieval plan
    primary_subject: str                                                # Main subject of the user goal
    learning_goal: str                                                  # Normalized learning goal
    subject_relation_summary: str                                       # How subjects relate to the goal
    search_rag_query: str                                               # Initial rewritten query for local course retrieval
    search_web_query: str                                               # Initial rewritten query for web search
    expanded_keypoints: list[str]                                       # Query rewriter expanded concrete keypoints
    search_query_rewrite_reason: str                                    # Query rewriter rationale
    search_query_rewrite_error: str                                     # Query rewriter failure reason, if any
    search_query_rewrite_raw_preview: str                               # Truncated raw query-rewriter output for diagnostics
    web_supplement_decisions: list[dict]                                # Dynamic web supplement coverage decisions
    web_supplement_results: list[dict]                                  # Dynamic web supplement result docs
    coverage_decision_summary: str                                      # Summary of coverage risk and web supplement decision
    retrieval_branch_mode: str                                          # multi_subject_plan / single_subject_synthetic
    web_supplement_provider: str                                        # Web supplement provider, e.g. tavily
    web_supplement_failed: bool                                         # Dynamic web supplement was needed but produced no usable result
    web_supplement_failure_reason: str                                  # Reason for failed dynamic web supplement
    web_supplement_status_by_subject: dict                              # Per-subject dynamic web supplement status
    web_supplement_success_subjects: list[str]                          # Subjects with usable web supplement
    web_supplement_failed_subjects: list[str]                           # Subjects that needed but failed web supplement
    web_supplement_partial_failed: bool                                 # At least one subject failed while another succeeded
    web_evidence_count: int                                             # Approved source_type=web evidence count
    web_supplement_count: int                                           # Legacy-compatible alias for web_evidence_count
    web_judge_provider: str                                             # Search Result Judge provider
    web_judge_model: str                                                # Search Result Judge model
    web_judge_failed_subjects: list[str]                                 # Subjects where Search Result Judge failed
    web_judge_rejected_all_subjects: list[str]                           # Subjects where Judge worked but rejected every result
    evidence_candidates: list[dict]                                      # Dual-source local/web EvidenceCandidate snapshots
    evidence_judge_output: dict                                          # Raw structured Evidence Judge output
    evidence_judge_rounds: int                                           # Evidence Judge rounds executed
    evidence_judge_state: str                                            # sufficient / partially_sufficient / insufficient
    evidence_coverage_gaps: list[dict]                                   # Coverage gaps reserved for future search optimization
    search_refinement_needed: bool                                       # Evidence Judge requested more search
    search_refinement_deferred: bool                                     # Follow-up search is intentionally deferred
    search_refinement_deferred_reason: str                               # Why refinement was not executed
    proposed_followup_search_queries: list[dict]                         # Reserved future search queries from coverage gaps
    search_optimization_reserved: bool                                   # Search optimization hook is reserved
    search_optimization_status: str                                      # reserved_not_implemented / disabled
    dual_source_mode: bool                                               # rag_retrieve used dual_source_evidence mode
    evidence_judge_failed: bool                                          # Evidence Judge failed and no evidence was admitted
    degraded_generation: bool                                            # Generation proceeds without approved evidence
    degraded_reason: str                                                 # Reason for degraded generation
    plan: str                                                           # Generated plans
    retry_count: int                                                    # Hallucination retry counter
    hallucination_detected: bool                                        # Hallucination flag
    rewritten_query: str                                                # Rewritten query on retry
    hallucination_reason: str                                           # Reason from hallucination eval
    emotional_intel: str                                                # Emotional state summary (gather_intel)
    resource_intel: str                                                 # Resource intel summary (gather_intel)
    intel_summary: str                                                  # Combined intel for adversarial planner
    # ── Adversarial planning (flattened SubGraph — AC-01) ────────────
    draft: str                                                          # Current plan draft text
    academic_verdict: str                                               # "approve" / "reject"
    academic_reason: str                                                # Reviewer reasoning
    emotional_verdict: str                                              # "approve" / "reject"
    emotional_reason: str                                               # Reviewer reasoning
    adv_round: int                                                      # Current review round
    consensus: bool                                                     # Both reviewers approved?
    revision_notes: str                                                 # Combined feedback for drafter
    # ── HIL feedback loop ────────────────────────────────────────────
    hil_action: str                                                     # "confirm" or "feedback" — set by plan_output
    hil_feedback: str                                                   # User's raw feedback text — set by plan_output
    hil_summary: str                                                    # Compressed summary of all prior feedback rounds (overwritten, not appended)
    feedback_route: str                                                 # "tweak" or "rewrite" — set by feedback_router
