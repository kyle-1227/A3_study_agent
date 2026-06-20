"""LearningState: the shared state object that flows through all nodes in the LangGraph, acting as the single source of truth for the system."""

from __future__ import annotations

from typing import Annotated, Literal

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# Sentinel value: returning this from a node signals "clear all context"
CONTEXT_CLEAR: list[dict] = [{"__clear__": True}]

# Sentinel value: returning this to evidence memory reducers clears the list.
MEMORY_CLEAR: list[dict] = [{"__memory_clear__": True}]

# Sentinel value: returning this to resource result reducers clears the list.
RESOURCE_RESULTS_CLEAR: list[dict] = [{"__resource_results_clear__": True}]

# Evidence memory reducer
EVIDENCE_MEMORY_MAX_ENTRIES = 20


def evidence_memory_reducer(existing: list[dict], update: list[dict]) -> list[dict]:
    """Idempotent, bounded, deduplicated evidence memory reducer.

    Rules:
    - Dedupe by ``memory_id`` (latest wins).
    - Retain only the most recent ``EVIDENCE_MEMORY_MAX_ENTRIES`` entries.
    - Never store raw docs or full old context.
    """
    if update and update[0].get("__memory_clear__"):
        return []

    merged: dict[str, dict] = {}
    for entry in existing:
        mid = entry.get("memory_id", "")
        if mid:
            merged[mid] = entry
    for entry in update:
        mid = entry.get("memory_id", "")
        if mid:
            merged[mid] = entry
    sorted_entries = sorted(
        merged.values(),
        key=lambda e: e.get("created_at", ""),
        reverse=True,
    )
    return sorted_entries[:EVIDENCE_MEMORY_MAX_ENTRIES]


# Current-turn transient state reset
def initial_request_reset_transient_state() -> dict:
    """Return reset values for current-turn-only fields.

    Called at the start of every new /stream user request so that
    routing, query, retrieval, evidence, and resource artifacts from
    a previous turn in the same thread do not contaminate the new request.

    Long-term fields (messages, conversation_summary,
    evidence_summary_memory, evidence_gap_memory, profile) are **not**
    reset by this function.
    """
    return {
        # routing
        "intent": "unknown",
        "subject": "",
        "subject_candidates": [],
        "keypoints": [],
        "requested_resource_type": "",
        "requested_resource_types": [],
        "needs_mindmap": False,
        # memory use policy for current query rewrite
        "memory_use_policy": "unset",
        "memory_use_reason": "",
        "memory_use_user_choice": "",
        "memory_confirmation_required": False,
        "memory_confirmation_question": "",
        "selected_evidence_memory_summaries": [],
        "episodic_memory_results": [],
        "semantic_memory_results": [],
        # query / retrieval plan
        "local_retrieval_query": "",
        "web_research_seed_query": "",
        "expanded_keypoints": [],
        "search_query_rewrite_reason": "",
        "search_query_rewrite_error": "",
        "search_query_rewrite_raw_preview": "",
        "retrieval_plan": [],
        "learning_goal": "",
        "primary_subject": "",
        "subject_relation_summary": "",
        "rewritten_query": "",
        "retry_count": 0,
        "hallucination_detected": False,
        "hallucination_reason": "",
        # retrieval / web research
        "retrieval_branch_mode": "",
        "web_evidence_count": 0,
        "web_research_debug": {},
        "web_research_outcome": "",
        # evidence
        "local_evidence_candidates": [],
        "web_evidence_candidates": [],
        "local_evidence_originals": {},
        "web_evidence_originals": {},
        "evidence_candidates": [],
        "evidence_judge_output": {},
        "evidence_judge_debug": {},
        "evidence_judge_rounds": 0,
        "evidence_judge_state": "",
        "evidence_coverage_gaps": [],
        "search_refinement_needed": False,
        "search_refinement_deferred": False,
        "search_refinement_deferred_reason": "",
        "proposed_followup_search_queries": [],
        "search_optimization_reserved": True,
        "search_optimization_status": "reserved_not_implemented",
        "dual_source_mode": False,
        "evidence_judge_failed": False,
        "degraded_generation": False,
        "degraded_reason": "",
        "evidence_controlled_stop": False,
        "evidence_controlled_stop_reason": "",
        # context
        "context": CONTEXT_CLEAR,
        # resource artifacts
        "mindmap_outline": "",
        "mindmap_tree": {},
        "mindmap_artifact": {},
        "mindmap_review_verdict": "",
        "mindmap_review_reason": "",
        "mindmap_revision_notes": "",
        "mindmap_round": 0,
        "exercise_outline": "",
        "exercise_items": [],
        "exercise_artifact": {},
        "exercise_review_verdict": "",
        "exercise_review_reason": "",
        "exercise_revision_notes": "",
        "exercise_round": 0,
        "review_doc_outline": "",
        "review_doc_markdown": "",
        "review_doc_markdowns": [],
        "review_doc_artifact": {},
        "review_doc_artifacts": [],
        "review_doc_review_verdict": "",
        "review_doc_review_reason": "",
        "review_doc_revision_notes": "",
        "review_doc_round": 0,
        "study_plan_emotional_intel": "",
        "study_plan_emotional_profile": {},
        "study_plan_outline": "",
        "study_plan_artifact": {},
        "study_plan_markdown": "",
        "study_plan_round": 0,
        "study_plan_academic_verdict": "",
        "study_plan_academic_reason": "",
        "study_plan_emotional_verdict": "",
        "study_plan_emotional_reason": "",
        "study_plan_consensus": False,
        "study_plan_revision_notes": "",
        "study_plan_document_artifact": {},
        "plan": "",
        "resource_generation_plan": {},
        "resource_branch_results": RESOURCE_RESULTS_CLEAR,
        "resource_bundle_artifact": {},
        "resource_generation_debug": {},
        "resource_generation_status": "",
    }


def context_reducer(existing: list[dict], update: list[dict]) -> list[dict]:
    """Merge context lists from fan-out branches.

    Returning CONTEXT_CLEAR resets context to empty (used on retry path).
    Normal updates are appended (same as operator.add).
    """
    if update and update[0].get("__clear__"):
        return []
    return existing + update


def resource_branch_results_reducer(existing: list[dict], update: list[dict]) -> list[dict]:
    """Merge resource worker results from dynamic fan-out branches."""
    if not update or update[0].get("__resource_results_clear__"):
        return []

    merged: dict[str, dict] = {}
    for entry in existing or []:
        resource_type = str((entry or {}).get("resource_type") or "").strip()
        if resource_type:
            merged[resource_type] = entry
    for entry in update or []:
        resource_type = str((entry or {}).get("resource_type") or "").strip()
        if resource_type:
            merged[resource_type] = entry

    order = ["mindmap", "quiz", "review_doc", "study_plan"]
    return sorted(
        merged.values(),
        key=lambda item: order.index(item.get("resource_type")) if item.get("resource_type") in order else len(order),
    )


class LearningState(TypedDict):
    messages: Annotated[list, add_messages]                             # Chat history
    conversation_summary: str                                            # Compact multi-turn conversation summary
    evidence_summary_memory: Annotated[list[dict], evidence_memory_reducer]  # Bounded evidence memory
    evidence_gap_memory: Annotated[list[dict], evidence_memory_reducer]     # Bounded gap memory
    request_id: str                                                      # Per-request trace identifier
    session_id: str                                                      # Session identifier for trace grouping
    thread_id: str                                                       # LangGraph thread identifier
    intent: Literal["academic", "emotional", "unknown"]    # User intent
    subject: str                                                        # The topic being discussed
    subject_candidates: list[str]                                       # Ordered available-subject candidates
    keypoints: list[str]                                                # Key points
    requested_resource_type: str                                        # Requested resource type, e.g. mindmap
    requested_resource_types: list[str]                                 # Ordered resource types requested for parallel generation
    needs_mindmap: bool                                                 # Route to mindmap collaboration chain when true
    memory_use_policy: Literal["use", "ignore", "ask_user", "unset"]    # Whether query rewrite may use selected memory
    memory_use_reason: str                                              # Reason for the memory use policy
    memory_use_user_choice: Literal["use", "ignore", ""]               # User choice after HIL confirmation
    memory_confirmation_required: bool                                  # Whether memory use confirmation was requested
    memory_confirmation_question: str                                   # HIL question shown to the user
    selected_evidence_memory_summaries: list[dict]                      # Compact evidence summaries selected for current query
    episodic_memory_results: list[dict]                                 # Top-K episodic memories with scores (from episodic_memory_retriever)
    semantic_memory_results: list[dict]                                 # Top-K semantic memories with scores (from episodic_memory_retriever)
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
    review_doc_markdowns: list[dict]                                    # Per-subject Markdown review documents
    review_doc_artifact: dict                                           # Generated review document content and artifact metadata
    review_doc_artifacts: list[dict]                                    # Per-subject review document artifact metadata
    review_doc_review_verdict: str                                      # "approve" / "reject"
    review_doc_review_reason: str                                       # Review document reviewer reasoning
    review_doc_revision_notes: str                                      # Feedback for review_doc_agent regeneration
    review_doc_round: int                                               # Review document generation/review round
    study_plan_emotional_intel: str                                     # Study-plan learner workload/emotional summary
    study_plan_emotional_profile: dict                                  # Structured emotional/workload profile
    study_plan_outline: str                                             # Planner-produced study-plan outline
    study_plan_artifact: dict                                           # Structured personalized study-plan artifact
    study_plan_markdown: str                                            # Rendered study-plan Markdown
    study_plan_round: int                                               # Study-plan generation/review round
    study_plan_academic_verdict: str                                    # "approve" / "reject"
    study_plan_academic_reason: str                                     # Academic reviewer reasoning
    study_plan_emotional_verdict: str                                   # "approve" / "reject"
    study_plan_emotional_reason: str                                    # Emotional/workload reviewer reasoning
    study_plan_consensus: bool                                          # Both study-plan reviewers approved
    study_plan_revision_notes: str                                      # Feedback for study_plan_agent regeneration
    study_plan_document_artifact: dict                                  # Markdown/DOCX artifact metadata
    context: Annotated[list[dict], context_reducer]                    # Merged retrieval context (fan-in)
    retrieval_plan: list[dict]                                          # Multi-subject retrieval plan
    primary_subject: str                                                # Main subject of the user goal
    learning_goal: str                                                  # Normalized learning goal
    subject_relation_summary: str                                       # How subjects relate to the goal
    local_retrieval_query: str                                               # Initial rewritten query for local course retrieval
    web_research_seed_query: str                                               # Initial rewritten query for Web Research
    expanded_keypoints: list[str]                                       # Query rewriter expanded concrete keypoints
    search_query_rewrite_reason: str                                    # Query rewriter rationale
    search_query_rewrite_error: str                                     # Query rewriter failure reason, if any
    search_query_rewrite_raw_preview: str                               # Truncated raw query-rewriter output for diagnostics
    retrieval_branch_mode: str                                          # multi_subject_plan / single_subject_synthetic
    web_evidence_count: int                                             # Approved source_type=web evidence count
    web_research_debug: dict                                             # Web Research V2 execution status/debug summary
    web_research_outcome: Literal["", "success", "failed", "skipped"]  # Web Research V2 outcome
    local_evidence_candidates: list[dict]                                # Local RAG EvidenceCandidate snapshots from rag_retrieve
    web_evidence_candidates: list[dict]                                  # Web EvidenceCandidate snapshots from Web Research V2
    local_evidence_originals: dict                                       # Original local RAG docs keyed by evidence_id
    web_evidence_originals: dict                                         # Original Tavily results keyed by evidence_id
    evidence_candidates: list[dict]                                      # Dual-source local/web EvidenceCandidate snapshots
    evidence_judge_output: dict                                          # Raw structured Evidence Judge output
    evidence_judge_debug: dict                                           # Evidence Judge execution status/debug summary
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
    evidence_controlled_stop: bool                                       # Controlled stop due to insufficient evidence
    evidence_controlled_stop_reason: str                                 # Reason for controlled stop
    plan: str                                                           # Generated plans
    resource_generation_plan: dict                                      # Parallel resource generation plan
    resource_branch_results: Annotated[list[dict], resource_branch_results_reducer]  # Parallel resource worker results
    resource_bundle_artifact: dict                                      # Aggregated multi-resource artifact metadata
    resource_generation_debug: dict                                     # Resource generation execution status/debug summary
    resource_generation_status: str                                     # success / partial_success / failed / skipped
    retry_count: int                                                    # Hallucination retry counter
    hallucination_detected: bool                                        # Hallucination flag
    rewritten_query: str                                                # Rewritten query on retry
    hallucination_reason: str                                           # Reason from hallucination eval
