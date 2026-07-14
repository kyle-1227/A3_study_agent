"""LearningState: the shared state object that flows through all nodes in the LangGraph, acting as the single source of truth for the system."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage, ToolMessage, message_to_dict
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.assessment.attempt_journal import assessment_attempt_journal_reducer
from src.assessment.checkpoint import assessment_checkpoint_resources_reducer
from src.context_engineering.workspace import (
    TASK_WORKSPACE_CLEAR as _TASK_WORKSPACE_CLEAR,
    merge_task_workspace,
)
from src.context_engineering.input_manifest import (
    merge_llm_input_manifest_history,
    merge_thread_context_ledger,
)
from src.context_engineering.influence import (
    CONTEXT_INFLUENCE_LEDGER_CLEAR as _CONTEXT_INFLUENCE_LEDGER_CLEAR,
    merge_context_influence_ledger,
)
from src.context_engineering.session_memory import (
    ContextInjectionRecordV1,
    ContextMemoryCompactionMutationV1,
    SessionContextMemoryLedgerV1,
    apply_context_memory_compaction,
    new_session_context_memory_ledger,
    record_context_injection,
)
from src.observability.activity import merge_activity_timeline
from src.observability.context_usage_report import (
    merge_context_usage_report_history,
)


# Sentinel value: returning this from a node signals "clear all context"
CONTEXT_CLEAR: list[dict] = [{"__clear__": True}]

# Sentinel value: returning this to evidence memory reducers clears the list.
MEMORY_CLEAR: list[dict] = [{"__memory_clear__": True}]

# Sentinel value: returning this to resource result reducers clears the list.
RESOURCE_RESULTS_CLEAR: list[dict] = [{"__resource_results_clear__": True}]

# Sentinel values for reducer-owned persistent context.
TASK_WORKSPACE_CLEAR: dict = _TASK_WORKSPACE_CLEAR
DICT_CLEAR: dict = {"__dict_clear__": True}
GENERATED_ARTIFACTS_CLEAR: list[dict] = [{"__generated_artifacts_clear__": True}]
WORKSPACE_EVENTS_CLEAR: list[dict] = [{"__workspace_events_clear__": True}]
LLM_INPUT_MANIFESTS_CLEAR: list[dict] = [{"__llm_input_manifests_clear__": True}]
CONTEXT_INFLUENCE_LEDGER_CLEAR: dict = _CONTEXT_INFLUENCE_LEDGER_CLEAR
ACTIVITY_TIMELINE_CLEAR: list[dict] = [{"__activity_timeline_clear__": True}]
CONTEXT_USAGE_REPORTS_CLEAR: list[dict] = [{"__context_usage_reports_clear__": True}]
SESSION_CONTEXT_MEMORY_LEDGER_CLEAR: dict = {
    "__session_context_memory_ledger_clear__": True
}

# Evidence memory reducer
EVIDENCE_MEMORY_MAX_ENTRIES = 20
EVIDENCE_MEMORY_CHAR_LIMIT = 64_000
CONTEXT_WINDOW_HISTORY_LIMIT = 30
CONTEXT_WINDOW_HISTORY_CHAR_LIMIT = 96_000
CONTEXT_WINDOW_EVENT_LIMIT = 120
CONTEXT_WINDOW_EVENT_CHAR_LIMIT = 96_000
GENERATED_ARTIFACT_HISTORY_CHAR_LIMIT = 96_000
THREAD_MESSAGE_HISTORY_LIMIT = 48
THREAD_MESSAGE_HISTORY_CHAR_LIMIT = 128_000


def _bounded_json_history(
    values: list[dict],
    *,
    max_items: int,
    max_chars: int,
    newest_first: bool = False,
) -> list[dict]:
    """Keep the newest JSON-safe entries within deterministic dual bounds."""
    candidates = values[:max_items] if newest_first else values[-max_items:]
    ordered = candidates if newest_first else list(reversed(candidates))
    bounded: list[dict] = []
    total_chars = 2
    for entry in ordered:
        if not isinstance(entry, dict):
            continue
        try:
            item_chars = len(
                json.dumps(
                    entry,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        except (TypeError, ValueError):
            continue
        added_chars = item_chars + (1 if bounded else 0)
        if item_chars > max_chars or total_chars + added_chars > max_chars:
            continue
        bounded.append(entry)
        total_chars += added_chars
    return bounded if newest_first else list(reversed(bounded))


def _message_json_chars(message: BaseMessage) -> int:
    return len(
        json.dumps(
            message_to_dict(message),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def bounded_messages_reducer(existing: list, update: list) -> list:
    """Apply LangGraph message semantics and retain only the recent window."""
    merged = list(add_messages(existing or [], update or []))
    candidates = merged[-THREAD_MESSAGE_HISTORY_LIMIT:]
    bounded_reversed: list[BaseMessage] = []
    total_chars = 2
    for message in reversed(candidates):
        if not isinstance(message, BaseMessage):
            continue
        item_chars = _message_json_chars(message)
        if item_chars > THREAD_MESSAGE_HISTORY_CHAR_LIMIT:
            if not bounded_reversed:
                raise ValueError(
                    "latest thread message exceeds persistent character limit"
                )
            break
        added_chars = item_chars + (1 if bounded_reversed else 0)
        if total_chars + added_chars > THREAD_MESSAGE_HISTORY_CHAR_LIMIT:
            break
        bounded_reversed.append(message)
        total_chars += added_chars
    bounded = list(reversed(bounded_reversed))
    while bounded and isinstance(bounded[0], ToolMessage):
        bounded.pop(0)
    return bounded


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
    return _bounded_json_history(
        sorted_entries,
        max_items=EVIDENCE_MEMORY_MAX_ENTRIES,
        max_chars=EVIDENCE_MEMORY_CHAR_LIMIT,
        newest_first=True,
    )


def bounded_context_window_reducer(
    existing: list[dict],
    update: list[dict],
) -> list[dict]:
    """Append bounded context-window telemetry without unbounded growth."""
    values = list(existing or []) + list(update or [])
    return _bounded_json_history(
        values,
        max_items=CONTEXT_WINDOW_HISTORY_LIMIT,
        max_chars=CONTEXT_WINDOW_HISTORY_CHAR_LIMIT,
    )


def bounded_context_event_reducer(
    existing: list[dict],
    update: list[dict],
) -> list[dict]:
    """Append bounded request-local CE events."""
    if update and update[0].get("__workspace_events_clear__"):
        return []
    values = list(existing or []) + list(update or [])
    return _bounded_json_history(
        values,
        max_items=CONTEXT_WINDOW_EVENT_LIMIT,
        max_chars=CONTEXT_WINDOW_EVENT_CHAR_LIMIT,
    )


def llm_input_manifests_reducer(
    existing: list[dict],
    update: list[dict],
) -> list[dict]:
    """Merge bounded LLM input manifests idempotently by manifest_id."""
    if update and update[0].get("__llm_input_manifests_clear__"):
        return []
    return merge_llm_input_manifest_history(existing, update)


def context_usage_reports_reducer(
    existing: list[dict],
    update: list[dict],
) -> list[dict]:
    """Merge versioned context usage reports by stable report_id."""
    if update and update[0].get("__context_usage_reports_clear__"):
        return []
    return merge_context_usage_report_history(existing, update)


def activity_timeline_reducer(
    existing: list[dict],
    update: list[dict],
) -> list[dict]:
    """Merge bounded thread activity events idempotently by activity_id."""
    if update and update[0].get("__activity_timeline_clear__"):
        return []
    return merge_activity_timeline(existing, update)


def merge_dict_reducer(existing: dict, update: dict) -> dict:
    """Reducer-safe shallow dict merge for keyed context-window state."""
    if isinstance(update, dict) and update.get("__dict_clear__") is True:
        return {}
    merged = dict(existing or {})
    merged.update(update or {})
    return merged


def thread_context_ledger_reducer(existing: dict, update: dict) -> dict:
    """Merge sanitized thread-level context ledger updates."""
    if isinstance(update, dict) and update.get("__dict_clear__") is True:
        return {}
    return merge_thread_context_ledger(existing, update)


def session_context_memory_ledger_reducer(existing: dict, update: dict) -> dict:
    """Apply strict, idempotent mutations to durable session memory accounting."""

    if not isinstance(update, dict):
        raise ValueError("session context memory ledger update must be a dict")
    if update.get("__session_context_memory_ledger_clear__") is True:
        return {}
    operation = update.get("operation")
    if operation == "record_dispatch":
        record = ContextInjectionRecordV1.model_validate(update.get("record"))
        ledger = (
            SessionContextMemoryLedgerV1.model_validate(existing)
            if existing
            else new_session_context_memory_ledger(record.thread_id)
        )
        return record_context_injection(ledger, record).model_dump(mode="json")
    if operation == "apply_compaction":
        if not existing:
            raise ValueError("compaction requires an existing session memory ledger")
        ledger = SessionContextMemoryLedgerV1.model_validate(existing)
        mutation = ContextMemoryCompactionMutationV1.model_validate(update)
        return apply_context_memory_compaction(
            ledger,
            boundary_id=mutation.boundary_id,
            retained_logical_item_ids=mutation.retained_logical_item_ids,
            compacted_at=mutation.compacted_at,
            before_tokens=mutation.before_tokens,
            after_tokens=mutation.after_tokens,
        ).model_dump(mode="json")
    if update.get("schema_version") == 1:
        return SessionContextMemoryLedgerV1.model_validate(update).model_dump(
            mode="json"
        )
    raise ValueError("session context memory ledger operation is invalid")


def latest_dict_reducer(_existing: dict, update: dict) -> dict:
    """Replace with the latest dict value."""
    if isinstance(update, dict) and update.get("__dict_clear__") is True:
        return {}
    return dict(update or {})


def latest_string_reducer(_existing: str, update: str) -> str:
    """Replace with the latest string value."""
    return str(update or "")


def generated_artifacts_reducer(
    existing: list[dict],
    update: list[dict],
) -> list[dict]:
    """Merge bounded generated artifact summaries idempotently by artifact_id."""
    if update and update[0].get("__generated_artifacts_clear__"):
        return []
    merged: dict[str, dict] = {}
    for entry in [*(existing or []), *(update or [])]:
        if not isinstance(entry, dict):
            continue
        artifact_id = str(entry.get("artifact_id") or "").strip()
        if not artifact_id:
            continue
        merged[artifact_id] = entry
    sorted_entries = sorted(
        merged.values(),
        key=lambda item: (
            str(item.get("created_at") or ""),
            str(item.get("artifact_id") or ""),
        ),
        reverse=True,
    )
    return _bounded_json_history(
        sorted_entries,
        max_items=CONTEXT_WINDOW_HISTORY_LIMIT,
        max_chars=GENERATED_ARTIFACT_HISTORY_CHAR_LIMIT,
        newest_first=True,
    )


def task_workspace_reducer(existing: dict, update: dict) -> dict:
    """Merge versioned task workspace updates without unbounded growth."""
    return merge_task_workspace(existing, update)


def context_influence_ledger_reducer(existing: dict, update: dict) -> dict:
    """Merge the bounded, idempotent cross-node influence ledger."""
    return merge_context_influence_ledger(existing, update)


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
        "user_id": "",
        "intent": "unknown",
        "response_mode": "",
        "qa_scope": "",
        "requires_live_verification": False,
        "final_response_type": "",
        "runtime_capability_metadata": {},
        "subject": "",
        "subject_candidates": [],
        "keypoints": [],
        "requested_resource_type": "",
        "requested_resource_types": [],
        "needs_mindmap": False,
        "workspace_continuation": {},
        "workspace_continuation_applied": False,
        "workspace_continuation_reason": "",
        "learner_profile_inferred": {},
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
        "graded_evidence": [],
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
        "parent_child_retrieval_result": {},
        "parent_child_local_refs": [],
        "parent_child_generation_id": "",
        "parent_child_retrieval_fingerprint": "",
        "parent_child_hydration": {},
        # resource-aware evidence orchestration candidate
        "evidence_orchestration_fingerprint": "",
        "evidence_orchestration_status": "",
        "evidence_requested_resource_types": [],
        "evidence_requested_subjects": [],
        "evidence_requirements": [],
        "evidence_current_round": 0,
        "evidence_current_tasks": [],
        "evidence_all_tasks": [],
        "evidence_retrieval_signatures": [],
        "evidence_candidate_records": [],
        "evidence_ledger": [],
        "evidence_coverage": {},
        "evidence_previous_coverage": {},
        "evidence_source_outcomes": [],
        "evidence_parent_child_rounds": [],
        "evidence_repair_plans": [],
        "evidence_consecutive_no_progress_rounds": 0,
        "evidence_orchestration_route": "",
        "evidence_terminal_status": "",
        "evidence_terminal_reason_code": "",
        "evidence_hydration_count": 0,
        "evidence_local_batch": {},
        "evidence_web_batch": {},
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
        "exercise_resource_v3": {},
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
        "code_practice_outline": "",
        "code_practice_markdown": "",
        "code_practice_artifact": {},
        "code_practice_review_verdict": "",
        "code_practice_review_reason": "",
        "code_practice_revision_notes": "",
        "code_practice_round": 0,
        "video_script_outline": "",
        "video_script_markdown": "",
        "video_script_srt": "",
        "video_script_artifact": {},
        "video_script_review_verdict": "",
        "video_script_review_reason": "",
        "video_script_revision_notes": "",
        "video_script_round": 0,
        "video_animation_spec": {},
        "video_animation_html": "",
        "video_animation_artifact": {},
        "video_animation_review_verdict": "",
        "video_animation_review_reason": "",
        "video_animation_revision_notes": "",
        "video_animation_round": 0,
        "video_animation_render_log": "",
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
        "resource_final_v3": {},
        "resource_generation_debug": {},
        "resource_generation_status": "",
        "resource_evidence_readiness": [],
        "resource_evidence_assignments": [],
        "ready_resource_types": [],
        "blocked_resource_types": [],
        "learning_path": {},
        "curriculum_context": "",
        "learner_path_planner_output": {},
        "quiz_results": [],
        "adaptive_tasks": [],
        "recommendations": [],
        "recommendation_resource_context": [],
        "resource_recommendation_output": {},
        "review_schedule": [],
        # run control
        "schema_version": "run_control_v1",
        "run_status": "running",
        "stop_requested": False,
        "stop_reason": "",
        "stop_requested_at": "",
        "current_node": "",
        "last_completed_node": "",
        "resume_available": False,
        "stopped_at": "",
        "pending_interrupt_type": "",
        "profile_completion_request": {},
        # context window telemetry
        "context_usage": {},
        "llm_input_manifest": {},
        "request_context_window": {
            "current_request_id": "",
            "current_node": "",
            "last_event_count": 0,
        },
    }


def context_reducer(existing: list[dict], update: list[dict]) -> list[dict]:
    """Merge context lists from fan-out branches.

    Returning CONTEXT_CLEAR resets context to empty (used on retry path).
    Normal updates are appended (same as operator.add).
    """
    if update and update[0].get("__clear__"):
        return update[1:]
    return existing + update


def resource_branch_results_reducer(
    existing: list[dict], update: list[dict]
) -> list[dict]:
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

    order = [
        "review_doc",
        "mindmap",
        "quiz",
        "code_practice",
        "video_script",
        "video_animation",
        "study_plan",
    ]

    def sort_key(item: dict) -> int:
        resource_type = str(item.get("resource_type") or "")
        return order.index(resource_type) if resource_type in order else len(order)

    return sorted(
        merged.values(),
        key=sort_key,
    )


class LearningState(TypedDict):
    messages: Annotated[list, bounded_messages_reducer]  # Recent chat history
    conversation_summary: Annotated[
        str, latest_string_reducer
    ]  # Compact multi-turn conversation summary
    conversation_summary_v2: Annotated[
        dict, latest_dict_reducer
    ]  # Validated semantic summary bound to the active compact boundary
    compact_boundary: Annotated[
        dict, latest_dict_reducer
    ]  # Content-free identities replaced in provider-bound model views
    compaction_result: Annotated[
        dict, latest_dict_reducer
    ]  # Latest committed full-compaction measurement and recovery descriptor
    last_provider_dispatch: Annotated[
        dict, latest_dict_reducer
    ]  # Latest actual trigger-eligible provider-bound input measurement
    evidence_summary_memory: Annotated[
        list[dict], evidence_memory_reducer
    ]  # Bounded evidence memory
    evidence_gap_memory: Annotated[
        list[dict], evidence_memory_reducer
    ]  # Bounded gap memory
    task_workspace: Annotated[
        dict, task_workspace_reducer
    ]  # Versioned durable task-level workspace
    workspace_events: Annotated[
        list[dict], bounded_context_event_reducer
    ]  # Bounded workspace update/context events
    request_id: str  # Per-request trace identifier
    session_id: str  # Session identifier for trace grouping
    thread_id: str  # LangGraph thread identifier
    user_id: str  # Explicit authenticated learner id; never derived from thread_id
    graph_version: str  # Deterministic compiled graph manifest version
    schema_version: str  # Run-control state schema version
    run_status: str  # running / stopping / stopped / completed / error
    stop_requested: bool  # Whether user requested safe stop
    stop_reason: str  # User-visible safe-stop reason
    stop_requested_at: str  # UTC timestamp for stop request
    current_node: str  # Current LangGraph node, for status UI
    last_completed_node: str  # Last completed LangGraph node, for status UI
    resume_available: bool  # True only when a real checkpoint interrupt can continue
    stopped_at: str  # UTC timestamp for saved stop checkpoint
    pending_interrupt_type: str  # user_stop / plan_review / memory_confirmation
    profile_completion_request: dict  # Profile-completion interrupt payload
    profile_summary: str  # Compact learner profile summary for CE profile source
    learner_profile_summary: str  # User-supplied compact learner profile summary
    learner_profile: dict  # User-supplied learner profile facts
    learner_profile_inferred: dict  # Current-request inferred learner facts
    context_usage: dict  # Most recent LLM context window usage
    context_usage_history: Annotated[
        list[dict], bounded_context_window_reducer
    ]  # Cross-request bounded usage history
    context_usage_report: Annotated[
        dict, latest_dict_reducer
    ]  # Most recent reconciled provider-input usage report
    context_usage_reports: Annotated[
        list[dict], context_usage_reports_reducer
    ]  # Bounded reconciled usage report history
    activity_timeline: Annotated[
        list[dict], activity_timeline_reducer
    ]  # Thread-level normalized activity timeline
    llm_input_manifest: Annotated[
        dict, latest_dict_reducer
    ]  # Most recent provider-bound LLM input manifest
    llm_input_manifests: Annotated[
        list[dict], llm_input_manifests_reducer
    ]  # Cross-request bounded manifest history
    thread_context_ledger: Annotated[
        dict, thread_context_ledger_reducer
    ]  # Thread-level context source ledger
    session_context_memory_ledger: Annotated[
        dict, session_context_memory_ledger_reducer
    ]  # Durable CE items actually dispatched during this thread
    thread_context_window_v3: Annotated[
        dict, latest_dict_reducer
    ]  # Public projection of durable session context memory
    background_context_window: Annotated[
        dict, latest_dict_reducer
    ]  # Codex-like thread-level background context window
    context_continuity: Annotated[
        dict, latest_dict_reducer
    ]  # Task-continuity diagnostics for current request
    context_influence_ledger: Annotated[
        dict, context_influence_ledger_reducer
    ]  # Bounded cross-node context influence metadata
    request_context_window: Annotated[
        dict, latest_dict_reducer
    ]  # Current request CE window summary
    context_window_events: Annotated[
        list[dict], bounded_context_event_reducer
    ]  # Current/recent CE event summaries
    last_context_policy_by_node: Annotated[
        dict, merge_dict_reducer
    ]  # Last resolved policy by node
    last_provider_supply_by_node: Annotated[
        dict, merge_dict_reducer
    ]  # Last provider supply by node
    last_context_selection_by_node: Annotated[
        dict, merge_dict_reducer
    ]  # Last apply selection by node
    last_context_applied_by_node: Annotated[
        dict, merge_dict_reducer
    ]  # Last applied context by node
    last_drop_reasons_by_node: Annotated[
        dict, merge_dict_reducer
    ]  # Last drop reason counts by node
    last_resource_subnodes: Annotated[
        list[dict], bounded_context_window_reducer
    ]  # Bounded resource subnode events
    resource_artifacts_by_type: Annotated[
        dict, merge_dict_reducer
    ]  # Artifact summaries by resource type
    last_generated_artifacts: Annotated[
        list[dict], generated_artifacts_reducer
    ]  # Bounded generated artifact summaries
    last_resource_final_payload: Annotated[
        dict, latest_dict_reducer
    ]  # Last sanitized renderable resource_final payload
    assessment_checkpoint_resources: Annotated[
        dict, assessment_checkpoint_resources_reducer
    ]  # Durable server-only quiz cards and private answer keys for this thread
    assessment_attempt_journal: Annotated[
        dict, assessment_attempt_journal_reducer
    ]  # Durable idempotency records containing hashes and public finals only
    last_qa_response: Annotated[
        dict, latest_dict_reducer
    ]  # Last bounded renderable qa_final payload
    intent: Literal["academic", "emotional", "unknown"]  # User intent
    response_mode: Literal["qa", "resource", "emotional", ""]
    qa_scope: Literal["academic", "general", "a3_agent", ""]
    requires_live_verification: bool
    final_response_type: str
    runtime_capability_metadata: dict
    subject: str  # The topic being discussed
    subject_candidates: list[str]  # Ordered available-subject candidates
    keypoints: list[str]  # Key points
    requested_resource_type: str  # Requested resource type, e.g. mindmap
    requested_resource_types: list[
        str
    ]  # Ordered resource types requested for parallel generation
    needs_mindmap: bool  # Route to mindmap collaboration chain when true
    workspace_continuation: dict  # Safe current-turn workspace continuation diagnostics
    workspace_continuation_applied: bool  # True when subject inherited from workspace
    workspace_continuation_reason: str  # Continuation skip/apply diagnostic reason
    memory_use_policy: Literal[
        "use", "ignore", "ask_user", "unset"
    ]  # Whether query rewrite may use selected memory
    memory_use_reason: str  # Reason for the memory use policy
    memory_use_user_choice: Literal[
        "use", "ignore", ""
    ]  # User choice after HIL confirmation
    memory_confirmation_required: bool  # Whether memory use confirmation was requested
    memory_confirmation_question: str  # HIL question shown to the user
    selected_evidence_memory_summaries: list[
        dict
    ]  # Compact evidence summaries selected for current query
    episodic_memory_results: list[
        dict
    ]  # Top-K episodic memories with scores (from episodic_memory_retriever)
    semantic_memory_results: list[
        dict
    ]  # Top-K semantic memories with scores (from episodic_memory_retriever)
    mindmap_outline: str  # Planner-produced knowledge structure blueprint
    mindmap_tree: dict  # Reviewed JSON tree draft
    mindmap_artifact: dict  # Generated mindmap tree and artifact metadata
    mindmap_review_verdict: str  # "approve" / "reject"
    mindmap_review_reason: str  # Reviewer reasoning
    mindmap_revision_notes: str  # Feedback for mindmap_agent regeneration
    mindmap_round: int  # Mindmap generation/review round
    exercise_outline: str  # Planner-produced exercise blueprint
    exercise_items: list[dict]  # Reviewed exercise item drafts
    exercise_artifact: dict  # Generated exercise metadata/content
    exercise_resource_v3: dict  # Public Resource Final V3 quiz projection
    exercise_review_verdict: str  # "approve" / "reject"
    exercise_review_reason: str  # Exercise reviewer reasoning
    exercise_revision_notes: str  # Feedback for exercise_agent regeneration
    exercise_round: int  # Exercise generation/review round
    review_doc_outline: str  # Planner-produced review document blueprint
    review_doc_markdown: str  # Reviewed Markdown review document draft
    review_doc_markdowns: list[dict]  # Per-subject Markdown review documents
    review_doc_artifact: dict  # Generated review document content and artifact metadata
    review_doc_artifacts: list[dict]  # Per-subject review document artifact metadata
    review_doc_review_verdict: str  # "approve" / "reject"
    review_doc_review_reason: str  # Review document reviewer reasoning
    review_doc_revision_notes: str  # Feedback for review_doc_agent regeneration
    review_doc_round: int  # Review document generation/review round
    code_practice_outline: str  # Planner-produced code practice case blueprint
    code_practice_markdown: str  # Reviewed Markdown code practice case draft
    code_practice_artifact: (
        dict  # Generated code practice content and artifact metadata
    )
    code_practice_review_verdict: str  # "approve" / "reject"
    code_practice_review_reason: str  # Code practice reviewer reasoning
    code_practice_revision_notes: str  # Feedback for code_practice_agent regeneration
    code_practice_round: int  # Code practice generation/review round
    video_script_outline: (
        str  # Planner-produced teaching video / animation script blueprint
    )
    video_script_markdown: (
        str  # Reviewed Markdown teaching video / animation script draft
    )
    video_script_srt: str  # Generated subtitle/caption text in SRT format
    video_script_artifact: dict  # Generated video script content and artifact metadata
    video_script_review_verdict: str  # "approve" / "reject"
    video_script_review_reason: str  # Video script reviewer reasoning
    video_script_revision_notes: str  # Feedback for video_script_agent regeneration
    video_script_round: int  # Video script generation/review round
    video_animation_spec: dict  # Structured animation/MP4 generation specification
    video_animation_html: str  # Generated animation HTML source for rendering
    video_animation_artifact: dict  # Generated animation/MP4 artifact metadata
    video_animation_review_verdict: str  # "approve" / "reject"
    video_animation_review_reason: str  # Video animation reviewer reasoning
    video_animation_revision_notes: str  # Feedback for video_animation regeneration
    video_animation_round: int  # Video animation generation/review round
    video_animation_render_log: (
        str  # Renderer log or failure details for animation export
    )
    study_plan_emotional_intel: str  # Study-plan learner workload/emotional summary
    study_plan_emotional_profile: dict  # Structured emotional/workload profile
    study_plan_outline: str  # Planner-produced study-plan outline
    study_plan_artifact: dict  # Structured personalized study-plan artifact
    study_plan_markdown: str  # Rendered study-plan Markdown
    study_plan_round: int  # Study-plan generation/review round
    study_plan_academic_verdict: str  # "approve" / "reject"
    study_plan_academic_reason: str  # Academic reviewer reasoning
    study_plan_emotional_verdict: str  # "approve" / "reject"
    study_plan_emotional_reason: str  # Emotional/workload reviewer reasoning
    study_plan_consensus: bool  # Both study-plan reviewers approved
    study_plan_revision_notes: str  # Feedback for study_plan_agent regeneration
    study_plan_document_artifact: dict  # Markdown/DOCX artifact metadata
    context: Annotated[list[dict], context_reducer]  # Merged retrieval context (fan-in)
    retrieval_plan: list[dict]  # Multi-subject retrieval plan
    primary_subject: str  # Main subject of the user goal
    learning_goal: str  # Normalized learning goal
    subject_relation_summary: str  # How subjects relate to the goal
    local_retrieval_query: str  # Initial rewritten query for local course retrieval
    web_research_seed_query: str  # Initial rewritten query for Web Research
    expanded_keypoints: list[str]  # Query rewriter expanded concrete keypoints
    search_query_rewrite_reason: str  # Query rewriter rationale
    search_query_rewrite_error: str  # Query rewriter failure reason, if any
    search_query_rewrite_raw_preview: (
        str  # Truncated raw query-rewriter output for diagnostics
    )
    retrieval_branch_mode: str  # multi_subject_plan / single_subject_synthetic
    web_evidence_count: int  # Approved source_type=web evidence count
    web_research_debug: dict  # Web Research V2 execution status/debug summary
    web_research_outcome: Literal[
        "", "success", "failed", "skipped"
    ]  # Web Research V2 outcome
    local_evidence_candidates: list[
        dict
    ]  # Local RAG EvidenceCandidate snapshots from rag_retrieve
    web_evidence_candidates: list[
        dict
    ]  # Web EvidenceCandidate snapshots from Web Research V2
    local_evidence_originals: dict  # Original local RAG docs keyed by evidence_id
    web_evidence_originals: dict  # Original Tavily results keyed by evidence_id
    evidence_candidates: list[dict]  # Dual-source local/web EvidenceCandidate snapshots
    graded_evidence: list[dict]  # LLM-judged evidence snapshots for CE provider handoff
    evidence_judge_output: dict  # Raw structured Evidence Judge output
    evidence_judge_debug: dict  # Evidence Judge execution status/debug summary
    evidence_judge_rounds: int  # Evidence Judge rounds executed
    evidence_judge_state: str  # sufficient / partially_sufficient / insufficient
    evidence_coverage_gaps: list[
        dict
    ]  # Coverage gaps reserved for future search optimization
    search_refinement_needed: bool  # Evidence Judge requested more search
    search_refinement_deferred: bool  # Follow-up search is intentionally deferred
    search_refinement_deferred_reason: str  # Why refinement was not executed
    proposed_followup_search_queries: list[
        dict
    ]  # Reserved future search queries from coverage gaps
    search_optimization_reserved: bool  # Search optimization hook is reserved
    search_optimization_status: str  # reserved_not_implemented / disabled
    dual_source_mode: bool  # rag_retrieve used dual_source_evidence mode
    evidence_judge_failed: bool  # Evidence Judge failed and no evidence was admitted
    degraded_generation: bool  # Generation proceeds without approved evidence
    degraded_reason: str  # Reason for degraded generation
    evidence_controlled_stop: bool  # Controlled stop due to insufficient evidence
    evidence_controlled_stop_reason: str  # Reason for controlled stop
    parent_child_retrieval_result: dict  # Strict child-only multi-branch result
    parent_child_local_refs: list[dict]  # Judge-safe child references; no parent body
    parent_child_generation_id: str  # Generation pinned for this request
    parent_child_retrieval_fingerprint: str  # Retrieval policy fingerprint
    parent_child_hydration: dict  # Content-free post-Judge hydration summary
    evidence_orchestration_fingerprint: str  # Joint policy/profile/index fingerprint
    evidence_orchestration_status: str  # planned / retrieving / repairing / terminal
    evidence_requested_resource_types: list[str]  # Original canonical resource request
    evidence_requested_subjects: list[str]  # Canonical subjects bound to requirements
    evidence_requirements: list[dict]  # Strict compiled EvidenceRequirement rows
    evidence_current_round: int  # Initial round=0; supplements=1..configured maximum
    evidence_current_tasks: list[dict]  # Strict tasks dispatched in the active round
    evidence_all_tasks: list[dict]  # Cumulative bounded RetrievalTask inventory
    evidence_retrieval_signatures: list[str]  # Exact no-repeat signatures
    evidence_candidate_records: list[dict]  # Requirement-bound candidate snapshots
    evidence_ledger: list[dict]  # Content-free accepted/rejected evidence ledger
    evidence_coverage: dict  # Current RequirementCoverageBatch
    evidence_previous_coverage: dict  # Prior round RequirementCoverageBatch
    evidence_source_outcomes: list[dict]  # Completed versus empty source outcomes
    evidence_parent_child_rounds: list[dict]  # Child-only snapshots awaiting hydration
    evidence_repair_plans: list[dict]  # Strict EvidenceRepairPlan history
    evidence_consecutive_no_progress_rounds: int  # Supplement no-progress guard
    evidence_orchestration_route: str  # retrieve / repair / terminal
    evidence_terminal_status: str  # Explicit bounded terminal state
    evidence_terminal_reason_code: str  # Content-free terminal reason
    evidence_hydration_count: int  # Must transition from zero to one exactly once
    evidence_local_batch: dict  # Active-round local branch output
    evidence_web_batch: dict  # Active-round web branch output
    plan: str  # Generated plans
    resource_generation_plan: dict  # Parallel resource generation plan
    resource_branch_results: Annotated[
        list[dict], resource_branch_results_reducer
    ]  # Parallel resource worker results
    resource_task: dict  # Dynamic Send payload owned by one resource worker
    resource_bundle_artifact: dict  # Aggregated multi-resource artifact metadata
    resource_final_v3: dict  # Strict authoritative Resource Final V3 payload
    resource_generation_debug: (
        dict  # Resource generation execution status/debug summary
    )
    resource_generation_status: str  # success / partial_success / failed / skipped
    resource_evidence_readiness: list[dict]  # Per-resource code-derived readiness
    resource_evidence_assignments: list[dict]  # Evidence refs for ready resources
    ready_resource_types: list[str]  # Workers permitted to run
    blocked_resource_types: list[str]  # Explicit insufficient-evidence resources
    learning_path: dict  # Curriculum Engine: LearningPath serialized
    curriculum_context: str  # KG-aware context string for study_plan_planner
    learner_path_planner_output: dict  # Strict learner_path_planner_output_v1
    quiz_results: list[
        dict
    ]  # Assessment: recent quiz attempts with error classifications
    adaptive_tasks: list[dict]  # Assessment: generated adaptive practice tasks
    recommendations: list[dict]  # Recommendation Engine: ranked recommendations
    recommendation_resource_context: list[
        dict
    ]  # Generated resources explicitly available to automatic recommendation
    resource_recommendation_output: dict  # Strict resource_recommendation_output_v1
    review_schedule: list[dict]  # Assessment: due spaced repetition review items
    retry_count: int  # Hallucination retry counter
    hallucination_detected: bool  # Hallucination flag
    rewritten_query: str  # Rewritten query on retry
    hallucination_reason: str  # Reason from hallucination eval
