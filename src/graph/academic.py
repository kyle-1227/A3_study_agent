"""SubGraph A — Academic Learning Assistant: parallel retrieval (fan-out/fan-in),
answer generation, and hallucination evaluation with retry loop.

Keypoint extraction is handled by the supervisor node (merged for latency),
so this subgraph starts at the academic_router which fans out to both
rag_retrieve and web_search in parallel.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from typing import Any, Literal

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config import get_setting, load_prompt
from src.graph.evidence import EvidenceCandidate, EvidenceJudgeItem, EvidenceJudgeOutput
from src.graph.llm import async_invoke_with_fallback, get_fallback_llm, get_node_llm
from src.graph.state import CONTEXT_CLEAR, EVIDENCE_MEMORY_MAX_ENTRIES, LearningState
from src.llm.structured_output import (
    StructuredLLMResult,
    StructuredOutputError,
    get_fallback_modes,
    get_llm_output_mode,
    get_max_raw_chars,
    invoke_structured_llm,
)
from src.llm.http_messages import normalize_openai_messages, validate_openai_messages
from src.observability.a3_trace import emit_a3_trace
from src.rag.course_catalog import get_available_subjects_from_data, normalize_subject
from src.rag.retriever import retrieve
from src.tools.search_tool import sanitize_error_message, search_with_diagnostics as web_search_fn
from src.tracing import traced_llm_call, traced_node, traced_retrieval, traced_search

logger = logging.getLogger(__name__)

MAX_RETRIES = get_setting("academic.max_retries", 2)


# ── Structured output schema for hallucination evaluation ─────────
class HallucinationEvaluation(BaseModel):
    """LLM-evaluated faithfulness judgment."""

    is_faithful: bool = Field(
        description="True if the answer is grounded in the retrieved context "
        "and addresses the student's question without fabrication",
    )
    reason: str = Field(
        description="Brief explanation of the evaluation judgment",
    )


class RetrievalPlanItem(BaseModel):
    """Structured per-subject retrieval instruction."""

    subject: str = ""
    role: str = ""
    rag_query: str = ""
    web_search_query: str = ""
    purpose: str = ""
    relation_to_goal: str = ""
    priority: float = 0.5
    coverage_hint: str = ""
    expected_coverage: list[str] = Field(default_factory=list)


class SearchQueryRewriteOutput(BaseModel):
    """Structured initial retrieval-query rewrite result."""

    rag_query: str = Field(description="Query optimized for local course/RAG retrieval")
    web_search_query: str = Field(description="Query optimized for external web search")
    expanded_keypoints: list[str] = Field(description="Expanded concrete knowledge points")
    reason: str = Field(description="Brief rationale for the rewrite")
    learning_goal: str = Field(default="", description="Normalized learning goal")
    primary_subject: str = Field(default="", description="Main subject for the user goal")
    subject_relation_summary: str = Field(default="", description="How subjects relate to the goal")
    retrieval_plan: list[RetrievalPlanItem] = Field(
        default_factory=list,
        description="Per-subject retrieval plan",
    )
    memory_context_notes: list[str] = Field(
        default_factory=list,
        description="Notes about how conversation/evidence memory relates to current query",
    )
    memory_used_for_retrieval: bool = Field(
        default=False,
        description="Whether evidence memory influenced the retrieval plan",
    )
    memory_use_reason: str = Field(
        default="",
        description="Why memory was or was not used for retrieval",
    )


_HISTORY_REFERENCE_PATTERNS = (
    "之前", "上次", "刚才", "刚刚", "前面", "前面说", "前面讲",
    "历史", "刚才说", "刚才讲", "之前说", "之前讲",
    "前述", "前文", "上文", "上回", "继续", "接着说", "接着讲",
    "previously", "before", "last time", "earlier", "history",
    "previous", "above", "aforementioned", "继续上面的",
)


def _has_explicit_history_reference(query: str) -> bool:
    """Check if the user query contains explicit history-reference language.

    This is a lightweight pattern match — no hardcoded discipline keywords.
    """
    lowered = (query or "").lower()
    return any(pattern.lower() in lowered for pattern in _HISTORY_REFERENCE_PATTERNS)


def validate_search_query_rewrite_output(
    parsed: BaseModel,
    *,
    current_query: str = "",
) -> str:
    """Business validation for retrieval query rewriting.

    If memory_used_for_retrieval is true but the current query does NOT
    contain explicit history-reference language, fail validation.
    """
    if not isinstance(parsed, SearchQueryRewriteOutput):
        return "root expected SearchQueryRewriteOutput"
    if not str(parsed.rag_query or "").strip():
        return "rag_query must be non-empty"
    if not str(parsed.web_search_query or "").strip():
        return "web_search_query must be non-empty"
    for idx, item in enumerate(parsed.retrieval_plan or []):
        prefix = f"retrieval_plan.{idx}"
        if item.priority < 0 or item.priority > 1:
            return f"{prefix}.priority must be between 0 and 1"
        if item.subject and not str(item.subject).strip():
            return f"{prefix}.subject must be a string"
        if item.role and not str(item.role).strip():
            return f"{prefix}.role must be a string"
    # ── Memory use validation ───────────────────────────────────────
    # Two valid paths for memory to influence retrieval:
    # 1. Current query contains explicit history-reference language, OR
    # 2. LLM marks memory_used_for_retrieval=true with a non-empty reason.
    if parsed.memory_used_for_retrieval:
        has_explicit_ref = _has_explicit_history_reference(current_query)
        has_valid_reason = bool((parsed.memory_use_reason or "").strip())
        if not has_explicit_ref and not has_valid_reason:
            return (
                "memory_used_for_retrieval=true but current query does not "
                "contain explicit history-reference language and "
                "memory_use_reason is empty. "
                "Memory may only influence retrieval when the user "
                "explicitly references previous conversation or the LLM "
                "provides a valid reason."
            )
    return ""


def validate_hallucination_eval(parsed: BaseModel) -> str:
    """Business validation for hallucination evaluation."""
    if not isinstance(parsed, HallucinationEvaluation):
        return "root expected HallucinationEvaluation"
    if not isinstance(parsed.is_faithful, bool):
        return "is_faithful must be a boolean"
    if not str(parsed.reason or "").strip():
        return "reason must be non-empty"
    return ""


class SupplementPlanItem(BaseModel):
    """One candidate Web Search query for a supplement purpose."""

    purpose: str = ""
    query: str = ""
    priority: float = 0.5
    reason: str = ""


class SubjectCoverageDecision(BaseModel):
    """LLM decision for one retrieval branch."""

    subject: str = ""
    role: str = ""
    local_evidence_strength: str = "unknown"
    coverage_risk: str = "low"
    web_supplement_needed: bool = False
    supplement_purposes: list[str] = Field(default_factory=list)
    supplement_plan: list[SupplementPlanItem] = Field(default_factory=list)
    reason: str = ""
    priority: float = 0.5


class CoverageDecisionOutput(BaseModel):
    """LLM output for branch-aware Web supplement decisions."""

    overall_need_web: bool = False
    decision_summary: str = ""
    subject_decisions: list[SubjectCoverageDecision] = Field(default_factory=list)


class SearchResultJudgeItem(BaseModel):
    """Strict judgment for one Tavily search result."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(..., description="Index of the Tavily result in the input list.")
    title: str = ""
    url: str = ""
    keep: bool = Field(..., description="Whether this result should enter context.")
    final_quality: Literal["high", "medium", "low"] = "low"
    relevance: Literal["high", "medium", "low"] = "low"
    authority: Literal["high", "medium", "low"] = "low"
    usefulness: Literal["high", "medium", "low"] = "low"
    risk: Literal["high", "medium", "low"] = "low"
    evidence_type: Literal[
        "university_course_pdf",
        "textbook_or_notes",
        "official_documentation",
        "open_exercise_set",
        "github_or_notebook",
        "educational_platform",
        "quiz_or_practice_site",
        "video",
        "blog_or_article",
        "unknown",
    ] = "unknown"
    use_case: Literal[
        "core_evidence",
        "exercise_material",
        "implementation_reference",
        "background_context",
        "inspiration_only",
        "discard",
    ] = "discard"
    reason: str = Field(..., description="Specific reason for keep/drop decision.")


class SearchResultJudgeOutput(BaseModel):
    """Strict Search Result Judge response."""

    model_config = ConfigDict(extra="forbid")

    judged_results: list[SearchResultJudgeItem] = Field(default_factory=list)


ALLOWED_SUPPLEMENT_PURPOSES = {
    "repair",
    "coverage_expansion",
    "application_context",
    "tool_ecosystem",
    "latest_practice",
    "case_example",
    "implementation_detail",
    "comparison",
    "planning_support",
    "resource_enrichment",
}


def _last_human_query(state: LearningState) -> str:
    """Extract the last HumanMessage content (robust for retry loops)."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _render_prompt(prompt_name: str, replacements: dict[str, str]) -> str:
    """Render named placeholders without interpreting JSON braces in prompts."""
    prompt = load_prompt(prompt_name)
    for key, value in replacements.items():
        prompt = prompt.replace("{" + key + "}", str(value))
    return prompt


def _message_content_to_text(content) -> str:
    """Convert chat message content into text for diagnostics."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content or "")


_RETRIEVAL_ROLES = {
    "core_concept",
    "implementation_tool",
    "application_context",
    "prerequisite",
    "comparison",
    "extension",
    "method_for_domain",
    "case_carrier",
    "constraint",
    "supporting_context",
}


def _clear_retrieval_plan_state() -> dict:
    """Clear multi-subject retrieval fields to avoid checkpointer residue."""
    return {
        "retrieval_plan": [],
        "learning_goal": "",
        "primary_subject": "",
        "subject_relation_summary": "",
        "web_supplement_decisions": [],
        "web_supplement_results": [],
        "coverage_decision_summary": "",
        "retrieval_branch_mode": "",
        "web_supplement_provider": "tavily",
        "web_supplement_failed": False,
        "web_supplement_failure_reason": "",
        "web_supplement_status_by_subject": {},
        "web_supplement_success_subjects": [],
        "web_supplement_failed_subjects": [],
        "web_supplement_partial_failed": False,
        "web_judge_provider": "openrouter",
        "web_judge_model": "deepseek/deepseek-v4-flash",
        "web_judge_failed_subjects": [],
        "web_judge_rejected_all_subjects": [],
        "web_evidence_count": 0,
        "web_supplement_count": 0,
        "evidence_candidates": [],
        "evidence_judge_output": {},
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
    }


def _is_retry_rewrite_active(state: LearningState) -> bool:
    """True only when a hallucination retry rewrite is in progress."""
    return bool(
        (state.get("retry_count") or 0) > 0
        or state.get("hallucination_reason", "")
    )


def select_relevant_memory_summaries(
    state: LearningState,
    current_query: str,
    subject: str,
    requested_resource_type: str,
    *,
    max_selected: int = 3,
) -> list[dict]:
    """Select compact evidence memory summaries relevant to the current query.

    Reads ``summary`` first, falls back to ``decision_summary``.
    Tolerates missing fields and traces missing-field counts.
    Returns only compact summaries — never raw docs, full old context, or
    full historical answers.
    """
    memory_entries = state.get("evidence_summary_memory") or []
    if not memory_entries:
        return []

    query_lower = (current_query or "").lower()
    subject_lower = (subject or "").lower()
    resource_lower = (requested_resource_type or "").lower()
    missing_field_counts: dict[str, int] = {}

    def _get_summary(entry: dict) -> str:
        val = str(entry.get("summary") or "")
        if val.strip():
            return val.strip()
        val = str(entry.get("decision_summary") or "")
        return val.strip()

    scored: list[tuple[float, dict]] = []

    for idx, entry in enumerate(memory_entries):
        # ── Track missing fields ────────────────────────────────────
        for field in ("summary", "subject", "resource_type", "decision_summary"):
            if not entry.get(field):
                missing_field_counts[field] = missing_field_counts.get(field, 0) + 1

        score = 0.0
        entry_subject = str(entry.get("subject") or "").lower()
        entry_resource = str(entry.get("resource_type") or entry.get("requested_resource_type") or "").lower()
        entry_summary = _get_summary(entry).lower()

        # Recency bonus
        score += 0.2

        # Subject match
        if subject_lower and entry_subject == subject_lower:
            score += 0.3
        elif subject_lower and entry_subject and subject_lower in entry_subject:
            score += 0.15

        # Resource type match
        if resource_lower and entry_resource == resource_lower:
            score += 0.2

        # Query term overlap (lightweight signal only)
        if query_lower and entry_summary:
            query_terms = set(query_lower.split())
            summary_terms = set(entry_summary.split())
            if query_terms and summary_terms:
                overlap = len(query_terms & summary_terms) / max(len(query_terms), 1)
                score += min(overlap * 0.3, 0.3)

        scored.append((score, entry))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [
        entry
        for _, entry in scored[:max_selected]
        if _ > 0.0
    ]

    selected_ids = [e.get("memory_id", "") for e in selected]
    emit_a3_trace(
        logger,
        "memory_summary_selection",
        {
            "available_count": len(memory_entries),
            "selected_count": len(selected),
            "selected_ids": selected_ids,
            "selection_reason": f"scored {len(memory_entries)} entries, selected top {len(selected)}",
            "missing_field_counts": missing_field_counts,
            "prompt_chars_added": sum(len(_get_summary(e)) for e in selected),
        },
        state=state,
        env_flag="LOG_A3_TRACE",
    )
    return selected


def _query_source(state: LearningState) -> tuple[str, str]:
    """Priority: search_rag_query > active retry rewritten_query > expanded_keypoints > keypoints > original query."""
    rewritten = state.get("rewritten_query", "")
    search_rag_query = state.get("search_rag_query", "")
    expanded_keypoints = state.get("expanded_keypoints", [])
    keypoints = state.get("keypoints", [])
    if search_rag_query:
        return search_rag_query, "search_rag_query"
    # rewritten_query is diagnostic only; used for retrieval only when retry rewrite is active
    if rewritten and _is_retry_rewrite_active(state):
        return rewritten, "rewritten_query"
    if expanded_keypoints:
        return " ".join(expanded_keypoints), "expanded_keypoints"
    if keypoints:
        return " ".join(keypoints), "keypoints"
    return _last_human_query(state), "original_query"


def _doc_subject(doc: dict) -> str | None:
    return (doc.get("metadata") or {}).get("subject")


def _subject_mismatch_count(docs: list[dict], subject: str | None) -> int:
    if not subject:
        return 0
    return sum(1 for doc in docs if _doc_subject(doc) != subject)


def _top_doc_summaries(docs: list[dict], limit: int = 5) -> list[dict]:
    return [
        {
            "rank": i + 1,
            "source": doc.get("source"),
            "metadata_subject": _doc_subject(doc),
            "raw_vector_score": doc.get("raw_vector_score"),
            "raw_vector_score_source": doc.get("raw_vector_score_source"),
            "raw_vector_score_direction": doc.get("raw_vector_score_direction"),
            "bm25_score": doc.get("bm25_score"),
            "bm25_score_direction": doc.get("bm25_score_direction"),
            "rerank_score": doc.get("rerank_score"),
        }
        for i, doc in enumerate(docs[:limit])
    ]


def _subjects_used(docs: list[dict]) -> list[str]:
    return sorted({str(doc.get("retrieval_subject")) for doc in docs if doc.get("retrieval_subject")})


def _roles_used(docs: list[dict]) -> list[str]:
    return sorted({str(doc.get("retrieval_role")) for doc in docs if doc.get("retrieval_role")})


def _is_web_evidence(item: dict) -> bool:
    return (
        item.get("source_type") == "web"
        or item.get("type") in {"web_evidence", "web_supplement"}
        or item.get("legacy_type") == "web_supplement"
        or item.get("type_legacy") == "web_supplement"
    )


def _web_evidence_items(items: list[dict]) -> list[dict]:
    return [item for item in items if _is_web_evidence(item)]


def _score_doc(doc: dict) -> float:
    """Best available score for sorting retrieved docs."""
    if doc.get("rerank_score") is not None:
        value = doc.get("rerank_score")
    elif doc.get("bm25_score") is not None:
        value = doc.get("bm25_score")
    else:
        value = 0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _best_doc_score(docs: list[dict]) -> float:
    """Return the best rerank score, or a non-authoritative fallback signal."""
    if not docs:
        return 0.0
    rerank_scores = [
        _score_doc(doc)
        for doc in docs
        if doc.get("rerank_score") is not None
    ]
    if rerank_scores:
        return max(rerank_scores)
    return max(_score_doc(doc) for doc in docs)


def _has_rerank_score(docs: list[dict]) -> bool:
    return any(doc.get("rerank_score") is not None for doc in docs)


def _branch_status_score_source(docs: list[dict]) -> str:
    return "rerank_score" if _has_rerank_score(docs) else "fallback_raw_retrieval_signal"


def _evaluate_retrieval_branch(
    *,
    subject: str,
    role: str,
    docs: list[dict],
    is_hit: bool,
    subject_mismatch_count: int,
    reranker_failed: bool = False,
) -> dict:
    """
    Classify one retrieval_plan branch by local evidence quality.

    ``role`` is accepted for future policy tuning; V1 keeps the threshold rules
    subject-agnostic and role-agnostic.
    """
    del subject, role
    doc_count = len(docs)
    score_source = _branch_status_score_source(docs)
    has_rerank_score = score_source == "rerank_score"
    best_score = _best_doc_score(docs)
    best_rerank_score = best_score if has_rerank_score else 0.0
    usable_threshold = float(get_setting("rag.branch_usable_threshold", 0.45))
    strong_threshold = float(get_setting("rag.branch_strong_threshold", 0.7))

    if doc_count == 0:
        branch_status = "missing"
        weak_reason = "no_docs"
    elif subject_mismatch_count > 0:
        branch_status = "weak"
        weak_reason = "subject_mismatch"
    elif not has_rerank_score:
        branch_status = "weak" if reranker_failed or not is_hit else "usable"
        weak_reason = "reranker_failed" if reranker_failed else ("retrieve_is_hit_false" if not is_hit else "")
    elif not is_hit:
        branch_status = "weak"
        weak_reason = "retrieve_is_hit_false"
    elif best_score < usable_threshold:
        branch_status = "weak"
        weak_reason = "low_rerank_score"
    elif best_score >= strong_threshold:
        branch_status = "strong"
        weak_reason = ""
    else:
        branch_status = "usable"
        weak_reason = ""

    return {
        "branch_status": branch_status,
        "weak_reason": weak_reason,
        "best_rerank_score": best_rerank_score,
        "best_retrieval_score": best_score,
        "branch_status_score_source": score_source,
        "reranker_failed": bool(reranker_failed),
        "doc_count": doc_count,
        "should_use_in_generation": branch_status in {"strong", "usable", "weak"},
        "needs_supplement": branch_status in {"weak", "missing"},
    }


def _doc_dedupe_key(doc: dict) -> str:
    source = str(doc.get("source") or (doc.get("metadata") or {}).get("source_file") or "")
    content = str(doc.get("content") or "")
    digest = hashlib.md5(content.encode("utf-8")).hexdigest()
    return f"{source}:{digest}"


def _clamp_priority(value) -> float:
    try:
        priority = float(value)
    except (TypeError, ValueError):
        priority = 0.5
    return max(0.0, min(1.0, priority))


def _allowed_retrieval_subjects(state: LearningState) -> set[str]:
    """Build the subject hard boundary for retrieval plans."""
    available = set(get_available_subjects_from_data())
    if available:
        return available
    subject = normalize_subject(str(state.get("subject") or ""))
    return {subject} if subject and subject != "other" else set()


def _normalize_retrieval_plan(
    raw_plan: list[RetrievalPlanItem],
    state: LearningState,
) -> tuple[list[dict], dict]:
    """Filter and normalize LLM-produced per-subject retrieval plan."""
    allowed_subjects = _allowed_retrieval_subjects(state)
    by_subject: dict[str, dict] = {}
    rejected_items: list[dict] = []

    for item in raw_plan or []:
        subject = normalize_subject(item.subject)
        rag_query = item.rag_query.strip()
        if not subject:
            rejected_items.append({"subject": subject, "reason": "empty_subject"})
            continue
        if not rag_query:
            rejected_items.append({"subject": subject, "reason": "empty_rag_query"})
            continue
        if subject not in allowed_subjects:
            rejected_items.append({"subject": subject, "reason": "subject_not_in_available_subjects"})
            continue

        role = item.role.strip() or "supporting_context"
        if role not in _RETRIEVAL_ROLES:
            rejected_items.append({"subject": subject, "reason": "invalid_role_fallback_to_supporting_context"})
            role = "supporting_context"

        normalized = {
            "subject": subject,
            "role": role,
            "rag_query": rag_query,
            "web_search_query": item.web_search_query.strip(),
            "purpose": item.purpose.strip(),
            "relation_to_goal": item.relation_to_goal.strip(),
            "priority": _clamp_priority(item.priority),
            "coverage_hint": item.coverage_hint.strip(),
            "expected_coverage": [
                str(value).strip()
                for value in item.expected_coverage
                if str(value).strip()
            ],
        }

        existing = by_subject.get(subject)
        if existing is None or normalized["priority"] > existing["priority"]:
            if existing is not None:
                rejected_items.append({"subject": subject, "reason": "duplicate_subject_lower_priority"})
            by_subject[subject] = normalized
        else:
            rejected_items.append({"subject": subject, "reason": "duplicate_subject_lower_priority"})

    plan = sorted(by_subject.values(), key=lambda item: item["priority"], reverse=True)[:4]

    return plan, {
        "raw_plan_count": len(raw_plan or []),
        "normalized_plan_count": len(plan),
        "accepted_subjects": [item["subject"] for item in plan],
        "rejected_items": rejected_items,
    }


def _normalize_primary_subject(parsed_primary: str, plan: list[dict]) -> str:
    primary = normalize_subject(parsed_primary)
    plan_subjects = {item["subject"] for item in plan}
    if primary and primary in plan_subjects:
        return primary
    return plan[0]["subject"] if plan else ""


def _maybe_fail_subject_conflict(
    *,
    parsed_primary: str,
    normalized_primary: str,
    supervisor_subject: str,
    available_subjects: list[str],
    retrieval_plan: list[dict],
) -> None:
    """Fail-fast if the LLM subject conflicts with supervisor/available subjects
    in a way normalization cannot justify."""
    raw = (parsed_primary or "").strip().lower()
    if not raw:
        return  # LLM made no subject claim — no conflict to check

    sv = (supervisor_subject or "").strip().lower()
    if not sv or sv in ("unknown", "other"):
        return  # Supervisor did not classify — no conflict baseline

    norm = (normalized_primary or "").strip().lower()
    available_lower = {s.lower() for s in available_subjects}
    plan_subjects_lower = {item.get("subject", "").lower() for item in retrieval_plan}

    # No conflict: normalized matches supervisor's subject
    if norm == sv:
        return
    # No conflict: normalized is in available subjects
    if norm and norm in available_lower:
        return
    # No conflict: LLM raw matches supervisor (normalization lost it)
    if raw == sv:
        return

    # Conflict: raw is plausible (in available) but normalized mismatched
    # — that's a normalization issue, not a conflict
    if raw in available_lower:
        return

    # Genuine conflict: LLM proposes a subject that is neither the
    # supervisor's subject nor in the available/plan set
    if norm and plan_subjects_lower and norm not in plan_subjects_lower:
        if norm not in available_lower:
            raise ValueError(
                f"search_query_rewriter subject conflict: "
                f"LLM proposed '{parsed_primary}' (normalized '{normalized_primary}'), "
                f"but supervisor subject is '{supervisor_subject}' "
                f"and normalized subject is not in available subjects."
            )


def _web_query_source(state: LearningState) -> tuple[str, str]:
    search_web_query = state.get("search_web_query", "")
    rewritten = state.get("rewritten_query", "")
    if search_web_query:
        return search_web_query, "search_web_query"
    if rewritten and _is_retry_rewrite_active(state):
        return rewritten, "rewritten_query"
    return _last_human_query(state), "original_query"


def _build_retrieval_branches(state: LearningState) -> tuple[list[dict], dict]:
    """Build unified retrieval branches for multi- and single-subject paths.

    retrieval_plan always wins when non-empty.
    Stale rewritten_query never suppresses retrieval plan.
    """
    retrieval_plan = state.get("retrieval_plan") or []
    retry_active = _is_retry_rewrite_active(state)
    rewritten_query = state.get("rewritten_query", "")

    if retrieval_plan:
        branches = [dict(item, _synthetic_single_subject=False) for item in retrieval_plan]
        debug = {
            "mode": "multi_subject_plan",
            "branch_count": len(branches),
            "subjects": [item.get("subject") for item in branches],
            "synthetic_single_subject": False,
            "query_source": "retrieval_plan",
            "rewritten_query_present": bool(rewritten_query),
            "retry_rewrite_active": retry_active,
            "ignored_stale_rewritten_query": bool(rewritten_query and not retry_active),
            "used_retrieval_plan": True,
            "retrieval_plan_count": len(branches),
        }
        return branches, debug

    query, query_source = _query_source(state)
    web_query, _web_source = _web_query_source(state)
    subject = normalize_subject(str(state.get("subject") or "other")) or "other"
    branch = {
        "subject": subject,
        "role": "core_concept",
        "rag_query": query,
        "web_search_query": web_query,
        "purpose": "Retrieve local course evidence for the current single-subject question.",
        "relation_to_goal": "This subject is the main evidence source for the current question.",
        "priority": 1.0,
        "coverage_hint": "",
        "expected_coverage": [],
        "_synthetic_single_subject": True,
    }
    debug = {
        "mode": "single_subject_synthetic",
        "branch_count": 1 if query else 0,
        "subjects": [subject] if query else [],
        "synthetic_single_subject": True,
        "query_source": query_source,
        "rewritten_query_present": bool(rewritten_query),
        "retry_rewrite_active": retry_active,
        "ignored_stale_rewritten_query": False,
        "used_retrieval_plan": False,
        "retrieval_plan_count": 0,
    }
    return ([branch] if query else []), debug


_BRANCH_STATUS_RANK = {
    "strong": 3,
    "usable": 2,
    "weak": 1,
    "missing": 0,
}


def _select_docs_with_subject_quota(
    docs: list[dict],
    max_docs: int,
    *,
    primary_subject: str = "",
) -> tuple[list[dict], dict]:
    """Keep a balanced, quality-aware multi-subject context."""
    if max_docs <= 0:
        return [], {
            "quota_used": {},
            "subject_quota": {},
            "dropped_docs_count": len(docs),
        }

    deduped: list[dict] = []
    seen: set[str] = set()
    for doc in docs:
        key = (
            f"diagnostic:{doc.get('retrieval_subject')}:{doc.get('retrieval_role')}"
            if doc.get("type") == "rag_diagnostic"
            else _doc_dedupe_key(doc)
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for doc in deduped:
        grouped[str(doc.get("retrieval_subject") or "unknown")].append(doc)

    subject_max_docs = int(get_setting("rag.multi_subject_subject_max_docs", 3))
    primary_extra_docs = int(get_setting("rag.multi_subject_primary_extra_docs", 1))
    weak_max_docs = int(get_setting("rag.multi_subject_weak_max_docs", 1))

    subject_quota: dict[str, int] = {}
    for subject in grouped:
        quota = subject_max_docs + (primary_extra_docs if subject == primary_subject else 0)
        subject_quota[subject] = max(1, quota)

    def _sort_key(doc: dict) -> tuple:
        status = str(doc.get("branch_status") or "usable")
        return (
            _BRANCH_STATUS_RANK.get(status, 0),
            float(doc.get("retrieval_priority") or 0),
            _score_doc(doc),
        )

    for subject_docs in grouped.values():
        subject_docs.sort(key=_sort_key, reverse=True)

    selected: list[dict] = []
    selected_keys: set[str] = set()
    quota_used: Counter = Counter()

    def _doc_key(doc: dict) -> str:
        if doc.get("type") == "rag_diagnostic":
            return f"diagnostic:{doc.get('retrieval_subject')}:{doc.get('retrieval_role')}"
        return _doc_dedupe_key(doc)

    def _can_select(doc: dict) -> bool:
        subject = str(doc.get("retrieval_subject") or "unknown")
        status = str(doc.get("branch_status") or "usable")
        if quota_used[subject] >= subject_quota.get(subject, subject_max_docs):
            return False
        if status == "weak":
            weak_used = sum(
                1
                for selected_doc in selected
                if selected_doc.get("retrieval_subject") == subject
                and selected_doc.get("branch_status") == "weak"
            )
            if weak_used >= weak_max_docs:
                return False
        if status == "missing":
            missing_used = any(
                selected_doc.get("retrieval_subject") == subject
                and selected_doc.get("branch_status") == "missing"
                for selected_doc in selected
            )
            if missing_used:
                return False
        return True

    def _add_doc(doc: dict) -> bool:
        if len(selected) >= max_docs:
            return False
        key = _doc_key(doc)
        if key in selected_keys or not _can_select(doc):
            return False
        selected.append(doc)
        selected_keys.add(key)
        quota_used[str(doc.get("retrieval_subject") or "unknown")] += 1
        return True

    subjects_by_priority = sorted(
        grouped,
        key=lambda subject: (
            _sort_key(grouped[subject][0]),
        ),
        reverse=True,
    )

    for subject in subjects_by_priority:
        for doc in grouped[subject]:
            if _add_doc(doc):
                break

    remaining = [
        doc
        for subject_docs in grouped.values()
        for doc in subject_docs
        if _doc_key(doc) not in selected_keys
    ]
    remaining.sort(key=_sort_key, reverse=True)

    for doc in remaining:
        _add_doc(doc)

    branch_status_distribution = Counter(doc.get("branch_status", "usable") for doc in selected)
    branch_status_by_subject: dict[str, dict[str, int]] = defaultdict(dict)
    for subject, subject_docs in grouped.items():
        status_counter = Counter(doc.get("branch_status", "usable") for doc in subject_docs)
        branch_status_by_subject[subject] = dict(status_counter)

    quota_debug = {
        "quota_used": dict(quota_used),
        "subject_quota": subject_quota,
        "branch_status_distribution": dict(branch_status_distribution),
        "branch_status_by_subject": dict(branch_status_by_subject),
        "dropped_docs_count": max(0, len(deduped) - len(selected)),
        "weak_subjects": sorted({
            str(doc.get("retrieval_subject"))
            for doc in deduped
            if doc.get("branch_status") == "weak"
        }),
        "missing_subjects": sorted({
            str(doc.get("retrieval_subject"))
            for doc in deduped
            if doc.get("branch_status") == "missing"
        }),
    }
    return selected, quota_debug


def _web_setting(key: str, default):
    return get_setting(f"web_search.{key}", default)


def _web_conditional_enabled() -> bool:
    return bool(_web_setting("conditional_supplement_enabled", True))


def _web_timeout_seconds() -> float:
    try:
        return max(1.0, float(_web_setting("timeout_seconds", get_setting("academic.search_timeout", 6))))
    except (TypeError, ValueError):
        return 6.0


def _tavily_exception_diagnostics(
    query: str,
    exc: Exception,
    *,
    original_user_query: str = "",
    subject: str = "",
    role: str = "",
    purpose: str = "",
    elapsed_ms=None,
) -> dict:
    return {
        "provider": "tavily",
        "query": query,
        "original_user_query": original_user_query,
        "subject": subject,
        "role": role,
        "purpose": purpose,
        "ok": False,
        "results": [],
        "result_count": 0,
        "error_type": type(exc).__name__,
        "error_message": sanitize_error_message(exc),
        "raw_type": "",
        "raw_count": None,
        "elapsed_ms": elapsed_ms,
        "status_code": None,
    }


def _coerce_web_search_diagnostics(
    value: Any,
    *,
    query: str,
    original_user_query: str = "",
    subject: str = "",
    role: str = "",
    purpose: str = "",
) -> dict:
    """Normalize older list-style mocks into Tavily diagnostics shape."""
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {
            "provider": "tavily",
            "query": query,
            "original_user_query": original_user_query,
            "subject": subject,
            "role": role,
            "purpose": purpose,
            "ok": True,
            "results": value,
            "result_count": len(value),
            "error_type": "",
            "error_message": "",
            "raw_type": "list",
            "raw_count": len(value),
            "elapsed_ms": None,
            "status_code": None,
        }
    return _tavily_exception_diagnostics(
        query,
        TypeError(f"Unexpected web search diagnostics type: {type(value).__name__}"),
        original_user_query=original_user_query,
        subject=subject,
        role=role,
        purpose=purpose,
    )


def _clip_text(value: Any, limit: int) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def _compact_web_query(query: str, *, purpose: str = "", subject: str = "", max_chars: int = 160) -> str:
    """Pure compression only — never add subject, purpose, or discipline terms.

    - Normalize whitespace
    - Remove duplicate tokens while preserving input order
    - Enforce max length
    - Preserve terms already present in the input
    """
    text = " ".join(str(query or "").replace("\n", " ").split())
    if len(text) <= max_chars and len(text.split()) <= 8:
        return text

    raw_tokens = text.split()
    seen: set[str] = set()
    english_tokens: list[str] = []
    other_tokens: list[str] = []
    filler_tokens = {
        "with",
        "tutorial",
        "tutorials",
        "course",
        "courses",
        "notes",
        "note",
        "practice",
        "problem",
        "problems",
        "coding",
        "and",
        "or",
    }
    for token in raw_tokens:
        cleaned = token.strip(" ,;，；。.!?()[]{}<>\"'`")
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen or key in filler_tokens:
            continue
        seen.add(key)
        if any(ch.isascii() and (ch.isalnum() or ch in {"-", "_", ".", "+", "#"}) for ch in cleaned):
            english_tokens.append(cleaned)
        else:
            other_tokens.append(cleaned)

    selected: list[str] = []
    prioritized = list(english_tokens)
    prioritized.extend(other_tokens[:4])
    for token in prioritized:
        candidate = " ".join([*selected, token]).strip()
        if len(candidate) > max_chars:
            continue
        selected.append(token)

    compacted = " ".join(selected).strip()
    return compacted[:max_chars] if compacted else text[:max_chars]


def _build_branch_summaries(
    *,
    retrieval_plan: list[dict],
    branch_evals: dict[str, dict],
    docs_by_subject: dict[str, list[dict]],
) -> list[dict]:
    """Build compact summaries for LLM coverage decision."""
    summaries: list[dict] = []
    for item in retrieval_plan:
        subject = str(item.get("subject") or "")
        docs = docs_by_subject.get(subject, [])
        branch_eval = branch_evals.get(subject, {})
        summaries.append({
            "subject": subject,
            "role": item.get("role", ""),
            "priority": item.get("priority", 0.5),
            "branch_status": branch_eval.get("branch_status", "unknown"),
            "weak_reason": branch_eval.get("weak_reason", ""),
            "best_rerank_score": branch_eval.get("best_rerank_score", 0.0),
            "branch_status_score_source": branch_eval.get("branch_status_score_source", ""),
            "reranker_failed": branch_eval.get("reranker_failed", False),
            "used_doc_count": len(docs),
            "top_docs": [
                {
                    "source": doc.get("source"),
                    "metadata_subject": _doc_subject(doc),
                    "rerank_score": doc.get("rerank_score"),
                    "raw_vector_score": doc.get("raw_vector_score"),
                    "bm25_score": doc.get("bm25_score"),
                    "preview": _clip_text(doc.get("content", ""), 160),
                }
                for doc in docs[:3]
            ],
            "coverage_hint": item.get("coverage_hint", ""),
            "expected_coverage": item.get("expected_coverage", []),
            "web_search_query": item.get("web_search_query", ""),
        })
    return summaries


def _allowed_plan_subjects(branches: list[dict]) -> set[str]:
    return {str(item.get("subject") or "") for item in branches if item.get("subject")}


def _normalize_purposes(values: list[str]) -> list[str]:
    max_purposes = int(_web_setting("max_purposes_per_subject", 4))
    purposes: list[str] = []
    for value in values or []:
        purpose = str(value).strip()
        if purpose in ALLOWED_SUPPLEMENT_PURPOSES and purpose not in purposes:
            purposes.append(purpose)
        if len(purposes) >= max_purposes:
            break
    return purposes


def _normalize_supplement_query(query: str) -> str:
    max_chars = int(_web_setting("max_query_chars", 180))
    return _compact_web_query(query, max_chars=max_chars)


def _build_web_attempt_schedule(targets: list[dict]) -> list[dict]:
    """Build a fair schedule: one first-pass query per subject, then retries."""
    schedule: list[dict] = []
    retry_items: list[dict] = []
    max_chars = int(_web_setting("max_query_chars", 160))
    sorted_targets = sorted(
        targets,
        key=lambda target: _clamp_priority(target.get("subject_priority", 0.5)),
        reverse=True,
    )
    for target in sorted_targets:
        subject = str(target.get("subject") or "")
        queries = sorted(
            target.get("supplement_queries", []) or [],
            key=lambda item: _clamp_priority(item.get("priority", 0.5)),
            reverse=True,
        )
        for idx, query_item in enumerate(queries):
            purpose = query_item.get("purpose") or (target.get("supplement_purposes") or ["coverage_expansion"])[0]
            raw_query = str(query_item.get("query", "") or "")
            compacted_query = _compact_web_query(
                raw_query,
                purpose=purpose,
                subject=subject,
                max_chars=max_chars,
            )
            attempt = {
                "subject": subject,
                "role": target.get("role", ""),
                "purpose": purpose,
                "raw_query": raw_query,
                "query": compacted_query,
                "query_compacted": raw_query.strip() != compacted_query.strip(),
                "query_priority": _clamp_priority(query_item.get("priority", 0.5)),
                "subject_priority": _clamp_priority(target.get("subject_priority", 0.5)),
                "reason": query_item.get("reason") or target.get("decision_reason", ""),
                "target": target,
                "attempt_group": "first_pass" if idx == 0 else "retry_pass",
            }
            if not attempt["query"]:
                continue
            if idx == 0:
                schedule.append(attempt)
            else:
                retry_items.append(attempt)
    retry_items.sort(key=lambda item: (item.get("subject_priority", 0), item.get("query_priority", 0)), reverse=True)
    return schedule + retry_items


def _empty_web_subject_status(target: dict) -> dict:
    return {
        "subject": target.get("subject", ""),
        "role": target.get("role", ""),
        "attempts": 0,
        "success": False,
        "used_result_count": 0,
        "failed_attempts": 0,
        "last_error_type": "",
        "last_error_message": "",
        "purposes": target.get("supplement_purposes", []),
        "purposes_attempted": [],
        "purposes_succeeded": [],
        "queries_attempted": [],
        "queries_succeeded": [],
    }


def _targets_from_decision(
    *,
    parsed: CoverageDecisionOutput,
    branches: list[dict],
    branch_evals: dict[str, dict],
) -> list[dict]:
    allowed_subjects = _allowed_plan_subjects(branches)
    branch_by_subject = {str(item.get("subject")): item for item in branches}
    max_subjects = int(_web_setting("max_supplement_subjects", 2))
    max_items = int(_web_setting("max_plan_items_per_subject", 3))
    targets: list[dict] = []

    for decision in parsed.subject_decisions or []:
        subject = normalize_subject(decision.subject)
        if subject not in allowed_subjects or not decision.web_supplement_needed:
            continue
        branch = branch_by_subject.get(subject, {})
        purposes = _normalize_purposes(decision.supplement_purposes)
        query_items: list[dict] = []
        for item in sorted(decision.supplement_plan or [], key=lambda value: _clamp_priority(value.priority), reverse=True):
            purpose = item.purpose if item.purpose in ALLOWED_SUPPLEMENT_PURPOSES else (purposes[0] if purposes else "coverage_expansion")
            query = _compact_web_query(
                item.query,
                purpose=purpose,
                subject=subject,
                max_chars=int(_web_setting("max_query_chars", 160)),
            )
            if not query:
                continue
            query_items.append({
                "purpose": purpose,
                "query": query,
                "priority": _clamp_priority(item.priority),
                "reason": item.reason.strip(),
            })
            if len(query_items) >= max_items:
                break
        if not query_items and decision.web_supplement_needed:
            fallback_query = branch.get("web_search_query") or branch.get("rag_query") or ""
            if fallback_query:
                purpose = purposes[0] if purposes else "coverage_expansion"
                query_items.append({
                    "purpose": purpose,
                    "query": _compact_web_query(
                        fallback_query,
                        purpose=purpose,
                        subject=subject,
                        max_chars=int(_web_setting("max_query_chars", 160)),
                    ),
                    "priority": _clamp_priority(decision.priority),
                    "reason": decision.reason.strip() or "Fallback query from retrieval branch.",
                })
        if not query_items:
            continue
        branch_eval = branch_evals.get(subject, {})
        targets.append({
            "subject": subject,
            "role": decision.role.strip() or branch.get("role", "supporting_context"),
            "coverage_risk": decision.coverage_risk if decision.coverage_risk in {"low", "medium", "high"} else "low",
            "local_evidence_strength": decision.local_evidence_strength or branch_eval.get("branch_status", "unknown"),
            "supplement_purposes": purposes,
            "supplement_queries": query_items,
            "decision_reason": decision.reason.strip(),
            "subject_priority": _clamp_priority(decision.priority),
            "branch_status": branch_eval.get("branch_status", "unknown"),
        })

    targets.sort(
        key=lambda target: (
            _clamp_priority(target.get("subject_priority")),
            max((_clamp_priority(item.get("priority")) for item in target.get("supplement_queries", [])), default=0),
        ),
        reverse=True,
    )
    return targets[:max_subjects]


async def _decide_web_supplement_with_llm(
    *,
    state: LearningState,
    retrieval_plan: list[dict],
    branch_evals: dict[str, dict],
    docs_by_subject: dict[str, list[dict]],
    branch_mode: str,
) -> tuple[list[dict], dict]:
    """Return selected Web supplement targets and diagnostics."""
    enabled = bool(_web_setting("llm_decision_enabled", True))
    if not enabled:
        raise RuntimeError("web coverage decision disabled and rule fallback is not allowed")

    branch_summaries = _build_branch_summaries(
        retrieval_plan=retrieval_plan,
        branch_evals=branch_evals,
        docs_by_subject=docs_by_subject,
    )
    web_budget = {
        "max_total_attempts": int(_web_setting("max_total_attempts", 10)),
        "max_attempts_per_subject": int(_web_setting("max_attempts_per_subject", 3)),
        "max_supplement_subjects": int(_web_setting("max_supplement_subjects", 2)),
        "max_results_per_attempt": int(_web_setting("max_results_per_attempt", 5)),
        "timeout_seconds": _web_timeout_seconds(),
    }
    prompt = _render_prompt(
        "web_coverage_decision",
        {
            "question": _last_human_query(state),
            "intent": str(state.get("intent", "")),
            "requested_resource_type": str(state.get("requested_resource_type", "")),
            "learning_goal": str(state.get("learning_goal", "")),
            "subject_relation_summary": str(state.get("subject_relation_summary", "")),
            "branch_mode": branch_mode,
            "branch_summaries": json.dumps(branch_summaries, ensure_ascii=False),
            "web_budget": json.dumps(web_budget, ensure_ascii=False),
        },
    )
    messages = [
        SystemMessage(content="You are a coverage decision agent. Return only schema-valid JSON."),
        HumanMessage(content=prompt),
    ]

    structured_result = await invoke_structured_llm(
        node_name="web_coverage_decision",
        llm_node="web_coverage_decision",
        schema=CoverageDecisionOutput,
        messages=messages,
        output_mode=get_llm_output_mode("web_coverage_decision"),
        fallback_modes=get_fallback_modes("web_coverage_decision"),
        state=state,
        max_raw_chars=get_max_raw_chars("web_coverage_decision"),
    )
    parsed = structured_result.parsed
    if not isinstance(parsed, CoverageDecisionOutput):
        raise TypeError("web_coverage_decision parsed result is not CoverageDecisionOutput")
    targets = _targets_from_decision(
        parsed=parsed,
        branches=retrieval_plan,
        branch_evals=branch_evals,
    )
    subject_decisions = [
        {
            "subject": decision.subject,
            "role": decision.role,
            "local_evidence_strength": decision.local_evidence_strength,
            "coverage_risk": decision.coverage_risk,
            "web_supplement_needed": decision.web_supplement_needed,
            "supplement_purposes": decision.supplement_purposes,
            "supplement_plan_count": len(decision.supplement_plan),
            "priority": decision.priority,
            "reason": decision.reason,
        }
        for decision in parsed.subject_decisions
    ]
    return targets, {
        "enabled": True,
        "llm_used": True,
        "success": True,
        "fallback_used": False,
        "overall_need_web": parsed.overall_need_web,
        "decision_summary": parsed.decision_summary,
        "subject_decisions": subject_decisions,
        "selected_targets": targets,
        "error_type": "",
        "parsing_error": "",
        "raw_preview": structured_result.raw_output[:500],
    }


def _judge_setting(key: str, default: Any) -> Any:
    return get_setting(f"llm.search_result_judge.{key}", default)


def _judge_provider() -> str:
    return os.getenv("SEARCH_RESULT_JUDGE_PROVIDER", str(_judge_setting("provider", "openrouter"))).strip() or "openrouter"


def _judge_model() -> str:
    return (
        os.getenv("SEARCH_RESULT_JUDGE_MODEL", str(_judge_setting("model", "deepseek/deepseek-v4-flash"))).strip()
        or "deepseek/deepseek-v4-flash"
    )


def _judge_base_url() -> str:
    return str(_judge_setting("base_url", "https://openrouter.ai/api/v1")).rstrip("/")


def _judge_api_key_env() -> str:
    return str(_judge_setting("api_key_env", "OPENROUTER_API_KEY") or "OPENROUTER_API_KEY")


def _judge_api_key() -> str:
    return os.getenv(_judge_api_key_env(), "").strip()


def _judge_max_tokens() -> int:
    try:
        return int(_judge_setting("max_tokens", 1200))
    except (TypeError, ValueError):
        return 1200


def _judge_temperature() -> float:
    try:
        return float(_judge_setting("temperature", 0))
    except (TypeError, ValueError):
        return 0.0


def _judge_result_preview(results: list[dict], *, limit: int = 8) -> list[dict]:
    previews: list[dict] = []
    for index, result in enumerate(results[:limit]):
        previews.append({
            "index": index,
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "content_preview": str(result.get("content", ""))[:600],
            "tavily_score": result.get("score"),
        })
    return previews


def _judge_result_payload(results: list[dict]) -> list[dict]:
    return [
        {
            "index": index,
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "content_preview": str(result.get("content", ""))[:900],
            "tavily_score": result.get("score"),
        }
        for index, result in enumerate(results)
    ]


def _judge_http_headers(api_key: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    app_title = os.getenv("OPENROUTER_APP_TITLE", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if app_title:
        headers["X-Title"] = app_title
    return headers


def _judge_response_schema() -> dict[str, Any]:
    return SearchResultJudgeOutput.model_json_schema()


def _build_judge_messages(
    *,
    state: LearningState,
    subject: str,
    role: str,
    purpose: str,
    search_query: str,
    raw_query: str,
    original_user_query: str,
    tavily_results: list[dict],
    coverage_risk: str,
    local_evidence_strength: str,
) -> list[dict]:
    prompt = _render_prompt(
        "search_result_judge",
        {
            "original_user_query": original_user_query,
            "learning_goal": str(state.get("learning_goal", "")),
            "requested_resource_type": str(state.get("requested_resource_type", "")),
            "subject": subject,
            "role": role,
            "purpose": purpose,
            "coverage_risk": coverage_risk,
            "local_evidence_strength": local_evidence_strength,
            "raw_query": raw_query,
            "search_query": search_query,
            "tavily_results": json.dumps(_judge_result_payload(tavily_results), ensure_ascii=False),
        },
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a strict Web Search Result Judge. Return only a valid JSON object "
                "matching the provided JSON schema. Do not answer the user."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def _judge_request_payload(messages: list[dict]) -> dict[str, Any]:
    openai_messages = normalize_openai_messages(messages)
    validate_openai_messages(openai_messages)
    return {
        "model": _judge_model(),
        "messages": openai_messages,
        "temperature": _judge_temperature(),
        "max_tokens": _judge_max_tokens(),
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "search_result_judge_output",
                "strict": True,
                "schema": _judge_response_schema(),
            },
        },
    }


def _openrouter_chat_completion(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    api_key = _judge_api_key()
    if not api_key:
        raise RuntimeError(f"{_judge_api_key_env()} is not configured")
    timeout = max(5.0, _web_timeout_seconds() + 8.0)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{_judge_base_url()}/chat/completions",
            headers=_judge_http_headers(api_key),
            json=payload,
        )
        status_code = response.status_code
        response.raise_for_status()
        return response.json(), status_code


def _extract_openrouter_response_meta(raw_response: dict[str, Any]) -> tuple[str, str, int, int]:
    """Return (content, finish_reason, prompt_tokens, completion_tokens) from OpenRouter response."""
    choices = raw_response.get("choices") if isinstance(raw_response, dict) else None
    if not choices:
        return "", "", 0, 0
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = _message_content_to_text(message.get("content", ""))
    finish_reason = str(first.get("finish_reason", ""))
    usage = raw_response.get("usage") if isinstance(raw_response, dict) else {}
    prompt_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
    completion_tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0
    return content, finish_reason, prompt_tokens, completion_tokens


def _extract_openrouter_content(raw_response: dict[str, Any]) -> str:
    content, _, _, _ = _extract_openrouter_response_meta(raw_response)
    return content


def _judge_failure_debug(
    *,
    failure_phase: str,
    original_user_query: str,
    raw_query: str,
    search_query: str,
    subject: str,
    role: str,
    purpose: str,
    tavily_results: list[dict],
    error_type: str = "",
    error_message: str = "",
    status_code: Any = None,
    parsing_error: str = "",
    validation_error: str = "",
    raw_output: str = "",
) -> dict:
    return {
        "provider": _judge_provider(),
        "model": _judge_model(),
        "success": False,
        "search_result_judge_failed": True,
        "failure_phase": failure_phase,
        "original_user_query": original_user_query,
        "raw_query": raw_query,
        "search_query": search_query,
        "subject": subject,
        "role": role,
        "purpose": purpose,
        "input_result_count": len(tavily_results),
        "input_results_preview": _judge_result_preview(tavily_results),
        "accepted_count": 0,
        "rejected_count": 0,
        "error_type": error_type,
        "error_message": sanitize_error_message(error_message, max_chars=2000),
        "status_code": status_code,
        "parsing_error": sanitize_error_message(parsing_error, max_chars=2000),
        "validation_error": sanitize_error_message(validation_error, max_chars=4000),
        "raw_output": raw_output[:12000],
    }


async def _judge_tavily_search_results_with_llm(
    *,
    state: LearningState,
    subject: str,
    role: str,
    purpose: str,
    search_query: str,
    raw_query: str,
    original_user_query: str,
    tavily_results: list[dict],
    coverage_risk: str = "",
    local_evidence_strength: str = "",
) -> tuple[list[dict], dict]:
    """Judge Tavily results with OpenRouter strict schema. No rule fallback."""
    if not tavily_results:
        debug = {
            "provider": _judge_provider(),
            "model": _judge_model(),
            "success": True,
            "search_result_judge_failed": False,
            "judge_rejected_all": False,
            "original_user_query": original_user_query,
            "raw_query": raw_query,
            "search_query": search_query,
            "subject": subject,
            "role": role,
            "purpose": purpose,
            "input_result_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "judged_results": [],
            "raw_preview": "",
            "parsing_error": None,
            "validation_error": None,
        }
        return [], debug

    messages = _build_judge_messages(
        state=state,
        subject=subject,
        role=role,
        purpose=purpose,
        search_query=search_query,
        raw_query=raw_query,
        original_user_query=original_user_query,
        tavily_results=tavily_results,
        coverage_risk=coverage_risk,
        local_evidence_strength=local_evidence_strength,
    )
    request_payload = _judge_request_payload(messages)
    raw_output = ""
    status_code = None
    try:
        raw_response, status_code = await asyncio.to_thread(_openrouter_chat_completion, request_payload)
        raw_output = _extract_openrouter_content(raw_response)
        if not raw_output:
            debug = _judge_failure_debug(
                failure_phase="missing_judged_results",
                original_user_query=original_user_query,
                raw_query=raw_query,
                search_query=search_query,
                subject=subject,
                role=role,
                purpose=purpose,
                tavily_results=tavily_results,
                error_type="EmptyRawOutput",
                error_message="Search Result Judge returned empty content",
                status_code=status_code,
                raw_output=json.dumps(raw_response, ensure_ascii=False, default=str)[:12000],
            )
            emit_a3_trace(logger, "search_result_judge", debug, state=state, env_flag="LOG_WEB_SEARCH_RESULT", max_chars=12000)
            return [], debug
        try:
            parsed_json = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            debug = _judge_failure_debug(
                failure_phase="parsing_error",
                original_user_query=original_user_query,
                raw_query=raw_query,
                search_query=search_query,
                subject=subject,
                role=role,
                purpose=purpose,
                tavily_results=tavily_results,
                error_type=type(exc).__name__,
                error_message=str(exc),
                status_code=status_code,
                parsing_error=str(exc),
                raw_output=raw_output,
            )
            emit_a3_trace(logger, "search_result_judge", debug, state=state, env_flag="LOG_WEB_SEARCH_RESULT", max_chars=12000)
            return [], debug
        try:
            parsed = SearchResultJudgeOutput.model_validate(parsed_json)
        except ValidationError as exc:
            debug = _judge_failure_debug(
                failure_phase="validation_error",
                original_user_query=original_user_query,
                raw_query=raw_query,
                search_query=search_query,
                subject=subject,
                role=role,
                purpose=purpose,
                tavily_results=tavily_results,
                error_type=type(exc).__name__,
                error_message=str(exc),
                status_code=status_code,
                validation_error=str(exc),
                raw_output=raw_output,
            )
            emit_a3_trace(logger, "search_result_judge", debug, state=state, env_flag="LOG_WEB_SEARCH_RESULT", max_chars=12000)
            return [], debug

        judged = parsed.judged_results or []
        indexes = [item.index for item in judged]
        expected_indexes = set(range(len(tavily_results)))
        if not judged:
            failure_phase = "missing_judged_results"
        elif len(indexes) != len(set(indexes)) or set(indexes) != expected_indexes:
            failure_phase = "index_mismatch"
        else:
            failure_phase = ""
        if failure_phase:
            debug = _judge_failure_debug(
                failure_phase=failure_phase,
                original_user_query=original_user_query,
                raw_query=raw_query,
                search_query=search_query,
                subject=subject,
                role=role,
                purpose=purpose,
                tavily_results=tavily_results,
                error_type="InvalidJudgedResults",
                error_message=f"Expected indexes {sorted(expected_indexes)}, got {indexes}",
                status_code=status_code,
                validation_error=f"Expected indexes {sorted(expected_indexes)}, got {indexes}",
                raw_output=raw_output,
            )
            emit_a3_trace(logger, "search_result_judge", debug, state=state, env_flag="LOG_WEB_SEARCH_RESULT", max_chars=12000)
            return [], debug

        judged_by_index = {item.index: item for item in judged}
        accepted_results: list[dict] = []
        judged_payload: list[dict] = []
        for index, tavily_result in enumerate(tavily_results):
            item = judged_by_index[index]
            item_payload = item.model_dump()
            judged_payload.append(item_payload)
            if not item.keep:
                continue
            accepted_results.append({
                **tavily_result,
                "judge_keep": True,
                "judge_quality": item.final_quality,
                "judge_relevance": item.relevance,
                "judge_authority": item.authority,
                "judge_usefulness": item.usefulness,
                "judge_risk": item.risk,
                "evidence_type": item.evidence_type,
                "use_case": item.use_case,
                "judge_reason": item.reason,
                "judge_index": index,
                "judge_title": item.title,
                "judge_url": item.url,
            })

        debug = {
            "provider": _judge_provider(),
            "model": _judge_model(),
            "success": True,
            "search_result_judge_failed": False,
            "judge_rejected_all": not bool(accepted_results),
            "original_user_query": original_user_query,
            "raw_query": raw_query,
            "search_query": search_query,
            "subject": subject,
            "role": role,
            "purpose": purpose,
            "input_result_count": len(tavily_results),
            "accepted_count": len(accepted_results),
            "rejected_count": len(tavily_results) - len(accepted_results),
            "judged_results": judged_payload,
            "raw_preview": raw_output[:4000],
            "parsing_error": None,
            "validation_error": None,
            "status_code": status_code,
        }
        emit_a3_trace(logger, "search_result_judge", debug, state=state, env_flag="LOG_WEB_SEARCH_RESULT", max_chars=8000)
        return accepted_results, debug
    except Exception as exc:
        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code
            error_message = exc.response.text or str(exc)
        else:
            error_message = str(exc)
        error_type = "MissingApiKey" if _judge_api_key_env() in str(exc) else type(exc).__name__
        failure_phase = "missing_api_key" if error_type == "MissingApiKey" else "structured_output_request_failed"
        debug = _judge_failure_debug(
            failure_phase=failure_phase,
            original_user_query=original_user_query,
            raw_query=raw_query,
            search_query=search_query,
            subject=subject,
            role=role,
            purpose=purpose,
            tavily_results=tavily_results,
            error_type=error_type,
            error_message=error_message,
            status_code=status_code,
            raw_output=raw_output,
        )
        emit_a3_trace(logger, "search_result_judge", debug, state=state, env_flag="LOG_WEB_SEARCH_RESULT", max_chars=12000)
        return [], debug


def _retrieval_setting(key: str, default: Any) -> Any:
    return get_setting(f"retrieval.{key}", default)


def _dual_source_enabled() -> bool:
    return str(_retrieval_setting("mode", "")).strip() == "dual_source_evidence"


def _block_generation_when_evidence_judge_failed() -> bool:
    return bool(_retrieval_setting("dual_source_evidence.block_generation_when_evidence_judge_failed", True))


def _fail_fast_evidence_judge() -> bool:
    return bool(get_setting("development.fail_fast_evidence_judge", True))


def _evidence_failure_phase(state: LearningState) -> str:
    output = state.get("evidence_judge_output") or {}
    if isinstance(output, dict):
        return str(output.get("failure_phase") or output.get("degraded_reason") or "")
    return ""


def _evidence_judge_setting(key: str, default: Any) -> Any:
    return get_setting(f"llm.evidence_judge.{key}", default)


def _evidence_judge_output_setting(key: str, default: Any) -> Any:
    return get_setting(f"llm_outputs.evidence_judge.{key}", default)


def _evidence_judge_provider() -> str:
    return str(_evidence_judge_setting("provider", "openrouter") or "openrouter").strip()


def _evidence_judge_model() -> str:
    return str(_evidence_judge_setting("model", "deepseek/deepseek-v4-flash") or "deepseek/deepseek-v4-flash").strip()


def _evidence_judge_base_url() -> str:
    return str(
        _evidence_judge_setting("base_url", "https://openrouter.ai/api/v1")
    ).rstrip("/")


def _evidence_judge_api_key_env() -> str:
    return str(_evidence_judge_setting("api_key_env", "OPENROUTER_API_KEY") or "OPENROUTER_API_KEY")


def _evidence_judge_api_key() -> str:
    return os.getenv(_evidence_judge_api_key_env(), "").strip()


def _evidence_judge_max_tokens() -> int:
    try:
        return int(_evidence_judge_setting("max_tokens", 1800))
    except (TypeError, ValueError):
        return 1800


def _evidence_judge_temperature() -> float:
    try:
        return float(_evidence_judge_setting("temperature", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _candidate_preview(candidates: list[EvidenceCandidate], *, limit: int = 10) -> list[dict]:
    previews: list[dict] = []
    for candidate in candidates[:limit]:
        item = candidate.model_dump(mode="json")
        item["content_preview"] = str(item.get("content_preview", ""))[:300]
        previews.append(item)
    return previews


def _evidence_judge_response_schema() -> dict[str, Any]:
    return EvidenceJudgeOutput.model_json_schema()


def _build_evidence_judge_messages(
    *,
    candidates: list[EvidenceCandidate],
    original_user_query: str,
    learning_goal: str,
    requested_resource_type: str,
    round_index: int,
) -> list[dict]:
    payload = [candidate.model_dump(mode="json") for candidate in candidates]
    prompt = _render_prompt(
        "evidence_judge",
        {
            "original_user_query": original_user_query,
            "learning_goal": learning_goal,
            "requested_resource_type": requested_resource_type,
            "round_index": round_index,
            "evidence_candidates": json.dumps(payload, ensure_ascii=False),
        },
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a strict Evidence Judge. Return only valid JSON matching the "
                "provided JSON schema. Do not answer the user or perform search."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def _evidence_judge_request_payload(messages: list[dict]) -> dict[str, Any]:
    openai_messages = normalize_openai_messages(messages)
    validate_openai_messages(openai_messages)
    return {
        "model": _evidence_judge_model(),
        "messages": openai_messages,
        "temperature": _evidence_judge_temperature(),
        "max_tokens": _evidence_judge_max_tokens(),
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "evidence_judge_output",
                "strict": True,
                "schema": _evidence_judge_response_schema(),
            },
        },
    }


def _evidence_judge_schema_size_chars() -> int:
    return len(json.dumps(_evidence_judge_response_schema(), ensure_ascii=False))


def _messages_chars(messages: list[dict]) -> int:
    return sum(len(_message_content_to_text(message.get("content", ""))) for message in messages)


def _provider_name_from_error_body(raw_body: str) -> str:
    try:
        parsed = json.loads(raw_body)
    except Exception:
        return ""

    def _walk(value: Any) -> str:
        if isinstance(value, dict):
            for key in ("provider_name", "provider", "providerName"):
                if value.get(key):
                    return str(value.get(key))
            for nested in value.values():
                found = _walk(nested)
                if found:
                    return found
        if isinstance(value, list):
            for nested in value:
                found = _walk(nested)
                if found:
                    return found
        return ""

    return _walk(parsed)


def _classify_evidence_judge_failure(
    *,
    status_code: Any,
    raw_error_body: str,
    candidate_count: int,
    schema_size_chars: int,
    prompt_chars: int,
    default_phase: str,
) -> tuple[str, str, str]:
    text = str(raw_error_body or "").lower()
    if status_code == 404 and ("no endpoints found" in text or "can handle the requested parameters" in text):
        return (
            "structured_output_unsupported_by_provider",
            "OpenRouter routing layer rejected the request: no provider supports the required parameters (likely json_schema response_format).",
            "switch_to_prompt_json_pydantic_or_choose_another_model",
        )
    if status_code == 400:
        if any(term in text for term in ("unsupported", "not support", "does not support")) and any(
            term in text for term in ("json_schema", "response_format", "structured")
        ):
            return (
                "structured_output_unsupported_by_provider",
                "Provider rejected strict json_schema response_format.",
                "switch_to_prompt_json_pydantic_or_choose_another_model",
            )
        if any(term in text for term in ("too large", "context length", "maximum context", "token", "payload")):
            return (
                "payload_too_large_or_rejected",
                "Provider rejected the request size, prompt size, or candidate payload.",
                "reduce_candidate_count_or_preview_size",
            )
        if any(term in text for term in ("schema", "json_schema", "response_format", "strict")):
            return (
                "schema_too_complex_or_rejected",
                "Provider rejected the strict schema shape or response_format payload.",
                "simplify_schema_or_split_judge_batch",
            )
        if "provider returned error" in text or "provider_name" in text:
            return (
                "structured_output_unsupported_by_provider",
                "OpenRouter upstream provider rejected the strict structured-output request.",
                "run_schema_probe_then_change_model_or_explicitly_approve_prompt_json_mode",
            )
        if candidate_count > 8 or prompt_chars > 20000:
            return (
                "payload_too_large_or_rejected",
                "HTTP 400 occurred with a large candidate/prompt payload.",
                "reduce_candidate_count_or_preview_size",
            )
        if schema_size_chars > 8000:
            return (
                "schema_too_complex_or_rejected",
                "HTTP 400 occurred with a large strict schema.",
                "simplify_schema_or_split_judge_batch",
            )
    return (
        default_phase,
        "Evidence Judge request failed before producing valid judged evidence.",
        "inspect_provider_error_body",
    )


def _openrouter_evidence_chat_completion(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    api_key = _evidence_judge_api_key()
    if not api_key:
        raise RuntimeError(f"{_evidence_judge_api_key_env()} is not configured")
    with httpx.Client(timeout=max(10.0, _web_timeout_seconds() + 12.0)) as client:
        response = client.post(
            f"{_evidence_judge_base_url()}/chat/completions",
            headers=_judge_http_headers(api_key),
            json=payload,
        )
        status_code = response.status_code
        response.raise_for_status()
        return response.json(), status_code


def _evidence_judge_failure_debug(
    *,
    failure_phase: str,
    original_user_query: str,
    candidates: list[EvidenceCandidate],
    error_type: str = "",
    error_message: str = "",
    status_code: Any = None,
    parsing_error: str = "",
    validation_error: str = "",
    raw_output: str = "",
    raw_error_body: str = "",
    provider_error_body: str = "",
    provider_name: str = "",
    prompt_chars: int = 0,
    schema_size_chars: int = 0,
    message_count: int = 0,
    inferred_failure_reason: str = "",
    action_needed: str = "",
    finish_reason: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    using_direct_openrouter_http: bool = False,
    provider_request_mode: str = "",
) -> dict:
    schema_size = schema_size_chars or _evidence_judge_schema_size_chars()
    output_mode = str(
        _evidence_judge_output_setting(
            "output_mode",
            _evidence_judge_setting("output_mode", "native_json_schema_pydantic"),
        )
        or "native_json_schema_pydantic"
    )
    recommendation = ""
    if failure_phase == "structured_output_unsupported_by_provider":
        recommendation = "switch_evidence_judge_output_mode_to_prompt_json_pydantic"
    return {
        "stage": "evidence_judge",
        "provider": _evidence_judge_provider(),
        "model": _evidence_judge_model(),
        "round_index": 1,
        "success": False,
        "failure_phase": failure_phase,
        "inferred_failure_reason": inferred_failure_reason,
        "action_needed": action_needed,
        "recommendation": recommendation,
        "structured_output_method": output_mode,
        "output_mode": output_mode,
        "using_langchain_with_structured_output": False,
        "using_direct_openrouter_http": using_direct_openrouter_http,
        "provider_request_mode": provider_request_mode,
        "candidate_count": len(candidates),
        "schema_name": "EvidenceJudgeOutput",
        "schema_size_chars": schema_size,
        "prompt_chars": prompt_chars,
        "message_count": message_count,
        "provider_error_body": sanitize_error_message(provider_error_body or raw_error_body, max_chars=12000),
        "raw_error_body": sanitize_error_message(raw_error_body or provider_error_body, max_chars=12000),
        "provider_name": provider_name,
        "original_user_query": original_user_query,
        "input_candidate_count": len(candidates),
        "candidate_preview": _candidate_preview(candidates),
        "error_type": error_type,
        "error_message": sanitize_error_message(error_message, max_chars=2000),
        "status_code": status_code,
        "raw_output": raw_output[:12000],
        "parsing_error": sanitize_error_message(parsing_error, max_chars=2000),
        "validation_error": sanitize_error_message(validation_error, max_chars=4000),
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def _validate_evidence_judge_business_result(
    parsed: BaseModel,
    *,
    expected_ids: list[str],
) -> str:
    if not isinstance(parsed, EvidenceJudgeOutput):
        return "parsed result is not EvidenceJudgeOutput"
    judged_ids = [item.evidence_id for item in parsed.judged_evidence]
    problems: list[str] = []
    if len(judged_ids) != len(set(judged_ids)):
        problems.append("duplicate evidence_id values in judged_evidence")
    missing = [evidence_id for evidence_id in expected_ids if evidence_id not in judged_ids]
    extra = [evidence_id for evidence_id in judged_ids if evidence_id not in expected_ids]
    if missing:
        problems.append(f"missing evidence_id values: {missing}")
    if extra:
        problems.append(f"unknown evidence_id values: {extra}")
    if len(judged_ids) != len(expected_ids):
        problems.append(f"expected {len(expected_ids)} judged evidence items, got {len(judged_ids)}")
    return "; ".join(problems)


def _structured_result_to_evidence_failure_debug(
    *,
    result: StructuredLLMResult,
    original_user_query: str,
    candidates: list[EvidenceCandidate],
) -> dict:
    failure_phase = result.failure_phase or "structured_llm_failed"
    if result.business_validation_error and "evidence_id" in result.business_validation_error:
        failure_phase = "evidence_id_mismatch"
    return _evidence_judge_failure_debug(
        failure_phase=failure_phase,
        original_user_query=original_user_query,
        candidates=candidates,
        error_type=result.error_type,
        error_message=result.error_message,
        status_code=result.status_code,
        parsing_error=result.parsing_error,
        validation_error=result.validation_error or result.business_validation_error,
        raw_output=result.raw_output,
        raw_error_body=result.provider_error_body,
        provider_error_body=result.provider_error_body,
        prompt_chars=0,
        schema_size_chars=_evidence_judge_schema_size_chars(),
        message_count=0,
        inferred_failure_reason=(
            result.business_validation_error
            or result.validation_error
            or result.parsing_error
            or result.error_message
            or "Structured Evidence Judge call failed."
        ),
        action_needed="inspect_structured_llm_output_trace",
        using_direct_openrouter_http=result.using_direct_openrouter_http
        or any(attempt.using_direct_openrouter_http for attempt in result.attempts),
        provider_request_mode=result.provider_request_mode
        or next(
            (attempt.provider_request_mode for attempt in reversed(result.attempts) if attempt.provider_request_mode),
            "",
        ),
    )


def _fallback_evidence_type(candidate: EvidenceCandidate) -> str:
    if candidate.source_type == "local_rag":
        return "local_course_material"
    return "web_article"


def _build_fallback_evidence_judge_output(
    candidates: list[EvidenceCandidate],
    reason: str,
) -> EvidenceJudgeOutput:
    candidate_count = len(candidates)
    fallback_reason = sanitize_error_message(reason, max_chars=800) or (
        "Evidence Judge structured output failed validation."
    )
    return EvidenceJudgeOutput(
        overall_evidence_state="partially_sufficient" if candidate_count > 0 else "insufficient",
        need_more_web_search=candidate_count == 0,
        judged_evidence=[
            EvidenceJudgeItem(
                evidence_id=candidate.evidence_id,
                keep=True,
                final_quality="medium",
                relevance="medium",
                authority="medium",
                usefulness="medium",
                risk="low",
                evidence_type=_fallback_evidence_type(candidate),
                use_case="background_context",
                coverage_contribution=(
                    "Fallback retained this candidate because Evidence Judge structured output failed."
                ),
                reason=(
                    "Evidence Judge failed validation, so this candidate was retained by "
                    f"deterministic fallback. Reason: {fallback_reason}"
                ),
            )
            for candidate in candidates
        ],
        coverage_gaps=[],
        decision_summary=(
            "Evidence Judge validation failed; deterministic fallback retained available "
            f"evidence candidates for degraded generation. Reason: {fallback_reason}"
        ),
    )


def _fallback_failure_phase(debug: dict | None, reason: str) -> str:
    debug = debug or {}
    failure_phase = str(debug.get("failure_phase") or "").strip()
    if failure_phase:
        return failure_phase
    lowered = str(reason or "").lower()
    if "evidence_id" in lowered or "missing evidence_id" in lowered:
        return "evidence_id_mismatch"
    if "judged_evidence" in lowered:
        return "judged_evidence_empty"
    if "parsed" in lowered and "none" in lowered:
        return "parsed_none"
    return "structured_llm_failed"


def _fallback_evidence_judge_debug(
    *,
    base_debug: dict | None,
    fallback_output: EvidenceJudgeOutput,
    candidates: list[EvidenceCandidate],
    reason: str,
) -> dict:
    debug = dict(base_debug or {})
    failure_phase = _fallback_failure_phase(debug, reason)
    judged_count = len(fallback_output.judged_evidence)
    debug.update(
        {
            "stage": "evidence_judge",
            "success": False,
            "fallback_used": True,
            "failure_phase": failure_phase,
            "input_candidate_count": len(candidates),
            "fallback_judged_count": judged_count,
            "overall_evidence_state": fallback_output.overall_evidence_state,
            "need_more_web_search": fallback_output.need_more_web_search,
            "kept_count": judged_count,
            "rejected_count": 0,
            "coverage_gap_count": 0,
            "coverage_gaps": [],
            "fallback_reason": sanitize_error_message(reason, max_chars=2000),
            "fallback_evidence_ids": [item.evidence_id for item in fallback_output.judged_evidence],
        }
    )
    if "error_type" not in debug or not debug.get("error_type"):
        debug["error_type"] = "EvidenceJudgeFallback"
    return debug


def _fallback_evidence_judge_result(
    *,
    state: LearningState,
    candidates: list[EvidenceCandidate],
    reason: str,
    base_debug: dict | None = None,
) -> tuple[EvidenceJudgeOutput, dict]:
    fallback_output = _build_fallback_evidence_judge_output(candidates, reason)
    debug = _fallback_evidence_judge_debug(
        base_debug=base_debug,
        fallback_output=fallback_output,
        candidates=candidates,
        reason=reason,
    )
    emit_a3_trace(
        logger,
        "evidence_judge",
        debug,
        state=state,
        env_flag="LOG_WEB_SEARCH_RESULT",
        max_chars=12000,
    )
    return fallback_output, debug


async def _judge_evidence_candidates_with_llm(
    *,
    state: LearningState,
    candidates: list[EvidenceCandidate],
    original_user_query: str,
    learning_goal: str,
    requested_resource_type: str,
    round_index: int,
) -> tuple[EvidenceJudgeOutput | None, dict]:
    """Judge local RAG and Web evidence through the unified structured runtime."""
    if not candidates:
        parsed = EvidenceJudgeOutput(
            overall_evidence_state="insufficient",
            need_more_web_search=False,
            judged_evidence=[],
            coverage_gaps=[],
            decision_summary="No evidence candidates were available.",
        )
        debug = {
            "provider": _evidence_judge_provider(),
            "model": _evidence_judge_model(),
            "round_index": round_index,
            "success": True,
            "overall_evidence_state": parsed.overall_evidence_state,
            "need_more_web_search": False,
            "input_candidate_count": 0,
            "kept_count": 0,
            "rejected_count": 0,
            "kept_source_distribution": {},
            "coverage_gap_count": 0,
            "coverage_gaps": [],
            "raw_preview": "",
            "raw_output_chars": 0,
            "output_mode": str(_evidence_judge_output_setting("output_mode", "native_json_schema_pydantic") or "native_json_schema_pydantic"),
            "fallback_modes": [],
            "fallback_used": False,
            "default_used": False,
            "retry_count": 0,
            "failure_phase": "",
            "error_type": "",
            "provider_error_body": "",
            "parsing_error": None,
            "validation_error": None,
        }
        emit_a3_trace(logger, "evidence_judge", debug, state=state, env_flag="LOG_WEB_SEARCH_RESULT")
        return parsed, debug

    messages = _build_evidence_judge_messages(
        candidates=candidates,
        original_user_query=original_user_query,
        learning_goal=learning_goal,
        requested_resource_type=requested_resource_type,
        round_index=round_index,
    )
    expected_ids = [candidate.evidence_id for candidate in candidates]
    output_mode = str(_evidence_judge_output_setting("output_mode", "native_json_schema_pydantic") or "native_json_schema_pydantic")
    fallback_modes = _evidence_judge_output_setting("fallback_modes", [])
    if not isinstance(fallback_modes, list):
        fallback_modes = []

    try:
        structured_result = await invoke_structured_llm(
            node_name="evidence_judge",
            llm_node="evidence_judge",
            schema=EvidenceJudgeOutput,
            messages=messages,
            output_mode=output_mode,
            fallback_modes=[str(mode) for mode in fallback_modes],
            business_validator=lambda parsed: _validate_evidence_judge_business_result(parsed, expected_ids=expected_ids),
            state=state,
            max_raw_chars=int(_evidence_judge_output_setting("max_raw_chars", 12000) or 12000),
        )
    except StructuredOutputError as exc:
        debug = _structured_result_to_evidence_failure_debug(
            result=exc.result,
            original_user_query=original_user_query,
            candidates=candidates,
        )
        reason = (
            exc.result.business_validation_error
            or exc.result.validation_error
            or exc.result.parsing_error
            or exc.result.error_message
            or str(exc)
        )
        return _fallback_evidence_judge_result(
            state=state,
            candidates=candidates,
            reason=reason,
            base_debug=debug,
        )
    except Exception as exc:
        reason = str(exc)
        if (
            "BusinessValidationError" in type(exc).__name__
            or "BusinessValidationError" in reason
            or "evidence_id_mismatch" in reason
            or "missing evidence_id values" in reason
            or "judged_evidence" in reason
        ):
            return _fallback_evidence_judge_result(
                state=state,
                candidates=candidates,
                reason=reason,
                base_debug={
                    "stage": "evidence_judge",
                    "provider": _evidence_judge_provider(),
                    "model": _evidence_judge_model(),
                    "round_index": round_index,
                    "failure_phase": _fallback_failure_phase({}, reason),
                    "error_type": type(exc).__name__,
                    "error_message": sanitize_error_message(reason, max_chars=2000),
                },
            )
        raise

    if not structured_result.success or not isinstance(structured_result.parsed, EvidenceJudgeOutput):
        debug = _structured_result_to_evidence_failure_debug(
            result=structured_result,
            original_user_query=original_user_query,
            candidates=candidates,
        )
        reason = (
            structured_result.business_validation_error
            or structured_result.validation_error
            or structured_result.parsing_error
            or structured_result.error_message
            or "Evidence Judge returned no parsed result."
        )
        return _fallback_evidence_judge_result(
            state=state,
            candidates=candidates,
            reason=reason,
            base_debug=debug,
        )

    parsed = structured_result.parsed
    business_error = _validate_evidence_judge_business_result(parsed, expected_ids=expected_ids)
    if business_error or (len(candidates) > 0 and not parsed.judged_evidence):
        if business_error and "evidence_id" in business_error:
            failure_phase = "evidence_id_mismatch"
        elif not parsed.judged_evidence:
            failure_phase = "judged_evidence_empty"
        else:
            failure_phase = "business_validation_error"
        return _fallback_evidence_judge_result(
            state=state,
            candidates=candidates,
            reason=business_error or "judged_evidence is empty while candidates were provided.",
            base_debug={
                "stage": "evidence_judge",
                "provider": _evidence_judge_provider(),
                "model": _evidence_judge_model(),
                "round_index": round_index,
                "failure_phase": failure_phase,
                "error_type": "BusinessValidationError",
                "validation_error": business_error,
                "raw_preview": structured_result.raw_output[:4000],
                "raw_output_chars": len(structured_result.raw_output or ""),
            },
        )
    candidate_by_id = {candidate.evidence_id: candidate for candidate in candidates}
    kept = [item for item in parsed.judged_evidence if item.keep]
    kept_distribution = Counter(candidate_by_id[item.evidence_id].source_type for item in kept)
    debug = {
        "stage": "evidence_judge",
        "provider": _evidence_judge_provider(),
        "model": _evidence_judge_model(),
        "round_index": round_index,
        "success": True,
        "output_mode": structured_result.output_mode,
        "fallback_modes": structured_result.fallback_modes,
        "fallback_used": structured_result.fallback_used,
        "default_used": structured_result.default_used,
        "retry_count": structured_result.retry_count,
        "failure_phase": "",
        "error_type": "",
        "provider_error_body": "",
        "overall_evidence_state": parsed.overall_evidence_state,
        "need_more_web_search": parsed.need_more_web_search,
        "input_candidate_count": len(candidates),
        "kept_count": len(kept),
        "rejected_count": len(candidates) - len(kept),
        "kept_source_distribution": dict(kept_distribution),
        "coverage_gap_count": len(parsed.coverage_gaps),
        "coverage_gaps": [gap.model_dump() for gap in parsed.coverage_gaps],
        "raw_preview": structured_result.raw_output[:4000],
        "raw_output_chars": len(structured_result.raw_output or ""),
        "parsing_error": None,
        "validation_error": None,
        "structured_output_method": structured_result.output_mode,
        "using_direct_openrouter_http": structured_result.using_direct_openrouter_http
        or any(attempt.using_direct_openrouter_http for attempt in structured_result.attempts),
        "provider_request_mode": structured_result.provider_request_mode
        or next(
            (attempt.provider_request_mode for attempt in reversed(structured_result.attempts) if attempt.provider_request_mode),
            "",
        ),
        "schema_name": "EvidenceJudgeOutput",
        "schema_size_chars": _evidence_judge_schema_size_chars(),
        "prompt_chars": _messages_chars(messages),
        "message_count": len(messages),
    }
    emit_a3_trace(logger, "evidence_judge", debug, state=state, env_flag="LOG_WEB_SEARCH_RESULT", max_chars=8000)
    return parsed, debug


def _evidence_quality_rank(value: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(str(value), 0)


def _candidate_rank(candidate: EvidenceCandidate) -> float:
    if candidate.source_type == "local_rag":
        return float(candidate.rerank_score or 0)
    return float(candidate.tavily_score or 0)


def _build_local_evidence_candidates(
    *,
    docs: list[dict],
    subject: str,
    role: str,
    branch_status: str,
    branch_status_score_source: str,
) -> list[EvidenceCandidate]:
    candidates: list[EvidenceCandidate] = []
    for rank, doc in enumerate(docs):
        metadata = dict(doc.get("metadata") or {})
        source = str(doc.get("source") or metadata.get("source") or "")
        content = str(doc.get("content") or doc.get("page_content") or "")
        candidates.append(EvidenceCandidate(
            evidence_id=f"local:{subject or 'other'}:{rank}",
            source_type="local_rag",
            provider="chroma_rag",
            subject=subject,
            role=role,
            purpose=str(doc.get("retrieval_purpose") or "local_course_retrieval"),
            title=source,
            source=source,
            content_preview=_clip_text(content, 800),
            raw_vector_score=doc.get("raw_vector_score"),
            raw_vector_score_source=doc.get("raw_vector_score_source"),
            raw_vector_score_direction=doc.get("raw_vector_score_direction"),
            rerank_score=doc.get("rerank_score"),
            branch_status=branch_status,
            branch_status_score_source=branch_status_score_source,
            metadata={
                "metadata": metadata,
                "retrieval_query": doc.get("retrieval_query", ""),
                "weak_reason": doc.get("weak_reason", ""),
                "relation_to_goal": doc.get("relation_to_goal", ""),
                "retrieval_priority": doc.get("retrieval_priority", 0),
            },
        ))
    return candidates


def _build_web_evidence_candidates(
    *,
    tavily_results: list[dict],
    subject: str,
    role: str,
    purpose: str,
    query: str,
    attempt_index: int = 0,
) -> list[EvidenceCandidate]:
    candidates: list[EvidenceCandidate] = []
    for rank, result in enumerate(tavily_results):
        candidates.append(EvidenceCandidate(
            evidence_id=f"web:{subject or 'other'}:{attempt_index}:{rank}",
            source_type="web",
            provider="tavily",
            subject=subject,
            role=role,
            purpose=purpose,
            title=str(result.get("title") or ""),
            source=str(result.get("url") or result.get("title") or "tavily"),
            url=str(result.get("url") or ""),
            content_preview=_clip_text(str(result.get("content") or ""), 800),
            tavily_score=result.get("score"),
            tavily_query=query,
            metadata={"favicon": result.get("favicon", "")},
        ))
    return candidates


def _cap_evidence_candidates(candidates: list[EvidenceCandidate]) -> list[EvidenceCandidate]:
    max_candidates = int(_retrieval_setting("fusion.max_evidence_candidates", 16))
    max_local = int(_retrieval_setting("fusion.max_local_candidates", 8))
    max_web = int(_retrieval_setting("fusion.max_web_candidates", 8))
    local = sorted(
        [candidate for candidate in candidates if candidate.source_type == "local_rag"],
        key=_candidate_rank,
        reverse=True,
    )[:max_local]
    web = sorted(
        [candidate for candidate in candidates if candidate.source_type == "web"],
        key=_candidate_rank,
        reverse=True,
    )[:max_web]
    combined = local + web
    if len(combined) <= max_candidates:
        return combined
    preserve_balance = bool(_retrieval_setting("fusion.preserve_source_type_balance", True))
    if not preserve_balance:
        return sorted(combined, key=_candidate_rank, reverse=True)[:max_candidates]
    selected: list[EvidenceCandidate] = []
    if local and max_candidates > 0:
        selected.append(local[0])
    if web and len(selected) < max_candidates:
        selected.append(web[0])
    seen = {candidate.evidence_id for candidate in selected}
    for candidate in sorted(combined, key=_candidate_rank, reverse=True):
        if candidate.evidence_id in seen:
            continue
        selected.append(candidate)
        seen.add(candidate.evidence_id)
        if len(selected) >= max_candidates:
            break
    return selected


def _judge_context_rank(item: dict) -> tuple[int, int, int, float]:
    return (
        _evidence_quality_rank(item.get("judge_quality", "")),
        _evidence_quality_rank(item.get("judge_relevance", "")),
        _evidence_quality_rank(item.get("judge_usefulness", "")),
        float(item.get("rerank_score") or item.get("tavily_score") or 0),
    )


def _context_item_from_evidence(
    *,
    candidate: EvidenceCandidate,
    judge_item: Any,
    original: dict,
) -> dict:
    judge_fields = {
        "evidence_id": candidate.evidence_id,
        "judge_keep": True,
        "judge_quality": judge_item.final_quality,
        "judge_relevance": judge_item.relevance,
        "judge_authority": judge_item.authority,
        "judge_usefulness": judge_item.usefulness,
        "judge_risk": judge_item.risk,
        "evidence_type": judge_item.evidence_type,
        "use_case": judge_item.use_case,
        "coverage_contribution": judge_item.coverage_contribution,
        "judge_reason": judge_item.reason,
    }
    if candidate.source_type == "local_rag":
        return {
            **original,
            "type": "rag",
            "source_type": "local_rag",
            "provider": "chroma_rag",
            "subject": candidate.subject,
            "role": candidate.role,
            "retrieval_subject": candidate.subject,
            "retrieval_role": candidate.role,
            "branch_status_score_source": candidate.branch_status_score_source,
            **judge_fields,
        }
    return {
        "type": "web_evidence",
        "legacy_type": "web_supplement",
        "type_legacy": "web_supplement",
        "source_type": "web",
        "provider": "tavily",
        "title": original.get("title", candidate.title),
        "url": original.get("url", candidate.url),
        "content": original.get("content", ""),
        "source": original.get("url") or original.get("title") or candidate.source,
        "subject": candidate.subject,
        "role": candidate.role,
        "supplement_for_subject": candidate.subject,
        "supplement_for_role": candidate.role,
        "supplement_purpose": candidate.purpose,
        "supplement_purposes": [candidate.purpose] if candidate.purpose else [],
        "retrieval_subject": candidate.subject,
        "retrieval_role": candidate.role,
        "retrieval_query": candidate.tavily_query,
        "tavily_score": candidate.tavily_score,
        **judge_fields,
    }


def _select_judged_context(
    *,
    parsed: EvidenceJudgeOutput,
    candidates: list[EvidenceCandidate],
    originals: dict[str, dict],
) -> list[dict]:
    candidate_by_id = {candidate.evidence_id: candidate for candidate in candidates}
    items: list[dict] = []
    for judge_item in parsed.judged_evidence:
        if not judge_item.keep:
            continue
        candidate = candidate_by_id.get(judge_item.evidence_id)
        if not candidate:
            continue
        items.append(_context_item_from_evidence(
            candidate=candidate,
            judge_item=judge_item,
            original=originals.get(candidate.evidence_id, {}),
        ))
    max_docs = int(_retrieval_setting("fusion.max_context_docs", 8))
    preserve_balance = bool(_retrieval_setting("fusion.preserve_source_type_balance", True))
    if len(items) <= max_docs:
        return sorted(items, key=_judge_context_rank, reverse=True)
    sorted_items = sorted(items, key=_judge_context_rank, reverse=True)
    if not preserve_balance:
        return sorted_items[:max_docs]
    selected: list[dict] = []
    local = [item for item in sorted_items if item.get("source_type") == "local_rag"]
    web = [item for item in sorted_items if item.get("source_type") == "web"]
    if local:
        selected.append(local[0])
    if web and len(selected) < max_docs:
        selected.append(web[0])
    seen = {item.get("evidence_id") for item in selected}
    for item in sorted_items:
        if item.get("evidence_id") in seen:
            continue
        selected.append(item)
        seen.add(item.get("evidence_id"))
        if len(selected) >= max_docs:
            break
    return selected


def _followups_from_coverage_gaps(parsed: EvidenceJudgeOutput) -> list[dict]:
    followups: list[dict] = []
    for gap in parsed.coverage_gaps:
        followups.append({
            "subject": gap.subject,
            "role": gap.role,
            "gap": gap.gap,
            "suggested_search_query": gap.suggested_search_query,
            "purpose": gap.purpose,
            "priority": gap.priority,
            "source": "evidence_judge_coverage_gap",
            "status": "reserved_not_executed",
        })
    return followups


async def _run_dynamic_web_supplement(
    *,
    state: LearningState,
    targets: list[dict],
    decision_debug: dict,
    branch_mode: str,
) -> tuple[list[dict], dict]:
    """Run dynamic Web supplement with bounded attempts."""
    del decision_debug
    max_total = int(_web_setting("max_total_attempts", 3))
    max_per_subject = int(_web_setting("max_attempts_per_subject", 2))
    max_results = int(_web_setting("max_results_per_attempt", 2))
    min_results_per_subject = int(_web_setting("min_results_per_subject", 1))
    stop_after_success = bool(_web_setting("stop_subject_after_success", True))
    retry_failed_first = bool(_web_setting("retry_failed_subjects_first", True))
    timeout = _web_timeout_seconds()
    attempts = 0
    attempts_by_subject: Counter = Counter()
    docs: list[dict] = []
    attempt_logs: list[dict] = []
    schedule = _build_web_attempt_schedule(targets)
    original_user_query = _last_human_query(state)
    status_by_subject = {
        str(target.get("subject") or ""): _empty_web_subject_status(target)
        for target in targets
        if target.get("subject")
    }

    def _status(subject: str, target: dict) -> dict:
        if subject not in status_by_subject:
            status_by_subject[subject] = _empty_web_subject_status(target)
        return status_by_subject[subject]

    while attempts < max_total:
        candidates = [
            item for item in schedule
            if not item.get("_used") and attempts_by_subject[item.get("subject", "")] < max_per_subject
        ]
        if not candidates:
            break
        active_candidates = []
        for item in candidates:
            status = _status(item.get("subject", ""), item.get("target", {}))
            if stop_after_success and status.get("success") and int(status.get("used_result_count") or 0) >= min_results_per_subject:
                continue
            active_candidates.append(item)
        if active_candidates:
            candidates = active_candidates
        if retry_failed_first:
            candidates.sort(
                key=lambda item: (
                    1 if _status(item.get("subject", ""), item.get("target", {})).get("failed_attempts") else 0,
                    0 if _status(item.get("subject", ""), item.get("target", {})).get("success") else 1,
                    1 if item.get("attempt_group") == "first_pass" else 0,
                    item.get("subject_priority", 0),
                    item.get("query_priority", 0),
                ),
                reverse=True,
            )
        else:
            candidates.sort(
                key=lambda item: (
                    1 if item.get("attempt_group") == "first_pass" else 0,
                    item.get("subject_priority", 0),
                    item.get("query_priority", 0),
                ),
                reverse=True,
            )

        query_item = candidates[0]
        query_item["_used"] = True
        target = query_item.get("target", {})
        subject = query_item.get("subject", "")
        attempts += 1
        attempts_by_subject[subject] += 1
        query = query_item.get("query", "")
        purpose = query_item.get("purpose") or (target.get("supplement_purposes") or ["coverage_expansion"])[0]
        raw_query = query_item.get("raw_query", query)
        subject_status = _status(subject, target)
        subject_status["attempts"] = int(subject_status.get("attempts") or 0) + 1
        if purpose not in subject_status["purposes_attempted"]:
            subject_status["purposes_attempted"].append(purpose)
        subject_status["queries_attempted"].append(query)

        started = time.perf_counter()
        diagnostics: dict
        timed_out = False
        try:
            diagnostics = await asyncio.wait_for(
                asyncio.to_thread(
                    web_search_fn,
                    query,
                    original_user_query=original_user_query,
                    subject=subject,
                    role=str(target.get("role", "")),
                    purpose=purpose,
                    max_results=max_results,
                    timeout_seconds=timeout,
                ),
                timeout=timeout,
            )
            diagnostics = _coerce_web_search_diagnostics(
                diagnostics,
                query=query,
                original_user_query=original_user_query,
                subject=subject,
                role=str(target.get("role", "")),
                purpose=purpose,
            )
        except asyncio.TimeoutError:
            timed_out = True
            diagnostics = _tavily_exception_diagnostics(
                query,
                TimeoutError(f"tavily search exceeded {timeout}s"),
                original_user_query=original_user_query,
                subject=subject,
                role=str(target.get("role", "")),
                purpose=purpose,
                elapsed_ms=round(timeout * 1000, 2),
            )
        except Exception as exc:
            diagnostics = _tavily_exception_diagnostics(
                query,
                exc,
                original_user_query=original_user_query,
                subject=subject,
                role=str(target.get("role", "")),
                purpose=purpose,
            )

        elapsed_ms = diagnostics.get("elapsed_ms")
        if elapsed_ms is None:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        results = diagnostics.get("results", []) or []
        accepted_results: list[dict] = []
        judge_debug: dict = {
            "success": False,
            "search_result_judge_failed": False,
            "judge_rejected_all": False,
            "accepted_count": 0,
            "rejected_count": 0,
            "failure_phase": "",
        }
        if results:
            accepted_results, judge_debug = await _judge_tavily_search_results_with_llm(
                state=state,
                subject=subject,
                role=str(target.get("role", "")),
                purpose=purpose,
                search_query=query,
                raw_query=raw_query,
                original_user_query=original_user_query,
                tavily_results=results,
                coverage_risk=str(target.get("coverage_risk", "")),
                local_evidence_strength=str(target.get("local_evidence_strength", "")),
            )
        used_results = accepted_results[:max_results]
        for result in used_results:
            docs.append({
                "type": "web_supplement",
                "source_type": "web",
                "provider": "tavily",
                "content": result.get("content", ""),
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "source": result.get("url") or result.get("title") or "tavily",
                "supplement_for_subject": subject,
                "supplement_for_role": target.get("role", ""),
                "supplement_purpose": purpose,
                "supplement_purposes": target.get("supplement_purposes", []),
                "supplement_reason": query_item.get("reason") or target.get("decision_reason", ""),
                "retrieval_subject": subject,
                "retrieval_role": target.get("role", ""),
                "retrieval_query": query,
                "branch_status": target.get("branch_status", ""),
                "coverage_risk": target.get("coverage_risk", ""),
                "local_evidence_strength": target.get("local_evidence_strength", ""),
                "judge_keep": True,
                "judge_quality": result.get("judge_quality", "low"),
                "judge_relevance": result.get("judge_relevance", "low"),
                "judge_authority": result.get("judge_authority", "low"),
                "judge_usefulness": result.get("judge_usefulness", "low"),
                "judge_risk": result.get("judge_risk", "low"),
                "evidence_type": result.get("evidence_type", "unknown"),
                "use_case": result.get("use_case", "discard"),
                "judge_reason": result.get("judge_reason", ""),
                "tavily_score": result.get("score"),
            })
        if used_results:
            subject_status["success"] = True
            subject_status["used_result_count"] = int(subject_status.get("used_result_count") or 0) + len(used_results)
            if purpose not in subject_status["purposes_succeeded"]:
                subject_status["purposes_succeeded"].append(purpose)
            subject_status["queries_succeeded"].append(query)
            subject_status["last_error_type"] = ""
            subject_status["last_error_message"] = ""
        else:
            subject_status["failed_attempts"] = int(subject_status.get("failed_attempts") or 0) + 1
            if judge_debug.get("search_result_judge_failed"):
                subject_status["last_error_type"] = "SearchResultJudgeFailed"
                subject_status["last_failure_reason"] = judge_debug.get("failure_phase", "search_result_judge_failed")
                subject_status["last_error_message"] = judge_debug.get("error_message", "")
            elif judge_debug.get("judge_rejected_all"):
                subject_status["last_error_type"] = "JudgeRejectedAll"
                subject_status["last_failure_reason"] = "judge_rejected_all"
                subject_status["last_error_message"] = "search result judge rejected all Tavily results"
            else:
                subject_status["last_error_type"] = diagnostics.get("error_type") or "NoWebResults"
                subject_status["last_failure_reason"] = diagnostics.get("error_type") or "timeout_or_no_results"
                subject_status["last_error_message"] = diagnostics.get("error_message") or "no Tavily results"

        attempt_payload = {
            "branch_mode": branch_mode,
            "attempt_group": query_item.get("attempt_group", ""),
            "subject": subject,
            "role": target.get("role", ""),
            "purpose": purpose,
            "attempt": attempts,
            "subject_attempt": attempts_by_subject[subject],
            "max_total_attempts": max_total,
            "max_attempts_per_subject": max_per_subject,
            "provider": diagnostics.get("provider", "tavily"),
            "original_user_query": original_user_query[:2000],
            "raw_query": raw_query,
            "query": query,
            "query_compacted": bool(query_item.get("query_compacted")),
            "ok": diagnostics.get("ok", False),
            "timed_out": timed_out or diagnostics.get("error_type") == "TimeoutError",
            "status_code": diagnostics.get("status_code"),
            "raw_result_count": len(results),
            "result_count": diagnostics.get("result_count", len(results)),
            "search_result_judge_enabled": True,
            "search_result_judge_success": bool(judge_debug.get("success")),
            "search_result_judge_failed": bool(judge_debug.get("search_result_judge_failed")),
            "judge_rejected_all": bool(judge_debug.get("judge_rejected_all")),
            "judge_failure_phase": judge_debug.get("failure_phase", ""),
            "judge_accepted_count": judge_debug.get("accepted_count", 0),
            "judge_rejected_count": judge_debug.get("rejected_count", 0),
            "legacy_quality_filter_disabled": True,
            "used_result_count": len(used_results),
            "elapsed_ms": elapsed_ms,
            "error_type": diagnostics.get("error_type", ""),
            "error_message": diagnostics.get("error_message", ""),
            "top_results": [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "judge_quality": item.get("judge_quality"),
                    "judge_relevance": item.get("judge_relevance"),
                    "judge_authority": item.get("judge_authority"),
                    "evidence_type": item.get("evidence_type"),
                    "use_case": item.get("use_case"),
                    "judge_reason": item.get("judge_reason", ""),
                }
                for item in used_results[:3]
            ],
        }
        attempt_logs.append(attempt_payload)
        # TEMP A3_TRACE: remove after multi-subject retrieval validation.
        emit_a3_trace(
            logger,
            "dynamic_web_supplement",
            attempt_payload,
            state=state,
            env_flag="LOG_WEB_SEARCH_RESULT",
        )

    success_subjects = sorted(subject for subject, status in status_by_subject.items() if status.get("success"))
    failed_subjects = sorted(
        subject for subject, status in status_by_subject.items()
        if not status.get("success") and int(status.get("attempts") or 0) > 0
    )

    return docs, {
        "provider": "tavily",
        "attempts_used": attempts,
        "max_total_attempts": max_total,
        "attempts_by_subject": dict(attempts_by_subject),
        "result_doc_count": len(docs),
        "timeout_count": sum(1 for item in attempt_logs if item.get("timed_out")),
        "no_result_count": sum(1 for item in attempt_logs if not item.get("used_result_count")),
        "attempts": attempt_logs,
        "status_by_subject": status_by_subject,
        "success_subjects": success_subjects,
        "failed_subjects": failed_subjects,
        "judge_failed_subjects": sorted(
            subject for subject, status in status_by_subject.items()
            if status.get("last_error_type") == "SearchResultJudgeFailed"
        ),
        "judge_rejected_all_subjects": sorted(
            subject for subject, status in status_by_subject.items()
            if status.get("last_error_type") == "JudgeRejectedAll"
        ),
        "partial_failed": bool(success_subjects and failed_subjects),
    }


# ── Node 0: academic router (fan-out trigger) ─────────────────────

def _dual_source_web_query(state: LearningState, branch: dict) -> tuple[str, str]:
    if branch.get("web_search_query"):
        return str(branch.get("web_search_query")), "retrieval_branch_web_search_query"
    if state.get("search_web_query"):
        return str(state.get("search_web_query")), "search_web_query"
    if branch.get("rag_query"):
        return str(branch.get("rag_query")), "retrieval_branch_rag_query"
    return _last_human_query(state), "original_user_query"


def _source_distribution(items: list[dict]) -> dict:
    return dict(Counter(str(item.get("source_type") or item.get("type") or "unknown") for item in items))


async def _run_dual_source_first_round_web(
    *,
    state: LearningState,
    branch: dict,
    query: str,
    original_user_query: str,
) -> tuple[list[dict], dict]:
    max_results = int(_retrieval_setting("web.max_results_per_query", 3))
    timeout = float(_retrieval_setting("web.timeout_seconds", _web_timeout_seconds()))
    subject = str(branch.get("subject") or "")
    role = str(branch.get("role") or "")
    purpose = str(branch.get("purpose") or "first_round_dual_source")
    started = time.perf_counter()
    try:
        diagnostics = await asyncio.wait_for(
            asyncio.to_thread(
                web_search_fn,
                query,
                original_user_query=original_user_query,
                subject=subject,
                role=role,
                purpose=purpose,
                max_results=max_results,
                timeout_seconds=timeout,
            ),
            timeout=timeout,
        )
        diagnostics = _coerce_web_search_diagnostics(
            diagnostics,
            query=query,
            original_user_query=original_user_query,
            subject=subject,
            role=role,
            purpose=purpose,
        )
    except asyncio.TimeoutError:
        diagnostics = _tavily_exception_diagnostics(
            query,
            TimeoutError(f"tavily search exceeded {timeout}s"),
            original_user_query=original_user_query,
            subject=subject,
            role=role,
            purpose=purpose,
            elapsed_ms=round(timeout * 1000, 2),
        )
    except Exception as exc:
        diagnostics = _tavily_exception_diagnostics(
            query,
            exc,
            original_user_query=original_user_query,
            subject=subject,
            role=role,
            purpose=purpose,
        )
    diagnostics.setdefault("elapsed_ms", round((time.perf_counter() - started) * 1000, 2))
    return (diagnostics.get("results") or [])[:max_results], diagnostics


async def _rag_retrieve_dual_source(state: LearningState, branches: list[dict], branch_debug: dict) -> dict:
    original_user_query = _last_human_query(state)
    per_subject_top_k = int(_retrieval_setting("local_rag.per_subject_top_k", get_setting("rag.multi_subject_per_subject_top_k", 3)))
    local_enabled = bool(_retrieval_setting("local_rag.enabled", True))
    local_candidates_all: list[EvidenceCandidate] = []
    originals: dict[str, dict] = {}

    with traced_retrieval(
        query=original_user_query,
        subject=str(state.get("subject", "")),
        top_k=per_subject_top_k,
    ) as span:
        span.set_attribute("rag.mode", "dual_source_evidence")
        span.set_attribute("rag.branch_count", len(branches))
        for branch_index, branch in enumerate(branches):
            subject = str(branch.get("subject") or "")
            role = str(branch.get("role") or "supporting_context")
            rag_query = str(branch.get("rag_query") or original_user_query)
            retrieve_subject = None if subject == "other" else subject

            if local_enabled:
                result = retrieve(query=rag_query, subject=retrieve_subject, top_k=per_subject_top_k)
                raw_docs = result.get("docs", []) or []
                used_docs = raw_docs[:per_subject_top_k]
                subject_mismatch_count = _subject_mismatch_count(used_docs, retrieve_subject)
                branch_eval = _evaluate_retrieval_branch(
                    subject=subject,
                    role=role,
                    docs=used_docs,
                    is_hit=bool(result.get("is_hit", False)),
                    subject_mismatch_count=subject_mismatch_count,
                    reranker_failed=bool(result.get("reranker_failed")),
                )
                local_docs: list[dict] = []
                for doc in used_docs:
                    local_docs.append({
                        "type": "rag",
                        **doc,
                        "retrieval_subject": subject,
                        "retrieval_role": role,
                        "retrieval_query": rag_query,
                        "retrieval_purpose": branch.get("purpose", ""),
                        "relation_to_goal": branch.get("relation_to_goal", ""),
                        "retrieval_priority": _clamp_priority(branch.get("priority", 0.5)),
                        "branch_status": branch_eval["branch_status"],
                        "weak_reason": branch_eval["weak_reason"],
                        "best_rerank_score": branch_eval["best_rerank_score"],
                        "needs_supplement": branch_eval["needs_supplement"],
                        "branch_status_score_source": branch_eval["branch_status_score_source"],
                        "reranker_failed": branch_eval["reranker_failed"],
                    })
                local_candidates = _build_local_evidence_candidates(
                    docs=local_docs,
                    subject=subject,
                    role=role,
                    branch_status=branch_eval["branch_status"],
                    branch_status_score_source=branch_eval["branch_status_score_source"],
                )
                for candidate, original in zip(local_candidates, local_docs):
                    local_candidates_all.append(candidate)
                    originals[candidate.evidence_id] = original
                emit_a3_trace(
                    logger,
                    "rag_retrieve_plan_item",
                    {
                        "branch_mode": "dual_source_evidence",
                        "subject": subject,
                        "role": role,
                        "priority": branch.get("priority", 0.5),
                        "query": rag_query,
                        "top_k": per_subject_top_k,
                        "raw_doc_count": len(raw_docs),
                        "used_doc_count": len(used_docs),
                        "doc_count": len(used_docs),
                        "is_hit": result.get("is_hit", False),
                        "subject_mismatch_count": subject_mismatch_count,
                        "branch_status": branch_eval["branch_status"],
                        "weak_reason": branch_eval["weak_reason"],
                        "best_rerank_score": branch_eval["best_rerank_score"],
                        "needs_supplement": branch_eval["needs_supplement"],
                        "branch_status_score_source": branch_eval["branch_status_score_source"],
                        "reranker_failed": branch_eval["reranker_failed"],
                        "top_docs": _top_doc_summaries(used_docs),
                    },
                    state=state,
                    env_flag="LOG_RAG_RESULT",
                )

    candidates = _cap_evidence_candidates(local_candidates_all)
    emit_a3_trace(
        logger,
        "local_evidence_candidate_build",
        {
            "branch_mode": "dual_source_evidence",
            "local_candidate_count": len(candidates),
            "subjects": sorted({candidate.subject for candidate in candidates if candidate.subject}),
            "candidate_preview": [
                {
                    "evidence_id": candidate.evidence_id,
                    "source_type": candidate.source_type,
                    "subject": candidate.subject,
                    "rerank_score": candidate.rerank_score,
                    "tavily_score": candidate.tavily_score,
                    "source": candidate.source,
                    "url": candidate.url,
                }
                for candidate in candidates[:10]
            ],
        },
        state=state,
        env_flag="LOG_RAG_RESULT",
    )

    return {
        "local_evidence_candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        "local_evidence_originals": {
            candidate.evidence_id: originals[candidate.evidence_id]
            for candidate in candidates
            if candidate.evidence_id in originals
        },
        "retrieval_branch_mode": branch_debug.get("mode", ""),
    }


async def _web_search_dual_source(state: LearningState, branches: list[dict], branch_debug: dict) -> dict:
    original_user_query = _last_human_query(state)
    web_enabled = bool(_retrieval_setting("web.enabled", True))
    web_candidates_all: list[EvidenceCandidate] = []
    originals: dict[str, dict] = {}

    if not web_enabled:
        emit_a3_trace(
            logger,
            "web_search",
            {
                "branch_mode": "dual_source_evidence",
                "skipped": True,
                "skip_reason": "retrieval_web_disabled",
                "provider": "tavily",
                "result_count": 0,
                "used_result_count": 0,
            },
            state=state,
            env_flag="LOG_WEB_SEARCH_RESULT",
        )
        return {
            "web_evidence_candidates": [],
            "web_evidence_originals": {},
        }

    for branch_index, branch in enumerate(branches):
        subject = str(branch.get("subject") or "")
        role = str(branch.get("role") or "supporting_context")
        web_query, query_source = _dual_source_web_query(state, branch)
        web_results, diagnostics = await _run_dual_source_first_round_web(
            state=state,
            branch=branch,
            query=web_query,
            original_user_query=original_user_query,
        )
        emit_a3_trace(
            logger,
            "web_search",
            {
                "branch_mode": "dual_source_evidence",
                "subject": subject,
                "role": role,
                "query_source": query_source,
                "query": web_query,
                "provider": diagnostics.get("provider", "tavily"),
                "ok": diagnostics.get("ok", False),
                "result_count": diagnostics.get("result_count", len(web_results)),
                "used_result_count": len(web_results),
                "status_code": diagnostics.get("status_code"),
                "elapsed_ms": diagnostics.get("elapsed_ms"),
                "error_type": diagnostics.get("error_type", ""),
                "error_message": diagnostics.get("error_message", ""),
                "search_result_judge_disabled_by_dual_source": True,
            },
            state=state,
            env_flag="LOG_WEB_SEARCH_RESULT",
        )
        web_candidates = _build_web_evidence_candidates(
            tavily_results=web_results,
            subject=subject,
            role=role,
            purpose=str(branch.get("purpose") or "first_round_dual_source"),
            query=web_query,
            attempt_index=branch_index,
        )
        for candidate, original in zip(web_candidates, web_results):
            web_candidates_all.append(candidate)
            originals[candidate.evidence_id] = original

    candidates = _cap_evidence_candidates(web_candidates_all)
    emit_a3_trace(
        logger,
        "web_evidence_candidate_build",
        {
            "branch_mode": "dual_source_evidence",
            "web_candidate_count": len(candidates),
            "subjects": sorted({candidate.subject for candidate in candidates if candidate.subject}),
            "candidate_preview": [
                {
                    "evidence_id": candidate.evidence_id,
                    "source_type": candidate.source_type,
                    "subject": candidate.subject,
                    "tavily_score": candidate.tavily_score,
                    "source": candidate.source,
                    "url": candidate.url,
                }
                for candidate in candidates[:10]
            ],
        },
        state=state,
        env_flag="LOG_WEB_SEARCH_RESULT",
    )
    return {
        "web_evidence_candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        "web_evidence_originals": {
            candidate.evidence_id: originals[candidate.evidence_id]
            for candidate in candidates
            if candidate.evidence_id in originals
        },
    }


@traced_node
async def evidence_judge(state: LearningState) -> dict:
    """Barrier fan-in: judge local and web candidates, then assemble final context."""
    original_user_query = _last_human_query(state)
    local_candidate_dicts = state.get("local_evidence_candidates") or []
    web_candidate_dicts = state.get("web_evidence_candidates") or []
    candidates = [
        EvidenceCandidate.model_validate(item)
        for item in [*local_candidate_dicts, *web_candidate_dicts]
    ]
    candidates = _cap_evidence_candidates(candidates)
    local_originals = dict(state.get("local_evidence_originals") or {})
    web_originals = dict(state.get("web_evidence_originals") or {})
    all_originals = {**local_originals, **web_originals}
    originals = {candidate.evidence_id: all_originals.get(candidate.evidence_id, {}) for candidate in candidates}

    emit_a3_trace(
        logger,
        "evidence_candidate_build",
        {
            "branch_mode": "dual_source_evidence",
            "candidate_count": len(candidates),
            "local_candidate_count": sum(1 for candidate in candidates if candidate.source_type == "local_rag"),
            "web_candidate_count": sum(1 for candidate in candidates if candidate.source_type == "web"),
            "subjects": sorted({candidate.subject for candidate in candidates if candidate.subject}),
            "source_type_distribution": dict(Counter(candidate.source_type for candidate in candidates)),
            "candidate_preview": [
                {
                    "evidence_id": candidate.evidence_id,
                    "source_type": candidate.source_type,
                    "subject": candidate.subject,
                    "rerank_score": candidate.rerank_score,
                    "tavily_score": candidate.tavily_score,
                    "source": candidate.source,
                    "url": candidate.url,
                }
                for candidate in candidates[:10]
            ],
        },
        state=state,
        env_flag="LOG_RAG_RESULT",
    )

    try:
        parsed, judge_debug = await _judge_evidence_candidates_with_llm(
            state=state,
            candidates=candidates,
            original_user_query=original_user_query,
            learning_goal=str(state.get("learning_goal", "")),
            requested_resource_type=str(state.get("requested_resource_type", "")),
            round_index=1,
        )
    except StructuredOutputError as exc:
        parsed, judge_debug = _fallback_evidence_judge_result(
            state=state,
            candidates=candidates,
            reason=(
                exc.result.business_validation_error
                or exc.result.validation_error
                or exc.result.parsing_error
                or exc.result.error_message
                or str(exc)
            ),
            base_debug=_structured_result_to_evidence_failure_debug(
                result=exc.result,
                original_user_query=original_user_query,
                candidates=candidates,
            ),
        )

    if parsed is None:
        parsed, judge_debug = _fallback_evidence_judge_result(
            state=state,
            candidates=candidates,
            reason=(
                "Evidence Judge returned no parsed result: "
                f"{judge_debug.get('failure_phase', 'unknown')}"
            ),
            base_debug=judge_debug,
        )

    fallback_used = bool(judge_debug.get("fallback_used"))
    context_docs = _select_judged_context(parsed=parsed, candidates=candidates, originals=originals)
    followups = _followups_from_coverage_gaps(parsed)
    refinement_needed = bool(parsed.need_more_web_search or followups)
    refinement_deferred = refinement_needed and bool(_retrieval_setting("evidence_refinement.reserved", True))
    deferred_reason = "search_optimization_loop_not_implemented_in_this_phase" if refinement_deferred else ""
    emit_a3_trace(
        logger,
        "evidence_refinement_reserved",
        {
            "reserved": True,
            "search_refinement_needed": refinement_needed,
            "search_refinement_deferred": refinement_deferred,
            "deferred_reason": deferred_reason,
            "evidence_judge_state": parsed.overall_evidence_state,
            "coverage_gap_count": len(parsed.coverage_gaps),
            "proposed_followup_search_queries": followups,
        },
        state=state,
        env_flag="LOG_RAG_RESULT",
    )

    web_context_docs = _web_evidence_items(context_docs)
    local_context_docs = [doc for doc in context_docs if doc.get("source_type") == "local_rag"]
    web_evidence_count = len(web_context_docs)
    web_failed = bool(web_candidate_dicts and not web_context_docs)
    emit_a3_trace(
        logger,
        "context_assembly",
        {
            "mode": "dual_source_evidence",
            "final_doc_count": len(context_docs),
            "local_rag_context_count": len(local_context_docs),
            "web_context_count": web_evidence_count,
            "evidence_judge_state": parsed.overall_evidence_state,
            "evidence_judge_rounds": 1,
            "source_type_distribution": _source_distribution(context_docs),
            "search_refinement_needed": refinement_needed,
            "search_refinement_deferred": refinement_deferred,
            "search_optimization_reserved": True,
            "web_evidence_count": web_evidence_count,
            "web_supplement_count": web_evidence_count,
            "web_supplement_provider": "tavily",
            "web_supplement_failed": web_failed,
            "evidence_candidate_count": len(candidates),
        },
        state=state,
        env_flag="LOG_CONTEXT_ASSEMBLY",
    )

    # ── Evidence memory ──────────────────────────────────────────────
    request_id = state.get("request_id", "")
    thread_id = state.get("thread_id", "")
    new_evidence, new_gaps = build_evidence_memory_summary(
        state=state,
        parsed=parsed,
        request_id=request_id,
        thread_id=thread_id,
    )

    # ── Controlled stop logic ────────────────────────────────────────
    evidence_state = parsed.overall_evidence_state
    controlled_stop = False
    controlled_stop_reason = ""
    degraded_generation = False
    degraded_reason = ""

    fail_fast_on_insufficient = bool(
        get_setting("retrieval.evidence_memory.fail_fast_on_insufficient_evidence", False)
    )

    if fallback_used:
        degraded_generation = True
        degraded_reason = "Evidence Judge validation failed; fallback evidence selection was used."
    elif evidence_state == "insufficient":
        if fail_fast_on_insufficient:
            raise RuntimeError(
                "Evidence Judge declared evidence insufficient and "
                "fail_fast_on_insufficient_evidence is enabled."
            )
        controlled_stop = True
        controlled_stop_reason = "evidence_insufficient"
    elif evidence_state == "partially_sufficient":
        degraded_generation = True
        degraded_reason = "evidence_partially_sufficient"

    return {
        "context": context_docs,
        "evidence_candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        "evidence_judge_output": parsed.model_dump(mode="json"),
        "evidence_judge_rounds": 1,
        "evidence_judge_state": evidence_state,
        "evidence_coverage_gaps": []
        if fallback_used
        else [gap.model_dump(mode="json") for gap in parsed.coverage_gaps],
        "search_refinement_needed": refinement_needed,
        "search_refinement_deferred": refinement_deferred,
        "search_refinement_deferred_reason": deferred_reason,
        "proposed_followup_search_queries": followups,
        "search_optimization_reserved": True,
        "search_optimization_status": "reserved_not_implemented",
        "dual_source_mode": True,
        "evidence_judge_failed": fallback_used,
        "degraded_generation": degraded_generation,
        "degraded_reason": degraded_reason,
        "evidence_controlled_stop": controlled_stop,
        "evidence_controlled_stop_reason": controlled_stop_reason,
        "evidence_summary_memory": new_evidence,
        "evidence_gap_memory": new_gaps,
        "web_supplement_provider": "tavily",
        "web_supplement_results": web_context_docs,
        "web_evidence_count": web_evidence_count,
        "web_supplement_count": web_evidence_count,
        "web_supplement_failed": web_failed,
        "web_supplement_failure_reason": "judge_rejected_all_or_no_web_kept" if web_failed else "",
        "web_supplement_status_by_subject": {},
        "web_supplement_success_subjects": sorted({doc.get("subject") for doc in web_context_docs if doc.get("subject")}),
        "web_supplement_failed_subjects": [],
        "web_supplement_partial_failed": False,
        "web_judge_provider": _evidence_judge_provider(),
        "web_judge_model": _evidence_judge_model(),
        "web_judge_failed_subjects": [],
        "web_judge_rejected_all_subjects": [],
        "coverage_decision_summary": parsed.decision_summary,
    }


# ── Evidence memory builder ────────────────────────────────────────────────

def build_evidence_memory_summary(
    *,
    state: LearningState,
    parsed: EvidenceJudgeOutput,
    request_id: str,
    thread_id: str,
    round_index: int = 1,
) -> tuple[list[dict], list[dict]]:
    """Build compact evidence memory and gap memory entries.

    Includes selector-facing fields: subject, resource_type, summary,
    decision_summary, evidence_state, followup_search_queries, and
    kept_evidence_summary with short safe metadata only.

    Never stores raw docs, content, full context, full historical
    answers, or raw originals.
    Returns (new_evidence_entries, new_gap_entries).
    """
    memory_id = f"{thread_id}:{request_id}:evidence_judge_round_{round_index}"
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # ── kept_evidence_summary: short safe metadata only ──────────────
    kept_evidence_summary: list[dict] = []
    judged_by_id: dict[str, EvidenceJudgeItem] = {}
    for item in parsed.judged_evidence:
        judged_by_id[item.evidence_id] = item

    candidates = state.get("evidence_candidates") or []
    originals = {**state.get("local_evidence_originals", {}), **state.get("web_evidence_originals", {})}
    candidate_by_id: dict[str, dict] = {}
    for c in candidates:
        cid = c.get("evidence_id", "")
        if cid:
            candidate_by_id[cid] = c

    for eid, judge_item in judged_by_id.items():
        if not judge_item.keep:
            continue
        candidate = candidate_by_id.get(eid, {})
        source = candidate.get("source", "")
        url = candidate.get("url", "")
        # Pull source/url from originals if not in candidate
        if (not source or not url) and eid in originals:
            orig = originals[eid]
            source = source or orig.get("source", "")
            url = url or orig.get("url", "")
        kept_evidence_summary.append({
            "evidence_id": eid,
            "subject": candidate.get("subject", ""),
            "source_type": candidate.get("source_type", ""),
            "source": source[:300] if source else "",
            "url": url[:500] if url else "",
            "final_quality": judge_item.final_quality,
            "use_case": judge_item.use_case,
            "short_summary": (judge_item.reason or "")[:200],
        })

    # ── followup queries from coverage gaps ──────────────────────────
    followup_queries: list[str] = []
    for gap in parsed.coverage_gaps:
        q = gap.suggested_search_query.strip()
        if q and q not in followup_queries:
            followup_queries.append(q)

    decision_summary_text = (parsed.decision_summary or "")[:1000]
    summary_text = decision_summary_text

    evidence_entry = {
        "memory_id": memory_id,
        "created_at": created_at,
        "request_id": request_id,
        "thread_id": thread_id,
        "evidence_judge_round": round_index,
        # ── Selector-facing fields ────────────────────────────────
        "subject": state.get("subject", ""),
        "requested_resource_type": state.get("requested_resource_type", ""),
        "resource_type": state.get("requested_resource_type", ""),
        "summary": summary_text,
        "decision_summary": decision_summary_text,
        "evidence_state": parsed.overall_evidence_state,
        "overall_evidence_state": parsed.overall_evidence_state,
        "need_more_web_search": parsed.need_more_web_search,
        "coverage_gap_count": len(parsed.coverage_gaps),
        "followup_search_queries": followup_queries,
        "evidence_count": len(parsed.judged_evidence),
        "kept_count": sum(1 for item in parsed.judged_evidence if item.keep),
        # ── Compact metadata only (no raw docs/content) ──────────
        "kept_evidence_summary": kept_evidence_summary,
    }

    gap_entries: list[dict] = []
    for gap in parsed.coverage_gaps:
        gap_entries.append({
            "memory_id": f"{memory_id}:gap:{gap.subject}:{gap.role}",
            "created_at": created_at,
            "request_id": request_id,
            "thread_id": thread_id,
            "subject": gap.subject,
            "role": gap.role,
            "gap": gap.gap,
            "suggested_search_query": gap.suggested_search_query,
            "purpose": gap.purpose,
            "priority": gap.priority,
        })

    emit_a3_trace(
        logger,
        "evidence_memory_summary_build",
        {
            "evidence_state": parsed.overall_evidence_state,
            "kept_count": evidence_entry["kept_count"],
            "gap_count": len(gap_entries),
            "summary_chars": len(summary_text),
            "memory_id": memory_id,
            "persisted": True,
        },
        state=state,
        env_flag="LOG_A3_TRACE",
    )

    return [evidence_entry], gap_entries


# ── Evidence summary output (controlled stop) ──────────────────────────────

def _render_evidence_summary_output(state: LearningState) -> str:
    """Render a short Markdown output when evidence is insufficient."""
    gaps = state.get("evidence_coverage_gaps") or []
    judge_output = state.get("evidence_judge_output") or {}
    decision = judge_output.get("decision_summary", "") or "证据不足，无法提供完整回答。"

    lines = [
        "## 📋 证据检索摘要",
        "",
        f"**状态**: {decision}",
        "",
    ]

    kept_ids = []
    evidence_candidates = state.get("evidence_candidates") or []
    for candidate in evidence_candidates:
        eid = candidate.get("evidence_id", "")
        if candidate.get("keep"):
            kept_ids.append(eid)

    if kept_ids:
        lines.append(f"**已保存的证据**: {len(kept_ids)} 条")
        lines.append("")

    if gaps:
        lines.append("### 🔍 发现的覆盖缺口")
        lines.append("")
        for gap in gaps[:5]:
            lines.append(f"- **{gap.get('subject', '')}** ({gap.get('role', '')}): {gap.get('gap', '')}")
        lines.append("")

    lines.append("### 💡 建议的后续搜索")
    followups = state.get("proposed_followup_search_queries") or []
    for fq in followups[:5]:
        q = fq.get("query", "") or fq.get("suggested_search_query", "")
        if q:
            lines.append(f"- `{q}`")

    lines.append("")
    lines.append("> ℹ️ 已保存当前证据摘要，您可以尝试更具体的问题或稍后重试。")

    return "\n".join(lines)


@traced_node
async def evidence_summary_output(state: LearningState) -> dict:
    """Controlled stop: emit summary when evidence is insufficient.

    This is a successful controlled stop, NOT a server error.
    Returns messages so the frontend displays it as a normal response,
    with metadata marking it as a controlled stop.
    """
    markdown = _render_evidence_summary_output(state)
    return {
        "plan": markdown,
        "messages": [AIMessage(content=markdown)],
        "evidence_controlled_stop": True,
        "final_response_type": "evidence_summary",
        "evidence_controlled_stop_reason": state.get("evidence_controlled_stop_reason", "evidence_insufficient"),
    }


# ── Node 0a: academic router ──────────────────────────────────────────────

@traced_node
async def academic_router(state: LearningState) -> dict:
    """Router node for parallel fan-out.

    Clears context on retry path only — NOT on new requests (that is
    handled by initial_request_reset_transient_state at /stream entry).
    """
    if _is_retry_rewrite_active(state):
        return {"context": CONTEXT_CLEAR}
    return {}


# ── Node 0b: query rewriting (retry path only, fail-fast) ─────────────────

@traced_node
async def rewrite_query(state: LearningState) -> dict:
    """Rewrite the user's query using hallucination feedback.

    Uses invoke_plain_llm_fail_fast — on failure, raises instead of
    falling back to the original query.  Does NOT clear persistent
    state or current judged context via CONTEXT_CLEAR; that is the
    academic_router's responsibility on the retry path.
    """
    from src.graph.llm import invoke_plain_llm_fail_fast

    original_query = _last_human_query(state)
    reason = state.get("hallucination_reason", "")
    retry_count = state.get("retry_count", 0)

    rewrite_prompt = load_prompt("rewrite_query").format(
        original_query=original_query,
        hallucination_reason=reason,
    )

    try:
        rewritten = await invoke_plain_llm_fail_fast(
            node_name="rewrite_query",
            llm_node="supervisor",
            messages=[
                SystemMessage(content="你是一个查询改写助手。根据反馈改进用户的搜索查询。"),
                HumanMessage(content=rewrite_prompt),
            ],
            state=state,
        )
    except Exception as exc:
        emit_a3_trace(
            logger,
            "rewrite_query_retry_failed",
            {
                "fallback_used": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:2000],
                "retry_count": retry_count,
                "hallucination_reason": reason,
            },
            state=state,
            env_flag="LOG_RETRY_TRACE",
        )
        raise

    emit_a3_trace(
        logger,
        "rewrite_query_retry",
        {
            "retry_count": retry_count,
            "hallucination_reason": reason,
            "rewritten_query": rewritten,
            "fallback_used": False,
        },
        state=state,
        env_flag="LOG_RETRY_TRACE",
    )

    # rewritten_query is diagnostic only; actual retrieval uses
    # search_rag_query / search_web_query.
    cleared = _clear_retrieval_plan_state()
    return {
        "rewritten_query": rewritten,
        "search_rag_query": rewritten,
        "search_web_query": rewritten,
        "retrieval_plan": [],
        **{k: v for k, v in cleared.items() if k not in ("retrieval_plan",)},
    }


# ── Node 0c: initial search-query rewriting ───────────────────────────────

async def _maintain_conversation_summary(state: LearningState) -> str:
    """Update the compact conversation summary before query rewrite.

    Only runs when the message history is long enough to justify
    summarization.  Returns the updated summary text.
    """
    messages = state.get("messages") or []
    existing_summary = str(state.get("conversation_summary") or "").strip()

    # Only summarize if we have enough messages
    human_messages = [
        m for m in messages
        if isinstance(m, HumanMessage)
        or (isinstance(m, dict) and m.get("type") == "human")
    ]
    if len(human_messages) < 2:
        return existing_summary or ""

    # Build a compact prompt for the LLM
    recent_texts: list[str] = []
    for m in messages[-10:]:
        content = ""
        if isinstance(m, HumanMessage):
            content = str(m.content or "")
        elif isinstance(m, AIMessage):
            content = str(m.content or "")[:200]
        elif isinstance(m, dict):
            content = str(m.get("content", ""))
            if m.get("type") == "ai":
                content = content[:200]
        if content.strip():
            role = "用户" if (isinstance(m, HumanMessage) or (isinstance(m, dict) and m.get("type") == "human")) else "助手"
            recent_texts.append(f"{role}: {content.strip()[:300]}")

    if not recent_texts:
        return existing_summary or ""

    try:
        from src.graph.llm import invoke_plain_llm_fail_fast

        prompt = (
            "将以下对话总结为一段简洁的中文摘要（不超过200字）。"
            "保留用户的学习目标和关键话题，忽略闲聊。\n\n"
            + ("现有摘要: " + existing_summary + "\n\n" if existing_summary else "")
            + "\n".join(recent_texts[-8:])
        )
        summary = await invoke_plain_llm_fail_fast(
            node_name="conversation_summary",
            llm_node="supervisor",
            messages=[HumanMessage(content=prompt)],
            state=state,
            temperature=0.0,
            max_raw_chars=800,
        )
        result = summary.strip()[:500]
        emit_a3_trace(
            logger,
            "conversation_summary",
            {
                "success": True,
                "summary_chars": len(result),
            },
            state=state,
            env_flag="LOG_A3_TRACE",
        )
        return result
    except Exception as exc:
        emit_a3_trace(
            logger,
            "conversation_summary",
            {
                "success": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
                "action": "keep_existing_summary",
                "fallback_used": False,
                "enhancement_only": True,
                "summary_chars": len(existing_summary),
            },
            state=state,
            env_flag="LOG_A3_TRACE",
        )
        fail_fast = bool(get_setting("development.fail_fast_conversation_summary", False))
        if fail_fast:
            raise
        return existing_summary or ""


@traced_node
async def search_query_rewriter(state: LearningState) -> dict:
    """Rewrite the original request into RAG and web-search queries.

    Query rewrite runs for every new request — stale rewritten_query from
    a previous turn does NOT skip it.
    """
    original_query = _last_human_query(state)
    keypoints = state.get("keypoints", [])
    requested_resource_type = state.get("requested_resource_type", "")
    subject = state.get("subject", "")
    subject_candidates = state.get("subject_candidates", [])
    available_subjects = get_available_subjects_from_data()

    # Maintain conversation summary before query rewrite
    conversation_summary = await _maintain_conversation_summary(state)

    # Select compact memory summaries — never full history
    selected_memories = select_relevant_memory_summaries(
        state,
        current_query=original_query,
        subject=subject,
        requested_resource_type=requested_resource_type,
    )

    prompt = _render_prompt(
        "search_query_rewriter",
        {
            "question": original_query,
            "keypoints": "、".join(keypoints) if keypoints else "未提取到明确关键词",
            "requested_resource_type": requested_resource_type or "none",
            "subject": subject or "other",
            "subject_candidates": "、".join(subject_candidates) if subject_candidates else "无",
            "available_subjects": "、".join(available_subjects) if available_subjects else "无",
            "conversation_summary": conversation_summary or "无",
            "evidence_memory_summaries": json.dumps(
                [
                    {
                        "summary": m.get("summary", ""),
                        "subject": m.get("subject", ""),
                        "resource_type": m.get("resource_type", ""),
                    }
                    for m in selected_memories
                ],
                ensure_ascii=False,
            ) if selected_memories else "无",
        },
    )
    messages = [
        SystemMessage(content="You are a retrieval query rewriter for a university learning agent. Return only schema-valid JSON. Current user query is highest priority. Conversation/evidence memory is optional background only. Never rewrite a current request into an old topic because of history. If current query says one topic and memory contains another, follow the current query unless the user explicitly asks to connect them."),
        HumanMessage(content=prompt),
    ]

    raw_preview = ""
    parsing_error = None
    try:
        with traced_llm_call(
            model_name=get_setting("query_rewrite.model", os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")),
            node_name="search_query_rewriter",
            temperature=0.0,
        ):
            structured_result = await invoke_structured_llm(
                node_name="search_query_rewriter",
                llm_node="query_rewrite",
                schema=SearchQueryRewriteOutput,
                messages=messages,
                output_mode=get_llm_output_mode("search_query_rewriter"),
                fallback_modes=get_fallback_modes("search_query_rewriter"),
                business_validator=lambda p: validate_search_query_rewrite_output(
                    p, current_query=original_query
                ),
                state=state,
                max_raw_chars=get_max_raw_chars("search_query_rewriter"),
            )
        parsed = structured_result.parsed
        if not isinstance(parsed, SearchQueryRewriteOutput):
            raise TypeError("search_query_rewriter parsed result is not SearchQueryRewriteOutput")
        raw_preview = structured_result.raw_output[:2000] if structured_result.raw_output else ""

        # ── Memory use trace ─────────────────────────────────────────
        history_ref = _has_explicit_history_reference(original_query)
        has_reason = bool((parsed.memory_use_reason or "").strip())
        if parsed.memory_used_for_retrieval:
            if history_ref:
                action = "allow"
            elif has_reason:
                action = "allow_by_llm_reason"
            else:
                action = "reject"
        else:
            action = "background_only"
        emit_a3_trace(
            logger,
            "query_rewrite_memory_use",
            {
                "memory_count": len(selected_memories),
                "memory_used_for_retrieval": parsed.memory_used_for_retrieval,
                "memory_use_reason": parsed.memory_use_reason,
                "current_query_has_history_reference": history_ref,
                "action": action,
            },
            state=state,
            env_flag="LOG_A3_TRACE",
        )

        result_payload = {
            "rag_query": parsed.rag_query.strip(),
            "web_search_query": parsed.web_search_query.strip(),
            "expanded_keypoints": [
                str(item).strip()
                for item in parsed.expanded_keypoints
                if str(item).strip()
            ],
            "reason": parsed.reason.strip(),
        }
        retrieval_plan, normalize_debug = _normalize_retrieval_plan(parsed.retrieval_plan, state)
        primary_subject = _normalize_primary_subject(parsed.primary_subject, retrieval_plan)

        # ── Subject conflict fail-fast ──────────────────────────────────
        _maybe_fail_subject_conflict(
            parsed_primary=parsed.primary_subject,
            normalized_primary=primary_subject,
            supervisor_subject=subject,
            available_subjects=available_subjects,
            retrieval_plan=retrieval_plan,
        )

        multi_subject_payload = {
            "retrieval_plan": retrieval_plan,
            "learning_goal": parsed.learning_goal.strip(),
            "primary_subject": primary_subject,
            "subject_relation_summary": parsed.subject_relation_summary.strip(),
        }

        # TEMP A3_TRACE: remove after multi-subject retrieval validation.
        emit_a3_trace(
            logger,
            "query_rewrite",
            {
                "intent": state.get("intent"),
                "subject": state.get("subject"),
                "subject_candidates": state.get("subject_candidates", []),
                "available_subjects": available_subjects,
                "learning_goal": parsed.learning_goal,
                "primary_subject": primary_subject,
                "subject_relation_summary": parsed.subject_relation_summary,
                "search_rag_query": result_payload["rag_query"],
                "search_web_query": result_payload["web_search_query"],
                "expanded_keypoints": result_payload["expanded_keypoints"],
                "retrieval_plan_count": len(retrieval_plan),
                "retrieval_plan": retrieval_plan,
                "reason": result_payload["reason"],
                "parsing_error": str(parsing_error) if parsing_error else None,
                "raw_preview": raw_preview,
            },
            state=state,
            env_flag="LOG_QUERY_REWRITE_RESULT",
            max_chars=800,
        )
        # TEMP A3_TRACE: remove after multi-subject retrieval validation.
        emit_a3_trace(
            logger,
            "retrieval_plan_normalize",
            {
                "available_subjects": available_subjects,
                "subject_candidates": subject_candidates,
                "raw_plan_count": normalize_debug["raw_plan_count"],
                "normalized_plan_count": normalize_debug["normalized_plan_count"],
                "accepted_subjects": normalize_debug["accepted_subjects"],
                "rejected_items": normalize_debug["rejected_items"],
                "primary_subject": primary_subject,
            },
            state=state,
            env_flag="LOG_RETRIEVAL_PLAN",
        )
    except Exception as exc:
        logger.exception("Initial search query rewrite failed; fallback disabled")
        # TEMP A3_TRACE: remove after multi-subject retrieval validation.
        emit_a3_trace(
            logger,
            "query_rewrite_failed",
            {
                "error": str(exc),
                "fallback": "disabled_fail_fast_structured_output",
                "retrieval_plan": [],
                "learning_goal": "",
                "primary_subject": "",
                "subject_relation_summary": "",
                "raw_preview": raw_preview,
            },
            state=state,
            env_flag="LOG_QUERY_REWRITE_RESULT",
        )
        raise


    return {
        "search_rag_query": result_payload["rag_query"],
        "search_web_query": result_payload["web_search_query"],
        "expanded_keypoints": result_payload["expanded_keypoints"],
        "search_query_rewrite_reason": result_payload["reason"],
        "search_query_rewrite_error": "",
        "search_query_rewrite_raw_preview": raw_preview,
        "conversation_summary": conversation_summary,
        **multi_subject_payload,
    }


# ── Node 1: RAG retrieval (parallel branch A) ─────────────────────

@traced_node
async def rag_retrieve(state: LearningState) -> dict:
    """Retrieve local course evidence, then run branch-aware Web supplement when needed."""
    branches, branch_debug = _build_retrieval_branches(state)
    branch_mode = branch_debug.get("mode", "unknown")
    per_subject_top_k = int(get_setting("rag.multi_subject_per_subject_top_k", 3))
    max_docs = int(get_setting("rag.multi_subject_max_docs", 8))

    # TEMP A3_TRACE: remove after multi-subject retrieval validation.
    emit_a3_trace(
        logger,
        "retrieval_branch_build",
        branch_debug,
        state=state,
        env_flag="LOG_RAG_RESULT",
    )

    if _dual_source_enabled():
        return await _rag_retrieve_dual_source(state, branches, branch_debug)

    if not branches:
        query, query_source = _query_source(state)
        subj = state.get("subject") if state.get("subject") != "other" else None
        with traced_retrieval(query=query, subject=subj) as span:
            result = await asyncio.to_thread(retrieve, query=query, subject=subj)
            raw_docs = result.get("docs", []) or []
            mismatch_count = _subject_mismatch_count(raw_docs, subj)
            branch_eval = _evaluate_retrieval_branch(
                subject=str(subj or ""),
                role="core_concept",
                docs=raw_docs,
                is_hit=result.get("is_hit", False),
                subject_mismatch_count=mismatch_count,
                reranker_failed=bool(result.get("reranker_failed")),
            )
            # TEMP A3_TRACE: remove after multi-subject retrieval validation.
            emit_a3_trace(
                logger,
                "rag_retrieve_single_subject",
                {
                    "subject": subj,
                    "query": query,
                    "query_source": query_source,
                    "raw_doc_count": len(raw_docs),
                    "used_doc_count": len(raw_docs),
                    "doc_count": len(raw_docs),
                    "is_hit": result.get("is_hit", False),
                    "subject_mismatch_count": mismatch_count,
                    "branch_status": branch_eval["branch_status"],
                    "weak_reason": branch_eval["weak_reason"],
                    "best_rerank_score": branch_eval["best_rerank_score"],
                    "branch_status_score_source": branch_eval["branch_status_score_source"],
                    "reranker_failed": branch_eval["reranker_failed"],
                    "top_docs": _top_doc_summaries(raw_docs),
                },
                state=state,
                env_flag="LOG_RAG_RESULT",
            )
            span.set_attribute("rag.doc_count", len(raw_docs))
            span.set_attribute("rag.is_hit", result.get("is_hit", False))
        return {"context": [{"type": "rag", **doc} for doc in raw_docs]}

    subjects = [str(item.get("subject", "")) for item in branches if item.get("subject")]
    query, _query_source_name = _query_source(state)
    with traced_retrieval(query=query, subject=branch_mode) as span:
        span.set_attribute("rag.branch_mode", branch_mode)
        span.set_attribute("rag.branch_count", len(branches))
        span.set_attribute("rag.retrieval_subjects", ",".join(subjects))

        local_docs: list[dict] = []
        docs_by_subject: dict[str, list[dict]] = {}
        branch_evals: dict[str, dict] = {}

        for item in branches:
            plan_subject = str(item.get("subject") or "other")
            plan_query = str(item.get("rag_query") or "").strip()
            if not plan_query:
                continue
            retrieve_subject = None if plan_subject == "other" else plan_subject
            result = await asyncio.to_thread(
                retrieve,
                query=plan_query,
                subject=retrieve_subject,
                top_k=per_subject_top_k,
            )
            raw_docs = result.get("docs", []) or []
            used_docs = raw_docs[:per_subject_top_k]
            docs_by_subject[plan_subject] = used_docs
            role = item.get("role", "supporting_context")
            priority = item.get("priority", 0.5)
            subject_mismatch_count = _subject_mismatch_count(used_docs, retrieve_subject)
            branch_eval = _evaluate_retrieval_branch(
                subject=plan_subject,
                role=role,
                docs=used_docs,
                is_hit=result.get("is_hit", False),
                subject_mismatch_count=subject_mismatch_count,
                reranker_failed=bool(result.get("reranker_failed")),
            )
            branch_evals[plan_subject] = branch_eval

            # TEMP A3_TRACE: remove after multi-subject retrieval validation.
            emit_a3_trace(
                logger,
                "rag_retrieve_plan_item",
                {
                    "branch_mode": branch_mode,
                    "subject": plan_subject,
                    "role": role,
                    "priority": priority,
                    "query": plan_query,
                    "top_k": per_subject_top_k,
                    "raw_doc_count": len(raw_docs),
                    "used_doc_count": len(used_docs),
                    "doc_count": len(used_docs),
                    "is_hit": result.get("is_hit", False),
                    "subject_mismatch_count": subject_mismatch_count,
                    "branch_status": branch_eval["branch_status"],
                    "weak_reason": branch_eval["weak_reason"],
                    "best_rerank_score": branch_eval["best_rerank_score"],
                    "branch_status_score_source": branch_eval["branch_status_score_source"],
                    "reranker_failed": branch_eval["reranker_failed"],
                    "needs_supplement": branch_eval["needs_supplement"],
                    "top_docs": _top_doc_summaries(used_docs),
                },
                state=state,
                env_flag="LOG_RAG_RESULT",
            )

            if branch_eval["branch_status"] == "missing":
                local_docs.append({
                    "type": "rag_diagnostic",
                    "retrieval_subject": plan_subject,
                    "retrieval_role": role,
                    "retrieval_query": plan_query,
                    "retrieval_purpose": item.get("purpose", ""),
                    "relation_to_goal": item.get("relation_to_goal", ""),
                    "retrieval_priority": priority,
                    "branch_status": "missing",
                    "weak_reason": "no_docs",
                    "best_rerank_score": 0.0,
                    "branch_status_score_source": "fallback_raw_retrieval_signal",
                    "reranker_failed": bool(result.get("reranker_failed")),
                    "needs_supplement": True,
                    "content": "No effective local course material was retrieved for this subject branch.",
                    "source": "local_rag_diagnostic",
                })
                continue

            for doc in used_docs:
                local_docs.append({
                    "type": "rag",
                    "retrieval_subject": plan_subject,
                    "retrieval_role": role,
                    "retrieval_query": plan_query,
                    "retrieval_purpose": item.get("purpose", ""),
                    "relation_to_goal": item.get("relation_to_goal", ""),
                    "retrieval_priority": priority,
                    "coverage_hint": item.get("coverage_hint", ""),
                    "expected_coverage": item.get("expected_coverage", []),
                    "branch_status": branch_eval["branch_status"],
                    "weak_reason": branch_eval["weak_reason"],
                    "best_rerank_score": branch_eval["best_rerank_score"],
                    "branch_status_score_source": branch_eval["branch_status_score_source"],
                    "reranker_failed": branch_eval["reranker_failed"],
                    "needs_supplement": branch_eval["needs_supplement"],
                    **doc,
                })

        targets, decision_debug = await _decide_web_supplement_with_llm(
            state=state,
            retrieval_plan=branches,
            branch_evals=branch_evals,
            docs_by_subject=docs_by_subject,
            branch_mode=branch_mode,
        )
        # TEMP A3_TRACE: remove after multi-subject retrieval validation.
        emit_a3_trace(
            logger,
            "coverage_decision",
            {
                "branch_mode": branch_mode,
                "branch_count": len(branches),
                "selected_target_count": len(targets),
                "selected_targets": [
                    {
                        "subject": target.get("subject"),
                        "purposes": target.get("supplement_purposes", []),
                        "query_count": len(target.get("supplement_queries", [])),
                    }
                    for target in targets
                ],
                **decision_debug,
            },
            state=state,
            env_flag="LOG_WEB_SEARCH_RESULT",
        )

        web_supplement_docs, web_debug = await _run_dynamic_web_supplement(
            state=state,
            targets=targets,
            decision_debug=decision_debug,
            branch_mode=branch_mode,
        )
        web_supplement_needed = bool(targets)
        web_status_by_subject = web_debug.get("status_by_subject", {})
        web_success_subjects = web_debug.get("success_subjects", [])
        web_failed_subjects = web_debug.get("failed_subjects", [])
        web_judge_failed_subjects = web_debug.get("judge_failed_subjects", [])
        web_judge_rejected_all_subjects = web_debug.get("judge_rejected_all_subjects", [])
        web_supplement_failed = web_supplement_needed and not web_success_subjects
        web_supplement_partial_failed = bool(web_success_subjects and web_failed_subjects)
        if web_supplement_failed and web_judge_failed_subjects:
            web_supplement_failure_reason = "search_result_judge_failed"
        elif web_supplement_failed and web_judge_rejected_all_subjects:
            web_supplement_failure_reason = "judge_rejected_all"
        else:
            web_supplement_failure_reason = "tavily_timeout_or_error" if web_supplement_failed else ""

        selected_local_docs, quota_debug = _select_docs_with_subject_quota(
            local_docs,
            max_docs,
            primary_subject=str(state.get("primary_subject") or ""),
        )
        selected_docs = selected_local_docs + web_supplement_docs
        subject_counter = Counter(doc.get("retrieval_subject") for doc in selected_docs)
        role_counter = Counter(doc.get("retrieval_role") for doc in selected_docs)
        web_supplement_purposes = Counter(
            doc.get("supplement_purpose") for doc in web_supplement_docs if doc.get("supplement_purpose")
        )
        web_evidence_use_cases = Counter(
            doc.get("use_case") for doc in web_supplement_docs if doc.get("use_case")
        )
        web_evidence_types = Counter(
            doc.get("evidence_type") for doc in web_supplement_docs if doc.get("evidence_type")
        )
        # TEMP A3_TRACE: remove after multi-subject retrieval validation.
        emit_a3_trace(
            logger,
            "context_assembly",
            {
                "mode": "branch_aware",
                "branch_mode": branch_mode,
                "branch_count": len(branches),
                "retrieval_plan_count": len(state.get("retrieval_plan") or []),
                "raw_doc_count": len(local_docs),
                "final_doc_count": len(selected_docs),
                "max_docs": max_docs,
                "subject_doc_distribution": dict(subject_counter),
                "role_distribution": dict(role_counter),
                "web_supplement_count": len(web_supplement_docs),
                "web_supplement_subjects": sorted({doc.get("supplement_for_subject") for doc in web_supplement_docs if doc.get("supplement_for_subject")}),
                "web_supplement_purposes": dict(web_supplement_purposes),
                "web_supplement_needed": web_supplement_needed,
                "web_supplement_provider": "tavily",
                "web_supplement_failed": web_supplement_failed,
                "web_supplement_failure_reason": web_supplement_failure_reason,
                "web_supplement_partial_failed": web_supplement_partial_failed,
                "web_supplement_status_by_subject": web_status_by_subject,
                "web_supplement_success_subjects": web_success_subjects,
                "web_supplement_failed_subjects": web_failed_subjects,
                "web_evidence_count": len(web_supplement_docs),
                "web_evidence_provider": "tavily",
                "web_judge_provider": _judge_provider(),
                "web_judge_model": _judge_model(),
                "web_judge_failed": bool(web_judge_failed_subjects),
                "web_judge_failed_subjects": web_judge_failed_subjects,
                "web_judge_rejected_all_subjects": web_judge_rejected_all_subjects,
                "web_evidence_subjects": sorted({doc.get("supplement_for_subject") for doc in web_supplement_docs if doc.get("supplement_for_subject")}),
                "web_evidence_use_cases": dict(web_evidence_use_cases),
                "web_evidence_types": dict(web_evidence_types),
                "coverage_decision_summary": decision_debug.get("decision_summary", ""),
                "dynamic_web_attempts_used": web_debug.get("attempts_used", 0),
                "dynamic_web_max_total_attempts": web_debug.get("max_total_attempts", 0),
                **quota_debug,
                "selected_docs": [
                    {
                        "type": doc.get("type"),
                        "subject": doc.get("retrieval_subject"),
                        "role": doc.get("retrieval_role"),
                        "branch_status": doc.get("branch_status"),
                        "weak_reason": doc.get("weak_reason"),
                        "supplement_purpose": doc.get("supplement_purpose"),
                        "judge_quality": doc.get("judge_quality"),
                        "judge_relevance": doc.get("judge_relevance"),
                        "evidence_type": doc.get("evidence_type"),
                        "use_case": doc.get("use_case"),
                        "source": doc.get("source"),
                        "raw_vector_score": doc.get("raw_vector_score"),
                        "raw_vector_score_source": doc.get("raw_vector_score_source"),
                        "raw_vector_score_direction": doc.get("raw_vector_score_direction"),
                        "bm25_score": doc.get("bm25_score"),
                        "bm25_score_direction": doc.get("bm25_score_direction"),
                        "rerank_score": doc.get("rerank_score"),
                        "branch_status_score_source": doc.get("branch_status_score_source"),
                        "reranker_failed": doc.get("reranker_failed"),
                    }
                    for doc in selected_docs
                ],
            },
            state=state,
            env_flag="LOG_CONTEXT_ASSEMBLY",
        )
        span.set_attribute("rag.doc_count", len(selected_docs))
        span.set_attribute("rag.is_hit", bool(selected_docs))
        if selected_docs:
            span.set_attribute("rag.top_retrieval_sort_score", _score_doc(selected_docs[0]))

    return {
        "context": selected_docs,
        "web_supplement_decisions": targets,
        "web_supplement_results": web_supplement_docs,
        "web_supplement_provider": "tavily",
        "coverage_decision_summary": decision_debug.get("decision_summary", ""),
        "retrieval_branch_mode": branch_mode,
        "web_supplement_failed": web_supplement_failed,
        "web_supplement_failure_reason": web_supplement_failure_reason,
        "web_supplement_status_by_subject": web_status_by_subject,
        "web_supplement_success_subjects": web_success_subjects,
        "web_supplement_failed_subjects": web_failed_subjects,
        "web_supplement_partial_failed": web_supplement_partial_failed,
        "web_judge_provider": _judge_provider(),
        "web_judge_model": _judge_model(),
        "web_judge_failed_subjects": web_judge_failed_subjects,
        "web_judge_rejected_all_subjects": web_judge_rejected_all_subjects,
    }

_SEARCH_TIMEOUT = _web_timeout_seconds()


@traced_node
async def web_search(state: LearningState) -> dict:
    """Fan-out web search — runs in parallel with rag_retrieve."""
    rewritten = state.get("rewritten_query", "")
    search_web_query = state.get("search_web_query", "")
    retrieval_plan = state.get("retrieval_plan") or []
    if _dual_source_enabled():
        branches, branch_debug = _build_retrieval_branches(state)
        return await _web_search_dual_source(state, branches, branch_debug)

    if (
        _web_conditional_enabled()
        and bool(_web_setting("skip_general_when_conditional", True))
        and state.get("intent") == "academic"
    ):
        branch_mode = "multi_subject_plan" if retrieval_plan else "single_subject_synthetic"
        skip_reason = (
            "dual_source_evidence_web_search_handled_in_rag_retrieve"
            if _dual_source_enabled()
            else "conditional_web_supplement_handled_in_rag_retrieve"
        )
        # TEMP A3_TRACE: remove after multi-subject retrieval validation.
        emit_a3_trace(
            logger,
            "web_search",
            {
                "query_source": "skipped_conditional_branch_mode",
                "skipped": True,
                "skip_reason": skip_reason,
                "legacy_node": True,
                "has_retrieval_plan": bool(retrieval_plan),
                "branch_mode": branch_mode,
                "retrieval_plan_count": len(retrieval_plan),
                "result_count": 0,
                "timed_out": False,
                "provider": "tavily",
                "ok": True,
                "error_type": "",
                "error_message": "",
            },
            state=state,
            env_flag="LOG_WEB_SEARCH_RESULT",
        )
        return {"context": []}

    selected_subject = ""
    if rewritten:
        query = rewritten
        query_source = "rewritten_query"
    elif search_web_query:
        query = search_web_query
        query_source = "search_web_query"
    elif retrieval_plan:
        best_item = max(
            retrieval_plan,
            key=lambda item: float(item.get("priority") or 0),
        )
        selected_subject = best_item.get("subject", "")
        query = best_item.get("web_search_query") or best_item.get("rag_query") or _last_human_query(state)
        query_source = "retrieval_plan_top_priority"
    else:
        query = _last_human_query(state)
        query_source = "original_query"

    with traced_search(query=query, timeout=_SEARCH_TIMEOUT) as span:
        diagnostics: dict = {
            "provider": "tavily",
            "query": query,
            "original_user_query": _last_human_query(state),
            "ok": False,
            "results": [],
            "result_count": 0,
            "error_type": "",
            "error_message": "",
            "raw_type": "",
            "raw_count": None,
            "elapsed_ms": None,
            "status_code": None,
        }
        try:
            diagnostics = await asyncio.wait_for(
                asyncio.to_thread(
                    web_search_fn,
                    query,
                    original_user_query=_last_human_query(state),
                    max_results=int(_web_setting("tavily.max_results", 5)),
                    timeout_seconds=_SEARCH_TIMEOUT,
                ),
                timeout=_SEARCH_TIMEOUT,
            )
            diagnostics = _coerce_web_search_diagnostics(
                diagnostics,
                query=query,
                original_user_query=_last_human_query(state),
            )
            search_results = diagnostics.get("results", [])
            span.set_attribute("search.result_count", len(search_results))
            span.set_attribute("search.timed_out", False)
        except asyncio.TimeoutError:
            diagnostics = _tavily_exception_diagnostics(
                query,
                TimeoutError(f"tavily search exceeded {_SEARCH_TIMEOUT}s"),
                original_user_query=_last_human_query(state),
                elapsed_ms=round(_SEARCH_TIMEOUT * 1000, 2),
            )
            search_results = []
            span.set_attribute("search.result_count", 0)
            span.set_attribute("search.timed_out", True)
        except Exception as exc:
            diagnostics = _tavily_exception_diagnostics(
                query,
                exc,
                original_user_query=_last_human_query(state),
            )
            search_results = []
            span.set_attribute("search.result_count", 0)
            span.set_attribute("search.timed_out", False)

    # TEMP A3_TRACE: remove after multi-subject retrieval validation.
    emit_a3_trace(
        logger,
        "web_search",
        {
            "query_source": query_source,
            "query": query,
            "original_user_query": _last_human_query(state)[:2000],
            "retrieval_plan_count": len(retrieval_plan),
            "selected_subject": selected_subject,
            "result_count": len(search_results),
            "timed_out": diagnostics.get("error_type") == "TimeoutError",
            "provider": diagnostics.get("provider", "tavily"),
            "ok": diagnostics.get("ok", False),
            "raw_type": diagnostics.get("raw_type", ""),
            "raw_count": diagnostics.get("raw_count"),
            "elapsed_ms": diagnostics.get("elapsed_ms"),
            "status_code": diagnostics.get("status_code"),
            "error_type": diagnostics.get("error_type", ""),
            "error_message": diagnostics.get("error_message", ""),
        },
        state=state,
        env_flag="LOG_WEB_SEARCH_RESULT",
    )

    return {"context": [{"type": "web", **r} for r in search_results]}


# ── Node 3: generate answer ──────────────────────────────────────

def _format_retrieval_score_note(doc: dict) -> str:
    """Format retrieval diagnostics without treating raw Chroma scores as relevance."""
    if doc.get("rerank_score") is not None:
        return f"rerank_score={doc.get('rerank_score')}"
    if doc.get("bm25_score") is not None:
        return f"bm25_score={doc.get('bm25_score')} (higher_is_better)"
    if doc.get("raw_vector_score") is not None:
        source = doc.get("raw_vector_score_source") or "chroma_similarity_search_with_score"
        direction = doc.get("raw_vector_score_direction") or "backend_specific"
        return f"raw_vector_score={doc.get('raw_vector_score')} ({source}; {direction}; not normalized relevance)"
    return "score unavailable"


def _format_retrieved(docs: list[dict]) -> str:
    if not docs:
        return "无相关参考资料。"
    if any(doc.get("type") in {"web_supplement", "web_evidence"} or doc.get("source_type") == "web" for doc in docs):
        purpose_notes = {
            "repair": "用于修补本地课程资料不足或相关性较弱的问题。",
            "coverage_expansion": "用于拓展本地课程资料之外的知识覆盖。",
            "application_context": "用于补充应用场景、行业落地或实践背景。",
            "tool_ecosystem": "用于补充工具、框架、库或技术栈生态。",
            "latest_practice": "用于补充较新的实践、趋势或前沿资料。",
            "case_example": "用于补充案例、项目或示例。",
            "implementation_detail": "用于补充代码、步骤或工程流程。",
            "planning_support": "用于补充学习路线或规划依据。",
            "resource_enrichment": "用于丰富思维导图、练习题、项目案例等学习资源素材。",
        }
        parts = []
        for d in docs:
            if d.get("type") not in {"web_supplement", "web_evidence"} and d.get("source_type") != "web":
                parts.append(_format_retrieved([d]))
                continue
            subject = d.get("supplement_for_subject") or d.get("retrieval_subject", "unknown")
            role = d.get("supplement_for_role") or d.get("retrieval_role", "supporting_context")
            purpose = d.get("supplement_purpose", "coverage_expansion")
            judge_quality = d.get("judge_quality", "unknown")
            judge_relevance = d.get("judge_relevance", "unknown")
            evidence_type = d.get("evidence_type", "unknown")
            use_case = d.get("use_case", "unknown")
            judge_reason = d.get("judge_reason", "")
            parts.append(
                f"【{subject}｜{role}｜Web 补充｜{purpose}】\n"
                "说明：以下资料用于补充该 subject 的覆盖广度、工具生态或实践背景，不属于本地课程知识库。\n"
                f"用途：{purpose_notes.get(purpose, '仅作为外部补充资料谨慎使用。')}\n"
                f"Judge: quality={judge_quality}, relevance={judge_relevance}, evidence_type={evidence_type}, use_case={use_case}\n"
                f"Judge reason: {judge_reason}\n"
                f"补充原因：{d.get('supplement_reason', '')}\n"
                f"来源：{d.get('title') or d.get('source', 'web_search')} {d.get('url', '')}\n"
                f"内容：{d.get('content', '')}"
            )
        return "\n\n".join(parts)
    if any(doc.get("retrieval_subject") for doc in docs):
        parts = []
        for i, d in enumerate(docs, 1):
            subject = d.get("retrieval_subject", "unknown")
            role = d.get("retrieval_role", "supporting_context")
            branch_status = d.get("branch_status", "usable")
            weak_reason = d.get("weak_reason", "")
            if branch_status == "weak":
                evidence_note = f"证据状态：弱证据（{weak_reason or '相关性不足'}），只能谨慎补充，不要当作强课程依据。"
            elif branch_status == "missing":
                evidence_note = f"证据状态：本地资料不足（{weak_reason or 'no_docs'}），只能说明资料缺口，不要当作课程依据。"
            elif branch_status == "strong":
                evidence_note = "证据状态：强证据，可作为核心课程依据。"
            else:
                evidence_note = "证据状态：可用证据，可作为课程依据但需结合其它资料。"
            purpose = d.get("retrieval_purpose") or "提供该学科相关课程依据"
            relation = d.get("relation_to_goal") or "与学习目标相关"
            parts.append(
                f"【{subject}｜{role}｜依据】\n"
                f"{evidence_note}\n"
                f"[{i}] 来源：{d.get('source', '未知')}（检索分数诊断：{_format_retrieval_score_note(d)}）\n"
                f"用途：{purpose}\n"
                f"关系：{relation}\n"
                f"检索 query：{d.get('retrieval_query', '')}\n"
                f"内容：{d.get('content', '')}"
            )
        return "\n\n".join(parts)

    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(f"[{i}] 来源：{d.get('source', '未知')}（检索分数诊断：{_format_retrieval_score_note(d)}）\n{d.get('content', '')}")
    return "\n\n".join(parts)


def _format_search(results: list[dict]) -> str:
    if not results:
        return "无网络搜索结果。"
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r.get('title', '无标题')} ({r.get('url', '')})\n{r.get('content', '')}")
    return "\n\n".join(parts)


_RESOURCE_OFFER_SECTION = """请在回答末尾追加以下小节。注意：这里只能询问用户是否需要继续生成资源，不能直接生成资源。

---

## 还可以继续生成的个性化学习资源

根据你刚才的问题，我还可以继续帮你生成：

1. 知识点思维导图：梳理核心概念、前置知识、易错点和实践任务；
2. 分层练习题：生成基础题、进阶题和应用题；
3. 代码实操案例：用一个小案例帮助你动手理解；
4. 学习路径建议：告诉你接下来应该按什么顺序学习。

你可以直接回复：“生成思维导图” / “生成练习题” / “生成代码案例”。
"""

_NO_RESOURCE_OFFER = "不要追加“还可以继续生成的个性化学习资源”小节，只处理用户当前明确要求的资源或问题。"


def _resource_offer_instruction(state: LearningState) -> str:
    """Return prompt instruction for optional follow-up resource offers."""
    if state.get("needs_mindmap") or state.get("requested_resource_type"):
        return _NO_RESOURCE_OFFER
    return _RESOURCE_OFFER_SECTION


@traced_node
async def generate_answer(state: LearningState) -> dict:
    """Synthesize final answer from merged context (RAG + web) via LLM."""
    question = _last_human_query(state)
    if state.get("evidence_judge_failed") and _block_generation_when_evidence_judge_failed():
        failure_output = state.get("evidence_judge_output") or {}
        failure_phase = _evidence_failure_phase(state)
        error_type = failure_output.get("error_type", "") if isinstance(failure_output, dict) else ""
        status_code = failure_output.get("status_code", "") if isinstance(failure_output, dict) else ""
        action_needed = failure_output.get("action_needed", "") if isinstance(failure_output, dict) else ""
        recommendation = failure_output.get("recommendation", "") if isinstance(failure_output, dict) else ""

        if action_needed is None:
            action_needed = ""

        if recommendation is None:
            recommendation = ""

        emit_a3_trace(
            logger,
            "generation_blocked",
            {
                "reason": "evidence_judge_failed",
                "evidence_judge_failure_phase": failure_phase,
                "error_type": error_type,
                "status_code": status_code,
                "action_needed": action_needed,
                "recommendation": recommendation,
                "context_count": len(state.get("context", [])),
                "question_preview": question[:500],
            },
            state=state,
            env_flag="LOG_GENERATION_SUMMARY",
            max_chars=2000,
        )
        diagnostic = (
            "[开发诊断] Evidence Judge 失败，已按配置阻断普通回答生成。\n\n"
            f"- failure_phase: {failure_phase or 'unknown'}\n"
            f"- error_type: {error_type or 'unknown'}\n"
            f"- status_code: {status_code or 'unknown'}\n"
            f"- action_needed: {action_needed or 'inspect evidence_judge A3_TRACE logs'}\n\n"
            "本次未使用未裁决的 local RAG 或 Tavily Web evidence 生成普通答案。"
        )
        return {"messages": [AIMessage(content=diagnostic)]}

    llm = get_node_llm("academic")

    # Split merged context by source type
    context = state.get("context", [])
    rag_docs = [c for c in context if c.get("type") == "rag"]
    retrieved_docs = [
        c
        for c in context
        if c.get("type") in {"rag", "rag_diagnostic", "web_supplement", "web_evidence"}
        or c.get("source_type") == "web"
    ]
    web_results = [c for c in context if c.get("type") == "web"]
    web_evidence = _web_evidence_items(context)
    # TEMP A3_TRACE: remove after multi-subject retrieval validation.
    emit_a3_trace(
        logger,
        "generate_answer",
        {
            "context_rag_count": len(rag_docs),
            "context_web_count": len(web_evidence),
            "context_web_supplement_count": len(web_evidence),
            "web_supplement_needed": bool(state.get("web_supplement_decisions")),
            "web_supplement_count": len(web_evidence),
            "web_supplement_provider": state.get("web_supplement_provider", "tavily"),
            "web_supplement_failed": bool(state.get("web_supplement_failed")),
            "web_supplement_failure_reason": state.get("web_supplement_failure_reason", ""),
            "web_supplement_partial_failed": bool(state.get("web_supplement_partial_failed")),
            "web_supplement_status_by_subject": state.get("web_supplement_status_by_subject", {}),
            "web_supplement_success_subjects": state.get("web_supplement_success_subjects", []),
            "web_supplement_failed_subjects": state.get("web_supplement_failed_subjects", []),
            "web_evidence_count": len(web_evidence),
            "web_evidence_provider": "tavily",
            "web_judge_provider": state.get("web_judge_provider", _judge_provider()),
            "web_judge_model": state.get("web_judge_model", _judge_model()),
            "web_judge_failed_subjects": state.get("web_judge_failed_subjects", []),
            "web_judge_rejected_all_subjects": state.get("web_judge_rejected_all_subjects", []),
            "web_evidence_use_cases": sorted({doc.get("use_case") for doc in web_evidence if doc.get("use_case")}),
            "web_evidence_types": sorted({doc.get("evidence_type") for doc in web_evidence if doc.get("evidence_type")}),
            "dual_source_mode": bool(state.get("dual_source_mode")),
            "evidence_judge_state": state.get("evidence_judge_state", ""),
            "search_refinement_needed": bool(state.get("search_refinement_needed")),
            "search_refinement_deferred": bool(state.get("search_refinement_deferred")),
            "subjects_used": _subjects_used(rag_docs),
            "roles_used": _roles_used(rag_docs),
            "branch_mode": state.get("retrieval_branch_mode", ""),
            "web_supplement_subjects": sorted({doc.get("supplement_for_subject") for doc in web_evidence if doc.get("supplement_for_subject")}),
            "web_supplement_purposes": sorted({doc.get("supplement_purpose") for doc in web_evidence if doc.get("supplement_purpose")}),
            "learning_goal": state.get("learning_goal", ""),
            "primary_subject": state.get("primary_subject", ""),
            "resource_offer": not bool(state.get("requested_resource_type") or state.get("needs_mindmap")),
            "model_group": "academic",
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )

    temperature = get_setting("academic.temperature", 0.7)
    user_prompt = load_prompt("academic_answer").format(
        retrieved_context=_format_retrieved(retrieved_docs),
        search_context=_format_search(web_results),
        question=question,
        resource_offer_instruction=_resource_offer_instruction(state),
    )

    max_tokens = get_setting("academic.max_tokens", None)
    fallback_kwargs = {"temperature": temperature}
    if max_tokens is not None:
        fallback_kwargs["max_tokens"] = max_tokens
    fallback = get_fallback_llm(**fallback_kwargs)
    messages = [
        SystemMessage(content=load_prompt("academic_system")),
        HumanMessage(content=user_prompt),
    ]

    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        node_name="generate_answer",
        temperature=temperature,
    ) as span:
        response = await async_invoke_with_fallback(
            llm, messages, fallback=fallback, span=span,
        )

    return {"messages": [AIMessage(content=response.content)]}


# ── Node 4: hallucination evaluation (reflection loop) ─────────


# TEMP A3_TRACE: remove after diagnostics validation.
def _coerce_hallucination_evaluation(value: Any) -> HallucinationEvaluation | None:
    if isinstance(value, HallucinationEvaluation):
        return value
    if isinstance(value, dict):
        return HallucinationEvaluation.model_validate(value)
    return None


def _hallucination_pack_parts(result_pack: Any) -> tuple[HallucinationEvaluation | None, Any, str]:
    if isinstance(result_pack, HallucinationEvaluation):
        return result_pack, None, ""
    if not isinstance(result_pack, dict):
        return _coerce_hallucination_evaluation(result_pack), None, ""

    raw_message = result_pack.get("raw")
    raw_text = _message_content_to_text(getattr(raw_message, "content", raw_message))
    parsed = _coerce_hallucination_evaluation(result_pack.get("parsed"))
    return parsed, result_pack.get("parsing_error"), raw_text[:500] if raw_text else ""


@traced_node
async def evaluate_hallucination(state: LearningState) -> dict:
    """Evaluate whether the generated answer hallucinates beyond retrieved context.

    Uses fail-fast structured LLM output to judge faithfulness. On detection,
    increments retry_count to signal the conditional edge for re-retrieval.
    Structured-output failures are surfaced instead of being treated as faithful.
    """
    if state.get("evidence_judge_failed") and _block_generation_when_evidence_judge_failed():
        emit_a3_trace(
            logger,
            "hallucination_eval",
            {
                "skipped": True,
                "skip_reason": "skipped_due_to_evidence_judge_failed",
                "evidence_judge_failure_phase": _evidence_failure_phase(state),
                "context_count": len(state.get("context", [])),
                "success": False,
                "is_faithful": None,
            },
            state=state,
            env_flag="LOG_RETRY_TRACE",
            max_chars=1000,
        )
        return {
            "hallucination_detected": False,
            "hallucination_reason": "skipped_due_to_evidence_judge_failed",
        }

    eval_temp = get_setting("hallucination_eval.temperature", 0.0)
    eval_model = get_setting("hallucination_eval.model", os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"))

    # Extract the generated answer (last message) and original question
    answer = state["messages"][-1].content
    question = _last_human_query(state)

    # Build context from all retrieval sources
    docs = state.get("context", [])
    context = "\n".join(d.get("content", "") for d in docs) if docs else ""

    eval_prompt = load_prompt("hallucination_eval").format(
        question=question, context=context, answer=answer,
    )

    retry_count = state.get("retry_count", 0)
    eval_messages = [
        SystemMessage(content=load_prompt("hallucination_system")),
        HumanMessage(content=eval_prompt),
    ]
    rag_docs = [d for d in docs if d.get("type") == "rag"]
    web_evidence = _web_evidence_items(docs)

    try:
        with traced_llm_call(
            model_name=eval_model,
            node_name="evaluate_hallucination",
            temperature=eval_temp,
        ):
            structured_result = await invoke_structured_llm(
                node_name="hallucination_eval",
                llm_node="hallucination_eval",
                schema=HallucinationEvaluation,
                messages=eval_messages,
                output_mode=get_llm_output_mode("hallucination_eval"),
                fallback_modes=get_fallback_modes("hallucination_eval"),
                business_validator=validate_hallucination_eval,
                state=state,
                max_raw_chars=get_max_raw_chars("hallucination_eval"),
            )
    except StructuredOutputError as exc:
        emit_a3_trace(
            logger,
            "hallucination_eval",
            {
                "success": False,
                "hallucination_eval_failed": True,
                "failure_phase": exc.result.failure_phase,
                "error_type": exc.result.error_type,
                "error_message": exc.result.error_message,
                "retry_count": retry_count,
                "model_group": "academic",
                "eval_model": eval_model,
                "context_rag_count": len(rag_docs),
                "context_web_count": len(web_evidence),
                "answer_chars": len(str(answer)),
                "prompt_chars": len(eval_prompt),
            },
            state=state,
            env_flag="LOG_RETRY_TRACE",
            max_chars=12000,
        )
        raise

    evaluation = structured_result.parsed
    if not isinstance(evaluation, HallucinationEvaluation):
        raise TypeError("hallucination_eval parsed result is not HallucinationEvaluation")
    is_faithful = evaluation.is_faithful
    failure_phase = ""

    hallucination_detected = not is_faithful
    # TEMP A3_TRACE: remove after multi-subject retrieval validation.
    emit_a3_trace(
        logger,
        "hallucination_eval",
        {
            "success": True,
            "is_faithful": is_faithful,
            "retry_count": retry_count,
            "reason": evaluation.reason,
            "failure_phase": failure_phase,
            "primary_called": True,
            "fallback_called": False,
            "fallback_used": False,
            "parsing_error": "",
            "raw_preview": structured_result.raw_output[:2000],
            "parsed_is_none": False,
            "model_group": "academic",
            "eval_model": eval_model,
            "fallback_model": "",
            "context_rag_count": len(rag_docs),
            "context_web_count": len(web_evidence),
            "answer_chars": len(str(answer)),
            "prompt_chars": len(eval_prompt),
        },
        state=state,
        env_flag="LOG_RETRY_TRACE",
        max_chars=500,
    )

    result: dict = {"hallucination_detected": hallucination_detected}
    if hallucination_detected:
        result["retry_count"] = retry_count + 1
        result["hallucination_reason"] = evaluation.reason

    return result


def should_retry_or_end(state: LearningState) -> str:
    """Conditional edge: retry via academic_router or route to END.

    Allows up to MAX_RETRIES re-retrieval attempts when hallucination
    is detected. After exhausting retries, routes to END regardless.
    """
    if (
        state.get("hallucination_detected", False)
        and state.get("retry_count", 0) <= MAX_RETRIES
    ):
        return "retry"
    return "end"
