"""SubGraph A — Academic Tutor: parallel retrieval (fan-out/fan-in),
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
import time
from collections import Counter, defaultdict
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.config import get_setting, load_prompt
from src.graph.llm import async_invoke_with_fallback, get_fallback_llm, get_node_llm
from src.graph.state import CONTEXT_CLEAR, TutorState
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


def _last_human_query(state: TutorState) -> str:
    """Extract the last HumanMessage content (robust for retry loops)."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _render_prompt(prompt_name: str, replacements: dict[str, str]) -> str:
    """Render named placeholders without interpreting JSON braces in prompts."""
    prompt = load_prompt(prompt_name)
    for key, value in replacements.items():
        prompt = prompt.replace("{" + key + "}", value)
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
    }


def _query_source(state: TutorState) -> tuple[str, str]:
    rewritten = state.get("rewritten_query", "")
    search_rag_query = state.get("search_rag_query", "")
    expanded_keypoints = state.get("expanded_keypoints", [])
    keypoints = state.get("keypoints", [])
    if rewritten:
        return rewritten, "rewritten_query"
    if search_rag_query:
        return search_rag_query, "search_rag_query"
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
            "score": doc.get("score"),
            "rerank_score": doc.get("rerank_score"),
        }
        for i, doc in enumerate(docs[:limit])
    ]


def _subjects_used(docs: list[dict]) -> list[str]:
    return sorted({str(doc.get("retrieval_subject")) for doc in docs if doc.get("retrieval_subject")})


def _roles_used(docs: list[dict]) -> list[str]:
    return sorted({str(doc.get("retrieval_role")) for doc in docs if doc.get("retrieval_role")})


def _score_doc(doc: dict) -> float:
    """Best available score for sorting retrieved docs."""
    value = doc.get("rerank_score", doc.get("score", 0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _best_doc_score(docs: list[dict]) -> float:
    """Return the best rerank/score value available for a doc list."""
    if not docs:
        return 0.0
    return max(_score_doc(doc) for doc in docs)


def _evaluate_retrieval_branch(
    *,
    subject: str,
    role: str,
    docs: list[dict],
    is_hit: bool,
    subject_mismatch_count: int,
) -> dict:
    """
    Classify one retrieval_plan branch by local evidence quality.

    ``role`` is accepted for future policy tuning; V1 keeps the threshold rules
    subject-agnostic and role-agnostic.
    """
    del subject, role
    doc_count = len(docs)
    best_score = _best_doc_score(docs)
    usable_threshold = float(get_setting("rag.branch_usable_threshold", 0.45))
    strong_threshold = float(get_setting("rag.branch_strong_threshold", 0.7))

    if doc_count == 0:
        branch_status = "missing"
        weak_reason = "no_docs"
    elif subject_mismatch_count > 0:
        branch_status = "weak"
        weak_reason = "subject_mismatch"
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
        "best_rerank_score": best_score,
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


def _allowed_retrieval_subjects(state: TutorState) -> set[str]:
    """Build the subject hard boundary for retrieval plans."""
    available = set(get_available_subjects_from_data())
    if available:
        return available
    subject = normalize_subject(str(state.get("subject") or ""))
    return {subject} if subject and subject != "other" else set()


def _normalize_retrieval_plan(
    raw_plan: list[RetrievalPlanItem],
    state: TutorState,
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


def _web_query_source(state: TutorState) -> tuple[str, str]:
    search_web_query = state.get("search_web_query", "")
    rewritten = state.get("rewritten_query", "")
    if search_web_query:
        return search_web_query, "search_web_query"
    if rewritten:
        return rewritten, "rewritten_query"
    return _last_human_query(state), "original_query"


def _build_retrieval_branches(state: TutorState) -> tuple[list[dict], dict]:
    """Build unified retrieval branches for multi- and single-subject paths."""
    retrieval_plan = state.get("retrieval_plan") or []
    if retrieval_plan and not state.get("rewritten_query"):
        branches = [dict(item, _synthetic_single_subject=False) for item in retrieval_plan]
        debug = {
            "mode": "multi_subject_plan",
            "branch_count": len(branches),
            "subjects": [item.get("subject") for item in branches],
            "synthetic_single_subject": False,
            "query_source": "retrieval_plan",
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


# TEMP A3_TRACE: remove after diagnostics validation.
def _web_search_diagnostics_from_legacy_result(result, query: str) -> dict:
    """Normalize old list mocks and new diagnostic dictionaries."""
    if isinstance(result, dict):
        return result
    if isinstance(result, list):
        return {
            "provider": "duckduckgo",
            "query": query,
            "ok": True,
            "results": result,
            "result_count": len(result),
            "error_type": "",
            "error_message": "",
            "raw_type": "legacy_list",
            "raw_count": len(result),
            "elapsed_ms": None,
        }
    return {
        "provider": "duckduckgo",
        "query": query,
        "ok": False,
        "results": [],
        "result_count": 0,
        "error_type": "UnexpectedSearchDiagnosticsType",
        "error_message": sanitize_error_message(f"Unexpected diagnostics type: {type(result).__name__}"),
        "raw_type": type(result).__name__,
        "raw_count": None,
        "elapsed_ms": None,
    }


# TEMP A3_TRACE: remove after diagnostics validation.
def _web_search_exception_diagnostics(query: str, exc: Exception, *, elapsed_ms=None) -> dict:
    return {
        "provider": "duckduckgo",
        "query": query,
        "ok": False,
        "results": [],
        "result_count": 0,
        "error_type": type(exc).__name__,
        "error_message": sanitize_error_message(exc),
        "raw_type": "",
        "raw_count": None,
        "elapsed_ms": elapsed_ms,
    }


def _clip_text(value: Any, limit: int) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text[:limit] + ("..." if len(text) > limit else "")


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
            "used_doc_count": len(docs),
            "top_docs": [
                {
                    "source": doc.get("source"),
                    "metadata_subject": _doc_subject(doc),
                    "rerank_score": doc.get("rerank_score"),
                    "score": doc.get("score"),
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
    return _clip_text(query, max_chars)


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
            query = _normalize_supplement_query(item.query)
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
                    "query": _normalize_supplement_query(fallback_query),
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


_OPEN_APPLICATION_TERMS = (
    "应用", "怎么用", "案例", "项目", "实践", "路线", "规划", "最新", "工具", "框架",
    "场景", "扩展", "行业", "前沿", "发展", "趋势", "真实", "落地",
    "apply", "application", "case", "project", "practice", "tool", "framework",
    "latest", "trend", "industry", "roadmap", "plan", "ecosystem",
)


def _question_has_any(question: str, terms: tuple[str, ...]) -> bool:
    lower = question.lower()
    return any(term.lower() in lower for term in terms)


def _rule_based_web_supplement_targets(
    *,
    state: TutorState,
    retrieval_plan: list[dict],
    branch_evals: dict[str, dict],
) -> tuple[list[dict], dict]:
    """Fallback when LLM coverage decision fails."""
    question = _last_human_query(state)
    open_application = _question_has_any(question, _OPEN_APPLICATION_TERMS)
    requested_resource = bool(state.get("requested_resource_type") or state.get("needs_mindmap"))
    intent = state.get("intent", "")
    targets: list[dict] = []
    for item in retrieval_plan:
        subject = str(item.get("subject") or "")
        branch_eval = branch_evals.get(subject, {})
        status = str(branch_eval.get("branch_status") or "unknown")
        purposes: list[str] = []
        if status in {"weak", "missing"}:
            purposes.append("repair")
        if open_application:
            purposes.extend(["coverage_expansion", "application_context"])
        if _question_has_any(question, ("工具", "框架", "库", "技术栈", "tool", "framework", "library", "ecosystem")):
            purposes.append("tool_ecosystem")
        if _question_has_any(question, ("案例", "项目", "实践", "示例", "case", "project", "example", "practice")):
            purposes.extend(["case_example", "implementation_detail"])
        if _question_has_any(question, ("最新", "前沿", "趋势", "当前", "latest", "trend", "current")):
            purposes.append("latest_practice")
        if intent == "planning":
            purposes.append("planning_support")
        if requested_resource:
            purposes.append("resource_enrichment")

        purposes = _normalize_purposes(purposes)
        if not purposes:
            continue
        query = item.get("web_search_query") or item.get("rag_query") or question
        targets.append({
            "subject": subject,
            "role": item.get("role", "supporting_context"),
            "coverage_risk": "high" if open_application or requested_resource else ("medium" if status in {"weak", "missing"} else "low"),
            "local_evidence_strength": status,
            "supplement_purposes": purposes,
            "supplement_queries": [
                {
                    "purpose": purposes[0],
                    "query": _normalize_supplement_query(query),
                    "priority": _clamp_priority(item.get("priority", 0.5)),
                    "reason": "Rule fallback selected this branch for Web supplement.",
                }
            ],
            "decision_reason": "Rule fallback based on branch status and task wording.",
            "subject_priority": _clamp_priority(item.get("priority", 0.5)),
            "branch_status": status,
        })

    max_subjects = int(_web_setting("max_supplement_subjects", 2))
    targets.sort(key=lambda target: _clamp_priority(target.get("subject_priority")), reverse=True)
    return targets[:max_subjects], {
        "fallback_reason": "rule_based_coverage_decision",
        "open_application": open_application,
        "requested_resource": requested_resource,
        "target_count": len(targets[:max_subjects]),
    }


async def _decide_web_supplement_with_llm(
    *,
    state: TutorState,
    retrieval_plan: list[dict],
    branch_evals: dict[str, dict],
    docs_by_subject: dict[str, list[dict]],
    branch_mode: str,
) -> tuple[list[dict], dict]:
    """Return selected Web supplement targets and diagnostics."""
    enabled = bool(_web_setting("llm_decision_enabled", True))
    if not enabled:
        targets, fallback_debug = _rule_based_web_supplement_targets(
            state=state,
            retrieval_plan=retrieval_plan,
            branch_evals=branch_evals,
        )
        return targets, {
            "enabled": False,
            "llm_used": False,
            "success": True,
            "fallback_used": True,
            "overall_need_web": bool(targets),
            "decision_summary": "LLM coverage decision disabled; used rule fallback.",
            "subject_decisions": [],
            "selected_targets": targets,
            **fallback_debug,
        }

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
    raw_preview = ""
    parsing_error = ""
    try:
        llm = get_node_llm(
            "web_coverage_decision",
            temperature=0.0,
            max_tokens=1000,
            streaming=False,
        )
        structured_llm = llm.with_structured_output(
            CoverageDecisionOutput,
            method="json_mode",
            include_raw=True,
        )
        fallback = get_fallback_llm(temperature=0.0, max_tokens=1000, streaming=False)
        structured_fallback = fallback.with_structured_output(
            CoverageDecisionOutput,
            method="json_mode",
            include_raw=True,
        )
        messages = [
            SystemMessage(content="You are a coverage decision agent. Return only valid JSON."),
            HumanMessage(content=prompt),
        ]
        result_pack = await async_invoke_with_fallback(
            structured_llm,
            messages,
            fallback=structured_fallback,
        )
        raw_message = result_pack.get("raw") if isinstance(result_pack, dict) else None
        parsed = result_pack.get("parsed") if isinstance(result_pack, dict) else result_pack
        parsing_error = str(result_pack.get("parsing_error") or "") if isinstance(result_pack, dict) else ""
        raw_preview = _message_content_to_text(getattr(raw_message, "content", raw_message))[:500] if raw_message else ""
        if parsing_error:
            raise ValueError(f"coverage decision parsing_error: {parsing_error}")
        if parsed is None:
            raise ValueError("coverage decision parsed result is None")
        if not isinstance(parsed, CoverageDecisionOutput):
            parsed = CoverageDecisionOutput.model_validate(parsed)
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
            "parsing_error": parsing_error,
            "raw_preview": raw_preview,
        }
    except Exception as exc:
        targets, fallback_debug = _rule_based_web_supplement_targets(
            state=state,
            retrieval_plan=retrieval_plan,
            branch_evals=branch_evals,
        )
        return targets, {
            "enabled": True,
            "llm_used": True,
            "success": False,
            "fallback_used": True,
            "overall_need_web": bool(targets),
            "decision_summary": "Coverage decision LLM failed; used rule fallback.",
            "subject_decisions": [],
            "selected_targets": targets,
            "error_type": type(exc).__name__,
            "error_message": sanitize_error_message(exc),
            "parsing_error": parsing_error,
            "raw_preview": raw_preview,
            **fallback_debug,
        }


async def _run_dynamic_web_supplement(
    *,
    state: TutorState,
    targets: list[dict],
    decision_debug: dict,
    branch_mode: str,
) -> tuple[list[dict], dict]:
    """Run dynamic Web supplement with bounded attempts."""
    del decision_debug
    max_total = int(_web_setting("max_total_attempts", 10))
    max_per_subject = int(_web_setting("max_attempts_per_subject", 3))
    max_results = int(_web_setting("max_results_per_attempt", 5))
    timeout = _web_timeout_seconds()
    attempts = 0
    attempts_by_subject: Counter = Counter()
    docs: list[dict] = []
    attempt_logs: list[dict] = []

    for target in targets:
        subject = target.get("subject", "")
        if attempts >= max_total:
            break
        for query_item in sorted(target.get("supplement_queries", []), key=lambda item: _clamp_priority(item.get("priority")), reverse=True):
            if attempts >= max_total or attempts_by_subject[subject] >= max_per_subject:
                break
            attempts += 1
            attempts_by_subject[subject] += 1
            query = _normalize_supplement_query(query_item.get("query", ""))
            purpose = query_item.get("purpose") or (target.get("supplement_purposes") or ["coverage_expansion"])[0]
            started = time.perf_counter()
            diagnostics: dict
            timed_out = False
            try:
                raw = await asyncio.wait_for(
                    asyncio.to_thread(web_search_fn, query),
                    timeout=timeout,
                )
                diagnostics = _web_search_diagnostics_from_legacy_result(raw, query)
            except asyncio.TimeoutError:
                timed_out = True
                diagnostics = _web_search_exception_diagnostics(
                    query,
                    TimeoutError(f"web supplement exceeded {timeout}s"),
                    elapsed_ms=round(timeout * 1000, 2),
                )
            except Exception as exc:
                diagnostics = _web_search_exception_diagnostics(query, exc)

            elapsed_ms = diagnostics.get("elapsed_ms")
            if elapsed_ms is None:
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            results = diagnostics.get("results", []) or []
            used_results = results[:max_results]
            for result in used_results:
                docs.append({
                    "type": "web_supplement",
                    "content": result.get("content", ""),
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "source": result.get("url") or result.get("title") or "web_search",
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
                })
            attempt_payload = {
                "branch_mode": branch_mode,
                "subject": subject,
                "role": target.get("role", ""),
                "purpose": purpose,
                "attempt": attempts,
                "subject_attempt": attempts_by_subject[subject],
                "max_total_attempts": max_total,
                "max_attempts_per_subject": max_per_subject,
                "query": query,
                "ok": diagnostics.get("ok", False),
                "timed_out": timed_out or diagnostics.get("error_type") == "TimeoutError",
                "result_count": diagnostics.get("result_count", len(results)),
                "used_result_count": len(used_results),
                "elapsed_ms": elapsed_ms,
                "error_type": diagnostics.get("error_type", ""),
                "error_message": diagnostics.get("error_message", ""),
                "top_results": [
                    {"title": item.get("title", ""), "url": item.get("url", "")}
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
            if used_results:
                break

    return docs, {
        "attempts_used": attempts,
        "max_total_attempts": max_total,
        "attempts_by_subject": dict(attempts_by_subject),
        "result_doc_count": len(docs),
        "attempts": attempt_logs,
    }


# ── Node 0: academic router (fan-out trigger) ─────────────────────

@traced_node
async def academic_router(state: TutorState) -> dict:
    """Router node for parallel fan-out. Clears context on retry path."""
    if state.get("retry_count", 0) > 0:
        return {"context": CONTEXT_CLEAR}
    return {}


# ── Node 0b: query rewriting (retry path only) ──────────────────

@traced_node
async def rewrite_query(state: TutorState) -> dict:
    """Rewrite the user's query using hallucination feedback for better retrieval."""
    original_query = _last_human_query(state)
    reason = state.get("hallucination_reason", "")

    llm = get_node_llm("supervisor")
    rewrite_prompt = load_prompt("rewrite_query").format(
        original_query=original_query,
        hallucination_reason=reason,
    )

    try:
        response = await llm.ainvoke([
            SystemMessage(content="你是一个查询改写助手。根据反馈改进用户的搜索查询。"),
            HumanMessage(content=rewrite_prompt),
        ])
        rewritten = response.content.strip()
    except Exception:
        logger.warning("Query rewrite failed, using original query")
        rewritten = original_query

    # TEMP A3_TRACE: remove after multi-subject retrieval validation.
    emit_a3_trace(
        logger,
        "rewrite_query_retry",
        {
            "retry_count": state.get("retry_count", 0),
            "hallucination_reason": reason,
            "rewritten_query": rewritten,
            "retrieval_plan_cleared": True,
        },
        state=state,
        env_flag="LOG_RETRY_TRACE",
    )

    return {"rewritten_query": rewritten, **_clear_retrieval_plan_state()}


# ── Node 0c: initial search-query rewriting ───────────────────────────────

@traced_node
async def search_query_rewriter(state: TutorState) -> dict:
    """Rewrite the original request into RAG and web-search queries."""
    if state.get("rewritten_query"):
        return _clear_retrieval_plan_state()

    original_query = _last_human_query(state)
    keypoints = state.get("keypoints", [])
    requested_resource_type = state.get("requested_resource_type", "")
    subject = state.get("subject", "")
    subject_candidates = state.get("subject_candidates", [])
    available_subjects = get_available_subjects_from_data()

    prompt = _render_prompt(
        "search_query_rewriter",
        {
            "question": original_query,
            "keypoints": "、".join(keypoints) if keypoints else "未提取到明确关键词",
            "requested_resource_type": requested_resource_type or "none",
            "subject": subject or "other",
            "subject_candidates": "、".join(subject_candidates) if subject_candidates else "无",
            "available_subjects": "、".join(available_subjects) if available_subjects else "无",
        },
    )

    llm = get_node_llm("query_rewrite", temperature=0.0)
    messages = [
        SystemMessage(content="你是高校个性化学习资源系统中的检索查询改写智能体，只输出结构化查询结果。"),
        HumanMessage(content=prompt),
    ]

    structured_llm = llm.with_structured_output(
        SearchQueryRewriteOutput,
        method="json_mode",
        include_raw=True,
    )
    fallback_llm = get_fallback_llm(temperature=0.0)
    structured_fallback = fallback_llm.with_structured_output(
        SearchQueryRewriteOutput,
        method="json_mode",
        include_raw=True,
    )

    try:
        with traced_llm_call(
            model_name=get_setting("query_rewrite.model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
            node_name="search_query_rewriter",
            temperature=0.0,
        ) as span:
            result_pack = await async_invoke_with_fallback(
                structured_llm,
                messages,
                fallback=structured_fallback,
                span=span,
            )
        raw_message = result_pack.get("raw")
        parsed = result_pack.get("parsed")
        parsing_error = result_pack.get("parsing_error")
        raw_text = _message_content_to_text(getattr(raw_message, "content", raw_message))
        raw_preview = raw_text[:2000] if raw_text else ""

        if parsing_error is not None:
            raise ValueError(f"search_query_rewriter parsing_error: {parsing_error}")
        if parsed is None:
            raise ValueError("search_query_rewriter parsed result is None")

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
        logger.warning("Initial search query rewrite failed; continuing with original query", exc_info=True)
        # TEMP A3_TRACE: remove after multi-subject retrieval validation.
        emit_a3_trace(
            logger,
            "query_rewrite_failed",
            {
                "error": str(exc),
                "fallback": "empty_retrieval_plan_and_single_query_fallback",
                "retrieval_plan": [],
                "learning_goal": "",
                "primary_subject": "",
                "subject_relation_summary": "",
                "raw_preview": raw_preview if "raw_preview" in locals() else "",
            },
            state=state,
            env_flag="LOG_QUERY_REWRITE_RESULT",
        )
        return {
            "search_query_rewrite_error": str(exc),
            "search_rag_query": "",
            "search_web_query": "",
            "expanded_keypoints": [],
            "search_query_rewrite_reason": "",
            "search_query_rewrite_raw_preview": raw_preview if "raw_preview" in locals() else "",
            **_clear_retrieval_plan_state(),
        }

    return {
        "search_rag_query": result_payload["rag_query"],
        "search_web_query": result_payload["web_search_query"],
        "expanded_keypoints": result_payload["expanded_keypoints"],
        "search_query_rewrite_reason": result_payload["reason"],
        "search_query_rewrite_error": "",
        "search_query_rewrite_raw_preview": raw_preview,
        **multi_subject_payload,
    }


# ── Node 1: RAG retrieval (parallel branch A) ─────────────────────

@traced_node
async def rag_retrieve(state: TutorState) -> dict:
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
                    "needs_supplement": True,
                    "content": "No effective local course material was retrieved for this subject branch.",
                    "source": "local_rag_diagnostic",
                    "score": 0.0,
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
                        "source": doc.get("source"),
                        "score": doc.get("score"),
                        "rerank_score": doc.get("rerank_score"),
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
            span.set_attribute("rag.top_score", _score_doc(selected_docs[0]))

    return {
        "context": selected_docs,
        "web_supplement_decisions": targets,
        "web_supplement_results": web_supplement_docs,
        "coverage_decision_summary": decision_debug.get("decision_summary", ""),
        "retrieval_branch_mode": branch_mode,
    }

_SEARCH_TIMEOUT = _web_timeout_seconds()


@traced_node
async def web_search(state: TutorState) -> dict:
    """Fan-out web search — runs in parallel with rag_retrieve."""
    rewritten = state.get("rewritten_query", "")
    search_web_query = state.get("search_web_query", "")
    retrieval_plan = state.get("retrieval_plan") or []
    if (
        _web_conditional_enabled()
        and bool(_web_setting("skip_general_when_conditional", True))
        and state.get("intent") in {"academic", "planning"}
    ):
        branch_mode = "multi_subject_plan" if retrieval_plan else "single_subject_synthetic"
        # TEMP A3_TRACE: remove after multi-subject retrieval validation.
        emit_a3_trace(
            logger,
            "web_search",
            {
                "query_source": "skipped_conditional_branch_mode",
                "skipped": True,
                "skip_reason": "conditional_web_supplement_handled_in_rag_retrieve",
                "has_retrieval_plan": bool(retrieval_plan),
                "branch_mode": branch_mode,
                "retrieval_plan_count": len(retrieval_plan),
                "result_count": 0,
                "timed_out": False,
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
            "provider": "duckduckgo",
            "query": query,
            "ok": False,
            "results": [],
            "result_count": 0,
            "error_type": "",
            "error_message": "",
            "raw_type": "",
            "raw_count": None,
            "elapsed_ms": None,
        }
        try:
            raw_diagnostics = await asyncio.wait_for(
                asyncio.to_thread(web_search_fn, query),
                timeout=_SEARCH_TIMEOUT,
            )
            diagnostics = _web_search_diagnostics_from_legacy_result(raw_diagnostics, query)
            search_results = diagnostics.get("results", [])
            span.set_attribute("search.result_count", len(search_results))
            span.set_attribute("search.timed_out", False)
        except asyncio.TimeoutError:
            diagnostics = _web_search_exception_diagnostics(
                query,
                TimeoutError(f"web search exceeded {_SEARCH_TIMEOUT}s"),
                elapsed_ms=round(_SEARCH_TIMEOUT * 1000, 2),
            )
            search_results = []
            span.set_attribute("search.result_count", 0)
            span.set_attribute("search.timed_out", True)
        except Exception as exc:
            diagnostics = _web_search_exception_diagnostics(query, exc)
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
            "retrieval_plan_count": len(retrieval_plan),
            "selected_subject": selected_subject,
            "result_count": len(search_results),
            "timed_out": diagnostics.get("error_type") == "TimeoutError",
            "provider": diagnostics.get("provider", "duckduckgo"),
            "ok": diagnostics.get("ok", False),
            "raw_type": diagnostics.get("raw_type", ""),
            "raw_count": diagnostics.get("raw_count"),
            "elapsed_ms": diagnostics.get("elapsed_ms"),
            "error_type": diagnostics.get("error_type", ""),
            "error_message": diagnostics.get("error_message", ""),
        },
        state=state,
        env_flag="LOG_WEB_SEARCH_RESULT",
    )

    return {"context": [{"type": "web", **r} for r in search_results]}


# ── Node 3: generate answer ──────────────────────────────────────

def _format_retrieved(docs: list[dict]) -> str:
    if not docs:
        return "无相关参考资料。"
    if any(doc.get("type") == "web_supplement" for doc in docs):
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
            if d.get("type") != "web_supplement":
                parts.append(_format_retrieved([d]))
                continue
            subject = d.get("supplement_for_subject") or d.get("retrieval_subject", "unknown")
            role = d.get("supplement_for_role") or d.get("retrieval_role", "supporting_context")
            purpose = d.get("supplement_purpose", "coverage_expansion")
            parts.append(
                f"【{subject}｜{role}｜Web 补充｜{purpose}】\n"
                "说明：以下资料用于补充该 subject 的覆盖广度、工具生态或实践背景，不属于本地课程知识库。\n"
                f"用途：{purpose_notes.get(purpose, '仅作为外部补充资料谨慎使用。')}\n"
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
                f"[{i}] 来源：{d.get('source', '未知')}（相关度：{d.get('score', 'N/A')}）\n"
                f"用途：{purpose}\n"
                f"关系：{relation}\n"
                f"检索 query：{d.get('retrieval_query', '')}\n"
                f"内容：{d.get('content', '')}"
            )
        return "\n\n".join(parts)

    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(f"[{i}] 来源：{d.get('source', '未知')}（相关度：{d.get('score', 'N/A')}）\n{d.get('content', '')}")
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


def _resource_offer_instruction(state: TutorState) -> str:
    """Return prompt instruction for optional follow-up resource offers."""
    if state.get("needs_mindmap") or state.get("requested_resource_type"):
        return _NO_RESOURCE_OFFER
    return _RESOURCE_OFFER_SECTION


@traced_node
async def generate_answer(state: TutorState) -> dict:
    """Synthesize final answer from merged context (RAG + web) via LLM."""
    llm = get_node_llm("academic")

    question = _last_human_query(state)

    # Split merged context by source type
    context = state.get("context", [])
    rag_docs = [c for c in context if c.get("type") == "rag"]
    retrieved_docs = [
        c
        for c in context
        if c.get("type") in {"rag", "rag_diagnostic", "web_supplement"}
    ]
    web_results = [c for c in context if c.get("type") == "web"]
    web_supplements = [c for c in context if c.get("type") == "web_supplement"]
    # TEMP A3_TRACE: remove after multi-subject retrieval validation.
    emit_a3_trace(
        logger,
        "generate_answer",
        {
            "context_rag_count": len(rag_docs),
            "context_web_count": len(web_results),
            "context_web_supplement_count": len(web_supplements),
            "subjects_used": _subjects_used(rag_docs),
            "roles_used": _roles_used(rag_docs),
            "branch_mode": state.get("retrieval_branch_mode", ""),
            "web_supplement_subjects": sorted({doc.get("supplement_for_subject") for doc in web_supplements if doc.get("supplement_for_subject")}),
            "web_supplement_purposes": sorted({doc.get("supplement_purpose") for doc in web_supplements if doc.get("supplement_purpose")}),
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

    fallback = get_fallback_llm(temperature=temperature)
    messages = [
        SystemMessage(content=load_prompt("academic_system")),
        HumanMessage(content=user_prompt),
    ]

    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
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


async def _invoke_hallucination_eval(
    structured_llm,
    messages: list,
    *,
    label: str,
) -> tuple[HallucinationEvaluation | None, dict]:
    """Invoke one hallucination evaluator and expose parsing diagnostics."""
    diagnostics = {
        "called": True,
        "error_type": "",
        "error_message": "",
        "parsing_error": None,
        "parsed_is_none": False,
        "raw_preview": "",
        "failure_phase": "",
    }
    try:
        result_pack = await structured_llm.ainvoke(messages)
        parsed, parsing_error, raw_preview = _hallucination_pack_parts(result_pack)
        diagnostics["raw_preview"] = raw_preview
        if parsing_error is not None:
            diagnostics["parsing_error"] = sanitize_error_message(parsing_error)
            diagnostics["failure_phase"] = (
                "structured_parsing_error"
                if label == "primary"
                else "fallback_structured_parsing_error"
            )
            return None, diagnostics
        if parsed is None:
            diagnostics["parsed_is_none"] = True
            diagnostics["failure_phase"] = "parsed_none" if label == "primary" else "fallback_parsed_none"
            return None, diagnostics
        return parsed, diagnostics
    except Exception as exc:
        diagnostics["error_type"] = type(exc).__name__
        diagnostics["error_message"] = sanitize_error_message(exc)
        diagnostics["failure_phase"] = f"{label}_call_failed"
        return None, diagnostics

@traced_node
async def evaluate_hallucination(state: TutorState) -> dict:
    """Evaluate whether the generated answer hallucinates beyond retrieved context.

    Uses structured LLM output to judge faithfulness. On detection,
    increments retry_count to signal the conditional edge for re-retrieval.
    Defaults to valid on any parsing/model failure (safe fallback).
    """
    eval_temp = get_setting("hallucination_eval.temperature", 0.0)
    eval_model = get_setting("hallucination_eval.model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    llm = get_node_llm(
        "hallucination_eval",
        temperature=0.0,
        max_tokens=256,
        streaming=False,
    )
    structured_primary = llm.with_structured_output(
        HallucinationEvaluation,
        method="json_mode",
        include_raw=True,
    )

    fallback_llm = get_fallback_llm(
        temperature=0.0,
        max_tokens=256,
        streaming=False,
    )
    structured_fallback = fallback_llm.with_structured_output(
        HallucinationEvaluation,
        method="json_mode",
        include_raw=True,
    )

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
    web_results = [d for d in docs if d.get("type") == "web"]
    primary_diag: dict = {"called": False}
    fallback_diag: dict = {"called": False}
    fallback_called = False
    fallback_used = False
    defaulted_to_valid = False
    failure_phase = ""

    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        node_name="evaluate_hallucination",
        temperature=eval_temp,
    ) as span:
        evaluation, primary_diag = await _invoke_hallucination_eval(
            structured_primary,
            eval_messages,
            label="primary",
        )
        if evaluation is None:
            fallback_called = True
            evaluation, fallback_diag = await _invoke_hallucination_eval(
                structured_fallback,
                eval_messages,
                label="fallback",
            )
            fallback_used = evaluation is not None

        if evaluation is None:
            logger.warning("Hallucination evaluation failed, defaulting to valid")
            defaulted_to_valid = True
            failure_phase = (
                fallback_diag.get("failure_phase")
                or primary_diag.get("failure_phase")
                or "primary_and_fallback_failed"
            )
            evaluation = HallucinationEvaluation(
                is_faithful=True,
                reason="evaluation_failed",
            )
            is_faithful = True
        else:
            failure_phase = primary_diag.get("failure_phase", "")
            is_faithful = evaluation.is_faithful

    hallucination_detected = not is_faithful
    # TEMP A3_TRACE: remove after multi-subject retrieval validation.
    raw_preview = primary_diag.get("raw_preview") or fallback_diag.get("raw_preview") or ""
    parsing_error = primary_diag.get("parsing_error") or fallback_diag.get("parsing_error")
    emit_a3_trace(
        logger,
        "hallucination_eval",
        {
            "success": not defaulted_to_valid,
            "defaulted_to_valid": defaulted_to_valid,
            "is_faithful": is_faithful,
            "retry_count": retry_count,
            "reason": evaluation.reason,
            "failure_phase": failure_phase,
            "primary_called": primary_diag.get("called", False),
            "fallback_called": fallback_called,
            "fallback_used": fallback_used,
            "primary_error_type": primary_diag.get("error_type", ""),
            "primary_error_message": primary_diag.get("error_message", ""),
            "fallback_error_type": fallback_diag.get("error_type", ""),
            "fallback_error_message": fallback_diag.get("error_message", ""),
            "parsing_error": parsing_error,
            "raw_preview": raw_preview,
            "parsed_is_none": primary_diag.get("parsed_is_none", False)
            or fallback_diag.get("parsed_is_none", False),
            "model_group": "academic",
            "eval_model": eval_model,
            "fallback_model": os.getenv("FALLBACK_MODEL", os.getenv("DEEPSEEK_MODEL", "")),
            "context_rag_count": len(rag_docs),
            "context_web_count": len(web_results),
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


def should_retry_or_end(state: TutorState) -> str:
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

