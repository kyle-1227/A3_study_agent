"""SubGraph A — Academic Tutor: parallel retrieval (fan-out/fan-in),
answer generation, and hallucination evaluation with retry loop.

Keypoint extraction is handled by the supervisor node (merged for latency),
so this subgraph starts at the academic_router which fans out to both
rag_retrieve and web_search in parallel.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
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
    """Query ChromaDB with keypoints extracted by the supervisor node."""
    rewritten = state.get("rewritten_query", "")
    retrieval_plan = state.get("retrieval_plan") or []
    subject = state.get("subject")
    query, query_source = _query_source(state)

    subj = subject if subject != "other" else None

    if not rewritten and retrieval_plan:
        per_subject_top_k = get_setting("rag.multi_subject_per_subject_top_k", 3)
        max_docs = get_setting("rag.multi_subject_max_docs", 8)
        subjects = [str(item.get("subject", "")) for item in retrieval_plan if item.get("subject")]

        with traced_retrieval(query=query, subject="multi") as span:
            span.set_attribute("rag.retrieval_plan_count", len(retrieval_plan))
            span.set_attribute("rag.retrieval_subjects", ",".join(subjects))
            all_docs: list[dict] = []
            for item in retrieval_plan:
                plan_subject = item.get("subject")
                plan_query = item.get("rag_query")
                if not plan_subject or not plan_query:
                    continue

                result = await asyncio.to_thread(
                    retrieve,
                    query=plan_query,
                    subject=plan_subject,
                    top_k=per_subject_top_k,
                )
                raw_docs = result.get("docs", []) or []
                used_docs = raw_docs[:per_subject_top_k]
                role = item.get("role", "supporting_context")
                priority = item.get("priority", 0.5)
                subject_mismatch_count = _subject_mismatch_count(used_docs, plan_subject)
                branch_eval = _evaluate_retrieval_branch(
                    subject=plan_subject,
                    role=role,
                    docs=used_docs,
                    is_hit=result.get("is_hit", False),
                    subject_mismatch_count=subject_mismatch_count,
                )
                # TEMP A3_TRACE: remove after multi-subject retrieval validation.
                emit_a3_trace(
                    logger,
                    "rag_retrieve_plan_item",
                    {
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
                    all_docs.append({
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
                        "content": "本地知识库中暂未检索到该学科分支的有效资料。",
                        "source": "local_rag_diagnostic",
                        "score": 0.0,
                    })
                    continue

                for doc in used_docs:
                    all_docs.append({
                        "type": "rag",
                        "retrieval_subject": plan_subject,
                        "retrieval_role": role,
                        "retrieval_query": plan_query,
                        "retrieval_purpose": item.get("purpose", ""),
                        "relation_to_goal": item.get("relation_to_goal", ""),
                        "retrieval_priority": priority,
                        "branch_status": branch_eval["branch_status"],
                        "weak_reason": branch_eval["weak_reason"],
                        "best_rerank_score": branch_eval["best_rerank_score"],
                        "needs_supplement": branch_eval["needs_supplement"],
                        **doc,
                    })

            selected_docs, quota_debug = _select_docs_with_subject_quota(
                all_docs,
                max_docs,
                primary_subject=str(state.get("primary_subject") or ""),
            )
            subject_counter = Counter(doc.get("retrieval_subject") for doc in selected_docs)
            role_counter = Counter(doc.get("retrieval_role") for doc in selected_docs)
            # TEMP A3_TRACE: remove after multi-subject retrieval validation.
            emit_a3_trace(
                logger,
                "context_assembly",
                {
                    "mode": "multi_subject",
                    "retrieval_plan_count": len(retrieval_plan),
                    "raw_doc_count": len(all_docs),
                    "final_doc_count": len(selected_docs),
                    "max_docs": max_docs,
                    "subject_doc_distribution": dict(subject_counter),
                    "role_distribution": dict(role_counter),
                    **quota_debug,
                    "selected_docs": [
                        {
                            "subject": doc.get("retrieval_subject"),
                            "role": doc.get("retrieval_role"),
                            "branch_status": doc.get("branch_status"),
                            "weak_reason": doc.get("weak_reason"),
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

        return {"context": selected_docs}

    with traced_retrieval(query=query, subject=subj) as span:
        result = await asyncio.to_thread(retrieve, query=query, subject=subj)
        docs = result.get("docs", []) or []
        subject_mismatch_count = _subject_mismatch_count(docs, subj)
        branch_eval = _evaluate_retrieval_branch(
            subject=subj or "",
            role="single_subject",
            docs=docs,
            is_hit=result.get("is_hit", False),
            subject_mismatch_count=subject_mismatch_count,
        )
        # TEMP A3_TRACE: remove after multi-subject retrieval validation.
        emit_a3_trace(
            logger,
            "rag_retrieve_single_subject",
            {
                "subject": subj,
                "query": query,
                "query_source": query_source,
                "raw_doc_count": len(docs),
                "used_doc_count": len(docs),
                "doc_count": len(docs),
                "is_hit": result.get("is_hit", False),
                "subject_mismatch_count": subject_mismatch_count,
                "branch_status": branch_eval["branch_status"],
                "weak_reason": branch_eval["weak_reason"],
                "best_rerank_score": branch_eval["best_rerank_score"],
                "top_docs": _top_doc_summaries(docs),
            },
            state=state,
            env_flag="LOG_RAG_RESULT",
        )
        # TEMP A3_TRACE: remove after multi-subject retrieval validation.
        emit_a3_trace(
            logger,
            "context_assembly",
            {
                "mode": "single_subject",
                "subject": subj,
                "final_doc_count": len(docs),
            },
            state=state,
            env_flag="LOG_CONTEXT_ASSEMBLY",
        )
        span.set_attribute("rag.doc_count", len(result.get("docs", [])))
        span.set_attribute("rag.is_hit", result.get("is_hit", False))
        if result.get("docs"):
            span.set_attribute("rag.top_score", result["docs"][0].get("score", 0))

    return {"context": [{"type": "rag", **doc} for doc in docs]}


# ── Node 2: web search (parallel branch B) ────────────────────────

_SEARCH_TIMEOUT = get_setting("academic.search_timeout", 15)


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


@traced_node
async def web_search(state: TutorState) -> dict:
    """Fan-out web search — runs in parallel with rag_retrieve."""
    rewritten = state.get("rewritten_query", "")
    search_web_query = state.get("search_web_query", "")
    retrieval_plan = state.get("retrieval_plan") or []
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
    web_results = [c for c in context if c.get("type") == "web"]
    # TEMP A3_TRACE: remove after multi-subject retrieval validation.
    emit_a3_trace(
        logger,
        "generate_answer",
        {
            "context_rag_count": len(rag_docs),
            "context_web_count": len(web_results),
            "subjects_used": _subjects_used(rag_docs),
            "roles_used": _roles_used(rag_docs),
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
        retrieved_context=_format_retrieved(rag_docs),
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
