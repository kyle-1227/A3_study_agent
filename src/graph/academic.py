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
from collections import defaultdict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.config import get_setting, load_prompt
from src.graph.llm import async_invoke_with_fallback, get_fallback_llm, get_node_llm
from src.graph.state import CONTEXT_CLEAR, TutorState
from src.rag.course_catalog import get_available_subjects_from_data, normalize_subject
from src.rag.retriever import retrieve
from src.tools.search_tool import search as web_search_fn
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


def _score_doc(doc: dict) -> float:
    """Best available score for sorting retrieved docs."""
    value = doc.get("rerank_score", doc.get("score", 0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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
) -> tuple[list[dict], str]:
    """Filter and normalize LLM-produced per-subject retrieval plan."""
    allowed_subjects = _allowed_retrieval_subjects(state)
    by_subject: dict[str, dict] = {}

    for item in raw_plan or []:
        subject = normalize_subject(item.subject)
        rag_query = item.rag_query.strip()
        if not subject or not rag_query:
            continue
        if subject not in allowed_subjects:
            continue

        role = item.role.strip() or "supporting_context"
        if role not in _RETRIEVAL_ROLES:
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
            by_subject[subject] = normalized

    plan = sorted(by_subject.values(), key=lambda item: item["priority"], reverse=True)[:4]

    return plan, ""


def _normalize_primary_subject(parsed_primary: str, plan: list[dict]) -> str:
    primary = normalize_subject(parsed_primary)
    plan_subjects = {item["subject"] for item in plan}
    if primary and primary in plan_subjects:
        return primary
    return plan[0]["subject"] if plan else ""


def _select_docs_with_subject_quota(docs: list[dict], max_docs: int) -> list[dict]:
    """Keep at least one doc per subject when possible, then fill by priority/score."""
    if max_docs <= 0:
        return []

    deduped: list[dict] = []
    seen: set[str] = set()
    for doc in docs:
        key = _doc_dedupe_key(doc)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for doc in deduped:
        grouped[str(doc.get("retrieval_subject") or "unknown")].append(doc)

    for subject_docs in grouped.values():
        subject_docs.sort(
            key=lambda doc: (
                float(doc.get("retrieval_priority") or 0),
                _score_doc(doc),
            ),
            reverse=True,
        )

    selected: list[dict] = []
    selected_keys: set[str] = set()
    subjects_by_priority = sorted(
        grouped,
        key=lambda subject: (
            float(grouped[subject][0].get("retrieval_priority") or 0),
            _score_doc(grouped[subject][0]),
        ),
        reverse=True,
    )

    for subject in subjects_by_priority:
        if len(selected) >= max_docs:
            break
        doc = grouped[subject][0]
        key = _doc_dedupe_key(doc)
        selected.append(doc)
        selected_keys.add(key)

    remaining = [
        doc
        for subject_docs in grouped.values()
        for doc in subject_docs
        if _doc_dedupe_key(doc) not in selected_keys
    ]
    remaining.sort(
        key=lambda doc: (
            float(doc.get("retrieval_priority") or 0),
            _score_doc(doc),
        ),
        reverse=True,
    )

    for doc in remaining:
        if len(selected) >= max_docs:
            break
        selected.append(doc)
    return selected


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

    log_query_rewrite_result = os.getenv("LOG_QUERY_REWRITE_RESULT", "").strip().lower() == "true"
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

        if log_query_rewrite_result:
            logger.warning("search_query_rewriter raw result preview: %s", raw_preview)
            logger.warning("search_query_rewriter parsing_error: %s", parsing_error)

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
        retrieval_plan, _ = _normalize_retrieval_plan(parsed.retrieval_plan, state)
        primary_subject = _normalize_primary_subject(parsed.primary_subject, retrieval_plan)
        multi_subject_payload = {
            "retrieval_plan": retrieval_plan,
            "learning_goal": parsed.learning_goal.strip(),
            "primary_subject": primary_subject,
            "subject_relation_summary": parsed.subject_relation_summary.strip(),
        }

        if log_query_rewrite_result:
            logger.warning(
                "search_query_rewriter parsed result: %s",
                json.dumps({**result_payload, **multi_subject_payload}, ensure_ascii=False),
            )
    except Exception as exc:
        logger.warning("Initial search query rewrite failed; continuing with original query", exc_info=True)
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
    search_rag_query = state.get("search_rag_query", "")
    expanded_keypoints = state.get("expanded_keypoints", [])
    keypoints = state.get("keypoints", [])
    subject = state.get("subject")

    if rewritten:
        query = rewritten
    elif search_rag_query:
        query = search_rag_query
    elif expanded_keypoints:
        query = " ".join(expanded_keypoints)
    elif keypoints:
        query = " ".join(keypoints)
    else:
        query = _last_human_query(state)

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
                for doc in result.get("docs", []):
                    all_docs.append({
                        "type": "rag",
                        "retrieval_subject": plan_subject,
                        "retrieval_role": item.get("role", "supporting_context"),
                        "retrieval_query": plan_query,
                        "retrieval_purpose": item.get("purpose", ""),
                        "relation_to_goal": item.get("relation_to_goal", ""),
                        "retrieval_priority": item.get("priority", 0.5),
                        **doc,
                    })

            selected_docs = _select_docs_with_subject_quota(all_docs, max_docs)
            span.set_attribute("rag.doc_count", len(selected_docs))
            span.set_attribute("rag.is_hit", bool(selected_docs))
            if selected_docs:
                span.set_attribute("rag.top_score", _score_doc(selected_docs[0]))

        return {"context": selected_docs}

    with traced_retrieval(query=query, subject=subj) as span:
        result = await asyncio.to_thread(retrieve, query=query, subject=subj)
        span.set_attribute("rag.doc_count", len(result.get("docs", [])))
        span.set_attribute("rag.is_hit", result.get("is_hit", False))
        if result.get("docs"):
            span.set_attribute("rag.top_score", result["docs"][0].get("score", 0))

    docs = result["docs"]
    return {"context": [{"type": "rag", **doc} for doc in docs]}


# ── Node 2: web search (parallel branch B) ────────────────────────

_SEARCH_TIMEOUT = get_setting("academic.search_timeout", 15)


@traced_node
async def web_search(state: TutorState) -> dict:
    """Fan-out web search — runs in parallel with rag_retrieve."""
    rewritten = state.get("rewritten_query", "")
    search_web_query = state.get("search_web_query", "")
    retrieval_plan = state.get("retrieval_plan") or []
    if rewritten:
        query = rewritten
    elif search_web_query:
        query = search_web_query
    elif retrieval_plan:
        best_item = max(
            retrieval_plan,
            key=lambda item: float(item.get("priority") or 0),
        )
        query = best_item.get("web_search_query") or best_item.get("rag_query") or _last_human_query(state)
    else:
        query = _last_human_query(state)

    with traced_search(query=query, timeout=_SEARCH_TIMEOUT) as span:
        try:
            search_results = await asyncio.wait_for(
                asyncio.to_thread(web_search_fn, query),
                timeout=_SEARCH_TIMEOUT,
            )
            span.set_attribute("search.result_count", len(search_results))
            span.set_attribute("search.timed_out", False)
        except asyncio.TimeoutError:
            search_results = []
            span.set_attribute("search.result_count", 0)
            span.set_attribute("search.timed_out", True)
        except Exception:
            search_results = []
            span.set_attribute("search.result_count", 0)
            span.set_attribute("search.timed_out", False)

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
            purpose = d.get("retrieval_purpose") or "提供该学科相关课程依据"
            relation = d.get("relation_to_goal") or "与学习目标相关"
            parts.append(
                f"【{subject}｜{role}｜依据】\n"
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

@traced_node
async def evaluate_hallucination(state: TutorState) -> dict:
    """Evaluate whether the generated answer hallucinates beyond retrieved context.

    Uses structured LLM output to judge faithfulness. On detection,
    increments retry_count to signal the conditional edge for re-retrieval.
    Defaults to valid on any parsing/model failure (safe fallback).
    """
    eval_temp = get_setting("academic.hallucination_eval_temperature", 0.0)
    llm = get_node_llm("academic", temperature=eval_temp)
    structured_primary = llm.with_structured_output(HallucinationEvaluation)

    fallback_llm = get_fallback_llm(temperature=eval_temp)
    structured_fallback = fallback_llm.with_structured_output(HallucinationEvaluation)

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

    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        node_name="evaluate_hallucination",
        temperature=eval_temp,
    ) as span:
        try:
            evaluation = await async_invoke_with_fallback(
                structured_primary,
                [
                    SystemMessage(content=load_prompt("hallucination_system")),
                    HumanMessage(content=eval_prompt),
                ],
                fallback=structured_fallback,
                span=span,
            )
            is_faithful = evaluation.is_faithful
        except Exception:
            logger.warning("Hallucination evaluation failed, defaulting to valid")
            is_faithful = True

    hallucination_detected = not is_faithful

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
