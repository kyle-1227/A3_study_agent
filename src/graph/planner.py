"""SubGraph B — Study Planner: policy search, then single-call plan generation.

Notes:
- 2-step (search → generate)

Future Roadmap:
- [Local RAG] Replace/Augment web search with a VectorDB (ChromaDB) containing
  official PDF documents from provincial education examination authorities.
- [Context Filtering] Implement a re-ranking stage to prioritize official
  .gov.cn domains over social media/marketing content.
- [Provincial Routing] Automatically inject user's provincial context into
  search queries to handle diverse Gaokao schemas (e.g., 3+1+2 vs. 3+3).
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_setting, load_prompt
from src.graph.llm import async_invoke_with_fallback, get_fallback_llm, get_node_llm
from src.graph.state import TutorState
from src.observability.a3_trace import emit_a3_trace
from src.rag.retriever import retrieve
from src.tools.search_tool import search as web_search_fn
from src.tracing import traced_llm_call, traced_node, traced_search

logger = logging.getLogger(__name__)


# ── Node 1: search latest Gaokao policies ─────────────────────────

# Time Limit to prevent search too long
_SEARCH_TIMEOUT = get_setting("planner.search_timeout", 15)


@traced_node
async def search_policy(state: TutorState) -> dict:
    """Use the configured Web Search provider to fetch policy information. Times out after 15s."""
    year = datetime.now().year
    query = state.get("search_web_query") or f"{year}年高校课程学习资源 专业入门路径"

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

    return {"search_results": search_results}


# ── Node: gather_intel (Phase2a — parallel fan-out) ──────────────

def _last_human_query(state: TutorState) -> str:
    """Extract the last HumanMessage content."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _build_planning_retrieval_query(state: TutorState) -> str:
    """Build a planning retrieval query without subject-specific hardcoding."""
    search_rag_query = state.get("search_rag_query", "")
    expanded_keypoints = state.get("expanded_keypoints", [])
    keypoints = state.get("keypoints", [])

    if search_rag_query:
        return search_rag_query
    if expanded_keypoints:
        return " ".join(str(item) for item in expanded_keypoints if str(item).strip())
    if keypoints:
        return " ".join(str(item) for item in keypoints if str(item).strip())
    return _last_human_query(state)


async def _gather_emotional_intel(state: TutorState) -> str:
    """Call LLM to summarize user's emotional state from conversation history."""
    llm = get_node_llm("emotional")
    fallback = get_fallback_llm(temperature=get_setting("emotional.temperature", 0.8))

    history_text = "\n".join(
        f"{'学习者' if isinstance(m, HumanMessage) else '学业导师'}: {m.content}"
        for m in state["messages"]
        if hasattr(m, "content")
    )

    messages = [
        SystemMessage(content=load_prompt("gather_emotional_intel")),
        HumanMessage(content=history_text),
    ]

    try:
        with traced_llm_call(
            model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            node_name="gather_emotional_intel",
            temperature=get_setting("emotional.temperature", 0.8),
        ) as span:
            response = await async_invoke_with_fallback(
                llm, messages, fallback=fallback, span=span,
            )
        return response.content.strip()
    except Exception:
        logger.warning("Emotional intel LLM call failed, using fallback", exc_info=True)
        return "无法获取情绪分析，建议按常规方式安排计划。"


async def _gather_resource_intel(state: TutorState) -> str:
    """Retrieve RAG + web search results in parallel, format as resource summary."""
    query = _build_planning_retrieval_query(state)
    web_query = state.get("search_web_query") or query
    subject = state.get("subject")
    subj = subject if subject and subject != "other" else None
    retrieval_plan = state.get("retrieval_plan") or []
    rag_sections_count = 0

    async def _rag():
        nonlocal rag_sections_count
        try:
            if retrieval_plan:
                sections: list[str] = []
                top_k = get_setting("rag.multi_subject_per_subject_top_k", 3)
                for item in retrieval_plan:
                    plan_subject = str(item.get("subject") or "").strip()
                    plan_query = str(item.get("rag_query") or "").strip()
                    if not plan_subject or not plan_query:
                        continue

                    result = await asyncio.to_thread(
                        retrieve,
                        query=plan_query,
                        subject=plan_subject,
                        top_k=top_k,
                    )
                    docs = result.get("docs", [])
                    if not docs:
                        continue

                    role = item.get("role") or "supporting_context"
                    purpose = item.get("purpose") or "补充学习规划所需课程依据"
                    relation = item.get("relation_to_goal") or "服务学习目标"
                    doc_lines = [
                        f"  - {d.get('content', '')[:220]}"
                        for d in docs[:2]
                    ]
                    rag_sections_count += 1
                    sections.append(
                        f"【知识库资源｜{plan_subject}｜{role}】\n"
                        f"用途：{purpose}\n"
                        f"关系：{relation}\n"
                        f"检索 query：{plan_query}\n"
                        + "\n".join(doc_lines)
                    )

                return "\n\n".join(sections)

            result = await asyncio.to_thread(retrieve, query=query, subject=subj)
            docs = result.get("docs", [])
            if not docs:
                return ""
            rag_sections_count = 1
            parts = [f"- {d.get('content', '')[:200]}" for d in docs[:3]]
            return "【知识库资源】\n" + "\n".join(parts)
        except Exception:
            logger.warning("RAG retrieval failed in gather_intel", exc_info=True)
            return ""

    async def _web():
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(web_search_fn, web_query),
                timeout=_SEARCH_TIMEOUT,
            )
            if not results:
                return ""
            parts = [f"- {r.get('title', '')}: {r.get('content', '')[:200]}" for r in results[:3]]
            return "【网络搜索】\n" + "\n".join(parts)
        except Exception:
            logger.warning("Web search failed in gather_intel", exc_info=True)
            return ""

    rag_text, web_text = await asyncio.gather(_rag(), _web())
    # TEMP A3_TRACE: remove after multi-subject retrieval validation.
    emit_a3_trace(
        logger,
        "planning_gather_intel",
        {
            "mode": "multi_subject" if retrieval_plan else "single_query",
            "retrieval_plan_count": len(retrieval_plan),
            "subjects": [item.get("subject") for item in retrieval_plan],
            "roles": [item.get("role") for item in retrieval_plan],
            "rag_sections_count": rag_sections_count,
            "web_query": web_query,
        },
        state=state,
        env_flag="LOG_PLANNING_INTEL",
    )

    combined = "\n\n".join(part for part in [rag_text, web_text] if part)
    return combined if combined else "未获取到相关资源信息。"


@traced_node
async def gather_intel(state: TutorState) -> dict:
    """Phase2a: Gather emotional + resource intel in parallel.

    Stores emotional_intel, resource_intel, and combined intel_summary
    in TutorState for the adversarial planning SubGraph.
    """
    emotional_intel, resource_intel = await asyncio.gather(
        _gather_emotional_intel(state),
        _gather_resource_intel(state),
    )

    intel_summary = f"【情绪分析】\n{emotional_intel}\n\n{resource_intel}"

    return {
        "emotional_intel": emotional_intel,
        "resource_intel": resource_intel,
        "intel_summary": intel_summary,
        # Initialize adversarial planning state
        "adv_round": 0,
        "draft": "",
        "academic_verdict": "",
        "academic_reason": "",
        "emotional_verdict": "",
        "emotional_reason": "",
        "consensus": False,
        "revision_notes": "",
    }
