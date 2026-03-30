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

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from src.config import get_setting, load_prompt
from src.graph.llm import async_invoke_with_fallback, get_fallback_llm, get_node_llm
from src.graph.state import TutorState
from src.rag.retriever import retrieve
from src.tools.search_tool import search as web_search_fn
from src.tracing import traced_llm_call, traced_node, traced_search

logger = logging.getLogger(__name__)


# ── Node 1: search latest Gaokao policies ─────────────────────────

# Time Limit to prevent search too long
_SEARCH_TIMEOUT = get_setting("planner.search_timeout", 15)


@traced_node
async def search_policy(state: TutorState) -> dict:
    """Use DuckDuckGo to fetch the latest Gaokao policy information. Times out after 15s."""
    year = datetime.now().year
    query = f"{year}年高考最新政策 考试时间安排 科目改革"

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


# ── Node 2: generate complete plan ────────────────────────────────

@traced_node
async def generate_plan(state: TutorState) -> dict:
    """Produce a complete study plan from user request + policy context in one LLM call."""
    llm = get_node_llm("planner")

    last_msg = state["messages"][-1]
    user_request = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    search_results = state.get("search_results", [])
    policy_info = "\n\n".join(
        f"- {r.get('title', '无标题')}: {r.get('content', '')}"
        for r in search_results
    ) if search_results else "未获取到最新政策信息，请基于通用经验给出建议。"

    prompt = load_prompt("planner_generate").format(
        user_request=user_request,
        policy_info=policy_info,
    )

    temperature = get_setting("planner.temperature", 0.7)
    fallback = get_fallback_llm(temperature=temperature)
    messages = [
        SystemMessage(content=load_prompt("planner_system")),
        HumanMessage(content=prompt),
    ]

    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        node_name="generate_plan",
        temperature=temperature,
    ) as span:
        response = await async_invoke_with_fallback(
            llm, messages, fallback=fallback, span=span,
        )

    return {"messages": [AIMessage(content=response.content)]}


# ── Node: gather_intel (Phase2a — parallel fan-out) ──────────────

def _last_human_query(state: TutorState) -> str:
    """Extract the last HumanMessage content."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


async def _gather_emotional_intel(state: TutorState) -> str:
    """Call LLM to summarize user's emotional state from conversation history."""
    llm = get_node_llm("emotional")
    fallback = get_fallback_llm(temperature=get_setting("emotional.temperature", 0.8))

    history_text = "\n".join(
        f"{'学生' if isinstance(m, HumanMessage) else '老师'}: {m.content}"
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
    query = _last_human_query(state)
    subject = state.get("subject")
    subj = subject if subject and subject != "other" else None

    async def _rag():
        try:
            result = retrieve(query=query, subject=subj)
            docs = result.get("docs", [])
            if not docs:
                return ""
            parts = [f"- {d.get('content', '')[:200]}" for d in docs[:3]]
            return "【知识库资源】\n" + "\n".join(parts)
        except Exception:
            logger.warning("RAG retrieval failed in gather_intel", exc_info=True)
            return ""

    async def _web():
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(web_search_fn, query),
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
    }


# ── Node: plan_adversarial_node (Phase2b — SubGraph wrapper) ─────

@traced_node
async def plan_adversarial_node(state: TutorState) -> dict:
    """Invoke the adversarial planning SubGraph and return the final plan.

    Bridges TutorState → PlanAdversarialState → TutorState.
    """
    from src.graph.plan_adversarial import PlanAdversarialState, build_adversarial_subgraph

    user_request = _last_human_query(state)
    intel_summary = state.get("intel_summary", "")
    max_rounds = get_setting("planner.adversarial_max_rounds", 3)

    sub_input: PlanAdversarialState = {
        "intel_summary": intel_summary,
        "user_request": user_request,
        "draft": "",
        "academic_verdict": "",
        "emotional_verdict": "",
        "round": 0,
        "max_rounds": max_rounds,
        "consensus": False,
        "revision_notes": "",
    }

    sub_graph = build_adversarial_subgraph()
    result = await sub_graph.ainvoke(sub_input)

    plan_text = result.get("draft", "")

    # HIL: pause for human review. interrupt() returns the user's edited plan
    # when resumed via Command(resume=edited_plan), or the original draft
    # on first invocation (before resume).
    edited_plan = interrupt(plan_text)

    final_plan = edited_plan if isinstance(edited_plan, str) and edited_plan else plan_text
    return {
        "plan": final_plan,
        "messages": [AIMessage(content=final_plan)],
    }
