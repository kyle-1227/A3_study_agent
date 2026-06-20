"""
Context Builder — memory-augmented system prompt assembly for LLM calls.

This is the central integration point where memory retrieval meets prompt
engineering. Every LLM node that needs memory context calls build_memory_context()
to get a formatted prefix string that is prepended to the node's system prompt.

Flow:
    current_query → retrieve_top_k_memories() → format sections → fit to budget
    → return [记忆上下文] prefix string
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.config import get_setting
from src.context.token_manager import TokenBudget, estimate_tokens, fit_to_budget
from src.memory.embeddings import get_embedding_provider
from src.memory.prompts import (
    MEMORY_CONTEXT_EPISODIC_HEADER,
    MEMORY_CONTEXT_HEADER,
    MEMORY_CONTEXT_SEMANTIC_HEADER,
    MEMORY_INFLUENCE_EXPLANATION_TEMPLATE,
)
from src.memory.retrieval import retrieve_top_k_memories
from src.memory.schema import (
    EpisodicMemoryRecord,
    MemoryContextInjection,
    MemoryRetrievalResult,
    SemanticMemorySummary,
)
from src.memory.storage import create_memory_store

logger = logging.getLogger(__name__)


# ── Primary API ────────────────────────────────────────────────────────────


async def build_memory_context(
    user_id: str,
    current_query: str,
    *,
    subject: str = "",
    profile_context: str = "",
    conversation_summary: str = "",
    budget: TokenBudget | None = None,
    top_k_episodic: int | None = None,
    top_k_semantic: int | None = None,
) -> MemoryContextInjection:
    """Build the memory-augmented context prefix for LLM system messages.

    Retrieves top-K episodic and semantic memories, formats them into a
    structured markdown block, and fits each section to its token budget.

    Args:
        user_id: The user/thread identifier for memory retrieval.
        current_query: The current user query to match against memories.
        subject: Current academic subject (improves retrieval accuracy).
        profile_context: Already-formatted user profile text (from ProfileManager).
        conversation_summary: Recent conversation summary (from graph state).
        budget: Token budget configuration. Loaded from settings if None.
        top_k_episodic: Max episodic memories to include. From settings if None.
        top_k_semantic: Max semantic summaries to include. From settings if None.

    Returns:
        MemoryContextInjection with the formatted context text and retrieval results.
    """
    start_time = time.monotonic()
    budget = budget or TokenBudget.from_settings()

    if top_k_episodic is None:
        top_k_episodic = int(get_setting("memory.retrieval.top_k_episodic", 3))
    if top_k_semantic is None:
        top_k_semantic = int(get_setting("memory.retrieval.top_k_semantic", 2))

    # ── 1. Retrieve memories ──────────────────────────────────────────
    all_results = await retrieve_top_k_memories(
        user_id=user_id,
        query=current_query,
        top_k=top_k_episodic + top_k_semantic,
        include_episodic=True,
        include_semantic=True,
    )

    episodic_results = [r for r in all_results if r.memory_type == "episodic"][:top_k_episodic]
    semantic_results = [r for r in all_results if r.memory_type == "semantic"][:top_k_semantic]

    # ── 2. Build sections ─────────────────────────────────────────────
    sections: list[str] = []

    # Episodic memories section
    if episodic_results:
        lines: list[str] = []
        for i, result in enumerate(episodic_results, 1):
            mem = result.memory
            if isinstance(mem, EpisodicMemoryRecord):
                date_str = mem.created_at[:10] if mem.created_at else ""
                score_str = f"(relevance={result.score:.2f})"
                lines.append(
                    f"  {i}. [{mem.memory_type}] {date_str} {score_str}\n"
                    f"     {mem.content[:300]}"
                )
        episodic_text = f"{MEMORY_CONTEXT_EPISODIC_HEADER}\n" + "\n".join(lines)
        episodic_text = fit_to_budget(episodic_text, budget.episodic_memories)
        sections.append(episodic_text)

    # Semantic memories section
    if semantic_results:
        lines = []
        for i, result in enumerate(semantic_results, 1):
            mem = result.memory
            if isinstance(mem, SemanticMemorySummary):
                lines.append(f"  {i}. {mem.content[:400]}")
                if mem.weak_knowledge_points:
                    lines.append(f"     薄弱点: {', '.join(mem.weak_knowledge_points[:5])}")
        if lines:
            semantic_text = f"{MEMORY_CONTEXT_SEMANTIC_HEADER}\n" + "\n".join(lines)
            semantic_text = fit_to_budget(semantic_text, budget.semantic_summary)
            sections.append(semantic_text)

    # Conversation summary section
    if conversation_summary:
        conv_text = fit_to_budget(
            f"近期对话概要: {conversation_summary}",
            budget.conversation_summary,
        )
        sections.append(conv_text)

    # Profile context (already formatted by ProfileManager, just pass through)
    if profile_context:
        profile_text = fit_to_budget(profile_context, budget.user_profile)
        sections.append(profile_text)

    # ── 3. Assemble ───────────────────────────────────────────────────
    if not sections:
        return MemoryContextInjection(
            context_text="",
            episodic_results=episodic_results,
            semantic_results=semantic_results,
            total_estimated_tokens=0,
            retrieval_time_ms=(time.monotonic() - start_time) * 1000,
        )

    context_text = f"{MEMORY_CONTEXT_HEADER}\n\n" + "\n\n".join(sections)

    retrieval_time_ms = (time.monotonic() - start_time) * 1000

    return MemoryContextInjection(
        context_text=context_text,
        episodic_results=episodic_results,
        semantic_results=semantic_results,
        total_estimated_tokens=estimate_tokens(context_text),
        retrieval_time_ms=retrieval_time_ms,
    )


# ── Memory Influence Explanation ──────────────────────────────────────────


def build_memory_explanation(
    results: list[MemoryRetrievalResult],
    *,
    max_sources: int = 3,
) -> str:
    """Generate a user-facing footer explaining how memory influenced the response.

    This is appended to AI responses to make memory influence transparent.
    It shows which historical memories were retrieved and why they were relevant.

    Args:
        results: The MemoryRetrievalResult list used for this response.
        max_sources: Max number of memory sources to mention.

    Returns:
        Formatted markdown footer string, or empty string if no memories used.
    """
    if not results:
        return ""

    relevant = [r for r in results if r.score > 0.1][:max_sources]
    if not relevant:
        return ""

    items: list[str] = []
    for r in relevant:
        mem = r.memory
        content_preview = (
            mem.content[:120] + "..." if len(mem.content) > 120 else mem.content
        )
        match_label = _match_reason_label(r.match_reason)
        items.append(
            f"- {match_label} (score={r.score:.2f}): {content_preview}"
        )

    return MEMORY_INFLUENCE_EXPLANATION_TEMPLATE.format(items="\n".join(items))


def format_memory_context_for_llm_node(
    injection: MemoryContextInjection,
    *,
    verbose: bool = True,
) -> str | None:
    """Format a MemoryContextInjection for insertion into an LLM system message.

    Args:
        injection: The result from build_memory_context().
        verbose: If True, includes section headers. If False, just the raw text.

    Returns:
        Formatted string or None if the injection is empty.
    """
    if not injection.context_text:
        return None

    if verbose:
        return injection.context_text

    # Compact mode: strip headers, just give the memory content lines
    lines = injection.context_text.split("\n")
    compact = [
        line for line in lines
        if not line.startswith("[记忆") and not line.startswith("相关学习") and not line.startswith("知识摘要")
        and line.strip()
    ]
    return "\n".join(compact) if compact else None


# ── Helpers ────────────────────────────────────────────────────────────────


def _match_reason_label(reason: str) -> str:
    """Convert match_reason code to human-readable Chinese label."""
    label_map = {
        "keyword_overlap": "关键词匹配",
        "vector_similarity": "语义相似",
        "high_importance": "高重要性记忆",
        "fallback": "历史记录",
    }
    parts = reason.split("+")
    labels = [label_map.get(p, p) for p in parts]
    return "+".join(labels)
