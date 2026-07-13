"""
Semantic Memory — LLM-powered consolidation of episodic events into
structured long-term summaries.

The core function, consolidate_episodic_to_semantic(), takes N unconsolidated
episodic memories, feeds them to an LLM via invoke_structured_llm(), and
produces a SemanticMemorySummary with extracted weak points, style changes,
and skill growth trajectory.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_setting
from src.llm.structured_output import (
    invoke_structured_llm,
    get_llm_output_mode,
    get_max_raw_chars,
)
from src.memory.embeddings import get_embedding_provider
from src.memory.prompts import SEMANTIC_SUMMARY_SYSTEM_PROMPT
from src.memory.schema import (
    EpisodicMemoryRecord,
    SemanticMemorySummary,
    SemanticSummaryStrictOutput,
)
from src.memory.storage import MemoryStore, create_memory_store

logger = logging.getLogger(__name__)

# LLM node name for the semantic summarizer (must match settings.yaml)
_LLM_NODE = "memory_semantic_summarizer"


async def consolidate_episodic_to_semantic(
    user_id: str,
    store: MemoryStore | None = None,
    max_episodic: int | None = None,
) -> SemanticMemorySummary | None:
    """Take the oldest N unconsolidated episodic memories and summarize them.

    Calls the LLM (via invoke_structured_llm) to produce a structured semantic
    summary. Embeds the resulting summary for future retrieval.

    Args:
        user_id: The user whose memories to consolidate.
        store: Optional MemoryStore (uses singleton if not provided).
        max_episodic: Max episodic events per batch. Default from settings.

    Returns:
        A new SemanticMemorySummary, or None if not enough unconsolidated
        memories exist or the LLM call fails.
    """
    store = store or create_memory_store()

    if max_episodic is None:
        max_episodic = int(get_setting(
            "memory.consolidation_max_per_batch", 5,
        ))

    # Fetch oldest unconsolidated episodic memories
    unconsolidated = await store.get_unconsolidated(user_id, limit=max_episodic)

    if len(unconsolidated) < 2:
        logger.debug(
            "Not enough unconsolidated memories for user=%s (%d < 2), skipping",
            user_id, len(unconsolidated),
        )
        return None

    # Format episodic contents for the LLM
    episodic_texts = _format_episodic_for_llm(unconsolidated)
    user_prompt = (
        f"将以下 {len(unconsolidated)} 个学习事件整合为语义记忆摘要:\n\n"
        f"{episodic_texts}"
    )

    # Call LLM for structured summarization
    try:
        result = await invoke_structured_llm(
            node_name=_LLM_NODE,
            llm_node=_LLM_NODE,
            schema=SemanticSummaryStrictOutput,
            messages=[
                SystemMessage(content=SEMANTIC_SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ],
            output_mode=get_llm_output_mode(_LLM_NODE),
            state={"thread_id": user_id},
            max_raw_chars=get_max_raw_chars(_LLM_NODE),
        )
    except Exception as exc:
        logger.exception(
            "LLM call failed for memory consolidation user=%s", user_id,
        )
        return None

    if not result.success or result.parsed is None:
        logger.warning(
            "Memory consolidation LLM returned no valid output for user=%s",
            user_id,
        )
        return None

    parsed = result.parsed
    if not isinstance(parsed, SemanticSummaryStrictOutput):
        logger.error(
            "Memory consolidation parsed result is not SemanticSummaryStrictOutput: %s",
            type(parsed),
        )
        return None

    # Generate embedding for the summary
    embedding = None
    try:
        provider = get_embedding_provider()
        embed_text = _build_embed_text(parsed)
        embeddings = await provider.embed([embed_text])
        if embeddings and embeddings[0]:
            embedding = embeddings[0]
    except Exception as exc:
        logger.debug("Failed to embed semantic summary: %s", exc)

    # Build and persist the summary
    summary = SemanticMemorySummary(
        user_id=user_id,
        source_episodic_ids=[m.memory_id for m in unconsolidated],
        content=parsed.content,
        weak_knowledge_points=list(parsed.weak_knowledge_points),
        learning_style_changes=parsed.learning_style_changes,
        skill_growth_trajectory=parsed.skill_growth_trajectory,
        embedding=embedding,
        confidence=parsed.confidence,
    )

    try:
        await store.save_semantic(summary)
        group_id = summary.summary_id
        await store.mark_consolidated(
            [m.memory_id for m in unconsolidated], group_id,
        )
        logger.info(
            "Consolidated %d episodic → semantic summary id=%s for user=%s (confidence=%.2f)",
            len(unconsolidated), summary.summary_id, user_id, summary.confidence,
        )
    except Exception as exc:
        logger.exception(
            "Failed to persist semantic summary for user=%s", user_id,
        )
        return None

    return summary


# ── Helpers ────────────────────────────────────────────────────────────────


def _format_episodic_for_llm(episodics: list[EpisodicMemoryRecord]) -> str:
    """Format episodic memories as a readable text block for LLM input."""
    parts: list[str] = []
    for i, mem in enumerate(episodics, 1):
        date_part = mem.created_at[:10] if mem.created_at else "unknown"
        parts.append(
            f"[{i}] type={mem.memory_type} date={date_part} "
            f"importance={mem.importance:.2f} subject={mem.subject or 'general'}\n"
            f"    content: {mem.content}"
        )
    return "\n\n".join(parts)


def _build_embed_text(parsed: SemanticSummaryStrictOutput) -> str:
    """Build a single text string for embedding the semantic summary.

    Concatenates all structured fields into a single dense representation.
    """
    parts = [parsed.content]
    if parsed.weak_knowledge_points:
        parts.append("薄弱知识点: " + ", ".join(parsed.weak_knowledge_points))
    if parsed.learning_style_changes:
        parts.append("学习风格变化: " + parsed.learning_style_changes)
    if parsed.skill_growth_trajectory:
        parts.append("技能成长轨迹: " + parsed.skill_growth_trajectory)
    return " ".join(parts)
