"""
Memory Consolidation & Forgetting — background tasks for memory hygiene.

Features:
- maybe_consolidate(): triggers episodic → semantic summarization when
  unconsolidated count reaches threshold.
- apply_forgetting(): deletes old low-importance memories and merges
  near-duplicate memories based on embedding similarity.

Both are designed to be called as fire-and-forget background tasks
(via asyncio.create_task()) so they never block the request path.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.config import get_setting
from src.memory.retrieval import _cosine_similarity
from src.memory.retention import (
    PROTECTED_EPISODIC_MEMORY_ID_PREFIXES,
    is_protected_episodic_memory_id,
)
from src.memory.semantic import consolidate_episodic_to_semantic
from src.memory.storage import MemoryStore, create_memory_store

logger = logging.getLogger(__name__)


async def maybe_consolidate(
    user_id: str,
    store: MemoryStore | None = None,
) -> int:
    """Check if consolidation is needed and run it.

    Consolidation runs when the number of unconsolidated episodic memories
    reaches the configured threshold.

    Args:
        user_id: The user to check.
        store: Optional MemoryStore (uses singleton if not provided).

    Returns:
        Number of new semantic summaries created (0 or 1).
    """
    store = store or create_memory_store()

    threshold = int(get_setting(
        "memory.consolidation_episodic_threshold", 5,
    ))
    max_per_batch = int(get_setting(
        "memory.consolidation_max_per_batch", 5,
    ))

    unconsolidated = await store.get_unconsolidated(
        user_id,
        limit=threshold,
        excluded_memory_id_prefixes=PROTECTED_EPISODIC_MEMORY_ID_PREFIXES,
    )

    if len(unconsolidated) < threshold:
        logger.debug(
            "Consolidation not triggered: %d/%d unconsolidated for user=%s",
            len(unconsolidated), threshold, user_id,
        )
        return 0

    logger.info(
        "Running consolidation for user=%s: %d unconsolidated (threshold=%d)",
        user_id, len(unconsolidated), threshold,
    )

    summary = await consolidate_episodic_to_semantic(
        user_id, store=store, max_episodic=max_per_batch,
    )
    return 1 if summary else 0


async def apply_forgetting(
    user_id: str,
    store: MemoryStore | None = None,
) -> dict[str, int]:
    """Apply forgetting rules to keep the memory store clean.

    1. **Age-based forgetting**: Delete episodic memories older than
       ``forgetting_retention_days`` with importance below
       ``forgetting_importance_min``.

    2. **Duplicate detection**: Find and merge near-duplicate episodic
       memories (cosine similarity above ``forgetting_duplicate_similarity``).
       Keeps the more recent one, deletes the older.

    Args:
        user_id: The user to apply forgetting to.
        store: Optional MemoryStore instance.

    Returns:
        Dict with stats: ``{"low_importance_deleted": N, "duplicates_merged": M}``
    """
    store = store or create_memory_store()

    retention_days = int(get_setting("memory.forgetting_retention_days", 30))
    importance_min = float(get_setting("memory.forgetting_importance_min", 0.2))
    duplicate_threshold = float(get_setting("memory.forgetting_duplicate_similarity", 0.95))

    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()

    # ── Age-based forgetting ──────────────────────────────────────────────
    low_importance_deleted = 0
    try:
        low_importance_deleted = await store.delete_low_importance_old(
            user_id,
            before_ts=cutoff,
            importance_max=importance_min,
            excluded_memory_id_prefixes=PROTECTED_EPISODIC_MEMORY_ID_PREFIXES,
        )
    except Exception as exc:
        logger.warning(
            "Age-based forgetting failed for user=%s: %s", user_id, exc,
        )

    # ── Duplicate detection ───────────────────────────────────────────────
    duplicates_merged = 0
    try:
        all_records = [
            record
            for record in await store.get_all_episodic_for_user(user_id, limit=200)
            if not is_protected_episodic_memory_id(record.memory_id)
        ]

        # Group records with embeddings (only compare embeddable ones)
        embeddable = [r for r in all_records if r.embedding and len(r.embedding) > 0]
        non_embeddable = [r for r in all_records if not r.embedding or len(r.embedding) == 0]

        # For non-embeddable records, do simple content-based dedup
        seen_contents: set[str] = set()
        for rec in non_embeddable:
            content_key = rec.content[:100].strip().lower()
            if content_key in seen_contents:
                await store.delete_episodic(rec.memory_id)
                duplicates_merged += 1
            else:
                seen_contents.add(content_key)

        # For embeddable records, compare pairwise
        for i in range(len(embeddable)):
            if not embeddable[i].embedding:
                continue
            for j in range(i + 1, len(embeddable)):
                if not embeddable[j].embedding:
                    continue
                sim = _cosine_similarity(
                    embeddable[i].embedding, embeddable[j].embedding,
                )
                if sim > duplicate_threshold:
                    # Keep the more recent one, delete the older
                    a = embeddable[i]
                    b = embeddable[j]
                    if a.created_at < b.created_at:
                        await store.delete_episodic(a.memory_id)
                        embeddable[i] = embeddable[j]  # Replace with kept one for future comparisons
                    else:
                        await store.delete_episodic(b.memory_id)
                    duplicates_merged += 1
                    break  # Only delete one per pair

    except Exception as exc:
        logger.warning(
            "Duplicate detection forgetting failed for user=%s: %s", user_id, exc,
        )

    if low_importance_deleted or duplicates_merged:
        logger.info(
            "Forgetting applied for user=%s: %d low-importance deleted, %d duplicates merged",
            user_id, low_importance_deleted, duplicates_merged,
        )

    return {
        "low_importance_deleted": low_importance_deleted,
        "duplicates_merged": duplicates_merged,
    }


async def run_consolidation_and_forgetting(
    user_id: str,
    store: MemoryStore | None = None,
) -> dict:
    """Run full memory maintenance cycle: consolidate + forget.

    Designed for fire-and-forget background execution.

    Args:
        user_id: The user to maintain.
        store: Optional MemoryStore instance.

    Returns:
        Dict with combined stats from both operations.
    """
    store = store or create_memory_store()

    summary_count = await maybe_consolidate(user_id, store=store)
    forgetting_stats = await apply_forgetting(user_id, store=store)

    return {
        "summaries_created": summary_count,
        **forgetting_stats,
    }
