"""
Episodic Memory Writer — records learning events and observations.

Each call to write_episodic_memory() creates a persistent EpisodicMemoryRecord
with an embedding vector for future retrieval. Importance is computed from
graph state signals (hallucination, resource generation, etc.).
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import get_setting
from src.memory.errors import MemoryEmbeddingRuntimeError
from src.memory.schema import EpisodicMemoryRecord
from src.memory.storage import MemoryStore, create_memory_store
from src.memory.embeddings import get_embedding_provider

logger = logging.getLogger(__name__)

# Singleton store instance (lazy)
_store: MemoryStore | None = None


def _get_store() -> MemoryStore:
    """Get or create the singleton memory store."""
    global _store
    if _store is None:
        backend = get_setting("memory.backend", "sqlite")
        db_path = get_setting("memory.db_path", "data/memory.db")
        _store = create_memory_store(backend=backend, db_path=db_path)
    return _store


def reset_store() -> None:
    """Reset store singleton (for testing)."""
    global _store
    _store = None


# ── Primary API ────────────────────────────────────────────────────────────


async def write_episodic_memory(
    state: dict[str, Any],
    *,
    memory_type: str,
    content: str,
    importance: float = 0.5,
    subject: str = "",
    metadata: dict[str, Any] | None = None,
    store: MemoryStore | None = None,
) -> EpisodicMemoryRecord:
    """Write one episodic memory event with embedding.

    Args:
        state: Graph state dict (LearningState or similar). Must contain at
               least ``thread_id``. May also contain ``subject``.
        memory_type: One of ``quiz_attempt``, ``learning_behavior``, ``error``,
                     ``key_conversation``, ``system_event``.
        content: Natural language description of the event.
        importance: 0.0–1.0 importance rating.
        subject: Academic subject (auto-detected from state if empty).
        metadata: Optional arbitrary metadata dict.
        store: Optional MemoryStore instance (uses singleton if not provided).

    Returns:
        The persisted EpisodicMemoryRecord.
    """
    store = store or _get_store()

    raw_thread_id = state.get("thread_id")
    if (
        not isinstance(raw_thread_id, str)
        or not raw_thread_id.strip()
        or raw_thread_id != raw_thread_id.strip()
    ):
        raise ValueError("episodic memory requires a normalized thread_id")
    user_id = raw_thread_id

    # Extract subject from state if not provided
    if not subject:
        subject = state.get("subject", "") or state.get("primary_subject", "")

    record = EpisodicMemoryRecord(
        user_id=user_id,
        memory_type=memory_type,
        content=content,
        importance=importance,
        subject=subject,
        metadata=metadata or {},
    )

    # Generate embedding for content. Empty input is not valid for persisted memory.
    provider = get_embedding_provider()
    embeddings = await provider.embed([content])
    if not embeddings or not embeddings[0]:
        raise MemoryEmbeddingRuntimeError(
            "Embedding provider returned no episodic memory embedding"
        )
    record.embedding = embeddings[0]

    # Persist
    try:
        await store.save_episodic(record)
        logger.debug(
            "Wrote episodic memory id=%s type=%s importance=%.2f",
            record.memory_id,
            memory_type,
            importance,
        )
    except Exception:
        logger.exception("Failed to persist episodic memory id=%s", record.memory_id)
        fail_fast = get_setting("memory.fail_fast_memory_write", False)
        if fail_fast:
            raise
        # Non-fatal by default — graph continues even if memory write fails

    return record


# ── Importance Heuristics ─────────────────────────────────────────────────


def compute_importance_from_state(state: dict[str, Any]) -> tuple[float, str, str]:
    """Compute importance, memory_type, and content from graph state.

    Uses signals in the state to determine how notable the current turn is.

    Args:
        state: Full LearningState dict after the graph run.

    Returns:
        Tuple of (importance 0.0–1.0, memory_type, content_description).
    """
    hallucination = state.get("hallucination_detected", False)
    resource_type = state.get("requested_resource_type", "")
    resource_types = state.get("requested_resource_types", [])
    subject = state.get("subject", "") or state.get("primary_subject", "")
    evidence_state = state.get("evidence_judge_state", "")
    retry_count = state.get("retry_count", 0)

    # Hallucination detected → high importance error memory
    if hallucination:
        reason = state.get("hallucination_reason", "")[:200]
        return (
            0.9,
            "error",
            f"回答幻觉检测触发 (重试{retry_count}次): 主题={subject}, 原因={reason}",
        )

    # Resource generation → medium-high importance
    all_resource_types = list(resource_types) if resource_types else []
    if resource_type and resource_type not in all_resource_types:
        all_resource_types.append(resource_type)

    if all_resource_types:
        types_str = "+".join(all_resource_types)
        return (
            0.6,
            "learning_behavior",
            f"生成了{types_str}类型学习资源, 主题={subject}",
        )

    # Evidence insufficient → notable system event
    if evidence_state == "insufficient":
        return (
            0.7,
            "system_event",
            f"证据不足, 主题={subject}, 无法充分回答",
        )

    # Default: standard Q&A
    return (
        0.4,
        "key_conversation",
        f"问答交互, 主题={subject}",
    )


def compute_importance_for_user_query(
    query: str,
    subject: str = "",
    resource_types: list[str] | None = None,
) -> tuple[float, str, str]:
    """Compute importance for the user's input (recorded at stream start).

    Args:
        query: The user's raw query text.
        subject: Detected subject.
        resource_types: Optional resource types requested.

    Returns:
        Tuple of (importance, memory_type, content_description).
    """
    # Resource requests are notable
    if resource_types:
        types_str = "+".join(resource_types)
        return (
            0.5,
            "learning_behavior",
            f"用户请求生成{types_str}学习资源: {query[:200]}",
        )

    # Very short queries are less notable
    if len(query) < 10:
        return (0.2, "key_conversation", f"简短提问: {query}")

    # Standard query
    importance = 0.4
    if len(query) > 100:
        importance = 0.5  # Detailed questions are more notable

    return (
        importance,
        "key_conversation",
        f"用户提问 (主题={subject}): {query[:200]}",
    )
