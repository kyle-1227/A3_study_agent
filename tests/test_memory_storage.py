"""Tests for memory storage layer (SQLiteMemoryStore CRUD)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.memory.schema import EpisodicMemoryRecord, SemanticMemorySummary
from src.memory.storage import SQLiteMemoryStore, create_memory_store


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_memory.db"


@pytest.fixture
async def store(test_db_path: Path) -> SQLiteMemoryStore:
    """Provide a fresh SQLiteMemoryStore for each test."""
    s = SQLiteMemoryStore(db_path=test_db_path)
    yield s
    # Cleanup
    test_db_path.unlink(missing_ok=True)


@pytest.mark.anyio
async def test_save_and_load_episodic(store: SQLiteMemoryStore):
    """Basic episodic CRUD: save, query, get by id."""
    rec = EpisodicMemoryRecord(
        user_id="user1",
        memory_type="key_conversation",
        content="Test memory content",
        importance=0.7,
        subject="python",
    )
    await store.save_episodic(rec)

    # Query
    results = await store.query_episodic("user1", limit=10)
    assert len(results) == 1
    assert results[0].memory_id == rec.memory_id
    assert results[0].content == rec.content
    assert results[0].importance == 0.7

    # Get by ID
    by_id = await store.get_episodic_by_ids([rec.memory_id])
    assert len(by_id) == 1
    assert by_id[0].memory_id == rec.memory_id


@pytest.mark.anyio
async def test_episodic_query_filters(store: SQLiteMemoryStore):
    """Test episodic query with various filters."""
    # Create diverse records
    rec1 = EpisodicMemoryRecord(
        user_id="user1", memory_type="key_conversation", content="Q&A session",
        importance=0.5, created_at="2025-01-01T00:00:00Z",
    )
    rec2 = EpisodicMemoryRecord(
        user_id="user1", memory_type="error", content="Hallucination detected",
        importance=0.9, created_at="2025-06-01T00:00:00Z",
    )
    rec3 = EpisodicMemoryRecord(
        user_id="user2", memory_type="key_conversation", content="Other user",
        importance=0.3, created_at="2025-03-01T00:00:00Z",
    )
    for r in [rec1, rec2, rec3]:
        await store.save_episodic(r)

    # Filter by memory_type
    errors = await store.query_episodic("user1", memory_type="error")
    assert len(errors) == 1
    assert errors[0].memory_type == "error"

    # Filter by time range
    mid = await store.query_episodic(
        "user1", start_time="2025-02-01T00:00:00Z", end_time="2025-12-01T00:00:00Z",
    )
    assert len(mid) == 1
    assert mid[0].memory_id == rec2.memory_id

    # Filter by importance
    important = await store.query_episodic("user1", importance_min=0.8)
    assert len(important) == 1
    assert important[0].importance == 0.9


@pytest.mark.anyio
async def test_save_and_load_semantic(store: SQLiteMemoryStore):
    """Basic semantic CRUD: save, query."""
    sem = SemanticMemorySummary(
        user_id="user1",
        content="User struggles with recursion but excels at data structures",
        source_episodic_ids=["ep1", "ep2"],
        weak_knowledge_points=["recursion", "dynamic programming"],
        learning_style_changes="Prefers more examples now",
        skill_growth_trajectory="Python proficiency improved from 0.3 to 0.6",
        confidence=0.8,
    )
    await store.save_semantic(sem)

    results = await store.get_semantic_for_user("user1", limit=10)
    assert len(results) == 1
    assert results[0].summary_id == sem.summary_id
    assert results[0].content == sem.content
    assert "recursion" in results[0].weak_knowledge_points
    assert results[0].confidence == 0.8


@pytest.mark.anyio
async def test_unconsolidated_and_mark(store: SQLiteMemoryStore):
    """Test unconsolidated tracking and mark_consolidated."""
    # Create 3 episodic memories
    for i in range(3):
        rec = EpisodicMemoryRecord(
            user_id="user1", memory_type="key_conversation",
            content=f"Memory {i}", importance=0.5,
        )
        await store.save_episodic(rec)

    # All should be unconsolidated
    uncons = await store.get_unconsolidated("user1", limit=10)
    assert len(uncons) == 3

    # Mark 2 as consolidated
    ids = [r.memory_id for r in uncons[:2]]
    await store.mark_consolidated(ids, "group-1")

    # Only 1 should remain unconsolidated
    uncons2 = await store.get_unconsolidated("user1", limit=10)
    assert len(uncons2) == 1


@pytest.mark.anyio
async def test_delete_episodic(store: SQLiteMemoryStore):
    """Test episodic deletion."""
    rec = EpisodicMemoryRecord(
        user_id="user1", memory_type="key_conversation", content="Delete me",
        importance=0.1,
    )
    await store.save_episodic(rec)

    count_before = await store.get_episodic_count("user1")
    assert count_before == 1

    ok = await store.delete_episodic(rec.memory_id)
    assert ok is True

    count_after = await store.get_episodic_count("user1")
    assert count_after == 0

    # Delete non-existent
    ok2 = await store.delete_episodic("nonexistent")
    assert ok2 is False


@pytest.mark.anyio
async def test_delete_low_importance_old(store: SQLiteMemoryStore):
    """Test batch deletion of old, low-importance memories."""
    # Old, low importance → should be deleted
    rec1 = EpisodicMemoryRecord(
        user_id="user1", memory_type="key_conversation",
        content="Old and trivial", importance=0.1,
        created_at="2024-01-01T00:00:00Z",
    )
    # New, low importance → should NOT be deleted
    rec2 = EpisodicMemoryRecord(
        user_id="user1", memory_type="key_conversation",
        content="New but trivial", importance=0.1,
        created_at="2026-06-01T00:00:00Z",
    )
    # Old, high importance → should NOT be deleted
    rec3 = EpisodicMemoryRecord(
        user_id="user1", memory_type="error",
        content="Old but important", importance=0.9,
        created_at="2024-01-01T00:00:00Z",
    )
    for r in [rec1, rec2, rec3]:
        await store.save_episodic(r)

    # Delete records before 2025-01-01 with importance < 0.5
    deleted = await store.delete_low_importance_old(
        "user1", before_ts="2025-01-01T00:00:00Z", importance_max=0.5,
    )
    assert deleted == 1  # Only rec1 should be deleted

    remaining = await store.get_all_episodic_for_user("user1")
    assert len(remaining) == 2


@pytest.mark.anyio
async def test_count_methods(store: SQLiteMemoryStore):
    """Test episodic and semantic counts."""
    assert await store.get_episodic_count("user1") == 0
    assert await store.get_semantic_count("user1") == 0

    rec = EpisodicMemoryRecord(
        user_id="user1", memory_type="key_conversation", content="test",
    )
    await store.save_episodic(rec)
    assert await store.get_episodic_count("user1") == 1

    sem = SemanticMemorySummary(user_id="user1", content="test summary")
    await store.save_semantic(sem)
    assert await store.get_semantic_count("user1") == 1


@pytest.mark.anyio
async def test_factory_creates_sqlite():
    """Factory creates SQLiteMemoryStore by default."""
    store = create_memory_store(backend="sqlite", db_path=":memory:")
    assert isinstance(store, SQLiteMemoryStore)


def test_factory_rejects_unknown():
    """Factory raises for unknown backend."""
    with pytest.raises(ValueError, match="Unknown memory store backend"):
        create_memory_store(backend="unknown")


@pytest.mark.anyio
async def test_embedding_serialization(store: SQLiteMemoryStore):
    """Test that embeddings are correctly serialized/deserialized."""
    rec = EpisodicMemoryRecord(
        user_id="user1",
        memory_type="key_conversation",
        content="Test with embedding",
        embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
    )
    await store.save_episodic(rec)

    loaded = await store.get_episodic_by_ids([rec.memory_id])
    assert len(loaded) == 1
    assert loaded[0].embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

    # None embedding
    rec2 = EpisodicMemoryRecord(
        user_id="user1", memory_type="key_conversation", content="No embedding",
    )
    await store.save_episodic(rec2)
    loaded2 = await store.get_episodic_by_ids([rec2.memory_id])
    assert loaded2[0].embedding is None
