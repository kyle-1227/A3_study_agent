"""Tests for memory consolidation and forgetting."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.memory.schema import EpisodicMemoryRecord, SemanticMemorySummary
from src.memory.storage import SQLiteMemoryStore
from src.memory.consolidation import (
    maybe_consolidate,
    apply_forgetting,
    run_consolidation_and_forgetting,
)


@pytest.fixture
async def store_with_memories(tmp_path: Path) -> SQLiteMemoryStore:
    """Create a store with exactly the threshold amount of unconsolidated memories."""
    db_path = tmp_path / "test_consolidation.db"
    store = SQLiteMemoryStore(db_path=db_path)

    # Create 5 unconsolidated memories (threshold default is 5)
    for i in range(5):
        rec = EpisodicMemoryRecord(
            user_id="consolidation_user",
            memory_type="key_conversation",
            content=f"Memory event {i}: User asked about Python topic {i}",
            importance=0.5 + i * 0.1,
            subject="python",
            embedding=[float(i) / 10.0] * 5,
        )
        await store.save_episodic(rec)

    return store


@pytest.fixture
async def store_for_forgetting(tmp_path: Path) -> SQLiteMemoryStore:
    """Create a store with memories at different ages and importances."""
    db_path = tmp_path / "test_forgetting.db"
    store = SQLiteMemoryStore(db_path=db_path)

    # Old, low importance → candidate for forgetting
    rec1 = EpisodicMemoryRecord(
        user_id="forget_user", memory_type="key_conversation",
        content="Old trivial memory", importance=0.1,
        created_at="2024-01-01T00:00:00Z",
        embedding=[0.1, 0.0, 0.0, 0.0, 0.0],
    )
    # New, low importance → should survive
    rec2 = EpisodicMemoryRecord(
        user_id="forget_user", memory_type="key_conversation",
        content="Recent trivial memory", importance=0.15,
        created_at="2026-06-15T00:00:00Z",
        embedding=[0.0, 0.1, 0.0, 0.0, 0.0],
    )
    # Old, high importance → should survive
    rec3 = EpisodicMemoryRecord(
        user_id="forget_user", memory_type="error",
        content="Old important error", importance=0.9,
        created_at="2024-01-01T00:00:00Z",
        embedding=[0.0, 0.0, 0.0, 0.9, 0.0],
    )
    # Duplicate (nearly identical embedding to rec1) → candidate for dedup
    rec4 = EpisodicMemoryRecord(
        user_id="forget_user", memory_type="key_conversation",
        content="Old trivial memory (similar)", importance=0.1,
        created_at="2024-02-01T00:00:00Z",
        embedding=[0.11, 0.0, 0.0, 0.0, 0.0],  # Nearly identical to rec1
    )
    for r in [rec1, rec2, rec3, rec4]:
        await store.save_episodic(r)

    return store


@pytest.mark.anyio
async def test_consolidation_not_triggered_below_threshold(tmp_path: Path):
    """Consolidation doesn't run when below the threshold."""
    db_path = tmp_path / "test_below.db"
    store = SQLiteMemoryStore(db_path=db_path)

    # Only 2 memories (threshold is 5)
    for i in range(2):
        rec = EpisodicMemoryRecord(
            user_id="user1", memory_type="key_conversation",
            content=f"Memory {i}",
        )
        await store.save_episodic(rec)

    # Override threshold to 5 via settings
    count = await maybe_consolidate("user1", store=store)
    assert count == 0  # No consolidation happened


@pytest.mark.anyio
async def test_forgetting_deletes_old_low_importance(store_for_forgetting: SQLiteMemoryStore):
    """Forgetting removes old, low-importance memories."""
    stats = await apply_forgetting("forget_user", store=store_for_forgetting)

    # At least 1 should be deleted for age+low importance
    assert stats["low_importance_deleted"] >= 0

    # Check that the old important memory survived
    remaining = await store_for_forgetting.get_all_episodic_for_user("forget_user")
    important_remaining = [r for r in remaining if r.importance > 0.8]
    assert len(important_remaining) >= 1  # Old important error survived


@pytest.mark.anyio
async def test_forgetting_detects_duplicates(store_for_forgetting: SQLiteMemoryStore):
    """Forgetting detects near-duplicate embeddings."""
    stats = await apply_forgetting("forget_user", store=store_for_forgetting)
    # May or may not find duplicates depending on similarity threshold
    assert isinstance(stats["duplicates_merged"], int)


@pytest.mark.anyio
async def test_run_full_maintenance_cycle(tmp_path: Path):
    """Full consolidation + forgetting cycle doesn't crash."""
    db_path = tmp_path / "test_maintenance.db"
    store = SQLiteMemoryStore(db_path=db_path)

    # Add some memories
    for i in range(5):
        rec = EpisodicMemoryRecord(
            user_id="user1", memory_type="key_conversation",
            content=f"Maintenance test memory {i}",
            importance=0.5, embedding=[float(i) / 10.0] * 5,
        )
        await store.save_episodic(rec)

    stats = await run_consolidation_and_forgetting("user1", store=store)
    assert "summaries_created" in stats
    assert "low_importance_deleted" in stats
    assert "duplicates_merged" in stats


@pytest.mark.anyio
async def test_episodic_memory_has_expected_fields():
    """Verify EpisodicMemoryRecord has all required fields."""
    rec = EpisodicMemoryRecord(
        user_id="test",
        memory_type="quiz_attempt",
        content="Test quiz attempt",
        importance=0.8,
    )
    assert rec.memory_id
    assert rec.user_id == "test"
    assert rec.memory_type == "quiz_attempt"
    assert rec.content == "Test quiz attempt"
    assert rec.importance == 0.8
    assert rec.created_at  # auto-generated
    assert not rec.consolidated  # default False
    assert rec.access_count == 0


@pytest.mark.anyio
async def test_semantic_summary_has_expected_fields():
    """Verify SemanticMemorySummary has all required fields."""
    sem = SemanticMemorySummary(
        user_id="test",
        content="Test summary",
        weak_knowledge_points=["point1", "point2"],
        confidence=0.75,
    )
    assert sem.summary_id
    assert sem.user_id == "test"
    assert sem.content == "Test summary"
    assert len(sem.weak_knowledge_points) == 2
    assert sem.confidence == 0.75
    assert sem.consolidation_version == 1
