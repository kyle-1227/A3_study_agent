"""Fail-closed tests for bounded read-only episodic metadata queries."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

from src.memory.schema import EpisodicMemoryRecord
from src.memory.storage import MemoryStorageReadError, SQLiteMemoryStore


def _create_episodic_table(path: Path, *, metadata_column: bool = True) -> None:
    statement = (
        "CREATE TABLE episodic_memories ("
        "memory_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, "
        "subject TEXT NOT NULL, metadata_json TEXT NOT NULL, "
        "created_at TEXT NOT NULL)"
        if metadata_column
        else "CREATE TABLE episodic_memories ("
        "memory_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, "
        "subject TEXT NOT NULL, payload TEXT, created_at TEXT NOT NULL)"
    )
    with sqlite3.connect(path) as db:
        db.execute(statement)


def _insert_raw_memory(
    path: Path,
    *,
    memory_id: str,
    user_id: str,
    subject: str,
    metadata_json: str,
    created_at: str,
) -> None:
    with sqlite3.connect(path) as db:
        db.execute(
            "INSERT INTO episodic_memories "
            "(memory_id, user_id, subject, metadata_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (memory_id, user_id, subject, metadata_json, created_at),
        )


@pytest.mark.asyncio
async def test_memory_query_strict_round_trip_uses_only_metadata_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    record = EpisodicMemoryRecord(
        memory_id="memory-1",
        user_id="user-1",
        memory_type="quiz_attempt",
        content="private content must not be projected",
        subject="math",
        metadata={"learning_guidance_v1": {"schema_version": "event_v1"}},
    )
    await store.save_episodic(record)

    async def forbidden_init() -> None:
        raise AssertionError("strict query called _ensure_init")

    monkeypatch.setattr(store, "_ensure_init", forbidden_init)
    result = await store.query_episodic_metadata_strict(
        user_id="user-1",
        subject="math",
        limit=10,
    )

    assert len(result) == 1
    assert result[0].memory_id == "memory-1"
    assert result[0].metadata == record.metadata
    assert "content" not in type(result[0]).model_fields
    assert "embedding" not in type(result[0]).model_fields


@pytest.mark.asyncio
async def test_memory_query_strict_missing_database_is_not_created(
    tmp_path: Path,
) -> None:
    database = tmp_path / "missing.sqlite"

    with pytest.raises(MemoryStorageReadError) as error:
        await SQLiteMemoryStore(database).query_episodic_metadata_strict(
            user_id="user-1",
            subject="math",
            limit=10,
        )

    assert error.value.code == "memory_database_missing"
    assert not database.exists()


@pytest.mark.asyncio
async def test_memory_query_strict_distinguishes_missing_table_and_schema(
    tmp_path: Path,
) -> None:
    empty_database = tmp_path / "empty.sqlite"
    sqlite3.connect(empty_database).close()
    with pytest.raises(MemoryStorageReadError) as missing:
        await SQLiteMemoryStore(empty_database).query_episodic_metadata_strict(
            user_id="user-1",
            subject="math",
            limit=10,
        )
    assert missing.value.code == "episodic_table_missing"

    invalid_database = tmp_path / "invalid.sqlite"
    _create_episodic_table(invalid_database, metadata_column=False)
    with pytest.raises(MemoryStorageReadError) as invalid:
        await SQLiteMemoryStore(invalid_database).query_episodic_metadata_strict(
            user_id="user-1",
            subject="math",
            limit=10,
        )
    assert invalid.value.code == "episodic_table_schema_invalid"


@pytest.mark.asyncio
async def test_memory_query_strict_isolates_identity_and_orders_deterministically(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.sqlite"
    _create_episodic_table(database)
    rows = (
        ("memory-b", "user-1", "math", "2026-07-15T10:00:00+00:00"),
        ("memory-a", "user-1", "math", "2026-07-15T10:00:00+00:00"),
        ("memory-old", "user-1", "math", "2026-07-14T10:00:00+00:00"),
        ("memory-python", "user-1", "python", "2026-07-16T10:00:00+00:00"),
        ("memory-other", "user-2", "math", "2026-07-16T10:00:00+00:00"),
    )
    for memory_id, user_id, subject, created_at in rows:
        _insert_raw_memory(
            database,
            memory_id=memory_id,
            user_id=user_id,
            subject=subject,
            metadata_json=json.dumps({"id": memory_id}),
            created_at=created_at,
        )

    result = await SQLiteMemoryStore(database).query_episodic_metadata_strict(
        user_id="user-1",
        subject="math",
        limit=2,
    )

    assert tuple(item.memory_id for item in result) == ("memory-a", "memory-b")
    assert (
        await SQLiteMemoryStore(database).query_episodic_metadata_strict(
            user_id="absent-user",
            subject="math",
            limit=2,
        )
        == ()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    (
        "{malformed-sensitive-marker",
        '{"id":"first","id":"duplicate-sensitive-marker"}',
        '{"value":Infinity,"marker":"non-finite-sensitive-marker"}',
    ),
)
async def test_memory_query_strict_rejects_invalid_json_without_leaking_content(
    tmp_path: Path,
    raw: str,
) -> None:
    database = tmp_path / "memory.sqlite"
    _create_episodic_table(database)
    _insert_raw_memory(
        database,
        memory_id="memory-1",
        user_id="user-1",
        subject="math",
        metadata_json=raw,
        created_at="2026-07-15T10:00:00+00:00",
    )

    with pytest.raises(MemoryStorageReadError) as error:
        await SQLiteMemoryStore(database).query_episodic_metadata_strict(
            user_id="user-1",
            subject="math",
            limit=10,
        )

    assert error.value.code == "episodic_metadata_json_invalid"
    assert "sensitive-marker" not in str(error.value)
    assert error.value.__cause__ is None


@pytest.mark.asyncio
async def test_memory_query_strict_rejects_non_object_metadata(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.sqlite"
    _create_episodic_table(database)
    _insert_raw_memory(
        database,
        memory_id="memory-1",
        user_id="user-1",
        subject="math",
        metadata_json="[]",
        created_at="2026-07-15T10:00:00+00:00",
    )

    with pytest.raises(MemoryStorageReadError) as error:
        await SQLiteMemoryStore(database).query_episodic_metadata_strict(
            user_id="user-1",
            subject="math",
            limit=10,
        )

    assert error.value.code == "episodic_metadata_schema_invalid"


@pytest.mark.asyncio
async def test_memory_query_strict_does_not_return_partial_results(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.sqlite"
    _create_episodic_table(database)
    _insert_raw_memory(
        database,
        memory_id="memory-valid",
        user_id="user-1",
        subject="math",
        metadata_json="{}",
        created_at="2026-07-15T11:00:00+00:00",
    )
    _insert_raw_memory(
        database,
        memory_id="memory-invalid",
        user_id="user-1",
        subject="math",
        metadata_json="{invalid",
        created_at="2026-07-15T10:00:00+00:00",
    )

    with pytest.raises(MemoryStorageReadError) as error:
        await SQLiteMemoryStore(database).query_episodic_metadata_strict(
            user_id="user-1",
            subject="math",
            limit=10,
        )

    assert error.value.code == "episodic_metadata_json_invalid"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_id", "subject", "limit"),
    (
        ("", "math", 1),
        (" user-1", "math", 1),
        ("user-1", "math ", 1),
        ("user-1", "math", 0),
        ("user-1", "math", True),
    ),
)
async def test_memory_query_strict_validates_inputs_before_file_access(
    tmp_path: Path,
    user_id: str,
    subject: str,
    limit: int,
) -> None:
    database = tmp_path / "missing.sqlite"

    with pytest.raises(ValueError):
        await SQLiteMemoryStore(database).query_episodic_metadata_strict(
            user_id=user_id,
            subject=subject,
            limit=limit,
        )

    assert not database.exists()
