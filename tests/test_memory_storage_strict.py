"""Fail-closed tests for bounded read-only episodic metadata queries."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import traceback

import pytest

import src.memory.storage as memory_storage_module
from src.memory.retention import LEARNING_GUIDANCE_HISTORY_ID_PREFIX
from src.memory.schema import EpisodicMemoryRecord
from src.memory.storage import (
    MemoryStorageReadError,
    MemoryStorageWriteError,
    SQLiteMemoryStore,
)


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


def _strict_record(
    *,
    memory_id: str = "strict-memory-1",
    metadata: dict[str, object] | None = None,
) -> EpisodicMemoryRecord:
    return EpisodicMemoryRecord(
        memory_id=memory_id,
        user_id="user-1",
        memory_type="quiz_attempt",
        content="bounded fact",
        importance=0.1,
        subject="math",
        metadata=(
            {"learning_guidance_v1": {"status": "verified"}}
            if metadata is None
            else metadata
        ),
        embedding=[0.1, 0.2],
        created_at="2026-07-15T10:00:00+00:00",
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
        memory_id_prefix="memory-",
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
            memory_id_prefix="memory-",
            limit=10,
        )

    assert error.value.code == "memory_database_missing"
    assert not database.exists()


@pytest.mark.asyncio
async def test_explicit_initialization_creates_an_empty_strict_store(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.sqlite"
    store = SQLiteMemoryStore(database)

    await store.initialize()

    assert database.is_file()
    assert (
        await SQLiteMemoryStore(database).query_episodic_metadata_strict(
            user_id="user-1",
            subject="math",
            memory_id_prefix="memory-",
            limit=10,
        )
        == ()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("column", "corrupted_json"),
    [
        ("metadata_json", "{malformed"),
        ("metadata_json", '{"value":1,"value":2}'),
        ("metadata_json", '{"value":Infinity}'),
        ("metadata_json", '{"value":1e400}'),
        ("embedding_json", "{malformed"),
        ("embedding_json", '{"not":"an-array"}'),
        ("embedding_json", "[NaN]"),
        ("embedding_json", "[1e400]"),
    ],
)
async def test_insert_once_replay_rejects_corrupted_json_without_repair(
    tmp_path: Path,
    column: str,
    corrupted_json: str,
) -> None:
    database = tmp_path / "memory.sqlite"
    store = SQLiteMemoryStore(database)
    record = EpisodicMemoryRecord(
        memory_id="strict-memory-1",
        user_id="user-1",
        memory_type="quiz_attempt",
        content="bounded fact",
        subject="math",
        metadata={"learning_guidance_v1": {"status": "verified"}},
        embedding=[0.1, 0.2],
    )
    assert await store.insert_episodic_once_strict(record) is True
    if column == "metadata_json":
        statement = "UPDATE episodic_memories SET metadata_json = ? WHERE memory_id = ?"
    elif column == "embedding_json":
        statement = (
            "UPDATE episodic_memories SET embedding_json = ? WHERE memory_id = ?"
        )
    else:
        raise AssertionError(f"unsupported corruption column: {column}")
    with sqlite3.connect(database) as db:
        db.execute(statement, (corrupted_json, record.memory_id))

    with pytest.raises(MemoryStorageWriteError) as error:
        await SQLiteMemoryStore(database).insert_episodic_once_strict(record)

    assert error.value.code == "episodic_insert_conflict"
    assert "malformed" not in str(error.value)
    assert error.value.__cause__ is None


@pytest.mark.asyncio
async def test_insert_once_rolls_back_when_stored_row_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "memory.sqlite"
    store = SQLiteMemoryStore(database)
    await store.initialize()

    def reject_stored_row(_row: object) -> EpisodicMemoryRecord:
        raise ValueError("sensitive-row-marker")

    monkeypatch.setattr(
        memory_storage_module,
        "_row_to_episodic_strict",
        reject_stored_row,
    )

    with pytest.raises(MemoryStorageWriteError) as error:
        await store.insert_episodic_once_strict(_strict_record())

    assert error.value.code == "episodic_insert_conflict"
    assert error.value.__cause__ is None
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT COUNT(*) FROM episodic_memories").fetchone() == (0,)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_value", "drifted_value"),
    ((True, 1), (1, 1.0), (0.0, -0.0)),
)
async def test_insert_once_rejects_json_type_or_sign_drift(
    tmp_path: Path,
    first_value: object,
    drifted_value: object,
) -> None:
    database = tmp_path / "memory.sqlite"
    store = SQLiteMemoryStore(database)
    first = _strict_record(metadata={"value": first_value})
    drifted = _strict_record(metadata={"value": drifted_value})

    assert await store.insert_episodic_once_strict(first) is True
    with pytest.raises(MemoryStorageWriteError) as error:
        await store.insert_episodic_once_strict(drifted)

    assert error.value.code == "episodic_insert_conflict"
    with sqlite3.connect(database) as db:
        stored_json = db.execute(
            "SELECT metadata_json FROM episodic_memories WHERE memory_id = ?",
            (first.memory_id,),
        ).fetchone()[0]
    assert stored_json == json.dumps(
        first.metadata,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@pytest.mark.asyncio
async def test_insert_once_rejects_non_json_native_metadata_before_write(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.sqlite"
    store = SQLiteMemoryStore(database)
    await store.initialize()
    record = _strict_record(metadata={"items": ("one", "two")})

    with pytest.raises(TypeError, match="JSON-native"):
        await store.insert_episodic_once_strict(record)

    with sqlite3.connect(database) as db:
        assert db.execute("SELECT COUNT(*) FROM episodic_memories").fetchone() == (0,)


@pytest.mark.asyncio
async def test_insert_once_maps_initialization_io_failure_to_typed_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")

    async def fail_initialization() -> None:
        raise OSError("storage unavailable")

    monkeypatch.setattr(store, "_ensure_init", fail_initialization)

    with pytest.raises(MemoryStorageWriteError) as error:
        await store.insert_episodic_once_strict(_strict_record())

    assert error.value.code == "episodic_insert_failed"


@pytest.mark.asyncio
async def test_protected_history_namespace_is_insert_once_and_not_mutable(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.sqlite"
    store = SQLiteMemoryStore(database)
    memory_id = f"{LEARNING_GUIDANCE_HISTORY_ID_PREFIX}{'a' * 64}"
    record = _strict_record(memory_id=memory_id, metadata={"value": "original"})
    drifted = _strict_record(memory_id=memory_id, metadata={"value": "drifted"})

    await store.save_episodic(record)
    with pytest.raises(MemoryStorageWriteError) as conflict:
        await store.save_episodic(drifted)
    assert conflict.value.code == "episodic_insert_conflict"
    with pytest.raises(ValueError, match="protected episodic facts"):
        await store.mark_consolidated([memory_id], "summary-1")
    with pytest.raises(ValueError, match="protected episodic facts"):
        await store.delete_episodic(memory_id)
    assert await store.get_unconsolidated("user-1", limit=10) == []
    assert (
        await store.delete_low_importance_old(
            "user-1",
            before_ts="9999-12-31T23:59:59+00:00",
            importance_max=1.0,
        )
        == 0
    )


@pytest.mark.asyncio
async def test_protected_history_concurrent_strict_and_ordinary_writers_do_not_overwrite(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    await store.initialize()
    memory_id = f"{LEARNING_GUIDANCE_HISTORY_ID_PREFIX}{'b' * 64}"
    strict_record = _strict_record(memory_id=memory_id, metadata={"writer": "strict"})
    ordinary_record = _strict_record(
        memory_id=memory_id,
        metadata={"writer": "ordinary"},
    )

    outcomes = await asyncio.gather(
        store.insert_episodic_once_strict(strict_record),
        store.save_episodic(ordinary_record),
        return_exceptions=True,
    )

    assert sum(isinstance(item, MemoryStorageWriteError) for item in outcomes) == 1
    records = await store.query_episodic_metadata_strict(
        user_id="user-1",
        subject="math",
        memory_id_prefix=LEARNING_GUIDANCE_HISTORY_ID_PREFIX,
        limit=10,
    )
    assert len(records) == 1
    assert records[0].metadata in (strict_record.metadata, ordinary_record.metadata)


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
            memory_id_prefix="memory-",
            limit=10,
        )
    assert missing.value.code == "episodic_table_missing"

    invalid_database = tmp_path / "invalid.sqlite"
    _create_episodic_table(invalid_database, metadata_column=False)
    with pytest.raises(MemoryStorageReadError) as invalid:
        await SQLiteMemoryStore(invalid_database).query_episodic_metadata_strict(
            user_id="user-1",
            subject="math",
            memory_id_prefix="memory-",
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
        memory_id_prefix="memory-",
        limit=2,
    )

    assert tuple(item.memory_id for item in result) == ("memory-a", "memory-b")
    assert (
        await SQLiteMemoryStore(database).query_episodic_metadata_strict(
            user_id="absent-user",
            subject="math",
            memory_id_prefix="memory-",
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
        '{"value":1e400,"marker":"overflow-sensitive-marker"}',
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
            memory_id_prefix="memory-",
            limit=10,
        )

    assert error.value.code == "episodic_metadata_json_invalid"
    assert "sensitive-marker" not in str(error.value)
    assert error.value.__cause__ is None
    assert "sensitive-marker" not in "".join(traceback.format_exception(error.value))


@pytest.mark.asyncio
async def test_memory_query_strict_uses_case_sensitive_prefix_matching(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.sqlite"
    _create_episodic_table(database)
    prefix = LEARNING_GUIDANCE_HISTORY_ID_PREFIX
    _insert_raw_memory(
        database,
        memory_id=f"{prefix}{'c' * 64}",
        user_id="user-1",
        subject="math",
        metadata_json='{"status":"authoritative"}',
        created_at="2026-07-15T10:00:00+00:00",
    )
    _insert_raw_memory(
        database,
        memory_id=f"{prefix.upper()}{'d' * 64}",
        user_id="user-1",
        subject="math",
        metadata_json='{"status":"near-match"}',
        created_at="2026-07-15T11:00:00+00:00",
    )

    records = await SQLiteMemoryStore(database).query_episodic_metadata_strict(
        user_id="user-1",
        subject="math",
        memory_id_prefix=prefix,
        limit=10,
    )

    assert len(records) == 1
    assert records[0].metadata == {"status": "authoritative"}


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
            memory_id_prefix="memory-",
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
            memory_id_prefix="memory-",
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
            memory_id_prefix="memory-",
            limit=limit,
        )

    assert not database.exists()
