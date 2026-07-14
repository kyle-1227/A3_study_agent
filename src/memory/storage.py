"""
Memory storage — SQLite-backed persistence with migration-ready interface.

Design:
- Abstract base class (MemoryStore) for future PostgreSQL swap
- SQLite implementation for initial deployment
- All methods are async for future compatibility
- JSON serialization via Pydantic's model_dump/model_validate
- Follows the same pattern as src/profile/storage.py
"""

from __future__ import annotations

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.memory.schema import EpisodicMemoryRecord, SemanticMemorySummary

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/memory.db")

MemoryStorageReadErrorCode: TypeAlias = Literal[
    "memory_database_missing",
    "episodic_table_missing",
    "episodic_table_schema_invalid",
    "episodic_metadata_json_invalid",
    "episodic_metadata_schema_invalid",
    "episodic_record_identity_mismatch",
    "memory_database_read_failed",
]


class MemoryStorageReadError(RuntimeError):
    """Content-safe failure from the strict read-only episodic path."""

    def __init__(self, *, code: MemoryStorageReadErrorCode) -> None:
        self.code = code
        super().__init__(f"{code}: strict memory storage read failed")


class StrictEpisodicMetadataRecord(BaseModel):
    """Content-free projection required by learning-guidance history."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    memory_id: str = Field(min_length=1, max_length=200)
    user_id: str = Field(min_length=1, max_length=200)
    subject: str = Field(min_length=1, max_length=200)
    metadata: dict[str, object]

    @field_validator("memory_id", "user_id", "subject")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("identity fields must be normalized and non-blank")
        return value


def _reject_non_finite_metadata_json(value: str) -> object:
    raise ValueError("non-finite JSON number")


def _strict_metadata_json_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _strict_metadata_json_value(raw: object) -> object:
    if not isinstance(raw, str):
        raise TypeError("stored metadata JSON must be text")
    return json.loads(
        raw,
        object_pairs_hook=_strict_metadata_json_pairs,
        parse_constant=_reject_non_finite_metadata_json,
    )


def _validate_strict_query_text(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be normalized and non-blank")


# ── Abstract interface ─────────────────────────────────────────────────────


class MemoryStore(ABC):
    """Abstract interface for memory persistence.

    Implementations: SQLiteMemoryStore, PostgresMemoryStore (future).
    """

    @abstractmethod
    async def save_episodic(self, record: EpisodicMemoryRecord) -> None:
        """Persist an episodic memory record (insert or update)."""
        ...

    @abstractmethod
    async def save_semantic(self, summary: SemanticMemorySummary) -> None:
        """Persist a semantic memory summary (insert or update)."""
        ...

    @abstractmethod
    async def query_episodic(
        self,
        user_id: str,
        *,
        memory_type: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        importance_min: float = 0.0,
        limit: int = 50,
        offset: int = 0,
    ) -> list[EpisodicMemoryRecord]:
        """Query episodic memories with optional filters."""
        ...

    @abstractmethod
    async def get_episodic_by_ids(
        self,
        memory_ids: list[str],
    ) -> list[EpisodicMemoryRecord]:
        """Fetch episodic memories by their IDs."""
        ...

    @abstractmethod
    async def get_all_episodic_for_user(
        self,
        user_id: str,
        limit: int = 200,
    ) -> list[EpisodicMemoryRecord]:
        """Get all episodic memories for a user (for BM25 corpus building)."""
        ...

    @abstractmethod
    async def get_unconsolidated(
        self,
        user_id: str,
        limit: int = 10,
    ) -> list[EpisodicMemoryRecord]:
        """Get oldest unconsolidated episodic memories for a user."""
        ...

    @abstractmethod
    async def get_semantic_for_user(
        self,
        user_id: str,
        limit: int = 20,
    ) -> list[SemanticMemorySummary]:
        """Get semantic memory summaries for a user, newest first."""
        ...

    @abstractmethod
    async def mark_consolidated(
        self,
        memory_ids: list[str],
        group_id: str,
    ) -> None:
        """Mark a batch of episodic memories as consolidated."""
        ...

    @abstractmethod
    async def delete_episodic(self, memory_id: str) -> bool:
        """Delete a single episodic memory. Returns True if it existed."""
        ...

    @abstractmethod
    async def delete_low_importance_old(
        self,
        user_id: str,
        before_ts: str,
        importance_max: float,
    ) -> int:
        """Delete episodic memories older than before_ts with importance <= importance_max.
        Returns count deleted."""
        ...

    @abstractmethod
    async def delete_semantic(self, summary_id: str) -> bool:
        """Delete a semantic memory summary. Returns True if it existed."""
        ...

    @abstractmethod
    async def get_episodic_count(self, user_id: str) -> int:
        """Total episodic memories for a user."""
        ...

    @abstractmethod
    async def get_semantic_count(self, user_id: str) -> int:
        """Total semantic summaries for a user."""
        ...


# ── SQLite implementation ──────────────────────────────────────────────────


class SQLiteMemoryStore(MemoryStore):
    """SQLite-backed memory storage.

    Tables::

        episodic_memories (
            memory_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            content TEXT NOT NULL,
            importance REAL NOT NULL DEFAULT 0.5,
            subject TEXT DEFAULT '',
            metadata_json TEXT DEFAULT '{}',
            embedding_json TEXT,
            created_at TEXT NOT NULL,
            last_accessed_at TEXT DEFAULT '',
            access_count INTEGER DEFAULT 0,
            consolidated INTEGER DEFAULT 0,
            consolidation_group TEXT DEFAULT ''
        )

        semantic_memories (
            summary_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            source_episodic_ids_json TEXT DEFAULT '[]',
            content TEXT NOT NULL,
            weak_knowledge_points_json TEXT DEFAULT '[]',
            learning_style_changes TEXT DEFAULT '',
            skill_growth_trajectory TEXT DEFAULT '',
            embedding_json TEXT,
            created_at TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            consolidation_version INTEGER DEFAULT 1
        )
    """

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = Path(db_path or DEFAULT_DB_PATH)
        self._initialized = False

    @property
    def db_path(self) -> Path:
        return self._db_path

    async def _ensure_init(self) -> None:
        """Lazy initialization — create tables on first access."""
        if self._initialized:
            return
        import aiosqlite

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    subject TEXT DEFAULT '',
                    metadata_json TEXT DEFAULT '{}',
                    embedding_json TEXT,
                    created_at TEXT NOT NULL,
                    last_accessed_at TEXT DEFAULT '',
                    access_count INTEGER DEFAULT 0,
                    consolidated INTEGER DEFAULT 0,
                    consolidation_group TEXT DEFAULT ''
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS semantic_memories (
                    summary_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    source_episodic_ids_json TEXT DEFAULT '[]',
                    content TEXT NOT NULL,
                    weak_knowledge_points_json TEXT DEFAULT '[]',
                    learning_style_changes TEXT DEFAULT '',
                    skill_growth_trajectory TEXT DEFAULT '',
                    embedding_json TEXT,
                    created_at TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    consolidation_version INTEGER DEFAULT 1
                )
            """)
            # Indexes for common access patterns
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodic_user_time "
                "ON episodic_memories(user_id, created_at)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodic_user_type "
                "ON episodic_memories(user_id, memory_type)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodic_user_consolidated "
                "ON episodic_memories(user_id, consolidated)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_semantic_user_time "
                "ON semantic_memories(user_id, created_at)"
            )
            await db.commit()
        self._initialized = True

    # ── Episodic CRUD ──────────────────────────────────────────────────

    async def save_episodic(self, record: EpisodicMemoryRecord) -> None:
        """Insert or update an episodic memory record."""
        import aiosqlite

        await self._ensure_init()
        embedding_json = json.dumps(record.embedding) if record.embedding else None
        metadata_json = json.dumps(record.metadata, ensure_ascii=False)
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(
                """
                INSERT INTO episodic_memories (
                    memory_id, user_id, memory_type, content, importance,
                    subject, metadata_json, embedding_json, created_at,
                    last_accessed_at, access_count, consolidated, consolidation_group
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    content = excluded.content,
                    importance = excluded.importance,
                    subject = excluded.subject,
                    metadata_json = excluded.metadata_json,
                    embedding_json = excluded.embedding_json,
                    last_accessed_at = excluded.last_accessed_at,
                    access_count = excluded.access_count,
                    consolidated = excluded.consolidated,
                    consolidation_group = excluded.consolidation_group
            """,
                (
                    record.memory_id,
                    record.user_id,
                    record.memory_type,
                    record.content,
                    record.importance,
                    record.subject,
                    metadata_json,
                    embedding_json,
                    record.created_at,
                    record.last_accessed_at,
                    record.access_count,
                    int(record.consolidated),
                    record.consolidation_group,
                ),
            )
            await db.commit()
        logger.debug(
            "Saved episodic memory id=%s type=%s", record.memory_id, record.memory_type
        )

    async def query_episodic(
        self,
        user_id: str,
        *,
        memory_type: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        importance_min: float = 0.0,
        limit: int = 50,
        offset: int = 0,
    ) -> list[EpisodicMemoryRecord]:
        """Query episodic memories with optional filters."""
        import aiosqlite

        await self._ensure_init()
        where_clauses = ["user_id = ?"]
        params: list = [user_id]

        if memory_type:
            where_clauses.append("memory_type = ?")
            params.append(memory_type)
        if start_time:
            where_clauses.append("created_at >= ?")
            params.append(start_time)
        if end_time:
            where_clauses.append("created_at <= ?")
            params.append(end_time)
        if importance_min > 0.0:
            where_clauses.append("importance >= ?")
            params.append(importance_min)

        where_sql = " AND ".join(where_clauses)
        query_sql = (
            f"SELECT * FROM episodic_memories "
            f"WHERE {where_sql} "
            f"ORDER BY created_at DESC "
            f"LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query_sql, params)
            rows = await cursor.fetchall()

        return [_row_to_episodic(row) for row in rows]

    async def query_episodic_metadata_strict(
        self,
        *,
        user_id: str,
        subject: str,
        limit: int,
    ) -> tuple[StrictEpisodicMetadataRecord, ...]:
        """Read a bounded metadata projection without repair or DB creation."""

        _validate_strict_query_text(user_id, field_name="user_id")
        _validate_strict_query_text(subject, field_name="subject")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if not self._db_path.is_file():
            raise MemoryStorageReadError(code="memory_database_missing")

        import aiosqlite

        database_uri = f"{self._db_path.resolve().as_uri()}?mode=ro"
        try:
            async with aiosqlite.connect(database_uri, uri=True) as db:
                db.row_factory = aiosqlite.Row
                table_cursor = await db.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'episodic_memories'"
                )
                if await table_cursor.fetchone() is None:
                    raise MemoryStorageReadError(code="episodic_table_missing")

                schema_cursor = await db.execute("PRAGMA table_info(episodic_memories)")
                schema_rows = await schema_cursor.fetchall()
                columns = frozenset(str(row[1]) for row in schema_rows)
                required_columns = {
                    "memory_id",
                    "user_id",
                    "subject",
                    "metadata_json",
                    "created_at",
                }
                if not required_columns.issubset(columns):
                    raise MemoryStorageReadError(code="episodic_table_schema_invalid")

                cursor = await db.execute(
                    "SELECT memory_id, user_id, subject, metadata_json "
                    "FROM episodic_memories "
                    "WHERE user_id = ? AND subject = ? "
                    "ORDER BY created_at DESC, memory_id ASC LIMIT ?",
                    (user_id, subject, limit),
                )
                rows = await cursor.fetchall()
        except MemoryStorageReadError:
            raise
        except sqlite3.Error as exc:
            raise MemoryStorageReadError(code="memory_database_read_failed") from exc

        records: list[StrictEpisodicMetadataRecord] = []
        for row in rows:
            if row["user_id"] != user_id or row["subject"] != subject:
                raise MemoryStorageReadError(code="episodic_record_identity_mismatch")
            try:
                decoded = _strict_metadata_json_value(row["metadata_json"])
            except (json.JSONDecodeError, TypeError, ValueError):
                raise MemoryStorageReadError(
                    code="episodic_metadata_json_invalid"
                ) from None
            if not isinstance(decoded, dict):
                raise MemoryStorageReadError(code="episodic_metadata_schema_invalid")
            metadata = decoded
            try:
                record = StrictEpisodicMetadataRecord.model_validate(
                    {
                        "memory_id": row["memory_id"],
                        "user_id": row["user_id"],
                        "subject": row["subject"],
                        "metadata": metadata,
                    }
                )
            except ValidationError:
                raise MemoryStorageReadError(
                    code="episodic_metadata_schema_invalid"
                ) from None
            records.append(record)
        return tuple(records)

    async def get_episodic_by_ids(
        self,
        memory_ids: list[str],
    ) -> list[EpisodicMemoryRecord]:
        """Fetch episodic memories by their IDs."""
        if not memory_ids:
            return []
        import aiosqlite

        await self._ensure_init()
        placeholders = ",".join("?" for _ in memory_ids)
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                f"SELECT * FROM episodic_memories WHERE memory_id IN ({placeholders})",
                memory_ids,
            )
            rows = await cursor.fetchall()
        return [_row_to_episodic(row) for row in rows]

    async def get_all_episodic_for_user(
        self,
        user_id: str,
        limit: int = 200,
    ) -> list[EpisodicMemoryRecord]:
        """Get all episodic memories for a user, newest first."""
        import aiosqlite

        await self._ensure_init()
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM episodic_memories WHERE user_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            )
            rows = await cursor.fetchall()
        return [_row_to_episodic(row) for row in rows]

    async def get_unconsolidated(
        self,
        user_id: str,
        limit: int = 10,
    ) -> list[EpisodicMemoryRecord]:
        """Get oldest unconsolidated episodic memories."""
        import aiosqlite

        await self._ensure_init()
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM episodic_memories "
                "WHERE user_id = ? AND consolidated = 0 "
                "ORDER BY created_at ASC LIMIT ?",
                (user_id, limit),
            )
            rows = await cursor.fetchall()
        return [_row_to_episodic(row) for row in rows]

    async def mark_consolidated(
        self,
        memory_ids: list[str],
        group_id: str,
    ) -> None:
        """Mark a batch of episodic memories as consolidated."""
        if not memory_ids:
            return
        import aiosqlite

        await self._ensure_init()
        placeholders = ",".join("?" for _ in memory_ids)
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(
                f"UPDATE episodic_memories SET consolidated = 1, "
                f"consolidation_group = ? WHERE memory_id IN ({placeholders})",
                [group_id] + memory_ids,
            )
            await db.commit()
        logger.debug(
            "Marked %d episodic memories as consolidated (group=%s)",
            len(memory_ids),
            group_id,
        )

    async def delete_episodic(self, memory_id: str) -> bool:
        """Delete a single episodic memory."""
        import aiosqlite

        await self._ensure_init()
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(
                "DELETE FROM episodic_memories WHERE memory_id = ?",
                (memory_id,),
            )
            await db.commit()
            deleted = cursor.rowcount > 0
        if deleted:
            logger.debug("Deleted episodic memory id=%s", memory_id)
        return deleted

    async def delete_low_importance_old(
        self,
        user_id: str,
        before_ts: str,
        importance_max: float,
    ) -> int:
        """Delete old, low-importance episodic memories. Returns count deleted."""
        import aiosqlite

        await self._ensure_init()
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(
                "DELETE FROM episodic_memories "
                "WHERE user_id = ? AND created_at < ? AND importance < ?",
                (user_id, before_ts, importance_max),
            )
            await db.commit()
            deleted = cursor.rowcount
        if deleted:
            logger.debug(
                "Forgetting: deleted %d low-importance episodic memories for user=%s",
                deleted,
                user_id,
            )
        return deleted

    async def get_episodic_count(self, user_id: str) -> int:
        """Total episodic memories for a user."""
        import aiosqlite

        await self._ensure_init()
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM episodic_memories WHERE user_id = ?",
                (user_id,),
            )
            row = await cursor.fetchone()
        return row[0] if row else 0

    # ── Semantic CRUD ──────────────────────────────────────────────────

    async def save_semantic(self, summary: SemanticMemorySummary) -> None:
        """Insert or update a semantic memory summary."""
        import aiosqlite

        await self._ensure_init()
        embedding_json = json.dumps(summary.embedding) if summary.embedding else None
        source_ids_json = json.dumps(summary.source_episodic_ids, ensure_ascii=False)
        weak_points_json = json.dumps(summary.weak_knowledge_points, ensure_ascii=False)
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(
                """
                INSERT INTO semantic_memories (
                    summary_id, user_id, source_episodic_ids_json, content,
                    weak_knowledge_points_json, learning_style_changes,
                    skill_growth_trajectory, embedding_json, created_at,
                    confidence, consolidation_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(summary_id) DO UPDATE SET
                    content = excluded.content,
                    weak_knowledge_points_json = excluded.weak_knowledge_points_json,
                    learning_style_changes = excluded.learning_style_changes,
                    skill_growth_trajectory = excluded.skill_growth_trajectory,
                    embedding_json = excluded.embedding_json,
                    confidence = excluded.confidence,
                    consolidation_version = excluded.consolidation_version
            """,
                (
                    summary.summary_id,
                    summary.user_id,
                    source_ids_json,
                    summary.content,
                    weak_points_json,
                    summary.learning_style_changes,
                    summary.skill_growth_trajectory,
                    embedding_json,
                    summary.created_at,
                    summary.confidence,
                    summary.consolidation_version,
                ),
            )
            await db.commit()
        logger.debug(
            "Saved semantic summary id=%s (v%d)",
            summary.summary_id,
            summary.consolidation_version,
        )

    async def get_semantic_for_user(
        self,
        user_id: str,
        limit: int = 20,
    ) -> list[SemanticMemorySummary]:
        """Get semantic memory summaries for a user, newest first."""
        import aiosqlite

        await self._ensure_init()
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM semantic_memories WHERE user_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            )
            rows = await cursor.fetchall()
        return [_row_to_semantic(row) for row in rows]

    async def delete_semantic(self, summary_id: str) -> bool:
        """Delete a semantic memory summary."""
        import aiosqlite

        await self._ensure_init()
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(
                "DELETE FROM semantic_memories WHERE summary_id = ?",
                (summary_id,),
            )
            await db.commit()
            deleted = cursor.rowcount > 0
        if deleted:
            logger.debug("Deleted semantic summary id=%s", summary_id)
        return deleted

    async def get_semantic_count(self, user_id: str) -> int:
        """Total semantic summaries for a user."""
        import aiosqlite

        await self._ensure_init()
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM semantic_memories WHERE user_id = ?",
                (user_id,),
            )
            row = await cursor.fetchone()
        return row[0] if row else 0


# ── Row deserialization helpers ────────────────────────────────────────────


def _row_to_episodic(row) -> EpisodicMemoryRecord:
    """Deserialize an aiosqlite.Row to an EpisodicMemoryRecord."""
    data = dict(row)
    # Parse JSON fields
    try:
        data["metadata"] = json.loads(data.get("metadata_json", "{}") or "{}")
    except json.JSONDecodeError:
        data["metadata"] = {}
    try:
        data["embedding"] = json.loads(data.get("embedding_json") or "null")
    except json.JSONDecodeError:
        data["embedding"] = None
    data["consolidated"] = bool(data.get("consolidated", 0))
    # Remove JSON raw columns
    data.pop("metadata_json", None)
    data.pop("embedding_json", None)
    return EpisodicMemoryRecord(**data)


def _row_to_semantic(row) -> SemanticMemorySummary:
    """Deserialize an aiosqlite.Row to a SemanticMemorySummary."""
    data = dict(row)
    # Parse JSON fields
    try:
        data["source_episodic_ids"] = json.loads(
            data.get("source_episodic_ids_json", "[]") or "[]"
        )
    except json.JSONDecodeError:
        data["source_episodic_ids"] = []
    try:
        data["weak_knowledge_points"] = json.loads(
            data.get("weak_knowledge_points_json", "[]") or "[]"
        )
    except json.JSONDecodeError:
        data["weak_knowledge_points"] = []
    try:
        data["embedding"] = json.loads(data.get("embedding_json") or "null")
    except json.JSONDecodeError:
        data["embedding"] = None
    # Remove JSON raw columns
    data.pop("source_episodic_ids_json", None)
    data.pop("weak_knowledge_points_json", None)
    data.pop("embedding_json", None)
    return SemanticMemorySummary(**data)


# ── Factory ────────────────────────────────────────────────────────────────


def create_memory_store(backend: str = "sqlite", **kwargs) -> MemoryStore:
    """Factory: create a MemoryStore instance.

    Args:
        backend: "sqlite" (default) or "postgres" (future).
        **kwargs: Passed to the store constructor.

    Returns:
        A MemoryStore instance.
    """
    if backend == "sqlite":
        return SQLiteMemoryStore(**kwargs)
    if backend == "postgres":
        raise NotImplementedError("PostgreSQL memory store not yet implemented")
    raise ValueError(f"Unknown memory store backend: {backend}")
