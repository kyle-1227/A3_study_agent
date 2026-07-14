"""
Profile storage — SQLite-backed persistence with migration-ready interface.

Design:
- Abstract base class (ProfileStore) for future PostgreSQL swap
- SQLite implementation for initial deployment
- All methods are async for future compatibility
- JSON serialization via Pydantic's model_dump/model_validate
"""

from __future__ import annotations

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import ValidationError

from src.profile.schema import (
    AgentObservation,
    BehaviorProfile,
    Goal,
    LearningStyle,
    SkillEntry,
    UserProfile,
)

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/profile.db")

ProfileStorageReadErrorCode: TypeAlias = Literal[
    "profile_database_missing",
    "profile_table_missing",
    "profile_table_schema_invalid",
    "profile_record_json_invalid",
    "profile_record_schema_invalid",
    "profile_record_identity_mismatch",
    "profile_database_read_failed",
]


class ProfileStorageReadError(RuntimeError):
    """Content-safe failure from the strict read-only profile path."""

    def __init__(self, *, code: ProfileStorageReadErrorCode) -> None:
        self.code = code
        super().__init__(f"{code}: strict profile storage read failed")


def _reject_non_finite_json(value: str) -> object:
    raise ValueError("non-finite JSON number")


def _strict_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _strict_json_value(raw: object) -> object:
    if not isinstance(raw, str):
        raise TypeError("stored JSON must be text")
    return json.loads(
        raw,
        object_pairs_hook=_strict_json_pairs,
        parse_constant=_reject_non_finite_json,
    )


def _require_exact_fields(
    value: object,
    *,
    expected: frozenset[str],
) -> dict[str, object]:
    if not isinstance(value, dict) or frozenset(value) != expected:
        raise ValueError("stored profile object fields do not match the schema")
    return value


def _validate_complete_profile_payload(payload: dict[str, object]) -> None:
    root = _require_exact_fields(
        payload,
        expected=frozenset(UserProfile.model_fields),
    )
    skills = root["skills"]
    if not isinstance(skills, dict):
        raise TypeError("stored skills must be an object")
    for skill in skills.values():
        _require_exact_fields(skill, expected=frozenset(SkillEntry.model_fields))

    _require_exact_fields(
        root["learning_style"],
        expected=frozenset(LearningStyle.model_fields),
    )
    goals = root["goals"]
    if not isinstance(goals, list):
        raise TypeError("stored goals must be a list")
    for goal in goals:
        _require_exact_fields(goal, expected=frozenset(Goal.model_fields))

    _require_exact_fields(
        root["behavior"],
        expected=frozenset(BehaviorProfile.model_fields),
    )
    observations = root["agent_observations"]
    if not isinstance(observations, list):
        raise TypeError("stored observations must be a list")
    for observation in observations:
        _require_exact_fields(
            observation,
            expected=frozenset(AgentObservation.model_fields),
        )


def _validate_read_identity(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be normalized and non-blank")


# ── Abstract interface ─────────────────────────────────────────────────────


class ProfileStore(ABC):
    """Abstract interface for profile persistence.

    Implementations: SQLiteProfileStore, PostgresProfileStore, etc.
    """

    @abstractmethod
    async def save(self, profile: UserProfile) -> None:
        """Persist a user profile (insert or update)."""
        ...

    @abstractmethod
    async def load(self, user_id: str) -> UserProfile | None:
        """Load a user profile, or None if not found."""
        ...

    @abstractmethod
    async def delete(self, user_id: str) -> bool:
        """Delete a user profile. Returns True if it existed."""
        ...

    @abstractmethod
    async def list_users(self, limit: int = 100, offset: int = 0) -> list[str]:
        """List user IDs with profiles."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Total number of profiles."""
        ...


# ── SQLite implementation ──────────────────────────────────────────────────


class SQLiteProfileStore(ProfileStore):
    """SQLite-backed profile storage.

    Schema::

        CREATE TABLE IF NOT EXISTS profiles (
            user_id TEXT PRIMARY KEY,
            profile_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = Path(db_path or DEFAULT_DB_PATH)
        self._initialized = False

    @property
    def db_path(self) -> Path:
        return self._db_path

    async def _ensure_init(self) -> None:
        """Lazy initialization — create table on first access."""
        if self._initialized:
            return
        import aiosqlite

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_profiles_updated
                ON profiles(updated_at)
            """)
            await db.commit()
        self._initialized = True

    async def save(self, profile: UserProfile) -> None:
        """Insert or update a profile."""
        import aiosqlite

        await self._ensure_init()
        profile.touch()
        profile_json = profile.model_dump_json(exclude_none=True)
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(
                """
                INSERT INTO profiles (user_id, profile_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at = excluded.updated_at
            """,
                (profile.user_id, profile_json, profile.updated_at),
            )
            await db.commit()
        logger.debug("Saved profile for user=%s", profile.user_id)

    async def load(self, user_id: str) -> UserProfile | None:
        """Load a profile by user ID."""
        import aiosqlite

        await self._ensure_init()
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT profile_json FROM profiles WHERE user_id = ?",
                (user_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        try:
            data = json.loads(row[0])
            return UserProfile(**data)
        except Exception as exc:
            logger.warning("Failed to parse profile for user=%s: %s", user_id, exc)
            return None

    async def load_strict(self, user_id: str) -> UserProfile | None:
        """Read one profile without creating storage or repairing bad records."""

        _validate_read_identity(user_id, field_name="user_id")
        if not self._db_path.is_file():
            raise ProfileStorageReadError(code="profile_database_missing")

        import aiosqlite

        database_uri = f"{self._db_path.resolve().as_uri()}?mode=ro"
        try:
            async with aiosqlite.connect(database_uri, uri=True) as db:
                db.row_factory = aiosqlite.Row
                table_cursor = await db.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'profiles'"
                )
                if await table_cursor.fetchone() is None:
                    raise ProfileStorageReadError(code="profile_table_missing")

                schema_cursor = await db.execute("PRAGMA table_info(profiles)")
                schema_rows = await schema_cursor.fetchall()
                columns = frozenset(str(row[1]) for row in schema_rows)
                if not {"user_id", "profile_json"}.issubset(columns):
                    raise ProfileStorageReadError(code="profile_table_schema_invalid")

                cursor = await db.execute(
                    "SELECT user_id, profile_json FROM profiles WHERE user_id = ?",
                    (user_id,),
                )
                row = await cursor.fetchone()
        except ProfileStorageReadError:
            raise
        except sqlite3.Error as exc:
            raise ProfileStorageReadError(code="profile_database_read_failed") from exc

        if row is None:
            return None
        stored_user_id = row["user_id"]
        if stored_user_id != user_id:
            raise ProfileStorageReadError(code="profile_record_identity_mismatch")
        try:
            decoded = _strict_json_value(row["profile_json"])
        except (json.JSONDecodeError, TypeError, ValueError):
            raise ProfileStorageReadError(code="profile_record_json_invalid") from None
        if not isinstance(decoded, dict):
            raise ProfileStorageReadError(code="profile_record_schema_invalid")
        payload = decoded
        try:
            _validate_complete_profile_payload(payload)
            profile = UserProfile.model_validate(payload, strict=True)
        except (TypeError, ValueError, ValidationError):
            raise ProfileStorageReadError(
                code="profile_record_schema_invalid"
            ) from None
        if profile.user_id != user_id:
            raise ProfileStorageReadError(code="profile_record_identity_mismatch")
        return profile

    async def delete(self, user_id: str) -> bool:
        """Delete a profile. Returns True if it existed."""
        import aiosqlite

        await self._ensure_init()
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(
                "DELETE FROM profiles WHERE user_id = ?",
                (user_id,),
            )
            await db.commit()
            deleted = cursor.rowcount > 0
        if deleted:
            logger.debug("Deleted profile for user=%s", user_id)
        return deleted

    async def list_users(self, limit: int = 100, offset: int = 0) -> list[str]:
        """List user IDs, most recently updated first."""
        import aiosqlite

        await self._ensure_init()
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(
                "SELECT user_id FROM profiles ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def count(self) -> int:
        """Total profiles in storage."""
        import aiosqlite

        await self._ensure_init()
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM profiles")
            row = await cursor.fetchone()
        return row[0] if row else 0


# ── Factory ────────────────────────────────────────────────────────────────


def create_store(backend: str = "sqlite", **kwargs) -> ProfileStore:
    """Factory: create a ProfileStore instance.

    Args:
        backend: "sqlite" (default) or "postgres" (future).
        **kwargs: Passed to the store constructor.

    Returns:
        A ProfileStore instance.
    """
    if backend == "sqlite":
        return SQLiteProfileStore(**kwargs)
    if backend == "postgres":
        raise NotImplementedError("PostgreSQL profile store not yet implemented")
    raise ValueError(f"Unknown profile store backend: {backend}")
