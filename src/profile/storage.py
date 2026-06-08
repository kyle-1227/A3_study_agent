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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from src.profile.schema import UserProfile

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/profile.db")


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
            await db.execute("""
                INSERT INTO profiles (user_id, profile_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at = excluded.updated_at
            """, (profile.user_id, profile_json, profile.updated_at))
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
