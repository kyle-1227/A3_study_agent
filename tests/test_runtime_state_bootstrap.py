from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import app as a3_app
from src.profile.storage import SQLiteProfileStore


def test_legacy_runtime_state_migration_is_atomic_and_never_overwrites(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy" / "profile.db"
    target = tmp_path / "state" / "profile.db"
    legacy.parent.mkdir()
    target.parent.mkdir()
    legacy.write_bytes(b"first-database")

    assert a3_app._migrate_legacy_runtime_state_database(
        legacy_path=legacy,
        target_path=target,
    )
    assert target.read_bytes() == b"first-database"
    assert legacy.read_bytes() == b"first-database"
    assert not list(target.parent.glob("*.migrating"))

    legacy.write_bytes(b"changed-legacy")
    assert not a3_app._migrate_legacy_runtime_state_database(
        legacy_path=legacy,
        target_path=target,
    )
    assert target.read_bytes() == b"first-database"


def test_legacy_runtime_state_migration_allows_clean_install(tmp_path: Path) -> None:
    target = tmp_path / "state" / "memory.db"
    target.parent.mkdir()

    assert not a3_app._migrate_legacy_runtime_state_database(
        legacy_path=tmp_path / "missing" / "memory.db",
        target_path=target,
    )
    assert not target.exists()


@pytest.mark.parametrize("invalid_side", ["legacy", "target"])
def test_legacy_runtime_state_migration_rejects_directories(
    tmp_path: Path,
    invalid_side: str,
) -> None:
    legacy = tmp_path / "legacy.db"
    target = tmp_path / "target.db"
    legacy.write_bytes(b"database")
    if invalid_side == "legacy":
        legacy.unlink()
        legacy.mkdir()
    else:
        target.mkdir()

    with pytest.raises(RuntimeError, match="must be a regular file"):
        a3_app._migrate_legacy_runtime_state_database(
            legacy_path=legacy,
            target_path=target,
        )


@pytest.mark.asyncio
async def test_profile_store_explicit_initialize_creates_schema(tmp_path: Path) -> None:
    database = tmp_path / "profile.db"
    store = SQLiteProfileStore(database)

    await store.initialize()

    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'profiles'"
        ).fetchone()
    assert row == ("profiles",)
