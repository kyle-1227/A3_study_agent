"""Runtime-path contract for default SQLite memory stores."""

from __future__ import annotations

from pathlib import Path

import pytest

import src.memory.storage as memory_storage
from src.memory.storage import SQLiteMemoryStore, create_memory_store


def test_factory_uses_the_required_configured_sqlite_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configured_path = tmp_path / "runtime" / "memory.db"
    monkeypatch.setattr(
        memory_storage,
        "get_setting",
        lambda key, default: str(configured_path)
        if (key, default) == ("memory.db_path", None)
        else default,
    )

    store = create_memory_store()

    assert isinstance(store, SQLiteMemoryStore)
    assert store.db_path == configured_path


def test_factory_rejects_missing_default_sqlite_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(memory_storage, "get_setting", lambda _key, _default: None)

    with pytest.raises(RuntimeError, match="memory.db_path"):
        create_memory_store()
