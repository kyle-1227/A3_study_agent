from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_unwired_checkpoint_migration_runtime_is_removed() -> None:
    retired_paths = (
        REPO_ROOT / "src" / "checkpoint_migration" / "__init__.py",
        REPO_ROOT / "src" / "checkpoint_migration" / "cli.py",
        REPO_ROOT / "src" / "checkpoint_migration" / "contracts.py",
        REPO_ROOT / "src" / "checkpoint_migration" / "service.py",
        REPO_ROOT / "scripts" / "migrate_checkpoints.py",
        REPO_ROOT / "tests" / "test_checkpoint_migration.py",
    )

    for path in retired_paths:
        assert not path.exists(), path


def test_checkpoint_clear_runbook_requires_backup_and_preserves_schema_table() -> None:
    runbook = (
        REPO_ROOT / "docs" / "runbooks" / "checkpoint_clear_cutover.md"
    ).read_text(encoding="utf-8")

    assert "--format=custom" in runbook
    assert "pg_restore --list" in runbook
    assert "TRUNCATE TABLE checkpoint_writes, checkpoint_blobs, checkpoints;" in runbook
    assert "Do not add `checkpoint_migrations`" in runbook
    assert "Do not use `CASCADE`" in runbook
