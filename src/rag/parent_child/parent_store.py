"""Authoritative immutable SQLite storage for parent records."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sqlite3
from types import TracebackType
from typing import Sequence
from uuid import uuid4

from src.rag.parent_child._storage_io import resolve_under_root
from src.rag.parent_child.models import ParentRecord


class ParentStoreError(RuntimeError):
    """Base error for parent-store contract failures."""


class ParentStoreIntegrityError(ParentStoreError):
    """Raised when the SQLite store or a persisted record is inconsistent."""


class MissingParentError(ParentStoreError):
    """Raised when hydration requests parent IDs absent from the generation."""


_SCHEMA = """
CREATE TABLE store_metadata (
    key TEXT PRIMARY KEY NOT NULL,
    value TEXT NOT NULL CHECK (length(value) > 0)
) STRICT;

CREATE TABLE parents (
    parent_id TEXT PRIMARY KEY NOT NULL CHECK (length(parent_id) > 0),
    generation_id TEXT NOT NULL CHECK (length(generation_id) > 0),
    subject TEXT NOT NULL CHECK (length(subject) > 0),
    policy_id TEXT NOT NULL CHECK (length(policy_id) > 0),
    doc_id TEXT NOT NULL CHECK (length(doc_id) > 0),
    parent_index INTEGER NOT NULL CHECK (parent_index >= 0),
    content_sha1 TEXT NOT NULL CHECK (length(content_sha1) = 40),
    record_json TEXT NOT NULL CHECK (length(record_json) > 0),
    UNIQUE (doc_id, parent_index)
) STRICT;

CREATE INDEX parents_generation_subject_idx
    ON parents (generation_id, subject, parent_id);
"""


def _validate_timeout(busy_timeout_seconds: float) -> float:
    if isinstance(busy_timeout_seconds, bool) or busy_timeout_seconds <= 0:
        raise ValueError("busy_timeout_seconds must be greater than zero")
    return float(busy_timeout_seconds)


def _validate_record_for_store(
    record: ParentRecord,
    *,
    expected_generation_id: str,
) -> None:
    if record.generation_id != expected_generation_id:
        raise ParentStoreIntegrityError(
            f"parent {record.parent_id} belongs to a different generation"
        )
    expected_hash = hashlib.sha1(record.content.encode("utf-8")).hexdigest()
    if record.content_sha1 != expected_hash:
        raise ParentStoreIntegrityError(
            f"parent {record.parent_id} content hash does not match"
        )
    if record.parent_chars != len(record.content):
        raise ParentStoreIntegrityError(
            f"parent {record.parent_id} character count does not match"
        )


def create_parent_store(
    root: str | Path,
    relative_path: str,
    records: Sequence[ParentRecord],
    *,
    store_schema_version: str,
    expected_generation_id: str,
    busy_timeout_seconds: float,
) -> Path:
    """Create and atomically publish a complete immutable parent SQLite file."""

    if not store_schema_version:
        raise ValueError("store_schema_version is required")
    if not expected_generation_id:
        raise ValueError("expected_generation_id is required")
    timeout = _validate_timeout(busy_timeout_seconds)
    output_path = resolve_under_root(root, relative_path, must_exist=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)

    sorted_records = sorted(records, key=lambda item: item.parent_id)
    parent_ids = [record.parent_id for record in sorted_records]
    if len(parent_ids) != len(set(parent_ids)):
        raise ParentStoreIntegrityError("duplicate parent_id values are forbidden")
    for record in sorted_records:
        _validate_record_for_store(
            record,
            expected_generation_id=expected_generation_id,
        )

    temporary_path = output_path.parent / f".{output_path.name}.{uuid4().hex}.tmp"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            temporary_path,
            timeout=timeout,
            isolation_level=None,
        )
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA user_version = 1")
        connection.executescript(_SCHEMA)
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.executemany(
                "INSERT INTO store_metadata(key, value) VALUES (?, ?)",
                (
                    ("store_schema_version", store_schema_version),
                    ("generation_id", expected_generation_id),
                    ("record_count", str(len(sorted_records))),
                ),
            )
            connection.executemany(
                """
                INSERT INTO parents(
                    parent_id,
                    generation_id,
                    subject,
                    policy_id,
                    doc_id,
                    parent_index,
                    content_sha1,
                    record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        record.parent_id,
                        record.generation_id,
                        record.subject,
                        record.policy_id,
                        record.doc_id,
                        record.parent_index,
                        record.content_sha1,
                        record.model_dump_json(),
                    )
                    for record in sorted_records
                ),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        integrity_result = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity_result != ("ok",):
            raise ParentStoreIntegrityError("new parent store failed integrity_check")
        connection.close()
        connection = None
        if output_path.exists():
            raise FileExistsError(output_path)
        os.replace(temporary_path, output_path)
        return output_path
    finally:
        if connection is not None:
            connection.close()
        if temporary_path.exists():
            temporary_path.unlink()


class ParentStore:
    """Read-only parent hydration interface for one immutable generation."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        expected_schema_version: str,
        expected_generation_id: str,
    ) -> None:
        self._connection = connection
        self._expected_schema_version = expected_schema_version
        self._expected_generation_id = expected_generation_id
        self._closed = False

    @classmethod
    def open_readonly(
        cls,
        root: str | Path,
        relative_path: str,
        *,
        expected_schema_version: str,
        expected_generation_id: str,
        busy_timeout_seconds: float,
    ) -> ParentStore:
        """Open a ParentStore through SQLite's read-only URI mode."""

        if not expected_schema_version:
            raise ValueError("expected_schema_version is required")
        if not expected_generation_id:
            raise ValueError("expected_generation_id is required")
        timeout = _validate_timeout(busy_timeout_seconds)
        store_path = resolve_under_root(root, relative_path, must_exist=True)
        if not store_path.is_file() or store_path.is_symlink():
            raise ParentStoreError("parent store must be a regular file")
        connection = sqlite3.connect(
            f"{store_path.as_uri()}?mode=ro&immutable=1",
            uri=True,
            timeout=timeout,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA foreign_keys = ON")
        instance = cls(
            connection,
            expected_schema_version=expected_schema_version,
            expected_generation_id=expected_generation_id,
        )
        try:
            instance._verify_metadata()
        except BaseException:
            connection.close()
            raise
        return instance

    def _ensure_open(self) -> None:
        if self._closed:
            raise ParentStoreError("parent store is closed")

    def _verify_metadata(self) -> None:
        self._ensure_open()
        rows = self._connection.execute(
            "SELECT key, value FROM store_metadata ORDER BY key"
        ).fetchall()
        metadata = {str(row["key"]): str(row["value"]) for row in rows}
        expected_keys = {"generation_id", "record_count", "store_schema_version"}
        if set(metadata) != expected_keys:
            raise ParentStoreIntegrityError("parent store metadata keys are invalid")
        if metadata["store_schema_version"] != self._expected_schema_version:
            raise ParentStoreIntegrityError("parent store schema version mismatch")
        if metadata["generation_id"] != self._expected_generation_id:
            raise ParentStoreIntegrityError("parent store generation mismatch")
        try:
            declared_count = int(metadata["record_count"])
        except ValueError as exc:
            raise ParentStoreIntegrityError(
                "parent store record count is not an integer"
            ) from exc
        actual_count = int(
            self._connection.execute("SELECT COUNT(*) FROM parents").fetchone()[0]
        )
        if declared_count != actual_count:
            raise ParentStoreIntegrityError("parent store record count mismatch")

    def verify_integrity(self) -> None:
        """Run SQLite and record-level integrity checks without mutating the store."""

        self._ensure_open()
        result = self._connection.execute("PRAGMA integrity_check").fetchall()
        if [tuple(row) for row in result] != [("ok",)]:
            raise ParentStoreIntegrityError("parent store integrity_check failed")
        foreign_key_rows = self._connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()
        if foreign_key_rows:
            raise ParentStoreIntegrityError("parent store foreign_key_check failed")
        self._verify_metadata()
        rows = self._connection.execute(
            """
            SELECT parent_id, generation_id, subject, policy_id, doc_id,
                   parent_index, content_sha1, record_json
            FROM parents
            ORDER BY parent_id
            """
        ).fetchall()
        for row in rows:
            self._record_from_row(row)

    def _record_from_row(self, row: sqlite3.Row) -> ParentRecord:
        record = ParentRecord.model_validate_json(str(row["record_json"]))
        persisted_values = {
            "parent_id": row["parent_id"],
            "generation_id": row["generation_id"],
            "subject": row["subject"],
            "policy_id": row["policy_id"],
            "doc_id": row["doc_id"],
            "parent_index": row["parent_index"],
            "content_sha1": row["content_sha1"],
        }
        for field, persisted in persisted_values.items():
            if getattr(record, field) != persisted:
                raise ParentStoreIntegrityError(
                    f"parent {record.parent_id} persisted {field} mismatch"
                )
        _validate_record_for_store(
            record,
            expected_generation_id=self._expected_generation_id,
        )
        return record

    def get_many(self, parent_ids: Sequence[str]) -> tuple[ParentRecord, ...]:
        """Hydrate all requested parents in request order or fail explicitly."""

        self._ensure_open()
        requested = tuple(parent_ids)
        if any(not parent_id for parent_id in requested):
            raise ValueError("parent_ids cannot contain an empty identifier")
        if len(requested) != len(set(requested)):
            raise ValueError("parent_ids cannot contain duplicates")
        if not requested:
            return ()

        variable_limit = self._connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
        by_id: dict[str, ParentRecord] = {}
        for start in range(0, len(requested), variable_limit):
            batch = requested[start : start + variable_limit]
            placeholders = ",".join("?" for _ in batch)
            rows = self._connection.execute(
                f"""
                SELECT parent_id, generation_id, subject, policy_id, doc_id,
                       parent_index, content_sha1, record_json
                FROM parents
                WHERE parent_id IN ({placeholders})
                """,
                batch,
            ).fetchall()
            for row in rows:
                record = self._record_from_row(row)
                by_id[record.parent_id] = record

        missing = tuple(parent_id for parent_id in requested if parent_id not in by_id)
        if missing:
            raise MissingParentError(f"missing parent IDs: {', '.join(missing)}")
        return tuple(by_id[parent_id] for parent_id in requested)

    def close(self) -> None:
        """Close the read-only SQLite connection."""

        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> ParentStore:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
