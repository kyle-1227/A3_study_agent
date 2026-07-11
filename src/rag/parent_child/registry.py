"""Transactional deployment registry for immutable RAG generations."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import re
import shutil
import sqlite3
from types import TracebackType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.rag.parent_child._storage_io import resolve_under_root, validate_generation_id
from src.rag.parent_child.generation import (
    GenerationOwnershipMarker,
    generation_final_relative_path,
    generation_staging_relative_path,
    validate_sealed_generation,
)
from src.rag.parent_child.manifests import read_strict_model


GenerationState = Literal[
    "BUILDING", "VALIDATING", "READY", "FAILED", "DELETING", "DELETED"
]
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_REGISTRY_SCHEMA = """
CREATE TABLE registry_metadata (
    key TEXT PRIMARY KEY NOT NULL,
    value TEXT NOT NULL CHECK (length(value) > 0)
) STRICT;

CREATE TABLE generations (
    generation_id TEXT PRIMARY KEY NOT NULL,
    state TEXT NOT NULL CHECK (
        state IN ('BUILDING', 'VALIDATING', 'READY', 'FAILED', 'DELETING', 'DELETED')
    ),
    directory_relative_path TEXT NOT NULL,
    manifest_sha256 TEXT,
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    failure_code TEXT,
    failure_type TEXT,
    CHECK (
        (state = 'READY' AND manifest_sha256 IS NOT NULL)
        OR state <> 'READY'
    )
) STRICT;

CREATE TABLE deployment (
    singleton INTEGER PRIMARY KEY NOT NULL CHECK (singleton = 1),
    primary_generation_id TEXT,
    previous_generation_id TEXT,
    shadow_generation_id TEXT,
    revision INTEGER NOT NULL CHECK (revision >= 0),
    updated_at_utc TEXT NOT NULL,
    FOREIGN KEY(primary_generation_id) REFERENCES generations(generation_id),
    FOREIGN KEY(previous_generation_id) REFERENCES generations(generation_id),
    FOREIGN KEY(shadow_generation_id) REFERENCES generations(generation_id)
) STRICT;

CREATE TABLE activation_history (
    activation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id TEXT NOT NULL,
    replaced_generation_id TEXT,
    revision INTEGER NOT NULL CHECK (revision > 0),
    activated_at_utc TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('activate', 'rollback')),
    FOREIGN KEY(generation_id) REFERENCES generations(generation_id),
    FOREIGN KEY(replaced_generation_id) REFERENCES generations(generation_id)
) STRICT;
"""


class GenerationRegistryError(RuntimeError):
    """Base class for typed registry and deployment failures."""


class GenerationTransitionError(GenerationRegistryError):
    """Raised when a requested lifecycle transition is not allowed."""


class GenerationActivationError(GenerationRegistryError):
    """Raised when a generation cannot become primary or shadow."""


class GenerationCleanupError(GenerationRegistryError):
    """Raised when cleanup would touch a protected or unowned generation."""


class GenerationRegistryRecord(BaseModel):
    """One lifecycle record with sanitized failure diagnostics."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    generation_id: str
    state: GenerationState
    directory_relative_path: str
    manifest_sha256: str | None
    created_at_utc: datetime
    updated_at_utc: datetime
    failure_code: str | None
    failure_type: str | None


class DeploymentSnapshot(BaseModel):
    """Atomic primary/previous/shadow deployment view."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    primary_generation_id: str | None
    previous_generation_id: str | None
    shadow_generation_id: str | None
    revision: int = Field(ge=0)
    updated_at_utc: datetime


class ActivationRecord(BaseModel):
    """Auditable result of an explicit activation transaction."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    generation_id: str
    replaced_generation_id: str | None
    revision: int = Field(gt=0)
    action: Literal["activate", "rollback"]
    activated_at_utc: datetime


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _validate_timeout(value: float) -> float:
    if isinstance(value, bool) or value <= 0:
        raise ValueError("busy_timeout_seconds must be greater than zero")
    return float(value)


def create_generation_registry(
    registry_path: Path,
    *,
    schema_version: str,
    busy_timeout_seconds: float,
) -> Path:
    """Create a new registry atomically at an explicit path."""

    if not schema_version:
        raise ValueError("schema_version is required")
    timeout = _validate_timeout(busy_timeout_seconds)
    if registry_path.is_symlink():
        raise GenerationRegistryError("registry path must not be a symlink")
    path = registry_path.resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.building")
    if temporary.exists():
        raise FileExistsError(temporary)
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(temporary, timeout=timeout, isolation_level=None)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        connection.executescript(_REGISTRY_SCHEMA)
        now = _timestamp()
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "INSERT INTO registry_metadata(key, value) VALUES (?, ?)",
                ("schema_version", schema_version),
            )
            connection.execute(
                """
                INSERT INTO deployment(
                    singleton, primary_generation_id, previous_generation_id,
                    shadow_generation_id, revision, updated_at_utc
                ) VALUES (1, NULL, NULL, NULL, 0, ?)
                """,
                (now,),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise GenerationRegistryError("new registry failed integrity_check")
        connection.close()
        connection = None
        temporary.replace(path)
        return path
    finally:
        if connection is not None:
            connection.close()
        if temporary.exists():
            temporary.unlink()


class GenerationRegistry:
    """Transactional registry; no method performs request-level fallback."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        index_root: Path,
        expected_schema_version: str,
        marker_schema_version: str,
    ) -> None:
        self._connection = connection
        self._index_root = index_root
        self._expected_schema_version = expected_schema_version
        self._marker_schema_version = marker_schema_version
        self._closed = False

    @classmethod
    def open(
        cls,
        registry_path: Path,
        *,
        index_root: Path,
        expected_schema_version: str,
        marker_schema_version: str,
        busy_timeout_seconds: float,
    ) -> GenerationRegistry:
        """Open and verify an existing deployment registry."""

        timeout = _validate_timeout(busy_timeout_seconds)
        if registry_path.is_symlink():
            raise GenerationRegistryError("registry must be a regular non-symlink file")
        path = registry_path.resolve(strict=True)
        resolved_root = index_root.resolve(strict=True)
        if not path.is_file():
            raise GenerationRegistryError("registry must be a regular file")
        if not path.is_relative_to(resolved_root):
            raise GenerationRegistryError("registry must be contained by index_root")
        connection = sqlite3.connect(path, timeout=timeout, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        instance = cls(
            connection,
            index_root=resolved_root,
            expected_schema_version=expected_schema_version,
            marker_schema_version=marker_schema_version,
        )
        try:
            instance.verify_integrity()
        except BaseException:
            connection.close()
            raise
        return instance

    def _ensure_open(self) -> None:
        if self._closed:
            raise GenerationRegistryError("generation registry is closed")

    def verify_integrity(self) -> None:
        """Verify SQLite integrity and the exact registry schema identity."""

        self._ensure_open()
        integrity_row = self._connection.execute("PRAGMA integrity_check").fetchone()
        if integrity_row is None or tuple(integrity_row) != ("ok",):
            raise GenerationRegistryError("registry integrity_check failed")
        if self._connection.execute("PRAGMA foreign_key_check").fetchall():
            raise GenerationRegistryError("registry foreign_key_check failed")
        rows = self._connection.execute(
            "SELECT key, value FROM registry_metadata"
        ).fetchall()
        metadata = {str(row["key"]): str(row["value"]) for row in rows}
        if metadata != {"schema_version": self._expected_schema_version}:
            raise GenerationRegistryError("registry schema version mismatch")
        if (
            self._connection.execute("SELECT COUNT(*) FROM deployment").fetchone()[0]
            != 1
        ):
            raise GenerationRegistryError("registry deployment singleton is missing")

    def register_building(self, generation_id: str) -> GenerationRegistryRecord:
        """Register one newly created immutable generation workspace."""

        self._ensure_open()
        generation_id = validate_generation_id(generation_id)
        now = _timestamp()
        relative_path = generation_final_relative_path(generation_id)
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._connection.execute(
                """
                INSERT INTO generations(
                    generation_id, state, directory_relative_path,
                    manifest_sha256, created_at_utc, updated_at_utc,
                    failure_code, failure_type
                ) VALUES (?, 'BUILDING', ?, NULL, ?, ?, NULL, NULL)
                """,
                (generation_id, relative_path, now, now),
            )
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise
        return self.get_generation(generation_id)

    def transition(self, generation_id: str, target_state: GenerationState) -> None:
        """Apply an allowed non-deployment lifecycle transition."""

        allowed: dict[GenerationState, set[GenerationState]] = {
            "BUILDING": {"VALIDATING", "FAILED"},
            "VALIDATING": {"READY", "FAILED"},
            "READY": {"DELETING"},
            "FAILED": {"DELETING"},
            "DELETING": {"DELETED"},
            "DELETED": set(),
        }
        record = self.get_generation(generation_id)
        if target_state not in allowed[record.state]:
            raise GenerationTransitionError(
                f"transition {record.state}->{target_state} is not allowed"
            )
        if target_state == "READY":
            raise GenerationTransitionError(
                "mark_ready requires a sealed manifest digest"
            )
        now = _timestamp()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            updated = self._connection.execute(
                """
                UPDATE generations SET state = ?, updated_at_utc = ?
                WHERE generation_id = ? AND state = ?
                """,
                (target_state, now, generation_id, record.state),
            ).rowcount
            if updated != 1:
                raise GenerationTransitionError("generation state changed concurrently")
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise

    def mark_failed(
        self,
        generation_id: str,
        *,
        failure_code: str,
        failure_type: str,
    ) -> None:
        """Record bounded diagnostics for a failed build or validation."""

        if not failure_code or not failure_type:
            raise ValueError("failure_code and failure_type are required")
        if len(failure_code) > 128 or len(failure_type) > 128:
            raise ValueError("failure diagnostics exceed the bounded registry contract")
        record = self.get_generation(generation_id)
        if record.state not in {"BUILDING", "VALIDATING"}:
            raise GenerationTransitionError("only an in-progress generation can fail")
        now = _timestamp()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            updated = self._connection.execute(
                """
                UPDATE generations
                SET state='FAILED', updated_at_utc=?, failure_code=?, failure_type=?
                WHERE generation_id=? AND state=?
                """,
                (now, failure_code, failure_type, generation_id, record.state),
            ).rowcount
            if updated != 1:
                raise GenerationTransitionError("generation state changed concurrently")
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise

    def mark_ready(self, generation_id: str, *, manifest_sha256: str) -> None:
        """Mark a validating, sealed generation READY after full verification."""

        if _SHA256_PATTERN.fullmatch(manifest_sha256) is None:
            raise ValueError("manifest_sha256 must be a lowercase SHA-256 value")
        record = self.get_generation(generation_id)
        if record.state != "VALIDATING":
            raise GenerationTransitionError("only VALIDATING can become READY")
        validate_sealed_generation(
            self._index_root,
            generation_id,
            expected_manifest_sha256=manifest_sha256,
            expected_marker_schema_version=self._marker_schema_version,
        )
        now = _timestamp()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            updated = self._connection.execute(
                """
                UPDATE generations
                SET state='READY', manifest_sha256=?, updated_at_utc=?
                WHERE generation_id=? AND state='VALIDATING'
                """,
                (manifest_sha256, now, generation_id),
            ).rowcount
            if updated != 1:
                raise GenerationTransitionError("generation state changed concurrently")
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise

    def get_generation(self, generation_id: str) -> GenerationRegistryRecord:
        """Return one lifecycle row or fail explicitly."""

        self._ensure_open()
        row = self._connection.execute(
            """
            SELECT generation_id, state, directory_relative_path, manifest_sha256,
                   created_at_utc, updated_at_utc, failure_code, failure_type
            FROM generations WHERE generation_id=?
            """,
            (generation_id,),
        ).fetchone()
        if row is None:
            raise GenerationRegistryError(f"unknown generation: {generation_id}")
        return GenerationRegistryRecord(
            generation_id=str(row["generation_id"]),
            state=str(row["state"]),
            directory_relative_path=str(row["directory_relative_path"]),
            manifest_sha256=(
                str(row["manifest_sha256"])
                if row["manifest_sha256"] is not None
                else None
            ),
            created_at_utc=datetime.fromisoformat(str(row["created_at_utc"])),
            updated_at_utc=datetime.fromisoformat(str(row["updated_at_utc"])),
            failure_code=(
                str(row["failure_code"]) if row["failure_code"] is not None else None
            ),
            failure_type=(
                str(row["failure_type"]) if row["failure_type"] is not None else None
            ),
        )

    def deployment(self) -> DeploymentSnapshot:
        """Return the one atomic deployment pointer snapshot."""

        self._ensure_open()
        row = self._connection.execute(
            """
            SELECT primary_generation_id, previous_generation_id,
                   shadow_generation_id, revision, updated_at_utc
            FROM deployment WHERE singleton=1
            """
        ).fetchone()
        if row is None:
            raise GenerationRegistryError("deployment singleton is missing")
        return DeploymentSnapshot(
            primary_generation_id=row["primary_generation_id"],
            previous_generation_id=row["previous_generation_id"],
            shadow_generation_id=row["shadow_generation_id"],
            revision=int(row["revision"]),
            updated_at_utc=datetime.fromisoformat(str(row["updated_at_utc"])),
        )

    def _activate(
        self,
        generation_id: str,
        *,
        action: Literal["activate", "rollback"],
    ) -> ActivationRecord:
        record = self.get_generation(generation_id)
        if record.state != "READY" or record.manifest_sha256 is None:
            raise GenerationActivationError(
                "only a verified READY generation can activate"
            )
        validate_sealed_generation(
            self._index_root,
            generation_id,
            expected_manifest_sha256=record.manifest_sha256,
            expected_marker_schema_version=self._marker_schema_version,
        )
        now = _timestamp()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            current = self._connection.execute(
                "SELECT primary_generation_id, revision FROM deployment WHERE singleton=1"
            ).fetchone()
            replaced = current["primary_generation_id"]
            revision = int(current["revision"]) + 1
            self._connection.execute(
                """
                UPDATE deployment
                SET previous_generation_id=primary_generation_id,
                    primary_generation_id=?, revision=?, updated_at_utc=?
                WHERE singleton=1
                """,
                (generation_id, revision, now),
            )
            self._connection.execute(
                """
                INSERT INTO activation_history(
                    generation_id, replaced_generation_id, revision,
                    activated_at_utc, action
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (generation_id, replaced, revision, now, action),
            )
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise
        return ActivationRecord(
            generation_id=generation_id,
            replaced_generation_id=replaced,
            revision=revision,
            action=action,
            activated_at_utc=datetime.fromisoformat(now),
        )

    def activate(self, generation_id: str) -> ActivationRecord:
        """Explicitly activate a READY generation after revalidation."""

        return self._activate(generation_id, action="activate")

    def rollback(self) -> ActivationRecord:
        """Explicitly reactivate the recorded previous READY generation."""

        previous = self.deployment().previous_generation_id
        if previous is None:
            raise GenerationActivationError("no previous generation is recorded")
        return self._activate(previous, action="rollback")

    def set_shadow(self, generation_id: str | None) -> DeploymentSnapshot:
        """Set or clear an explicit shadow pointer; it never serves by failure."""

        if generation_id is not None:
            record = self.get_generation(generation_id)
            if record.state != "READY" or record.manifest_sha256 is None:
                raise GenerationActivationError("shadow generation must be READY")
            validate_sealed_generation(
                self._index_root,
                generation_id,
                expected_manifest_sha256=record.manifest_sha256,
                expected_marker_schema_version=self._marker_schema_version,
            )
        now = _timestamp()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            self._connection.execute(
                """
                UPDATE deployment SET shadow_generation_id=?, updated_at_utc=?
                WHERE singleton=1
                """,
                (generation_id, now),
            )
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise
        return self.deployment()

    def cleanup_generation(self, generation_id: str) -> None:
        """Delete one registry-owned, unprotected READY/FAILED generation."""

        record = self.get_generation(generation_id)
        deployment = self.deployment()
        protected = {
            deployment.primary_generation_id,
            deployment.previous_generation_id,
            deployment.shadow_generation_id,
        }
        if generation_id in protected:
            raise GenerationCleanupError(
                "protected deployment generation cannot be cleaned"
            )
        if record.state not in {"READY", "FAILED"}:
            raise GenerationCleanupError(
                "only READY or FAILED generations can be cleaned"
            )
        if record.state == "FAILED":
            staging_relative = generation_staging_relative_path(generation_id)
            final_relative = generation_final_relative_path(generation_id)
            staging_path = resolve_under_root(
                self._index_root,
                staging_relative,
                must_exist=False,
            )
            final_path = resolve_under_root(
                self._index_root,
                final_relative,
                must_exist=False,
            )
            existing = tuple(
                relative
                for relative, path in (
                    (staging_relative, staging_path),
                    (final_relative, final_path),
                )
                if path.exists()
            )
            if len(existing) != 1:
                raise GenerationCleanupError(
                    "FAILED generation must own exactly one staging or final directory"
                )
            relative_path = existing[0]
        else:
            relative_path = generation_final_relative_path(generation_id)
        generation_path = resolve_under_root(
            self._index_root,
            relative_path,
            must_exist=True,
        )
        owner_path = resolve_under_root(
            generation_path,
            ".a3_generation_owner.json",
            must_exist=True,
        )
        if not owner_path.is_file() or owner_path.is_symlink():
            raise GenerationCleanupError("generation ownership marker is missing")
        try:
            marker = read_strict_model(
                generation_path,
                ".a3_generation_owner.json",
                GenerationOwnershipMarker,
            )
        except Exception as exc:
            raise GenerationCleanupError(
                "generation ownership marker is invalid"
            ) from exc
        if (
            marker.generation_id != generation_id
            or marker.owner != "a3_parent_child_generation"
            or marker.schema_version != self._marker_schema_version
        ):
            raise GenerationCleanupError("generation ownership marker mismatch")

        self.transition(generation_id, "DELETING")
        try:
            resolved_path = generation_path.resolve(strict=True)
            if not resolved_path.is_relative_to(self._index_root):
                raise GenerationCleanupError("cleanup path escapes index_root")
            if resolved_path == self._index_root:
                raise GenerationCleanupError("cleanup cannot target index_root")
            shutil.rmtree(resolved_path)
            self._connection.execute("BEGIN IMMEDIATE")
            self._connection.execute(
                """
                UPDATE generations
                SET state='DELETED', updated_at_utc=?
                WHERE generation_id=? AND state='DELETING'
                """,
                (_timestamp(), generation_id),
            )
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> GenerationRegistry:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = [
    "ActivationRecord",
    "DeploymentSnapshot",
    "GenerationActivationError",
    "GenerationCleanupError",
    "GenerationRegistry",
    "GenerationRegistryError",
    "GenerationRegistryRecord",
    "GenerationTransitionError",
    "create_generation_registry",
]
