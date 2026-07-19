"""Contained disposable Chroma copies for the mutable primary index."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.rag.parent_child._storage_io import sha256_path
from src.rag.parent_child.manifests import GenerationManifest

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_OWNER_FILE = ".a3_chroma_runtime_owner.json"
_RUNTIME_DIRECTORY = ".runtime_chroma"
CHROMA_RUNTIME_OWNER_SCHEMA_VERSION = "chroma_runtime_snapshot_v1"


class ChromaRuntimeSnapshotError(RuntimeError):
    """A primary Chroma artifact cannot be safely copied or cleaned."""


class _SnapshotOwner(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _reject_symlinks(path: Path) -> None:
    if path.is_symlink():
        raise ChromaRuntimeSnapshotError("Chroma snapshot paths must not be symlinks")
    for child in path.rglob("*"):
        if child.is_symlink():
            raise ChromaRuntimeSnapshotError(
                "primary Chroma artifacts must not contain symlinks"
            )


def _contained_directory(path: Path, *, root: Path) -> Path:
    if not path.is_absolute() or not root.is_absolute():
        raise ChromaRuntimeSnapshotError("snapshot paths must be absolute")
    if path.is_symlink() or root.is_symlink():
        raise ChromaRuntimeSnapshotError("snapshot paths must not be symlinks")
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise ChromaRuntimeSnapshotError("snapshot path escapes its containment root")
    if not resolved.is_dir():
        raise ChromaRuntimeSnapshotError("snapshot source must be a directory")
    _reject_symlinks(resolved)
    return resolved


class ChromaRuntimeSnapshot:
    """A marker-owned runtime copy; it never selects a legacy data directory."""

    def __init__(
        self,
        *,
        index_root: Path,
        snapshot_root: Path,
        persist_directory: Path,
        owner: _SnapshotOwner,
    ) -> None:
        self.index_root = index_root
        self.snapshot_root = snapshot_root
        self.persist_directory = persist_directory
        self._owner = owner
        self._closed = False

    @property
    def source_sha256(self) -> str:
        """Digest of the source directory that this owned copy verified."""

        return self._owner.source_sha256

    @classmethod
    def create(
        cls,
        *,
        index_root: Path,
        source_directory: Path,
        owner_schema_version: str,
        expected_source_sha256: str | None = None,
    ) -> "ChromaRuntimeSnapshot":
        """Copy a contained primary Chroma directory and verify copied bytes.

        expected_source_sha256 exists only for offline artifact tooling
        compatibility. The primary runtime deliberately omits it: the source is
        verified by containment, no-symlink checks, and a before/after digest
        computed in this one operation rather than any sealed manifest.
        """

        if not owner_schema_version:
            raise ChromaRuntimeSnapshotError("owner_schema_version is required")
        if (
            expected_source_sha256 is not None
            and _SHA256_PATTERN.fullmatch(expected_source_sha256) is None
        ):
            raise ChromaRuntimeSnapshotError(
                "expected_source_sha256 must be a lowercase SHA-256 value"
            )
        if index_root.is_symlink():
            raise ChromaRuntimeSnapshotError("index_root must not be a symlink")
        root = index_root.resolve(strict=True)
        source = _contained_directory(source_directory, root=root)
        source_sha256 = sha256_path(source)
        if (
            expected_source_sha256 is not None
            and source_sha256 != expected_source_sha256
        ):
            raise ChromaRuntimeSnapshotError(
                "runtime source differs from expected offline artifact"
            )
        runtime_root = root / _RUNTIME_DIRECTORY
        if runtime_root.exists() and runtime_root.is_symlink():
            raise ChromaRuntimeSnapshotError(
                "runtime Chroma root must not be a symlink"
            )
        runtime_root.mkdir(parents=False, exist_ok=True)
        runtime_root = _contained_directory(runtime_root, root=root)
        snapshot_id = uuid4().hex
        snapshot_root = runtime_root / snapshot_id
        persist_directory = snapshot_root / "chroma"
        owner = _SnapshotOwner(
            schema_version=owner_schema_version,
            snapshot_id=snapshot_id,
            source_sha256=source_sha256,
        )
        try:
            snapshot_root.mkdir(exist_ok=False)
            (snapshot_root / _OWNER_FILE).write_text(
                json.dumps(
                    owner.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            shutil.copytree(source, persist_directory)
            _reject_symlinks(persist_directory)
            if sha256_path(persist_directory) != source_sha256:
                raise ChromaRuntimeSnapshotError(
                    "runtime Chroma copy differs from primary source"
                )
            return cls(
                index_root=root,
                snapshot_root=snapshot_root,
                persist_directory=persist_directory,
                owner=owner,
            )
        except BaseException:
            if snapshot_root.exists():
                resolved = snapshot_root.resolve(strict=True)
                if not resolved.is_relative_to(runtime_root):
                    raise ChromaRuntimeSnapshotError(
                        "partial snapshot cleanup escaped runtime root"
                    )
                shutil.rmtree(resolved)
            raise

    def close(self) -> None:
        if self._closed:
            return
        runtime_root = (self.index_root / _RUNTIME_DIRECTORY).resolve(strict=True)
        snapshot_root = self.snapshot_root.resolve(strict=True)
        if not snapshot_root.is_relative_to(runtime_root):
            raise ChromaRuntimeSnapshotError("snapshot cleanup escapes runtime root")
        marker_path = snapshot_root / _OWNER_FILE
        if marker_path.is_symlink() or not marker_path.is_file():
            raise ChromaRuntimeSnapshotError("snapshot ownership marker is missing")
        try:
            marker = _SnapshotOwner.model_validate_json(
                marker_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise ChromaRuntimeSnapshotError(
                "snapshot ownership marker is invalid"
            ) from exc
        if marker != self._owner or snapshot_root.name != marker.snapshot_id:
            raise ChromaRuntimeSnapshotError("snapshot ownership marker mismatch")
        shutil.rmtree(snapshot_root)
        self._closed = True

    def __enter__(self) -> "ChromaRuntimeSnapshot":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def chroma_artifact_sha256(manifest: GenerationManifest) -> str:
    """Offline compatibility helper; the primary runtime never calls this."""

    matches = [
        descriptor
        for descriptor in manifest.artifacts
        if descriptor.artifact_type == "chroma_children"
        and descriptor.relative_path == "chroma_children"
    ]
    if len(matches) != 1 or _SHA256_PATTERN.fullmatch(matches[0].sha256) is None:
        raise ChromaRuntimeSnapshotError(
            "generation requires one canonical Chroma descriptor"
        )
    return matches[0].sha256


__all__ = [
    "CHROMA_RUNTIME_OWNER_SCHEMA_VERSION",
    "ChromaRuntimeSnapshot",
    "ChromaRuntimeSnapshotError",
    "chroma_artifact_sha256",
]
