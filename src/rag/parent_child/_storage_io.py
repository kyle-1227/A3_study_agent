"""Filesystem primitives shared by immutable parent-child generation artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any
from uuid import uuid4

from pydantic import BaseModel


_GENERATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ArtifactPathError(ValueError):
    """Raised when an artifact path escapes its configured storage root."""


def validate_generation_id(generation_id: str) -> str:
    """Validate a generation identifier before using it in a filesystem path."""

    if not _GENERATION_ID_PATTERN.fullmatch(generation_id):
        raise ArtifactPathError(
            "generation_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}"
        )
    if generation_id in {".", ".."}:
        raise ArtifactPathError("generation_id cannot be a dot path")
    return generation_id


def validate_relative_path(relative_path: str) -> PurePosixPath:
    """Return a validated, canonical POSIX relative path."""

    if not relative_path or "\\" in relative_path:
        raise ArtifactPathError("artifact path must be a non-empty POSIX path")
    parsed = PurePosixPath(relative_path)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise ArtifactPathError("artifact path must stay below its storage root")
    return parsed


def resolve_under_root(
    root: str | Path,
    relative_path: str,
    *,
    must_exist: bool,
) -> Path:
    """Resolve a canonical relative path and reject symlink/path traversal escapes."""

    logical_root = Path(root)
    if logical_root.is_symlink():
        raise ArtifactPathError("storage root must not be a symlink")
    root_path = logical_root.resolve(strict=False)
    parsed = validate_relative_path(relative_path)
    logical_candidate = root_path
    for part in parsed.parts:
        logical_candidate /= part
        if logical_candidate.is_symlink():
            raise ArtifactPathError("artifact paths must not contain symlinks")
    candidate = logical_candidate.resolve(strict=must_exist)
    if not candidate.is_relative_to(root_path):
        raise ArtifactPathError("resolved artifact path escapes its storage root")
    return candidate


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically for fingerprints and immutable artifacts."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def model_json_bytes(model: BaseModel) -> bytes:
    """Encode a validated Pydantic model using the canonical JSON contract."""

    return canonical_json_bytes(model.model_dump(mode="json"))


def sha256_bytes(value: bytes) -> str:
    """Return the hexadecimal SHA-256 digest for bytes."""

    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a regular file without loading it fully into memory."""

    file_path = Path(path)
    if not file_path.is_file() or file_path.is_symlink():
        raise ArtifactPathError(f"artifact is not a regular file: {file_path.name}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    """Hash a file or directory tree using relative names and file digests."""

    target = Path(path)
    if target.is_symlink():
        raise ArtifactPathError(f"symlink artifacts are forbidden: {target.name}")
    if target.is_file():
        return sha256_file(target)
    if not target.is_dir():
        raise ArtifactPathError(f"artifact does not exist: {target.name}")

    entries: list[list[str]] = []
    for child in sorted(target.rglob("*"), key=lambda item: item.as_posix()):
        if child.is_symlink():
            raise ArtifactPathError(
                f"symlink inside artifact directory is forbidden: {child.name}"
            )
        if child.is_file():
            entries.append([child.relative_to(target).as_posix(), sha256_file(child)])
    return sha256_bytes(canonical_json_bytes(entries))


def atomic_write_bytes(
    root: str | Path,
    relative_path: str,
    value: bytes,
    *,
    overwrite: bool,
) -> Path:
    """Atomically write bytes below root, fsyncing before the final replace."""

    root_path = Path(root).resolve(strict=False)
    root_path.mkdir(parents=True, exist_ok=True)
    output_path = resolve_under_root(root_path, relative_path, must_exist=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)

    temporary_path = output_path.parent / f".{output_path.name}.{uuid4().hex}.tmp"
    try:
        with temporary_path.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        if output_path.exists() and not overwrite:
            raise FileExistsError(output_path)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path
