"""Project-contained, symlink-safe filesystem primitives for RAG tooling."""

from __future__ import annotations

import os
from pathlib import Path
import stat
from uuid import uuid4


class ProjectPathError(ValueError):
    """A tooling path is invalid, escapes its project root, or is a link."""


_REPARSE_POINT_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)


def resolve_project_root(project_root: Path) -> Path:
    """Return an existing non-link project directory without creating it."""

    if not isinstance(project_root, Path):
        raise TypeError("project_root must be a pathlib.Path instance")
    logical_root = Path(os.path.abspath(str(project_root)))
    _reject_reparse_point(logical_root, field_name="project_root")
    if not logical_root.exists() or not logical_root.is_dir():
        raise ProjectPathError("project_root must be an existing directory")
    try:
        return logical_root.resolve(strict=True)
    except OSError as exc:
        raise ProjectPathError("project_root cannot be resolved") from exc


def resolve_project_path(
    project_root: Path,
    value: Path,
    *,
    must_exist: bool,
) -> Path:
    """Resolve one path below ``project_root`` while rejecting every link part."""

    if not isinstance(value, Path):
        raise TypeError("path value must be a pathlib.Path instance")
    root = resolve_project_root(project_root)
    candidate_input = value if value.is_absolute() else root / value
    candidate = Path(os.path.abspath(str(candidate_input)))
    try:
        relative_parts = candidate.relative_to(root).parts
    except ValueError as exc:
        raise ProjectPathError("path must remain inside project_root") from exc

    current = root
    _reject_reparse_point(current, field_name="project_root")
    for part in relative_parts:
        current /= part
        _reject_reparse_point(current, field_name="path")

    if must_exist:
        if not candidate.exists():
            raise ProjectPathError("required path does not exist")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ProjectPathError("required path cannot be resolved") from exc
        if not resolved.is_relative_to(root):
            raise ProjectPathError("resolved path escapes project_root")
        return resolved
    return candidate


def require_project_file(project_root: Path, value: Path) -> Path:
    """Return one existing regular file below the project root."""

    path = resolve_project_path(project_root, value, must_exist=True)
    if not path.is_file():
        raise ProjectPathError("path must be an existing regular file")
    return path


def require_project_directory(project_root: Path, value: Path) -> Path:
    """Return one existing directory below the project root."""

    path = resolve_project_path(project_root, value, must_exist=True)
    if not path.is_dir():
        raise ProjectPathError("path must be an existing directory")
    return path


def atomic_write_project_bytes(
    project_root: Path,
    output_path: Path,
    value: bytes,
    *,
    overwrite: bool,
) -> Path:
    """Atomically write bytes below a non-link project root.

    The helper does not create the project root and refuses to replace an existing
    output unless the caller deliberately supplies ``overwrite=True``.
    """

    if not isinstance(value, bytes):
        raise TypeError("value must be bytes")
    output = resolve_project_path(project_root, output_path, must_exist=False)
    if output.exists() and not output.is_file():
        raise ProjectPathError("output path must be a regular file when it exists")
    if output.exists() and not overwrite:
        raise FileExistsError(output)

    output.parent.mkdir(parents=True, exist_ok=True)
    output = resolve_project_path(project_root, output, must_exist=False)
    temporary = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        if output.exists() and not overwrite:
            raise FileExistsError(output)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def _reject_reparse_point(path: Path, *, field_name: str) -> None:
    """Reject symlinks and Windows junction/reparse points without following them."""

    try:
        path_status = path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ProjectPathError(f"{field_name} cannot be inspected") from exc
    attributes = getattr(path_status, "st_file_attributes", 0)
    if path.is_symlink() or attributes & _REPARSE_POINT_ATTRIBUTE:
        raise ProjectPathError(
            f"{field_name} must not contain a symlink or reparse point"
        )


__all__ = [
    "ProjectPathError",
    "atomic_write_project_bytes",
    "require_project_directory",
    "require_project_file",
    "resolve_project_path",
    "resolve_project_root",
]
