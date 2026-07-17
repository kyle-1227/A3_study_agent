"""Explicit development reload settings for the ASGI server entry points."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import get_setting


class ServerReloadConfig(BaseModel):
    """Validated reload scope owned by ``config/settings.yaml``."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    enabled: bool
    include_dirs: list[str] = Field(min_length=1, max_length=8)
    exclude_dirs: list[str] = Field(min_length=1, max_length=32)

    @field_validator("include_dirs", "exclude_dirs")
    @classmethod
    def validate_relative_directories(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            path = _normalize_relative_directory(value)
            if path not in normalized:
                normalized.append(path)
        if not normalized:
            raise ValueError("server reload directory list must not be empty")
        return normalized


class ServerRuntimeStateConfig(BaseModel):
    """Validated mutable-state directory owned by ``config/settings.yaml``."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    directory: str = Field(min_length=1, max_length=160)

    @field_validator("directory")
    @classmethod
    def validate_directory(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("runtime state directory must be normalized and stripped")
        normalized = _normalize_relative_directory(value)
        if normalized == ".":
            raise ValueError("runtime state directory must not be the workspace root")
        return normalized


@dataclass(frozen=True, slots=True)
class ServerRuntimeStatePaths:
    directory: Path
    profile_db_path: Path
    memory_db_path: Path


def load_server_reload_config() -> ServerReloadConfig:
    """Load the required server reload contract without inventing defaults."""

    raw = get_setting("server.reload", None)
    if not isinstance(raw, dict):
        raise RuntimeError("server.reload configuration is required")
    return ServerReloadConfig.model_validate(raw)


def load_server_runtime_state_config() -> ServerRuntimeStateConfig:
    """Load the required mutable-state contract without inventing defaults."""

    raw = get_setting("server.runtime_state", None)
    if not isinstance(raw, dict):
        raise RuntimeError("server.runtime_state configuration is required")
    return ServerRuntimeStateConfig.model_validate(raw)


def resolve_uvicorn_reload_options(
    config: ServerReloadConfig,
    *,
    workspace_root: Path,
    enabled_override: bool | None = None,
) -> dict[str, bool | list[str]]:
    """Build Uvicorn reload arguments from the explicit configuration."""

    enabled = config.enabled if enabled_override is None else enabled_override
    if not enabled:
        return {"reload": False}
    root = workspace_root.resolve()
    include_dirs = _resolve_directories(
        config.include_dirs,
        workspace_root=root,
        require_existing=True,
    )
    exclude_dirs = _resolve_directories(
        config.exclude_dirs,
        workspace_root=root,
        require_existing=False,
    )
    return {
        "reload": True,
        "reload_dirs": include_dirs,
        "reload_excludes": exclude_dirs,
    }


def resolve_server_runtime_state_paths(
    config: ServerRuntimeStateConfig,
    *,
    workspace_root: Path,
) -> ServerRuntimeStatePaths:
    """Resolve profile/memory databases inside a dedicated mutable directory."""

    if not isinstance(config, ServerRuntimeStateConfig):
        raise TypeError("config must be ServerRuntimeStateConfig")
    if not isinstance(workspace_root, Path):
        raise TypeError("workspace_root must be pathlib.Path")
    root = workspace_root.resolve()
    directory = (root / config.directory).resolve()
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("runtime state directory escaped the workspace") from exc
    immutable_data_root = (root / "data").resolve()
    if directory == immutable_data_root or directory.is_relative_to(
        immutable_data_root
    ):
        raise RuntimeError(
            "runtime state directory must remain outside immutable course data"
        )
    return ServerRuntimeStatePaths(
        directory=directory,
        profile_db_path=directory / "profile.db",
        memory_db_path=directory / "memory.db",
    )


def _normalize_relative_directory(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("server reload directories must be strings")
    candidate = value.strip()
    if not candidate:
        raise ValueError("server reload directory must not be blank")
    if candidate.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", candidate):
        raise ValueError("server reload directory must be workspace-relative")
    parts = [part for part in re.split(r"[\\/]+", candidate) if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise ValueError("server reload directory must not escape the workspace")
    return "." if not parts else "/".join(parts)


def _resolve_directories(
    values: list[str],
    *,
    workspace_root: Path,
    require_existing: bool,
) -> list[str]:
    resolved: list[str] = []
    for value in values:
        candidate = (workspace_root / value).resolve()
        try:
            candidate.relative_to(workspace_root)
        except ValueError as exc:
            raise RuntimeError("server reload directory escaped the workspace") from exc
        if require_existing and not candidate.is_dir():
            raise RuntimeError("server reload include directory does not exist")
        normalized = str(candidate)
        if normalized not in resolved:
            resolved.append(normalized)
    return resolved


__all__ = [
    "ServerReloadConfig",
    "ServerRuntimeStateConfig",
    "ServerRuntimeStatePaths",
    "load_server_reload_config",
    "load_server_runtime_state_config",
    "resolve_server_runtime_state_paths",
    "resolve_uvicorn_reload_options",
]
