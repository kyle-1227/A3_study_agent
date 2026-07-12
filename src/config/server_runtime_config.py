"""Explicit development reload settings for the ASGI server entry points."""

from __future__ import annotations

import re
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


def load_server_reload_config() -> ServerReloadConfig:
    """Load the required server reload contract without inventing defaults."""

    raw = get_setting("server.reload", None)
    if not isinstance(raw, dict):
        raise RuntimeError("server.reload configuration is required")
    return ServerReloadConfig.model_validate(raw)


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
    "load_server_reload_config",
    "resolve_uvicorn_reload_options",
]
