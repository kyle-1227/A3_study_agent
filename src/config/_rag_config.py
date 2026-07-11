"""Shared fail-fast primitives for production RAG YAML configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, TypeVar

import yaml  # type: ignore[import-untyped]
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
)


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("value must contain non-whitespace characters")
    if value != value.strip():
        raise ValueError("value must not contain leading or trailing whitespace")
    return value


def _yaml_path(value: object) -> Path:
    """Convert the one scalar representation YAML has for a filesystem path."""
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str):
        if not value or "\x00" in value:
            raise ValueError("path must be a non-empty string without NUL bytes")
        if value != value.strip():
            raise ValueError("path must not contain leading or trailing whitespace")
        path = Path(value)
    else:
        raise TypeError("path must be represented by a YAML string")
    return path


def _yaml_sequence(value: object) -> tuple[object, ...]:
    """Freeze a YAML sequence before strict item validation."""
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise TypeError("value must be a YAML sequence")


NonBlankStr = Annotated[str, Field(min_length=1), AfterValidator(_non_blank)]
NonEmptyStr = Annotated[str, Field(min_length=1)]
ConfigPath = Annotated[Path, BeforeValidator(_yaml_path)]
NonBlankStrTuple = Annotated[
    tuple[NonBlankStr, ...],
    BeforeValidator(_yaml_sequence),
]
NonEmptyStrTuple = Annotated[
    tuple[NonEmptyStr, ...],
    BeforeValidator(_yaml_sequence),
]
ConfigPathTuple = Annotated[
    tuple[ConfigPath, ...],
    BeforeValidator(_yaml_sequence),
]
PositiveIntTuple = Annotated[
    tuple[Annotated[int, Field(gt=0)], ...],
    BeforeValidator(_yaml_sequence),
]
NonNegativeIntTuple = Annotated[
    tuple[Annotated[int, Field(ge=0)], ...],
    BeforeValidator(_yaml_sequence),
]
PositiveFloatTuple = Annotated[
    tuple[Annotated[float, Field(gt=0)], ...],
    BeforeValidator(_yaml_sequence),
]


class StrictRagConfigModel(BaseModel):
    """Base contract shared by all production RAG configuration models."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class RagConfigError(RuntimeError):
    """Base class for typed RAG configuration loading failures."""

    def __init__(self, *, config_path: Path, reason: str) -> None:
        self.config_path = config_path
        self.reason = reason
        super().__init__(f"RAG configuration error at {config_path}: {reason}")


class RagConfigPathError(RagConfigError):
    """The explicitly supplied configuration path cannot be read."""


class RagConfigYamlError(RagConfigError):
    """The configuration is not valid YAML."""


class RagConfigYamlRootError(RagConfigError):
    """The YAML document root is not a mapping."""


class RagConfigValidationError(RagConfigError):
    """Pydantic rejected the strict configuration contract."""

    def __init__(
        self,
        *,
        config_path: Path,
        validation_errors: tuple[tuple[str, str], ...],
    ) -> None:
        self.validation_errors = validation_errors
        summary = "; ".join(
            f"{location}: {error_type}" for location, error_type in validation_errors
        )
        super().__init__(config_path=config_path, reason=summary)


ConfigModelT = TypeVar("ConfigModelT", bound=StrictRagConfigModel)


def load_strict_rag_yaml(
    config_path: Path,
    model_type: type[ConfigModelT],
) -> ConfigModelT:
    """Load one explicit YAML file and validate it without fallback behavior."""
    if not isinstance(config_path, Path):
        raise RagConfigPathError(
            config_path=Path("<invalid-path-type>"),
            reason="config_path must be a pathlib.Path instance",
        )
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RagConfigPathError(
            config_path=config_path,
            reason=f"{type(exc).__name__} while reading the file",
        ) from exc

    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RagConfigYamlError(
            config_path=config_path,
            reason="invalid YAML syntax",
        ) from exc
    if not isinstance(payload, dict):
        raise RagConfigYamlRootError(
            config_path=config_path,
            reason="YAML document root must be a mapping",
        )

    try:
        return model_type.model_validate(payload)
    except ValidationError as exc:
        details = tuple(
            (
                ".".join(str(part) for part in error["loc"]),
                str(error["type"]),
            )
            for error in exc.errors(include_input=False, include_url=False)
        )
        raise RagConfigValidationError(
            config_path=config_path,
            validation_errors=details,
        ) from exc


class RagConfigSecretError(RuntimeError):
    """A required secret environment variable is absent or blank.

    The exception deliberately contains only the environment-variable name.
    """

    def __init__(self, *, environment_name: str) -> None:
        self.environment_name = environment_name
        super().__init__(
            f"required RAG secret environment variable is missing: {environment_name}"
        )


def resolve_required_secret(environment_name: str) -> str:
    """Resolve one explicitly configured secret without logging its value."""

    if not environment_name or not environment_name.isidentifier():
        raise ValueError("environment_name must be a valid non-empty identifier")
    value = os.environ.get(environment_name)
    if value is None or not value.strip():
        raise RagConfigSecretError(environment_name=environment_name)
    return value
