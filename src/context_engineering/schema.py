"""Strict schemas for Context Engineering telemetry and context items."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)(cookie\s*[:=]\s*)[^;\n]+"),
    re.compile(r"(?i)(x-api-key\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]+"),
    re.compile(r"nvapi-[A-Za-z0-9_-]+"),
    re.compile(r"(?i)(postgres(?:ql)?://)[^\s]+"),
)

SENSITIVE_METADATA_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "cookie",
        "password",
        "secret",
        "db_uri",
        "database_url",
        "raw_prompt",
        "raw_messages",
        "raw_output",
        "schema",
        "prompt",
        "messages",
    }
)


def normalize_metadata_key(key: object) -> str:
    """Normalize a metadata key for exact sensitive-key matching."""
    return str(key or "").strip().lower().replace("-", "_")


def is_sensitive_metadata_key(key: object) -> bool:
    """Return whether a metadata key is explicitly sensitive."""
    return normalize_metadata_key(key) in SENSITIVE_METADATA_KEYS


def sanitize_error_message(message: object, *, max_chars: int = 300) -> str:
    """Redact obvious secrets from short diagnostic messages."""
    text = str(message or "").replace("\n", " ").strip()
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text[:max_chars]


class ContextConfigError(RuntimeError):
    """Context Engineering configuration error."""

    def __init__(self, reason: str, warning: str) -> None:
        self.reason = reason
        self.warning = warning
        super().__init__(f"{reason}: {warning}")


class ContextUsageError(RuntimeError):
    """Context usage calculation error."""

    def __init__(self, reason: str, warning: str) -> None:
        self.reason = reason
        self.warning = warning
        super().__init__(f"{reason}: {warning}")


class ContextProviderError(RuntimeError):
    """Context provider collection error."""

    def __init__(
        self,
        *,
        provider: str,
        source_type: str,
        stage: str,
        message: object,
        original_exception_type: str = "",
    ) -> None:
        self.provider = str(provider or "").strip() or "unknown"
        self.source_type = str(source_type or "").strip() or "unknown"
        self.stage = str(stage or "").strip() or "collect"
        self.original_exception_type = str(original_exception_type or "").strip()
        self.sanitized_message = sanitize_error_message(message)
        super().__init__(f"{self.provider}:{self.stage}: {self.sanitized_message}")


class TokenCount(BaseModel):
    """Estimated token count result."""

    model_config = ConfigDict(extra="forbid")

    value: int = Field(ge=0)
    estimated: bool
    method: str = Field(min_length=1)


class ContextBudget(BaseModel):
    """Budget for one model call."""

    model_config = ConfigDict(extra="forbid")

    node_name: str
    llm_node: str
    model: str = Field(min_length=1)
    max_context_tokens: int = Field(gt=0)
    reserved_output_tokens: int = Field(ge=0)
    max_input_tokens: int = Field(ge=0)
    warning_ratio: float = Field(ge=0, le=1)
    critical_ratio: float = Field(ge=0, le=1)
    compact_ratio: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _validate_relationships(self) -> "ContextBudget":
        if self.warning_ratio >= self.critical_ratio:
            raise ValueError("warning_ratio must be less than critical_ratio")
        if self.critical_ratio > self.compact_ratio:
            raise ValueError("critical_ratio must not exceed compact_ratio")
        if self.reserved_output_tokens >= self.max_context_tokens:
            raise ValueError(
                "reserved_output_tokens must be less than max_context_tokens"
            )
        expected_input = self.max_context_tokens - self.reserved_output_tokens
        if self.max_input_tokens != expected_input:
            raise ValueError(
                "max_input_tokens must equal max_context_tokens - reserved_output_tokens"
            )
        return self


ContextSourceType = Literal[
    "message",
    "memory",
    "evidence",
    "artifact",
    "profile",
    "trajectory",
    "rules",
    "curriculum",
    "unknown",
]

ContextScope = Literal["node", "turn", "session", "project", "global"]
ContextLifetime = Literal[
    "ephemeral",
    "turn",
    "session",
    "cross_session",
    "long_term",
]
ContextDisclosureLevel = Literal["index", "summary", "snippet", "full"]


class ContextItem(BaseModel):
    """Candidate context item produced by a ContextProvider."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    source_type: ContextSourceType
    title: str = Field(default="")
    content: str = Field(default="")
    token_estimate: int = Field(ge=0)
    estimated: bool
    tokenizer_mode: str = Field(min_length=1)
    priority: int = Field(ge=0, le=100)
    relevance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    recency_score: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    scope: ContextScope
    lifetime: ContextLifetime
    compressible: bool
    can_drop: bool
    disclosure_level: ContextDisclosureLevel
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_item(self) -> "ContextItem":
        if not self.content.strip() and not self.title.strip() and not self.metadata:
            raise ValueError(
                "content can be empty only when title or metadata identifies the item"
            )
        sensitive = [key for key in self.metadata if is_sensitive_metadata_key(key)]
        if sensitive:
            raise ValueError(
                "metadata contains sensitive keys: " + ", ".join(sorted(sensitive))
            )
        return self


class ContextUsageReport(BaseModel):
    """Pre-call context usage report."""

    model_config = ConfigDict(extra="forbid")

    node_name: str
    llm_node: str
    provider: str
    model: str
    input_estimated_tokens: int = Field(ge=0)
    reserved_output_tokens: int = Field(ge=0)
    used_tokens: int = Field(ge=0)
    max_context_tokens: int = Field(gt=0)
    available_tokens: int = Field(ge=0)
    used_ratio: float = Field(ge=0)
    warning_level: Literal["ok", "warning", "critical", "overflow"]
    estimated: bool
    tokenizer_mode: str = Field(min_length=1)
    message_count: int = Field(ge=0)
    schema_size_chars: int | None = Field(default=None, ge=0)
    breakdown: dict[str, int]

    @model_validator(mode="after")
    def _validate_totals(self) -> "ContextUsageReport":
        expected_used = self.input_estimated_tokens + self.reserved_output_tokens
        if self.used_tokens != expected_used:
            raise ValueError(
                "used_tokens must equal input_estimated_tokens + reserved_output_tokens"
            )
        expected_available = max(self.max_context_tokens - self.used_tokens, 0)
        if self.available_tokens != expected_available:
            raise ValueError("available_tokens must equal remaining context capacity")
        return self
