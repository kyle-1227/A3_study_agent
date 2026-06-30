"""Strict schemas for Context Engineering telemetry."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
