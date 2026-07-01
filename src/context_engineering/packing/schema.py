"""Strict schemas for ContextPacker shadow mode."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.context_engineering.schema import ContextItem, sanitize_error_message

PackingStrategy = Literal["priority_budget"]
PackingReason = Literal[
    "required",
    "fits_budget",
    "over_budget",
    "source_disabled",
]


class ContextPackingError(RuntimeError):
    """Context packing failure with sanitized diagnostics."""

    def __init__(
        self,
        *,
        reason: str,
        warning: object,
        node_name: str,
        llm_node: str,
        selected_tokens: int | None = None,
        budget_tokens: int | None = None,
        original_exception_type: str = "",
    ) -> None:
        self.reason = str(reason or "").strip() or "context_packing_error"
        self.warning = sanitize_error_message(warning)
        self.node_name = str(node_name or "").strip()
        self.llm_node = str(llm_node or "").strip()
        self.selected_tokens = selected_tokens
        self.budget_tokens = budget_tokens
        self.original_exception_type = str(original_exception_type or "").strip()
        super().__init__(f"{self.reason}: {self.warning}")


class PackingDecision(BaseModel):
    """Safe per-item packing decision."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    title: str = Field(default="")
    selected: bool
    reason: PackingReason
    token_estimate: int = Field(ge=0)
    priority: int = Field(ge=0, le=100)
    can_drop: bool
    budget_before: int | None = Field(default=None, ge=0)
    budget_after: int | None = Field(default=None, ge=0)


class PackedContext(BaseModel):
    """Internal ContextPacker output for shadow mode."""

    model_config = ConfigDict(extra="forbid")

    node_name: str
    llm_node: str
    strategy: PackingStrategy
    selected_items: list[ContextItem]
    dropped_items: list[ContextItem]
    decisions: list[PackingDecision]
    rendered_context: str
    max_context_block_tokens: int = Field(gt=0)
    selected_tokens: int = Field(ge=0)
    dropped_tokens: int = Field(ge=0)
    required_tokens: int = Field(ge=0)
    optional_tokens: int = Field(ge=0)
    remaining_tokens: int = Field(ge=0)
    overflow: bool
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_totals(self) -> "PackedContext":
        expected_selected = sum(item.token_estimate for item in self.selected_items)
        expected_dropped = sum(item.token_estimate for item in self.dropped_items)
        if self.selected_tokens != expected_selected:
            raise ValueError("selected_tokens must equal selected item token sum")
        if self.dropped_tokens != expected_dropped:
            raise ValueError("dropped_tokens must equal dropped item token sum")
        expected_remaining = max(
            self.max_context_block_tokens - self.selected_tokens, 0
        )
        if self.remaining_tokens != expected_remaining:
            raise ValueError("remaining_tokens must equal remaining packing budget")
        if self.selected_tokens > self.max_context_block_tokens and not self.overflow:
            raise ValueError("overflow must be true when selected tokens exceed budget")
        return self
