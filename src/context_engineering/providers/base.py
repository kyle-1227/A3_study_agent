"""Base contracts for ContextProvider implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from src.context_engineering.schema import ContextItem, ContextSourceType


@dataclass(frozen=True)
class ProviderContext:
    """Input available to context providers during shadow collection."""

    node_name: str
    llm_node: str | None
    user_query: str | None
    current_user_message_index: int | None
    state: dict[str, Any]
    messages: list[Any]
    request_id: str | None
    thread_id: str | None
    max_items_per_provider: int
    max_content_chars_per_item: int


class ContextProvider(Protocol):
    """Protocol implemented by all context item providers."""

    name: str
    source_type: ContextSourceType

    def collect(self, context: ProviderContext) -> list[ContextItem]:
        """Collect candidate context items from existing state/messages only."""
