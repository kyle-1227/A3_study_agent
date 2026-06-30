"""Context provider for already-recorded trajectory summaries."""

from __future__ import annotations

from typing import Any

from src.context_engineering.itemizer import make_context_item
from src.context_engineering.providers.base import ProviderContext
from src.context_engineering.schema import ContextItem, ContextProviderError


class TrajectoryContextProvider:
    """Objectize existing trajectory/step summaries from state."""

    name = "trajectory_provider"
    source_type = "trajectory"

    def collect(self, context: ProviderContext) -> list[ContextItem]:
        if context.max_items_per_provider <= 0:
            return []
        value = context.state.get("trajectory") or context.state.get("step_summaries")
        if not value:
            return []
        if not isinstance(value, list):
            raise ContextProviderError(
                provider=self.name,
                source_type=self.source_type,
                stage="decode_state",
                message="trajectory must be a list when present",
                original_exception_type="TypeError",
            )
        items: list[ContextItem] = []
        for index, entry in enumerate(value):
            if len(items) >= context.max_items_per_provider:
                break
            if isinstance(entry, str):
                title = f"trajectory_step_{index}"
                content = entry
                metadata: dict[str, Any] = {"step_index": index}
            elif isinstance(entry, dict):
                title = str(
                    entry.get("title")
                    or entry.get("node")
                    or f"trajectory_step_{index}"
                )
                content = str(entry.get("summary") or entry.get("content") or "")
                metadata = {
                    "step_index": index,
                    "node": entry.get("node", ""),
                    "status": entry.get("status", ""),
                }
            else:
                raise ContextProviderError(
                    provider=self.name,
                    source_type=self.source_type,
                    stage="decode_state",
                    message="trajectory item must be str or dict",
                    original_exception_type="TypeError",
                )
            if not content:
                continue
            items.append(
                make_context_item(
                    source_type="trajectory",
                    title=title,
                    content=content,
                    priority=50,
                    scope="session",
                    lifetime="session",
                    compressible=True,
                    can_drop=True,
                    disclosure_level="summary",
                    metadata=metadata,
                    max_content_chars=context.max_content_chars_per_item,
                )
            )
        return items
