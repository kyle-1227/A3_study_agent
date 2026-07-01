"""Context provider for existing curriculum state."""

from __future__ import annotations

from src.context_engineering.itemizer import make_context_item
from src.context_engineering.providers.base import ProviderContext
from src.context_engineering.schema import ContextItem


class CurriculumContextProvider:
    """Objectize curriculum context already present in state."""

    name = "curriculum_provider"
    source_type = "curriculum"

    def collect(self, context: ProviderContext) -> list[ContextItem]:
        if context.max_items_per_provider <= 0:
            return []
        content = context.state.get("curriculum_context")
        if not isinstance(content, str) or not content.strip():
            return []
        learning_path = context.state.get("learning_path")
        path_id = ""
        if isinstance(learning_path, dict):
            path_id = str(learning_path.get("path_id") or learning_path.get("id") or "")
        return [
            make_context_item(
                source_type="curriculum",
                title="curriculum_context",
                content=content,
                priority=65,
                scope="project",
                lifetime="session",
                compressible=True,
                can_drop=True,
                disclosure_level="summary",
                metadata={"path_id": path_id},
                max_content_chars=context.max_content_chars_per_item,
            )
        ]
