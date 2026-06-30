"""Context provider for already-produced artifact summaries."""

from __future__ import annotations

from src.context_engineering.itemizer import make_context_item
from src.context_engineering.providers.base import ProviderContext
from src.context_engineering.schema import ContextItem


class ArtifactContextProvider:
    """Objectize stable artifact metadata already present in state."""

    name = "artifact_provider"
    source_type = "artifact"

    def collect(self, context: ProviderContext) -> list[ContextItem]:
        if context.max_items_per_provider <= 0:
            return []
        items: list[ContextItem] = []
        for key in (
            "mindmap_artifact",
            "exercise_artifact",
            "review_doc_artifact",
            "study_plan_artifact",
            "resource_bundle_artifact",
        ):
            artifact = context.state.get(key)
            if not isinstance(artifact, dict) or not artifact:
                continue
            title = str(artifact.get("title") or key)
            summary = str(
                artifact.get("summary")
                or artifact.get("status")
                or artifact.get("resource_generation_status")
                or title
            )
            items.append(
                make_context_item(
                    source_type="artifact",
                    title=title,
                    content=summary,
                    priority=55,
                    scope="session",
                    lifetime="session",
                    compressible=True,
                    can_drop=True,
                    disclosure_level="summary",
                    metadata={
                        "artifact_source": key,
                        "artifact_id": artifact.get("artifact_id", ""),
                        "filename": artifact.get("filename", ""),
                    },
                    max_content_chars=context.max_content_chars_per_item,
                )
            )
            if len(items) >= context.max_items_per_provider:
                break
        return items
