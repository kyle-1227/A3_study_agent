"""Context provider for already-produced artifact summaries."""

from __future__ import annotations

from typing import Any

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
        for key, artifact in _existing_artifacts(context.state):
            if len(items) >= context.max_items_per_provider:
                break
            title = str(artifact.get("title") or key)
            summary = str(
                artifact.get("summary")
                or artifact.get("status")
                or artifact.get("resource_generation_status")
                or artifact.get("markdown_url")
                or artifact.get("filename")
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
                        "resource_type": artifact.get("resource_type", ""),
                    },
                    max_content_chars=context.max_content_chars_per_item,
                )
            )
        return items


def _existing_artifacts(state: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    artifacts: list[tuple[str, dict[str, Any]]] = []
    by_type = state.get("resource_artifacts_by_type")
    if isinstance(by_type, dict):
        for resource_type, artifact in by_type.items():
            if isinstance(artifact, dict) and artifact:
                artifacts.append(
                    (f"resource_artifacts_by_type.{resource_type}", artifact)
                )
    generated = state.get("last_generated_artifacts")
    if isinstance(generated, list):
        for index, artifact in enumerate(generated):
            if isinstance(artifact, dict) and artifact:
                artifacts.append((f"last_generated_artifacts.{index}", artifact))
    for key in (
        "mindmap_artifact",
        "exercise_artifact",
        "review_doc_artifact",
        "review_doc_artifacts",
        "code_practice_artifact",
        "video_script_artifact",
        "video_animation_artifact",
        "study_plan_artifact",
        "study_plan_document_artifact",
        "resource_bundle_artifact",
    ):
        artifact = state.get(key)
        if isinstance(artifact, dict) and artifact:
            artifacts.append((key, artifact))
            continue
        if isinstance(artifact, list):
            for index, item in enumerate(artifact):
                if isinstance(item, dict) and item:
                    artifacts.append((f"{key}.{index}", item))
    return artifacts
