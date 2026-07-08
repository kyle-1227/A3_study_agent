"""Context provider for already-produced artifact summaries."""

from __future__ import annotations

from typing import Any

from src.context_engineering.itemizer import make_context_item
from src.context_engineering.providers.base import ProviderContext
from src.context_engineering.schema import ContextItem
from src.context_engineering.workspace import (
    sanitize_workspace_text,
    workspace_scope_from_state,
)
from src.rag.course_catalog import normalize_subject


class ArtifactContextProvider:
    """Objectize stable artifact metadata already present in state."""

    name = "artifact_provider"
    source_type = "artifact"

    def collect(self, context: ProviderContext) -> list[ContextItem]:
        if context.max_items_per_provider <= 0:
            return []
        items: list[ContextItem] = []
        seen_ids: set[str] = set()
        for key, artifact in _existing_artifacts(context.state):
            if len(items) >= context.max_items_per_provider:
                break
            artifact_id = _artifact_id(key, artifact)
            if artifact_id in seen_ids:
                continue
            seen_ids.add(artifact_id)
            title = str(artifact.get("title") or key)
            summary = str(
                artifact.get("summary")
                or artifact.get("message_preview")
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
                    relevance_score=_artifact_relevance(artifact, context.state),
                    metadata={
                        "artifact_source": key,
                        "artifact_id": artifact_id,
                        "filename": artifact.get("filename", ""),
                        "resource_type": artifact.get("resource_type", ""),
                        "task_type": artifact.get("resource_type", ""),
                        "subject": artifact.get("subject", ""),
                        "normalized_subject": artifact.get("normalized_subject", ""),
                        "thread_id": artifact.get("thread_id", ""),
                        "request_id": artifact.get("request_id", ""),
                        "purpose": artifact.get("purpose") or "artifact_reference",
                        "created_at": artifact.get("created_at", ""),
                        "workspace_id": artifact.get("workspace_id", ""),
                        "artifact_refs": artifact.get("artifact_refs") or {},
                    },
                    max_content_chars=context.max_content_chars_per_item,
                )
            )
        return items


def _existing_artifacts(state: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    artifacts: list[tuple[str, dict[str, Any]]] = []
    workspace = state.get("task_workspace")
    if isinstance(workspace, dict):
        workspace_id = sanitize_workspace_text(
            workspace.get("workspace_id"),
            max_chars=160,
            fallback="",
        )
        artifacts_by_id = workspace.get("artifacts_by_id")
        if isinstance(artifacts_by_id, dict):
            for artifact_id, artifact in artifacts_by_id.items():
                if isinstance(artifact, dict) and artifact:
                    artifacts.append(
                        (
                            f"task_workspace.artifacts_by_id.{artifact_id}",
                            {**artifact, "workspace_id": workspace_id},
                        )
                    )
        ordered = workspace.get("artifacts")
        if isinstance(ordered, list):
            for index, artifact in enumerate(ordered):
                if isinstance(artifact, dict) and artifact:
                    artifacts.append(
                        (
                            f"task_workspace.artifacts.{index}",
                            {**artifact, "workspace_id": workspace_id},
                        )
                    )
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


def _artifact_id(key: str, artifact: dict[str, Any]) -> str:
    return sanitize_workspace_text(
        artifact.get("artifact_id") or key,
        max_chars=220,
        fallback=key,
    )


def _artifact_relevance(artifact: dict[str, Any], state: dict[str, Any]) -> float:
    scope = workspace_scope_from_state(state)
    state_thread = scope.get("thread_id", "")
    artifact_thread = sanitize_workspace_text(
        artifact.get("thread_id"),
        max_chars=120,
        fallback="",
    )
    state_subject = scope.get("normalized_subject", "")
    artifact_subject = sanitize_workspace_text(
        artifact.get("normalized_subject"),
        max_chars=120,
        fallback="",
    )
    if not artifact_subject:
        raw_subject = sanitize_workspace_text(
            artifact.get("subject"),
            max_chars=120,
            fallback="",
        )
        artifact_subject = normalize_subject(raw_subject) if raw_subject else ""

    if state_thread and artifact_thread and state_thread != artifact_thread:
        return 0.05
    if state_thread and artifact_thread == state_thread:
        if state_subject and artifact_subject and state_subject == artifact_subject:
            return 0.82
        if state_subject and artifact_subject and state_subject != artifact_subject:
            return 0.25
        return 0.55
    if state_subject and artifact_subject and state_subject == artifact_subject:
        return 0.45
    return 0.35
