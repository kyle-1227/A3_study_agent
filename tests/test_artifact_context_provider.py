"""Tests for artifact ContextItem provider workspace support."""

from __future__ import annotations

from src.context_engineering.providers.artifact_provider import ArtifactContextProvider
from src.context_engineering.providers.base import ProviderContext


def _context(state: dict) -> ProviderContext:
    return ProviderContext(
        node_name="node",
        llm_node="llm",
        user_query="query",
        current_user_message_index=None,
        state=state,
        messages=[],
        request_id="request-2",
        thread_id="thread-1",
        max_items_per_provider=10,
        max_content_chars_per_item=4000,
    )


def test_artifact_provider_reads_workspace_first_and_dedupes_sources():
    artifact = {
        "artifact_id": "artifact:v1:shared",
        "resource_type": "review_doc",
        "title": "Prior Review",
        "summary": "Compact summary only.",
        "subject": "machine_learning",
        "normalized_subject": "machine_learning",
        "thread_id": "thread-1",
        "request_id": "request-1",
        "purpose": "artifact_reference",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    state = {
        "thread_id": "thread-1",
        "subject": "Machine Learning",
        "task_workspace": {
            "schema_version": 1,
            "workspace_id": "workspace:v1:one",
            "artifacts_by_id": {"artifact:v1:shared": artifact},
            "artifacts": [artifact],
        },
        "resource_artifacts_by_type": {"review_doc": artifact},
        "review_doc_artifact": artifact,
    }

    items = ArtifactContextProvider().collect(_context(state))

    assert len(items) == 1
    item = items[0]
    assert item.title == "Prior Review"
    assert item.content == "Compact summary only."
    assert item.metadata["artifact_source"].startswith("task_workspace.artifacts_by_id")
    assert item.metadata["artifact_id"] == "artifact:v1:shared"
    assert item.metadata["purpose"] == "artifact_reference"
    assert item.metadata["thread_id"] == "thread-1"
    assert item.relevance_score and item.relevance_score > 0.8


def test_artifact_provider_skips_corrupt_workspace_entries():
    state = {
        "thread_id": "thread-1",
        "task_workspace": {
            "schema_version": 1,
            "artifacts_by_id": {"bad": "not-a-dict"},
            "artifacts": ["also-bad"],
        },
        "resource_artifacts_by_type": {
            "quiz": {
                "artifact_id": "artifact:v1:quiz",
                "title": "Quiz",
                "summary": "Quiz summary",
                "resource_type": "quiz",
                "thread_id": "thread-1",
                "purpose": "artifact_reference",
            }
        },
    }

    items = ArtifactContextProvider().collect(_context(state))

    assert [item.metadata["artifact_id"] for item in items] == ["artifact:v1:quiz"]
