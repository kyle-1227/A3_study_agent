from __future__ import annotations

import pytest

from src.graph.resource_final import (
    STRUCTURED_RESOURCE_TYPES,
    completed_without_resource_payload,
    normalize_resource_final_payload,
)


def test_resource_final_builds_stable_id_hash_and_normalized_resource():
    legacy_payload = {
        "type": "resource_final",
        "resource_type": "review_doc",
        "thread_id": "thread-1",
        "request_id": "request-1",
        "answer": "review doc ready",
        "review_doc_artifacts": [
            {
                "title": "ML Review",
                "markdown_url": "/artifacts/review-docs/r1/ml.md",
                "docx_url": "https://example.com/ml.docx?token=secret",
                "filename": "C:/Users/kyle/secret.md",
            }
        ],
    }

    first = normalize_resource_final_payload(legacy_payload)
    second = normalize_resource_final_payload(dict(legacy_payload))

    assert first is not None
    assert second is not None
    assert first["schema_version"] == 1
    assert first["resource_id"].startswith("resource:v1:")
    assert first["payload_hash"].startswith("payload:v1:")
    assert first["resource_id"] == second["resource_id"]
    assert first["payload_hash"] == second["payload_hash"]
    assert first["resource"]["kind"] == "review_doc"
    assert first["resource"]["payload"]["review_doc_artifacts"][0]["markdown_url"] == (
        "/artifacts/review-docs/r1/ml.md"
    )
    artifact = first["resource"]["payload"]["review_doc_artifacts"][0]
    assert "docx_url" not in artifact
    assert "filename" not in artifact
    assert "secret" not in str(first)
    assert "C:/Users" not in str(first)


def test_resource_final_id_changes_for_distinct_requests_same_type():
    base = {
        "type": "resource_final",
        "resource_type": "mindmap",
        "thread_id": "thread-1",
        "answer": "done",
        "mindmap": {"title": "ML Map", "tree": {"title": "ML"}},
    }

    first = normalize_resource_final_payload({**base, "request_id": "request-1"})
    second = normalize_resource_final_payload({**base, "request_id": "request-2"})

    assert first is not None
    assert second is not None
    assert first["payload_hash"] == second["payload_hash"]
    assert first["resource_id"] != second["resource_id"]


def test_completed_without_resource_only_for_resource_runs():
    assert completed_without_resource_payload({"messages": ["plain answer"]}) is None

    diagnostic = completed_without_resource_payload(
        {
            "thread_id": "thread-1",
            "request_id": "request-1",
            "requested_resource_type": "mindmap",
            "resource_generation_status": "failed",
        }
    )

    assert diagnostic is not None
    assert diagnostic["type"] == "resource_final_diagnostic"
    assert diagnostic["status"] == "completed_without_resource"
    assert diagnostic["requested_resource_types"] == ["mindmap"]


# ---------------------------------------------------------------------------
# Structured resource type empty-payload rejection
# ---------------------------------------------------------------------------


def test_normalize_returns_none_for_structured_type_answer_only():
    """resource_type=study_plan with only answer text must return None."""
    payload = {
        "type": "resource_final",
        "resource_type": "study_plan",
        "answer": "Here is a detailed analysis of your study needs.",
    }
    result = normalize_resource_final_payload(payload)
    assert result is None


def test_normalize_returns_valid_for_evidence_summary_answer_only():
    """resource_type=evidence_summary with answer text must return a valid payload."""
    payload = {
        "type": "resource_final",
        "resource_type": "evidence_summary",
        "answer": "Evidence is insufficient for resource generation.",
        "controlled_stop": True,
        "controlled_stop_reason": "evidence_insufficient",
    }
    result = normalize_resource_final_payload(payload)
    assert result is not None
    assert result["resource_type"] == "evidence_summary"
    assert result["resource"]["kind"] == "evidence_summary"
    assert result["resource"]["payload"]  # non-empty


@pytest.mark.parametrize(
    "resource_type",
    sorted(STRUCTURED_RESOURCE_TYPES),
)
def test_all_structured_types_rejected_when_payload_empty(resource_type: str):
    """Every structured type with only answer text must return None."""
    payload = {
        "type": "resource_final",
        "resource_type": resource_type,
        "answer": "generic answer text",
    }
    result = normalize_resource_final_payload(payload)
    assert result is None, f"{resource_type} should be rejected"


def test_structured_resource_types_includes_bundle():
    assert "bundle" in STRUCTURED_RESOURCE_TYPES


def test_structured_resource_types_consistent_with_artifact_keys():
    """Every key in STRUCTURED_RESOURCE_ARTIFACT_KEYS must be in STRUCTURED_RESOURCE_TYPES."""
    from app import STRUCTURED_RESOURCE_ARTIFACT_KEYS

    for key in STRUCTURED_RESOURCE_ARTIFACT_KEYS:
        assert key in STRUCTURED_RESOURCE_TYPES, (
            f"{key} in STRUCTURED_RESOURCE_ARTIFACT_KEYS but missing from "
            f"STRUCTURED_RESOURCE_TYPES"
        )
