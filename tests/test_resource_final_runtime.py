from __future__ import annotations

import json

import pytest

from src.assessment.identity import stable_exercise_question_id
from src.graph.resource_final_runtime import (
    ResourceFinalRuntimeError,
    build_resource_final_v3_from_bundle,
)
from src.graph.resource_final_v3 import (
    ResourceFinalV3ResourceValidation,
    ResourceFinalV3TerminalStatus,
    build_resource_final_v3_resource,
    validate_resource_final_v3,
)


THREAD_ID = "thread-1"
REQUEST_ID = "request-1"


def _validation(
    resource_type: str,
    *,
    status: str = "success",
    downloadable_count: int = 1,
) -> dict:
    return {
        "schema_version": "resource_validation_v1",
        "resource_type": resource_type,
        "valid": True,
        "terminal_status": status,
        "renderable_count": 1,
        "downloadable_count": downloadable_count,
        "verified_local_count": downloadable_count,
        "remote_unverified_count": 0,
        "failure_reason": "",
        "warnings": [],
    }


def _mindmap_result(*, status: str = "success") -> dict:
    return {
        "resource_type": "mindmap",
        "status": status,
        "title": "Machine Learning Map",
        "artifact": {
            "title": "Machine Learning Map",
            "tree": {
                "title": "Machine Learning",
                "children": [{"title": "Models"}],
            },
            "xmind_url": "/artifacts/mindmaps/map-1/map.xmind",
        },
        "artifacts": [],
        "state_updates": {
            "mindmap_tree": {
                "title": "Machine Learning",
                "children": [{"title": "Models"}],
            }
        },
        "message_content": "A validated machine learning concept map is ready.",
        "validation": _validation("mindmap", status=status),
    }


def _failed_result(resource_type: str) -> dict:
    return {
        "resource_type": resource_type,
        "status": "failed",
        "title": resource_type,
        "artifact": {},
        "artifacts": [],
        "state_updates": {},
        "message_content": "",
        "error_code": f"{resource_type}.generation_failed",
        "error_type": "ProviderError",
        "error_message_sanitized": "The provider request failed.",
        "validation": None,
    }


def _build(
    *,
    terminal_status: ResourceFinalV3TerminalStatus,
    requested: list[str],
    results: list[dict],
):
    return build_resource_final_v3_from_bundle(
        thread_id=THREAD_ID,
        request_id=REQUEST_ID,
        requested_resource_types=requested,
        terminal_status=terminal_status,
        branch_results=results,
        blocked_resources=[],
        recommendations=[],
        summary="Resource generation reached an authoritative terminal state.",
    )


def test_runtime_builds_stable_success_and_validates_persisted_json_shape():
    first = _build(
        terminal_status="success",
        requested=["mindmap"],
        results=[_mindmap_result()],
    )
    second = _build(
        terminal_status="success",
        requested=["mindmap"],
        results=[_mindmap_result()],
    )

    assert first.payload_hash == second.payload_hash
    assert first.resource_final_id == second.resource_final_id
    assert first.validation.success_count == 1
    assert first.resources[0].kind == "mindmap"
    restored = validate_resource_final_v3(first.model_dump(mode="json"))
    assert restored == first


def test_runtime_preserves_real_partial_success_and_typed_failure():
    result = _build(
        terminal_status="partial_success",
        requested=["mindmap", "quiz"],
        results=[_mindmap_result(), _failed_result("quiz")],
    )

    assert result.terminal_status == "partial_success"
    assert [resource.kind for resource in result.resources] == ["mindmap"]
    assert result.validation.success_count == 1
    assert result.validation.failed_count == 1
    assert result.errors[0].error_code == "quiz.generation_failed"


def test_runtime_builds_failed_terminal_without_fake_resource():
    result = _build(
        terminal_status="failed",
        requested=["quiz"],
        results=[_failed_result("quiz")],
    )

    assert result.terminal_status == "failed"
    assert result.resources == ()
    assert result.errors[0].resource_type == "quiz"


def test_runtime_builds_controlled_stop_from_explicit_blocked_evidence():
    result = build_resource_final_v3_from_bundle(
        thread_id=THREAD_ID,
        request_id=REQUEST_ID,
        requested_resource_types=["study_plan"],
        terminal_status="controlled_stop",
        branch_results=[],
        blocked_resources=[
            {
                "resource_type": "study_plan",
                "status": "blocked_insufficient_evidence",
                "reason_code": "required_evidence_incomplete",
                "blocked_requirement_ids": ["requirement-1"],
            }
        ],
        recommendations=[],
        summary="Study plan generation stopped because required evidence is missing.",
    )

    assert result.terminal_status == "controlled_stop"
    assert result.resources == ()
    assert result.blocked_resources[0].resource_type == "study_plan"


def test_runtime_requires_exact_requested_result_partition():
    with pytest.raises(ResourceFinalRuntimeError, match="exactly match"):
        _build(
            terminal_status="success",
            requested=["mindmap", "quiz"],
            results=[_mindmap_result()],
        )


def test_runtime_rejects_success_without_strict_validation():
    result = _mindmap_result()
    result["validation"] = None
    with pytest.raises(ResourceFinalRuntimeError, match="requires validation"):
        _build(
            terminal_status="success",
            requested=["mindmap"],
            results=[result],
        )


def test_runtime_drops_unsafe_references_before_public_hashing():
    result = _mindmap_result()
    result["artifact"]["source_url"] = (
        "https://provider.invalid/resource?token=secret-value"
    )
    result["artifact"]["local_path"] = "C:\\private\\resource.json"

    final = _build(
        terminal_status="success",
        requested=["mindmap"],
        results=[result],
    )
    payload = final.resources[0].payload["mindmap"]

    assert isinstance(payload, dict)
    assert "source_url" not in payload
    assert "local_path" not in payload
    assert "secret-value" not in final.model_dump_json()


def test_runtime_reuses_bound_public_quiz_resource_without_answers():
    question = "What does a Python list store?"
    tags = ("python", "collections")
    card = {
        "schema_version": "exercise_card_v1",
        "question_id": stable_exercise_question_id(
            level="basic",
            question_type="free_text",
            question=question,
            choices=(),
            tags=tags,
        ),
        "question_type": "free_text",
        "level": "basic",
        "question": question,
        "choices": [],
        "tags": list(tags),
    }
    validation = ResourceFinalV3ResourceValidation.model_validate_json(
        json.dumps(_validation("quiz")),
        strict=True,
    )
    quiz = build_resource_final_v3_resource(
        thread_id=THREAD_ID,
        request_id=REQUEST_ID,
        kind="quiz",
        status="success",
        title="Python Quiz",
        summary="One validated public exercise card.",
        payload={
            "exercise_artifact": {
                "schema_version": "exercise_public_artifact_v1",
                "title": "Python Quiz",
                "items": [card],
            },
            "exercise_items": [card],
        },
        artifact_refs={},
        validation=validation,
    )
    result = {
        "resource_type": "quiz",
        "status": "success",
        "title": "Python Quiz",
        "artifact": {
            "schema_version": "exercise_public_artifact_v1",
            "title": "Python Quiz",
            "items": [card],
        },
        "artifacts": [],
        "state_updates": {"exercise_resource_v3": quiz.model_dump(mode="json")},
        "message_content": "Python quiz ready.",
        "validation": _validation("quiz"),
    }

    final = _build(
        terminal_status="success",
        requested=["quiz"],
        results=[result],
    )

    assert final.resources == (quiz,)
    assert "answer" not in final.model_dump_json()
