from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.graph.resource_final_v3 import (
    ResourceFinalV3,
    ResourceFinalV3BlockedResource,
    ResourceFinalV3Error,
    ResourceFinalV3ResourceValidation,
    ResourceFinalV3Validation,
    build_resource_final_v3,
    build_resource_final_v3_resource,
)


THREAD_ID = "thread-1"
REQUEST_ID = "request-1"


def _resource_validation(
    *, status: str = "success"
) -> ResourceFinalV3ResourceValidation:
    return ResourceFinalV3ResourceValidation.model_validate(
        {
            "schema_version": "resource_validation_v1",
            "resource_type": "mindmap",
            "valid": True,
            "terminal_status": status,
            "renderable_count": 1,
            "downloadable_count": 1,
            "verified_local_count": 1,
            "remote_unverified_count": 0,
            "failure_reason": "",
            "warnings": (),
        },
        strict=True,
    )


def _mindmap_resource(*, status: str = "success"):
    return build_resource_final_v3_resource(
        thread_id=THREAD_ID,
        request_id=REQUEST_ID,
        kind="mindmap",
        status=status,
        title="Machine Learning Map",
        summary="A validated machine learning concept map.",
        payload={
            "mindmap": {
                "title": "Machine Learning",
                "tree": {"title": "ML", "children": [{"title": "Models"}]},
            }
        },
        artifact_refs={"xmind_url": "/artifacts/mindmaps/map-1/map.xmind"},
        validation=_resource_validation(status=status),
    )


def _validation(
    *,
    resources: int,
    successes: int,
    partial_successes: int = 0,
    failures: int = 0,
    blocked: int = 0,
) -> ResourceFinalV3Validation:
    return ResourceFinalV3Validation(
        schema_version="resource_final_validation_v3",
        resource_count=resources,
        success_count=successes,
        partial_success_count=partial_successes,
        failed_count=failures,
        blocked_count=blocked,
        renderable_count=resources,
        downloadable_count=resources,
    )


def test_resource_final_v3_builder_is_strict_and_stable():
    resource = _mindmap_resource()
    kwargs = {
        "thread_id": THREAD_ID,
        "request_id": REQUEST_ID,
        "terminal_status": "success",
        "resources": [resource],
        "recommendations": [],
        "blocked_resources": [],
        "errors": [],
        "validation": _validation(resources=1, successes=1),
        "summary": "One validated resource was generated.",
    }

    first = build_resource_final_v3(**kwargs)
    second = build_resource_final_v3(**kwargs)

    assert first.schema_version == "resource_final_v3"
    assert first.terminal_status == "success"
    assert first.payload_hash == second.payload_hash
    assert first.resource_final_id == second.resource_final_id
    assert first.payload_hash.startswith("payload:v3:")
    assert first.resource_final_id.startswith("resource-final:v3:")
    assert resource.payload_hash.startswith("payload:v3:")
    assert resource.resource_id.startswith("resource:v3:")


def test_resource_final_v3_forbids_extra_fields():
    payload = build_resource_final_v3(
        thread_id=THREAD_ID,
        request_id=REQUEST_ID,
        terminal_status="success",
        resources=[_mindmap_resource()],
        recommendations=[],
        blocked_resources=[],
        errors=[],
        validation=_validation(resources=1, successes=1),
        summary="One validated resource was generated.",
    ).model_dump()
    payload["legacy_review_doc_artifacts"] = []

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ResourceFinalV3.model_validate(payload)


def test_resource_final_v3_rejects_unknown_terminal_status():
    payload = build_resource_final_v3(
        thread_id=THREAD_ID,
        request_id=REQUEST_ID,
        terminal_status="success",
        resources=[_mindmap_resource()],
        recommendations=[],
        blocked_resources=[],
        errors=[],
        validation=_validation(resources=1, successes=1),
        summary="One validated resource was generated.",
    ).model_dump()
    payload["terminal_status"] = "unknown"

    with pytest.raises(ValidationError, match="Input should be"):
        ResourceFinalV3.model_validate(payload)


def test_resource_final_v3_rejects_tampered_hash():
    payload = build_resource_final_v3(
        thread_id=THREAD_ID,
        request_id=REQUEST_ID,
        terminal_status="success",
        resources=[_mindmap_resource()],
        recommendations=[],
        blocked_resources=[],
        errors=[],
        validation=_validation(resources=1, successes=1),
        summary="One validated resource was generated.",
    ).model_dump()
    payload["summary"] = "Tampered after hashing."

    with pytest.raises(ValidationError, match="payload_hash does not match"):
        ResourceFinalV3.model_validate(payload)


def test_resource_final_v3_does_not_promote_failure_to_success():
    error = ResourceFinalV3Error(
        resource_type="mindmap",
        error_code="mindmap.validation_failed",
        error_type="ResourceValidationError",
        message_sanitized="The resource did not pass business validation.",
    )

    result = build_resource_final_v3(
        thread_id=THREAD_ID,
        request_id=REQUEST_ID,
        terminal_status="failed",
        resources=[],
        recommendations=[],
        blocked_resources=[],
        errors=[error],
        validation=_validation(resources=0, successes=0, failures=1),
        summary="Resource generation failed validation.",
    )

    assert result.terminal_status == "failed"
    assert result.resources == ()
    assert result.errors == (error,)


def test_resource_final_v3_requires_real_resource_for_partial_success():
    blocked = ResourceFinalV3BlockedResource(
        resource_type="quiz",
        status="blocked_insufficient_evidence",
        reason_code="evidence.insufficient",
        blocked_requirement_ids=("requirement-1",),
    )

    with pytest.raises(
        ValidationError, match="partial_success requires at least one real resource"
    ):
        build_resource_final_v3(
            thread_id=THREAD_ID,
            request_id=REQUEST_ID,
            terminal_status="partial_success",
            resources=[],
            recommendations=[],
            blocked_resources=[blocked],
            errors=[],
            validation=_validation(
                resources=0,
                successes=0,
                blocked=1,
            ),
            summary="No resource was generated.",
        )


def test_resource_final_v3_resource_union_rejects_wrong_payload_shape():
    with pytest.raises(ValidationError, match="mindmap payload requires"):
        build_resource_final_v3_resource(
            thread_id=THREAD_ID,
            request_id=REQUEST_ID,
            kind="mindmap",
            status="success",
            title="Machine Learning Map",
            summary="A map with a mismatched payload.",
            payload={"study_plan": {"title": "Wrong payload"}},
            artifact_refs={},
            validation=_resource_validation(),
        )


def test_resource_final_v3_resource_rejects_empty_expected_payload():
    with pytest.raises(ValidationError, match="non-empty value"):
        build_resource_final_v3_resource(
            thread_id=THREAD_ID,
            request_id=REQUEST_ID,
            kind="mindmap",
            status="success",
            title="Machine Learning Map",
            summary="An empty map must not be reported as successful.",
            payload={"mindmap": {}},
            artifact_refs={},
            validation=_resource_validation(),
        )
