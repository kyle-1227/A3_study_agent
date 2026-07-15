"""Onboarding V2 API and OpenAPI integration tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

import app as app_module
from src.learning_guidance.profile_writer import LearningGuidanceProfileWriterV1
from src.profile.storage import SQLiteProfileStore


def _payload(
    *,
    request_id: str = "onboard-request-1",
    user_id: str = "user-1",
    topic_id: str = "python-basics",
    level: float = 0.25,
) -> dict[str, object]:
    return {
        "schema_version": "onboard_v2",
        "profile": {
            "schema_version": "learning_guidance_profile_write_request_v1",
            "request_id": request_id,
            "user_id": user_id,
            "skills": [
                {
                    "subject": "python",
                    "topic_id": topic_id,
                    "level": level,
                    "confidence": 0.8,
                }
            ],
            "goals": [
                {
                    "subject": "python",
                    "topic_id": topic_id,
                    "goal": "Master Python basics",
                    "importance": 0.9,
                    "progress": 0.0,
                }
            ],
            "preferences": [
                {
                    "subject": "python",
                    "topic_id": topic_id,
                    "dimension": "prefer_visual",
                    "strength": 0.8,
                }
            ],
        },
        "nickname": "learner",
        "grade": "grade-10",
        "dislikes": ["rote repetition"],
    }


@pytest.fixture
def onboarding_client(
    tmp_path: Path,
    learning_guidance_runtime,
) -> Iterator[TestClient]:
    state = app_module.app.state
    sentinel = object()
    prior_runtime = getattr(state, "learning_guidance_runtime", sentinel)
    prior_writer = getattr(state, "learning_guidance_profile_writer", sentinel)
    state.learning_guidance_runtime = learning_guidance_runtime
    state.learning_guidance_profile_writer = LearningGuidanceProfileWriterV1(
        store=SQLiteProfileStore(tmp_path / "profile.sqlite"),
        knowledge_graph=learning_guidance_runtime.knowledge_graph,
    )
    client = TestClient(app_module.app)
    try:
        yield client
    finally:
        client.close()
        if prior_runtime is sentinel:
            delattr(state, "learning_guidance_runtime")
        else:
            state.learning_guidance_runtime = prior_runtime
        if prior_writer is sentinel:
            delattr(state, "learning_guidance_profile_writer")
        else:
            state.learning_guidance_profile_writer = prior_writer


def test_catalog_exposes_exact_production_runtime_identity(
    onboarding_client: TestClient,
    learning_guidance_runtime,
) -> None:
    response = onboarding_client.get("/learning-guidance/catalog")

    assert response.status_code == 200
    assert response.json() == {
        "schema_version": "learning_guidance_catalog_v1",
        "data_version": learning_guidance_runtime.knowledge_graph.data_version,
        "artifact_fingerprint": (
            learning_guidance_runtime.knowledge_graph.artifact_fingerprint
        ),
        "subjects": [
            {
                "subject_id": "python",
                "title": "Python",
                "topics": [
                    {
                        "topic_id": "python-basics",
                        "title": "Python basics",
                    }
                ],
            }
        ],
    }


def test_onboard_create_and_identical_replay(onboarding_client: TestClient) -> None:
    first = onboarding_client.post("/onboard", json=_payload())
    replay = onboarding_client.post("/onboard", json=_payload())

    assert first.status_code == 200
    assert replay.status_code == 200
    assert first.json()["status"] == "created"
    assert replay.json()["status"] == "replayed"
    assert replay.json()["request_id"] == "onboard-request-1"
    assert replay.json()["user_id"] == "user-1"
    assert replay.json()["skills_count"] == 1
    assert replay.json()["goals_count"] == 1
    assert replay.json()["preferences_count"] == 1


def test_onboard_request_drift_is_a_conflict(onboarding_client: TestClient) -> None:
    assert onboarding_client.post("/onboard", json=_payload()).status_code == 200

    conflict = onboarding_client.post("/onboard", json=_payload(level=0.5))

    assert conflict.status_code == 409
    assert conflict.json() == {"detail": "profile_write_request_conflict"}


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("nickname", "different learner"),
        ("grade", "grade-11"),
        ("dislikes", ["video lectures"]),
    ],
)
def test_onboard_top_level_drift_is_a_conflict(
    onboarding_client: TestClient,
    field_name: str,
    replacement: object,
) -> None:
    assert onboarding_client.post("/onboard", json=_payload()).status_code == 200
    drifted = _payload()
    drifted[field_name] = replacement

    conflict = onboarding_client.post("/onboard", json=drifted)

    assert conflict.status_code == 409
    assert conflict.json() == {"detail": "profile_write_request_conflict"}


def test_onboard_unknown_topic_is_rejected(onboarding_client: TestClient) -> None:
    response = onboarding_client.post(
        "/onboard",
        json=_payload(user_id="user-unknown", topic_id="python-unknown"),
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "profile_write_topic_invalid"}


@pytest.mark.parametrize(
    "payload",
    [
        {
            "user_id": "legacy-user",
            "nickname": "legacy",
            "subjects": ["python"],
            "skill_levels": {"python": 0.5},
            "goals": ["Learn Python"],
            "learning_style": {},
            "grade": "grade-10",
            "dislikes": [],
        },
        {**_payload(), "unexpected": True},
    ],
    ids=["legacy-v1", "extra-field"],
)
def test_onboard_rejects_legacy_and_schema_drift(
    onboarding_client: TestClient,
    payload: dict[str, object],
) -> None:
    response = onboarding_client.post("/onboard", json=payload)

    assert response.status_code == 422


def test_catalog_and_writer_fail_closed_when_state_is_invalid(
    onboarding_client: TestClient,
) -> None:
    app_module.app.state.learning_guidance_runtime = object()
    app_module.app.state.learning_guidance_profile_writer = object()

    catalog = onboarding_client.get("/learning-guidance/catalog")
    onboard = onboarding_client.post("/onboard", json=_payload())

    assert catalog.status_code == 503
    assert catalog.json() == {"detail": "learning_guidance_runtime_unavailable"}
    assert onboard.status_code == 503
    assert onboard.json() == {"detail": "learning_guidance_profile_writer_unavailable"}


def test_openapi_exposes_only_v2_onboarding_contracts() -> None:
    schema = app_module.app.openapi()

    onboard = schema["paths"]["/onboard"]["post"]
    catalog = schema["paths"]["/learning-guidance/catalog"]["get"]
    assert onboard["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/OnboardRequest"
    }
    assert onboard["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/OnboardResultV2"
    }
    assert catalog["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/LearningGuidanceCatalogV1"
    }
    properties = schema["components"]["schemas"]["OnboardRequest"]["properties"]
    assert set(properties) == {
        "schema_version",
        "profile",
        "nickname",
        "grade",
        "dislikes",
    }
