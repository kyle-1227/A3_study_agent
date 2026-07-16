from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import app as a3_app
from src.schemas import HealthLiveV1, HealthReadyV3


_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64


@dataclass(frozen=True)
class _FakeGraphManifest:
    graph_version: str


@dataclass(frozen=True)
class _FakeKnowledgeGraphSubject:
    subject_id: str


@dataclass(frozen=True)
class _FakeKnowledgeGraph:
    data_version: str
    artifact_fingerprint: str
    subjects: tuple[_FakeKnowledgeGraphSubject, ...] = ()


@dataclass(frozen=True)
class _FakeLearningGuidanceRuntime:
    knowledge_graph: _FakeKnowledgeGraph


@dataclass(frozen=True)
class _FakeParentChildRuntime:
    generation_id: str


@dataclass(frozen=True)
class _FakeEvidenceOrchestrationRuntime:
    learning_guidance: _FakeLearningGuidanceRuntime
    parent_child: _FakeParentChildRuntime
    orchestration_fingerprint: str


@dataclass(frozen=True)
class _FakeServedCandidateRuntime:
    orchestration: _FakeEvidenceOrchestrationRuntime
    generation_manifest_fingerprint: str
    deployment_mode: str
    rollout_activation_enabled: bool
    rollout_shadow_enabled: bool


def _ready_state() -> SimpleNamespace:
    guidance = _FakeLearningGuidanceRuntime(
        knowledge_graph=_FakeKnowledgeGraph(
            data_version="kg-2026-07-15",
            artifact_fingerprint=_DIGEST_A,
        )
    )
    orchestration = _FakeEvidenceOrchestrationRuntime(
        learning_guidance=guidance,
        parent_child=_FakeParentChildRuntime(generation_id="pc-generation-1"),
        orchestration_fingerprint=_DIGEST_B,
    )
    return SimpleNamespace(
        checkpointer_enabled=True,
        checkpointer_type="postgres",
        readiness_db_timeout_seconds=3.0,
        graph_manifest=_FakeGraphManifest(graph_version="graph-v1"),
        graph_version="graph-v1",
        learning_guidance_runtime=guidance,
        served_candidate_owner=_FakeServedCandidateRuntime(
            orchestration=orchestration,
            generation_manifest_fingerprint=_DIGEST_C,
            deployment_mode="active",
            rollout_activation_enabled=True,
            rollout_shadow_enabled=False,
        ),
        served_candidate_runtime=orchestration,
        parent_child_generation_id="pc-generation-1",
        parent_child_generation_manifest_fingerprint=_DIGEST_C,
    )


def _request(state: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(app=SimpleNamespace(state=state))


@pytest.fixture
def health_runtime_types(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(a3_app, "GraphManifest", _FakeGraphManifest)
    monkeypatch.setattr(
        a3_app,
        "LearningGuidanceRuntime",
        _FakeLearningGuidanceRuntime,
    )
    monkeypatch.setattr(
        a3_app,
        "EvidenceOrchestrationRuntime",
        _FakeEvidenceOrchestrationRuntime,
    )
    monkeypatch.setattr(
        a3_app,
        "ServedCandidateRuntime",
        _FakeServedCandidateRuntime,
    )


@pytest.mark.asyncio
async def test_health_live_is_strict_v1() -> None:
    payload = await a3_app.health_live_endpoint()

    assert payload == HealthLiveV1(schema_version="health_live_v1", status="live")
    with pytest.raises(ValidationError):
        HealthLiveV1.model_validate(
            {
                "schema_version": "health_live_v1",
                "status": "live",
                "unexpected": True,
            }
        )


@pytest.mark.asyncio
async def test_subjects_endpoint_uses_curated_knowledge_graph(
    health_runtime_types: None,
) -> None:
    subjects = (
        _FakeKnowledgeGraphSubject(subject_id="big_data"),
        _FakeKnowledgeGraphSubject(subject_id="computer"),
        _FakeKnowledgeGraphSubject(subject_id="machine_learning"),
        _FakeKnowledgeGraphSubject(subject_id="math"),
        _FakeKnowledgeGraphSubject(subject_id="python"),
    )
    state = SimpleNamespace(
        learning_guidance_runtime=_FakeLearningGuidanceRuntime(
            knowledge_graph=_FakeKnowledgeGraph(
                data_version="kg-2026-07-15",
                artifact_fingerprint=_DIGEST_A,
                subjects=subjects,
            )
        )
    )

    payload = await a3_app.get_subjects_endpoint(_request(state))

    assert payload == {
        "subjects": [
            "big_data",
            "computer",
            "machine_learning",
            "math",
            "python",
        ]
    }


@pytest.mark.asyncio
async def test_subjects_endpoint_fails_closed_without_curated_runtime(
    health_runtime_types: None,
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await a3_app.get_subjects_endpoint(_request(SimpleNamespace()))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "subject_catalog_runtime_unavailable"


@pytest.mark.asyncio
async def test_health_ready_returns_all_verified_identities(
    monkeypatch: pytest.MonkeyPatch,
    health_runtime_types: None,
) -> None:
    probes: list[tuple[str, float]] = []

    async def probe(*, db_uri: str, timeout_seconds: float) -> None:
        probes.append((db_uri, timeout_seconds))

    monkeypatch.setattr(a3_app, "get_db_uri", lambda: "postgresql:///a3_test")
    monkeypatch.setattr(a3_app, "_probe_postgres_readiness", probe)

    payload = await a3_app.health_ready_endpoint(_request(_ready_state()))

    assert payload == HealthReadyV3(
        schema_version="health_ready_v3",
        status="ready",
        checkpointer_type="postgres",
        graph_version="graph-v1",
        knowledge_graph_data_version="kg-2026-07-15",
        knowledge_graph_artifact_fingerprint=_DIGEST_A,
        parent_child_generation_id="pc-generation-1",
        parent_child_generation_manifest_fingerprint=_DIGEST_C,
        evidence_orchestration_fingerprint=_DIGEST_B,
        deployment_mode="active",
        rollout_activation_enabled=True,
        rollout_shadow_enabled=False,
    )
    assert probes == [("postgresql:///a3_test", 3.0)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (
            lambda state: setattr(state, "graph_manifest", None),
            "health_ready_graph_manifest_unavailable",
        ),
        (
            lambda state: setattr(state, "graph_version", "wrong-graph"),
            "health_ready_graph_manifest_invalid",
        ),
        (
            lambda state: setattr(
                state,
                "parent_child_generation_id",
                "wrong-generation",
            ),
            "health_ready_generation_invalid",
        ),
        (
            lambda state: setattr(
                state,
                "parent_child_generation_manifest_fingerprint",
                _DIGEST_A,
            ),
            "health_ready_generation_manifest_invalid",
        ),
        (
            lambda state: setattr(
                state,
                "served_candidate_owner",
                replace(
                    state.served_candidate_owner, deployment_mode="inactive_canary"
                ),
            ),
            "health_ready_deployment_mode_invalid",
        ),
        (
            lambda state: setattr(
                state,
                "served_candidate_owner",
                replace(
                    state.served_candidate_owner,
                    rollout_activation_enabled=False,
                ),
            ),
            "health_ready_rollout_state_invalid",
        ),
        (
            lambda state: setattr(
                state,
                "served_candidate_owner",
                replace(
                    state.served_candidate_owner,
                    rollout_shadow_enabled=True,
                ),
            ),
            "health_ready_rollout_state_invalid",
        ),
        (
            lambda state: setattr(state, "checkpointer_type", "memory"),
            "health_ready_postgres_checkpointer_required",
        ),
    ],
)
async def test_health_ready_fails_closed_on_runtime_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
    health_runtime_types: None,
    mutation,
    expected_code: str,
) -> None:
    state = _ready_state()
    mutation(state)
    monkeypatch.setattr(a3_app, "get_db_uri", lambda: "postgresql:///a3_test")

    with pytest.raises(HTTPException) as exc_info:
        await a3_app.health_ready_endpoint(_request(state))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == expected_code


@pytest.mark.asyncio
async def test_postgres_probe_redacts_driver_error_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_connect(*args, **kwargs):
        del args, kwargs
        raise a3_app.psycopg.OperationalError(
            "connection rejected for private-database.internal"
        )

    monkeypatch.setattr(a3_app.psycopg.AsyncConnection, "connect", fail_connect)

    with pytest.raises(HTTPException) as exc_info:
        await a3_app._probe_postgres_readiness(
            db_uri="postgresql://private-database.internal/a3",
            timeout_seconds=1.0,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "health_ready_database_unavailable"
    assert "private-database" not in str(exc_info.value.detail)


def test_readiness_timeout_is_explicit_and_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    def setting(*args):
        calls.append(args)
        return 2

    monkeypatch.setattr(a3_app, "get_setting", setting)
    assert a3_app._readiness_db_timeout_seconds() == 2.0
    assert calls == [("server.readiness_db_timeout_seconds",)]

    monkeypatch.setattr(a3_app, "get_setting", lambda *args: None)
    with pytest.raises(RuntimeError, match="must be an explicit positive number"):
        a3_app._readiness_db_timeout_seconds()


def _health_ready_payload() -> dict[str, object]:
    return {
        "schema_version": "health_ready_v3",
        "status": "ready",
        "checkpointer_type": "postgres",
        "graph_version": "graph-v1",
        "knowledge_graph_data_version": "kg-v1",
        "knowledge_graph_artifact_fingerprint": _DIGEST_A,
        "parent_child_generation_id": "generation-v1",
        "parent_child_generation_manifest_fingerprint": _DIGEST_C,
        "evidence_orchestration_fingerprint": _DIGEST_B,
        "deployment_mode": "active",
        "rollout_activation_enabled": True,
        "rollout_shadow_enabled": False,
    }


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("schema_version", "health_ready_v2"),
        ("knowledge_graph_artifact_fingerprint", "not-a-digest"),
        ("deployment_mode", "inactive_canary"),
        ("rollout_activation_enabled", False),
        ("rollout_shadow_enabled", True),
        ("rollout_activation_enabled", "true"),
        ("rollout_activation_enabled", 0),
        ("rollout_activation_enabled", None),
    ],
)
def test_health_ready_v3_rejects_invalid_identity(
    field_name: str,
    invalid_value: object,
) -> None:
    payload = _health_ready_payload()
    payload[field_name] = invalid_value

    with pytest.raises(ValidationError):
        HealthReadyV3.model_validate(payload, strict=True)


def test_health_ready_v3_rejects_missing_and_extra_fields() -> None:
    missing = _health_ready_payload()
    missing.pop("rollout_activation_enabled")
    extra = _health_ready_payload()
    extra["candidate_mode"] = "active"

    with pytest.raises(ValidationError):
        HealthReadyV3.model_validate(missing, strict=True)
    with pytest.raises(ValidationError):
        HealthReadyV3.model_validate(extra, strict=True)
