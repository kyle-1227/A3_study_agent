from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import app as a3_app
from src.schemas import HealthLiveV1, HealthReadyV4


_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64


@dataclass(frozen=True)
class _FakeGraphManifest:
    graph_version: str


@dataclass(frozen=True)
class _FakeKnowledgeGraph:
    data_version: str
    artifact_fingerprint: str


@dataclass(frozen=True)
class _FakeGuidance:
    knowledge_graph: _FakeKnowledgeGraph


@dataclass(frozen=True)
class _FakeParentChild:
    primary_revision: int
    primary_config_fingerprint: str


@dataclass(frozen=True)
class _FakeOrchestration:
    learning_guidance: _FakeGuidance
    parent_child: _FakeParentChild
    orchestration_fingerprint: str


@dataclass(frozen=True)
class _FakePrimaryOwner:
    orchestration: _FakeOrchestration
    primary_revision: int
    primary_updated_at: datetime
    primary_config_fingerprint: str


def _state() -> SimpleNamespace:
    updated = datetime(2026, 7, 19, tzinfo=UTC)
    guidance = _FakeGuidance(_FakeKnowledgeGraph("kg-v1", _DIGEST_A))
    orchestration = _FakeOrchestration(
        guidance,
        _FakeParentChild(1, _DIGEST_C),
        _DIGEST_B,
    )
    owner = _FakePrimaryOwner(orchestration, 1, updated, _DIGEST_C)
    return SimpleNamespace(
        checkpointer_enabled=True,
        checkpointer_type="postgres",
        readiness_db_timeout_seconds=3.0,
        graph_manifest=_FakeGraphManifest("graph-v1"),
        graph_version="graph-v1",
        learning_guidance_runtime=guidance,
        served_primary_owner=owner,
        served_primary_runtime=orchestration,
        parent_child_primary_revision=1,
        parent_child_primary_updated_at=updated,
        parent_child_primary_config_fingerprint=_DIGEST_C,
    )


def _request(state: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(app=SimpleNamespace(state=state))


@pytest.fixture
def runtime_types(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(a3_app, "GraphManifest", _FakeGraphManifest)
    monkeypatch.setattr(a3_app, "LearningGuidanceRuntime", _FakeGuidance)
    monkeypatch.setattr(a3_app, "EvidenceOrchestrationRuntime", _FakeOrchestration)
    monkeypatch.setattr(a3_app, "ServedPrimaryRuntime", _FakePrimaryOwner)


@pytest.mark.asyncio
async def test_health_live_is_strict_v1() -> None:
    assert await a3_app.health_live_endpoint() == HealthLiveV1(
        schema_version="health_live_v1",
        status="live",
    )


@pytest.mark.asyncio
async def test_health_ready_returns_verified_primary_v4(
    monkeypatch: pytest.MonkeyPatch,
    runtime_types: None,
) -> None:
    probes: list[tuple[str, float]] = []

    async def probe(*, db_uri: str, timeout_seconds: float) -> None:
        probes.append((db_uri, timeout_seconds))

    monkeypatch.setattr(a3_app, "get_db_uri", lambda: "postgresql:///a3_test")
    monkeypatch.setattr(a3_app, "_probe_postgres_readiness", probe)

    payload = await a3_app.health_ready_endpoint(_request(_state()))

    assert payload == HealthReadyV4(
        schema_version="health_ready_v4",
        status="ready",
        checkpointer_type="postgres",
        graph_version="graph-v1",
        knowledge_graph_data_version="kg-v1",
        knowledge_graph_artifact_fingerprint=_DIGEST_A,
        parent_child_primary_revision=1,
        parent_child_primary_updated_at=datetime(2026, 7, 19, tzinfo=UTC),
        parent_child_primary_config_fingerprint=_DIGEST_C,
        evidence_orchestration_fingerprint=_DIGEST_B,
    )
    assert probes == [("postgresql:///a3_test", 3.0)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda state: setattr(state, "parent_child_primary_revision", 2),
            "health_ready_primary_invalid",
        ),
        (
            lambda state: setattr(
                state,
                "parent_child_primary_config_fingerprint",
                _DIGEST_A,
            ),
            "health_ready_primary_invalid",
        ),
        (
            lambda state: setattr(state, "served_primary_owner", None),
            "health_ready_primary_unavailable",
        ),
        (
            lambda state: setattr(state, "checkpointer_type", "memory"),
            "health_ready_postgres_checkpointer_required",
        ),
    ],
)
async def test_health_ready_fails_closed_on_primary_drift(
    monkeypatch: pytest.MonkeyPatch,
    runtime_types: None,
    mutate,
    code: str,
) -> None:
    state = _state()
    mutate(state)
    monkeypatch.setattr(a3_app, "get_db_uri", lambda: "postgresql:///a3_test")

    with pytest.raises(HTTPException) as exc_info:
        await a3_app.health_ready_endpoint(_request(state))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == code


def _payload() -> dict[str, object]:
    return {
        "schema_version": "health_ready_v4",
        "status": "ready",
        "checkpointer_type": "postgres",
        "graph_version": "graph-v1",
        "knowledge_graph_data_version": "kg-v1",
        "knowledge_graph_artifact_fingerprint": _DIGEST_A,
        "parent_child_primary_revision": 1,
        "parent_child_primary_updated_at": "2026-07-19T00:00:00+00:00",
        "parent_child_primary_config_fingerprint": _DIGEST_C,
        "evidence_orchestration_fingerprint": _DIGEST_B,
    }


def test_health_ready_v4_is_strict() -> None:
    with pytest.raises(ValidationError):
        HealthReadyV4.model_validate(
            {**_payload(), "unexpected": True},
            strict=True,
        )
    with pytest.raises(ValidationError):
        HealthReadyV4.model_validate(
            {**_payload(), "parent_child_primary_revision": 0},
            strict=True,
        )
