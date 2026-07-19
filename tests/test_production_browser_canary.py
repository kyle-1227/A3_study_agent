from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import httpx
import pytest
from pydantic import ValidationError

import scripts.run_production_browser_canary as canary
from scripts.run_production_browser_canary import (
    ProductionBrowserCanaryReportV4,
    ProductionCanaryError,
    ProductionCanaryExpectedPrimaryV1,
    _fetch_served_identity,
    _require_stable_served_identity,
    _served_identity,
)
from src.schemas import HealthReadyV4


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64


def _payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "health_ready_v4",
        "status": "ready",
        "checkpointer_type": "postgres",
        "graph_version": "graph-v1",
        "knowledge_graph_data_version": "kg-v1",
        "knowledge_graph_artifact_fingerprint": DIGEST_A,
        "parent_child_primary_revision": 1,
        "parent_child_primary_updated_at": "2026-07-19T00:00:00+00:00",
        "parent_child_primary_config_fingerprint": DIGEST_B,
        "evidence_orchestration_fingerprint": DIGEST_C,
    }
    payload.update(updates)
    return payload


def _expected() -> ProductionCanaryExpectedPrimaryV1:
    return ProductionCanaryExpectedPrimaryV1(
        parent_child_primary_revision=1,
        parent_child_primary_config_fingerprint=DIGEST_B,
    )


async def _fetch(payload: dict[str, object]):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health/ready"
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await _fetch_served_identity(
            client=client,
            backend_url="http://backend.test",
            expected_primary=_expected(),
            expected_knowledge_graph_data_version="kg-v1",
            expected_knowledge_graph_artifact_fingerprint=DIGEST_A,
        )


def test_fetch_served_identity_binds_health_ready_v4() -> None:
    identity = asyncio.run(_fetch(_payload()))

    assert identity.health_ready_schema_version == "health_ready_v4"
    assert identity.parent_child_primary_revision == 1
    assert identity.parent_child_primary_config_fingerprint == DIGEST_B
    assert identity.identity_fingerprint == canary.canonical_sha256(
        identity.model_dump(mode="json", exclude={"identity_fingerprint"})
    )


@pytest.mark.parametrize(
    "payload",
    [
        _payload(schema_version="health_ready_v3"),
        _payload(parent_child_primary_revision=2),
        _payload(parent_child_primary_revision="1"),
        _payload(parent_child_primary_config_fingerprint=DIGEST_D),
        _payload(parent_child_primary_config_fingerprint="B" * 64),
        _payload(parent_child_primary_updated_at="not-a-date"),
    ],
)
def test_fetch_rejects_primary_contract_drift(payload: dict[str, object]) -> None:
    with pytest.raises(ProductionCanaryError):
        asyncio.run(_fetch(payload))


def test_canary_requires_identical_primary_identity_before_and_after() -> None:
    before = _served_identity(HealthReadyV4.model_validate(_payload(), strict=False))
    after = _served_identity(
        HealthReadyV4.model_validate(
            _payload(parent_child_primary_revision=2),
            strict=False,
        )
    )

    with pytest.raises(ProductionCanaryError, match="changed"):
        _require_stable_served_identity(before, after)


def test_report_binding_is_primary_identity_bound_and_secret_safe() -> None:
    identity = _served_identity(HealthReadyV4.model_validate(_payload(), strict=False))
    binding = canary.canonical_sha256(
        {
            "schema_version": "production_canary_binding_v3",
            "dataset_content_fingerprint": DIGEST_D,
            "served_identity_fingerprint": identity.identity_fingerprint,
        }
    )
    report = ProductionBrowserCanaryReportV4(
        schema_version="production_browser_canary_v4",
        created_at_utc=datetime(2026, 7, 19, tzinfo=UTC).isoformat(),
        dataset_id="smoke-v4",
        dataset_content_fingerprint=DIGEST_D,
        served_identity=identity,
        canary_binding_fingerprint=binding,
        readiness_observation_count=2,
        smoke_authoring_only=True,
        case_count=1,
        all_cases_completed=True,
        verified_download_count=1,
        cases=[{"case_id": "case-1"}],
    )
    serialized = report.model_dump_json().casefold()
    for forbidden in ("query", "authorization", "provider_body", "db_uri"):
        assert forbidden not in serialized

    invalid = report.model_dump(mode="python")
    invalid["canary_binding_fingerprint"] = DIGEST_A
    with pytest.raises(ValidationError):
        ProductionBrowserCanaryReportV4.model_validate(invalid, strict=True)
