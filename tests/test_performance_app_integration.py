from __future__ import annotations

import json

from fastapi.testclient import TestClient

import app as app_module
from src.observability.performance_config import PerformanceObservabilityConfig
from src.observability.performance_contracts import FrontendPerformanceBatchV1
from src.observability.performance_runtime import performance_request_recorder
from src.observability.performance_service import (
    PerformanceService,
    configure_performance_service,
    observe_request_performance,
    reset_performance_service_for_tests,
)


def _config(*, frontend_enabled: bool = True) -> PerformanceObservabilityConfig:
    return PerformanceObservabilityConfig.model_validate(
        {
            "enabled": True,
            "max_spans_per_request": 64,
            "report_retention_count": 32,
            "frontend_ingestion": {
                "enabled": frontend_enabled,
                "endpoint_path": "/observability/frontend-performance",
                "secret_env": "FRONTEND_PERFORMANCE_HMAC_SECRET",
                "allowed_origins": ["http://localhost:3000"],
                "token_ttl_seconds": 300,
                "max_payload_bytes": 16_384,
                "max_milestones_per_request": 16,
                "max_batches_per_request": 1,
            },
        }
    )


def _service_and_capability() -> tuple[PerformanceService, dict]:
    service = PerformanceService(_config(), secret=b"p" * 32)
    with performance_request_recorder(
        request_id="request-1",
        thread_id="thread-1",
        max_spans=64,
    ) as recorder:
        capability = service.register_request(recorder=recorder, user_id="user-1")
        service.mark_capability_exposed(capability["token"])
    service.store_report(recorder, token=capability["token"])
    return service, capability


def _payload(trace_id: str) -> dict:
    return FrontendPerformanceBatchV1.model_validate(
        {
            "schema_version": "frontend_performance_v1",
            "request_id": "request-1",
            "thread_id": "thread-1",
            "trace_id": trace_id,
            "milestones": [
                {"name": "submit_to_stream_context", "duration_ms": 10.0},
                {
                    "name": "submit_to_done",
                    "duration_ms": 50.0,
                    "status": "completed",
                },
            ],
        }
    ).model_dump(mode="json")


def test_stream_context_exposes_request_bound_capability_and_report(monkeypatch):
    monkeypatch.setenv("FRONTEND_PERFORMANCE_HMAC_SECRET", "z" * 32)
    service = configure_performance_service(_config())
    try:
        with observe_request_performance(
            request_id="request-1",
            thread_id="thread-1",
            user_id="user-1",
        ):
            payload = app_module._stream_context_payload(
                request_id="request-1",
                thread_id="thread-1",
                graph_version="graph:v1:test",
            )
            capability = payload["performance_telemetry"]
            assert capability["trace_id"].startswith("trace:v1:")
            assert capability["endpoint"] == "/observability/frontend-performance"
            assert capability["token"]

        report = service.get_report("request-1")
        assert report is not None
        assert report.frontend_sample_status == "pending"
        assert report.request_id == "request-1"
    finally:
        reset_performance_service_for_tests()


def test_stream_context_without_graph_version_preserves_legacy_event_contract(
    monkeypatch,
):
    monkeypatch.setenv("FRONTEND_PERFORMANCE_HMAC_SECRET", "z" * 32)
    service = configure_performance_service(_config())
    try:
        with observe_request_performance(
            request_id="request-1",
            thread_id="thread-1",
            user_id="user-1",
        ):
            payload = app_module._stream_context_payload(
                request_id="request-1",
                thread_id="thread-1",
                graph_version="",
            )
            assert "performance_telemetry" not in payload

        report = service.get_report("request-1")
        assert report is not None
        assert report.frontend_sample_status == "incomplete"
    finally:
        reset_performance_service_for_tests()


def test_frontend_performance_endpoint_accepts_once_and_rejects_missing_auth(
    monkeypatch,
):
    service, capability = _service_and_capability()
    monkeypatch.setattr(app_module, "get_performance_service", lambda: service)
    client = TestClient(app_module.app)
    headers = {
        "Authorization": f"Bearer {capability['token']}",
        "Origin": "http://localhost:3000",
        "Content-Type": "application/json",
    }
    payload = _payload(capability["trace_id"])

    accepted = client.post(
        "/observability/frontend-performance",
        headers=headers,
        content=json.dumps(payload),
    )
    replayed = client.post(
        "/observability/frontend-performance",
        headers=headers,
        content=json.dumps(payload),
    )

    assert accepted.status_code == 204
    assert accepted.content == b""
    assert replayed.status_code == 409
    assert replayed.json()["detail"] == "frontend_performance_replay_rejected"

    other_service, other_capability = _service_and_capability()
    monkeypatch.setattr(app_module, "get_performance_service", lambda: other_service)
    missing_auth = client.post(
        "/observability/frontend-performance",
        headers={"Origin": "http://localhost:3000"},
        json=_payload(other_capability["trace_id"]),
    )
    assert missing_auth.status_code == 401


def test_frontend_performance_endpoint_enforces_raw_body_limit(monkeypatch):
    service, capability = _service_and_capability()
    monkeypatch.setattr(app_module, "get_performance_service", lambda: service)
    client = TestClient(app_module.app)

    response = client.post(
        "/observability/frontend-performance",
        headers={
            "Authorization": f"Bearer {capability['token']}",
            "Origin": "http://localhost:3000",
        },
        content=b"x" * 16_385,
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "frontend_performance_payload_too_large"
