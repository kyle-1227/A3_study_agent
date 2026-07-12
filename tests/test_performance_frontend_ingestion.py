from __future__ import annotations

import pytest

from src.observability.performance_config import PerformanceObservabilityConfig
from src.observability.performance_contracts import FrontendPerformanceBatchV1
from src.observability.performance_runtime import performance_request_recorder
from src.observability.performance_service import (
    FrontendPerformanceRejected,
    PerformanceService,
)


def _config(*, enabled: bool = True) -> PerformanceObservabilityConfig:
    return PerformanceObservabilityConfig.model_validate(
        {
            "enabled": True,
            "max_spans_per_request": 64,
            "report_retention_count": 32,
            "frontend_ingestion": {
                "enabled": enabled,
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


def _batch(*, trace_id: str, thread_id: str = "thread-1") -> FrontendPerformanceBatchV1:
    return FrontendPerformanceBatchV1.model_validate(
        {
            "schema_version": "frontend_performance_v1",
            "request_id": "request-1",
            "thread_id": thread_id,
            "trace_id": trace_id,
            "milestones": [
                {"name": "submit_to_stream_context", "duration_ms": 12.5},
                {
                    "name": "submit_to_done",
                    "duration_ms": 83.0,
                    "status": "completed",
                },
            ],
        }
    )


def _registered_service() -> tuple[PerformanceService, dict, object]:
    service = PerformanceService(_config(), secret=b"s" * 32)
    with performance_request_recorder(
        request_id="request-1",
        thread_id="thread-1",
        max_spans=64,
    ) as recorder:
        capability = service.register_request(recorder=recorder, user_id="user-1")
        service.mark_capability_exposed(capability["token"])
    report = service.store_report(recorder, token=capability["token"])
    return service, capability, report


def test_frontend_batch_is_bound_and_single_use():
    service, capability, report = _registered_service()
    payload = _batch(trace_id=capability["trace_id"])

    event = service.accept_frontend_batch(
        authorization=f"Bearer {capability['token']}",
        origin="http://localhost:3000",
        raw_size=512,
        payload=payload,
    )

    assert report.frontend_sample_status == "pending"
    assert event.milestone_count == 2
    assert service.get_frontend_batch("request-1") == payload
    updated = service.get_report("request-1")
    assert updated is not None
    assert updated.frontend_sample_status == "accepted"
    assert updated.frontend_milestone_count == 2

    with pytest.raises(FrontendPerformanceRejected) as exc_info:
        service.accept_frontend_batch(
            authorization=f"Bearer {capability['token']}",
            origin="http://localhost:3000",
            raw_size=512,
            payload=payload,
        )
    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "frontend_performance_replay_rejected"


@pytest.mark.parametrize(
    ("origin", "raw_size", "thread_id", "expected_code"),
    [
        (
            "https://untrusted.example",
            512,
            "thread-1",
            "frontend_performance_origin_rejected",
        ),
        (
            "http://localhost:3000",
            20_000,
            "thread-1",
            "frontend_performance_payload_too_large",
        ),
        (
            "http://localhost:3000",
            512,
            "thread-other",
            "frontend_performance_binding_mismatch",
        ),
    ],
)
def test_frontend_batch_rejects_origin_size_and_binding_mismatch(
    origin: str,
    raw_size: int,
    thread_id: str,
    expected_code: str,
):
    service, capability, _ = _registered_service()

    with pytest.raises(FrontendPerformanceRejected) as exc_info:
        service.accept_frontend_batch(
            authorization=f"Bearer {capability['token']}",
            origin=origin,
            raw_size=raw_size,
            payload=_batch(trace_id=capability["trace_id"], thread_id=thread_id),
        )

    assert exc_info.value.code == expected_code


def test_disabled_frontend_ingestion_is_explicitly_rejected():
    service = PerformanceService(_config(enabled=False), secret=None)
    payload = _batch(
        trace_id="trace:v1:" + "0" * 64,
    )

    with pytest.raises(FrontendPerformanceRejected) as exc_info:
        service.accept_frontend_batch(
            authorization="Bearer unavailable",
            origin="http://localhost:3000",
            raw_size=512,
            payload=payload,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == "frontend_performance_disabled"


def test_expired_unsubmitted_frontend_sample_becomes_incomplete(monkeypatch):
    clock = {"value": 100}
    monkeypatch.setattr(
        "src.observability.performance_service.time.time", lambda: clock["value"]
    )
    service = PerformanceService(_config(), secret=b"s" * 32)
    with performance_request_recorder(
        request_id="request-1",
        thread_id="thread-1",
        max_spans=64,
    ) as recorder:
        capability = service.register_request(recorder=recorder, user_id="user-1")
        service.mark_capability_exposed(capability["token"])
    service.store_report(recorder, token=capability["token"])

    clock["value"] = 400
    report = service.get_report("request-1")

    assert report is not None
    assert report.frontend_sample_status == "incomplete"
