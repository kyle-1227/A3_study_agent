from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

import scripts.run_production_browser_canary as canary
from scripts.run_production_browser_canary import (
    ProductionBrowserCanaryReportV3,
    ProductionCanaryError,
    ProductionCanaryExpectedGenerationV1,
    _artifact_paths,
    _fetch_served_identity,
    _parse_sse_text,
    _refresh_projection_mode,
    _require_stable_served_identity,
    _safe_terminal_projection,
    _served_identity,
    _validate_stream_events,
    _verify_downloads,
    _verify_replay,
)
from src.graph.resource_final_v3 import (
    ResourceFinalV3ResourceValidation,
    ResourceFinalV3Validation,
    build_resource_final_v3,
    build_resource_final_v3_resource,
)


REQUEST_ID = "00000000-0000-4000-8000-000000000001"
RESUME_REQUEST_ID = "00000000-0000-4000-8000-000000000002"
THREAD_ID = "thread-1"
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64


def _health_ready_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "health_ready_v3",
        "status": "ready",
        "checkpointer_type": "postgres",
        "graph_version": "graph-v1",
        "knowledge_graph_data_version": "kg-2026-07-15",
        "knowledge_graph_artifact_fingerprint": DIGEST_A,
        "parent_child_generation_id": "pc-generation-1",
        "parent_child_generation_manifest_fingerprint": DIGEST_B,
        "evidence_orchestration_fingerprint": DIGEST_C,
        "deployment_mode": "active",
        "rollout_activation_enabled": True,
        "rollout_shadow_enabled": False,
    }
    payload.update(updates)
    return payload


def _expected_generation() -> ProductionCanaryExpectedGenerationV1:
    return ProductionCanaryExpectedGenerationV1(
        parent_child_generation_id="pc-generation-1",
        parent_child_generation_manifest_fingerprint=DIGEST_B,
    )


def _resource_final(*, request_id: str = REQUEST_ID) -> dict:
    resource = build_resource_final_v3_resource(
        thread_id=THREAD_ID,
        request_id=request_id,
        kind="review_doc",
        status="success",
        title="Bounded canary document",
        summary="One validated document.",
        payload={"review_doc": "bounded", "review_doc_artifacts": {}},
        artifact_refs={"docx": "/artifacts/review-docs/canary/document.docx"},
        validation=ResourceFinalV3ResourceValidation(
            schema_version="resource_validation_v1",
            resource_type="review_doc",
            valid=True,
            terminal_status="success",
            renderable_count=1,
            downloadable_count=1,
            verified_local_count=1,
            remote_unverified_count=0,
            failure_reason="",
            warnings=(),
        ),
    )
    return build_resource_final_v3(
        thread_id=THREAD_ID,
        request_id=request_id,
        terminal_status="success",
        resources=(resource,),
        recommendations=(),
        blocked_resources=(),
        errors=(),
        validation=ResourceFinalV3Validation(
            schema_version="resource_final_validation_v3",
            resource_count=1,
            success_count=1,
            partial_success_count=0,
            failed_count=0,
            blocked_count=0,
            renderable_count=1,
            downloadable_count=1,
        ),
        summary="One validated resource was generated.",
    ).model_dump(mode="json")


def _event(
    event_type: str,
    sequence: int,
    *,
    data: dict | None = None,
    stream_id: str = "stream-1",
    request_id: str = REQUEST_ID,
    thread_id: str = THREAD_ID,
) -> dict:
    return {
        "schema_version": "agent_stream_v2",
        "type": event_type,
        "stream_id": stream_id,
        "event_id": f"{stream_id}:{sequence}",
        "sequence": sequence,
        "request_id": request_id,
        "thread_id": thread_id,
        "created_at": "2026-07-15T12:00:00+00:00",
        "data": {} if data is None else data,
    }


def _completed_events(
    *,
    stream_id: str = "stream-1",
    request_id: str = REQUEST_ID,
) -> list[dict]:
    return [
        _event(
            "stream_start",
            1,
            stream_id=stream_id,
            request_id=request_id,
        ),
        _event(
            "activity_update",
            2,
            data={"kind": "canary_progress", "payload": {"completed": True}},
            stream_id=stream_id,
            request_id=request_id,
        ),
        _event(
            "resource_final",
            3,
            data=_resource_final(request_id=request_id),
            stream_id=stream_id,
            request_id=request_id,
        ),
        _event(
            "stream_done",
            4,
            stream_id=stream_id,
            request_id=request_id,
        ),
    ]


def test_validate_stream_events_accepts_strict_resource_final() -> None:
    summary = _validate_stream_events(_completed_events())

    assert summary["event_count"] == 4
    assert summary["stream_count"] == 1
    assert summary["interrupt_count"] == 0
    assert summary["terminal_type"] == "resource_final"
    assert summary["thread_id"] == THREAD_ID
    assert summary["initial_request_id"] == REQUEST_ID


def test_validate_stream_events_accepts_interrupt_then_resume() -> None:
    events = [
        _event("stream_start", 1),
        _event("interrupt", 2),
        _event("stream_done", 3),
        *_completed_events(
            stream_id="stream-2",
            request_id=RESUME_REQUEST_ID,
        ),
    ]

    summary = _validate_stream_events(events)

    assert summary["stream_count"] == 2
    assert summary["interrupt_count"] == 1
    assert summary["initial_request_id"] == REQUEST_ID
    assert summary["request_id"] == RESUME_REQUEST_ID


@pytest.mark.parametrize(
    "events",
    [
        [_event("evidence_progress", 1)],
        [_event("stream_start", 1), _event("resource_final", 3)],
        [
            _event("stream_start", 1),
            _event("resource_final", 2),
            _event("stream_error", 3),
            _event("stream_done", 4),
        ],
        [
            _event("stream_start", 1),
            _event("resource_final", 2, request_id=RESUME_REQUEST_ID),
            _event("stream_done", 3),
        ],
    ],
)
def test_validate_stream_events_rejects_invalid_terminal_contract(events) -> None:
    with pytest.raises(ProductionCanaryError):
        _validate_stream_events(events)


def test_safe_terminal_projection_validates_v3_and_drops_generated_body() -> None:
    payload = _resource_final()
    terminal_data = {key: value for key, value in payload.items() if key != "type"}

    result = _safe_terminal_projection(
        terminal_data,
        terminal_type="resource_final",
        expected_resource_types=("review_doc",),
        expected_request_id=REQUEST_ID,
        expected_thread_id=THREAD_ID,
    )

    assert result == {
        "schema_version": "resource_final_v3",
        "resource_final_id": payload["resource_final_id"],
        "payload_hash": payload["payload_hash"],
        "request_id": REQUEST_ID,
        "thread_id": THREAD_ID,
        "terminal_status": "success",
        "resources": [{"resource_type": "review_doc", "status": "success"}],
        "blocked_resources": [],
    }
    assert "payload" not in result
    assert "summary" not in result


def test_safe_terminal_projection_rejects_schema_drift_and_identity_drift() -> None:
    invalid_status = _resource_final()
    invalid_status.pop("type")
    invalid_status["terminal_status"] = "all_resources_ready"

    with pytest.raises(ProductionCanaryError, match="resource_final_v3"):
        _safe_terminal_projection(
            invalid_status,
            terminal_type="resource_final",
            expected_resource_types=("review_doc",),
            expected_request_id=REQUEST_ID,
            expected_thread_id=THREAD_ID,
        )
    with pytest.raises(ProductionCanaryError, match="identity"):
        _safe_terminal_projection(
            {key: value for key, value in _resource_final().items() if key != "type"},
            terminal_type="resource_final",
            expected_resource_types=("review_doc",),
            expected_request_id=RESUME_REQUEST_ID,
            expected_thread_id=THREAD_ID,
        )


def test_safe_terminal_projection_rejects_transport_type_drift() -> None:
    terminal_data = {
        key: value for key, value in _resource_final().items() if key != "type"
    }

    with pytest.raises(ProductionCanaryError, match="terminal type"):
        _safe_terminal_projection(
            terminal_data,
            terminal_type="qa_final",
            expected_resource_types=("review_doc",),
            expected_request_id=REQUEST_ID,
            expected_thread_id=THREAD_ID,
        )
    with pytest.raises(ProductionCanaryError, match="must not duplicate"):
        _safe_terminal_projection(
            _resource_final(),
            terminal_type="resource_final",
            expected_resource_types=("review_doc",),
            expected_request_id=REQUEST_ID,
            expected_thread_id=THREAD_ID,
        )


def test_parse_sse_text_and_artifact_paths_are_bounded() -> None:
    event = _event("stream_done", 1)
    body = f"event: stream_done\nid: stream-1:1\ndata: {json.dumps(event)}\n\n"

    assert _parse_sse_text(body) == [event]
    assert _artifact_paths(
        {
            "resources": [
                {
                    "artifact_refs": {
                        "docx": "/artifacts/review-docs/id/file.docx",
                        "remote": "https://example.com/not-local",
                    }
                }
            ]
        }
    ) == ("/artifacts/review-docs/id/file.docx",)


def test_refresh_projection_mode_requires_visible_ready_or_blocked_state() -> None:
    assert (
        _refresh_projection_mode(
            {"resources": [{"resource_type": "review_doc"}], "blocked_resources": []},
            ("/artifacts/review-docs/id/file.docx",),
        )
        == "artifact_download"
    )
    assert (
        _refresh_projection_mode(
            {
                "resources": [],
                "blocked_resources": [{"resource_type": "review_doc"}],
            },
            (),
        )
        == "blocked_status"
    )
    with pytest.raises(ProductionCanaryError, match="local artifact projection"):
        _refresh_projection_mode(
            {"resources": [{"resource_type": "review_doc"}], "blocked_resources": []},
            (),
        )


@pytest.mark.asyncio
async def test_verify_replay_requires_exact_validated_journal_tail() -> None:
    events = _completed_events()
    expected_tail = events[2:]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Last-Event-ID"] == "stream-1:2"
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in expected_tail)
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await _verify_replay(
            client=client,
            backend_url="http://backend.test",
            events=events,
            stream_id="stream-1",
        )

    assert result == {
        "after_sequence": 2,
        "replayed_count": 2,
        "tail_matches": True,
    }


@pytest.mark.asyncio
async def test_verify_downloads_consumes_nonempty_attachment() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/artifacts/review-docs/id/file.docx"
        return httpx.Response(
            200,
            content=b"bounded-artifact",
            headers={"content-disposition": 'attachment; filename="file.docx"'},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await _verify_downloads(
            client=client,
            backend_url="http://backend.test",
            artifact_paths=("/artifacts/review-docs/id/file.docx",),
        )

    assert result == {
        "referenced_count": 1,
        "verified_count": 1,
        "attachment_header_count": 1,
        "downloaded_bytes": len(b"bounded-artifact"),
    }


async def _identity_from_payload(
    payload: dict[str, object],
    *,
    status_code: int = 200,
):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health/ready"
        return httpx.Response(status_code, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await _fetch_served_identity(
            client=client,
            backend_url="http://backend.test",
            expected_generation=_expected_generation(),
            expected_knowledge_graph_data_version="kg-2026-07-15",
            expected_knowledge_graph_artifact_fingerprint=DIGEST_A,
        )


@pytest.mark.asyncio
async def test_fetch_served_identity_binds_exact_health_ready_v3() -> None:
    identity = await _identity_from_payload(_health_ready_payload())

    assert identity.health_ready_schema_version == "health_ready_v3"
    assert identity.parent_child_generation_id == "pc-generation-1"
    assert identity.knowledge_graph_artifact_fingerprint == DIGEST_A
    assert identity.deployment_mode == "active"
    assert identity.rollout_activation_enabled is True
    assert identity.rollout_shadow_enabled is False
    assert identity.identity_fingerprint == canary.canonical_sha256(
        identity.model_dump(mode="json", exclude={"identity_fingerprint"})
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        _health_ready_payload(schema_version="health_ready_v2"),
        _health_ready_payload(deployment_mode="inactive_canary"),
        _health_ready_payload(rollout_activation_enabled=False),
        _health_ready_payload(rollout_shadow_enabled=True),
        _health_ready_payload(rollout_activation_enabled="true"),
        _health_ready_payload(parent_child_generation_id="different-generation"),
        _health_ready_payload(parent_child_generation_manifest_fingerprint=DIGEST_D),
        _health_ready_payload(knowledge_graph_data_version="different-kg"),
        _health_ready_payload(knowledge_graph_artifact_fingerprint=DIGEST_D),
        _health_ready_payload(knowledge_graph_artifact_fingerprint="A" * 64),
    ],
)
async def test_fetch_served_identity_rejects_contract_and_identity_drift(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ProductionCanaryError):
        await _identity_from_payload(payload)


@pytest.mark.asyncio
async def test_fetch_served_identity_rejects_missing_extra_and_http_failure() -> None:
    missing = _health_ready_payload()
    missing.pop("rollout_activation_enabled")
    extra = _health_ready_payload(activation_enabled=False)

    for payload in (missing, extra):
        with pytest.raises(ProductionCanaryError, match="health_ready_v3"):
            await _identity_from_payload(payload)
    with pytest.raises(ProductionCanaryError, match="request failed"):
        await _identity_from_payload(_health_ready_payload(), status_code=503)


def test_expected_generation_rejects_defaults_and_aliases() -> None:
    with pytest.raises(ValidationError):
        ProductionCanaryExpectedGenerationV1(
            parent_child_generation_id=" pc-generation-1 ",
            parent_child_generation_manifest_fingerprint=DIGEST_B,
        )
    with pytest.raises(ValidationError):
        ProductionCanaryExpectedGenerationV1(
            parent_child_generation_id="pc-generation-1",
            parent_child_generation_manifest_fingerprint="B" * 64,
        )
    with pytest.raises(ValidationError):
        ProductionCanaryExpectedGenerationV1.model_validate(
            {
                "generation_id": "pc-generation-1",
                "manifest_fingerprint": DIGEST_B,
            },
            strict=True,
        )


def test_identity_and_report_fingerprints_fail_closed() -> None:
    readiness = canary.HealthReadyV3.model_validate(
        _health_ready_payload(),
        strict=True,
    )
    identity = _served_identity(readiness)
    binding = canary.canonical_sha256(
        {
            "schema_version": "production_canary_binding_v2",
            "dataset_content_fingerprint": DIGEST_D,
            "served_identity_fingerprint": identity.identity_fingerprint,
        }
    )
    report = ProductionBrowserCanaryReportV3(
        schema_version="production_browser_canary_v3",
        created_at_utc="2026-07-16T00:00:00+00:00",
        dataset_id="smoke-v2",
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

    serialized = report.model_dump_json()
    for forbidden in (
        "query",
        "http://",
        "authorization",
        "provider_body",
        "db_uri",
    ):
        assert forbidden not in serialized.casefold()

    invalid_binding = report.model_dump(mode="python")
    invalid_binding["canary_binding_fingerprint"] = DIGEST_A
    with pytest.raises(ValidationError):
        ProductionBrowserCanaryReportV3.model_validate(invalid_binding, strict=True)

    invalid_identity = report.model_dump(mode="python")
    invalid_identity["served_identity"]["identity_fingerprint"] = DIGEST_D
    with pytest.raises(ValidationError):
        ProductionBrowserCanaryReportV3.model_validate(invalid_identity, strict=True)

    missing_identity = report.model_dump(mode="python")
    missing_identity.pop("served_identity")
    with pytest.raises(ValidationError):
        ProductionBrowserCanaryReportV3.model_validate(missing_identity, strict=True)

    extra_identity = report.model_dump(mode="python")
    extra_identity["health"] = _health_ready_payload()
    with pytest.raises(ValidationError):
        ProductionBrowserCanaryReportV3.model_validate(extra_identity, strict=True)


def test_stable_served_identity_rejects_pre_post_drift() -> None:
    before = _served_identity(
        canary.HealthReadyV3.model_validate(_health_ready_payload(), strict=True)
    )
    after = _served_identity(
        canary.HealthReadyV3.model_validate(
            _health_ready_payload(graph_version="graph-v2"),
            strict=True,
        )
    )

    with pytest.raises(ProductionCanaryError, match="changed"):
        _require_stable_served_identity(before, after)


@pytest.mark.asyncio
async def test_run_aborts_before_user_or_browser_when_readiness_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = SimpleNamespace(
        cases=tuple(range(6)),
        knowledge_graph_data_version="kg-2026-07-15",
        knowledge_graph_artifact_fingerprint=DIGEST_A,
        model_dump=lambda *, mode: {"mode": mode},
    )
    calls: list[str] = []

    async def fail_readiness(**kwargs):
        del kwargs
        raise ProductionCanaryError("health readiness request failed")

    async def forbid_provision(**kwargs):
        del kwargs
        calls.append("provision")
        raise AssertionError("user provisioning must not run")

    def forbid_browser():
        calls.append("browser")
        raise AssertionError("browser must not run")

    monkeypatch.setattr(canary, "_contained_file", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(canary, "_contained_output", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(canary, "_load_dataset", lambda path: dataset)
    monkeypatch.setattr(canary, "_fetch_served_identity", fail_readiness)
    monkeypatch.setattr(canary, "_provision_user", forbid_provision)
    monkeypatch.setattr(canary, "async_playwright", forbid_browser)
    args = argparse.Namespace(
        project_root=tmp_path,
        dataset=Path("dataset.json"),
        output_dir=Path("output"),
        frontend_url="http://frontend.test",
        backend_url="http://backend.test",
        expected_generation_id="pc-generation-1",
        expected_generation_manifest_fingerprint=DIGEST_B,
        timeout_seconds=1.0,
        headless=True,
    )

    with pytest.raises(ProductionCanaryError, match="request failed"):
        await canary._run(args)

    assert calls == []
    assert not (tmp_path / "result.json").exists()
