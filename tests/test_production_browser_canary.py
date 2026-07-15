from __future__ import annotations

import json

import httpx
import pytest

from scripts.run_production_browser_canary import (
    ProductionCanaryError,
    _artifact_paths,
    _parse_sse_text,
    _refresh_projection_mode,
    _safe_terminal_projection,
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

    result = _safe_terminal_projection(
        payload,
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
    invalid_status["terminal_status"] = "all_resources_ready"

    with pytest.raises(ProductionCanaryError, match="resource_final_v3"):
        _safe_terminal_projection(
            invalid_status,
            expected_resource_types=("review_doc",),
            expected_request_id=REQUEST_ID,
            expected_thread_id=THREAD_ID,
        )
    with pytest.raises(ProductionCanaryError, match="identity"):
        _safe_terminal_projection(
            _resource_final(),
            expected_resource_types=("review_doc",),
            expected_request_id=RESUME_REQUEST_ID,
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
