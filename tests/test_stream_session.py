"""Producer/subscriber tests for disconnect-safe stream sessions."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.streaming.journal import StreamJournalSequenceError
from src.streaming.session import (
    StreamSessionConflictError,
    StreamSessionExpiredError,
    StreamSessionManager,
)
from src.streaming.settings import StreamingRuntimeConfig


REQUEST_ID = "00000000-0000-4000-8000-000000000001"
SECOND_REQUEST_ID = "00000000-0000-4000-8000-000000000002"
OPERATION = "new_request"
FINGERPRINT = "request-fingerprint-1"


async def _source():
    for payload in [
        {"type": "qa_provisional_start"},
        {"type": "qa_provisional_delta", "delta": "answer"},
        {"type": "qa_final", "payload_hash": "abc"},
        {"type": "done"},
    ]:
        await asyncio.sleep(0)
        yield f"data: {json.dumps(payload)}\n\n"


async def _broken_source():
    yield f"data: {json.dumps({'type': 'node_event', 'node': 'qa'})}\n\n"
    raise RuntimeError("private provider failure detail")


async def _blocking_source(release: asyncio.Event):
    await release.wait()
    async for frame in _source():
        yield frame


def _config(
    *,
    ttl_seconds: int = 60,
    max_events: int = 20,
    max_bytes: int = 100000,
) -> StreamingRuntimeConfig:
    return StreamingRuntimeConfig(
        retry_ms=1000,
        journal_max_events=max_events,
        journal_max_bytes=max_bytes,
        journal_ttl_seconds=ttl_seconds,
    )


def _sequence(frame: str) -> int:
    return _payload(frame)["sequence"]


def _payload(frame: str) -> dict:
    data = next(line for line in frame.splitlines() if line.startswith("data: "))
    return json.loads(data.removeprefix("data: "))


async def _create(
    manager: StreamSessionManager,
    *,
    stream_id: str,
    request_id: str = REQUEST_ID,
    thread_id: str = "thread-1",
    operation: str = OPERATION,
    request_fingerprint: str = FINGERPRINT,
    source=None,
):
    return await manager.create(
        stream_id=stream_id,
        request_id=request_id,
        thread_id=thread_id,
        operation=operation,
        request_fingerprint=request_fingerprint,
        source=_source() if source is None else source,
    )


@pytest.mark.anyio
async def test_second_subscriber_replays_from_last_event_id() -> None:
    manager = StreamSessionManager(_config())
    session = await _create(manager, stream_id="stream-1")
    first = [frame async for frame in session.subscribe(after_sequence=0)]
    replay = [frame async for frame in session.subscribe(after_sequence=3)]

    assert [_sequence(frame) for frame in first] == list(range(1, len(first) + 1))
    assert [_sequence(frame) for frame in replay] == list(range(4, len(first) + 1))


@pytest.mark.anyio
async def test_duplicate_request_id_attaches_existing_session() -> None:
    manager = StreamSessionManager(_config())
    first = await _create(manager, stream_id="stream-1")
    second = await _create(manager, stream_id="stream-2")
    assert second is first


@pytest.mark.anyio
async def test_duplicate_request_rejects_operation_or_payload_drift() -> None:
    manager = StreamSessionManager(_config())
    await _create(manager, stream_id="stream-1")

    with pytest.raises(StreamSessionConflictError, match="another operation"):
        await _create(manager, stream_id="stream-2", operation="resume")
    with pytest.raises(StreamSessionConflictError, match="payload does not match"):
        await _create(
            manager,
            stream_id="stream-3",
            request_fingerprint="different-fingerprint",
        )


@pytest.mark.anyio
async def test_thread_rejects_a_second_active_request() -> None:
    manager = StreamSessionManager(_config())
    release = asyncio.Event()
    first = await _create(
        manager,
        stream_id="stream-1",
        source=_blocking_source(release),
    )

    with pytest.raises(StreamSessionConflictError, match="active stream"):
        await _create(
            manager,
            stream_id="stream-2",
            request_id=SECOND_REQUEST_ID,
            request_fingerprint="request-fingerprint-2",
        )

    release.set()
    frames = [frame async for frame in first.subscribe(after_sequence=0)]
    assert _payload(frames[-1])["type"] == "stream_done"


@pytest.mark.anyio
async def test_source_failure_is_a_replayable_safe_terminal() -> None:
    manager = StreamSessionManager(_config())
    session = await _create(
        manager,
        stream_id="stream-1",
        source=_broken_source(),
    )
    payloads = [_payload(frame) async for frame in session.subscribe(after_sequence=0)]

    assert [payload["type"] for payload in payloads][-2:] == [
        "stream_error",
        "stream_done",
    ]
    assert payloads[-2]["data"]["error_type"] == "stream_producer_failed"
    assert "private provider failure detail" not in json.dumps(payloads)


@pytest.mark.anyio
async def test_failure_after_authoritative_terminal_only_appends_done() -> None:
    async def terminal_then_fail():
        yield f"data: {json.dumps({'type': 'qa_final', 'payload_hash': 'abc'})}\n\n"
        raise RuntimeError("failure after final")

    manager = StreamSessionManager(_config())
    session = await _create(
        manager,
        stream_id="stream-1",
        source=terminal_then_fail(),
    )
    payloads = [_payload(frame) async for frame in session.subscribe(after_sequence=0)]

    assert [payload["type"] for payload in payloads] == [
        "stream_start",
        "qa_final",
        "stream_done",
    ]
    assert payloads[-1]["data"]["terminal_type"] == "qa_final"


@pytest.mark.anyio
async def test_capacity_failure_reserves_error_and_done_events() -> None:
    async def many_events():
        for index in range(5):
            yield (
                "data: "
                + json.dumps({"type": "node_event", "node": str(index)})
                + "\n\n"
            )

    manager = StreamSessionManager(_config(max_events=4))
    session = await _create(
        manager,
        stream_id="stream-1",
        source=many_events(),
    )
    payloads = [_payload(frame) async for frame in session.subscribe(after_sequence=0)]

    assert [payload["type"] for payload in payloads] == [
        "stream_start",
        "activity_update",
        "stream_error",
        "stream_done",
    ]
    assert payloads[-2]["data"]["error_type"] == "stream_event_log_capacity_exhausted"


@pytest.mark.anyio
async def test_byte_capacity_failure_reserves_error_and_done_events() -> None:
    async def oversized_event():
        yield (
            "data: "
            + json.dumps({"type": "node_event", "details": "x" * 2000})
            + "\n\n"
        )

    manager = StreamSessionManager(_config(max_bytes=9000))
    session = await _create(
        manager,
        stream_id="stream-1",
        source=oversized_event(),
    )
    payloads = [_payload(frame) async for frame in session.subscribe(after_sequence=0)]

    assert [payload["type"] for payload in payloads] == [
        "stream_start",
        "stream_error",
        "stream_done",
    ]
    assert payloads[-2]["data"]["error_type"] == "stream_event_log_capacity_exhausted"


@pytest.mark.anyio
async def test_active_stream_does_not_expire(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 100.0}
    monkeypatch.setattr("src.streaming.journal.time.monotonic", lambda: clock["now"])
    manager = StreamSessionManager(_config(ttl_seconds=1))
    release = asyncio.Event()
    session = await _create(
        manager,
        stream_id="stream-1",
        source=_blocking_source(release),
    )
    await asyncio.sleep(0)

    clock["now"] = 1000.0
    assert await manager.get("stream-1") is session

    release.set()
    frames = [frame async for frame in session.subscribe(after_sequence=0)]
    assert _payload(frames[-1])["type"] == "stream_done"


@pytest.mark.anyio
async def test_expired_request_id_is_a_tombstone_not_a_new_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 100.0}
    monkeypatch.setattr("src.streaming.journal.time.monotonic", lambda: clock["now"])
    executions: list[str] = []

    async def counted_source():
        executions.append("executed")
        async for frame in _source():
            yield frame

    manager = StreamSessionManager(_config(ttl_seconds=1))
    session = await _create(
        manager,
        stream_id="stream-1",
        source=counted_source(),
    )
    _ = [frame async for frame in session.subscribe(after_sequence=0)]
    clock["now"] = 102.0

    with pytest.raises(StreamSessionExpiredError, match="expired stream"):
        await _create(
            manager,
            stream_id="stream-2",
            source=counted_source(),
        )
    with pytest.raises(StreamSessionExpiredError, match="expired"):
        await manager.get("stream-1")
    assert executions == ["executed"]


@pytest.mark.anyio
async def test_replay_gap_is_rejected_before_iterator_is_returned() -> None:
    manager = StreamSessionManager(_config())
    session = await _create(manager, stream_id="stream-1")
    _ = [frame async for frame in session.subscribe(after_sequence=0)]

    with pytest.raises(StreamJournalSequenceError, match="ahead"):
        session.subscribe(after_sequence=999)


@pytest.mark.anyio
async def test_reconnect_route_rejects_gap_before_streaming_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import reconnect_stream_endpoint

    manager = StreamSessionManager(_config())
    session = await _create(manager, stream_id="stream-1")
    _ = [frame async for frame in session.subscribe(after_sequence=0)]
    monkeypatch.setattr("app.stream_session_manager", manager)
    request = SimpleNamespace(headers={"last-event-id": "stream-1:999"})

    with pytest.raises(HTTPException) as exc_info:
        await reconnect_stream_endpoint("stream-1", request)
    assert exc_info.value.status_code == 409
    assert "ahead" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_reconnect_route_returns_gone_for_expired_journal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import reconnect_stream_endpoint

    clock = {"now": 100.0}
    monkeypatch.setattr("src.streaming.journal.time.monotonic", lambda: clock["now"])
    manager = StreamSessionManager(_config(ttl_seconds=1))
    session = await _create(manager, stream_id="stream-1")
    _ = [frame async for frame in session.subscribe(after_sequence=0)]
    clock["now"] = 102.0
    monkeypatch.setattr("app.stream_session_manager", manager)
    request = SimpleNamespace(headers={"last-event-id": "stream-1:1"})

    with pytest.raises(HTTPException) as exc_info:
        await reconnect_stream_endpoint("stream-1", request)
    assert exc_info.value.status_code == 410
    assert exc_info.value.detail == "stream_session_expired"
