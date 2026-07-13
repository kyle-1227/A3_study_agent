"""Migration adapter tests proving the external stream is V2-only."""

from __future__ import annotations

import json

import pytest

from src.streaming.adapter import adapt_legacy_sse_stream


REQUEST_ID = "00000000-0000-4000-8000-000000000001"


async def _legacy(payloads: list[dict]):
    for payload in payloads:
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _payload(frame: str) -> dict:
    data_line = next(line for line in frame.splitlines() if line.startswith("data: "))
    return json.loads(data_line.removeprefix("data: "))


@pytest.mark.anyio
async def test_adapter_maps_qa_provisional_and_final_without_raw_json() -> None:
    frames = [
        frame
        async for frame in adapt_legacy_sse_stream(
            _legacy(
                [
                    {"type": "qa_provisional_start"},
                    {"type": "qa_provisional_delta", "delta": "你好"},
                    {"type": "qa_provisional_stop"},
                    {
                        "type": "qa_final",
                        "payload_hash": "abc",
                        "response": {"answer": "你好"},
                    },
                    {"type": "done"},
                ]
            ),
            stream_id="stream-1",
            request_id=REQUEST_ID,
            thread_id="thread-1",
            retry_ms=1500,
        )
    ]
    payloads = [_payload(frame) for frame in frames]

    assert [item["type"] for item in payloads] == [
        "stream_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "qa_final",
        "stream_done",
    ]
    assert payloads[2]["data"]["delta"] == "你好"
    assert [item["sequence"] for item in payloads] == list(range(1, len(payloads) + 1))


@pytest.mark.anyio
async def test_adapter_fails_closed_when_legacy_done_has_no_terminal() -> None:
    payloads = [
        _payload(frame)
        async for frame in adapt_legacy_sse_stream(
            _legacy([{"type": "token", "content": "partial"}, {"type": "done"}]),
            stream_id="stream-1",
            request_id=REQUEST_ID,
            thread_id="thread-1",
            retry_ms=1500,
        )
    ]

    assert [item["type"] for item in payloads][-2:] == [
        "stream_error",
        "stream_done",
    ]
    assert payloads[-2]["data"]["error_type"] == "missing_authoritative_terminal"


@pytest.mark.anyio
async def test_adapter_maps_completed_without_resource_to_error_terminal() -> None:
    payloads = [
        _payload(frame)
        async for frame in adapt_legacy_sse_stream(
            _legacy(
                [
                    {
                        "type": "resource_final_diagnostic",
                        "status": "completed_without_resource",
                    },
                    {"type": "done"},
                ]
            ),
            stream_id="stream-1",
            request_id=REQUEST_ID,
            thread_id="thread-1",
            retry_ms=1500,
        )
    ]
    assert [item["type"] for item in payloads] == [
        "stream_start",
        "stream_error",
        "stream_done",
    ]
