"""Temporary migration adapter from legacy internal payloads to agent_stream_v2."""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any, cast

from src.streaming.contracts import AgentStreamEventType, StreamEventSequencer
from src.streaming.journal import StreamJournal
from src.streaming.sse import encode_sse_event


async def adapt_legacy_sse_stream(
    source: AsyncIterable[str],
    *,
    stream_id: str,
    request_id: str,
    thread_id: str,
    retry_ms: int,
    journal: StreamJournal | None = None,
) -> AsyncIterator[str]:
    """Expose one strict V2 stream while legacy producers are being replaced."""

    sequencer = StreamEventSequencer(
        stream_id=stream_id,
        request_id=request_id,
        thread_id=thread_id,
    )
    open_blocks: set[str] = set()

    start = sequencer.emit("stream_start", {"retry_ms": retry_ms})
    _append(journal, start)
    yield encode_sse_event(start, retry_ms=retry_ms)

    async for frame in source:
        payload = _legacy_payload(frame)
        if payload is None:
            continue
        legacy_type = str(payload.get("type") or "")
        if sequencer.terminal is not None and legacy_type != "done":
            continue

        if legacy_type in {"token", "text"}:
            block_id = f"{request_id}:assistant"
            if block_id not in open_blocks:
                open_blocks.add(block_id)
                async for encoded in _emit(
                    sequencer,
                    "content_block_start",
                    _block_data(block_id, 0, provisional=True),
                    journal,
                ):
                    yield encoded
            async for encoded in _emit(
                sequencer,
                "content_block_delta",
                _block_data(
                    block_id,
                    0,
                    provisional=True,
                    delta=str(payload.get("content") or ""),
                ),
                journal,
            ):
                yield encoded
            continue

        if legacy_type == "qa_provisional_start":
            block_id = f"{request_id}:qa-answer"
            if block_id not in open_blocks:
                open_blocks.add(block_id)
                async for encoded in _emit(
                    sequencer,
                    "content_block_start",
                    _block_data(block_id, 0, provisional=True),
                    journal,
                ):
                    yield encoded
            continue

        if legacy_type == "qa_provisional_delta":
            block_id = f"{request_id}:qa-answer"
            if block_id not in open_blocks:
                open_blocks.add(block_id)
                async for encoded in _emit(
                    sequencer,
                    "content_block_start",
                    _block_data(block_id, 0, provisional=True),
                    journal,
                ):
                    yield encoded
            async for encoded in _emit(
                sequencer,
                "content_block_delta",
                _block_data(
                    block_id,
                    0,
                    provisional=True,
                    delta=str(payload.get("delta") or ""),
                ),
                journal,
            ):
                yield encoded
            continue

        if legacy_type in {"qa_provisional_stop", "qa_provisional_reset"}:
            block_id = f"{request_id}:qa-answer"
            if block_id in open_blocks:
                open_blocks.remove(block_id)
                async for encoded in _emit(
                    sequencer,
                    "content_block_stop",
                    {
                        **_block_data(block_id, 0, provisional=True),
                        "reset": legacy_type == "qa_provisional_reset",
                        "reason": str(payload.get("reason") or ""),
                    },
                    journal,
                ):
                    yield encoded
            continue

        terminal_type = _terminal_type(payload)
        if terminal_type is not None:
            async for encoded in _close_blocks(sequencer, open_blocks, journal):
                yield encoded
            data = {key: value for key, value in payload.items() if key != "type"}
            if legacy_type == "resource_final_diagnostic":
                data = {
                    "error_type": "completed_without_resource",
                    "message": "Run completed without an authoritative resource payload",
                    **data,
                }
            async for encoded in _emit(sequencer, terminal_type, data, journal):
                yield encoded
            continue

        if legacy_type == "done":
            if sequencer.terminal is None:
                async for encoded in _close_blocks(sequencer, open_blocks, journal):
                    yield encoded
                async for encoded in _emit(
                    sequencer,
                    "stream_error",
                    {
                        "error_type": "missing_authoritative_terminal",
                        "message": "Legacy stream ended without an authoritative terminal",
                    },
                    journal,
                ):
                    yield encoded
            async for encoded in _emit(
                sequencer,
                "stream_done",
                {"terminal_type": sequencer.terminal},
                journal,
            ):
                yield encoded
            return

        event_type = _progress_type(legacy_type)
        data = {
            "kind": legacy_type or "legacy_event",
            "payload": {key: value for key, value in payload.items() if key != "type"},
        }
        async for encoded in _emit(sequencer, event_type, data, journal):
            yield encoded

    if sequencer.terminal is None:
        async for encoded in _close_blocks(sequencer, open_blocks, journal):
            yield encoded
        async for encoded in _emit(
            sequencer,
            "stream_error",
            {
                "error_type": "stream_ended_without_terminal",
                "message": "Stream source ended before an authoritative terminal",
            },
            journal,
        ):
            yield encoded
    async for encoded in _emit(
        sequencer,
        "stream_done",
        {"terminal_type": sequencer.terminal},
        journal,
    ):
        yield encoded


async def _emit(
    sequencer: StreamEventSequencer,
    event_type: AgentStreamEventType,
    data: dict[str, Any],
    journal: StreamJournal | None,
) -> AsyncIterator[str]:
    event = sequencer.emit(event_type, data)
    _append(journal, event)
    yield encode_sse_event(event)


async def _close_blocks(
    sequencer: StreamEventSequencer,
    open_blocks: set[str],
    journal: StreamJournal | None,
) -> AsyncIterator[str]:
    for index, block_id in enumerate(sorted(open_blocks)):
        async for encoded in _emit(
            sequencer,
            "content_block_stop",
            _block_data(block_id, index, provisional=True),
            journal,
        ):
            yield encoded
    open_blocks.clear()


def _append(journal: StreamJournal | None, event: Any) -> None:
    if journal is not None:
        journal.append(event)


def _legacy_payload(frame: str) -> dict[str, Any] | None:
    text = str(frame or "").strip()
    data_lines = [
        line[5:].lstrip() for line in text.splitlines() if line.startswith("data:")
    ]
    if not data_lines:
        return None
    payload = json.loads("\n".join(data_lines))
    if not isinstance(payload, dict):
        raise ValueError("legacy SSE data must be a JSON object")
    return payload


def _block_data(
    block_id: str,
    block_index: int,
    *,
    provisional: bool,
    delta: str = "",
) -> dict[str, Any]:
    return {
        "block_id": block_id,
        "block_index": block_index,
        "block_type": "markdown",
        "provisional": provisional,
        "delta": delta,
    }


def _terminal_type(payload: dict[str, Any]) -> AgentStreamEventType | None:
    legacy_type = str(payload.get("type") or "")
    if legacy_type in {"qa_final", "resource_final", "interrupt"}:
        return cast(AgentStreamEventType, legacy_type)
    if legacy_type == "error" or legacy_type == "resource_final_diagnostic":
        return "stream_error"
    if legacy_type == "run_status" and payload.get("run_status") == "stopped":
        return "stopped"
    return None


def _progress_type(legacy_type: str) -> AgentStreamEventType:
    if legacy_type in {"provider_retry", "resource_subnode"}:
        return "tool_progress"
    if legacy_type in {"resource_generation", "artifact"}:
        return "artifact_progress"
    return "activity_update"


__all__ = ["adapt_legacy_sse_stream"]
