"""Contract tests for ordered and replay-safe agent_stream_v2 events."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.streaming import (
    AgentStreamEventDraftV2,
    AgentStreamEventV2,
    StreamContractError,
    StreamEventSequencer,
    StreamJournal,
    StreamJournalCapacityError,
    StreamJournalExpiredError,
    StreamJournalSequenceError,
    encode_sse_event,
    parse_last_event_id,
)


REQUEST_ID = "00000000-0000-4000-8000-000000000001"


def _sequencer() -> StreamEventSequencer:
    return StreamEventSequencer(
        stream_id="stream-1",
        request_id=REQUEST_ID,
        thread_id="thread-1",
    )


def test_stream_events_are_ordered_and_done_requires_terminal() -> None:
    sequencer = _sequencer()
    start = sequencer.emit("stream_start")
    terminal = sequencer.emit("qa_final", {"payload_hash": "abc"})
    done = sequencer.emit("stream_done")

    assert [start.sequence, terminal.sequence, done.sequence] == [1, 2, 3]
    assert terminal.event_id == "stream-1:2"
    assert sequencer.terminal == "qa_final"

    with pytest.raises(StreamContractError, match="after stream_done"):
        sequencer.emit("activity_update")


def test_native_draft_rejects_session_owned_and_malformed_block_events() -> None:
    with pytest.raises(ValidationError):
        AgentStreamEventDraftV2.model_validate({"type": "stream_start", "data": {}})
    with pytest.raises(ValidationError):
        AgentStreamEventDraftV2.model_validate(
            {"type": "content_block_delta", "data": {"delta": "missing identity"}}
        )

    draft = AgentStreamEventDraftV2(
        type="content_block_delta",
        data={
            "block_id": "answer",
            "block_index": 0,
            "block_type": "markdown",
            "provisional": True,
            "delta": "ok",
        },
    )
    assert draft.type == "content_block_delta"


def test_stream_rejects_done_without_terminal_and_second_terminal() -> None:
    sequencer = _sequencer()
    sequencer.emit("stream_start")
    with pytest.raises(StreamContractError, match="requires an authoritative"):
        sequencer.emit("stream_done")

    sequencer.emit("interrupt", {"interrupt_type": "plan_review"})
    with pytest.raises(StreamContractError, match="only stream_done"):
        sequencer.emit("stream_error", {"error_type": "late_error"})


def test_event_contract_rejects_unknown_fields_and_invalid_identity() -> None:
    event = _sequencer().emit("stream_start")
    payload = event.model_dump(mode="json")
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        AgentStreamEventV2.model_validate(payload)

    payload.pop("unexpected")
    payload["event_id"] = "wrong"
    with pytest.raises(ValidationError, match="event_id"):
        AgentStreamEventV2.model_validate(payload)


def test_sse_encoder_and_last_event_id_are_strict() -> None:
    event = _sequencer().emit("stream_start")
    frame = encode_sse_event(event, retry_ms=1500)
    lines = frame.strip().splitlines()

    assert lines[:3] == [
        "event: stream_start",
        "id: stream-1:1",
        "retry: 1500",
    ]
    assert json.loads(lines[3].removeprefix("data: "))["sequence"] == 1
    assert parse_last_event_id("stream-1:1", expected_stream_id="stream-1") == 1
    with pytest.raises(StreamContractError, match="does not match"):
        parse_last_event_id("other:1", expected_stream_id="stream-1")


def test_journal_replays_without_silent_eviction() -> None:
    sequencer = _sequencer()
    journal = StreamJournal(
        stream_id="stream-1",
        max_events=4,
        max_bytes=10000,
        ttl_seconds=60,
    )
    first = sequencer.emit("stream_start")
    second = sequencer.emit("activity_update", {"kind": "node"})
    journal.append(first)
    journal.append(second)

    assert [event.sequence for event in journal.after(1)] == [2]
    with pytest.raises(StreamJournalCapacityError, match="event limit"):
        journal.append(sequencer.emit("activity_update", {"kind": "node"}))
    with pytest.raises(StreamJournalSequenceError, match="ahead"):
        journal.after(3)


def test_journal_expiry_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    journal = StreamJournal(
        stream_id="stream-1",
        max_events=4,
        max_bytes=10000,
        ttl_seconds=1,
    )
    journal.seal(completed_monotonic=100.0)
    monkeypatch.setattr("src.streaming.journal.time.monotonic", lambda: 102.0)
    with pytest.raises(StreamJournalExpiredError, match="expired"):
        journal.after(0)
