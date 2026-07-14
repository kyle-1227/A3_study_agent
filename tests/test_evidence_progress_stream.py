from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from src.observability.a3_trace import reset_trace_event_sink, set_trace_event_sink
from src.observability.evidence_trace import emit_evidence_trace
from src.streaming.contracts import (
    AgentStreamEventDraftV2,
    AgentStreamEventV2,
    StreamEventSequencer,
)
from src.streaming.evidence_progress import (
    EvidenceProgressV1,
    build_evidence_progress,
    reset_evidence_progress_sink,
    set_evidence_progress_sink,
)
from src.streaming.journal import StreamJournal
from src.streaming.sse import parse_last_event_id

REQUEST_ID = "00000000-0000-4000-8000-000000000001"
THREAD_ID = "thread-1"
SHA = "a" * 64


def _round_started() -> dict[str, object]:
    return {
        "schema_version": "evidence_orchestration_trace_v1",
        "stage": "evidence_orchestration.round.started",
        "round_index": 1,
        "task_count": 3,
        "local_task_count": 1,
        "web_task_count": 2,
        "budget_used_tasks": 6,
        "budget_remaining_tasks": 12,
    }


def _round_merged() -> dict[str, object]:
    return {
        "schema_version": "evidence_orchestration_trace_v1",
        "stage": "evidence_orchestration.round.merged",
        "round_index": 1,
        "local_candidate_count": 2,
        "web_candidate_count": 4,
        "deduplicated_count": 1,
        "ledger_count": 5,
        "ledger_fingerprint": SHA,
    }


def _source_completed() -> dict[str, object]:
    return {
        "schema_version": "evidence_orchestration_trace_v1",
        "stage": "evidence_orchestration.source.completed",
        "round_index": 1,
        "source": "web",
        "status": "completed",
        "task_count": 2,
        "query_batch_fingerprint": SHA,
        "candidate_count": 4,
        "latency_ms": 120,
    }


def test_round_start_and_merge_share_stable_progress_identity() -> None:
    started = build_evidence_progress(
        _round_started(),
        request_id=REQUEST_ID,
        thread_id=THREAD_ID,
    )
    merged = build_evidence_progress(
        _round_merged(),
        request_id=REQUEST_ID,
        thread_id=THREAD_ID,
    )

    assert started.progress_id == merged.progress_id
    assert started.lifecycle_key == merged.lifecycle_key == "round:1"
    assert started.phase_status == "running"
    assert merged.phase_status == "completed"


def test_public_projection_excludes_query_fingerprint_and_rejects_extra() -> None:
    progress = build_evidence_progress(
        _source_completed(),
        request_id=REQUEST_ID,
        thread_id=THREAD_ID,
    )
    payload = progress.model_dump(mode="json")

    assert "query_batch_fingerprint" not in payload["details"]
    payload["details"]["query"] = "private query"
    with pytest.raises(ValidationError):
        EvidenceProgressV1.model_validate(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.update(
                request_id="AAAAAAAA-0000-4000-8000-000000000001"
            ),
            "canonical UUID",
        ),
        (
            lambda payload: payload.update(phase_status="running"),
            "phase_status",
        ),
        (
            lambda payload: payload["details"].update(status="empty"),
            "status",
        ),
        (
            lambda payload: payload["details"].update(task_count=True),
            "task_count",
        ),
    ],
)
def test_progress_contract_rejects_identity_and_stage_drift(
    mutate: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    payload = build_evidence_progress(
        _source_completed(),
        request_id="aaaaaaaa-0000-4000-8000-000000000001",
        thread_id=THREAD_ID,
    ).model_dump(mode="python")
    mutate(payload)
    with pytest.raises(ValidationError, match=message):
        EvidenceProgressV1.model_validate(payload)


def test_reliable_sink_failure_propagates_before_best_effort_trace() -> None:
    def fail_sink(_: EvidenceProgressV1) -> None:
        raise RuntimeError("sink unavailable")

    token = set_evidence_progress_sink(fail_sink)
    try:
        with pytest.raises(RuntimeError, match="sink unavailable"):
            emit_evidence_trace(
                __import__("logging").getLogger(__name__),
                _source_completed(),
                state={"request_id": REQUEST_ID, "thread_id": THREAD_ID},
            )
    finally:
        reset_evidence_progress_sink(token)


def test_best_effort_trace_failure_does_not_drop_public_progress() -> None:
    class BrokenTraceSink:
        def append(self, _: object) -> None:
            raise RuntimeError("best effort trace sink failed")

    published: list[EvidenceProgressV1] = []
    progress_token = set_evidence_progress_sink(published.append)
    trace_token = set_trace_event_sink(BrokenTraceSink())
    try:
        emit_evidence_trace(
            __import__("logging").getLogger(__name__),
            _source_completed(),
            state={"request_id": REQUEST_ID, "thread_id": THREAD_ID},
        )
    finally:
        reset_trace_event_sink(trace_token)
        reset_evidence_progress_sink(progress_token)

    assert len(published) == 1
    assert published[0].details.stage == "evidence_orchestration.source.completed"


def test_agent_stream_validates_progress_binding_and_journal_replay() -> None:
    progress = build_evidence_progress(
        _source_completed(),
        request_id=REQUEST_ID,
        thread_id=THREAD_ID,
    )
    draft = AgentStreamEventDraftV2(
        type="evidence_progress",
        data=progress.model_dump(mode="json"),
    )
    sequencer = StreamEventSequencer(
        stream_id="stream-1",
        request_id=REQUEST_ID,
        thread_id=THREAD_ID,
    )
    start = sequencer.emit("stream_start", {})
    event = sequencer.emit(draft.type, draft.data)
    failure = sequencer.emit(
        "stream_error",
        {"error_type": "fixture", "message": "fixture", "recoverable": False},
    )
    done = sequencer.emit("stream_done", {"terminal_type": "stream_error"})
    journal = StreamJournal(
        stream_id="stream-1",
        max_events=6,
        max_bytes=20_000,
        ttl_seconds=60,
    )
    for item in (start, event, failure, done):
        journal.append(item)

    before_progress = parse_last_event_id("stream-1:1", expected_stream_id="stream-1")
    replayed = journal.after(before_progress)[0]
    assert replayed.event_id == event.event_id
    assert replayed.data == event.data
    after_progress = parse_last_event_id(event.event_id, expected_stream_id="stream-1")
    assert [item.type for item in journal.after(after_progress)] == [
        "stream_error",
        "stream_done",
    ]

    invalid = event.model_dump(mode="python")
    invalid["thread_id"] = "thread-2"
    with pytest.raises(ValidationError, match="thread_id"):
        AgentStreamEventV2.model_validate(invalid)
