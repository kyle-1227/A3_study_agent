"""Strict content-free evidence orchestration trace tests."""

from __future__ import annotations

import json
import logging

import pytest
from pydantic import ValidationError

from src.observability.a3_trace import reset_trace_event_sink, set_trace_event_sink
from src.observability.activity import activity_from_trace_event
from src.observability.evidence_trace import (
    EVIDENCE_TRACE_SCHEMA_VERSION,
    EVIDENCE_TRACE_STAGES,
    emit_evidence_trace,
    validate_evidence_trace_event,
)
from src.observability.node_registry import get_node_runtime_metadata

NOW = "2026-07-11T00:00:00+00:00"
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _valid_events() -> list[dict[str, object]]:
    base = {"schema_version": EVIDENCE_TRACE_SCHEMA_VERSION}
    return [
        {
            **base,
            "stage": "evidence_orchestration.plan.accepted",
            "orchestration_fingerprint": HASH_A,
            "profile_fingerprint": HASH_B,
            "requirement_count": 4,
            "resource_count": 2,
            "subject_count": 2,
            "budget_max_rounds": 2,
            "budget_max_tasks": 18,
        },
        {
            **base,
            "stage": "evidence_orchestration.round.started",
            "round_index": 0,
            "task_count": 4,
            "local_task_count": 2,
            "web_task_count": 2,
            "budget_used_tasks": 4,
            "budget_remaining_tasks": 14,
        },
        {
            **base,
            "stage": "evidence_orchestration.source.completed",
            "round_index": 0,
            "source": "local",
            "status": "completed",
            "task_count": 2,
            "query_batch_fingerprint": HASH_A,
            "candidate_count": 5,
            "latency_ms": 120,
        },
        {
            **base,
            "stage": "evidence_orchestration.source.empty",
            "round_index": 0,
            "source": "web",
            "status": "empty",
            "task_count": 2,
            "query_batch_fingerprint": HASH_B,
            "latency_ms": 100,
            "reason_code": "no_candidates",
        },
        {
            **base,
            "stage": "evidence_orchestration.source.failed",
            "round_index": 1,
            "source": "web",
            "status": "failed",
            "task_count": 1,
            "query_batch_fingerprint": HASH_C,
            "latency_ms": 80,
            "reason_code": "provider_error",
            "error_type": "ProviderTransportError",
        },
        {
            **base,
            "stage": "evidence_orchestration.round.merged",
            "round_index": 0,
            "local_candidate_count": 3,
            "web_candidate_count": 2,
            "deduplicated_count": 1,
            "ledger_count": 4,
            "ledger_fingerprint": HASH_A,
        },
        {
            **base,
            "stage": "evidence_orchestration.coverage.judged",
            "round_index": 0,
            "requirement_count": 4,
            "complete_count": 2,
            "partial_count": 1,
            "missing_count": 1,
            "accepted_evidence_count": 4,
            "coverage_fingerprint": HASH_B,
        },
        {
            **base,
            "stage": "evidence_orchestration.progress.evaluated",
            "round_index": 1,
            "previous_complete_count": 2,
            "current_complete_count": 3,
            "previous_partial_count": 1,
            "current_partial_count": 1,
            "previous_missing_count": 1,
            "current_missing_count": 0,
            "new_accepted_evidence_count": 1,
            "progressed": True,
            "consecutive_no_progress_rounds": 0,
        },
        {
            **base,
            "stage": "evidence_orchestration.route.decided",
            "round_index": 0,
            "status": "repair",
            "reason_code": "repair_required",
            "next_local_task_count": 1,
            "next_web_task_count": 1,
            "budget_remaining_rounds": 1,
            "budget_remaining_tasks": 12,
        },
        {
            **base,
            "stage": "evidence_orchestration.resource.assigned",
            "round_index": 1,
            "resource_type": "quiz",
            "status": "ready",
            "requirement_count": 2,
            "assigned_evidence_count": 3,
            "missing_requirement_count": 0,
            "assignment_fingerprint": HASH_C,
        },
        {
            **base,
            "stage": "evidence_orchestration.terminal",
            "orchestration_fingerprint": HASH_A,
            "status": "partial_resources_ready",
            "rounds_completed": 2,
            "ready_resource_count": 1,
            "blocked_resource_count": 1,
            "total_search_tasks": 6,
            "ledger_count": 8,
            "reason_code": "partial_resources_ready",
        },
        {
            **base,
            "stage": "evidence_orchestration.failed",
            "status": "failed",
            "round_index": 1,
            "source": "judge",
            "error_type": "EvidenceBusinessValidationError",
            "reason_code": "coverage_contract_invalid",
            "budget_used_tasks": 6,
            "budget_remaining_tasks": 12,
        },
    ]


def test_exact_evidence_trace_family_is_strictly_validated():
    events = _valid_events()

    assert {event["stage"] for event in events} == EVIDENCE_TRACE_STAGES
    for event in events:
        validated = validate_evidence_trace_event(event)
        assert validated.stage == event["stage"]
        assert validated.schema_version == EVIDENCE_TRACE_SCHEMA_VERSION


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("query", "raw search query"),
        ("url", "https://example.invalid/private"),
        ("evidence_body", "raw evidence text"),
        ("provider_body", {"response": "raw"}),
        ("headers", {"Authorization": "Bearer secret"}),
        ("exception_message", "full provider exception"),
    ],
)
def test_evidence_trace_rejects_forbidden_fields(field: str, value: object):
    event = {**_valid_events()[0], field: value}

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_evidence_trace_event(event)


@pytest.mark.parametrize(
    "unsafe_reason",
    [
        "https://example.invalid/private",
        "raw search query",
        "sk-secretvalue123456",
    ],
)
def test_evidence_trace_rejects_unsafe_values_in_safe_fields(unsafe_reason: str):
    event = {**_valid_events()[3], "reason_code": unsafe_reason}

    with pytest.raises(ValidationError):
        validate_evidence_trace_event(event)


def test_invalid_evidence_trace_has_no_sink_side_effect(monkeypatch):
    monkeypatch.delenv("LOG_A3_TRACE", raising=False)
    sink: list[dict[str, object]] = []
    token = set_trace_event_sink(sink)
    try:
        event = {**_valid_events()[0], "query": "private raw query"}
        with pytest.raises(ValidationError):
            emit_evidence_trace(logging.getLogger(__name__), event)
    finally:
        reset_trace_event_sink(token)

    assert sink == []


def test_emitted_evidence_trace_is_content_free(monkeypatch):
    monkeypatch.delenv("LOG_A3_TRACE", raising=False)
    sink: list[dict[str, object]] = []
    token = set_trace_event_sink(sink)
    try:
        emit_evidence_trace(
            logging.getLogger(__name__),
            _valid_events()[2],
            state={
                "request_id": "request-1",
                "session_id": "session-1",
                "thread_id": "thread-1",
                "query": "private raw query",
                "url": "https://example.invalid/private",
                "headers": {"Authorization": "Bearer secret-token"},
                "provider_body": "private provider response",
            },
        )
    finally:
        reset_trace_event_sink(token)

    assert len(sink) == 1
    serialized = json.dumps(sink[0], ensure_ascii=False, sort_keys=True)
    assert sink[0]["stage"] == "evidence_orchestration.source.completed"
    assert sink[0]["schema_version"] == EVIDENCE_TRACE_SCHEMA_VERSION
    assert "private raw query" not in serialized
    assert "example.invalid" not in serialized
    assert "Authorization" not in serialized
    assert "secret-token" not in serialized
    assert "provider response" not in serialized


def test_evidence_trace_maps_to_evidence_progress_activity():
    event = {
        **_valid_events()[2],
        "request_id": "request-1",
        "session_id": "session-1",
        "thread_id": "thread-1",
    }

    activity = activity_from_trace_event(
        event,
        thread_id="thread-1",
        request_id="request-1",
        sequence=1,
        now=NOW,
    )

    assert activity is not None
    assert activity.kind == "evidence_progress"
    assert activity.status == "completed"
    assert activity.node == "local_rag_search_batch"
    assert activity.duration_ms == 120
    assert activity.safe_details["query_batch_fingerprint"] == HASH_A
    assert "query" not in activity.safe_details


def test_evidence_failure_activity_preserves_the_failing_node():
    event = {
        **_valid_events()[-1],
        "request_id": "request-1",
        "session_id": "session-1",
        "thread_id": "thread-1",
    }

    activity = activity_from_trace_event(
        event,
        thread_id="thread-1",
        request_id="request-1",
        sequence=2,
        now=NOW,
    )

    assert activity is not None
    assert activity.status == "failed"
    assert activity.node == "requirement_evidence_judge"
    assert activity.safe_details["source"] == "judge"


def test_evidence_activity_adapter_does_not_silently_drop_unsafe_fields():
    event = {
        **_valid_events()[6],
        "request_id": "request-1",
        "session_id": "session-1",
        "thread_id": "thread-1",
        "raw_output": "private evidence body",
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        activity_from_trace_event(
            event,
            thread_id="thread-1",
            request_id="request-1",
            sequence=1,
            now=NOW,
        )


def test_evidence_orchestration_nodes_are_registered():
    expected = {
        "rag_generation_router",
        "resource_evidence_planner",
        "retrieval_round_router",
        "local_rag_search_batch",
        "web_research_search_batch",
        "retrieval_round_merge",
        "requirement_evidence_judge",
        "evidence_repair_planner",
        "parent_child_parent_hydration",
        "resource_evidence_assignment",
    }

    assert all(get_node_runtime_metadata(node_id) is not None for node_id in expected)
