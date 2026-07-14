"""Durable, content-free assessment attempt journal tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable

import pytest
from pydantic import ValidationError

from src.assessment.attempt_contracts import (
    AssessmentAttemptV1,
    AssessmentFinalV1,
    build_assessment_final_v1,
    stable_assessment_attempt_hash,
)
from src.assessment.attempt_journal import (
    AssessmentAttemptJournalError,
    AssessmentAttemptJournalV1,
    AssessmentAttemptRecordV1,
    AssessmentCheckpointIdempotencyExecutor,
    AssessmentJournalPersistenceError,
    assessment_attempt_journal_reducer,
    build_assessment_attempt_record_v1,
    merge_assessment_attempt_journal_v1,
    validate_assessment_attempt_journal_v1,
)
from src.assessment.attempt_service import AssessmentRequestConflict

THREAD_ID = "thread-assessment-journal-1"
REQUEST_ID = "request-assessment-journal-1"
RESOURCE_ID = f"resource:v3:{'a' * 64}"
QUESTION_ID = f"question:v1:{'b' * 64}"


def _attempt(*, answer: str = "4") -> AssessmentAttemptV1:
    return AssessmentAttemptV1(
        schema_version="assessment_attempt_v1",
        request_id=REQUEST_ID,
        resource_id=RESOURCE_ID,
        question_id=QUESTION_ID,
        answer=answer,
        time_spent_seconds=3.5,
    )


def _final() -> AssessmentFinalV1:
    return build_assessment_final_v1(
        thread_id=THREAD_ID,
        attempt=_attempt(),
        is_correct=True,
        error_classification=None,
        adaptive_tasks=(),
    )


def _request_hash(*, answer: str = "4") -> str:
    return stable_assessment_attempt_hash(
        thread_id=THREAD_ID,
        attempt=_attempt(answer=answer),
    )


class _MemoryJournalStore:
    def __init__(self, *, persist: bool = True) -> None:
        self.value: dict = {}
        self.persist = persist
        self.append_count = 0

    async def load(self, thread_id: str) -> object:
        assert thread_id == THREAD_ID
        return json.loads(json.dumps(self.value)) if self.value else {}

    async def append(
        self,
        thread_id: str,
        update: AssessmentAttemptJournalV1,
    ) -> None:
        assert thread_id == THREAD_ID
        self.append_count += 1
        if self.persist:
            self.value = assessment_attempt_journal_reducer(
                self.value,
                update.model_dump(mode="json"),
            )


def _executor(store: _MemoryJournalStore) -> AssessmentCheckpointIdempotencyExecutor:
    return AssessmentCheckpointIdempotencyExecutor(
        load_journal=store.load,
        append_journal=store.append,
    )


def _operation(counter: list[int]) -> Awaitable[AssessmentFinalV1]:
    async def run() -> AssessmentFinalV1:
        counter[0] += 1
        await asyncio.sleep(0)
        return _final()

    return run()


@pytest.mark.anyio
async def test_executor_serializes_duplicate_requests_and_replays_public_final():
    store = _MemoryJournalStore()
    executor = _executor(store)
    counter = [0]

    async def run_once() -> object:
        async def operation() -> AssessmentFinalV1:
            return await _operation(counter)

        return await executor.execute_once(
            thread_id=THREAD_ID,
            request_id=REQUEST_ID,
            request_hash=_request_hash(),
            operation=operation,
        )

    first, second = await asyncio.gather(run_once(), run_once())

    assert first == second == _final()
    assert counter == [1]
    assert store.append_count == 1
    journal = validate_assessment_attempt_journal_v1(
        store.value,
        thread_id=THREAD_ID,
    )
    assert len(journal.records) == 1
    serialized = json.dumps(journal.model_dump(mode="json"), ensure_ascii=False)
    for forbidden in (
        "answer_key",
        "accepted_answers",
        "canonical_correct_answer",
        "answer_explanation",
    ):
        assert forbidden not in serialized


def test_journal_record_does_not_retain_the_submitted_answer():
    submitted_answer = "private-submitted-answer-7f0b"
    attempt = _attempt(answer=submitted_answer)
    final = build_assessment_final_v1(
        thread_id=THREAD_ID,
        attempt=attempt,
        is_correct=True,
        error_classification=None,
        adaptive_tasks=(),
    )
    record = build_assessment_attempt_record_v1(
        request_hash=stable_assessment_attempt_hash(
            thread_id=THREAD_ID,
            attempt=attempt,
        ),
        final=final,
    )

    serialized = json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
    assert submitted_answer not in serialized


@pytest.mark.anyio
async def test_executor_rejects_request_id_reuse_with_different_hash():
    store = _MemoryJournalStore()
    executor = _executor(store)

    await executor.execute_once(
        thread_id=THREAD_ID,
        request_id=REQUEST_ID,
        request_hash=_request_hash(),
        operation=lambda: _operation([0]),
    )

    with pytest.raises(AssessmentRequestConflict):
        await executor.execute_once(
            thread_id=THREAD_ID,
            request_id=REQUEST_ID,
            request_hash=_request_hash(answer="5"),
            operation=lambda: _operation([0]),
        )


@pytest.mark.anyio
async def test_failed_operation_is_not_cached_and_can_be_retried():
    store = _MemoryJournalStore()
    executor = _executor(store)

    async def fail() -> AssessmentFinalV1:
        raise RuntimeError("private answer must not be persisted")

    with pytest.raises(RuntimeError, match="private answer"):
        await executor.execute_once(
            thread_id=THREAD_ID,
            request_id=REQUEST_ID,
            request_hash=_request_hash(),
            operation=fail,
        )
    assert store.value == {}

    final = await executor.execute_once(
        thread_id=THREAD_ID,
        request_id=REQUEST_ID,
        request_hash=_request_hash(),
        operation=lambda: _operation([0]),
    )
    assert final == _final()


@pytest.mark.anyio
async def test_executor_fails_closed_when_append_is_not_durable():
    store = _MemoryJournalStore(persist=False)
    executor = _executor(store)

    with pytest.raises(
        AssessmentJournalPersistenceError,
        match="assessment_attempt_record_not_durable",
    ):
        await executor.execute_once(
            thread_id=THREAD_ID,
            request_id=REQUEST_ID,
            request_hash=_request_hash(),
            operation=lambda: _operation([0]),
        )


def test_journal_json_contract_rejects_schema_drift_and_identity_conflicts():
    record = build_assessment_attempt_record_v1(
        request_hash=_request_hash(),
        final=_final(),
    )
    journal = AssessmentAttemptJournalV1(
        schema_version="assessment_attempt_journal_v1",
        thread_id=THREAD_ID,
        records=(record,),
    )
    restored = validate_assessment_attempt_journal_v1(
        journal.model_dump(mode="json"),
        thread_id=THREAD_ID,
    )
    assert restored == journal

    payload = journal.model_dump(mode="json")
    payload["unexpected"] = True
    with pytest.raises(AssessmentAttemptJournalError):
        validate_assessment_attempt_journal_v1(payload)

    with pytest.raises(AssessmentAttemptJournalError) as thread_error:
        validate_assessment_attempt_journal_v1(journal, thread_id="other-thread")
    assert thread_error.value.code == "assessment_attempt_journal_thread_mismatch"

    conflicting = AssessmentAttemptRecordV1(
        schema_version="assessment_attempt_record_v1",
        request_id=record.request_id,
        request_hash=f"assessment-attempt:v1:{'f' * 64}",
        final=record.final,
        committed_at=record.committed_at,
    )
    with pytest.raises(AssessmentAttemptJournalError) as conflict:
        merge_assessment_attempt_journal_v1(
            existing=journal,
            update=AssessmentAttemptJournalV1(
                schema_version="assessment_attempt_journal_v1",
                thread_id=THREAD_ID,
                records=(conflicting,),
            ),
        )
    assert conflict.value.code == "assessment_attempt_journal_request_conflict"


def test_record_rejects_request_identity_drift():
    record = build_assessment_attempt_record_v1(
        request_hash=_request_hash(),
        final=_final(),
    )
    with pytest.raises(ValidationError, match="request_id"):
        AssessmentAttemptRecordV1(
            schema_version="assessment_attempt_record_v1",
            request_id="different-request",
            request_hash=record.request_hash,
            final=record.final,
            committed_at=record.committed_at,
        )
