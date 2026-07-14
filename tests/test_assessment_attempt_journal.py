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
    LocalAssessmentExecutionLock,
    assessment_attempt_journal_reducer,
    build_assessment_attempt_claim_v1,
    complete_assessment_attempt_claim_v1,
    merge_assessment_attempt_journal_v1,
    validate_assessment_attempt_journal_v1,
)
from src.assessment.attempt_service import (
    AssessmentRecordedFailure,
    AssessmentRecoveryRequired,
    AssessmentRequestConflict,
)

THREAD_ID = "thread-assessment-journal-1"
REQUEST_ID = "00000000-0000-4000-8000-000000000102"
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


def _executor(
    store: _MemoryJournalStore,
    *,
    execution_lock: LocalAssessmentExecutionLock | None = None,
) -> AssessmentCheckpointIdempotencyExecutor:
    lock = execution_lock
    if lock is None:
        lock = LocalAssessmentExecutionLock()
    return AssessmentCheckpointIdempotencyExecutor(
        load_journal=store.load,
        append_journal=store.append,
        execution_lock=lock,
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
    assert store.append_count == 2
    journal = validate_assessment_attempt_journal_v1(
        store.value,
        thread_id=THREAD_ID,
    )
    assert len(journal.records) == 1
    assert journal.records[0].status == "completed"
    serialized = json.dumps(journal.model_dump(mode="json"), ensure_ascii=False)
    for forbidden in (
        "answer_key",
        "accepted_answers",
        "canonical_correct_answer",
        "answer_explanation",
    ):
        assert forbidden not in serialized


@pytest.mark.anyio
async def test_shared_backend_lock_serializes_independent_executors():
    store = _MemoryJournalStore()
    shared_lock = LocalAssessmentExecutionLock()
    first_executor = _executor(store, execution_lock=shared_lock)
    second_executor = _executor(store, execution_lock=shared_lock)
    counter = [0]

    async def run_once(
        executor: AssessmentCheckpointIdempotencyExecutor,
    ) -> object:
        return await executor.execute_once(
            thread_id=THREAD_ID,
            request_id=REQUEST_ID,
            request_hash=_request_hash(),
            operation=lambda: _operation(counter),
        )

    first, second = await asyncio.gather(
        run_once(first_executor),
        run_once(second_executor),
    )

    assert first == second == _final()
    assert counter == [1]


@pytest.mark.anyio
async def test_claim_is_durable_before_operation_dispatch():
    store = _MemoryJournalStore()
    executor = _executor(store)

    async def operation() -> AssessmentFinalV1:
        journal = validate_assessment_attempt_journal_v1(store.value)
        assert journal.records[0].status == "in_progress"
        assert journal.records[0].final is None
        return _final()

    final = await executor.execute_once(
        thread_id=THREAD_ID,
        request_id=REQUEST_ID,
        request_hash=_request_hash(),
        operation=operation,
    )

    assert final == _final()


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
    claim = build_assessment_attempt_claim_v1(
        request_id=attempt.request_id,
        request_hash=stable_assessment_attempt_hash(
            thread_id=THREAD_ID,
            attempt=attempt,
        ),
    )
    record = complete_assessment_attempt_claim_v1(
        claim=claim,
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
async def test_failed_operation_is_durably_replayed_without_retrying_operation():
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
    record = validate_assessment_attempt_journal_v1(store.value).records[0]
    assert record.status == "failed"
    serialized = json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
    assert "private answer" not in serialized

    counter = [0]
    with pytest.raises(AssessmentRecordedFailure) as exc_info:
        await executor.execute_once(
            thread_id=THREAD_ID,
            request_id=REQUEST_ID,
            request_hash=_request_hash(),
            operation=lambda: _operation(counter),
        )
    assert exc_info.value.code == "assessment_idempotency_failed"
    assert counter == [0]


@pytest.mark.anyio
async def test_cancelled_operation_leaves_claim_and_requires_explicit_recovery():
    store = _MemoryJournalStore()
    executor = _executor(store)

    async def cancel() -> AssessmentFinalV1:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await executor.execute_once(
            thread_id=THREAD_ID,
            request_id=REQUEST_ID,
            request_hash=_request_hash(),
            operation=cancel,
        )

    record = validate_assessment_attempt_journal_v1(store.value).records[0]
    assert record.status == "in_progress"
    counter = [0]
    with pytest.raises(AssessmentRecoveryRequired):
        await executor.execute_once(
            thread_id=THREAD_ID,
            request_id=REQUEST_ID,
            request_hash=_request_hash(),
            operation=lambda: _operation(counter),
        )
    assert counter == [0]


@pytest.mark.anyio
async def test_executor_fails_closed_when_append_is_not_durable():
    store = _MemoryJournalStore(persist=False)
    executor = _executor(store)

    with pytest.raises(
        AssessmentJournalPersistenceError,
        match="assessment_attempt_claim_not_durable",
    ):
        await executor.execute_once(
            thread_id=THREAD_ID,
            request_id=REQUEST_ID,
            request_hash=_request_hash(),
            operation=lambda: _operation([0]),
        )


def test_journal_json_contract_rejects_schema_drift_and_identity_conflicts():
    claim = build_assessment_attempt_claim_v1(
        request_id=REQUEST_ID,
        request_hash=_request_hash(),
    )
    record = complete_assessment_attempt_claim_v1(
        claim=claim,
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
        status="completed",
        started_at=record.started_at,
        committed_at=record.committed_at,
        final=record.final,
        error_code="",
        failure_stage="",
        exception_type="",
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
    claim = build_assessment_attempt_claim_v1(
        request_id=REQUEST_ID,
        request_hash=_request_hash(),
    )
    record = complete_assessment_attempt_claim_v1(
        claim=claim,
        final=_final(),
    )
    with pytest.raises(ValidationError, match="request_id"):
        AssessmentAttemptRecordV1(
            schema_version="assessment_attempt_record_v1",
            request_id="different-request",
            request_hash=record.request_hash,
            status=record.status,
            started_at=record.started_at,
            committed_at=record.committed_at,
            final=record.final,
            error_code=record.error_code,
            failure_stage=record.failure_stage,
            exception_type=record.exception_type,
        )
