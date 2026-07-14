"""Strict durable idempotency journal for assessment attempt terminals."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncContextManager, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.assessment.attempt_contracts import (
    ASSESSMENT_ATTEMPT_HASH_PATTERN,
    ASSESSMENT_REQUEST_ID_PATTERN,
    ASSESSMENT_THREAD_ID_PATTERN,
    AssessmentFinalV1,
)
from src.assessment.attempt_service import (
    AssessmentAttemptServiceError,
    AssessmentDependencyFailed,
    AssessmentOperation,
    AssessmentRecordedFailure,
    AssessmentRecoveryRequired,
    AssessmentRequestConflict,
)

ASSESSMENT_ATTEMPT_JOURNAL_SCHEMA_VERSION = "assessment_attempt_journal_v1"
ASSESSMENT_ATTEMPT_RECORD_SCHEMA_VERSION = "assessment_attempt_record_v1"
ASSESSMENT_ATTEMPT_JOURNAL_MAX_RECORDS = 1_000

AssessmentJournalLoader = Callable[[str], Awaitable[object]]
AssessmentJournalAppender = Callable[
    [str, "AssessmentAttemptJournalV1"], Awaitable[None]
]


class AssessmentExecutionLock(Protocol):
    """Serialize assessment execution for a thread across the active backend."""

    def hold(self, thread_id: str) -> AsyncContextManager[None]:
        """Hold the backend-specific thread lock for one complete operation."""


class AssessmentAttemptJournalError(ValueError):
    """Typed failure for invalid or conflicting durable journal state."""

    def __init__(self, *, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class AssessmentJournalPersistenceError(RuntimeError):
    """Raised when a journal append is not durably observable after writing."""

    def __init__(self, *, code: str) -> None:
        self.code = code
        super().__init__(f"assessment journal persistence failed: {code}")


class AssessmentAttemptRecordV1(BaseModel):
    """One content-free durable claim and its optional terminal outcome."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["assessment_attempt_record_v1"]
    request_id: str = Field(pattern=ASSESSMENT_REQUEST_ID_PATTERN, max_length=160)
    request_hash: str = Field(pattern=ASSESSMENT_ATTEMPT_HASH_PATTERN)
    status: Literal["in_progress", "completed", "failed"]
    started_at: datetime
    committed_at: datetime | None
    final: AssessmentFinalV1 | None
    error_code: str = Field(pattern=r"^assessment_[a-z0-9_]{1,120}$|^$")
    failure_stage: Literal[
        "",
        "assessment",
        "idempotency",
        "error_classification",
        "adaptive_practice",
    ]
    exception_type: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.]{0,159}$|^$")

    @model_validator(mode="after")
    def validate_record_identity(self) -> AssessmentAttemptRecordV1:
        if self.started_at.tzinfo is None:
            raise ValueError("started_at must include a timezone")
        if self.committed_at is not None and self.committed_at.tzinfo is None:
            raise ValueError("committed_at must include a timezone")
        if self.status == "in_progress":
            if (
                self.committed_at is not None
                or self.final is not None
                or self.error_code
                or self.failure_stage
                or self.exception_type
            ):
                raise ValueError("in_progress record cannot contain terminal data")
            return self
        if self.committed_at is None:
            raise ValueError("terminal record requires committed_at")
        if self.status == "completed":
            if self.final is None:
                raise ValueError("completed record requires final")
            if self.request_id != self.final.request_id:
                raise ValueError("record request_id must match final request_id")
            if self.error_code or self.failure_stage or self.exception_type:
                raise ValueError("completed record cannot contain failure data")
            return self
        if self.final is not None:
            raise ValueError("failed record cannot contain final")
        if not self.error_code or not self.failure_stage or not self.exception_type:
            raise ValueError("failed record requires content-free failure metadata")
        return self


class AssessmentAttemptJournalV1(BaseModel):
    """Bounded thread checkpoint journal containing no submitted answers."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["assessment_attempt_journal_v1"]
    thread_id: str = Field(pattern=ASSESSMENT_THREAD_ID_PATTERN, max_length=160)
    records: tuple[AssessmentAttemptRecordV1, ...] = Field(
        max_length=ASSESSMENT_ATTEMPT_JOURNAL_MAX_RECORDS
    )

    @model_validator(mode="after")
    def validate_journal_identity(self) -> AssessmentAttemptJournalV1:
        request_ids = [record.request_id for record in self.records]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("assessment journal request_id values must be unique")
        if any(
            record.final is not None and record.final.thread_id != self.thread_id
            for record in self.records
        ):
            raise ValueError("assessment journal final belongs to another thread")
        return self


def new_assessment_attempt_journal_v1(thread_id: str) -> AssessmentAttemptJournalV1:
    """Create one empty, strictly validated thread journal."""

    return AssessmentAttemptJournalV1(
        schema_version=ASSESSMENT_ATTEMPT_JOURNAL_SCHEMA_VERSION,
        thread_id=thread_id,
        records=(),
    )


def build_assessment_attempt_claim_v1(
    *,
    request_id: str,
    request_hash: str,
) -> AssessmentAttemptRecordV1:
    """Persist a content-free execution claim before any provider dispatch."""

    return AssessmentAttemptRecordV1(
        schema_version=ASSESSMENT_ATTEMPT_RECORD_SCHEMA_VERSION,
        request_id=request_id,
        request_hash=request_hash,
        status="in_progress",
        started_at=datetime.now(timezone.utc),
        committed_at=None,
        final=None,
        error_code="",
        failure_stage="",
        exception_type="",
    )


def complete_assessment_attempt_claim_v1(
    *,
    claim: AssessmentAttemptRecordV1,
    final: AssessmentFinalV1,
) -> AssessmentAttemptRecordV1:
    """Transition one exact durable claim to its public successful terminal."""

    if claim.status != "in_progress":
        raise AssessmentAttemptJournalError(
            code="assessment_attempt_claim_not_in_progress",
            message="only an in_progress assessment claim can complete",
        )
    return AssessmentAttemptRecordV1(
        schema_version=ASSESSMENT_ATTEMPT_RECORD_SCHEMA_VERSION,
        request_id=claim.request_id,
        request_hash=claim.request_hash,
        status="completed",
        started_at=claim.started_at,
        committed_at=datetime.now(timezone.utc),
        final=final,
        error_code="",
        failure_stage="",
        exception_type="",
    )


def fail_assessment_attempt_claim_v1(
    *,
    claim: AssessmentAttemptRecordV1,
    error_code: str,
    failure_stage: Literal[
        "assessment",
        "idempotency",
        "error_classification",
        "adaptive_practice",
    ],
    exception_type: str,
) -> AssessmentAttemptRecordV1:
    """Transition one exact durable claim to a content-free failed terminal."""

    if claim.status != "in_progress":
        raise AssessmentAttemptJournalError(
            code="assessment_attempt_claim_not_in_progress",
            message="only an in_progress assessment claim can fail",
        )
    return AssessmentAttemptRecordV1(
        schema_version=ASSESSMENT_ATTEMPT_RECORD_SCHEMA_VERSION,
        request_id=claim.request_id,
        request_hash=claim.request_hash,
        status="failed",
        started_at=claim.started_at,
        committed_at=datetime.now(timezone.utc),
        final=None,
        error_code=error_code,
        failure_stage=failure_stage,
        exception_type=exception_type,
    )


def validate_assessment_attempt_journal_v1(
    value: AssessmentAttemptJournalV1 | Mapping[str, object],
    *,
    thread_id: str | None = None,
) -> AssessmentAttemptJournalV1:
    """Validate live or JSON-restored journal state without key repair."""

    try:
        payload: object = (
            value.model_dump(mode="json")
            if isinstance(value, AssessmentAttemptJournalV1)
            else value
        )
        journal = AssessmentAttemptJournalV1.model_validate_json(
            _mapping_json(payload),
            strict=True,
        )
    except (TypeError, ValueError) as exc:
        raise AssessmentAttemptJournalError(
            code="assessment_attempt_journal_invalid",
            message="assessment attempt journal violates its strict JSON schema",
        ) from exc
    if thread_id is not None and journal.thread_id != thread_id:
        raise AssessmentAttemptJournalError(
            code="assessment_attempt_journal_thread_mismatch",
            message="assessment attempt journal belongs to another thread",
        )
    return journal


def find_assessment_attempt_record_v1(
    journal: AssessmentAttemptJournalV1,
    *,
    request_id: str,
) -> AssessmentAttemptRecordV1 | None:
    """Return the exact durable request record when present."""

    return next(
        (record for record in journal.records if record.request_id == request_id),
        None,
    )


def merge_assessment_attempt_journal_v1(
    *,
    existing: AssessmentAttemptJournalV1 | Mapping[str, object] | None,
    update: AssessmentAttemptJournalV1 | Mapping[str, object],
) -> AssessmentAttemptJournalV1:
    """Merge exact records idempotently and reject all identity conflicts."""

    incoming = validate_assessment_attempt_journal_v1(update)
    current = (
        new_assessment_attempt_journal_v1(incoming.thread_id)
        if existing is None or existing == {}
        else validate_assessment_attempt_journal_v1(
            existing,
            thread_id=incoming.thread_id,
        )
    )
    records = list(current.records)
    indexes_by_request_id = {
        record.request_id: index for index, record in enumerate(records)
    }
    for record in incoming.records:
        prior_index = indexes_by_request_id.get(record.request_id)
        if prior_index is not None:
            prior = records[prior_index]
            if prior == record:
                continue
            if (
                prior.status == "in_progress"
                and record.status in {"completed", "failed"}
                and prior.request_hash == record.request_hash
                and prior.started_at == record.started_at
            ):
                records[prior_index] = record
                continue
            raise AssessmentAttemptJournalError(
                code="assessment_attempt_journal_request_conflict",
                message="assessment request_id is bound to different journal data",
            )
        if record.status != "in_progress":
            raise AssessmentAttemptJournalError(
                code="assessment_attempt_terminal_without_claim",
                message="assessment terminal update requires a durable claim",
            )
        records.append(record)
        indexes_by_request_id[record.request_id] = len(records) - 1
    try:
        return AssessmentAttemptJournalV1(
            schema_version=ASSESSMENT_ATTEMPT_JOURNAL_SCHEMA_VERSION,
            thread_id=current.thread_id,
            records=tuple(records),
        )
    except ValidationError as exc:
        raise AssessmentAttemptJournalError(
            code="assessment_attempt_journal_capacity_exhausted",
            message="assessment attempt journal cannot accept another record",
        ) from exc


def assessment_attempt_journal_reducer(existing: dict, update: dict) -> dict:
    """LangGraph reducer for exact, bounded assessment terminal records."""

    if not update:
        return dict(existing or {})
    return merge_assessment_attempt_journal_v1(
        existing=existing,
        update=update,
    ).model_dump(mode="json")


@dataclass
class _LockEntry:
    lock: asyncio.Lock
    users: int


class LocalAssessmentExecutionLock:
    """Process-local execution lock used only with the memory checkpointer."""

    def __init__(self) -> None:
        self._entries: dict[str, _LockEntry] = {}
        self._entries_guard = asyncio.Lock()

    @asynccontextmanager
    async def hold(self, thread_id: str) -> AsyncIterator[None]:
        if not thread_id.strip():
            raise ValueError("assessment execution lock requires thread_id")
        entry = await self._reserve(thread_id)
        try:
            async with entry.lock:
                yield
        finally:
            await self._release(thread_id, entry)

    async def _reserve(self, thread_id: str) -> _LockEntry:
        async with self._entries_guard:
            entry = self._entries.get(thread_id)
            if entry is None:
                entry = _LockEntry(lock=asyncio.Lock(), users=0)
                self._entries[thread_id] = entry
            entry.users += 1
            return entry

    async def _release(self, thread_id: str, entry: _LockEntry) -> None:
        async with self._entries_guard:
            current = self._entries.get(thread_id)
            if current is not entry or current.users <= 0:
                raise RuntimeError("assessment execution lock registry is inconsistent")
            current.users -= 1
            if current.users == 0:
                self._entries.pop(thread_id, None)


class AssessmentCheckpointIdempotencyExecutor:
    """Claim before dispatch and verify every checkpoint journal transition."""

    def __init__(
        self,
        *,
        load_journal: AssessmentJournalLoader,
        append_journal: AssessmentJournalAppender,
        execution_lock: AssessmentExecutionLock,
    ) -> None:
        self._load_journal = load_journal
        self._append_journal = append_journal
        self._execution_lock = execution_lock

    async def execute_once(
        self,
        *,
        thread_id: str,
        request_id: str,
        request_hash: str,
        operation: AssessmentOperation,
    ) -> object:
        async with self._execution_lock.hold(thread_id):
            journal = await self._load(thread_id)
            existing = find_assessment_attempt_record_v1(
                journal,
                request_id=request_id,
            )
            if existing is not None:
                return _replay_record(existing, request_hash=request_hash)

            claim = build_assessment_attempt_claim_v1(
                request_id=request_id,
                request_hash=request_hash,
            )
            await self._persist_record(
                thread_id,
                claim,
                failure_code="assessment_attempt_claim_not_durable",
            )
            try:
                raw_final = await operation()
                final = AssessmentFinalV1.model_validate(raw_final, strict=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                error_code, failure_stage, exception_type = _failure_metadata(exc)
                failed = fail_assessment_attempt_claim_v1(
                    claim=claim,
                    error_code=error_code,
                    failure_stage=failure_stage,
                    exception_type=exception_type,
                )
                await self._persist_record(
                    thread_id,
                    failed,
                    failure_code="assessment_attempt_failure_not_durable",
                )
                raise

            completed = complete_assessment_attempt_claim_v1(
                claim=claim,
                final=final,
            )
            persisted_record = await self._persist_record(
                thread_id,
                completed,
                failure_code="assessment_attempt_final_not_durable",
            )
            if persisted_record.final is None:
                raise AssessmentJournalPersistenceError(
                    code="assessment_attempt_final_missing"
                )
            return persisted_record.final

    async def _persist_record(
        self,
        thread_id: str,
        record: AssessmentAttemptRecordV1,
        *,
        failure_code: str,
    ) -> AssessmentAttemptRecordV1:
        update = AssessmentAttemptJournalV1(
            schema_version=ASSESSMENT_ATTEMPT_JOURNAL_SCHEMA_VERSION,
            thread_id=thread_id,
            records=(record,),
        )
        await self._append_journal(thread_id, update)
        persisted = await self._load(thread_id)
        persisted_record = find_assessment_attempt_record_v1(
            persisted,
            request_id=record.request_id,
        )
        if persisted_record != record:
            raise AssessmentJournalPersistenceError(code=failure_code)
        return persisted_record

    async def _load(self, thread_id: str) -> AssessmentAttemptJournalV1:
        raw = await self._load_journal(thread_id)
        if raw is None or raw == {}:
            return new_assessment_attempt_journal_v1(thread_id)
        if not isinstance(raw, Mapping | AssessmentAttemptJournalV1):
            raise AssessmentAttemptJournalError(
                code="assessment_attempt_journal_invalid",
                message="assessment attempt journal must be an object",
            )
        return validate_assessment_attempt_journal_v1(raw, thread_id=thread_id)


def _replay_record(
    record: AssessmentAttemptRecordV1,
    *,
    request_hash: str,
) -> AssessmentFinalV1:
    if record.request_hash != request_hash:
        raise AssessmentRequestConflict()
    if record.status == "in_progress":
        raise AssessmentRecoveryRequired()
    if record.status == "failed":
        failure_stage = record.failure_stage
        if failure_stage == "":
            raise AssessmentAttemptJournalError(
                code="assessment_attempt_failure_stage_missing",
                message="failed assessment journal record has no stage",
            )
        raise AssessmentRecordedFailure(
            code=record.error_code,
            stage=failure_stage,
            exception_type=record.exception_type,
        )
    if record.final is None:
        raise AssessmentAttemptJournalError(
            code="assessment_attempt_completed_final_missing",
            message="completed assessment journal record has no final",
        )
    return record.final


def _failure_metadata(
    exc: Exception,
) -> tuple[
    str,
    Literal[
        "assessment",
        "idempotency",
        "error_classification",
        "adaptive_practice",
    ],
    str,
]:
    if isinstance(exc, AssessmentDependencyFailed):
        return exc.code, exc.stage, exc.exception_type
    if isinstance(exc, AssessmentAttemptServiceError):
        return exc.code, "assessment", type(exc).__name__
    return "assessment_idempotency_failed", "idempotency", type(exc).__name__


def _mapping_json(value: object) -> str:
    if not isinstance(value, Mapping):
        raise TypeError("value must be an object")
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "ASSESSMENT_ATTEMPT_JOURNAL_MAX_RECORDS",
    "ASSESSMENT_ATTEMPT_JOURNAL_SCHEMA_VERSION",
    "ASSESSMENT_ATTEMPT_RECORD_SCHEMA_VERSION",
    "AssessmentAttemptJournalError",
    "AssessmentAttemptJournalV1",
    "AssessmentAttemptRecordV1",
    "AssessmentCheckpointIdempotencyExecutor",
    "AssessmentExecutionLock",
    "AssessmentJournalAppender",
    "AssessmentJournalLoader",
    "AssessmentJournalPersistenceError",
    "LocalAssessmentExecutionLock",
    "assessment_attempt_journal_reducer",
    "build_assessment_attempt_claim_v1",
    "complete_assessment_attempt_claim_v1",
    "fail_assessment_attempt_claim_v1",
    "find_assessment_attempt_record_v1",
    "merge_assessment_attempt_journal_v1",
    "new_assessment_attempt_journal_v1",
    "validate_assessment_attempt_journal_v1",
]
