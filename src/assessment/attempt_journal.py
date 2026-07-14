"""Strict durable idempotency journal for assessment attempt terminals."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.assessment.attempt_contracts import (
    ASSESSMENT_ATTEMPT_HASH_PATTERN,
    ASSESSMENT_REQUEST_ID_PATTERN,
    ASSESSMENT_THREAD_ID_PATTERN,
    AssessmentFinalV1,
)
from src.assessment.attempt_service import (
    AssessmentOperation,
    AssessmentRequestConflict,
)

ASSESSMENT_ATTEMPT_JOURNAL_SCHEMA_VERSION = "assessment_attempt_journal_v1"
ASSESSMENT_ATTEMPT_RECORD_SCHEMA_VERSION = "assessment_attempt_record_v1"
ASSESSMENT_ATTEMPT_JOURNAL_MAX_RECORDS = 1_000

AssessmentJournalLoader = Callable[[str], Awaitable[object]]
AssessmentJournalAppender = Callable[
    [str, "AssessmentAttemptJournalV1"], Awaitable[None]
]


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
    """One content-free request identity bound to its public final payload."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["assessment_attempt_record_v1"]
    request_id: str = Field(pattern=ASSESSMENT_REQUEST_ID_PATTERN, max_length=160)
    request_hash: str = Field(pattern=ASSESSMENT_ATTEMPT_HASH_PATTERN)
    final: AssessmentFinalV1
    committed_at: datetime

    @model_validator(mode="after")
    def validate_record_identity(self) -> AssessmentAttemptRecordV1:
        if self.request_id != self.final.request_id:
            raise ValueError("record request_id must match final request_id")
        if self.committed_at.tzinfo is None:
            raise ValueError("committed_at must include a timezone")
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
        if any(record.final.thread_id != self.thread_id for record in self.records):
            raise ValueError("assessment journal final belongs to another thread")
        return self


def new_assessment_attempt_journal_v1(thread_id: str) -> AssessmentAttemptJournalV1:
    """Create one empty, strictly validated thread journal."""

    return AssessmentAttemptJournalV1(
        schema_version=ASSESSMENT_ATTEMPT_JOURNAL_SCHEMA_VERSION,
        thread_id=thread_id,
        records=(),
    )


def build_assessment_attempt_record_v1(
    *,
    request_hash: str,
    final: AssessmentFinalV1,
) -> AssessmentAttemptRecordV1:
    """Bind a request hash to a public final without retaining request content."""

    return AssessmentAttemptRecordV1(
        schema_version=ASSESSMENT_ATTEMPT_RECORD_SCHEMA_VERSION,
        request_id=final.request_id,
        request_hash=request_hash,
        final=final,
        committed_at=datetime.now(timezone.utc),
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
    by_request_id = {record.request_id: record for record in records}
    for record in incoming.records:
        prior = by_request_id.get(record.request_id)
        if prior is not None:
            if prior != record:
                raise AssessmentAttemptJournalError(
                    code="assessment_attempt_journal_request_conflict",
                    message="assessment request_id is bound to different journal data",
                )
            continue
        records.append(record)
        by_request_id[record.request_id] = record
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


class AssessmentCheckpointIdempotencyExecutor:
    """Serialize locally and verify every injected checkpoint journal append."""

    def __init__(
        self,
        *,
        load_journal: AssessmentJournalLoader,
        append_journal: AssessmentJournalAppender,
    ) -> None:
        self._load_journal = load_journal
        self._append_journal = append_journal
        self._lock_entries: dict[tuple[str, str], _LockEntry] = {}
        self._lock_entries_guard = asyncio.Lock()

    async def execute_once(
        self,
        *,
        thread_id: str,
        request_id: str,
        request_hash: str,
        operation: AssessmentOperation,
    ) -> object:
        key = (thread_id, request_id)
        entry = await self._reserve_lock(key)
        try:
            async with entry.lock:
                journal = await self._load(thread_id)
                existing = find_assessment_attempt_record_v1(
                    journal,
                    request_id=request_id,
                )
                if existing is not None:
                    if existing.request_hash != request_hash:
                        raise AssessmentRequestConflict()
                    return existing.final

                raw_final = await operation()
                final = AssessmentFinalV1.model_validate(raw_final, strict=True)
                record = build_assessment_attempt_record_v1(
                    request_hash=request_hash,
                    final=final,
                )
                update = AssessmentAttemptJournalV1(
                    schema_version=ASSESSMENT_ATTEMPT_JOURNAL_SCHEMA_VERSION,
                    thread_id=thread_id,
                    records=(record,),
                )
                await self._append_journal(thread_id, update)

                persisted = await self._load(thread_id)
                persisted_record = find_assessment_attempt_record_v1(
                    persisted,
                    request_id=request_id,
                )
                if persisted_record != record:
                    raise AssessmentJournalPersistenceError(
                        code="assessment_attempt_record_not_durable"
                    )
                return persisted_record.final
        finally:
            await self._release_lock(key, entry)

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

    async def _reserve_lock(self, key: tuple[str, str]) -> _LockEntry:
        async with self._lock_entries_guard:
            entry = self._lock_entries.get(key)
            if entry is None:
                entry = _LockEntry(lock=asyncio.Lock(), users=0)
                self._lock_entries[key] = entry
            entry.users += 1
            return entry

    async def _release_lock(
        self,
        key: tuple[str, str],
        entry: _LockEntry,
    ) -> None:
        async with self._lock_entries_guard:
            current = self._lock_entries.get(key)
            if current is not entry or current.users <= 0:
                raise RuntimeError(
                    "assessment idempotency lock registry is inconsistent"
                )
            current.users -= 1
            if current.users == 0:
                self._lock_entries.pop(key, None)


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
    "AssessmentJournalAppender",
    "AssessmentJournalLoader",
    "AssessmentJournalPersistenceError",
    "assessment_attempt_journal_reducer",
    "build_assessment_attempt_record_v1",
    "find_assessment_attempt_record_v1",
    "merge_assessment_attempt_journal_v1",
    "new_assessment_attempt_journal_v1",
    "validate_assessment_attempt_journal_v1",
]
