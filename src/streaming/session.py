"""Process-local producer/subscriber sessions for resumable SSE delivery."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone

from src.streaming.contracts import (
    AgentStreamEventDraftV2,
    AgentStreamEventType,
    AgentStreamEventV2,
    StreamContractError,
    StreamEventSequencer,
)
from src.streaming.journal import StreamJournal, StreamJournalCapacityError
from src.streaming.settings import StreamingRuntimeConfig
from src.streaming.sse import encode_sse_event


_AUTHORITATIVE_TERMINALS = frozenset(
    {
        "qa_final",
        "resource_final",
        "recommendation_final",
        "assessment_final",
        "interrupt",
        "stopped",
        "stream_error",
    }
)


class StreamSessionNotFoundError(StreamContractError):
    pass


class StreamSessionExpiredError(StreamContractError):
    pass


class StreamSessionConflictError(StreamContractError):
    pass


@dataclass(frozen=True)
class _RequestBinding:
    stream_id: str
    thread_id: str
    operation: str
    request_fingerprint: str


class StreamSession:
    def __init__(
        self,
        *,
        stream_id: str,
        request_id: str,
        thread_id: str,
        operation: str,
        request_fingerprint: str,
        config: StreamingRuntimeConfig,
    ) -> None:
        if not operation.strip() or not request_fingerprint.strip():
            raise StreamContractError(
                "stream operation and request fingerprint are required"
            )
        self.stream_id = stream_id
        self.request_id = request_id
        self.thread_id = thread_id
        self.operation = operation
        self.request_fingerprint = request_fingerprint
        self.config = config
        self.journal = StreamJournal(
            stream_id=stream_id,
            max_events=config.journal_max_events,
            max_bytes=config.journal_max_bytes,
            ttl_seconds=config.journal_ttl_seconds,
        )
        self._condition = asyncio.Condition()
        self._complete = False
        self._producer: asyncio.Task[None] | None = None
        self._failure: Exception | None = None
        self._recoverable_terminal_error = False

    def start(self, source: AsyncIterable[AgentStreamEventDraftV2]) -> None:
        if self._producer is not None:
            raise StreamSessionConflictError("stream producer already started")
        self._producer = asyncio.create_task(self._run(source))

    def subscribe(self, *, after_sequence: int) -> AsyncIterator[str]:
        """Validate replay synchronously, then retain it through delivery."""

        self.journal.after(after_sequence)
        self.journal.acquire_replay_lease()
        return self._subscribe(after_sequence=after_sequence)

    async def _subscribe(self, *, after_sequence: int) -> AsyncIterator[str]:
        sequence = after_sequence
        try:
            while True:
                events = self.journal.after(sequence)
                for event in events:
                    sequence = event.sequence
                    yield encode_sse_event(
                        event,
                        retry_ms=self.config.retry_ms if event.sequence == 1 else None,
                    )
                if self._complete:
                    if self._failure is not None:
                        raise StreamContractError(
                            "stream producer failed before a terminal could be recorded"
                        ) from self._failure
                    return
                async with self._condition:
                    if self.journal.size <= sequence and not self._complete:
                        await self._condition.wait()
        finally:
            self.journal.release_replay_lease()

    @property
    def complete(self) -> bool:
        return self._complete

    @property
    def expired(self) -> bool:
        return self._complete and self.journal.expired

    @property
    def recoverable_terminal_error(self) -> bool:
        return self._complete and self._recoverable_terminal_error

    async def _run(self, source: AsyncIterable[AgentStreamEventDraftV2]) -> None:
        cancellation: asyncio.CancelledError | None = None
        sequencer = StreamEventSequencer(
            stream_id=self.stream_id,
            request_id=self.request_id,
            thread_id=self.thread_id,
        )
        try:
            self.journal.append(
                sequencer.emit("stream_start", {"retry_ms": self.config.retry_ms})
            )
            async with self._condition:
                self._condition.notify_all()
            async for draft in source:
                validated = AgentStreamEventDraftV2.model_validate(draft)
                self.journal.append(sequencer.emit(validated.type, validated.data))
                if (
                    validated.type == "stream_error"
                    and validated.data.get("recoverable") is True
                ):
                    self._recoverable_terminal_error = True
                async with self._condition:
                    self._condition.notify_all()
            if sequencer.terminal is None:
                self._record_failure_terminal(
                    StreamContractError(
                        "native stream source ended without an authoritative terminal"
                    )
                )
            else:
                self.journal.append(
                    sequencer.emit(
                        "stream_done",
                        {"terminal_type": sequencer.terminal},
                    )
                )
        except asyncio.CancelledError as exc:
            cancellation = exc
            self._record_failure_terminal(exc)
        except Exception as exc:
            self._record_failure_terminal(exc)
        finally:
            self._complete = True
            self.journal.seal()
            async with self._condition:
                self._condition.notify_all()
        if cancellation is not None:
            raise cancellation

    def _record_failure_terminal(self, exc: BaseException) -> None:
        """Materialize a safe replayable terminal without exposing exception text."""

        try:
            last_event = self.journal.last_event
            if last_event is not None and last_event.type == "stream_done":
                return
            if last_event is not None and last_event.type in _AUTHORITATIVE_TERMINALS:
                terminal_type = last_event.type
            else:
                error_type = (
                    "stream_event_log_capacity_exhausted"
                    if isinstance(exc, StreamJournalCapacityError)
                    else "stream_producer_failed"
                )
                message = (
                    "Stream event log capacity was exhausted"
                    if isinstance(exc, StreamJournalCapacityError)
                    else "Stream producer failed before completion"
                )
                error_event = self._failure_event(
                    "stream_error",
                    {
                        "error_type": error_type,
                        "message": message,
                        "recoverable": False,
                    },
                )
                self.journal.append(error_event)
                terminal_type = "stream_error"
            done_event = self._failure_event(
                "stream_done",
                {"terminal_type": terminal_type},
            )
            self.journal.append(done_event)
        except Exception as terminal_exc:
            self._failure = terminal_exc

    def _failure_event(
        self,
        event_type: AgentStreamEventType,
        data: dict[str, object],
    ) -> AgentStreamEventV2:
        sequence = self.journal.size + 1
        return AgentStreamEventV2(
            type=event_type,
            stream_id=self.stream_id,
            event_id=f"{self.stream_id}:{sequence}",
            sequence=sequence,
            request_id=self.request_id,
            thread_id=self.thread_id,
            created_at=datetime.now(timezone.utc),
            data=data,
        )


class StreamSessionManager:
    def __init__(self, config: StreamingRuntimeConfig) -> None:
        self.config = config
        self._sessions: dict[str, StreamSession] = {}
        self._request_bindings: dict[str, _RequestBinding] = {}
        self._expired_streams: set[str] = set()
        self._active_thread_streams: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        *,
        stream_id: str,
        request_id: str,
        thread_id: str,
        operation: str,
        request_fingerprint: str,
        source: AsyncIterable[AgentStreamEventDraftV2],
        allow_recoverable_retry: bool = False,
    ) -> StreamSession:
        async with self._lock:
            self._purge_expired()
            binding = self._request_bindings.get(request_id)
            if binding is not None:
                self._validate_binding(
                    binding,
                    thread_id=thread_id,
                    operation=operation,
                    request_fingerprint=request_fingerprint,
                )
                existing = self._sessions.get(binding.stream_id)
                if existing is None:
                    raise StreamSessionExpiredError(
                        "request_id is bound to an expired stream session"
                    )
                if not (
                    allow_recoverable_retry and existing.recoverable_terminal_error
                ):
                    return existing
                self._request_bindings.pop(request_id)
                if self._active_thread_streams.get(thread_id) == existing.stream_id:
                    self._active_thread_streams.pop(thread_id, None)
            if stream_id in self._sessions or stream_id in self._expired_streams:
                raise StreamSessionConflictError("stream_id already exists")
            active_stream_id = self._active_thread_streams.get(thread_id)
            if active_stream_id is not None:
                active = self._sessions.get(active_stream_id)
                if active is not None and not active.complete:
                    raise StreamSessionConflictError(
                        "thread already has an active stream session"
                    )
                self._active_thread_streams.pop(thread_id, None)
            session = StreamSession(
                stream_id=stream_id,
                request_id=request_id,
                thread_id=thread_id,
                operation=operation,
                request_fingerprint=request_fingerprint,
                config=self.config,
            )
            self._sessions[stream_id] = session
            self._request_bindings[request_id] = _RequestBinding(
                stream_id=stream_id,
                thread_id=thread_id,
                operation=operation,
                request_fingerprint=request_fingerprint,
            )
            self._active_thread_streams[thread_id] = stream_id
            session.start(source)
            return session

    async def get(self, stream_id: str) -> StreamSession:
        async with self._lock:
            self._purge_expired()
            session = self._sessions.get(stream_id)
            if session is not None:
                return session
            if stream_id in self._expired_streams:
                raise StreamSessionExpiredError("stream session expired")
            raise StreamSessionNotFoundError("stream session not found")

    @staticmethod
    def _validate_binding(
        binding: _RequestBinding,
        *,
        thread_id: str,
        operation: str,
        request_fingerprint: str,
    ) -> None:
        if binding.thread_id != thread_id:
            raise StreamSessionConflictError(
                "request_id is already bound to another thread"
            )
        if binding.operation != operation:
            raise StreamSessionConflictError(
                "request_id is already bound to another operation"
            )
        if binding.request_fingerprint != request_fingerprint:
            raise StreamSessionConflictError(
                "request_id payload does not match the original request"
            )

    def _purge_expired(self) -> None:
        expired = [
            stream_id
            for stream_id, session in self._sessions.items()
            if session.expired
        ]
        for stream_id in expired:
            session = self._sessions.pop(stream_id)
            self._expired_streams.add(stream_id)
            binding = self._request_bindings.get(session.request_id)
            if (
                session.recoverable_terminal_error
                and binding is not None
                and binding.stream_id == stream_id
            ):
                self._request_bindings.pop(session.request_id, None)
            if self._active_thread_streams.get(session.thread_id) == stream_id:
                self._active_thread_streams.pop(session.thread_id, None)


__all__ = [
    "StreamSession",
    "StreamSessionConflictError",
    "StreamSessionExpiredError",
    "StreamSessionManager",
    "StreamSessionNotFoundError",
]
