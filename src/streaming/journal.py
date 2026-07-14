"""Bounded in-memory event journal for same-process SSE continuation."""

from __future__ import annotations

import json
import threading
import time
from collections import deque

from src.streaming.contracts import AgentStreamEventV2, StreamContractError


_AUTHORITATIVE_TERMINALS = frozenset(
    {
        "qa_final",
        "resource_final",
        "assessment_final",
        "interrupt",
        "stopped",
        "stream_error",
    }
)
_RESERVED_TERMINAL_EVENT_BYTES = 4096


class StreamJournalCapacityError(StreamContractError):
    """Raised instead of silently dropping an event when capacity is exhausted."""


class StreamJournalExpiredError(StreamContractError):
    """Raised when a caller attempts to replay an expired journal."""


class StreamJournalSequenceError(StreamContractError):
    """Raised when appended events are not strictly contiguous."""


class StreamJournal:
    """Keep validated UI-safe events through a configured terminal retention."""

    def __init__(
        self,
        *,
        stream_id: str,
        max_events: int,
        max_bytes: int,
        ttl_seconds: int,
    ) -> None:
        if not stream_id.strip():
            raise StreamContractError("stream_id is required")
        if max_events <= 2:
            raise StreamContractError(
                "journal event limit must reserve stream_error and stream_done"
            )
        if max_bytes <= 2 * _RESERVED_TERMINAL_EVENT_BYTES:
            raise StreamContractError(
                "journal byte limit must reserve stream_error and stream_done"
            )
        if ttl_seconds <= 0:
            raise StreamContractError("journal TTL must be positive")
        self.stream_id = stream_id
        self.max_events = max_events
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self._events: deque[AgentStreamEventV2] = deque()
        self._bytes = 0
        self._sealed_at: float | None = None
        self._replay_leases = 0
        self._lock = threading.Lock()

    def append(self, event: AgentStreamEventV2) -> None:
        """Append a contiguous event or fail before mutating the journal."""

        if event.stream_id != self.stream_id:
            raise StreamJournalSequenceError("event stream_id does not match journal")
        event_bytes = len(
            json.dumps(event.model_dump(mode="json"), ensure_ascii=False).encode(
                "utf-8"
            )
        )
        with self._lock:
            self._ensure_not_expired()
            expected = len(self._events) + 1
            if event.sequence != expected:
                raise StreamJournalSequenceError(
                    f"expected sequence {expected}, got {event.sequence}"
                )
            reserved_events, reserved_bytes = _terminal_reserve(event.type)
            if len(self._events) + 1 + reserved_events > self.max_events:
                raise StreamJournalCapacityError("stream event limit exceeded")
            if self._bytes + event_bytes + reserved_bytes > self.max_bytes:
                raise StreamJournalCapacityError("stream byte limit exceeded")
            self._events.append(event)
            self._bytes += event_bytes

    def after(self, sequence: int) -> list[AgentStreamEventV2]:
        """Return immutable validated events strictly after ``sequence``."""

        if sequence < 0:
            raise StreamJournalSequenceError("sequence must be non-negative")
        with self._lock:
            self._ensure_not_expired()
            if sequence > len(self._events):
                raise StreamJournalSequenceError(
                    "requested sequence is ahead of journal"
                )
            return list(self._events)[sequence:]

    @property
    def size(self) -> int:
        with self._lock:
            self._ensure_not_expired()
            return len(self._events)

    @property
    def last_event(self) -> AgentStreamEventV2 | None:
        with self._lock:
            self._ensure_not_expired()
            return self._events[-1] if self._events else None

    @property
    def expired(self) -> bool:
        with self._lock:
            return self._is_expired()

    def seal(self, *, completed_monotonic: float | None = None) -> None:
        """Start retention TTL only after the producer has completed."""

        with self._lock:
            if self._sealed_at is not None:
                raise StreamContractError("stream journal is already sealed")
            self._sealed_at = (
                time.monotonic() if completed_monotonic is None else completed_monotonic
            )

    def acquire_replay_lease(self) -> None:
        """Prevent a validated replay from expiring while it is delivered."""

        with self._lock:
            self._ensure_not_expired()
            self._replay_leases += 1

    def release_replay_lease(self) -> None:
        with self._lock:
            if self._replay_leases <= 0:
                raise StreamContractError("stream replay lease is not active")
            self._replay_leases -= 1

    def _ensure_not_expired(self) -> None:
        if self._is_expired():
            raise StreamJournalExpiredError("stream journal expired")

    def _is_expired(self) -> bool:
        return (
            self._sealed_at is not None
            and self._replay_leases == 0
            and time.monotonic() - self._sealed_at > self.ttl_seconds
        )


def _terminal_reserve(event_type: str) -> tuple[int, int]:
    if event_type == "stream_done":
        return 0, 0
    if event_type in _AUTHORITATIVE_TERMINALS:
        return 1, _RESERVED_TERMINAL_EVENT_BYTES
    return 2, 2 * _RESERVED_TERMINAL_EVENT_BYTES


__all__ = [
    "StreamJournal",
    "StreamJournalCapacityError",
    "StreamJournalExpiredError",
    "StreamJournalSequenceError",
]
