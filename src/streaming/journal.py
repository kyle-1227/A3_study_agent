"""Bounded in-memory event journal for same-process SSE continuation."""

from __future__ import annotations

import json
import threading
import time
from collections import deque

from src.streaming.contracts import AgentStreamEventV2, StreamContractError


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
        created_monotonic: float | None = None,
    ) -> None:
        if not stream_id.strip():
            raise StreamContractError("stream_id is required")
        if max_events <= 0 or max_bytes <= 0 or ttl_seconds <= 0:
            raise StreamContractError("journal limits must be positive")
        self.stream_id = stream_id
        self.max_events = max_events
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self._created = (
            time.monotonic() if created_monotonic is None else created_monotonic
        )
        self._events: deque[AgentStreamEventV2] = deque()
        self._bytes = 0
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
            if len(self._events) + 1 > self.max_events:
                raise StreamJournalCapacityError("stream event limit exceeded")
            if self._bytes + event_bytes > self.max_bytes:
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

    def _ensure_not_expired(self) -> None:
        if time.monotonic() - self._created > self.ttl_seconds:
            raise StreamJournalExpiredError("stream journal expired")


__all__ = [
    "StreamJournal",
    "StreamJournalCapacityError",
    "StreamJournalExpiredError",
    "StreamJournalSequenceError",
]
