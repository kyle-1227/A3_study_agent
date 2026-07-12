"""Request-local, non-logging sink for user-visible provisional stream events."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any, Callable, Literal, TypeAlias


ProvisionalEventType: TypeAlias = Literal[
    "qa_provisional_start",
    "qa_provisional_delta",
    "qa_provisional_stop",
    "qa_provisional_reset",
    "tool_progress",
]
ProvisionalSink: TypeAlias = Callable[[dict[str, Any]], None]

_SINK: ContextVar[ProvisionalSink | None] = ContextVar(
    "a3_provisional_stream_sink",
    default=None,
)


def set_provisional_event_sink(sink: ProvisionalSink) -> Token[ProvisionalSink | None]:
    """Bind a request-local sink without routing content through application logs."""

    return _SINK.set(sink)


def reset_provisional_event_sink(token: Token[ProvisionalSink | None]) -> None:
    _SINK.reset(token)


def emit_provisional_event(
    event_type: ProvisionalEventType,
    *,
    node_name: str,
    request_id: str,
    thread_id: str,
    delta: str = "",
    answer_chars: int = 0,
    reason: str = "",
) -> None:
    """Deliver a bounded UI event only when the active stream installed a sink."""

    if len(delta) > 65536:
        raise ValueError("provisional delta exceeds 65536 characters")
    sink = _SINK.get()
    if sink is None:
        return
    sink(
        {
            "type": event_type,
            "node_name": str(node_name)[:120],
            "request_id": str(request_id)[:160],
            "thread_id": str(thread_id)[:160],
            "delta": delta,
            "answer_chars": max(0, int(answer_chars)),
            "reason": str(reason)[:120],
        }
    )


__all__ = [
    "emit_provisional_event",
    "reset_provisional_event_sink",
    "set_provisional_event_sink",
]
