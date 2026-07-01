"""Run-control helpers for safe LangGraph stop/continue behavior."""

from __future__ import annotations

import inspect
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from langgraph.types import interrupt


RUN_CONTROL_SCHEMA_VERSION = "run_control_v1"
RUN_STATUS_RUNNING = "running"
RUN_STATUS_STOPPING = "stopping"
RUN_STATUS_STOPPED = "stopped"
RUN_STATUS_CONTINUING = "continuing"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_ERROR = "error"
RUN_STATUS_NOT_RESUMABLE = "not_resumable"
RUN_STATUS_UNKNOWN = "unknown"

RUN_CONTROL_FIELDS = [
    "schema_version",
    "run_status",
    "stop_requested",
    "stop_reason",
    "current_node",
    "last_completed_node",
    "resume_available",
    "stopped_at",
    "pending_interrupt_type",
    "context_usage",
    "context_usage_history",
]

CONTEXT_USAGE_HISTORY_LIMIT = 30


@dataclass(frozen=True)
class StopSignal:
    """Process-local marker used to ask the next graph node to interrupt."""

    thread_id: str
    reason: str
    requested_at: str

    def to_payload(self) -> dict[str, str]:
        return asdict(self)


class RunControlRegistry:
    """Thread-safe process-local stop signal registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._signals: dict[str, StopSignal] = {}

    def request_stop(self, thread_id: str, reason: str) -> StopSignal:
        signal = StopSignal(
            thread_id=thread_id,
            reason=reason,
            requested_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._signals[thread_id] = signal
        return signal

    def get_stop_signal(self, thread_id: str | None) -> StopSignal | None:
        if not thread_id:
            return None
        with self._lock:
            return self._signals.get(thread_id)

    def clear_stop_signal(self, thread_id: str | None) -> None:
        if not thread_id:
            return
        with self._lock:
            self._signals.pop(thread_id, None)


run_control_registry = RunControlRegistry()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def trim_context_usage_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(history or [])[-CONTEXT_USAGE_HISTORY_LIMIT:]


def _state_thread_id(state: dict[str, Any]) -> str:
    return str(state.get("thread_id") or state.get("session_id") or "")


def stop_requested_for_state(state: dict[str, Any]) -> StopSignal | None:
    """Return the active stop signal if this node should pause before work."""
    thread_id = _state_thread_id(state)
    signal = run_control_registry.get_stop_signal(thread_id)
    if signal is not None:
        return signal
    if state.get("stop_requested") is True:
        return StopSignal(
            thread_id=thread_id,
            reason=str(state.get("stop_reason") or "user_stop"),
            requested_at=str(state.get("stop_requested_at") or utc_now_iso()),
        )
    return None


def wrap_interruptible_node(node_name: str, node_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a sync or async LangGraph node with a pre-node user stop gate."""

    async def _wrapped(state: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        signal = stop_requested_for_state(state)
        if signal is not None:
            resume_value = interrupt(
                {
                    "type": "user_stop",
                    "node": node_name,
                    "thread_id": signal.thread_id,
                    "reason": signal.reason,
                    "requested_at": signal.requested_at,
                }
            )
            if isinstance(resume_value, dict):
                action = str(resume_value.get("action") or "")
            else:
                action = str(resume_value or "")
            if action != "continue":
                raise ValueError(f"Invalid user_stop resume action: {action!r}")

        result = node_fn(state, *args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    _wrapped.__name__ = getattr(node_fn, "__name__", f"{node_name}_run_control_wrapper")
    _wrapped.__doc__ = getattr(node_fn, "__doc__", None)
    return _wrapped
