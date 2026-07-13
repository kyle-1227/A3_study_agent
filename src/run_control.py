"""Run-control helpers for safe LangGraph stop/continue behavior."""

from __future__ import annotations

import inspect
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from langgraph.types import interrupt

from src.observability.activity import merge_activity_timeline
from src.context_engineering.compaction import (
    CompactBoundaryV1,
    CompactionResultV1,
    ConversationSummaryV2,
    ProviderBoundUsageV1,
)
from src.context_engineering.session_memory import SessionContextMemoryLedgerV1
from src.context_engineering.thread_window_v3 import ThreadContextWindowV3
from src.observability.context_usage_report import merge_context_usage_report_history
from src.observability.contracts import ContextUsageReport


RUN_CONTROL_SCHEMA_VERSION = "run_control_v1"
RUN_STATUS_RUNNING = "running"
RUN_STATUS_STOPPING = "stopping"
RUN_STATUS_STOPPED = "stopped"
RUN_STATUS_CONTINUING = "continuing"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_ERROR = "failed"
RUN_STATUS_IDLE = "idle"
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
        self._active_runs: dict[str, dict[str, Any]] = {}

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

    def start_active_run(
        self, thread_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        snapshot = _safe_active_run_payload(thread_id, payload)
        with self._lock:
            self._active_runs[thread_id] = snapshot
        return dict(snapshot)

    def update_active_run(
        self, thread_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        update = _safe_active_run_payload(thread_id, payload)
        with self._lock:
            current = dict(self._active_runs.get(thread_id) or {"thread_id": thread_id})
            current.update(update)
            self._active_runs[thread_id] = current
            return dict(current)

    def finish_active_run(
        self,
        thread_id: str,
        terminal_payload: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._active_runs.pop(thread_id, None)

    def get_active_run(self, thread_id: str | None) -> dict[str, Any] | None:
        if not thread_id:
            return None
        with self._lock:
            snapshot = self._active_runs.get(thread_id)
            return dict(snapshot) if snapshot is not None else None


run_control_registry = RunControlRegistry()


def _safe_active_run_payload(thread_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "thread_id",
        "schema_version",
        "run_status",
        "resume_available",
        "pending_interrupt_type",
        "current_node",
        "last_completed_node",
        "stopped_at",
        "stop_reason",
        "request_context_window",
        "thread_context_window",
        "thread_context_window_v2",
        "thread_context_window_v3",
        "session_context_memory_ledger",
        "last_provider_dispatch",
        "compact_boundary",
        "conversation_summary_v2",
        "compaction_result",
        "profile_completion_request",
        "context_usage",
        "context_usage_history",
        "context_usage_report",
        "context_usage_reports",
        "activity_timeline",
        "graph_version",
        "llm_input_manifest",
        "llm_input_manifests",
        "thread_context_ledger",
        "background_context_window",
        "context_continuity",
        "context_influence_ledger",
        "resource_artifacts_by_type",
        "last_generated_artifacts",
        "last_resource_final_payload",
        "last_qa_response",
        "missing_run_control_fields",
        "message",
    }
    result: dict[str, Any] = {"thread_id": str(thread_id)}
    for key, value in (payload or {}).items():
        if key not in allowed:
            continue
        if key == "activity_timeline" and isinstance(value, list):
            result[key] = merge_activity_timeline([], value)
            continue
        if key == "context_usage_reports" and isinstance(value, list):
            result[key] = merge_context_usage_report_history([], value)
            continue
        if key == "context_usage_report" and isinstance(value, dict):
            try:
                result[key] = ContextUsageReport.model_validate(value).model_dump(
                    mode="json"
                )
            except Exception:
                continue
            continue
        if key == "thread_context_window_v3" and isinstance(value, dict) and value:
            result[key] = ThreadContextWindowV3.model_validate(value).model_dump(
                mode="json"
            )
            continue
        if key == "session_context_memory_ledger" and isinstance(value, dict) and value:
            result[key] = SessionContextMemoryLedgerV1.model_validate(value).model_dump(
                mode="json"
            )
            continue
        if key == "last_provider_dispatch" and isinstance(value, dict) and value:
            result[key] = ProviderBoundUsageV1.model_validate(value).model_dump(
                mode="json"
            )
            continue
        if key == "compact_boundary" and isinstance(value, dict) and value:
            result[key] = CompactBoundaryV1.model_validate(value).model_dump(
                mode="json"
            )
            continue
        if key == "conversation_summary_v2" and isinstance(value, dict) and value:
            result[key] = ConversationSummaryV2.model_validate(value).model_dump(
                mode="json"
            )
            continue
        if key == "compaction_result" and isinstance(value, dict) and value:
            result[key] = CompactionResultV1.model_validate(value).model_dump(
                mode="json"
            )
            continue
        result[key] = _safe_active_value(value)
    return result


def _safe_active_value(value: Any) -> Any:
    if isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value[:512]
    if isinstance(value, list):
        return [_safe_active_value(item) for item in value[:50]]
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in list(value.items())[:50]:
            safe[str(key)[:120]] = _safe_active_value(item)
        return safe
    if value is None:
        return None
    return str(value)[:512]


def start_active_run(thread_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return run_control_registry.start_active_run(thread_id, payload)


def update_active_run(thread_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return run_control_registry.update_active_run(thread_id, payload)


def finish_active_run(
    thread_id: str,
    terminal_payload: dict[str, Any] | None = None,
) -> None:
    run_control_registry.finish_active_run(thread_id, terminal_payload)


def get_active_run(thread_id: str | None) -> dict[str, Any] | None:
    return run_control_registry.get_active_run(thread_id)


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


def wrap_interruptible_node(
    node_name: str, node_fn: Callable[..., Any]
) -> Callable[..., Any]:
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
