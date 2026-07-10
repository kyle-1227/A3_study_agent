"""Temporary A3_TRACE structured logs for multi-subject retrieval validation."""

from __future__ import annotations

import json
import logging
import os
from contextvars import ContextVar
from typing import Any


_TRACE_EVENT_SINK: ContextVar[Any] = ContextVar("a3_trace_event_sink", default=None)


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "")
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _truncate(value: Any, max_chars: int = 500, max_items: int = 20) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.replace("\n", " ").strip()
        return text[:max_chars] + ("..." if len(text) > max_chars else "")
    if isinstance(value, list):
        return [
            _truncate(item, max_chars=max_chars, max_items=max_items)
            for item in value[:max_items]
        ]
    if isinstance(value, dict):
        return {
            str(k): _truncate(v, max_chars=max_chars, max_items=max_items)
            for k, v in value.items()
        }
    return value


def _trace_ids_from_state(state: dict | None) -> dict[str, str]:
    state = state or {}
    configurable = (
        state.get("configurable") if isinstance(state.get("configurable"), dict) else {}
    )
    metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}

    def _pick(*keys: str) -> str:
        for source in (state, configurable, metadata):
            for key in keys:
                value = source.get(key)
                if value:
                    return str(value)
        return "unknown"

    thread_id = _pick("thread_id")
    session_id = _pick("session_id", "thread_id")
    request_id = _pick("request_id", "run_id")
    return {
        "request_id": request_id,
        "session_id": session_id,
        "thread_id": thread_id,
    }


def set_trace_event_sink(sink: Any):
    """Attach a best-effort per-context sink for selected trace events."""
    return _TRACE_EVENT_SINK.set(sink)


def reset_trace_event_sink(token) -> None:
    """Reset the per-context trace event sink."""
    _TRACE_EVENT_SINK.reset(token)


def emit_a3_trace(
    logger: logging.Logger,
    stage: str,
    payload: dict[str, Any],
    *,
    state: dict | None = None,
    env_flag: str = "LOG_A3_TRACE",
    level: str = "warning",
    max_chars: int = 500,
    max_items: int = 20,
) -> None:
    """
    Emit one structured A3_TRACE log line.

    Requirements:
    - Controlled by env_flag or LOG_A3_TRACE.
    - Never raise exception.
    - Truncate long values.
    """
    try:
        safe_payload = {
            "stage": stage,
            **_trace_ids_from_state(state),
            **_truncate(payload, max_chars=max_chars, max_items=max_items),
        }
        sink = _TRACE_EVENT_SINK.get()
        if sink is not None:
            try:
                sink.append(safe_payload)
            except Exception:
                pass

        if not (_env_enabled("LOG_A3_TRACE") or _env_enabled(env_flag)):
            return

        line = "A3_TRACE " + json.dumps(safe_payload, ensure_ascii=False, default=str)

        if level == "info":
            logger.info(line)
        else:
            logger.warning(line)
    except Exception:
        logger.debug("Failed to emit A3_TRACE log", exc_info=True)
