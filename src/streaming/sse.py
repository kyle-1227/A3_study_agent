"""SSE encoding and Last-Event-ID validation for agent_stream_v2."""

from __future__ import annotations

import json

from src.streaming.contracts import AgentStreamEventV2, StreamContractError


def encode_sse_event(
    event: AgentStreamEventV2,
    *,
    retry_ms: int | None = None,
) -> str:
    """Encode one validated event without exposing any extra transport data."""

    if retry_ms is not None and retry_ms <= 0:
        raise StreamContractError("retry_ms must be positive")
    lines = [f"event: {event.type}", f"id: {event.event_id}"]
    if retry_ms is not None:
        lines.append(f"retry: {retry_ms}")
    payload = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
    lines.append(f"data: {payload}")
    return "\n".join(lines) + "\n\n"


def parse_last_event_id(value: str, *, expected_stream_id: str) -> int:
    """Return the last delivered sequence after strict stream identity checks."""

    text = str(value or "").strip()
    if not text:
        return 0
    stream_id, separator, raw_sequence = text.rpartition(":")
    if not separator or stream_id != expected_stream_id:
        raise StreamContractError("Last-Event-ID does not match stream_id")
    try:
        sequence = int(raw_sequence)
    except ValueError as exc:
        raise StreamContractError("Last-Event-ID sequence must be an integer") from exc
    if sequence < 0:
        raise StreamContractError("Last-Event-ID sequence must be non-negative")
    return sequence


__all__ = ["encode_sse_event", "parse_last_event_id"]
