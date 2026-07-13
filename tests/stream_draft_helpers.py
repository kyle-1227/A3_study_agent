"""Helpers for asserting native ``agent_stream_v2`` producer drafts."""

from __future__ import annotations

from collections.abc import Iterable

from src.streaming.contracts import AgentStreamEventDraftV2


_PROGRESS_EVENTS = frozenset({"activity_update", "tool_progress", "artifact_progress"})


def draft_payload(event: AgentStreamEventDraftV2) -> dict:
    """Expose the semantic payload without recreating a legacy SSE frame."""

    if event.type in _PROGRESS_EVENTS:
        kind = event.data.get("kind")
        payload = event.data.get("payload")
        assert isinstance(kind, str) and kind
        assert isinstance(payload, dict)
        return {"type": kind, **payload}
    return {"type": event.type, **event.data}


def draft_payloads(events: Iterable[AgentStreamEventDraftV2]) -> list[dict]:
    return [draft_payload(event) for event in events]
