"""Strict public contracts for the agent_stream_v2 protocol."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, TypeAlias, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


AGENT_STREAM_SCHEMA_VERSION = "agent_stream_v2"

AgentStreamEventType: TypeAlias = Literal[
    "stream_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "activity_update",
    "tool_progress",
    "artifact_progress",
    "qa_final",
    "resource_final",
    "interrupt",
    "stopped",
    "stream_error",
    "stream_done",
]
AuthoritativeTerminal: TypeAlias = Literal[
    "qa_final",
    "resource_final",
    "interrupt",
    "stopped",
    "stream_error",
]
ContentBlockType: TypeAlias = Literal["markdown", "text", "tool"]

_TERMINAL_EVENTS: frozenset[str] = frozenset(
    {"qa_final", "resource_final", "interrupt", "stopped", "stream_error"}
)


class StreamContractError(RuntimeError):
    """Raised when a producer violates the public stream state machine."""


class ContentBlockPayloadV1(BaseModel):
    """Safe content-block metadata shared by block start/delta/stop events."""

    model_config = ConfigDict(extra="forbid")

    block_id: str = Field(min_length=1, max_length=160)
    block_index: int = Field(ge=0)
    block_type: ContentBlockType
    provisional: bool
    delta: str = Field(default="", max_length=65536)


class AgentStreamEventV2(BaseModel):
    """One ordered, replay-safe event in an agent stream."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["agent_stream_v2"] = AGENT_STREAM_SCHEMA_VERSION
    type: AgentStreamEventType
    stream_id: str = Field(min_length=1, max_length=160)
    event_id: str = Field(min_length=1, max_length=220)
    sequence: int = Field(ge=1)
    request_id: str = Field(min_length=1, max_length=160)
    thread_id: str = Field(min_length=1, max_length=160)
    created_at: datetime
    data: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_identity(self) -> "AgentStreamEventV2":
        try:
            UUID(self.request_id)
        except ValueError as exc:
            raise ValueError("request_id must be a UUID") from exc
        expected_event_id = f"{self.stream_id}:{self.sequence}"
        if self.event_id != expected_event_id:
            raise ValueError("event_id must equal '<stream_id>:<sequence>'")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        return self


class StreamEventSequencer:
    """Create ordered events while enforcing one authoritative terminal."""

    def __init__(self, *, stream_id: str, request_id: str, thread_id: str) -> None:
        if not stream_id.strip() or not thread_id.strip():
            raise StreamContractError("stream_id and thread_id are required")
        try:
            UUID(request_id)
        except ValueError as exc:
            raise StreamContractError("request_id must be a UUID") from exc
        self.stream_id = stream_id
        self.request_id = request_id
        self.thread_id = thread_id
        self._sequence = 0
        self._terminal: AuthoritativeTerminal | None = None
        self._done = False

    @property
    def sequence(self) -> int:
        return self._sequence

    @property
    def terminal(self) -> AuthoritativeTerminal | None:
        return self._terminal

    def emit(
        self,
        event_type: AgentStreamEventType,
        data: dict[str, Any] | None = None,
    ) -> AgentStreamEventV2:
        if self._done:
            raise StreamContractError("cannot emit after stream_done")
        if self._terminal is not None and event_type != "stream_done":
            raise StreamContractError("only stream_done may follow a terminal event")
        if event_type == "stream_done" and self._terminal is None:
            raise StreamContractError(
                "stream_done requires an authoritative terminal event"
            )
        if event_type in _TERMINAL_EVENTS:
            if self._terminal is not None:
                raise StreamContractError("only one authoritative terminal is allowed")
            self._terminal = cast(AuthoritativeTerminal, event_type)

        self._sequence += 1
        if event_type == "stream_done":
            self._done = True
        return AgentStreamEventV2(
            type=event_type,
            stream_id=self.stream_id,
            event_id=f"{self.stream_id}:{self._sequence}",
            sequence=self._sequence,
            request_id=self.request_id,
            thread_id=self.thread_id,
            created_at=datetime.now(timezone.utc),
            data=dict(data or {}),
        )


__all__ = [
    "AGENT_STREAM_SCHEMA_VERSION",
    "AgentStreamEventType",
    "AgentStreamEventV2",
    "AuthoritativeTerminal",
    "ContentBlockPayloadV1",
    "StreamContractError",
    "StreamEventSequencer",
]
