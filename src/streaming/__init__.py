"""Versioned streaming contracts and runtime helpers."""

from src.streaming.contracts import (
    AGENT_STREAM_SCHEMA_VERSION,
    AgentStreamEventType,
    AgentStreamEventV2,
    AuthoritativeTerminal,
    ContentBlockPayloadV1,
    StreamContractError,
    StreamEventSequencer,
)
from src.streaming.adapter import adapt_legacy_sse_stream
from src.streaming.journal import (
    StreamJournal,
    StreamJournalCapacityError,
    StreamJournalExpiredError,
    StreamJournalSequenceError,
)
from src.streaming.sse import encode_sse_event, parse_last_event_id
from src.streaming.provisional import (
    emit_provisional_event,
    reset_provisional_event_sink,
    set_provisional_event_sink,
)

__all__ = [
    "AGENT_STREAM_SCHEMA_VERSION",
    "AgentStreamEventType",
    "AgentStreamEventV2",
    "AuthoritativeTerminal",
    "ContentBlockPayloadV1",
    "StreamContractError",
    "StreamEventSequencer",
    "StreamJournal",
    "StreamJournalCapacityError",
    "StreamJournalExpiredError",
    "StreamJournalSequenceError",
    "adapt_legacy_sse_stream",
    "encode_sse_event",
    "emit_provisional_event",
    "parse_last_event_id",
    "reset_provisional_event_sink",
    "set_provisional_event_sink",
]
