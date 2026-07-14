"""Versioned streaming contracts and runtime helpers."""

from src.streaming.contracts import (
    AGENT_STREAM_SCHEMA_VERSION,
    AgentStreamEventDraftV2,
    AgentStreamEventType,
    AgentStreamEventV2,
    AuthoritativeTerminal,
    ContentBlockPayloadV1,
    StreamContractError,
    StreamEventSequencer,
)
from src.streaming.journal import (
    StreamJournal,
    StreamJournalCapacityError,
    StreamJournalExpiredError,
    StreamJournalSequenceError,
)
from src.streaming.sse import encode_sse_event, parse_last_event_id
from src.streaming.evidence_progress import (
    EVIDENCE_PROGRESS_SCHEMA_VERSION,
    EvidenceProgressV1,
    build_evidence_progress,
    evidence_progress_sink_active,
    publish_evidence_progress,
    reset_evidence_progress_sink,
    set_evidence_progress_sink,
)
from src.streaming.provisional import (
    emit_provisional_event,
    reset_provisional_event_sink,
    set_provisional_event_sink,
)

__all__ = [
    "AGENT_STREAM_SCHEMA_VERSION",
    "AgentStreamEventDraftV2",
    "AgentStreamEventType",
    "AgentStreamEventV2",
    "AuthoritativeTerminal",
    "ContentBlockPayloadV1",
    "EVIDENCE_PROGRESS_SCHEMA_VERSION",
    "EvidenceProgressV1",
    "StreamContractError",
    "StreamEventSequencer",
    "StreamJournal",
    "StreamJournalCapacityError",
    "StreamJournalExpiredError",
    "StreamJournalSequenceError",
    "encode_sse_event",
    "build_evidence_progress",
    "evidence_progress_sink_active",
    "emit_provisional_event",
    "parse_last_event_id",
    "publish_evidence_progress",
    "reset_evidence_progress_sink",
    "reset_provisional_event_sink",
    "set_provisional_event_sink",
    "set_evidence_progress_sink",
]
