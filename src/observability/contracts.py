"""Versioned, safe observability contracts shared by API, SSE, and state."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

GRAPH_MANIFEST_SCHEMA_VERSION = "graph_manifest_v1"
ACTIVITY_EVENT_SCHEMA_VERSION = "activity_event_v1"
CONTEXT_USAGE_REPORT_SCHEMA_VERSION = "context_usage_report_v1"

ActivityKind = Literal[
    "node",
    "llm",
    "tool",
    "retrieval",
    "evidence_progress",
    "context",
    "review",
    "interrupt",
    "artifact",
    "retry",
    "stream",
]
ActivityStatus = Literal[
    "queued",
    "running",
    "waiting",
    "completed",
    "retrying",
    "interrupted",
    "failed",
    "skipped",
]
ContextMainCategory = Literal[
    "system_prompt",
    "tool_definitions",
    "rules",
    "skills",
    "subagent_definitions",
    "conversation",
    "unclassified",
]
ContextReconciliationWarning = Literal[
    "segments_compacted",
    "unclassified_tokens_present",
]


def _validate_aware_utc_iso(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("timestamp is required")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    offset = parsed.utcoffset()
    if parsed.tzinfo is None or offset is None:
        raise ValueError("timestamp must be timezone-aware")
    if offset.total_seconds() != 0:
        raise ValueError("timestamp must use UTC")
    return parsed.isoformat()


class GraphManifestNode(BaseModel):
    """One executable graph node or registry-owned logical resource subnode."""

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, max_length=160)
    label: str = Field(min_length=1, max_length=160)
    description: str = Field(max_length=320)
    kind: str = Field(min_length=1, max_length=80)
    group: str = Field(min_length=1, max_length=80)
    parent: str = Field(default="", max_length=160)
    workflow: str = Field(default="", max_length=120)
    order: int = Field(ge=0)
    stage_rank: int = Field(ge=0)
    visible: bool
    logical: bool
    activity_running: str = Field(max_length=200)
    activity_completed: str = Field(max_length=200)


class GraphManifestEdge(BaseModel):
    """One deterministic executable or logical topology edge."""

    model_config = ConfigDict(extra="forbid")

    edge_id: str = Field(min_length=1, max_length=180)
    source: str = Field(min_length=1, max_length=160)
    target: str = Field(min_length=1, max_length=160)
    kind: Literal["graph", "logical"]
    conditional: bool
    label: str = Field(default="", max_length=160)
    workflow: str = Field(default="", max_length=120)


class GraphManifest(BaseModel):
    """Backend source of truth for the graph and logical resource topology."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["graph_manifest_v1"]
    graph_version: str = Field(min_length=1, max_length=180)
    generated_at: str
    nodes: list[GraphManifestNode]
    edges: list[GraphManifestEdge]
    capability_metadata: dict[str, Any]

    _generated_at_utc = field_validator("generated_at")(_validate_aware_utc_iso)

    @model_validator(mode="after")
    def validate_topology(self) -> "GraphManifest":
        if not self.graph_version.startswith("graph:v1:"):
            raise ValueError("graph_version prefix is invalid")
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("graph manifest contains duplicate node ids")
        valid_ids = set(node_ids)
        for edge in self.edges:
            if edge.source not in valid_ids or edge.target not in valid_ids:
                raise ValueError("graph manifest edge references an unknown node")
        return self


class GraphManifestUnavailable(BaseModel):
    """Typed error body used when startup introspection cannot build a manifest."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["graph_manifest_error_v1"]
    error: Literal["graph_manifest_unavailable"]
    reason: str = Field(min_length=1, max_length=160)
    error_type: str = Field(min_length=1, max_length=120)


class ActivityEvent(BaseModel):
    """One safe, idempotent timeline activity."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["activity_event_v1"]
    activity_id: str = Field(min_length=1, max_length=180)
    sequence: int = Field(ge=1)
    thread_id: str = Field(min_length=1, max_length=120)
    request_id: str = Field(min_length=1, max_length=120)
    kind: ActivityKind
    status: ActivityStatus
    node: str = Field(default="", max_length=160)
    parent: str = Field(default="", max_length=160)
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(default="", max_length=320)
    tool: str = Field(default="", max_length=120)
    model: str = Field(default="", max_length=160)
    started_at: str
    updated_at: str
    completed_at: str = ""
    duration_ms: int | None = Field(default=None, ge=0)
    safe_details: dict[str, Any] = Field(default_factory=dict)

    _started_at_utc = field_validator("started_at")(_validate_aware_utc_iso)
    _updated_at_utc = field_validator("updated_at")(_validate_aware_utc_iso)

    @field_validator("completed_at")
    @classmethod
    def validate_completed_at(cls, value: str) -> str:
        return _validate_aware_utc_iso(value) if value else ""

    @model_validator(mode="after")
    def validate_completion(self) -> "ActivityEvent":
        terminal = self.status in {"completed", "interrupted", "failed", "skipped"}
        if terminal and not self.completed_at:
            raise ValueError("terminal activity requires completed_at")
        if not self.activity_id.startswith("activity:v1:"):
            raise ValueError("activity_id prefix is invalid")
        return self


class ContextUsageCategory(BaseModel):
    """One reconciled context usage rollup."""

    model_config = ConfigDict(extra="forbid")

    category: str = Field(min_length=1, max_length=120)
    estimated_tokens: int = Field(ge=0)
    segment_count: int = Field(ge=0)
    message_count: int = Field(ge=0)


class ContextUsageSegment(BaseModel):
    """Content-free accounting for one provider message or CE provenance slice."""

    model_config = ConfigDict(extra="forbid")

    segment_id: str = Field(min_length=1, max_length=180)
    fingerprint: str = Field(min_length=16, max_length=80)
    message_index: int = Field(ge=0)
    role: str = Field(min_length=1, max_length=40)
    main_category: ContextMainCategory
    detailed_category: str = Field(min_length=1, max_length=120)
    char_count: int = Field(ge=0)
    estimated_tokens: int = Field(ge=0)
    provenance: dict[str, str] = Field(default_factory=dict)

    @field_validator("provenance")
    @classmethod
    def validate_provenance(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > 6:
            raise ValueError("segment provenance exceeds item bound")
        for key, item in value.items():
            if not key or len(key) > 40:
                raise ValueError("segment provenance key is invalid")
            if len(item) > 96:
                raise ValueError("segment provenance value exceeds character bound")
        return value


class ContextUsageReport(BaseModel):
    """Reconciled accounting for the exact provider-bound input."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["context_usage_report_v1"]
    report_id: str = Field(min_length=1, max_length=180)
    manifest_id: str = Field(min_length=1, max_length=180)
    created_at: str
    request_id: str = Field(max_length=120)
    thread_id: str = Field(max_length=120)
    node_name: str = Field(min_length=1, max_length=120)
    llm_node: str = Field(min_length=1, max_length=120)
    provider: str = Field(max_length=120)
    model: str = Field(max_length=160)
    input_estimated_tokens: int = Field(ge=0)
    output_reserved_tokens: int = Field(ge=0)
    used_tokens: int = Field(ge=0)
    max_context_tokens: int = Field(gt=0)
    available_tokens: int = Field(ge=0)
    used_ratio: float = Field(ge=0)
    warning_level: Literal["ok", "warning", "critical", "overflow"]
    estimated: bool
    tokenizer_mode: str = Field(min_length=1, max_length=80)
    message_count: int = Field(ge=0)
    schema_size_chars: int | None = Field(default=None, ge=0)
    main_categories: list[ContextUsageCategory] = Field(max_length=16)
    detailed_categories: list[ContextUsageCategory] = Field(max_length=32)
    overlap_rollups: list[ContextUsageCategory] = Field(max_length=8)
    segments: list[ContextUsageSegment] = Field(max_length=32)
    unclassified_tokens: int = Field(ge=0)
    reconciliation_ok: bool
    reconciliation_warnings: list[ContextReconciliationWarning] = Field(
        default_factory=list,
        max_length=4,
    )

    _created_at_utc = field_validator("created_at")(_validate_aware_utc_iso)

    @model_validator(mode="after")
    def validate_reconciliation(self) -> "ContextUsageReport":
        main_total = sum(item.estimated_tokens for item in self.main_categories)
        segment_total = sum(item.estimated_tokens for item in self.segments)
        if main_total != self.input_estimated_tokens:
            raise ValueError("main category tokens do not reconcile")
        if segment_total != self.input_estimated_tokens:
            raise ValueError("segment tokens do not reconcile")
        if (
            self.used_tokens
            != self.input_estimated_tokens + self.output_reserved_tokens
        ):
            raise ValueError("used_tokens does not reconcile")
        expected_available = max(self.max_context_tokens - self.used_tokens, 0)
        if self.available_tokens != expected_available:
            raise ValueError("available_tokens does not reconcile")
        if not self.reconciliation_ok:
            raise ValueError("context usage report does not reconcile")
        if not self.report_id.startswith("context_usage:v1:"):
            raise ValueError("report_id prefix is invalid")
        return self
