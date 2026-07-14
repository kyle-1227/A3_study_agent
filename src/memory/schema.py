"""
Memory schema — Long-term memory data models for episodic and semantic memory.

Design principles:
- Episodic memories are atomic events with importance scoring
- Semantic memories are LLM-generated summaries that aggregate N episodic events
- Every record carries an embedding for vector similarity search
- All timestamps are ISO 8601 UTC
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


# ── Episodic Memory ───────────────────────────────────────────────────────


class EpisodicMemoryRecord(BaseModel):
    """A single learning event or observation recorded by the agent.

    Examples:
        - Quiz attempt: correctness, subject, knowledge points
        - Learning behavior: studied a resource, asked follow-ups
        - Error: hallucination detected, misunderstanding
        - Key conversation: important interaction summary
    """

    memory_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique memory identifier",
    )
    user_id: str = Field(..., description="User this memory belongs to")
    memory_type: Literal[
        "quiz_attempt",
        "learning_behavior",
        "error",
        "key_conversation",
        "system_event",
    ] = Field(default="key_conversation", description="Category of the memory event")

    content: str = Field(..., description="Natural language description of the event")
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How important this memory is (0=trivial, 1=critical)",
    )
    subject: str = Field(
        default="", description="Academic subject, e.g. 'python', 'math'"
    )

    # Arbitrary metadata for extensibility
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible key-value metadata",
    )

    # Vector embedding for similarity search
    embedding: list[float] | None = Field(
        default=None,
        description="Vector embedding of content for similarity search",
    )

    # Lifecycle tracking
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp of record creation",
    )
    last_accessed_at: str = Field(
        default="", description="Last time this memory was retrieved"
    )
    access_count: int = Field(
        default=0, ge=0, description="How many times this memory was retrieved"
    )

    # Consolidation tracking
    consolidated: bool = Field(
        default=False,
        description="Whether this memory has been included in a semantic summary",
    )
    consolidation_group: str = Field(
        default="",
        description="ID of the semantic summary batch that consumed this memory",
    )


# ── Semantic Memory ───────────────────────────────────────────────────────


class SemanticMemorySummary(BaseModel):
    """LLM-generated summary that aggregates N episodic memories.

    This is the "compressed" form of episodic memory — it captures
    patterns, trends, and persistent traits rather than individual events.
    """

    summary_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique summary identifier",
    )
    user_id: str = Field(..., description="User this summary belongs to")
    source_episodic_ids: list[str] = Field(
        default_factory=list,
        description="IDs of episodic memories that were summarized",
    )

    content: str = Field(
        ...,
        description="Natural language summary of the aggregated episodic events",
    )

    # Structured extracted traits
    weak_knowledge_points: list[str] = Field(
        default_factory=list,
        description="Knowledge areas the user consistently struggles with",
    )
    learning_style_changes: str = Field(
        default="",
        description="Detected changes in learning preferences",
    )
    skill_growth_trajectory: str = Field(
        default="",
        description="Summary of skill improvement or stagnation",
    )

    # Embedding for retrieval
    embedding: list[float] | None = Field(
        default=None,
        description="Vector embedding of the summary content",
    )

    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp of summary creation",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How confident the LLM is in this summary",
    )
    consolidation_version: int = Field(
        default=1,
        ge=1,
        description="How many times this summary has been re-consolidated",
    )


# ── Strict-output DTO for LLM summarization ───────────────────────────────


_WeakPointText = Annotated[str, Field(max_length=80)]
_StyleChangeText = Annotated[str, Field(max_length=400)]
_TrajectoryText = Annotated[str, Field(max_length=400)]
_SummaryContentText = Annotated[str, Field(max_length=1000)]


class SemanticSummaryStrictOutput(BaseModel):
    """DeepSeek strict-tool schema for memory consolidation.

    The LLM receives N episodic memory texts and produces this structured output.
    """

    content: _SummaryContentText = Field(
        default="",
        description="Concise natural-language summary (2-4 sentences) capturing key learning patterns",
    )
    weak_knowledge_points: list[_WeakPointText] = Field(
        default_factory=list,
        max_length=10,
        description="Specific knowledge points the learner struggled with",
    )
    learning_style_changes: _StyleChangeText = Field(
        default="",
        description="Any detectable changes in how the user prefers to learn",
    )
    skill_growth_trajectory: _TrajectoryText = Field(
        default="",
        description="What skills improved or stayed flat across the summarized events",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How confident the LLM is in this consolidation",
    )


# ── Retrieval Result ──────────────────────────────────────────────────────


class MemoryRetrievalResult(BaseModel):
    """One matched memory with relevance scoring metadata.

    Carries the full memory object plus per-signal scores for transparency.
    """

    memory: EpisodicMemoryRecord | SemanticMemorySummary = Field(
        ..., description="The retrieved memory record"
    )
    memory_type: Literal["episodic", "semantic"] = Field(
        ..., description="Whether this is episodic or semantic memory"
    )
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Combined relevance score (keyword + vector + importance boost)",
    )
    keyword_score: float = Field(default=0.0, ge=0.0, le=1.0)
    vector_score: float = Field(default=0.0, ge=0.0, le=1.0)
    match_reason: str = Field(
        default="",
        description="Why this memory matched, e.g. 'subject_match+keyword_overlap'",
    )
