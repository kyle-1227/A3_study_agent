"""
Analytics types — data models for Learning Growth Analytics,
Cognitive Model Graph, and Explainable AI Panel.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── Growth Analytics ──────────────────────────────────────────────────────


class GrowthDataPoint(BaseModel):
    """A single point on the growth curve."""

    timestamp: str = Field(..., description="ISO 8601 timestamp")
    skill_level: float = Field(default=0.0, ge=0.0, le=1.0)
    accuracy: float = Field(default=0.0, ge=0.0, le=1.0, description="Rolling window accuracy")
    topic: str = ""
    event_count: int = Field(default=0, ge=0, description="Events in this window")
    subject: str = ""


class GrowthSeries(BaseModel):
    """Time series for one topic/subject."""

    topic: str
    subject: str = ""
    data_points: list[GrowthDataPoint] = Field(default_factory=list)
    trend: str = Field(default="stable", description="improving | stable | declining")
    current_level: float = Field(default=0.0, ge=0.0, le=1.0)


class GrowthAnalytics(BaseModel):
    """Complete growth analytics for a user."""

    user_id: str
    subject: str = ""
    days: int = 30
    series: list[GrowthSeries] = Field(default_factory=list)
    overall_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_trend: str = "stable"
    total_events: int = 0
    generated_at: str = ""


# ── Cognitive Graph ──────────────────────────────────────────────────────


class CognitiveNode(BaseModel):
    """A node in the user's cognitive graph."""

    id: str
    label: str
    type: Literal["skill", "weakness", "preference", "topic"] = "skill"
    size: float = Field(default=0.5, ge=0.0, le=1.0, description="Visual size for rendering")
    level: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    details: str = ""
    color: str = ""  # hex color hint


class CognitiveEdge(BaseModel):
    """An edge connecting cognitive nodes."""

    source: str
    target: str
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    label: str = Field(default="", description="requires | related_to | prefers | strengthens")


class CognitiveGraphData(BaseModel):
    """Complete cognitive graph for a user."""

    user_id: str
    nodes: list[CognitiveNode] = Field(default_factory=list)
    edges: list[CognitiveEdge] = Field(default_factory=list)
    summary: str = ""
    node_count: int = 0
    edge_count: int = 0


# ── Explainable AI ────────────────────────────────────────────────────────


class DecisionTrace(BaseModel):
    """A single agent decision with reasoning chain."""

    trace_id: str = Field(default="")
    node_name: str = ""
    timestamp: str = ""
    decision: str = ""
    evidence: str = ""
    reasoning_steps: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    thread_id: str = ""
    subject: str = ""


class DecisionTraceList(BaseModel):
    """Paginated list of decision traces."""

    user_id: str
    traces: list[DecisionTrace] = Field(default_factory=list)
    total: int = 0


# ── Dashboard Aggregator ──────────────────────────────────────────────────


class DashboardData(BaseModel):
    """Aggregated dashboard data for one API call."""

    user_id: str
    growth: GrowthAnalytics | None = None
    cognitive_graph: CognitiveGraphData | None = None
    recent_decisions: DecisionTraceList | None = None
    stats_summary: dict = Field(default_factory=dict)
