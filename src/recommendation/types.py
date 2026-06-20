"""
Recommendation types — data models for the Memory-driven Recommendation Engine.

Score breakdowns, individual recommendations, and ranked recommendation lists.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ScoreBreakdown(BaseModel):
    """Transparent decomposition of a recommendation's combined score."""

    weakness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    forgetting_score: float = Field(default=0.0, ge=0.0, le=1.0)
    preference_score: float = Field(default=0.0, ge=0.0, le=1.0)
    goal_score: float = Field(default=0.0, ge=0.0, le=1.0)
    combined_score: float = Field(default=0.0, ge=0.0, le=1.0)
    weights: dict[str, float] = Field(default_factory=dict)


class Recommendation(BaseModel):
    """A single ranked learning resource recommendation with explainability."""

    resource_type: Literal["doc", "quiz", "mindmap", "review_doc", "case"] = Field(
        default="quiz", description="Type of learning resource"
    )
    subject: str = Field(default="", description="Academic subject")
    topic: str = Field(default="", description="Specific topic name")
    title: str = Field(default="", description="Resource title/name")
    priority: float = Field(default=0.5, ge=0.0, le=1.0, description="Overall priority score")
    reason: str = Field(default="", description="Human-readable explanation of why this was recommended")
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    suggested_interval_days: int = Field(default=0, ge=0, description="Days until spaced repetition review")
    knowledge_points: list[str] = Field(default_factory=list)  # Associated KPs
    user_skill_gap: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="How much the user needs this (1-skill_level*confidence)",
    )


class RecommendationList(BaseModel):
    """Ranked list of recommendations with generation context."""

    user_id: str
    generated_at: str
    items: list[Recommendation] = Field(default_factory=list)
    context_summary: str = Field(
        default="",
        description="What profile/memory data drove this recommendation list",
    )
    total_candidates_considered: int = 0
    generation_time_ms: float = 0.0
