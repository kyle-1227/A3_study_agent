"""
Assessment types — data models for the Adaptive Learning Loop.

Quiz attempt recording, error classification, adaptive practice tasks,
and spaced repetition scheduling.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QuizAttemptResult(BaseModel):
    """A single quiz/exercise attempt, stored as episodic memory."""

    user_id: str
    subject: str = ""
    topic: str = ""
    question: str = ""
    user_answer: str = ""
    correct_answer: str = ""
    is_correct: bool = False
    knowledge_points: list[str] = Field(default_factory=list)
    difficulty_level: Literal["basic", "intermediate", "application", "self_check"] = "basic"
    attempt_number: int = Field(default=1, ge=1)
    time_spent_seconds: float = Field(default=0.0, ge=0.0)


class ErrorClassificationStrict(BaseModel):
    """DeepSeek strict-tool schema for error type classification.

    The LLM analyzes a failed quiz attempt and classifies the root cause.
    """

    error_type: Literal["concept", "logic", "implementation"] = Field(
        default="concept",
        description="concept=doesn't understand the idea, logic=wrong reasoning, implementation=syntax/detail error",
    )
    concept_gap: str = Field(
        default="",
        max_length=200,
        description="The specific concept or knowledge gap that caused the error",
    )
    suggestion: str = Field(
        default="",
        max_length=300,
        description="Brief suggestion for how to address this error",
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ErrorClassification(BaseModel):
    """Full error classification result (LLM output + quiz context)."""

    error_type: Literal["concept", "logic", "implementation"]
    concept_gap: str
    suggestion: str
    confidence: float
    quiz_topic: str = ""
    quiz_question: str = ""
    quiz_knowledge_points: list[str] = Field(default_factory=list)


class AdaptiveTask(BaseModel):
    """A generated adaptive practice task — similar, harder, or review."""

    task_type: Literal["similar", "harder", "review"]
    subject: str
    topic: str
    question: str
    answer: str = ""
    explanation: str = ""
    knowledge_points: list[str] = Field(default_factory=list)
    difficulty: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = Field(default="", description="Why this specific task was generated")
    source_error_type: str = ""  # The error type that triggered this task


class ReviewSchedule(BaseModel):
    """Spaced repetition schedule for a specific knowledge point."""

    knowledge_point: str
    subject: str
    topic: str = ""
    intervals: list[int] = Field(default_factory=list, description="Days between reviews")
    current_interval_index: int = Field(default=0, ge=0)
    next_review_at: str = ""  # ISO timestamp
    review_count: int = Field(default=0, ge=0)
    last_performance: float = Field(default=0.0, ge=0.0, le=1.0)
    last_reviewed_at: str = ""
    is_due: bool = False
