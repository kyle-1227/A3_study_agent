"""
Adaptive Learning Loop — quiz assessment → error classification → adaptive practice → spaced repetition.

Forms a closed loop: learning → assessment → memory update → adaptive practice →
recommendation → learning path update.
"""

from src.assessment.types import (
    AdaptiveTask,
    ErrorClassification,
    ErrorClassificationStrict,
    QuizAttemptResult,
    ReviewSchedule,
)
from src.assessment.classifier import classify_error
from src.assessment.practice_generator import generate_adaptive_practice
from src.assessment.scheduler import SpacedRepetitionScheduler, get_due_reviews

__all__ = [
    "AdaptiveTask",
    "ErrorClassification",
    "ErrorClassificationStrict",
    "QuizAttemptResult",
    "ReviewSchedule",
    "classify_error",
    "generate_adaptive_practice",
    "SpacedRepetitionScheduler",
    "get_due_reviews",
]
