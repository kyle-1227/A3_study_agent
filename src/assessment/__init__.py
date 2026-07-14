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
from src.assessment.errors import ErrorClassificationFailed
from src.assessment.scheduler import SpacedRepetitionScheduler, get_due_reviews


def __getattr__(name: str):
    """Preserve legacy package exports without eager structured-LLM imports."""

    if name == "classify_error":
        from src.assessment.classifier import classify_error

        return classify_error
    if name == "generate_adaptive_practice":
        from src.assessment.practice_generator import generate_adaptive_practice

        return generate_adaptive_practice
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AdaptiveTask",
    "ErrorClassification",
    "ErrorClassificationStrict",
    "QuizAttemptResult",
    "ReviewSchedule",
    "ErrorClassificationFailed",
    "classify_error",
    "generate_adaptive_practice",
    "SpacedRepetitionScheduler",
    "get_due_reviews",
]
