"""Typed assessment errors with bounded, sanitized diagnostics."""

from __future__ import annotations

import re
from typing import Any

_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)(cookie\s*[:=]\s*)[^;\n]+"),
    re.compile(r"(?i)(x-api-key\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]+"),
)


def sanitize_assessment_error(message: Any, *, max_chars: int = 500) -> str:
    """Redact common secret shapes and bound error text for diagnostics."""
    text = str(message or "").replace("\n", " ").strip()
    for pattern in _SECRET_PATTERNS:
        if pattern.pattern.startswith("sk-"):
            text = pattern.sub("sk-[REDACTED]", text)
        elif pattern.pattern.startswith("sk-or"):
            text = pattern.sub("sk-or-v1-[REDACTED]", text)
        else:
            text = pattern.sub(r"\1[REDACTED]", text)
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


class ErrorClassificationFailed(RuntimeError):
    """Raised when quiz-error classification cannot produce a valid result."""

    def __init__(
        self,
        *,
        quiz_topic: str,
        knowledge_points: list[str],
        failure_stage: str,
        original_exception_type: str,
        error_message: Any,
    ) -> None:
        self.quiz_topic = quiz_topic
        self.knowledge_points = list(knowledge_points)
        self.failure_stage = failure_stage
        self.original_exception_type = original_exception_type
        self.sanitized_error_message = sanitize_assessment_error(error_message)
        super().__init__(
            "Error classification failed "
            f"(quiz_topic={quiz_topic or 'unknown'}, "
            f"knowledge_points={self.knowledge_points}, "
            f"failure_stage={failure_stage}, "
            f"original_exception_type={original_exception_type}, "
            f"error_message={self.sanitized_error_message})"
        )
