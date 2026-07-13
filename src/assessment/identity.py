"""Stable public identities for assessment-capable exercise cards."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence


EXERCISE_QUESTION_ID_PREFIX = "question:v1"
_LEVELS = frozenset({"basic", "intermediate", "application", "self_check"})
_QUESTION_TYPES = frozenset({"free_text", "single_choice"})


def stable_exercise_question_id(
    *,
    level: str,
    question_type: str,
    question: str,
    choices: Sequence[str],
    tags: Sequence[str],
) -> str:
    """Hash the complete public question surface into a stable identity."""

    if not all(isinstance(value, str) for value in (level, question_type, question)):
        raise ValueError("exercise question identity text values must be strings")
    if any(not isinstance(value, str) for value in (*choices, *tags)):
        raise ValueError("exercise question identity choices and tags must be strings")

    normalized_level = level.strip()
    normalized_question_type = question_type.strip()
    normalized_question = question.strip()
    normalized_choices = tuple(choice.strip() for choice in choices)
    normalized_tags = tuple(sorted(tag.strip() for tag in tags))
    if (
        normalized_level not in _LEVELS
        or normalized_question_type not in _QUESTION_TYPES
        or not normalized_question
        or not normalized_tags
        or any(not choice for choice in normalized_choices)
        or any(not tag for tag in normalized_tags)
    ):
        raise ValueError(
            "exercise question identity requires a canonical level, question_type, "
            "question, choices, and tags"
        )
    if len(set(normalized_choices)) != len(normalized_choices):
        raise ValueError("exercise question identity choices must be unique")
    if len(set(normalized_tags)) != len(normalized_tags):
        raise ValueError("exercise question identity tags must be unique")
    if normalized_question_type == "free_text" and normalized_choices:
        raise ValueError("free_text exercise question identity cannot include choices")
    if normalized_question_type == "single_choice" and len(normalized_choices) < 2:
        raise ValueError(
            "single_choice exercise question identity requires at least two choices"
        )

    canonical = json.dumps(
        {
            "level": normalized_level,
            "question_type": normalized_question_type,
            "question": normalized_question,
            "choices": normalized_choices,
            "tags": normalized_tags,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{EXERCISE_QUESTION_ID_PREFIX}:{hashlib.sha256(canonical).hexdigest()}"


__all__ = ["EXERCISE_QUESTION_ID_PREFIX", "stable_exercise_question_id"]
