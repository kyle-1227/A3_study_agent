"""Retention rules for immutable episodic facts."""

from __future__ import annotations


LEARNING_GUIDANCE_HISTORY_ID_PREFIX = "learning-guidance-history:v1:"
PROTECTED_EPISODIC_MEMORY_ID_PREFIXES = (
    LEARNING_GUIDANCE_HISTORY_ID_PREFIX,
)


def is_protected_episodic_memory_id(memory_id: str) -> bool:
    return any(
        memory_id.startswith(prefix)
        for prefix in PROTECTED_EPISODIC_MEMORY_ID_PREFIXES
    )


__all__ = [
    "LEARNING_GUIDANCE_HISTORY_ID_PREFIX",
    "PROTECTED_EPISODIC_MEMORY_ID_PREFIXES",
    "is_protected_episodic_memory_id",
]
