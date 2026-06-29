"""Typed memory-layer errors."""

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


def sanitize_memory_error(message: Any, *, max_chars: int = 500) -> str:
    """Redact common secret shapes and bound memory-layer error text."""
    text = str(message or "").replace("\n", " ").strip()
    for pattern in _SECRET_PATTERNS:
        if pattern.pattern.startswith("sk-"):
            text = pattern.sub("sk-[REDACTED]", text)
        elif pattern.pattern.startswith("sk-or"):
            text = pattern.sub("sk-or-v1-[REDACTED]", text)
        else:
            text = pattern.sub(r"\1[REDACTED]", text)
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


class MemoryEmbeddingConfigError(RuntimeError):
    """Raised when memory embedding provider configuration is invalid."""


class MemoryEmbeddingRuntimeError(RuntimeError):
    """Raised when a configured memory embedding provider fails at runtime."""
