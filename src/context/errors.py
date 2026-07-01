"""Typed context configuration errors."""

from __future__ import annotations


class ContextConfigError(RuntimeError):
    """Raised when context-related configuration is missing or invalid."""
