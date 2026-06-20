"""
Context Engineering Layer — Memory-augmented prompt construction.

Components:
- Context Builder: assembles memory-augmented system prompts from retrieved memories
- Token Manager: token budget allocation and truncation for context window safety

Usage::

    from src.context import (
        build_memory_context, build_memory_explanation,
        TokenBudget, estimate_tokens, fit_to_budget,
    )
"""

from src.context.context_builder import (
    build_memory_context,
    build_memory_explanation,
)
from src.context.token_manager import (
    TokenBudget,
    estimate_tokens,
    fit_to_budget,
)

__all__ = [
    "build_memory_context",
    "build_memory_explanation",
    "TokenBudget",
    "estimate_tokens",
    "fit_to_budget",
]
