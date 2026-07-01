"""
Token Manager — token budget allocation and text truncation for context window safety.

Design:
- TokenBudget model with per-component allocations, loaded from settings.yaml
- estimate_tokens() provides rough token estimation for mixed Chinese/English text
- fit_to_budget() truncates text preserving sentence boundaries where possible

Note: This uses character-based estimation rather than a real tokenizer to avoid
adding a heavy dependency (e.g., tiktoken). For Chinese text (~1.5 chars/token)
and English text (~3.5 chars/token), this is accurate enough for budget management.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError, model_validator

from src.config import get_setting
from src.context.errors import ContextConfigError


_TOKEN_BUDGET_FIELDS = (
    "system_prompt",
    "user_profile",
    "episodic_memories",
    "semantic_summary",
    "current_task",
    "rag_evidence",
    "conversation_summary",
    "total_budget",
    "buffer",
)


class TokenBudget(BaseModel):
    """Legacy memory-level character allocations for context assembly.

    All values are approximate character counts (not actual LLM tokens).
    Characters are used because Chinese and English have different token ratios,
    and we want a simple, fast budget without a tokenizer dependency.

    This is a legacy memory-only budget. It is not the model's maximum context
    window and must not be used as an agent-wide Context Engineering budget.
    Model windows should be configured explicitly in the Phase 1
    context_engineering.model_limits work.
    """

    system_prompt: int = Field(ge=0)
    user_profile: int = Field(ge=0)
    episodic_memories: int = Field(ge=0)
    semantic_summary: int = Field(ge=0)
    current_task: int = Field(ge=0)
    rag_evidence: int = Field(ge=0)
    conversation_summary: int = Field(ge=0)
    total_budget: int = Field(gt=0)
    buffer: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_budget_relationships(self) -> "TokenBudget":
        if self.buffer > self.total_budget:
            raise ValueError("memory.token_budget.buffer must not exceed total_budget")
        return self

    @classmethod
    def from_settings(cls) -> "TokenBudget":
        """Load budget from settings.yaml memory.token_budget section."""
        values = {
            field_name: _required_non_negative_int(f"memory.token_budget.{field_name}")
            for field_name in _TOKEN_BUDGET_FIELDS
        }
        try:
            return cls(**values)
        except ValidationError as exc:
            raise ContextConfigError(
                f"Invalid memory.token_budget config: {exc}"
            ) from exc

    @property
    def available(self) -> int:
        """Remaining budget after all fixed allocations."""
        used = (
            self.system_prompt
            + self.user_profile
            + self.episodic_memories
            + self.semantic_summary
            + self.current_task
            + self.rag_evidence
            + self.conversation_summary
            + self.buffer
        )
        return max(0, self.total_budget - used)


def estimate_tokens(text: str) -> int:
    """Rough token estimation for mixed Chinese/English text.

    Heuristic:
    - Chinese characters (CJK): ~1 token per 1.5 characters
    - Other characters (English, numbers, punctuation): ~1 token per 3.5 characters

    This is NOT exact — it's designed to be fast and dependency-free.
    For accurate token counting, use tiktoken with the model's tokenizer.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Approximate token count.
    """
    if not text:
        return 0

    chinese_chars = 0
    other_chars = 0

    for c in text:
        if "一" <= c <= "鿿" or "㐀" <= c <= "䶿":
            chinese_chars += 1
        elif "　" <= c <= "〿":
            # CJK punctuation — closer to Chinese ratio
            chinese_chars += 1
        else:
            other_chars += 1

    # Chinese: ~1 token per 1.5 chars (common for DeepSeek/ChatGLM tokenizers)
    # English: ~1 token per 3.5 chars (conservative for subword tokenizers)
    chinese_tokens = chinese_chars / 1.5
    other_tokens = other_chars / 3.5

    return int(chinese_tokens + other_tokens)


def fit_to_budget(text: str, max_chars: int) -> str:
    """Truncate text to fit within a character budget.

    Tries to preserve sentence boundaries by cutting at the last
    sentence-ending punctuation before the budget limit.

    Args:
        text: The text to truncate.
        max_chars: Maximum characters allowed.

    Returns:
        Truncated text, potentially with an ellipsis note appended.
    """
    if not text:
        return ""

    if max_chars <= 0:
        return ""

    if len(text) <= max_chars:
        return text

    # Find a good truncation point
    target_len = max(50, max_chars - 20)  # leave room for truncation note

    # Try to cut at sentence boundary within the target region
    search_start = max(0, target_len - 100)
    search_region = text[search_start:max_chars]

    # Chinese sentence endings
    for sep in ("。", "！", "？", "\n\n", "\n"):
        pos = search_region.rfind(sep)
        if pos >= 0:
            cut = search_start + pos + len(sep)
            return text[:cut] + "\n...(truncated for token budget)"

    # English sentence endings
    for sep in (". ", "! ", "? "):
        pos = search_region.rfind(sep)
        if pos >= 0:
            cut = search_start + pos + len(sep)
            return text[:cut] + "\n...(truncated for token budget)"

    # Fallback: hard cut
    truncated = text[:target_len]
    return truncated + "...(truncated)"


def fit_to_budget_soft(text: str, max_chars: int) -> str:
    """Truncate text to fit budget, prefer word boundary.

    Unlike fit_to_budget, this is more aggressive — it cuts at any space
    or punctuation within the last 50 chars of the budget.

    Args:
        text: The text to truncate.
        max_chars: Maximum characters allowed.

    Returns:
        Truncated text.
    """
    if not text or max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    # Find last space within budget
    truncated = text[:max_chars]
    for sep in (" ", "，", "。", "\n", "、"):
        last = truncated.rfind(sep)
        if last > max(0, max_chars - 50):
            return truncated[:last] + "..."

    return truncated[: max_chars - 3] + "..."


def _required_non_negative_int(key: str) -> int:
    value = get_setting(key)
    if value is None:
        raise ContextConfigError(f"{key} is required")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContextConfigError(f"{key} must be a non-negative integer")
    if value < 0:
        raise ContextConfigError(f"{key} must be a non-negative integer")
    return value
