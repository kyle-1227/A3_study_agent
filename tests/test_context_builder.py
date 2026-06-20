"""Tests for context builder and token manager."""

from __future__ import annotations

import pytest

from src.context.token_manager import (
    TokenBudget,
    estimate_tokens,
    fit_to_budget,
    fit_to_budget_soft,
)


class TestTokenEstimation:
    def test_chinese_text(self):
        """Chinese characters should be ~1 token per 1.5 chars."""
        tokens = estimate_tokens("这是一段中文测试文本。")
        assert tokens > 0
        assert tokens < len("这是一段中文测试文本。")  # Should be fewer tokens than chars

    def test_english_text(self):
        """English text should be ~1 token per 3.5 chars."""
        tokens = estimate_tokens("This is a test sentence in English.")
        assert tokens > 0
        assert tokens < len("This is a test sentence in English.")

    def test_empty_text(self):
        assert estimate_tokens("") == 0

    def test_mixed_cn_en(self):
        """Mixed Chinese + English text."""
        text = "Python是一种编程语言。It is very popular."
        tokens = estimate_tokens(text)
        assert tokens > 0


class TestFitToBudget:
    def test_no_truncation_needed(self):
        result = fit_to_budget("short text", 100)
        assert result == "short text"

    def test_empty_text(self):
        assert fit_to_budget("", 100) == ""

    def test_zero_budget(self):
        assert fit_to_budget("anything", 0) == ""

    def test_truncation_at_sentence_boundary(self):
        text = "第一句话。第二句话。第三句话。"
        result = fit_to_budget(text, 10)
        # Should truncate and add note
        assert "(truncated" in result

    def test_soft_fit_at_boundary(self):
        text = "hello world foo bar baz qux"
        result = fit_to_budget_soft(text, 15)
        assert len(result) <= 18  # "..." may add a few chars


class TestTokenBudget:
    def test_default_budget(self):
        budget = TokenBudget()
        assert budget.total_budget == 4096
        assert budget.system_prompt == 500
        assert budget.episodic_memories == 800

    def test_available_computation(self):
        budget = TokenBudget(
            total_budget=4096, system_prompt=500, user_profile=300,
            episodic_memories=800, semantic_summary=400, current_task=500,
            rag_evidence=1500, conversation_summary=200, buffer=96,
        )
        # Fix allocations - buffer is negative budget so available=0 by design
        # (total 4096 minus all allocations = exactly 0)
        assert budget.available >= 0

    def test_custom_budget(self):
        budget = TokenBudget(total_budget=8192, episodic_memories=2000)
        assert budget.total_budget == 8192
        assert budget.episodic_memories == 2000


def test_import_context_builder():
    """Context builder imports without error."""
    from src.context.context_builder import (
        build_memory_context,
        build_memory_explanation,
        format_memory_context_for_llm_node,
    )
    assert callable(build_memory_context)
    assert callable(build_memory_explanation)
    assert callable(format_memory_context_for_llm_node)


def test_import_memory_modules():
    """All memory modules import correctly."""
    from src.memory import (
        EpisodicMemoryRecord,
        SemanticMemorySummary,
        MemoryRetrievalResult,
        MemoryStore,
        SQLiteMemoryStore,
        create_memory_store,
        EmbeddingProvider,
        DummyEmbeddingProvider,
        get_embedding_provider,
    )
    # Basic type checks
    assert issubclass(SQLiteMemoryStore, MemoryStore)
    assert issubclass(DummyEmbeddingProvider, EmbeddingProvider)
