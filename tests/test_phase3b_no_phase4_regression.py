"""Regression guards to keep Phase 3B-1 out of Phase 4 capabilities."""

from __future__ import annotations

from pathlib import Path


def test_context_apply_module_does_not_implement_phase4_capabilities():
    source = "\n".join(
        [
            Path("src/context_engineering/packing/apply.py").read_text(
                encoding="utf-8"
            ),
            Path("src/context_engineering/packing/apply_trace.py").read_text(
                encoding="utf-8"
            ),
        ]
    ).lower()

    forbidden = [
        "get_node_llm",
        "retrieve_top_k_memories",
        "get_embedding_provider",
        "web_search",
        "tavily",
        "write_memory",
        "save_memory",
        "upsert_memory",
        "summary",
        "summarizer",
        "compaction",
        "compact",
        "rerank",
        "embedding",
    ]

    for pattern in forbidden:
        assert pattern not in source


def test_context_window_and_memory_budget_are_not_changed():
    settings = Path("config/settings.yaml").read_text(encoding="utf-8")

    assert "deepseek-v4-pro: 1000000" in settings
    assert "total_budget: 4096" in settings
    assert "apply_to_llm: false" in settings
