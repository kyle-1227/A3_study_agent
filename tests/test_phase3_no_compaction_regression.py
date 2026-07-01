"""Regression guard for Phase 3A scope boundaries."""

from __future__ import annotations

from pathlib import Path


def test_packing_module_has_no_forbidden_runtime_capabilities():
    root = Path("src/context_engineering/packing")
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in root.glob("*.py")
        if path.name != "__init__.py"
    )

    forbidden = [
        "get_node_llm",
        "retrieve_top_k_memories",
        "get_embedding_provider",
        "web_search",
        "tavily",
        "write_memory",
        "save_memory",
        "upsert_memory",
        "compaction",
        "compact",
        "summarizer",
    ]
    for pattern in forbidden:
        assert pattern not in text


def test_phase3_does_not_modify_model_limits_or_memory_budget():
    settings = Path("config/settings.yaml").read_text(encoding="utf-8")

    assert "deepseek-v4-pro: 1000000" in settings
    assert "total_budget: 4096" in settings
