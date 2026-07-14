"""Regression guards to keep Phase 3B-1 out of Phase 4 capabilities."""

from __future__ import annotations

from pathlib import Path

import yaml


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
        "summarizer",
        "compaction",
        "compact",
        "rerank",
        "embedding",
    ]

    for pattern in forbidden:
        assert pattern not in source


def test_context_window_is_preserved_and_legacy_memory_budget_is_removed():
    settings_text = Path("config/settings.yaml").read_text(encoding="utf-8")
    settings = yaml.safe_load(settings_text)

    assert (
        settings["context_engineering"]["model_limits"]["deepseek-v4-pro"] == 1_000_000
    )
    assert "token_budget" not in settings["memory"]
    assert "apply_to_llm: false" in settings_text
