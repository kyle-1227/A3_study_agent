"""Regression tests for Phase 2 scope boundaries."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_phase2_providers_do_not_depend_on_context_packer_or_final_selection():
    offenders = []
    for path in Path("src/context_engineering/providers").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "context_engineering.packing" in text or "pack_context_items" in text:
            offenders.append(str(path))

    assert offenders == []


def test_phase2_keeps_model_window_and_memory_budget_config_unchanged():
    settings = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))

    assert (
        settings["context_engineering"]["model_limits"]["deepseek-v4-pro"] == 1_000_000
    )
    assert settings["memory"]["token_budget"]["total_budget"] == 4096


def test_phase2_provider_config_defaults_to_shadow_non_strict():
    settings = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
    providers = settings["context_engineering"]["providers"]

    assert providers["enabled"] is True
    assert providers["shadow_mode"] is True
    assert providers["strict"] is False


def test_phase2_providers_do_not_import_active_retrieval_or_llm_clients():
    forbidden = [
        "retrieve_top_k_memories",
        "get_embedding_provider",
        "get_node_llm",
        "invoke_plain_llm",
        "tavily",
        "web_search",
    ]
    offenders = []
    for path in Path("src/context_engineering/providers").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            if pattern in text:
                offenders.append(f"{path}:{pattern}")

    assert offenders == []


def test_src_python_still_does_not_hardcode_deepseek_window():
    offenders = []
    for path in Path("src").rglob("*.py"):
        if "1000000" in path.read_text(encoding="utf-8"):
            offenders.append(str(path))

    assert offenders == []
