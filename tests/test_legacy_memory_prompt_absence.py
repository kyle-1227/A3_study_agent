"""Regression guard for the removed legacy memory prompt implementation."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_legacy_memory_prompt_package_and_runtime_symbols_are_absent():
    assert not list(Path("src/context").glob("*.py"))

    forbidden_symbols = (
        "build_memory_" + "context",
        "build_memory_" + "explanation",
        "format_memory_context_" + "for_llm_node",
        "MemoryContext" + "Injection",
        "MEMORY_CONTEXT_" + "HEADER",
        "MEMORY_CONTEXT_EPISODIC_" + "HEADER",
        "MEMORY_CONTEXT_SEMANTIC_" + "HEADER",
        "MEMORY_CONTEXT_CONVERSATION_" + "HEADER",
        "MEMORY_CONTEXT_" + "FOOTER",
        "MEMORY_INFLUENCE_" + "EXPLANATION_TEMPLATE",
    )
    offenders: list[str] = []
    for path in Path("src").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for symbol in forbidden_symbols:
            if symbol in source:
                offenders.append(f"{path}:{symbol}")

    assert offenders == []


def test_legacy_memory_only_budget_is_absent():
    settings = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))

    assert "token_budget" not in settings["memory"]
