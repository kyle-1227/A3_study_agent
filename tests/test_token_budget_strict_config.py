"""Strict config tests for legacy memory TokenBudget."""

from __future__ import annotations

import pytest

from src.context.errors import ContextConfigError
from src.context.token_manager import TokenBudget


VALID_BUDGET = {
    "memory.token_budget.system_prompt": 500,
    "memory.token_budget.user_profile": 300,
    "memory.token_budget.episodic_memories": 800,
    "memory.token_budget.semantic_summary": 400,
    "memory.token_budget.current_task": 500,
    "memory.token_budget.rag_evidence": 1500,
    "memory.token_budget.conversation_summary": 200,
    "memory.token_budget.total_budget": 4096,
    "memory.token_budget.buffer": 96,
}


def _settings(values: dict[str, object]):
    def _get_setting(key: str, default=None):
        return values.get(key, default)

    return _get_setting


def test_complete_config_loads(monkeypatch):
    import src.context.token_manager as token_manager

    monkeypatch.setattr(token_manager, "get_setting", _settings(VALID_BUDGET))

    budget = TokenBudget.from_settings()

    assert budget.total_budget == 4096
    assert budget.buffer == 96


def test_missing_field_fails(monkeypatch):
    import src.context.token_manager as token_manager

    values = dict(VALID_BUDGET)
    values.pop("memory.token_budget.total_budget")
    monkeypatch.setattr(token_manager, "get_setting", _settings(values))

    with pytest.raises(
        ContextConfigError, match="memory.token_budget.total_budget is required"
    ):
        TokenBudget.from_settings()


def test_non_integer_fails(monkeypatch):
    import src.context.token_manager as token_manager

    values = dict(VALID_BUDGET)
    values["memory.token_budget.system_prompt"] = "500"
    monkeypatch.setattr(token_manager, "get_setting", _settings(values))

    with pytest.raises(ContextConfigError, match="system_prompt.*non-negative integer"):
        TokenBudget.from_settings()


def test_bool_is_not_integer(monkeypatch):
    import src.context.token_manager as token_manager

    values = dict(VALID_BUDGET)
    values["memory.token_budget.system_prompt"] = True
    monkeypatch.setattr(token_manager, "get_setting", _settings(values))

    with pytest.raises(ContextConfigError, match="system_prompt.*non-negative integer"):
        TokenBudget.from_settings()


def test_negative_value_fails(monkeypatch):
    import src.context.token_manager as token_manager

    values = dict(VALID_BUDGET)
    values["memory.token_budget.rag_evidence"] = -1
    monkeypatch.setattr(token_manager, "get_setting", _settings(values))

    with pytest.raises(ContextConfigError, match="rag_evidence.*non-negative integer"):
        TokenBudget.from_settings()


def test_total_budget_must_be_positive(monkeypatch):
    import src.context.token_manager as token_manager

    values = dict(VALID_BUDGET)
    values["memory.token_budget.total_budget"] = 0
    monkeypatch.setattr(token_manager, "get_setting", _settings(values))

    with pytest.raises(ContextConfigError, match="Invalid memory.token_budget config"):
        TokenBudget.from_settings()


def test_buffer_cannot_exceed_total(monkeypatch):
    import src.context.token_manager as token_manager

    values = dict(VALID_BUDGET)
    values["memory.token_budget.buffer"] = 5000
    monkeypatch.setattr(token_manager, "get_setting", _settings(values))

    with pytest.raises(ContextConfigError, match="buffer must not exceed total_budget"):
        TokenBudget.from_settings()
