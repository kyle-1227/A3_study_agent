"""Budget and configuration tests for Context Engineering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.context_engineering import budget, policies, tokenizer
from src.context_engineering.schema import ContextBudget, ContextConfigError


def _base_settings(
    *, strict: bool = True, model_limits: dict[str, int] | None = None
) -> dict[str, Any]:
    return {
        "context_engineering": {
            "enabled": True,
            "strict": strict,
            "tokenizer": {"mode": "estimated_mixed", "estimated": True},
            "model_limits": model_limits
            if model_limits is not None
            else {"deepseek-v4-pro": 1_000_000},
            "thresholds": {
                "warning_ratio": 0.70,
                "critical_ratio": 0.85,
                "compact_ratio": 0.90,
            },
            "default_reserved_output_tokens": 16000,
        },
        "llm": {"study_plan": {"max_tokens": 4096}},
    }


def _lookup(settings: dict[str, Any], key: str, default: Any = None) -> Any:
    current: Any = settings
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _patch_settings(monkeypatch, settings: dict[str, Any]) -> None:
    def fake_get_setting(key: str, default: Any = None) -> Any:
        return _lookup(settings, key, default)

    monkeypatch.setattr(budget, "get_setting", fake_get_setting)
    monkeypatch.setattr(policies, "get_setting", fake_get_setting)
    monkeypatch.setattr(tokenizer, "get_setting", fake_get_setting)


def test_reads_deepseek_v4_pro_window_from_formal_config():
    from src.config import clear_cache

    clear_cache()
    assert budget.get_model_context_limit("deepseek-v4-pro") == 1_000_000


def test_enabled_false_is_noop_and_does_not_check_model_limits(monkeypatch):
    _patch_settings(monkeypatch, {"context_engineering": {"enabled": False}})

    stage, payload = budget.build_context_usage_payload(
        node_name="node",
        llm_node="missing_llm",
        provider="provider",
        model="unknown-model",
        messages=[{"content": "hello"}],
    )

    assert stage == ""
    assert payload is None


def test_strict_missing_model_window_fails_fast(monkeypatch):
    _patch_settings(monkeypatch, _base_settings(strict=True, model_limits={}))

    with pytest.raises(ContextConfigError, match="model_window_unknown"):
        budget.build_context_usage_payload(
            node_name="node",
            llm_node="study_plan",
            provider="provider",
            model="unknown-model",
            messages=[],
        )


def test_non_strict_missing_model_window_emits_error_payload(monkeypatch):
    _patch_settings(monkeypatch, _base_settings(strict=False, model_limits={}))

    stage, payload = budget.build_context_usage_payload(
        node_name="node",
        llm_node="study_plan",
        provider="provider",
        model="unknown-model",
        messages=[],
    )

    assert stage == "context_usage_error"
    assert payload is not None
    assert payload["reason"] == "model_window_unknown"
    assert payload["warning"] == "model context window is unknown"
    assert "max_context_tokens" not in payload


def test_reserved_output_at_or_above_window_fails_fast(monkeypatch):
    _patch_settings(monkeypatch, _base_settings(model_limits={"small-model": 100}))

    with pytest.raises(ContextConfigError, match="reserved_output_tokens"):
        budget.build_context_budget(
            node_name="node",
            llm_node="study_plan",
            model="small-model",
        )


def test_warning_critical_and_overflow_levels(monkeypatch):
    _patch_settings(monkeypatch, _base_settings(model_limits={"model": 1000}))
    context_budget = ContextBudget(
        node_name="node",
        llm_node="llm",
        model="model",
        max_context_tokens=1000,
        reserved_output_tokens=100,
        max_input_tokens=900,
        warning_ratio=0.70,
        critical_ratio=0.85,
        compact_ratio=0.90,
    )

    warning = budget.compute_context_usage(
        messages=[{"content": "x" * 2100}],
        budget=context_budget,
        provider="provider",
    )
    critical = budget.compute_context_usage(
        messages=[{"content": "x" * 2625}],
        budget=context_budget,
        provider="provider",
    )
    overflow = budget.compute_context_usage(
        messages=[{"content": "x" * 3500}],
        budget=context_budget,
        provider="provider",
    )

    assert warning.warning_level == "warning"
    assert critical.warning_level == "critical"
    assert overflow.warning_level == "overflow"


def test_src_python_does_not_hardcode_deepseek_window():
    offenders = []
    for path in Path("src").rglob("*.py"):
        if "1000000" in path.read_text(encoding="utf-8"):
            offenders.append(str(path))

    assert offenders == []
