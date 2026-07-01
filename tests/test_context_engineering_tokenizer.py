"""Tokenizer tests for the Context Engineering Kernel."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.context_engineering import tokenizer
from src.context_engineering.schema import ContextConfigError


def _patch_tokenizer_settings(
    monkeypatch, *, mode: str = "estimated_mixed", estimated: bool = True
) -> None:
    config = {
        "enabled": True,
        "strict": True,
        "tokenizer": {"mode": mode, "estimated": estimated},
    }
    monkeypatch.setattr(
        tokenizer,
        "get_setting",
        lambda key, default=None: config if key == "context_engineering" else default,
    )


def test_estimated_mixed_counts_english_chinese_mixed_and_empty(monkeypatch):
    _patch_tokenizer_settings(monkeypatch)

    english = tokenizer.count_text_tokens("hello world")
    chinese = tokenizer.count_text_tokens("你好世界")
    mixed = tokenizer.count_text_tokens("hello你好")
    empty = tokenizer.count_text_tokens("")

    assert english.value == 4
    assert chinese.value == 3
    assert mixed.value == 4
    assert empty.value == 0
    assert english.estimated is True
    assert english.method == "estimated_mixed"


def test_count_messages_accepts_langchain_like_and_openai_style_messages(monkeypatch):
    _patch_tokenizer_settings(monkeypatch)

    result = tokenizer.count_messages_tokens(
        [
            SimpleNamespace(content="hello"),
            {"role": "user", "content": [{"type": "text", "text": "你好"}]},
            {"role": "assistant", "content": "world"},
        ]
    )

    assert result.value == 6
    assert result.estimated is True
    assert result.method == "estimated_mixed"


def test_count_messages_reads_tokenizer_config_once(monkeypatch):
    calls = 0
    config = {
        "enabled": True,
        "strict": True,
        "tokenizer": {"mode": "estimated_mixed", "estimated": True},
    }

    def fake_get_setting(key, default=None):
        nonlocal calls
        if key == "context_engineering":
            calls += 1
            return config
        return default

    monkeypatch.setattr(tokenizer, "get_setting", fake_get_setting)

    result = tokenizer.count_messages_tokens(
        [
            SimpleNamespace(content="hello"),
            {"role": "user", "content": "world"},
            {"role": "assistant", "content": [{"type": "text", "text": "abc"}]},
        ]
    )

    assert result.value == 5
    assert calls == 1


def test_legacy_adapter_estimate_is_pure_when_context_engineering_disabled(
    monkeypatch,
):
    import src.observability.context_usage as context_usage

    monkeypatch.setattr(
        tokenizer,
        "get_setting",
        lambda key, default=None: {"enabled": False}
        if key == "context_engineering"
        else default,
    )
    monkeypatch.setattr(
        context_usage,
        "emit_context_usage",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy estimate wrappers must not emit telemetry")
        ),
    )

    assert context_usage.estimate_tokens_from_text("hello") == 2
    assert (
        context_usage.estimate_messages_tokens(
            [
                SimpleNamespace(content="hello"),
                {"role": "user", "content": [{"type": "text", "text": "abc"}]},
            ]
        )
        == 3
    )


def test_tokenizer_rejects_unsupported_mode(monkeypatch):
    _patch_tokenizer_settings(monkeypatch, mode="not_real")

    with pytest.raises(ContextConfigError, match="tokenizer_mode_unsupported"):
        tokenizer.count_text_tokens("hello")
