"""Registry tests for ContextProvider shadow collection."""

from __future__ import annotations

import logging

import pytest

from src.context_engineering.itemizer import make_context_item
from src.context_engineering.providers.base import ProviderContext
from src.context_engineering.providers import registry
from src.context_engineering.providers.registry import (
    ContextProviderSettings,
    collect_context_items,
    emit_context_items_shadow,
    get_context_provider_settings,
)
from src.context_engineering.schema import ContextProviderError
from src.observability.a3_trace import reset_trace_event_sink, set_trace_event_sink


class _GoodProvider:
    name = "good"
    source_type = "message"

    def collect(self, context: ProviderContext):
        return [
            make_context_item(
                source_type="message",
                title="good",
                content="safe content",
                priority=80,
                scope="turn",
                lifetime="turn",
                compressible=True,
                can_drop=True,
                disclosure_level="snippet",
                metadata={"source": "good"},
                max_content_chars=context.max_content_chars_per_item,
            )
        ]


class _FailingProvider:
    name = "bad"
    source_type = "memory"

    def collect(self, context: ProviderContext):
        raise ContextProviderError(
            provider=self.name,
            source_type=self.source_type,
            stage="collect",
            message="boom api_key=sk-secret",
            original_exception_type="RuntimeError",
        )


def _settings(*, strict: bool = False, enabled: bool = True) -> ContextProviderSettings:
    return ContextProviderSettings(
        enabled=enabled,
        shadow_mode=True,
        strict=strict,
        enabled_sources=("message", "memory"),
        max_items_per_provider=10,
        max_content_chars_per_item=4000,
        trace_top_items=10,
    )


def _provider_context() -> ProviderContext:
    return ProviderContext(
        node_name="node",
        llm_node="llm",
        user_query="question",
        current_user_message_index=0,
        state={"request_id": "r1", "thread_id": "t1"},
        messages=[{"role": "user", "content": "question"}],
        request_id="r1",
        thread_id="t1",
        max_items_per_provider=10,
        max_content_chars_per_item=4000,
    )


def _fake_settings(
    *,
    provider_strict: bool,
    providers_enabled: bool = True,
    enabled_sources: list[str] | None = None,
):
    sources = enabled_sources or ["message", "memory"]

    def fake_get_setting(key: str, default=None):
        if key == "context_engineering":
            return {
                "enabled": True,
                "strict": True,
                "providers": {
                    "enabled": providers_enabled,
                    "shadow_mode": True,
                    "strict": provider_strict,
                    "enabled_sources": sources,
                    "max_items_per_provider": 10,
                    "max_content_chars_per_item": 4000,
                    "trace_top_items": 10,
                },
            }
        return default

    return fake_get_setting


def test_provider_settings_use_providers_strict_not_phase1_strict(monkeypatch):
    monkeypatch.setattr(registry, "get_setting", _fake_settings(provider_strict=False))

    settings = get_context_provider_settings()

    assert settings.strict is False


def test_collect_context_items_strict_false_keeps_good_items():
    items = collect_context_items(
        _provider_context(),
        providers=[_FailingProvider(), _GoodProvider()],
        settings=_settings(strict=False),
    )

    assert len(items) == 1
    assert items[0].title == "good"


def test_collect_context_items_strict_true_fails_fast():
    with pytest.raises(ContextProviderError):
        collect_context_items(
            _provider_context(),
            providers=[_FailingProvider(), _GoodProvider()],
            settings=_settings(strict=True),
        )


def test_emit_context_items_shadow_sends_error_and_collected_events(monkeypatch):
    monkeypatch.setattr(registry, "get_setting", _fake_settings(provider_strict=False))
    monkeypatch.setattr(
        registry,
        "get_default_providers",
        lambda settings=None: [_FailingProvider(), _GoodProvider()],
    )
    sink: list[dict] = []
    token = set_trace_event_sink(sink)
    try:
        items = emit_context_items_shadow(
            logging.getLogger("test_context_provider_registry"),
            node_name="node",
            llm_node="llm",
            messages=[{"role": "user", "content": "question"}],
            state={"request_id": "r1", "thread_id": "t1"},
        )
    finally:
        reset_trace_event_sink(token)

    assert len(items) == 1
    stages = [event["stage"] for event in sink]
    assert "context_provider_error" in stages
    assert "context_items_collected" in stages
    serialized = repr(sink).lower()
    assert "sk-secret" not in serialized
    assert "api_key" not in serialized


def test_emit_context_items_shadow_enabled_false_sends_no_events(monkeypatch):
    monkeypatch.setattr(
        registry,
        "get_setting",
        _fake_settings(provider_strict=False, providers_enabled=False),
    )
    sink: list[dict] = []
    token = set_trace_event_sink(sink)
    try:
        items = emit_context_items_shadow(
            logging.getLogger("test_context_provider_registry"),
            node_name="node",
            llm_node="llm",
            messages=[{"role": "user", "content": "question"}],
            state={"request_id": "r1", "thread_id": "t1"},
        )
    finally:
        reset_trace_event_sink(token)

    assert items == []
    assert sink == []


def test_emit_context_items_shadow_default_message_provider_does_not_duplicate_current_query(
    monkeypatch,
):
    monkeypatch.setattr(
        registry,
        "get_setting",
        _fake_settings(provider_strict=False, enabled_sources=["message"]),
    )
    sink: list[dict] = []
    token = set_trace_event_sink(sink)
    try:
        items = emit_context_items_shadow(
            logging.getLogger("test_context_provider_registry"),
            node_name="node",
            llm_node="llm",
            messages=[
                {"role": "assistant", "content": "previous answer"},
                {"role": "user", "content": "current question"},
            ],
            state={"request_id": "r1", "thread_id": "t1"},
        )
    finally:
        reset_trace_event_sink(token)

    assert [item.content for item in items].count("current question") == 1
    current_items = [item for item in items if item.title == "current_user_query"]
    assert len(current_items) == 1
    assert current_items[0].metadata["message_index"] == 1
    assert current_items[0].metadata["request_id"] == "r1"
    assert current_items[0].metadata["thread_id"] == "t1"
    assert "content_hash" in current_items[0].metadata
