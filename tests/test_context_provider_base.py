"""Base provider contract tests."""

from __future__ import annotations

from src.context_engineering.itemizer import make_context_item
from src.context_engineering.providers.base import ProviderContext
from src.context_engineering.providers.registry import (
    ContextProviderSettings,
    collect_context_items,
    collect_context_items_by_source,
)


class _Provider:
    name = "unit_provider"
    source_type = "message"

    def collect(self, context: ProviderContext):
        return [
            make_context_item(
                source_type="message",
                title="unit",
                content=context.user_query or "",
                priority=90,
                scope="turn",
                lifetime="turn",
                compressible=False,
                can_drop=False,
                disclosure_level="full",
                metadata={"source": "unit"},
                max_content_chars=context.max_content_chars_per_item,
            )
        ]


def _context() -> ProviderContext:
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


def _settings() -> ContextProviderSettings:
    return ContextProviderSettings(
        enabled=True,
        shadow_mode=True,
        strict=False,
        enabled_sources=("message",),
        max_items_per_provider=10,
        max_content_chars_per_item=4000,
        trace_top_items=10,
    )


def test_collect_context_items_returns_provider_items():
    items = collect_context_items(
        _context(), providers=[_Provider()], settings=_settings()
    )

    assert len(items) == 1
    assert items[0].content == "question"


def test_collect_context_items_by_source_groups_items():
    grouped = collect_context_items_by_source(
        _context(),
        providers=[_Provider()],
        settings=_settings(),
    )

    assert list(grouped) == ["message"]
    assert grouped["message"][0].title == "unit"
