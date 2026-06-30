"""Tests for message context provider."""

from __future__ import annotations

from src.context_engineering.providers.base import ProviderContext
from src.context_engineering.providers.message_provider import MessageContextProvider


def test_message_provider_objectizes_current_query_and_recent_messages():
    provider = MessageContextProvider()
    context = ProviderContext(
        node_name="generate_answer",
        llm_node="academic",
        user_query=None,
        current_user_message_index=None,
        state={},
        messages=[
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current question"},
        ],
        request_id=None,
        thread_id=None,
        max_items_per_provider=10,
        max_content_chars_per_item=4000,
    )

    items = provider.collect(context)

    assert items[0].title == "current_user_query"
    assert items[0].content == "current question"
    assert items[0].priority == 100
    assert items[0].can_drop is False
    assert all(item.source_type == "message" for item in items)
    assert not any(
        item.title != "current_user_query" and item.content == "current question"
        for item in items
    )


def test_message_provider_uses_explicit_user_query_without_mutating_messages():
    messages = [{"role": "assistant", "content": "previous answer"}]
    context = ProviderContext(
        node_name="node",
        llm_node="llm",
        user_query="explicit query",
        current_user_message_index=None,
        state={},
        messages=messages,
        request_id=None,
        thread_id=None,
        max_items_per_provider=10,
        max_content_chars_per_item=4000,
    )

    items = MessageContextProvider().collect(context)

    assert items[0].content == "explicit query"
    assert messages == [{"role": "assistant", "content": "previous answer"}]


def test_message_provider_does_not_duplicate_explicit_current_query_message():
    context = ProviderContext(
        node_name="node",
        llm_node="llm",
        user_query="current question",
        current_user_message_index=1,
        state={},
        messages=[
            {"role": "assistant", "content": "previous answer"},
            {"role": "user", "content": "current question"},
        ],
        request_id="r1",
        thread_id="t1",
        max_items_per_provider=10,
        max_content_chars_per_item=4000,
    )

    items = MessageContextProvider().collect(context)

    assert [item.content for item in items].count("current question") == 1
    assert items[0].metadata["message_index"] == 1


def test_current_user_query_id_uses_safe_identity_fields():
    messages = [{"role": "user", "content": "same visible question"}]
    first = MessageContextProvider().collect(
        ProviderContext(
            node_name="node",
            llm_node="llm",
            user_query="same visible question",
            current_user_message_index=0,
            state={},
            messages=messages,
            request_id="request-a",
            thread_id="thread-a",
            max_items_per_provider=10,
            max_content_chars_per_item=4000,
        )
    )[0]
    second = MessageContextProvider().collect(
        ProviderContext(
            node_name="node",
            llm_node="llm",
            user_query="same visible question",
            current_user_message_index=0,
            state={},
            messages=messages,
            request_id="request-b",
            thread_id="thread-a",
            max_items_per_provider=10,
            max_content_chars_per_item=4000,
        )
    )[0]
    different_query = MessageContextProvider().collect(
        ProviderContext(
            node_name="node",
            llm_node="llm",
            user_query="different visible question",
            current_user_message_index=0,
            state={},
            messages=[{"role": "user", "content": "different visible question"}],
            request_id="request-a",
            thread_id="thread-a",
            max_items_per_provider=10,
            max_content_chars_per_item=4000,
        )
    )[0]

    assert first.id != second.id
    assert first.id != different_query.id
    assert "same visible question" not in first.metadata.values()
    assert "content_hash" in first.metadata


def test_message_provider_respects_max_items_before_recent_items():
    context = ProviderContext(
        node_name="node",
        llm_node="llm",
        user_query=None,
        current_user_message_index=None,
        state={},
        messages=[
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current question"},
        ],
        request_id=None,
        thread_id=None,
        max_items_per_provider=1,
        max_content_chars_per_item=4000,
    )

    items = MessageContextProvider().collect(context)

    assert len(items) == 1
    assert items[0].title == "current_user_query"
