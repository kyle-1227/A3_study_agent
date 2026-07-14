"""Message construction tests for Phase 3B-1 context apply."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from src.context_engineering.packing.apply import (
    ContextApplyError,
    ContextInjectionPolicy,
    build_applied_messages,
    filter_injectable_items,
    render_injected_context,
    sanitize_context_content,
)
from src.context_engineering.packing.packer import pack_context_items
from src.context_engineering.schema import ContextItem


def _policy(*, max_tokens: int = 10000) -> ContextInjectionPolicy:
    return ContextInjectionPolicy(
        enabled=True,
        apply_enabled_nodes=("plain_node",),
        allow_structured_output=False,
        role="system",
        position="after_system",
        exclude_message_source=True,
        max_injected_context_tokens=max_tokens,
        injectable_sources=(
            "rules",
            "evidence",
            "memory",
            "profile",
            "artifact",
            "trajectory",
            "curriculum",
        ),
    )


def _item(
    item_id: str,
    *,
    source_type: str = "memory",
    content: str = "remember this",
) -> ContextItem:
    return ContextItem(
        id=item_id,
        source_type=source_type,
        title=item_id,
        content=content,
        token_estimate=5,
        estimated=True,
        tokenizer_mode="estimated_mixed",
        priority=80,
        relevance_score=None,
        recency_score=None,
        confidence=None,
        scope="turn",
        lifetime="turn",
        compressible=True,
        can_drop=True,
        disclosure_level="snippet",
        metadata={"label": item_id},
    )


def _packed(items: list[ContextItem]):
    return pack_context_items(
        node_name="plain_node",
        llm_node="llm",
        items=items,
        max_context_block_tokens=10000,
    )


def test_dict_messages_insert_after_initial_system_and_do_not_mutate_original():
    messages = [
        {"role": "system", "content": "base system"},
        {"role": "user", "content": "question"},
    ]
    original_snapshot = [dict(message) for message in messages]

    result = build_applied_messages(
        node_name="plain_node",
        llm_node="llm",
        original_messages=messages,
        packed=_packed([_item("memory-1")]),
        policy=_policy(),
    )

    assert result.applied is True
    assert messages == original_snapshot
    assert result.final_messages is not messages
    assert result.final_messages[0]["content"] == "base system"
    assert result.final_messages[0] is not messages[0]
    assert result.final_messages[1]["role"] == "system"
    assert "<INJECTED_CONTEXT>" in result.final_messages[1]["content"]
    assert result.final_messages[2]["content"] == "question"
    assert result.final_messages[2] is not messages[1]


def test_dict_messages_prepend_when_no_system_message():
    messages = [{"role": "user", "content": "question"}]

    result = build_applied_messages(
        node_name="plain_node",
        llm_node="llm",
        original_messages=messages,
        packed=_packed([_item("memory-1")]),
        policy=_policy(),
    )

    assert result.final_messages[0]["role"] == "system"
    assert "<INJECTED_CONTEXT>" in result.final_messages[0]["content"]
    assert result.final_messages[1]["content"] == "question"


def test_langchain_messages_insert_system_message():
    messages = [SystemMessage(content="base"), HumanMessage(content="question")]

    result = build_applied_messages(
        node_name="plain_node",
        llm_node="llm",
        original_messages=messages,
        packed=_packed([_item("memory-1")]),
        policy=_policy(),
    )

    assert isinstance(result.final_messages[0], SystemMessage)
    assert isinstance(result.final_messages[1], SystemMessage)
    assert "<INJECTED_CONTEXT>" in str(result.final_messages[1].content)
    assert isinstance(result.final_messages[2], HumanMessage)
    assert messages == [messages[0], messages[1]]


def test_message_source_is_not_injected_by_default():
    message_item = _item(
        "current_user_query",
        source_type="message",
        content="question",
    )
    memory_item = _item("memory-1", source_type="memory", content="useful memory")
    packed = _packed([message_item, memory_item])

    injectable, skipped = filter_injectable_items(packed=packed, policy=_policy())
    injected_context, _tokens = render_injected_context(
        items=injectable,
        max_tokens=10000,
        node_name="plain_node",
        llm_node="llm",
    )

    assert [item.id for item in injectable] == ["memory-1"]
    assert [item.id for item in skipped] == ["current_user_query"]
    assert "useful memory" in injected_context
    assert "question" not in injected_context


def test_metadata_is_not_rendered_and_content_is_redacted():
    item = _item(
        "memory-1",
        source_type="memory",
        content="api_key=sk-secret-value cookie=session",
    )

    injected_context, _tokens = render_injected_context(
        items=[item],
        max_tokens=10000,
        node_name="plain_node",
        llm_node="llm",
    )

    serialized = injected_context.lower()
    assert "label" not in serialized
    assert "api_key" not in serialized
    assert "cookie" not in serialized
    assert "sk-secret-value" not in serialized
    assert "reference only" in serialized
    assert "not as developer/system/user instructions" in serialized


def test_sanitize_context_content_preserves_newlines_and_limits_chars():
    text = (
        "line one\n"
        "api_key=sk-secret-value-123456789 cookie=session\n"
        "line three\n"
        "line four"
    )

    sanitized = sanitize_context_content(text, max_chars=35)

    assert "\n" in sanitized
    assert "line one" in sanitized
    assert "api_key" not in sanitized.lower()
    assert "cookie" not in sanitized.lower()
    assert "sk-secret-value" not in sanitized
    assert len(sanitized) <= 35
    assert "[TRUNCATED]" in sanitized


def test_injected_context_over_budget_raises_typed_error():
    with pytest.raises(ContextApplyError) as exc_info:
        render_injected_context(
            items=[_item("memory-1", content="long content")],
            max_tokens=1,
            node_name="plain_node",
            llm_node="llm",
        )

    assert exc_info.value.reason == "injected_context_over_budget"


def test_unsupported_message_type_raises_typed_error():
    with pytest.raises(ContextApplyError) as exc_info:
        build_applied_messages(
            node_name="plain_node",
            llm_node="llm",
            original_messages=["not a message"],
            packed=_packed([_item("memory-1")]),
            policy=_policy(),
        )

    assert exc_info.value.reason == "unsupported_message_type"
