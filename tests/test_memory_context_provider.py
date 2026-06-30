"""Tests for memory context provider."""

from __future__ import annotations

import inspect

import pytest

from src.context_engineering.providers.base import ProviderContext
from src.context_engineering.providers import memory_provider
from src.context_engineering.providers.memory_provider import MemoryContextProvider
from src.context_engineering.schema import ContextProviderError


def _context(state: dict) -> ProviderContext:
    return ProviderContext(
        node_name="node",
        llm_node="llm",
        user_query="query",
        current_user_message_index=None,
        state=state,
        messages=[],
        request_id=None,
        thread_id=None,
        max_items_per_provider=10,
        max_content_chars_per_item=4000,
    )


def test_memory_provider_returns_empty_for_no_existing_memory():
    assert MemoryContextProvider().collect(_context({})) == []


def test_memory_provider_objectizes_existing_state_results_only():
    state = {
        "episodic_memory_results": [
            {
                "memory_type": "episodic",
                "score": 0.8,
                "match_reason": "keyword_overlap",
                "memory": {
                    "memory_id": "m1",
                    "memory_type": "quiz_attempt",
                    "content": "User missed binary search boundary cases.",
                    "created_at": "2026-06-30T00:00:00Z",
                },
            }
        ],
        "semantic_memory_results": [
            {
                "memory_type": "semantic",
                "score": 0.6,
                "memory": {
                    "summary_id": "s1",
                    "content": "Learner prefers worked examples.",
                    "confidence": 0.7,
                },
            }
        ],
    }

    items = MemoryContextProvider().collect(_context(state))

    assert [item.source_type for item in items] == ["memory", "memory"]
    assert items[0].metadata["memory_id"] == "m1"
    assert items[0].relevance_score == 0.8
    assert items[1].priority == 65


def test_memory_provider_supports_summary_and_text_content_fields():
    items = MemoryContextProvider().collect(
        _context(
            {
                "episodic_memory_results": [
                    {"memory_id": "m1", "summary": "Summary memory."},
                    {"memory_id": "m2", "text": "Text memory."},
                ]
            }
        )
    )

    assert [item.content for item in items] == ["Summary memory.", "Text memory."]


def test_memory_provider_applies_limit_before_item_construction(monkeypatch):
    calls = 0
    original = memory_provider._memory_result_to_item

    def counting_to_item(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    context = _context(
        {
            "episodic_memory_results": [
                {"memory_id": "m1", "content": "one"},
                {"memory_id": "m2", "content": "two"},
            ]
        }
    )
    context = ProviderContext(
        node_name=context.node_name,
        llm_node=context.llm_node,
        user_query=context.user_query,
        current_user_message_index=context.current_user_message_index,
        state=context.state,
        messages=context.messages,
        request_id=context.request_id,
        thread_id=context.thread_id,
        max_items_per_provider=1,
        max_content_chars_per_item=context.max_content_chars_per_item,
    )
    monkeypatch.setattr(memory_provider, "_memory_result_to_item", counting_to_item)

    items = MemoryContextProvider().collect(context)

    assert len(items) == 1
    assert calls == 1


def test_memory_provider_schema_decode_failure_is_not_empty_result():
    with pytest.raises(ContextProviderError, match="episodic_memory_results"):
        MemoryContextProvider().collect(_context({"episodic_memory_results": {}}))


def test_memory_provider_does_not_import_retrieval_or_embedding_paths():
    source = inspect.getsource(memory_provider)

    assert "retrieve_top_k_memories" not in source
    assert "get_embedding_provider" not in source
    assert "embed(" not in source
