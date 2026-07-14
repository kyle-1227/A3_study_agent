"""Tests for memory context provider."""

from __future__ import annotations

import inspect

import pytest

from src.context_engineering.providers.base import ProviderContext
from src.context_engineering.providers import memory_provider
from src.context_engineering.providers.memory_provider import MemoryContextProvider
from src.context_engineering.schema import ContextProviderError


def _context(
    state: dict,
    *,
    thread_id: str | None = None,
    max_items: int = 10,
) -> ProviderContext:
    return ProviderContext(
        node_name="node",
        llm_node="llm",
        user_query="query",
        current_user_message_index=None,
        state=state,
        messages=[],
        request_id=None,
        thread_id=thread_id,
        max_items_per_provider=max_items,
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

    state["user_id"] = "user-1"
    items = MemoryContextProvider().collect(_context(state, thread_id="thread-1"))

    assert [item.source_type for item in items] == ["memory", "memory"]
    assert items[0].metadata["memory_id"] == "m1"
    assert items[0].metadata["thread_id"] == "thread-1"
    assert items[0].metadata["user_id"] == "user-1"
    assert items[0].metadata["purpose"] == ["continuity", "personalization"]
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


def test_memory_provider_unknown_bucket_fails_instead_of_using_silent_cap(
    monkeypatch,
):
    monkeypatch.setattr(
        memory_provider,
        "_existing_memory_results",
        lambda _state, *, limit: [("unknown_memory_bucket", {"content": "memory"})],
    )

    with pytest.raises(ContextProviderError) as exc_info:
        MemoryContextProvider().collect(_context({}))

    assert exc_info.value.original_exception_type == "KeyError"


def test_memory_provider_does_not_import_retrieval_or_embedding_paths():
    source = inspect.getsource(memory_provider)

    assert "retrieve_top_k_memories" not in source
    assert "get_embedding_provider" not in source
    assert "embed(" not in source


@pytest.mark.parametrize("policy", ["ignore", "ask_user"])
def test_memory_provider_blocks_ignore_and_pending_confirmation(policy):
    items = MemoryContextProvider().collect(
        _context(
            {
                "memory_use_policy": policy,
                "conversation_summary": "Do not inject this summary.",
                "episodic_memory_results": [
                    {"memory_id": "m1", "content": "Do not inject this memory."}
                ],
            },
            thread_id="thread-1",
        )
    )

    assert items == []


def test_memory_provider_preserves_source_identity_mismatch_for_filtering():
    items = MemoryContextProvider().collect(
        _context(
            {
                "user_id": "user-current",
                "episodic_memory_results": [
                    {
                        "memory_id": "m1",
                        "content": "Wrong-thread memory.",
                        "thread_id": "thread-other",
                        "user_id": "user-other",
                    }
                ],
            },
            thread_id="thread-current",
        )
    )

    assert items[0].metadata["thread_id"] == "thread-other"
    assert items[0].metadata["user_id"] == "user-other"


def test_memory_provider_uses_stable_logical_id_across_content_versions():
    first = MemoryContextProvider().collect(
        _context(
            {
                "episodic_memory_results": [
                    {"memory_id": "m1", "content": "Version one.", "score": 0.6}
                ]
            },
            thread_id="thread-1",
        )
    )[0]
    second = MemoryContextProvider().collect(
        _context(
            {
                "episodic_memory_results": [
                    {"memory_id": "m1", "content": "Version two.", "score": 0.9}
                ]
            },
            thread_id="thread-1",
        )
    )[0]

    assert first.id == second.id
    assert first.content != second.content


def test_memory_provider_fairly_represents_summary_episodic_and_semantic_buckets():
    items = MemoryContextProvider().collect(
        _context(
            {
                "conversation_summary": "Conversation summary.",
                "episodic_memory_results": [
                    {"memory_id": f"e{index}", "content": f"episodic {index}"}
                    for index in range(8)
                ],
                "semantic_memory_results": [
                    {"summary_id": f"s{index}", "content": f"semantic {index}"}
                    for index in range(4)
                ],
            },
            thread_id="thread-1",
            max_items=6,
        )
    )

    source_buckets = [item.metadata["source_bucket"] for item in items]
    assert source_buckets.count("conversation_summary") == 1
    assert source_buckets.count("episodic_memory_results") == 3
    assert source_buckets.count("semantic_memory_results") == 2
