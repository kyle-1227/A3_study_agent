"""Tests for evidence context provider."""

from __future__ import annotations

import pytest

from src.context_engineering.providers.base import ProviderContext
from src.context_engineering.providers import evidence_provider
from src.context_engineering.providers.evidence_provider import EvidenceContextProvider
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


def test_evidence_provider_objectizes_existing_candidates_without_reranking():
    state = {
        "evidence_candidates": [
            {
                "evidence_id": "local:1",
                "source_type": "local_rag",
                "provider": "local",
                "title": "Binary search notes",
                "url": "",
                "content_preview": "Binary search halves the search interval.",
                "rerank_score": 0.8,
            },
            {
                "evidence_id": "web:1",
                "source_type": "web",
                "provider": "tavily",
                "title": "Docs",
                "url": "https://example.test/docs",
                "content_preview": "Reference snippet.",
                "tavily_score": 0.5,
            },
        ]
    }

    items = EvidenceContextProvider().collect(_context(state))

    assert [item.metadata["evidence_id"] for item in items] == ["local:1", "web:1"]
    assert items[0].priority == 85
    assert items[1].priority == 75
    assert items[0].content == "Binary search halves the search interval."
    assert "raw_html" not in items[0].metadata


def test_evidence_provider_dedupes_existing_bucket_overlap():
    state = {
        "evidence_candidates": [{"evidence_id": "same", "content_preview": "first"}],
        "local_evidence_candidates": [
            {"evidence_id": "same", "content_preview": "second"}
        ],
    }

    items = EvidenceContextProvider().collect(_context(state))

    assert len(items) == 1
    assert items[0].content == "first"


def test_evidence_provider_supports_compatible_content_fields():
    fields = (
        "snippet",
        "excerpt",
        "summary",
        "content",
        "text",
        "page_content",
        "coverage_contribution",
    )
    state = {
        "evidence_candidates": [
            {"evidence_id": key, key: f"{key} value"} for key in fields
        ]
    }

    items = EvidenceContextProvider().collect(_context(state))

    assert [item.content for item in items] == [f"{key} value" for key in fields]


def test_evidence_provider_applies_limit_before_item_construction(monkeypatch):
    calls = 0
    original = evidence_provider._candidate_to_item

    def counting_to_item(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    context = _context(
        {
            "evidence_candidates": [
                {"evidence_id": "e1", "content_preview": "one"},
                {"evidence_id": "e2", "content_preview": "two"},
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
    monkeypatch.setattr(evidence_provider, "_candidate_to_item", counting_to_item)

    items = EvidenceContextProvider().collect(context)

    assert len(items) == 1
    assert calls == 1


def test_evidence_provider_decode_failure_is_not_empty_result():
    with pytest.raises(ContextProviderError, match="evidence_candidates"):
        EvidenceContextProvider().collect(_context({"evidence_candidates": "bad"}))
