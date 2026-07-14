"""Tests for profile and rules context providers."""

from __future__ import annotations

import pytest

from src.context_engineering.providers.base import ProviderContext
from src.context_engineering.providers.profile_provider import ProfileContextProvider
from src.context_engineering.providers.rules_provider import RulesContextProvider
from src.context_engineering.schema import ContextProviderError


def _context(state: dict, *, thread_id: str | None = None) -> ProviderContext:
    return ProviderContext(
        node_name="node",
        llm_node="llm",
        user_query="query",
        current_user_message_index=None,
        state=state,
        messages=[],
        request_id=None,
        thread_id=thread_id,
        max_items_per_provider=10,
        max_content_chars_per_item=4000,
    )


def test_profile_provider_objectizes_existing_profile_summary():
    items = ProfileContextProvider().collect(
        _context(
            {
                "user_id": "user-1",
                "profile_summary": "Learner likes examples.",
            },
            thread_id="thread-1",
        )
    )

    assert len(items) == 1
    assert items[0].source_type == "profile"
    assert items[0].lifetime == "long_term"
    assert items[0].content == "Learner likes examples."
    assert items[0].metadata["user_id"] == "user-1"
    assert items[0].metadata["thread_id"] == "thread-1"
    assert items[0].metadata["purpose"] == "personalization"


def test_profile_provider_does_not_load_profile_storage_when_absent():
    assert ProfileContextProvider().collect(_context({})) == []


def test_profile_provider_bad_state_fails_fast():
    with pytest.raises(ContextProviderError, match="profile must be a dict"):
        ProfileContextProvider().collect(_context({"profile": "bad"}))


def test_rules_provider_objectizes_existing_rule_summaries(monkeypatch):
    import src.context_engineering.providers.rules_provider as rules_provider

    monkeypatch.setattr(
        rules_provider, "get_setting", lambda _key, default=None: default
    )
    items = RulesContextProvider().collect(
        _context(
            {
                "node_rules": [
                    "Do not invent citations.",
                    {"title": "schema safety", "summary": "Do not trace schemas."},
                ]
            }
        )
    )

    assert [item.source_type for item in items] == ["rules", "rules"]
    assert all(item.can_drop is False for item in items)
    assert all(item.metadata["purpose"] == "instruction_support" for item in items)
    assert items[1].title == "schema safety"


def test_rules_provider_bad_state_fails_fast():
    with pytest.raises(ContextProviderError, match="runtime_rules"):
        RulesContextProvider().collect(_context({"runtime_rules": {"bad": True}}))


def test_profile_provider_uses_stable_logical_id_across_summary_versions():
    first = ProfileContextProvider().collect(
        _context(
            {"user_id": "user-1", "profile_summary": "Version one."},
            thread_id="thread-1",
        )
    )[0]
    second = ProfileContextProvider().collect(
        _context(
            {"user_id": "user-1", "profile_summary": "Version two."},
            thread_id="thread-1",
        )
    )[0]

    assert first.id == second.id
    assert first.content != second.content
