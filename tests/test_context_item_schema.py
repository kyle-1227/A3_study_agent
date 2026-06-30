"""Schema tests for Phase 2 ContextItem contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.context_engineering.schema import ContextItem


def _valid_item(**overrides) -> ContextItem:
    payload = {
        "id": "message:1",
        "source_type": "message",
        "title": "current_user_query",
        "content": "Explain binary search",
        "token_estimate": 5,
        "estimated": True,
        "tokenizer_mode": "estimated_mixed",
        "priority": 100,
        "relevance_score": 0.9,
        "recency_score": 0.8,
        "confidence": 0.7,
        "scope": "turn",
        "lifetime": "turn",
        "compressible": False,
        "can_drop": False,
        "disclosure_level": "full",
        "metadata": {"token_estimate": 5},
    }
    payload.update(overrides)
    return ContextItem(**payload)


def test_context_item_accepts_valid_strict_payload():
    item = _valid_item()

    assert item.source_type == "message"
    assert item.metadata["token_estimate"] == 5


def test_context_item_forbids_extra_fields():
    with pytest.raises(ValidationError):
        _valid_item(prompt="secret")


def test_context_item_rejects_invalid_token_and_priority_bounds():
    with pytest.raises(ValidationError):
        _valid_item(token_estimate=-1)

    with pytest.raises(ValidationError):
        _valid_item(priority=101)


def test_context_item_metadata_sensitive_key_matching_is_exact():
    allowed = _valid_item(metadata={"token_estimate": 12})
    assert allowed.metadata == {"token_estimate": 12}

    with pytest.raises(ValidationError, match="sensitive keys"):
        _valid_item(metadata={"token": "secret"})


def test_context_item_empty_content_requires_identifier_context():
    item = _valid_item(
        content="", title="profile_index", metadata={"source": "profile"}
    )
    assert item.content == ""

    with pytest.raises(ValidationError):
        _valid_item(content="", title="", metadata={})
