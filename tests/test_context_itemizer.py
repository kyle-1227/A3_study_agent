"""Tests for ContextItem construction helpers."""

from __future__ import annotations

from src.context_engineering.itemizer import (
    make_context_item,
    sanitize_metadata,
    stable_item_id,
)


def test_sanitize_metadata_uses_exact_normalized_sensitive_keys():
    metadata = sanitize_metadata(
        {
            "token": "secret",
            "Authorization": "Bearer secret",
            "token_estimate": 42,
            "safe": "value",
        }
    )

    assert "token" not in metadata
    assert "Authorization" not in metadata
    assert metadata["token_estimate"] == 42
    assert metadata["safe"] == "value"


def test_make_context_item_truncates_content_without_storing_original():
    original = "abcdef"
    item = make_context_item(
        source_type="evidence",
        title="doc",
        content=original,
        priority=75,
        scope="turn",
        lifetime="turn",
        compressible=True,
        can_drop=True,
        disclosure_level="snippet",
        metadata={},
        max_content_chars=3,
    )

    assert item.content == "abc"
    assert item.metadata["content_truncated"] is True
    assert item.metadata["original_content_chars"] == len(original)
    assert original not in repr(item.metadata)


def test_stable_item_id_uses_safe_identity_fields():
    first = stable_item_id(
        source_type="memory",
        title="remembered preference",
        metadata={"memory_id": "m1", "token": "secret"},
    )
    second = stable_item_id(
        source_type="memory",
        title="remembered preference",
        metadata={"memory_id": "m1"},
    )

    assert first == second
    assert first.startswith("memory:")
