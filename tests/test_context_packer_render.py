"""Render tests for Phase 3A ContextPacker."""

from __future__ import annotations

from src.context_engineering.packing.render import render_selected_context
from src.context_engineering.schema import ContextItem


def _item() -> ContextItem:
    return ContextItem(
        id="item-1",
        source_type="evidence",
        title="title with api_key=sk-secret-value",
        content="useful evidence before cookie=session",
        token_estimate=20,
        estimated=True,
        tokenizer_mode="estimated_mixed",
        priority=80,
        relevance_score=0.8,
        recency_score=None,
        confidence=None,
        scope="turn",
        lifetime="turn",
        compressible=True,
        can_drop=True,
        disclosure_level="snippet",
        metadata={"note": "must not render"},
    )


def test_render_selected_context_contains_source_title_and_content():
    rendered = render_selected_context([_item()])

    assert rendered.startswith("<CONTEXT_PACK>")
    assert "[evidence]" in rendered
    assert "useful evidence" in rendered
    assert rendered.endswith("</CONTEXT_PACK>")


def test_render_selected_context_redacts_secrets_and_excludes_metadata():
    rendered = render_selected_context([_item()]).lower()

    assert "metadata" not in rendered
    assert "must not render" not in rendered
    assert "api_key" not in rendered
    assert "cookie" not in rendered
    assert "sk-secret-value" not in rendered
