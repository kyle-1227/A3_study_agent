"""Schema tests for Phase 3A ContextPacker."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.context_engineering.packing.schema import (
    ContextPackingError,
    PackedContext,
    PackingDecision,
)
from src.context_engineering.schema import ContextItem


def _item(
    item_id: str = "item-1",
    *,
    can_drop: bool = True,
    token_estimate: int = 5,
) -> ContextItem:
    return ContextItem(
        id=item_id,
        source_type="message",
        title="safe title",
        content="secret content",
        token_estimate=token_estimate,
        estimated=True,
        tokenizer_mode="estimated_mixed",
        priority=50,
        relevance_score=None,
        recency_score=None,
        confidence=None,
        scope="turn",
        lifetime="turn",
        compressible=True,
        can_drop=can_drop,
        disclosure_level="snippet",
        metadata={},
    )


def test_packing_decision_forbids_content_and_metadata_fields():
    payload = {
        "item_id": "item-1",
        "source_type": "message",
        "title": "title",
        "selected": True,
        "reason": "fits_budget",
        "token_estimate": 5,
        "priority": 50,
        "can_drop": True,
        "budget_before": 10,
        "budget_after": 5,
        "content": "must not be accepted",
        "metadata": {"secret": "must not be accepted"},
    }

    with pytest.raises(ValidationError):
        PackingDecision.model_validate(payload)


def test_packed_context_forbids_extra_fields_and_validates_token_totals():
    item = _item()
    decision = PackingDecision(
        item_id=item.id,
        source_type=item.source_type,
        title=item.title,
        selected=True,
        reason="fits_budget",
        token_estimate=item.token_estimate,
        priority=item.priority,
        can_drop=item.can_drop,
        budget_before=20,
        budget_after=15,
    )

    packed = PackedContext(
        node_name="node",
        llm_node="llm",
        strategy="priority_budget",
        selected_items=[item],
        dropped_items=[],
        decisions=[decision],
        rendered_context="<CONTEXT_PACK>\n</CONTEXT_PACK>",
        max_context_block_tokens=20,
        selected_tokens=5,
        dropped_tokens=0,
        required_tokens=0,
        optional_tokens=5,
        remaining_tokens=15,
        overflow=False,
        warnings=[],
    )

    data = packed.model_dump()
    data["content"] = "must not be accepted"
    with pytest.raises(ValidationError):
        PackedContext.model_validate(data)


def test_packed_context_rejects_negative_tokens():
    item = _item()
    with pytest.raises(ValidationError):
        PackedContext(
            node_name="node",
            llm_node="llm",
            strategy="priority_budget",
            selected_items=[item],
            dropped_items=[],
            decisions=[],
            rendered_context="",
            max_context_block_tokens=20,
            selected_tokens=-1,
            dropped_tokens=0,
            required_tokens=0,
            optional_tokens=5,
            remaining_tokens=15,
            overflow=False,
            warnings=[],
        )


def test_context_packing_error_redacts_sensitive_values():
    error = ContextPackingError(
        reason="context_packing_error",
        warning="failed with api_key=sk-secret-value and cookie=session",
        node_name="node",
        llm_node="llm",
        selected_tokens=1,
        budget_tokens=2,
        original_exception_type="RuntimeError",
    )

    serialized = repr(error.warning).lower()
    assert "api_key" not in serialized
    assert "cookie" not in serialized
    assert "sk-secret-value" not in serialized
