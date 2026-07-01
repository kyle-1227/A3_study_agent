"""Budget tests for Phase 3A ContextPacker."""

from __future__ import annotations

import pytest

from src.context_engineering.packing.packer import pack_context_items
from src.context_engineering.packing.schema import ContextPackingError
from src.context_engineering.schema import ContextItem


def _item(
    item_id: str,
    *,
    source_type: str = "memory",
    token_estimate: int,
    can_drop: bool,
    priority: int = 50,
) -> ContextItem:
    return ContextItem(
        id=item_id,
        source_type=source_type,
        title=item_id,
        content=item_id,
        token_estimate=token_estimate,
        estimated=True,
        tokenizer_mode="estimated_mixed",
        priority=priority,
        relevance_score=None,
        recency_score=None,
        confidence=None,
        scope="turn",
        lifetime="turn",
        compressible=True,
        can_drop=can_drop,
        disclosure_level="summary",
        metadata={},
    )


def test_required_items_are_selected_before_optional_items():
    required = _item("required", token_estimate=5, can_drop=False, priority=1)
    optional = _item("optional", token_estimate=5, can_drop=True, priority=100)

    packed = pack_context_items(
        node_name="node",
        llm_node="llm",
        items=[optional, required],
        max_context_block_tokens=100,
    )

    assert [item.id for item in packed.selected_items] == ["required", "optional"]
    assert packed.decisions[0].reason == "required"


def test_required_items_over_budget_raise_typed_error():
    with pytest.raises(ContextPackingError) as exc_info:
        pack_context_items(
            node_name="node",
            llm_node="llm",
            items=[_item("required", token_estimate=50, can_drop=False)],
            max_context_block_tokens=10,
        )

    assert exc_info.value.reason == "required_over_budget"


def test_optional_items_over_budget_are_dropped_with_token_totals():
    packed = pack_context_items(
        node_name="node",
        llm_node="llm",
        items=[
            _item("required", token_estimate=1, can_drop=False),
            _item("fits", token_estimate=1, can_drop=True, priority=100),
            _item("drops", token_estimate=100, can_drop=True, priority=90),
        ],
        max_context_block_tokens=50,
    )

    assert [item.id for item in packed.selected_items] == ["required", "fits"]
    assert [item.id for item in packed.dropped_items] == ["drops"]
    assert packed.selected_tokens == 2
    assert packed.dropped_tokens == 100
    assert packed.remaining_tokens == 48
    assert packed.overflow is False
    assert any(decision.reason == "over_budget" for decision in packed.decisions)


def test_enabled_sources_only_affects_packing_decisions():
    packed = pack_context_items(
        node_name="node",
        llm_node="llm",
        items=[
            _item("message", source_type="message", token_estimate=5, can_drop=False),
            _item("memory", source_type="memory", token_estimate=5, can_drop=True),
        ],
        max_context_block_tokens=100,
        enabled_sources=("message",),
    )

    assert [item.id for item in packed.selected_items] == ["message"]
    assert [item.id for item in packed.dropped_items] == ["memory"]
    assert any(decision.reason == "source_disabled" for decision in packed.decisions)


def test_required_item_with_disabled_source_raises_typed_error():
    with pytest.raises(ContextPackingError) as exc_info:
        pack_context_items(
            node_name="node",
            llm_node="llm",
            items=[
                _item(
                    "required-memory",
                    source_type="memory",
                    token_estimate=5,
                    can_drop=False,
                )
            ],
            max_context_block_tokens=100,
            enabled_sources=("message",),
        )

    assert exc_info.value.reason == "required_source_disabled"


def test_rendered_context_over_budget_uses_actual_rendered_token_estimate():
    with pytest.raises(ContextPackingError) as exc_info:
        pack_context_items(
            node_name="node",
            llm_node="llm",
            items=[_item("tiny", token_estimate=1, can_drop=False)],
            max_context_block_tokens=2,
        )

    assert exc_info.value.reason == "rendered_context_over_budget"
