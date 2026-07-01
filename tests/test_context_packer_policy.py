"""Policy tests for Phase 3A ContextPacker."""

from __future__ import annotations

from src.context_engineering.packing.packer import pack_context_items
from src.context_engineering.schema import ContextItem


def _item(
    item_id: str,
    *,
    priority: int,
    token_estimate: int = 5,
    relevance_score: float | None = None,
    confidence: float | None = None,
    recency_score: float | None = None,
) -> ContextItem:
    return ContextItem(
        id=item_id,
        source_type="memory",
        title=item_id,
        content=item_id,
        token_estimate=token_estimate,
        estimated=True,
        tokenizer_mode="estimated_mixed",
        priority=priority,
        relevance_score=relevance_score,
        recency_score=recency_score,
        confidence=confidence,
        scope="session",
        lifetime="session",
        compressible=True,
        can_drop=True,
        disclosure_level="summary",
        metadata={},
    )


def _selected_ids(items: list[ContextItem]) -> list[str]:
    packed = pack_context_items(
        node_name="node",
        llm_node="llm",
        items=items,
        max_context_block_tokens=1000,
    )
    return [item.id for item in packed.selected_items]


def test_optional_items_sort_by_priority_desc():
    assert _selected_ids(
        [
            _item("low", priority=10),
            _item("high", priority=90),
        ]
    ) == ["high", "low"]


def test_optional_items_sort_by_scores_and_token_size_tie_breakers():
    items = [
        _item("id-b", priority=50, relevance_score=0.5, confidence=0.5),
        _item("id-a", priority=50, relevance_score=0.5, confidence=0.5),
        _item("relevant", priority=50, relevance_score=0.9),
        _item("confident", priority=50, relevance_score=0.5, confidence=0.9),
        _item(
            "recent",
            priority=50,
            relevance_score=0.5,
            confidence=0.5,
            recency_score=0.9,
        ),
        _item(
            "smaller",
            priority=50,
            relevance_score=0.5,
            confidence=0.5,
            token_estimate=1,
        ),
    ]

    assert _selected_ids(items) == [
        "relevant",
        "confident",
        "recent",
        "smaller",
        "id-a",
        "id-b",
    ]
