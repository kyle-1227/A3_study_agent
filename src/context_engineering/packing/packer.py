"""Deterministic ContextPacker implementation for shadow mode."""

from __future__ import annotations

from src.context_engineering.packing.render import render_selected_context
from src.context_engineering.packing.schema import (
    ContextPackingError,
    PackedContext,
    PackingDecision,
    PackingReason,
    PackingStrategy,
)
from src.context_engineering.schema import ContextItem, ContextSourceType
from src.context_engineering.tokenizer import estimate_text_tokens_mixed


def pack_context_items(
    *,
    node_name: str,
    llm_node: str,
    items: list[ContextItem],
    max_context_block_tokens: int,
    strategy: PackingStrategy = "priority_budget",
    enabled_sources: tuple[ContextSourceType, ...] | None = None,
) -> PackedContext:
    """Pack candidate ContextItems into an internal shadow PackedContext."""
    _validate_inputs(
        node_name=node_name,
        llm_node=llm_node,
        items=items,
        max_context_block_tokens=max_context_block_tokens,
        strategy=strategy,
    )
    enabled_source_set = set(enabled_sources) if enabled_sources is not None else None
    selected_items: list[ContextItem] = []
    dropped_items: list[ContextItem] = []
    decisions: list[PackingDecision] = []
    eligible_items: list[ContextItem] = []

    for item in items:
        if (
            enabled_source_set is not None
            and item.source_type not in enabled_source_set
        ):
            if not item.can_drop:
                raise ContextPackingError(
                    reason="required_source_disabled",
                    warning="required context item source is disabled by packing policy",
                    node_name=node_name,
                    llm_node=llm_node,
                    selected_tokens=item.token_estimate,
                    budget_tokens=max_context_block_tokens,
                )
            dropped_items.append(item)
            decisions.append(
                _decision(
                    item,
                    selected=False,
                    reason="source_disabled",
                    budget_before=None,
                    budget_after=None,
                )
            )
        else:
            eligible_items.append(item)

    required_items = [item for item in eligible_items if not item.can_drop]
    optional_items = [item for item in eligible_items if item.can_drop]
    required_tokens = sum(item.token_estimate for item in required_items)
    optional_tokens = sum(item.token_estimate for item in optional_items)
    if required_tokens > max_context_block_tokens:
        raise ContextPackingError(
            reason="required_over_budget",
            warning="required context items exceed packing budget",
            node_name=node_name,
            llm_node=llm_node,
            selected_tokens=required_tokens,
            budget_tokens=max_context_block_tokens,
        )

    remaining = max_context_block_tokens
    for item in required_items:
        budget_before = remaining
        remaining -= item.token_estimate
        selected_items.append(item)
        decisions.append(
            _decision(
                item,
                selected=True,
                reason="required",
                budget_before=budget_before,
                budget_after=remaining,
            )
        )

    for item in sorted(optional_items, key=_optional_sort_key):
        budget_before = remaining
        if item.token_estimate <= remaining:
            remaining -= item.token_estimate
            selected_items.append(item)
            decisions.append(
                _decision(
                    item,
                    selected=True,
                    reason="fits_budget",
                    budget_before=budget_before,
                    budget_after=remaining,
                )
            )
        else:
            dropped_items.append(item)
            decisions.append(
                _decision(
                    item,
                    selected=False,
                    reason="over_budget",
                    budget_before=budget_before,
                    budget_after=remaining,
                )
            )

    rendered_context = render_selected_context(selected_items)
    rendered_tokens = estimate_text_tokens_mixed(rendered_context)
    selected_tokens = sum(item.token_estimate for item in selected_items)
    if rendered_tokens > max_context_block_tokens:
        raise ContextPackingError(
            reason="rendered_context_over_budget",
            warning="rendered context exceeds packing budget after formatting",
            node_name=node_name,
            llm_node=llm_node,
            selected_tokens=rendered_tokens,
            budget_tokens=max_context_block_tokens,
        )

    warnings: list[str] = []
    if rendered_tokens > selected_tokens:
        warnings.append("formatting_token_overhead")

    return PackedContext(
        node_name=node_name,
        llm_node=llm_node,
        strategy=strategy,
        selected_items=selected_items,
        dropped_items=dropped_items,
        decisions=decisions,
        rendered_context=rendered_context,
        max_context_block_tokens=max_context_block_tokens,
        selected_tokens=selected_tokens,
        dropped_tokens=sum(item.token_estimate for item in dropped_items),
        required_tokens=required_tokens,
        optional_tokens=optional_tokens,
        remaining_tokens=max(max_context_block_tokens - selected_tokens, 0),
        overflow=False,
        warnings=warnings,
    )


def _validate_inputs(
    *,
    node_name: str,
    llm_node: str,
    items: list[ContextItem],
    max_context_block_tokens: int,
    strategy: str,
) -> None:
    if not isinstance(items, list):
        raise ContextPackingError(
            reason="invalid_items",
            warning="items must be a list",
            node_name=node_name,
            llm_node=llm_node,
        )
    if strategy != "priority_budget":
        raise ContextPackingError(
            reason="unsupported_strategy",
            warning="only priority_budget strategy is supported",
            node_name=node_name,
            llm_node=llm_node,
        )
    if (
        isinstance(max_context_block_tokens, bool)
        or not isinstance(max_context_block_tokens, int)
        or max_context_block_tokens <= 0
    ):
        raise ContextPackingError(
            reason="invalid_budget",
            warning="max_context_block_tokens must be a positive integer",
            node_name=node_name,
            llm_node=llm_node,
        )
    for item in items:
        if not isinstance(item, ContextItem):
            raise ContextPackingError(
                reason="invalid_item",
                warning="items must contain ContextItem instances",
                node_name=node_name,
                llm_node=llm_node,
                original_exception_type="TypeError",
            )
        if item.token_estimate < 0:
            raise ContextPackingError(
                reason="invalid_item_token_estimate",
                warning="ContextItem token_estimate must be non-negative",
                node_name=node_name,
                llm_node=llm_node,
            )


def _optional_sort_key(item: ContextItem) -> tuple[int, float, float, float, int, str]:
    return (
        -item.priority,
        -(item.relevance_score or 0.0),
        -(item.confidence or 0.0),
        -(item.recency_score or 0.0),
        item.token_estimate,
        item.id,
    )


def _decision(
    item: ContextItem,
    *,
    selected: bool,
    reason: PackingReason,
    budget_before: int | None,
    budget_after: int | None,
) -> PackingDecision:
    return PackingDecision(
        item_id=item.id,
        source_type=item.source_type,
        title=item.title,
        selected=selected,
        reason=reason,
        token_estimate=item.token_estimate,
        priority=item.priority,
        can_drop=item.can_drop,
        budget_before=budget_before,
        budget_after=budget_after,
    )
