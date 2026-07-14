"""Quality and budget tests for Phase 3B-2A context apply."""

from __future__ import annotations

from src.context_engineering.packing.apply import (
    ApplyBudgetPolicy,
    ApplyFormatPolicy,
    ApplyQualityPolicy,
    ContextInjectionPolicy,
    ImportanceScoringPolicy,
    RouteRolloutPolicy,
    prepare_context_apply_selection,
    render_injected_context,
)
from src.context_engineering.packing.packer import pack_context_items
from src.context_engineering.schema import ContextItem


def _policy(
    *,
    max_tokens: int = 10000,
    min_priority: int = 0,
    min_relevance_score: float | None = None,
    max_items_total: int = 8,
) -> ContextInjectionPolicy:
    return ContextInjectionPolicy(
        enabled=True,
        apply_enabled_nodes=("review_doc_agent",),
        allow_structured_output=False,
        role="system",
        position="after_system",
        exclude_message_source=True,
        max_injected_context_tokens=max_tokens,
        injectable_sources=("memory", "evidence", "rules"),
        route_rollout=RouteRolloutPolicy(
            enabled=True,
            route_name="single_resource_generation",
            apply_enabled_nodes=("review_doc_agent",),
            require_single_resource_request=True,
            sample_rate=1.0,
            min_injectable_items=1,
        ),
        quality=ApplyQualityPolicy(
            min_priority=min_priority,
            min_relevance_score=min_relevance_score,
            max_items_total=max_items_total,
            max_items_per_source={},
        ),
        budget=ApplyBudgetPolicy(
            graceful_degradation_enabled=True,
            drop_order=("priority_asc", "token_estimate_desc", "id_asc"),
        ),
        format=ApplyFormatPolicy(
            group_by_source=True,
            include_untrusted_context_warning=True,
            include_section_headers=True,
            max_content_chars_per_item=4000,
        ),
        importance_scoring=ImportanceScoringPolicy(
            enabled=False,
            shadow_mode=True,
            mode="shadow",
            llm_node="",
            max_items_to_score=0,
            max_content_preview_chars=0,
            timeout_seconds=0.0,
            emit_shadow_telemetry=False,
            min_shadow_score_for_analysis=0.0,
        ),
    )


def _item(
    item_id: str,
    *,
    source_type: str = "memory",
    content: str = "useful memory",
    priority: int = 80,
    token_estimate: int = 5,
    relevance_score: float | None = None,
    confidence: float | None = None,
    recency_score: float | None = None,
) -> ContextItem:
    return ContextItem(
        id=item_id,
        source_type=source_type,
        title=item_id,
        content=content,
        token_estimate=token_estimate,
        estimated=True,
        tokenizer_mode="estimated_mixed",
        priority=priority,
        relevance_score=relevance_score,
        recency_score=recency_score,
        confidence=confidence,
        scope="turn",
        lifetime="turn",
        compressible=True,
        can_drop=True,
        disclosure_level="snippet",
        metadata={},
    )


def _packed(items: list[ContextItem]):
    return pack_context_items(
        node_name="review_doc_agent",
        llm_node="review_doc",
        items=items,
        max_context_block_tokens=10000,
    )


def test_selection_skips_when_only_message_items_are_selected():
    selection = prepare_context_apply_selection(
        packed=_packed([_item("msg", source_type="message", content="question")]),
        policy=_policy(),
        node_name="review_doc_agent",
        llm_node="review_doc",
    )

    assert selection.skip_reason == "no_injectable_items"
    assert selection.final_injected_count == 0


def test_selection_reports_quality_filtered_all():
    selection = prepare_context_apply_selection(
        packed=_packed([_item("low", priority=10)]),
        policy=_policy(min_priority=50),
        node_name="review_doc_agent",
        llm_node="review_doc",
    )

    assert selection.skip_reason == "quality_filtered_all"
    assert selection.quality_filtered_count == 1
    assert selection.final_injected_count == 0


def test_quality_caps_apply_after_deterministic_sort():
    low_priority = _item("a-low", priority=10, token_estimate=1)
    high_priority = _item("b-high", priority=90, token_estimate=100)
    high_relevance = _item(
        "c-relevance",
        priority=90,
        relevance_score=0.9,
        confidence=0.1,
        recency_score=0.1,
        token_estimate=5,
    )

    selection = prepare_context_apply_selection(
        packed=_packed([low_priority, high_priority, high_relevance]),
        policy=_policy(max_items_total=1),
        node_name="review_doc_agent",
        llm_node="review_doc",
    )

    assert selection.final_injected_count == 1
    assert selection.final_items[0].id == "c-relevance"
    assert selection.quality_filtered_count == 2


def test_min_relevance_score_none_does_not_filter_missing_relevance():
    item = _item("missing-relevance", relevance_score=None)

    selection = prepare_context_apply_selection(
        packed=_packed([item]),
        policy=_policy(min_relevance_score=None),
        node_name="review_doc_agent",
        llm_node="review_doc",
    )

    assert selection.skip_reason == ""
    assert selection.final_items[0].id == "missing-relevance"
    assert selection.quality_filtered_count == 0


def test_min_relevance_score_filters_missing_and_low_relevance():
    missing = _item("missing-relevance", relevance_score=None, priority=90)
    low = _item("low-relevance", relevance_score=0.49, priority=80)
    high = _item("high-relevance", relevance_score=0.5, priority=70)

    zero_threshold = prepare_context_apply_selection(
        packed=_packed([missing, high]),
        policy=_policy(min_relevance_score=0.0),
        node_name="review_doc_agent",
        llm_node="review_doc",
    )
    strict_threshold = prepare_context_apply_selection(
        packed=_packed([missing, low, high]),
        policy=_policy(min_relevance_score=0.5),
        node_name="review_doc_agent",
        llm_node="review_doc",
    )

    assert [item.id for item in zero_threshold.final_items] == ["high-relevance"]
    assert zero_threshold.quality_filtered_count == 1
    assert [item.id for item in strict_threshold.final_items] == ["high-relevance"]
    assert strict_threshold.quality_filtered_count == 2


def test_group_by_source_renders_one_section_per_source_in_config_order():
    policy = _policy()
    items = [
        _item("memory-1", source_type="memory", content="memory one"),
        _item(
            "evidence-1",
            source_type="evidence",
            content="evidence one api_key=sk-secret-value-123456",
        ),
        _item("memory-2", source_type="memory", content="memory two"),
        _item("evidence-2", source_type="evidence", content="evidence two"),
        _item("rules-1", source_type="rules", content="rules one"),
    ]
    items[0].metadata["secret"] = "must not render"

    rendered, _tokens = render_injected_context(
        items=items,
        max_tokens=10000,
        node_name="review_doc_agent",
        llm_node="review_doc",
        format_policy=policy.format,
    )

    assert rendered.count("## Source: memory") == 1
    assert rendered.count("## Source: evidence") == 1
    assert rendered.count("## Source: rules") == 1
    assert rendered.index("## Source: memory") < rendered.index("## Source: evidence")
    assert rendered.index("## Source: evidence") < rendered.index("## Source: rules")

    memory_section = rendered.split("## Source: memory", 1)[1].split(
        "## Source: evidence",
        1,
    )[0]
    evidence_section = rendered.split("## Source: evidence", 1)[1].split(
        "## Source: rules",
        1,
    )[0]
    assert "memory-1" in memory_section
    assert "memory-2" in memory_section
    assert "evidence-1" in evidence_section
    assert "evidence-2" in evidence_section
    assert "must not render" not in rendered
    assert "api_key" not in rendered.lower()
    assert "sk-secret-value" not in rendered


def test_budget_graceful_degradation_drops_whole_items_and_reports_stats():
    large = _item(
        "large",
        content="long content " * 200,
        priority=10,
        token_estimate=1000,
    )
    small = _item("small", content="short", priority=90, token_estimate=5)

    selection = prepare_context_apply_selection(
        packed=_packed([large, small]),
        policy=_policy(max_tokens=200),
        node_name="review_doc_agent",
        llm_node="review_doc",
    )

    assert selection.skip_reason == ""
    assert selection.budget_dropped_count == 1
    assert selection.final_injected_count == 1
    assert selection.source_counts_after == {"memory": 1}
    assert selection.drop_reasons == {"over_budget": 1}
    assert "short" in selection.rendered_context
    assert "long content" not in selection.rendered_context
