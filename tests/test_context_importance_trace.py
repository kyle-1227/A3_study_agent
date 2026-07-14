"""Safe telemetry tests for Phase 3B-2A importance scoring."""

from __future__ import annotations

from src.context_engineering.packing.apply import ImportanceScoringPolicy
from src.context_engineering.packing.apply_trace import (
    build_context_importance_scored_event,
)
from src.context_engineering.packing.importance import (
    aggregate_importance_failure,
    build_importance_scorer_messages,
)
from src.context_engineering.schema import ContextItem


def _policy() -> ImportanceScoringPolicy:
    return ImportanceScoringPolicy(
        enabled=True,
        shadow_mode=True,
        mode="shadow",
        llm_node="importance_scorer",
        max_items_to_score=3,
        max_content_preview_chars=300,
        timeout_seconds=1.0,
        emit_shadow_telemetry=True,
        min_shadow_score_for_analysis=0.5,
    )


def test_importance_telemetry_and_prompt_preview_redact_item_secrets():
    item = ContextItem(
        id="memory-1",
        source_type="memory",
        title="title api_key=sk-title-secret",
        content=(
            "api_key=sk-secret-value cookie=session "
            "db_uri=postgresql://user:pass@localhost/db"
        ),
        token_estimate=20,
        estimated=True,
        tokenizer_mode="estimated_mixed",
        priority=80,
        relevance_score=None,
        recency_score=None,
        confidence=None,
        scope="turn",
        lifetime="turn",
        compressible=True,
        can_drop=True,
        disclosure_level="snippet",
        metadata={"token_estimate": 20},
    )
    policy = _policy()

    scorer_messages = build_importance_scorer_messages(items=[item], policy=policy)
    telemetry = aggregate_importance_failure(
        items=[item],
        started_at=None,
        reason="context_importance_llm_failed",
        warning="raw response had api_key=sk-secret-value",
    )
    event = build_context_importance_scored_event(
        node_name="review_doc_agent",
        llm_node="review_doc",
        telemetry=telemetry,
    )

    serialized_prompt = repr(scorer_messages).lower()
    serialized_event = repr(event).lower()
    assert "item_id" in serialized_prompt
    assert "source_type" in serialized_prompt
    assert "sanitized_title" in serialized_prompt
    assert "token_estimate" in serialized_prompt
    assert "priority" in serialized_prompt
    assert "relevance_score" in serialized_prompt
    assert "confidence" in serialized_prompt
    assert "recency_score" in serialized_prompt
    assert "disclosure_level" in serialized_prompt
    assert "content_preview" in serialized_prompt
    for serialized in (serialized_prompt, serialized_event):
        assert "api_key" not in serialized
        assert "cookie" not in serialized
        assert "db_uri" not in serialized
        assert "postgresql://" not in serialized
        assert "sk-secret-value" not in serialized
    assert "title api_key" not in serialized_event
    assert "content_preview" not in serialized_event
    assert "metadata" not in serialized_event
