"""Trace and SSE tests for Phase 3B-1 context apply."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.context_engineering.packing.apply import (
    ContextApplyError,
    ContextApplyResult,
    ContextApplySelection,
    ContextInjectionPolicy,
)
from src.context_engineering.packing.apply_trace import (
    build_context_applied_event,
    build_context_apply_error_event,
    build_context_apply_plan_event,
    build_context_apply_selection_event,
    build_context_importance_scored_event,
)
from src.context_engineering.packing.importance import ContextImportanceTelemetry
from src.observability.a3_trace import emit_a3_trace


def _policy() -> ContextInjectionPolicy:
    return ContextInjectionPolicy(
        enabled=True,
        apply_enabled_nodes=("plain_node",),
        fallback_on_error=True,
        allow_structured_output=False,
        role="system",
        position="after_system",
        exclude_message_source=True,
        max_injected_context_tokens=10000,
        injectable_sources=("memory",),
    )


def _snapshot(values: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(next=(), tasks=[], values=values or {})


def _payloads(collected: list[str]) -> list[dict]:
    return [json.loads(item.removeprefix("data: ").strip()) for item in collected]


def test_context_apply_plan_event_is_safe():
    event = build_context_apply_plan_event(
        node_name="node",
        llm_node="llm",
        policy=_policy(),
        original_message_count=2,
        selected_item_count=3,
        injectable_item_count=1,
        skipped_item_count=2,
    )

    serialized = repr(event).lower()
    assert event["apply_enabled"] is True
    assert "final_messages" not in serialized
    assert "content" not in serialized
    assert "metadata" not in serialized


def test_context_applied_event_does_not_dump_result_messages():
    result = ContextApplyResult(
        applied=True,
        fallback_used=False,
        original_message_count=1,
        final_message_count=2,
        injected_items_count=1,
        skipped_items_count=0,
        injected_context_tokens=10,
        final_messages=[
            {"role": "system", "content": "<INJECTED_CONTEXT>secret</INJECTED_CONTEXT>"}
        ],
        warnings=["api_key=sk-secret-value cookie=session"],
    )

    event = build_context_applied_event(
        node_name="node",
        llm_node="llm",
        policy=_policy(),
        result=result,
    )

    serialized = repr(event).lower()
    assert event["final_message_count"] == 2
    assert event["injected_context_tokens"] == 10
    assert "final_messages" not in serialized
    assert "<injected_context>" not in serialized
    assert "secret" not in serialized
    assert "api_key" not in serialized
    assert "cookie" not in serialized
    assert "sk-secret-value" not in serialized


def test_context_apply_error_event_is_redacted():
    event = build_context_apply_error_event(
        ContextApplyError(
            reason="apply_failed",
            warning="api_key=sk-secret-value cookie=session",
            node_name="node",
            llm_node="llm",
            fallback_used=True,
            original_exception_type="RuntimeError",
        )
    )

    serialized = repr(event).lower()
    assert event["fallback_used"] is True
    assert "api_key" not in serialized
    assert "cookie" not in serialized
    assert "sk-secret-value" not in serialized


def test_context_apply_selection_event_is_safe_and_aggregate_only():
    event = build_context_apply_selection_event(
        node_name="node",
        llm_node="llm",
        selection=ContextApplySelection(
            skip_reason="budget_fit_failed",
            single_resource_result="matched_single_resource",
            selected_item_count=3,
            injectable_item_count=2,
            skipped_item_count=1,
            quality_filtered_count=1,
            budget_dropped_count=1,
            final_injected_count=1,
            injected_context_tokens=20,
            original_estimated_tokens=100,
            final_estimated_tokens=125,
            token_delta=25,
            source_counts_before={"memory": 2},
            source_counts_after={"memory": 1},
            drop_reasons={"over_budget": 1},
            warnings=["api_key=sk-secret cookie=session"],
        ),
    )

    serialized = repr(event).lower()
    assert event["budget_dropped_count"] == 1
    assert event["final_injected_count"] == 1
    assert event["original_estimated_tokens"] == 100
    assert event["final_estimated_tokens"] == 125
    assert event["token_delta"] == 25
    assert "api_key" not in serialized
    assert "cookie" not in serialized
    assert "sk-secret" not in serialized
    assert "content" not in serialized
    assert "metadata" not in serialized


def test_context_importance_scored_event_is_aggregate_only():
    event = build_context_importance_scored_event(
        node_name="node",
        llm_node="llm",
        telemetry=ContextImportanceTelemetry(
            source_counts={"memory": 2},
            score_buckets={"0.75-1.00": 2},
            reason_code_counts={"useful": 2},
            candidate_count=2,
            scored_count=2,
            kept_count=1,
            dropped_count=1,
            fallback_to_rule_based=False,
            scoring_elapsed_ms=12.5,
            warnings=["db_uri=postgresql://u:p@h/db"],
        ),
    )

    serialized = repr(event).lower()
    assert event["candidate_count"] == 2
    assert "title" not in serialized
    assert "content_preview" not in serialized
    assert "metadata" not in serialized
    assert "db_uri" not in serialized
    assert "postgresql://" not in serialized


@pytest.mark.anyio
async def test_context_apply_events_are_forwarded_as_safe_sse():
    from app import generate_sse

    async def events():
        emit_a3_trace(
            logging.getLogger("test_context_apply_trace"),
            "context_applied",
            {
                "node_name": "generate_answer",
                "llm_node": "academic",
                "applied": True,
                "fallback_used": False,
                "original_message_count": 1,
                "final_message_count": 2,
                "injected_items_count": 1,
                "skipped_items_count": 0,
                "injected_context_tokens": 10,
                "original_estimated_tokens": 100,
                "final_estimated_tokens": 112,
                "token_delta": 12,
                "injection_role": "system",
                "injection_position": "after_system",
                "warnings": ["api_key=sk-secret-value cookie=session"],
                "final_messages": [{"content": "must not forward"}],
                "injected_context": "<INJECTED_CONTEXT>must not forward</INJECTED_CONTEXT>",
                "content": "must not forward",
                "metadata": {"secret": "must not forward"},
                "schema": "must not forward",
                "raw_output": "must not forward",
            },
            state={"request_id": "r1", "thread_id": "thread-1"},
        )
        emit_a3_trace(
            logging.getLogger("test_context_apply_trace"),
            "context_apply_selection",
            {
                "node_name": "generate_answer",
                "llm_node": "academic",
                "skip_reason": "budget_fit_failed",
                "single_resource_result": "matched_single_resource",
                "selected_item_count": 3,
                "injectable_item_count": 2,
                "skipped_item_count": 1,
                "quality_filtered_count": 1,
                "budget_dropped_count": 1,
                "final_injected_count": 1,
                "injected_context_tokens": 10,
                "original_estimated_tokens": 100,
                "final_estimated_tokens": 112,
                "token_delta": 12,
                "source_counts_before": {"memory": 2},
                "source_counts_after": {"memory": 1},
                "drop_reasons": {"over_budget": 1},
                "warnings": ["api_key=sk-secret-value cookie=session"],
                "content": "must not forward",
                "metadata": {"secret": "must not forward"},
            },
            state={"request_id": "r1", "thread_id": "thread-1"},
        )
        emit_a3_trace(
            logging.getLogger("test_context_apply_trace"),
            "context_apply_error",
            {
                "node_name": "generate_answer",
                "llm_node": "academic",
                "reason": "apply_failed",
                "warning": "api_key=sk-secret-value db_uri=postgresql://u:p@h/db",
                "fallback_used": True,
                "error_type": "RuntimeError",
                "final_messages": [{"content": "must not forward"}],
            },
            state={"request_id": "r1", "thread_id": "thread-1"},
        )
        emit_a3_trace(
            logging.getLogger("test_context_apply_trace"),
            "context_importance_scored",
            {
                "node_name": "generate_answer",
                "llm_node": "academic",
                "source_counts": {"memory": 2},
                "score_buckets": {"0.75-1.00": 2},
                "reason_code_counts": {"useful": 2},
                "candidate_count": 2,
                "scored_count": 2,
                "kept_count": 1,
                "dropped_count": 1,
                "fallback_to_rule_based": False,
                "scoring_elapsed_ms": 5,
                "disabled_reason": "",
                "error_reason": "",
                "error_type": "",
                "warnings": ["db_uri=postgresql://u:p@h/db"],
                "title": "must not forward",
                "content_preview": "must not forward",
                "raw_response": "must not forward",
                "schema": "must not forward",
                "metadata": {"secret": "must not forward"},
            },
            state={"request_id": "r1", "thread_id": "thread-1"},
        )
        yield {
            "event": "on_chain_start",
            "name": "generate_answer",
            "metadata": {"langgraph_node": "generate_answer"},
            "data": {"input": {}},
        }

    graph = MagicMock()
    graph.astream_events = MagicMock(return_value=events())
    graph.aget_state = AsyncMock(
        return_value=_snapshot({"schema_version": "run_control_v1"})
    )
    graph.aupdate_state = AsyncMock()

    collected = []
    async for item in generate_sse("q", graph, thread_id="thread-1"):
        collected.append(item)

    payloads = _payloads(collected)
    apply_payloads = [
        payload
        for payload in payloads
        if payload.get("type")
        in {
            "context_applied",
            "context_apply_error",
            "context_apply_selection",
            "context_importance_scored",
        }
    ]
    serialized = repr(apply_payloads).lower()
    assert len(apply_payloads) == 4
    assert apply_payloads[0]["injected_context_tokens"] == 10
    for payload in apply_payloads:
        assert "injected_context" not in payload
        assert "final_messages" not in payload
        assert "content" not in payload
        assert "metadata" not in payload
        assert "schema" not in payload
        assert "raw_output" not in payload
        assert "raw_response" not in payload
        assert "content_preview" not in payload
        assert "title" not in payload
    assert "must not forward" not in serialized
    assert "<injected_context>" not in serialized
    assert "api_key" not in serialized
    assert "db_uri" not in serialized
    assert "postgresql://" not in serialized
