"""Unified LLM input accounting and Context Usage Report tests."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from src.context_engineering.itemizer import make_context_item
from src.observability.contracts import ContextUsageReport
from src.observability.context_usage_report import (
    CONTEXT_USAGE_REPORT_SEGMENT_LIMIT,
)
from src.observability.llm_input import build_llm_input_observation


def _context_item(source_type: str, *, item_id: str, metadata: dict | None = None):
    return make_context_item(
        source_type=source_type,
        title=f"{source_type} summary",
        content="bounded safe summary",
        priority=70,
        scope="session",
        lifetime="session",
        compressible=True,
        can_drop=True,
        disclosure_level="summary",
        item_id=item_id,
        metadata=metadata or {},
    )


def _observation():
    items = (
        _context_item(
            "pipeline",
            item_id="influence:v1:planner",
            metadata={
                "influence_kind": "planner_output",
                "source_node": "mindmap_planner",
            },
        ),
        _context_item(
            "evidence",
            item_id="evidence:v1:kept",
            metadata={
                "retrieval_mode": "graded_evidence",
                "purpose": "factual_grounding",
            },
        ),
        _context_item("artifact", item_id="artifact:v1:prior"),
    )
    messages = [
        SystemMessage(content="Business policy"),
        SystemMessage(
            content=(
                "<INJECTED_CONTEXT>\n"
                "api_key=sk-this-must-never-be-persisted\n"
                "bounded context\n"
                "</INJECTED_CONTEXT>"
            )
        ),
        HumanMessage(content="Generate a new map from the prior learning task."),
    ]
    return build_llm_input_observation(
        node_name="mindmap_agent",
        llm_node="mindmap",
        provider="deepseek_official",
        model="deepseek-v4-pro",
        messages=messages,
        state={"request_id": "r1", "thread_id": "t1"},
        call_purpose="structured_llm",
        output_mode="deepseek_tool_call_strict",
        context_apply_applied=True,
        context_apply_status="applied",
        provider_bound_messages_mutated=True,
        context_items=items,
    )


def test_report_reconciles_main_segments_and_overlap_without_double_counting():
    observation = _observation()
    report = observation.context_usage_report
    assert report is not None

    assert (
        sum(item.estimated_tokens for item in report.main_categories)
        == report.input_estimated_tokens
    )
    assert (
        sum(item.estimated_tokens for item in report.segments)
        == report.input_estimated_tokens
    )
    overlap = next(
        item for item in report.overlap_rollups if item.category == "injected_context"
    )
    assert overlap.estimated_tokens > 0
    assert (
        report.used_tokens
        == report.input_estimated_tokens + report.output_reserved_tokens
    )
    assert report.reconciliation_ok is True
    assert report.reconciliation_warnings == []


def test_report_has_content_free_fingerprints_and_provenance():
    observation = _observation()
    report = observation.context_usage_report
    assert report is not None
    serialized = report.model_dump_json()

    assert "Generate a new map" not in serialized
    assert "Business policy" not in serialized
    assert "sk-this-must-never-be-persisted" not in serialized
    assert all(segment.fingerprint for segment in report.segments)
    assert any(
        segment.provenance.get("source_id") == "evidence:v1:kept"
        for segment in report.segments
    )
    details = {item.category for item in report.detailed_categories}
    assert {"original_query", "planner", "judge", "artifact"} <= details


def test_legacy_usage_is_projected_from_the_same_report():
    observation = _observation()
    report = observation.context_usage_report
    legacy = observation.legacy_context_usage
    assert report is not None and legacy is not None

    assert legacy["input_estimated_tokens"] == report.input_estimated_tokens
    assert legacy["reserved_output_tokens"] == report.output_reserved_tokens
    assert legacy["used_tokens"] == report.used_tokens
    assert legacy["max_context_tokens"] == report.max_context_tokens


def test_observation_counts_each_provider_message_once(monkeypatch):
    import src.context_engineering.input_accounting as accounting_module

    calls = 0
    original = accounting_module.estimate_text_tokens_mixed

    def counted(text: str) -> int:
        nonlocal calls
        calls += 1
        return original(text)

    monkeypatch.setattr(accounting_module, "estimate_text_tokens_mixed", counted)
    observation = _observation()

    assert observation.manifest["message_count"] == 3
    assert calls == 3


def test_report_contract_rejects_reconciliation_drift():
    observation = _observation()
    report = observation.context_usage_report
    assert report is not None
    payload = report.model_dump(mode="json")
    payload["input_estimated_tokens"] += 1

    with pytest.raises(ValidationError, match="reconcile"):
        ContextUsageReport.model_validate(payload)


def test_manifest_and_report_ids_are_stable_for_same_provider_input():
    first = _observation()
    second = _observation()

    assert first.manifest["manifest_id"] == second.manifest["manifest_id"]
    assert first.context_usage_report is not None
    assert second.context_usage_report is not None
    assert first.context_usage_report.report_id == second.context_usage_report.report_id
    assert "bounded safe summary" not in json.dumps(
        first.manifest,
        ensure_ascii=False,
    )


def test_large_provider_input_is_compacted_without_losing_token_reconciliation():
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": "x" * 80}
        for index in range(100)
    ]

    observation = build_llm_input_observation(
        node_name="qa_agent",
        llm_node="qa_agent",
        provider="deepseek_official",
        model="deepseek-v4-pro",
        messages=messages,
        state={"request_id": "r-large", "thread_id": "t1"},
        call_purpose="structured_llm",
    )
    report = observation.context_usage_report
    assert report is not None

    assert len(report.segments) <= CONTEXT_USAGE_REPORT_SEGMENT_LIMIT
    assert sum(item.estimated_tokens for item in report.segments) == (
        report.input_estimated_tokens
    )
    assert "segments_compacted" in report.reconciliation_warnings
    assert report.reconciliation_ok is True


def test_unclassified_input_is_explicit_and_does_not_break_reconciliation():
    observation = build_llm_input_observation(
        node_name="qa_agent",
        llm_node="qa_agent",
        provider="deepseek_official",
        model="deepseek-v4-pro",
        messages=[{"role": "unknown_role", "content": "bounded"}],
        state={"request_id": "r-unknown", "thread_id": "t1"},
        call_purpose="structured_llm",
    )
    report = observation.context_usage_report
    assert report is not None

    assert report.unclassified_tokens == report.input_estimated_tokens
    assert "unclassified_tokens_present" in report.reconciliation_warnings
    assert report.reconciliation_ok is True


def test_report_sse_payload_stays_below_ordinary_event_limit():
    items = tuple(
        _context_item(
            "pipeline",
            item_id=f"influence:v1:{index:064d}",
            metadata={
                "influence_kind": "planner_output",
                "source_node": "planner_node_with_bounded_metadata",
                "purpose": "pipeline_continuity",
            },
        )
        for index in range(64)
    )
    observation = build_llm_input_observation(
        node_name="mindmap_agent",
        llm_node="mindmap",
        provider="deepseek_official",
        model="deepseek-v4-pro",
        messages=[
            SystemMessage(
                content=(
                    "<INJECTED_CONTEXT>\n"
                    + ("bounded context\n" * 1200)
                    + "</INJECTED_CONTEXT>"
                )
            )
        ],
        state={"request_id": "r-size", "thread_id": "t1"},
        call_purpose="structured_llm",
        context_apply_applied=True,
        context_items=items,
    )
    report = observation.context_usage_report
    assert report is not None
    payload = {"type": "context_usage_report", **report.model_dump(mode="json")}

    assert len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) < 16 * 1024


def test_non_strict_observation_failure_is_explicit_and_keeps_manifest(monkeypatch):
    import src.observability.llm_input as llm_input_module

    monkeypatch.setattr(
        llm_input_module,
        "get_context_engineering_config",
        lambda: {"strict": False},
    )

    observation = build_llm_input_observation(
        node_name="qa_agent",
        llm_node="qa_agent",
        provider="deepseek_official",
        model="model-without-configured-window",
        messages=[{"role": "user", "content": "unchanged provider input"}],
        state={"request_id": "r-error", "thread_id": "t1"},
        call_purpose="structured_llm",
    )

    assert observation.manifest["message_count"] == 1
    assert observation.context_usage_report is None
    assert observation.legacy_context_usage is None
    assert observation.context_usage_error is not None
    assert observation.context_usage_error["reason"] == "model_window_unknown"
    assert "unchanged provider input" not in json.dumps(
        observation.context_usage_error,
        ensure_ascii=False,
    )
