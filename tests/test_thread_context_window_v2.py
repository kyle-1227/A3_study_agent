from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from src.context_engineering.input_accounting import build_llm_input_accounting
from src.context_engineering.thread_window import build_thread_context_window_v2


def _report(*, manifest_id: str, node_name: str, input_tokens: int) -> dict:
    return {
        "report_id": "context_usage:v1:report",
        "manifest_id": manifest_id,
        "created_at": "2026-07-12T00:00:00+00:00",
        "node_name": node_name,
        "llm_node": node_name,
        "model": "configured-model",
        "input_estimated_tokens": input_tokens,
        "output_reserved_tokens": 512,
        "used_tokens": input_tokens + 512,
        "max_context_tokens": 32_000,
        "used_ratio": (input_tokens + 512) / 32_000,
        "estimated": True,
        "tokenizer_mode": "estimated_mixed",
        "detailed_categories": [
            {
                "category": "recent_messages",
                "estimated_tokens": input_tokens,
                "segment_count": 1,
                "message_count": 1,
            }
        ],
    }


def test_thread_baseline_is_read_only_and_marks_unknown_sections():
    state = {
        "thread_id": "thread-1",
        "messages": [HumanMessage(content="Explain gradient descent")],
        "conversation_summary": "The learner is studying optimization.",
        "task_workspace": {
            "schema_version": 1,
            "workspace_id": "workspace:v1:abc",
            "active_subject": "machine learning",
            "evidence_summaries": [
                {
                    "evidence_id": "evidence:v1:abc",
                    "summary": "private source body must not escape",
                }
            ],
            "coverage_gaps": [],
            "artifacts_by_id": {},
        },
    }
    original = deepcopy(state)

    window = build_thread_context_window_v2(state)

    assert state == original
    assert window["schema_version"] == 2
    estimate = window["next_call_context_estimate"]
    assert estimate["basis"] == "thread_baseline"
    assert estimate["confidence"] == "low"
    assert "ce_block" in estimate["unknown_sections"]
    assert estimate["estimated_input_tokens"] == sum(
        section["estimated_tokens"] for section in estimate["sections"]
    )
    assert "private source body must not escape" not in str(window)


def test_known_node_reuses_report_only_when_message_fingerprint_matches():
    messages = [HumanMessage(content="Build a concept map")]
    accounting = build_llm_input_accounting(messages)
    manifest_id = "llm_input_manifest:v1:abc"
    report = _report(
        manifest_id=manifest_id,
        node_name="mindmap_agent",
        input_tokens=accounting.input_estimated_tokens,
    )
    state = {
        "thread_id": "thread-1",
        "messages": messages,
        "llm_input_manifest": {
            "manifest_id": manifest_id,
            "node_name": "mindmap_agent",
            "message_fingerprint": accounting.message_fingerprint,
        },
        "context_usage_report": report,
    }

    estimate = build_thread_context_window_v2(
        state,
        target_node="mindmap_agent",
    )["next_call_context_estimate"]

    assert estimate["basis"] == "known_next_node"
    assert estimate["confidence"] == "high"
    assert estimate["reused_manifest_statistics"] is True
    assert estimate["estimated_input_tokens"] == report["input_estimated_tokens"]
    assert estimate["unknown_sections"] == []


def test_fingerprint_mismatch_never_turns_hash_into_token_accounting():
    messages = [HumanMessage(content="A short current request")]
    manifest_id = "llm_input_manifest:v1:old"
    report = _report(
        manifest_id=manifest_id,
        node_name="review_doc_agent",
        input_tokens=19_999,
    )
    state = {
        "thread_id": "thread-1",
        "messages": messages,
        "llm_input_manifest": {
            "manifest_id": manifest_id,
            "node_name": "review_doc_agent",
            "message_fingerprint": "0" * 64,
        },
        "context_usage_report": report,
    }

    estimate = build_thread_context_window_v2(
        state,
        target_node="review_doc_agent",
    )["next_call_context_estimate"]

    assert estimate["reused_manifest_statistics"] is False
    assert estimate["estimated_input_tokens"] != 19_999
    assert estimate["estimated_input_tokens"] == sum(
        section["estimated_tokens"] for section in estimate["sections"]
    )
    assert estimate["max_context_tokens"] == 32_000


def test_corrupt_optional_history_degrades_to_a_safe_empty_projection():
    window = build_thread_context_window_v2(
        {
            "thread_id": "thread-1",
            "messages": "invalid",
            "context_usage_report": "invalid",
            "context_usage_reports": {"unexpected": "mapping"},
            "llm_input_manifests": "invalid",
            "task_workspace": {"schema_version": 999},
        }
    )

    assert window["last_llm_call_usage"]["present"] is False
    assert window["background_inventory"]["workspace_present"] is False
    assert window["next_call_context_estimate"]["estimated_input_tokens"] == 0


def test_thread_status_adds_v2_without_removing_legacy_fields():
    from app import _thread_status_from_snapshot

    snapshot = SimpleNamespace(
        next=(),
        tasks=[],
        values={
            "schema_version": "run_control_v1",
            "run_status": "completed",
            "stop_requested": False,
            "stop_reason": "",
            "current_node": "",
            "last_completed_node": "qa_agent",
            "resume_available": False,
            "stopped_at": "",
            "pending_interrupt_type": "",
            "context_usage": {},
            "context_usage_history": [],
            "thread_id": "thread-1",
            "messages": [HumanMessage(content="A thread question")],
        },
    )

    payload = _thread_status_from_snapshot("thread-1", snapshot).model_dump()

    assert "background_context_window" in payload
    assert "thread_context_window" in payload
    assert payload["thread_context_window_v2"]["schema_version"] == 2
    assert payload["thread_context_window_v2"]["thread_id"] == "thread-1"
