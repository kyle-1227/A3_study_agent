"""LLM input manifest and thread background-window tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import HumanMessage, SystemMessage
import pytest

from app import _context_window_status
from src.context_engineering.input_manifest import (
    build_background_context_window,
    build_llm_input_manifest,
    llm_input_manifest_trace_payload,
    merge_llm_input_manifest_history,
)
from src.context_engineering.influence import (
    build_influence_entry,
    build_influence_update,
    merge_context_influence_ledger,
)
from src.graph.state import initial_request_reset_transient_state


def test_manifest_trace_payload_is_safe_and_bounded():
    manifest = build_llm_input_manifest(
        node_name="review_doc_agent",
        llm_node="review_doc",
        provider="deepseek_official",
        model="deepseek-v4-pro",
        messages=[
            SystemMessage(content="system prompt with api_key=sk-secret-value"),
            HumanMessage(content="Please write the full machine learning notes."),
        ],
        state={
            "request_id": "r1",
            "thread_id": "t1",
            "conversation_summary": "prior compact summary",
        },
        call_purpose="plain_llm",
    )

    payload = llm_input_manifest_trace_payload(manifest)
    serialized = json.dumps(payload, ensure_ascii=False)

    assert payload["manifest_id"].startswith("llm_input_manifest:v1:")
    assert payload["section_names"] == [
        "provider_bound_messages",
        "conversation_summary",
    ]
    assert "sk-secret-value" not in serialized
    assert "Please write the full machine learning notes." not in serialized
    assert "system prompt" not in serialized


def test_manifest_history_dedupes_by_stable_id():
    manifest = build_llm_input_manifest(
        node_name="mindmap_agent",
        llm_node="mindmap",
        provider="deepseek_official",
        model="deepseek-v4-pro",
        messages=[HumanMessage(content="same prompt")],
        state={"request_id": "r1", "thread_id": "t1"},
        call_purpose="plain_llm",
    )
    payload = llm_input_manifest_trace_payload(manifest)

    history = merge_llm_input_manifest_history([payload], [payload])

    assert len(history) == 1
    assert history[0]["manifest_id"] == payload["manifest_id"]


def test_background_context_window_uses_manifest_workspace_metadata():
    manifest = build_llm_input_manifest(
        node_name="mindmap_agent",
        llm_node="mindmap",
        provider="deepseek_official",
        model="deepseek-v4-pro",
        messages=[HumanMessage(content="mindmap")],
        state={
            "request_id": "r1",
            "thread_id": "t1",
            "task_workspace": {
                "schema_version": 1,
                "workspace_id": "workspace:v1:abc",
                "active_subject": "机器学习",
                "evidence_summaries": [{"evidence_id": "e1"}],
                "coverage_gaps": [],
                "artifacts_by_id": {"a1": {"artifact_id": "a1"}},
            },
        },
        call_purpose="plain_llm",
    )

    window = build_background_context_window(
        manifest=llm_input_manifest_trace_payload(manifest),
        state={},
        manifest_count=1,
        max_context_tokens=1000,
    )

    assert window["workspace_present"] is True
    assert window["workspace_active_subject"] == "机器学习"
    assert window["workspace_evidence_summary_count"] == 1
    assert window["workspace_artifact_count"] == 1


def test_manifest_and_background_window_include_safe_influence_counts():
    influence_state = {"request_id": "r1", "thread_id": "t1"}
    influence_ledger = merge_context_influence_ledger(
        {},
        build_influence_update(
            state=influence_state,
            entries=[
                build_influence_entry(
                    state=influence_state,
                    kind="planner_output",
                    source_node="mindmap_planner",
                    preview="Compact outline only.",
                )
            ],
        ),
    )
    manifest = build_llm_input_manifest(
        node_name="mindmap_agent",
        llm_node="mindmap",
        provider="deepseek_official",
        model="deepseek-v4-pro",
        messages=[HumanMessage(content="mindmap")],
        state={**influence_state, "context_influence_ledger": influence_ledger},
        call_purpose="plain_llm",
    )

    assert "context_influence_ledger" in manifest["section_names"]
    section = next(
        item
        for item in manifest["sections"]
        if item["section"] == "context_influence_ledger"
    )
    assert section["item_count"] == 1
    assert "Compact outline only." not in repr(section)

    window = build_background_context_window(
        manifest=manifest,
        state={"context_influence_ledger": influence_ledger},
        manifest_count=1,
    )
    assert window["influence_entry_count"] == 1
    assert window["influence_source_node_count"] == 1


def test_request_reset_preserves_thread_background_context_fields():
    reset = initial_request_reset_transient_state()

    assert "task_workspace" not in reset
    assert "llm_input_manifests" not in reset
    assert "thread_context_ledger" not in reset
    assert "background_context_window" not in reset
    assert reset["llm_input_manifest"] == {}


def test_context_window_status_adds_background_fields():
    _, thread_window = _context_window_status(
        {
            "llm_input_manifests": [{"manifest_id": "m1"}],
            "background_context_window": {
                "last_manifest_id": "m1",
                "used_tokens": 147000,
                "max_context_tokens": 258000,
                "used_ratio": 0.57,
                "updated_at": "2026-07-08T00:00:00+00:00",
                "manifest_count": 1,
            },
        }
    )

    assert thread_window["llm_input_manifest_count"] == 1
    assert thread_window["background_context_window_present"] is True
    assert thread_window["background_context_window_used_tokens"] == 147000
    assert thread_window["background_context_window_max_tokens"] == 258000


@pytest.mark.anyio
async def test_manifest_trace_update_persists_background_context_checkpoint():
    from app import _update_llm_manifest_state_from_trace

    manifest = build_llm_input_manifest(
        node_name="mindmap_agent",
        llm_node="mindmap",
        provider="deepseek_official",
        model="deepseek-v4-pro",
        messages=[HumanMessage(content="mindmap")],
        state={"request_id": "r1", "thread_id": "t1"},
        call_purpose="plain_llm",
    )
    event = {
        "stage": "llm_input_manifest.built",
        **llm_input_manifest_trace_payload(manifest),
    }
    graph = MagicMock()
    graph.aupdate_state = AsyncMock()

    await _update_llm_manifest_state_from_trace(
        graph,
        {"configurable": {"thread_id": "t1"}},
        thread_id="t1",
        event=event,
        llm_input_manifests=[],
        state_context={"thread_id": "t1"},
    )

    graph.aupdate_state.assert_awaited_once()
    _config, values = graph.aupdate_state.await_args.args
    assert values["llm_input_manifest"]["manifest_id"] == event["manifest_id"]
    assert values["llm_input_manifests"]
    assert values["thread_context_ledger"]
    assert values["background_context_window"]


def test_task_continuity_avoids_long_term_memory_confirmation():
    from src.graph.academic import _deterministic_memory_use_decision

    decision = _deterministic_memory_use_decision(
        "再给我一份思维导图",
        selected_memory_count=2,
        task_continuity_resolved=True,
    )

    assert decision is not None
    assert decision.decision == "ignore"
    assert "task workspace" in decision.reason


def test_production_llm_calls_stay_inside_manifest_guarded_boundaries():
    root = Path("src")
    python_files = [path for path in root.rglob("*.py") if path.is_file()]
    provider_boundary_files = {
        Path("src/graph/llm.py"),
        Path("src/llm/structured_output.py"),
    }
    allowed_non_chat_http = {
        Path("src/memory/embeddings.py"),
        Path("src/rag/indexer.py"),
        Path("src/rag/reranker.py"),
        Path("src/tools/search_tool.py"),
    }
    raw_call_violations: list[str] = []
    chat_http_violations: list[str] = []
    fallback_usage_violations: list[str] = []

    for path in python_files:
        text = path.read_text(encoding="utf-8")
        normalized = Path(path.as_posix())
        if normalized not in provider_boundary_files:
            if ".ainvoke(" in text or ".invoke(" in text or "ChatOpenAI(" in text:
                raw_call_violations.append(path.as_posix())
        if "/chat/completions" in text and normalized not in provider_boundary_files:
            if normalized not in allowed_non_chat_http:
                chat_http_violations.append(path.as_posix())
        if "invoke_with_fallback(" in text and normalized != Path("src/graph/llm.py"):
            fallback_usage_violations.append(path.as_posix())

    assert raw_call_violations == []
    assert chat_http_violations == []
    assert fallback_usage_violations == []
