"""SSE integration tests for Phase 5 observability contracts."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage

from src.observability.a3_trace import emit_a3_trace
from src.observability.llm_input import (
    build_llm_input_observation,
    emit_llm_input_usage,
)

GRAPH_VERSION = "graph:v1:0123456789abcdef0123456789abcdef"


class AsyncIterator:
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


def _snapshot(
    values: dict | None = None,
    *,
    next_nodes: tuple[str, ...] = (),
    tasks: list | None = None,
):
    return SimpleNamespace(next=next_nodes, tasks=tasks or [], values=values or {})


def _node_event(node: str, event_type: str) -> dict:
    return {
        "event": event_type,
        "name": node,
        "metadata": {"langgraph_node": node},
        "data": {"output": {}} if event_type == "on_chain_end" else {"input": {}},
    }


def _payloads(events: list[str]) -> list[dict]:
    return [json.loads(item.removeprefix("data: ").strip()) for item in events]


def _graph(
    events,
    *,
    state_values: dict | None = None,
    activity: bool = True,
    snapshot=None,
):
    graph = MagicMock()
    graph._a3_activity_events_enabled = activity
    graph._a3_graph_version = GRAPH_VERSION
    graph._a3_checkpointer_enabled = False
    graph._a3_checkpointer_type = "disabled"
    stream = events if hasattr(events, "__aiter__") else AsyncIterator(events)
    graph.astream_events = MagicMock(return_value=stream)
    state_snapshot = snapshot or _snapshot(state_values or {})
    graph.aget_state = AsyncMock(return_value=state_snapshot)
    graph.aupdate_state = AsyncMock()
    return graph


@pytest.mark.anyio
async def test_activity_sse_updates_same_node_and_stream_ids_and_persists_timeline():
    from app import _stream_graph_events

    graph = _graph(
        [
            _node_event("supervisor", "on_chain_start"),
            _node_event("supervisor", "on_chain_end"),
        ]
    )
    collected = []
    async for item in _stream_graph_events(
        graph,
        {"request_id": "r1", "thread_id": "t1"},
        {"configurable": {"thread_id": "t1"}},
        "t1",
        request_id="r1",
        preserve_context_history=True,
    ):
        collected.append(item)

    payloads = _payloads(collected)
    activities = [item for item in payloads if item.get("type") == "activity_event"]
    node_activities = [item for item in activities if item.get("node") == "supervisor"]
    stream_activities = [item for item in activities if item.get("kind") == "stream"]

    assert [item["status"] for item in node_activities] == ["running", "completed"]
    assert len({item["activity_id"] for item in node_activities}) == 1
    assert [item["status"] for item in stream_activities] == ["running", "completed"]
    assert len({item["activity_id"] for item in stream_activities}) == 1
    updates = [call.args[1] for call in graph.aupdate_state.await_args_list]
    persisted = [item for item in updates if "activity_timeline" in item]
    assert persisted
    assert {item["status"] for item in persisted[-1]["activity_timeline"]} == {
        "completed"
    }


@pytest.mark.anyio
async def test_usage_report_and_legacy_usage_emit_from_same_snapshot_and_persist():
    from app import _stream_graph_events

    observation = build_llm_input_observation(
        node_name="qa_agent",
        llm_node="qa_agent",
        provider="deepseek_official",
        model="deepseek-v4-pro",
        messages=[
            SystemMessage(content="secret-system-body"),
            HumanMessage(content="secret-user-question"),
        ],
        state={"request_id": "r1", "thread_id": "t1"},
        call_purpose="structured_llm",
    )

    async def events():
        emit_llm_input_usage(
            logging.getLogger("test_usage_report_sse"),
            observation,
            state={"request_id": "r1", "thread_id": "t1"},
        )
        yield _node_event("qa_agent", "on_chain_start")
        yield _node_event("qa_agent", "on_chain_end")

    graph = _graph(events(), activity=True)
    collected = []
    async for item in _stream_graph_events(
        graph,
        {"request_id": "r1", "thread_id": "t1"},
        {"configurable": {"thread_id": "t1"}},
        "t1",
        request_id="r1",
        preserve_context_history=True,
    ):
        collected.append(item)

    payloads = _payloads(collected)
    report = next(
        item for item in payloads if item.get("type") == "context_usage_report"
    )
    legacy = next(item for item in payloads if item.get("type") == "context_usage")

    assert report["input_estimated_tokens"] == legacy["input_estimated_tokens"]
    assert report["used_tokens"] == legacy["used_tokens"]
    serialized_report = json.dumps(report, ensure_ascii=False)
    assert "secret-system-body" not in serialized_report
    assert "secret-user-question" not in serialized_report
    updates = [call.args[1] for call in graph.aupdate_state.await_args_list]
    assert any("context_usage_report" in item for item in updates)


@pytest.mark.anyio
async def test_usage_report_error_sse_preserves_versioned_contract(monkeypatch):
    from app import _stream_graph_events

    monkeypatch.setenv("LOG_A3_TRACE", "true")

    async def events():
        emit_a3_trace(
            logging.getLogger("test_usage_report_error_sse"),
            "context_usage_report_error",
            {
                "manifest_id": "llm_input_manifest:v1:failed",
                "node_name": "qa_agent",
                "llm_node": "qa_agent",
                "provider": "configured-provider",
                "model": "configured-model",
                "reason": "budget_unavailable",
                "warning": "budget accounting failed",
                "error_type": "ContextUsageError",
            },
            state={"request_id": "r1", "thread_id": "t1"},
            env_flag="LOG_A3_TRACE",
        )
        yield _node_event("qa_agent", "on_chain_start")
        yield _node_event("qa_agent", "on_chain_end")

    graph = _graph(events(), activity=True)
    collected = []
    async for item in _stream_graph_events(
        graph,
        {"request_id": "r1", "thread_id": "t1"},
        {"configurable": {"thread_id": "t1"}},
        "t1",
        request_id="r1",
        preserve_context_history=True,
    ):
        collected.append(item)

    error = next(
        item
        for item in _payloads(collected)
        if item.get("type") == "context_usage_report_error"
    )
    assert error["schema_version"] == "context_usage_report_error_v1"
    assert error["node_name"] == "qa_agent"
    assert error["node"] == "qa_agent"


@pytest.mark.anyio
async def test_stream_context_and_manifest_ref_precede_graph_execution():
    from app import generate_sse

    graph = _graph([], activity=False)
    collected = []
    async for item in generate_sse(
        "hello",
        graph,
        thread_id="t1",
        graph_version=GRAPH_VERSION,
    ):
        collected.append(item)

    payloads = _payloads(collected)
    types = [item.get("type") for item in payloads]
    assert types[:4] == [
        "thread_id",
        "stream_context",
        "graph_manifest_ref",
        "run_status",
    ]
    assert payloads[1]["thread_id"] == "t1"
    assert payloads[1]["request_id"]
    assert payloads[1]["graph_version"] == GRAPH_VERSION
    assert payloads[2]["endpoint"] == "/graph/manifest"


def test_thread_status_restores_report_activity_and_graph_version():
    from app import _thread_status_from_snapshot

    observation = build_llm_input_observation(
        node_name="qa_agent",
        llm_node="qa_agent",
        provider="deepseek_official",
        model="deepseek-v4-pro",
        messages=[HumanMessage(content="question")],
        state={"request_id": "r1", "thread_id": "t1"},
        call_purpose="structured_llm",
    )
    report = observation.context_usage_report
    assert report is not None
    activity = {
        "schema_version": "activity_event_v1",
        "activity_id": "activity:v1:0123456789abcdef0123456789abcdef",
        "sequence": 1,
        "thread_id": "t1",
        "request_id": "r1",
        "kind": "stream",
        "status": "completed",
        "node": "",
        "parent": "",
        "title": "Completed",
        "summary": "",
        "tool": "",
        "model": "",
        "started_at": "2026-07-10T00:00:00+00:00",
        "updated_at": "2026-07-10T00:00:01+00:00",
        "completed_at": "2026-07-10T00:00:01+00:00",
        "duration_ms": 1000,
        "safe_details": {},
    }
    values = {
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
        "context_usage_report": report.model_dump(mode="json"),
        "context_usage_reports": [report.model_dump(mode="json")],
        "activity_timeline": [activity],
        "graph_version": GRAPH_VERSION,
    }

    status = _thread_status_from_snapshot("t1", _snapshot(values))

    assert status.context_usage_report["report_id"] == report.report_id
    assert status.context_usage_report_count == 1
    assert status.activity_timeline_count == 1
    assert status.activity_timeline[0]["status"] == "completed"
    assert status.graph_version == GRAPH_VERSION


def test_missing_cached_manifest_returns_typed_503():
    from app import _app_graph_version

    fake_app = SimpleNamespace(
        state=SimpleNamespace(
            graph_version="",
            graph_manifest_error={
                "schema_version": "graph_manifest_error_v1",
                "error": "graph_manifest_unavailable",
                "reason": "topology unavailable",
                "error_type": "TopologyError",
            },
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        _app_graph_version(fake_app)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["error"] == "graph_manifest_unavailable"


@pytest.mark.anyio
async def test_manifest_endpoint_returns_cached_contract_and_typed_cache_error():
    from app import graph_manifest_endpoint
    from src.observability.graph_manifest import build_graph_manifest
    from src.graph.builder import get_compiled_graph

    manifest = build_graph_manifest(
        get_compiled_graph(),
        context_policy_mode="strict",
        checkpointer_enabled=False,
        checkpointer_type="disabled",
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(graph_manifest=manifest))
    )
    assert await graph_manifest_endpoint(request) is manifest

    broken_request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                graph_manifest=None,
                graph_manifest_error=None,
                graph_version=GRAPH_VERSION,
            )
        )
    )
    with pytest.raises(HTTPException) as exc_info:
        await graph_manifest_endpoint(broken_request)
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["error"] == "graph_manifest_unavailable"


@pytest.mark.anyio
async def test_interrupt_finalizes_stream_activity_as_waiting_and_persists():
    from app import _stream_graph_events

    interrupt = SimpleNamespace(
        value={
            "type": "memory_confirmation",
            "question": "Use selected memory?",
            "reason": "ambiguous",
            "selected_memory_count": 1,
            "options": [{"label": "Use", "value": "use"}],
        }
    )
    graph = _graph(
        [],
        activity=True,
        snapshot=_snapshot(
            {},
            next_nodes=("memory_use_decider",),
            tasks=[SimpleNamespace(interrupts=[interrupt])],
        ),
    )
    collected = []
    async for item in _stream_graph_events(
        graph,
        {"request_id": "r-interrupt", "thread_id": "t1"},
        {"configurable": {"thread_id": "t1"}},
        "t1",
        request_id="r-interrupt",
    ):
        collected.append(item)

    payloads = _payloads(collected)
    activities = [item for item in payloads if item.get("type") == "activity_event"]
    assert any(
        item["kind"] == "stream" and item["status"] == "interrupted"
        for item in activities
    )
    assert any(
        item["kind"] == "interrupt" and item["status"] == "waiting"
        for item in activities
    )
    updates = [call.args[1] for call in graph.aupdate_state.await_args_list]
    assert any("activity_timeline" in update for update in updates)


@pytest.mark.anyio
async def test_stream_exception_finalizes_failed_activity_without_raw_error():
    from app import _stream_graph_events

    async def failed_events():
        if False:
            yield {}
        raise RuntimeError("api_key=sk-secret-provider-body")

    graph = _graph(failed_events(), activity=True)
    collected = []
    async for item in _stream_graph_events(
        graph,
        {"request_id": "r-failed", "thread_id": "t1"},
        {"configurable": {"thread_id": "t1"}},
        "t1",
        request_id="r-failed",
    ):
        collected.append(item)

    payloads = _payloads(collected)
    failed = next(
        item
        for item in payloads
        if item.get("type") == "activity_event"
        and item.get("kind") == "stream"
        and item.get("status") == "failed"
    )
    assert failed["safe_details"] == {"error_type": "RuntimeError"}
    assert "sk-secret-provider-body" not in json.dumps(payloads, ensure_ascii=False)
