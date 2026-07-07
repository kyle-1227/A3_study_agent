"""Focused tests for Run Control stop/status/continue behavior."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.observability.a3_trace import emit_a3_trace


class AsyncIteratorMock:
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


def _payloads(collected: list[str]) -> list[dict]:
    return [json.loads(item.removeprefix("data: ").strip()) for item in collected]


def _snapshot(
    *, values: dict | None = None, interrupt_value=None, next_nodes=()
) -> SimpleNamespace:
    tasks = []
    if interrupt_value is not None:
        tasks = [SimpleNamespace(interrupts=[SimpleNamespace(value=interrupt_value)])]
    return SimpleNamespace(next=next_nodes, tasks=tasks, values=values or {})


@pytest.mark.anyio
async def test_stop_gate_wraps_sync_node_without_changing_output():
    from src.run_control import wrap_interruptible_node

    def sync_node(state):
        return {"answer": state["answer"]}

    wrapped = wrap_interruptible_node("sync_node", sync_node)

    assert await wrapped({"answer": "ok", "stop_requested": False}) == {"answer": "ok"}


@pytest.mark.anyio
async def test_stop_gate_wraps_async_node_without_changing_output():
    from src.run_control import wrap_interruptible_node

    async def async_node(state):
        return {"answer": state["answer"]}

    wrapped = wrap_interruptible_node("async_node", async_node)

    assert await wrapped({"answer": "ok", "stop_requested": False}) == {"answer": "ok"}


@pytest.mark.anyio
async def test_stop_gate_checks_before_original_node(monkeypatch):
    import src.run_control as run_control

    calls: list[str] = []

    def fake_interrupt(payload):
        calls.append(f"interrupt:{payload['type']}:{payload['node']}")
        return {"action": "continue"}

    def node(state):
        calls.append("node")
        return {"ok": True}

    monkeypatch.setattr(run_control, "interrupt", fake_interrupt)

    wrapped = run_control.wrap_interruptible_node("guarded_node", node)
    result = await wrapped(
        {
            "thread_id": "thread-1",
            "stop_requested": True,
            "stop_reason": "user_stop",
            "stop_requested_at": "2026-01-01T00:00:00+00:00",
        }
    )

    assert result == {"ok": True}
    assert calls == ["interrupt:user_stop:guarded_node", "node"]


def test_context_engineering_missing_config_fails_fast(monkeypatch):
    import src.context_engineering.budget as budget
    import src.observability.context_usage as context_usage
    from src.context_engineering.schema import ContextConfigError

    monkeypatch.setattr(budget, "get_setting", lambda key, default=None: default)

    with pytest.raises(ContextConfigError, match="context_engineering_missing"):
        context_usage.build_context_usage_payload(
            node_name="node",
            llm_node="llm",
            provider="provider",
            model="model",
            messages=[],
        )


def test_context_engineering_non_strict_unknown_model_returns_error_event(monkeypatch):
    import src.context_engineering.budget as budget
    import src.observability.context_usage as context_usage

    def fake_get_setting(key, default=None):
        if key == "context_engineering":
            return {
                "enabled": True,
                "strict": False,
                "model_limits": {"known-model": 1000},
            }
        return default

    monkeypatch.setattr(budget, "get_setting", fake_get_setting)

    stage, payload = context_usage.build_context_usage_payload(
        node_name="node",
        llm_node="llm",
        provider="provider",
        model="unknown-model",
        messages=[],
    )

    assert stage == "context_usage_error"
    assert payload["reason"] == "model_window_unknown"


@pytest.mark.anyio
async def test_status_returns_legacy_checkpoint_without_fake_resume():
    from app import get_thread_status_payload

    graph = MagicMock()
    graph.aget_state = AsyncMock(
        return_value=_snapshot(values={"messages": ["existing legacy state"]}),
    )

    status = await get_thread_status_payload(graph, "thread-1")

    assert status.schema_version == "legacy"
    assert status.run_status == "idle"
    assert status.resume_available is False
    assert "run_status" in status.missing_run_control_fields
    assert status.request_context_window["last_event_count"] == 0
    assert status.thread_context_window["context_usage_history_count"] == 0
    assert status.thread_context_window["artifact_count"] == 0


@pytest.mark.anyio
async def test_status_missing_checkpoint_is_404():
    from app import get_thread_status_payload

    graph = MagicMock()
    graph.aget_state = AsyncMock(return_value=_snapshot(values={}))

    with pytest.raises(Exception) as exc_info:
        await get_thread_status_payload(graph, "missing-thread")

    assert getattr(exc_info.value, "status_code", None) == 404


@pytest.mark.anyio
async def test_status_prefers_active_run_without_checkpoint_read():
    from app import get_thread_status_payload
    from src.run_control import finish_active_run, start_active_run

    graph = MagicMock()
    graph.aget_state = AsyncMock(side_effect=AssertionError("checkpoint not needed"))
    start_active_run(
        "active-thread",
        {
            "schema_version": "run_control_v1",
            "run_status": "running",
            "current_node": "review_doc_agent",
            "request_context_window": {
                "current_request_id": "req-1",
                "current_node": "review_doc_agent",
                "last_event_count": 2,
            },
            "thread_context_window": {
                "context_usage_history_count": 0,
                "artifact_count": 0,
                "conversation_summary_present": False,
                "last_context_policy_by_node_keys": ["review_doc_agent"],
                "last_provider_supply_by_node_keys": [],
                "last_context_selection_by_node_keys": [],
                "last_context_applied_by_node_keys": [],
                "last_resource_subnodes_count": 0,
            },
        },
    )

    try:
        status = await get_thread_status_payload(graph, "active-thread")
    finally:
        finish_active_run("active-thread")

    assert status.run_status == "running"
    assert status.current_node == "review_doc_agent"
    assert status.request_context_window["last_event_count"] == 2
    graph.aget_state.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize("run_status", ["completed", "failed", "stopped"])
async def test_status_preserves_terminal_checkpoint_status(run_status):
    from app import get_thread_status_payload

    graph = MagicMock()
    graph.aget_state = AsyncMock(
        return_value=_snapshot(
            values={
                "schema_version": "run_control_v1",
                "run_status": run_status,
                "stop_requested": False,
                "stop_reason": "",
                "current_node": "",
                "last_completed_node": "resource_bundle_output",
                "resume_available": False,
                "stopped_at": "",
                "pending_interrupt_type": "",
                "context_usage": {},
                "context_usage_history": [{"node": "review_doc_agent"}],
                "request_context_window": {
                    "current_request_id": "req-1",
                    "current_node": "",
                    "last_event_count": 3,
                },
                "resource_artifacts_by_type": {"review_doc": {"ok": True}},
            },
        )
    )

    status = await get_thread_status_payload(graph, f"thread-{run_status}")

    assert status.run_status == run_status
    assert status.thread_context_window["context_usage_history_count"] == 1
    assert status.thread_context_window["artifact_count"] == 1


@pytest.mark.anyio
async def test_stream_initializes_checkpoint_before_thread_id():
    from app import generate_sse
    from src.run_control import finish_active_run, get_active_run

    graph = MagicMock()
    graph.astream_events = MagicMock(return_value=AsyncIteratorMock([]))
    graph.aupdate_state = AsyncMock()
    graph.aget_state = AsyncMock(return_value=_snapshot(values={}))

    stream = generate_sse("q", graph, thread_id="init-thread")
    active_snapshot = None
    try:
        first = await stream.__anext__()
        first_payload = json.loads(first.removeprefix("data: ").strip())
        active_snapshot = get_active_run("init-thread")
    finally:
        await stream.aclose()
        finish_active_run("init-thread")

    assert first_payload == {"type": "thread_id", "thread_id": "init-thread"}
    first_update = graph.aupdate_state.await_args_list[0]
    assert first_update.args[0] == {"configurable": {"thread_id": "init-thread"}}
    assert first_update.args[1]["run_status"] == "running"
    assert first_update.args[1]["request_context_window"]["current_request_id"]
    assert first_update.kwargs == {"as_node": "supervisor"}
    assert active_snapshot is not None
    assert active_snapshot["run_status"] == "running"


@pytest.mark.anyio
async def test_stream_checkpoint_init_failure_does_not_emit_thread_id():
    from app import generate_sse

    graph = MagicMock()
    graph.astream_events = MagicMock(return_value=AsyncIteratorMock([]))
    graph.aupdate_state = AsyncMock(side_effect=RuntimeError("checkpoint down"))
    graph.aget_state = AsyncMock(return_value=_snapshot(values={}))

    collected = []
    async for sse in generate_sse("q", graph, thread_id="broken-thread"):
        collected.append(sse)

    payloads = _payloads(collected)
    assert payloads == [
        {
            "type": "error",
            "message": "thread_checkpoint_initialization_failed",
            "recoverable": False,
        }
    ]
    graph.astream_events.assert_not_called()


@pytest.mark.anyio
async def test_safe_update_thread_state_uses_registered_node_when_unspecified():
    from app import safe_update_thread_state

    graph = MagicMock()
    graph.aupdate_state = AsyncMock()

    await safe_update_thread_state(
        graph,
        {"configurable": {"thread_id": "thread-1"}},
        {"run_status": "failed"},
        state={},
    )

    graph.aupdate_state.assert_awaited_once_with(
        {"configurable": {"thread_id": "thread-1"}},
        {"run_status": "failed"},
        as_node="supervisor",
    )


@pytest.mark.anyio
async def test_dev_memory_clear_uses_registered_supervisor_writer(monkeypatch):
    import app as app_module

    graph = MagicMock()
    graph.aupdate_state = AsyncMock()
    monkeypatch.setattr(app_module, "_dev_memory_clear_enabled", lambda: True)

    result = await app_module.clear_persistent_memory_for_thread(graph, "thread-1")

    assert result["ok"] is True
    graph.aupdate_state.assert_awaited_once()
    assert graph.aupdate_state.await_args.kwargs == {"as_node": "supervisor"}
    assert graph.aupdate_state.await_args.args[0] == {
        "configurable": {"thread_id": "thread-1"}
    }


@pytest.mark.anyio
async def test_user_stop_interrupt_emits_stopped_without_done_or_completed():
    from app import generate_sse

    graph = MagicMock()
    graph.astream_events = MagicMock(return_value=AsyncIteratorMock([]))
    graph.aupdate_state = AsyncMock()
    graph.aget_state = AsyncMock(
        return_value=_snapshot(
            values={"schema_version": "run_control_v1", "run_status": "stopping"},
            interrupt_value={
                "type": "user_stop",
                "node": "study_plan_agent",
                "reason": "user_stop",
            },
            next_nodes=("study_plan_agent",),
        ),
    )

    collected = []
    async for sse in generate_sse("q", graph, thread_id="thread-1"):
        collected.append(sse)

    payloads = _payloads(collected)
    run_statuses = [
        payload for payload in payloads if payload.get("type") == "run_status"
    ]
    assert run_statuses[-1]["run_status"] == "stopped"
    assert not [payload for payload in payloads if payload.get("type") == "done"]
    assert not [
        payload
        for payload in payloads
        if payload.get("type") == "run_status"
        and payload.get("run_status") == "completed"
    ]


@pytest.mark.anyio
async def test_continue_requires_pending_user_stop_interrupt():
    from app import generate_continue_sse

    graph = MagicMock()
    graph.astream_events = MagicMock(return_value=AsyncIteratorMock([]))
    graph.aupdate_state = AsyncMock()
    graph.aget_state = AsyncMock(
        return_value=_snapshot(
            values={"schema_version": "run_control_v1"},
            interrupt_value={"type": "memory_confirmation", "question": "Use memory?"},
            next_nodes=("memory_use_decider",),
        ),
    )

    collected = []
    async for sse in generate_continue_sse(graph, "thread-1"):
        collected.append(sse)

    payloads = _payloads(collected)
    assert payloads == [
        {
            "type": "run_status",
            "run_status": "not_resumable",
            "thread_id": "thread-1",
            "resume_available": False,
            "pending_interrupt_type": "memory_confirmation",
            "message": "pending HIL interrupt must be resumed with /resume",
        }
    ]
    graph.astream_events.assert_not_called()


@pytest.mark.anyio
async def test_continue_user_stop_clears_stop_before_command_resume():
    from app import generate_continue_sse

    final_snapshot = _snapshot(
        values={
            "schema_version": "run_control_v1",
            "run_status": "running",
            "context_usage_history": [],
        },
    )
    graph = MagicMock()
    graph.astream_events = MagicMock(return_value=AsyncIteratorMock([]))
    graph.aupdate_state = AsyncMock()
    graph.aget_state = AsyncMock(
        side_effect=[
            _snapshot(
                values={"schema_version": "run_control_v1", "stop_requested": True},
                interrupt_value={"type": "user_stop", "node": "study_plan_agent"},
                next_nodes=("study_plan_agent",),
            ),
            final_snapshot,
            final_snapshot,
        ],
    )

    collected = []
    async for sse in generate_continue_sse(graph, "thread-1"):
        collected.append(sse)

    first_update = graph.aupdate_state.await_args_list[0].args[1]
    assert first_update["stop_requested"] is False
    assert first_update["run_status"] == "continuing"
    payloads = _payloads(collected)
    assert payloads[0]["run_status"] == "continuing"
    assert graph.astream_events.called


@pytest.mark.anyio
async def test_context_usage_trace_becomes_sse_and_bounded_state():
    from app import generate_sse

    async def events():
        emit_a3_trace(
            logging.getLogger("test_context_usage"),
            "context_usage",
            {
                "node_name": "study_plan_agent",
                "llm_node": "study_plan",
                "provider": "deepseek_official",
                "model": "synthetic-model",
                "input_estimated_tokens": 100,
                "reserved_output_tokens": 20,
                "used_tokens": 120,
                "max_context_tokens": 64000,
                "available_tokens": 63880,
                "used_ratio": 0.001,
                "warning_level": "ok",
                "estimated": True,
                "tokenizer_mode": "estimated_mixed",
                "message_count": 2,
                "breakdown": {
                    "input_estimated_tokens": 100,
                    "reserved_output_tokens": 20,
                },
            },
            state={"request_id": "r1", "thread_id": "thread-1"},
        )
        yield {
            "event": "on_chain_start",
            "name": "study_plan_agent",
            "metadata": {"langgraph_node": "study_plan_agent"},
            "data": {"input": {}},
        }

    graph = MagicMock()
    graph.astream_events = MagicMock(return_value=events())
    graph.aget_state = AsyncMock(
        return_value=_snapshot(values={"schema_version": "run_control_v1"})
    )
    graph.aupdate_state = AsyncMock()

    collected = []
    async for sse in generate_sse("q", graph, thread_id="thread-1"):
        collected.append(sse)

    payloads = _payloads(collected)
    context_events = [
        payload for payload in payloads if payload.get("type") == "context_usage"
    ]
    assert len(context_events) == 1
    assert context_events[0]["used_tokens"] == 120
    state_updates = [call.args[1] for call in graph.aupdate_state.await_args_list]
    context_updates = [
        update for update in state_updates if "context_usage_history" in update
    ]
    assert context_updates
    assert len(context_updates[-1]["context_usage_history"]) <= 30


@pytest.mark.anyio
async def test_context_usage_error_trace_becomes_warning_sse():
    from app import generate_sse

    async def events():
        emit_a3_trace(
            logging.getLogger("test_context_usage_error"),
            "context_usage_error",
            {
                "node_name": "study_plan_agent",
                "llm_node": "study_plan",
                "provider": "deepseek_official",
                "model": "unknown-model",
                "reason": "model_window_unknown",
                "warning": "context usage telemetry unavailable; model context window is unknown",
            },
            state={"request_id": "r1", "thread_id": "thread-1"},
        )
        yield {
            "event": "on_chain_start",
            "name": "study_plan_agent",
            "metadata": {"langgraph_node": "study_plan_agent"},
            "data": {"input": {}},
        }

    graph = MagicMock()
    graph.astream_events = MagicMock(return_value=events())
    graph.aget_state = AsyncMock(
        return_value=_snapshot(values={"schema_version": "run_control_v1"})
    )
    graph.aupdate_state = AsyncMock()

    collected = []
    async for sse in generate_sse("q", graph, thread_id="thread-1"):
        collected.append(sse)

    payloads = _payloads(collected)
    error_events = [
        payload for payload in payloads if payload.get("type") == "context_usage_error"
    ]
    assert error_events == [
        {
            "type": "context_usage_error",
            "node": "study_plan_agent",
            "llm_node": "study_plan",
            "provider": "deepseek_official",
            "model": "unknown-model",
            "reason": "model_window_unknown",
            "warning": "context usage telemetry unavailable; model context window is unknown",
        }
    ]
