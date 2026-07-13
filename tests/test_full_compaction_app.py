from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app as app_module
from src.context_engineering.compaction import (
    ConversationSummaryV2,
    ProviderBoundUsageV1,
)


class FakeGraph:
    def __init__(self, values: dict | None = None) -> None:
        self.values = dict(values or {})
        self.updates: list[tuple[dict, dict, str | None]] = []
        self.astream_events_called = False

    async def aupdate_state(self, config, values, as_node=None):
        self.updates.append((config, values, as_node))
        self.values.update(values)

    async def aget_state(self, _config):
        return SimpleNamespace(values=dict(self.values), tasks=(), next=())

    async def astream_events(self, *_args, **_kwargs):
        self.astream_events_called = True
        if False:
            yield None


def _provider_usage() -> dict:
    return ProviderBoundUsageV1(
        dispatch_id="dispatch-previous",
        call_id="call-previous",
        request_id="request-previous",
        thread_id="thread-1",
        attempt=1,
        provider="deepseek_official",
        model="deepseek-v4-pro",
        input_tokens=900_000,
        tokenizer_mode="estimated_mixed",
        estimated=True,
        trigger_eligible=True,
        dispatched_at=datetime.now(timezone.utc),
    ).model_dump(mode="json")


@pytest.mark.anyio
async def test_full_compaction_commits_boundary_summary_result_and_ledger_atomically(
    monkeypatch,
):
    history = [
        HumanMessage(content="old query " + "context " * 3000),
        AIMessage(content="old answer " + "detail " * 3000),
        HumanMessage(content="second query"),
        AIMessage(content="second answer"),
        HumanMessage(content="third query"),
        AIMessage(content="third answer"),
    ]
    snapshot_values = {
        "messages": history,
        "last_provider_dispatch": _provider_usage(),
    }
    state_input = {
        "messages": [HumanMessage(content="current query")],
        "request_id": "request-current",
        "thread_id": "thread-1",
        "session_id": "thread-1",
    }
    graph = FakeGraph(snapshot_values)

    async def fake_summary(*, boundary, **_kwargs):
        return ConversationSummaryV2(
            schema_version=2,
            boundary_id=boundary.boundary_id,
            summary="The learner completed the first discussion round.",
            learning_goals=[],
            preferences=[],
            facts=[],
            decisions=[],
            unfinished_tasks=[],
            evidence_ids=[],
            artifact_ids=[],
        )

    monkeypatch.setattr(
        app_module,
        "invoke_conversation_compaction",
        fake_summary,
    )

    updated_values, result = await app_module._prepare_full_compaction_for_new_request(
        graph,
        {"configurable": {"thread_id": "thread-1"}},
        thread_id="thread-1",
        request_id="request-current",
        snapshot_values=snapshot_values,
        state_input=state_input,
    )

    assert result is not None
    assert result["model_view_after_tokens"] < result["model_view_before_tokens"]
    assert len(graph.updates) == 1
    atomic_update = graph.updates[0][1]
    assert set(
        [
            "conversation_summary",
            "conversation_summary_v2",
            "compact_boundary",
            "compaction_result",
            "session_context_memory_ledger",
            "thread_context_window_v3",
        ]
    ).issubset(atomic_update)
    assert "messages" not in atomic_update
    assert atomic_update["thread_context_window_v3"]["compaction"]["status"] == (
        "compacted"
    )
    assert (
        atomic_update["session_context_memory_ledger"]["lifetime_injected_tokens"] == 0
    )
    assert state_input["compact_boundary"] == atomic_update["compact_boundary"]
    assert updated_values["compaction_result"] == result


@pytest.mark.anyio
async def test_actual_provider_dispatch_is_persisted_for_next_request():
    graph = FakeGraph()
    state_context: dict = {}
    event = {
        "stage": "provider_dispatch.started",
        "dispatch_id": "dispatch-1",
        "call_id": "call-1",
        "request_id": "request-1",
        "thread_id": "thread-1",
        "attempt": 1,
        "provider": "deepseek_official",
        "model": "deepseek-v4-pro",
        "input_tokens": 1234,
        "tokenizer_mode": "estimated_mixed",
        "estimated": True,
        "trigger_eligible": True,
        "dispatched_at": datetime.now(timezone.utc).isoformat(),
    }

    usage = await app_module._update_last_provider_dispatch_from_trace(
        graph,
        {"configurable": {"thread_id": "thread-1"}},
        thread_id="thread-1",
        event=event,
        state_context=state_context,
    )

    assert usage.input_tokens == 1234
    assert state_context["last_provider_dispatch"]["dispatch_id"] == "dispatch-1"
    assert graph.updates[0][1] == {
        "last_provider_dispatch": state_context["last_provider_dispatch"]
    }


@pytest.mark.anyio
async def test_compaction_failure_blocks_graph_before_execution(monkeypatch):
    graph = FakeGraph()

    async def fail_compaction(*_args, **_kwargs):
        raise RuntimeError("invalid summary body must not escape")

    monkeypatch.setattr(
        app_module,
        "_prepare_full_compaction_for_new_request",
        fail_compaction,
    )

    frames = [
        frame
        async for frame in app_module._generate_sse_impl(
            "query",
            graph,
            "thread-1",
            request_id="request-1",
        )
    ]

    payloads = [
        json.loads(
            "\n".join(
                line[5:].lstrip()
                for line in frame.strip().splitlines()
                if line.startswith("data:")
            )
        )
        for frame in frames
    ]
    assert payloads == [
        {
            "type": "error",
            "error_type": "full_compaction_failed",
            "message": "full_compaction_failed",
            "recoverable": False,
            "thread_id": "thread-1",
        }
    ]
    assert graph.astream_events_called is False
    assert any(
        update[1].get("run_status") == app_module.RUN_STATUS_ERROR
        for update in graph.updates
    )
