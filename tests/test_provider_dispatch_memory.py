from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from src.context_engineering.input_manifest import build_llm_input_manifest
from src.context_engineering.schema import ContextItem
from src.graph import llm as llm_module
from src.graph.llm import invoke_with_provider_transport_retry
from src.observability.a3_trace import (
    reset_trace_event_sink,
    set_trace_event_sink,
)


def _memory_item() -> ContextItem:
    return ContextItem(
        id="memory:goal",
        source_type="memory",
        content="private memory body",
        token_estimate=6,
        estimated=True,
        tokenizer_mode="estimated_mixed_v1",
        priority=80,
        scope="session",
        lifetime="session",
        compressible=True,
        can_drop=True,
        disclosure_level="summary",
    )


def _manifest(*, applied: bool = True):
    return build_llm_input_manifest(
        node_name="qa_agent",
        llm_node="qa",
        provider="deepseek_official",
        model="deepseek-v4-pro",
        messages=[
            SystemMessage(content="<INJECTED_CONTEXT>safe summary</INJECTED_CONTEXT>"),
            HumanMessage(content="question"),
        ],
        state={"request_id": "request-1", "thread_id": "thread-1"},
        call_purpose="plain_llm",
        context_apply_applied=applied,
        context_apply_status="applied" if applied else "observe_only",
        provider_bound_messages_mutated=applied,
        context_items=[_memory_item()],
    )


@pytest.mark.anyio
async def test_real_transport_retries_emit_one_injection_record_per_attempt(
    monkeypatch,
) -> None:
    monkeypatch.setattr(llm_module, "_provider_transport_max_retries", lambda _node: 2)
    monkeypatch.setattr(
        llm_module, "_provider_transport_delay_seconds", lambda _attempt: 0
    )
    monkeypatch.setattr(llm_module.asyncio, "sleep", AsyncMock())
    calls = 0
    events: list[dict] = []
    token = set_trace_event_sink(events)
    try:

        async def operation():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise TimeoutError("provider timed out")
            return "ok"

        result, retry_count = await invoke_with_provider_transport_retry(
            operation,
            node_name="qa_agent",
            llm_node="qa",
            provider="deepseek_official",
            model="deepseek-v4-pro",
            llm_input_manifest=_manifest(),
            state={"request_id": "request-1", "thread_id": "thread-1"},
        )
    finally:
        reset_trace_event_sink(token)

    injections = [
        event for event in events if event["stage"] == "context_injection.dispatched"
    ]
    dispatches = [
        event for event in events if event["stage"] == "provider_dispatch.started"
    ]
    assert result == "ok"
    assert retry_count == 2
    assert [event["attempt"] for event in dispatches] == [1, 2, 3]
    assert [event["attempt"] for event in injections] == [1, 2, 3]
    assert len({event["record_id"] for event in injections}) == 3
    assert all(
        event["item"]["logical_item_id"] == "memory:goal" for event in injections
    )
    assert "private memory body" not in repr(events)


@pytest.mark.anyio
async def test_observe_only_manifest_emits_no_injection_record(monkeypatch) -> None:
    monkeypatch.setattr(llm_module, "_provider_transport_max_retries", lambda _node: 0)
    events: list[dict] = []
    token = set_trace_event_sink(events)
    try:
        await invoke_with_provider_transport_retry(
            AsyncMock(return_value="ok"),
            node_name="qa_agent",
            llm_node="qa",
            provider="deepseek_official",
            model="deepseek-v4-pro",
            llm_input_manifest=_manifest(applied=False),
            state={"request_id": "request-1", "thread_id": "thread-1"},
        )
    finally:
        reset_trace_event_sink(token)

    assert not any(event["stage"] == "context_injection.dispatched" for event in events)


@pytest.mark.anyio
async def test_compaction_summary_dispatch_is_not_a_future_trigger(monkeypatch) -> None:
    monkeypatch.setattr(llm_module, "_provider_transport_max_retries", lambda _node: 0)
    events: list[dict] = []
    token = set_trace_event_sink(events)
    try:
        await invoke_with_provider_transport_retry(
            AsyncMock(return_value="ok"),
            node_name="conversation_compactor",
            llm_node="conversation_compactor",
            provider="deepseek_official",
            model="deepseek-v4-pro",
            llm_input_manifest=_manifest(applied=False),
            state={"request_id": "request-1", "thread_id": "thread-1"},
        )
    finally:
        reset_trace_event_sink(token)

    dispatch = next(
        event for event in events if event["stage"] == "provider_dispatch.started"
    )
    assert dispatch["trigger_eligible"] is False
