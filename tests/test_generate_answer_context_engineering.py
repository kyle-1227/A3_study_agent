"""Replacement gates for generate_answer memory injection through CE only."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from src.context_engineering.packing.node_policy import resolve_context_policy
from src.context_engineering.packing.source_policy import (
    filter_context_items_by_source_policy,
)
from src.context_engineering.influence_runtime import (
    begin_influence_capture,
    end_influence_capture,
)
from src.context_engineering.session_memory import (
    ContextInjectionRecordV1,
    new_session_context_memory_ledger,
    record_context_injection,
)
from src.context_engineering.providers.base import ProviderContext
from src.context_engineering.providers.memory_provider import MemoryContextProvider
from src.observability.a3_trace import reset_trace_event_sink, set_trace_event_sink


def _memory_state() -> dict:
    return {
        "request_id": "request-1",
        "thread_id": "thread-1",
        "user_id": "user-1",
        "memory_use_policy": "use",
        "conversation_summary": "The learner is revisiting binary search.",
        "episodic_memory_results": [
            {
                "memory_id": "episodic-1",
                "memory_type": "episodic",
                "content": "The learner missed the right-boundary case.",
                "score": 0.82,
            }
        ],
        "semantic_memory_results": [
            {
                "summary_id": "semantic-1",
                "memory_type": "semantic",
                "content": "The learner prefers worked examples.",
                "score": 0.76,
            }
        ],
        "profile_summary": "Second-year learner who prefers concise explanations.",
    }


def _message_text(message: object) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def test_generate_answer_policy_is_active_and_thread_scoped():
    resolved = resolve_context_policy(
        node_name="generate_answer",
        llm_node="academic",
        state=_memory_state(),
    )

    assert resolved.mode == "active"
    assert resolved.injection_policy.injectable_sources == (
        "rules",
        "memory",
        "profile",
    )
    assert resolved.injection_policy.required_sources == ("rules",)
    assert resolved.injection_policy.optional_sources == ("memory", "profile")
    assert resolved.source_policies["memory"].max_items == 6
    assert resolved.source_policies["memory"].max_tokens == 1600
    assert resolved.source_policies["memory"].min_relevance_score is None
    assert resolved.source_policies["memory"].require_user_match is False
    assert resolved.source_policies["memory"].require_thread_match is True


def test_long_memory_buckets_fit_policy_without_starving_summary_or_memory_type():
    state = {
        "request_id": "request-1",
        "thread_id": "thread-1",
        "user_id": "user-1",
        "memory_use_policy": "use",
        "conversation_summary": "摘要" * 3000,
        "episodic_memory_results": [
            {"memory_id": f"e{index}", "content": "经历" * 1000, "score": 0.8}
            for index in range(8)
        ],
        "semantic_memory_results": [
            {"summary_id": f"s{index}", "content": "偏好" * 1000, "score": 0.8}
            for index in range(4)
        ],
    }
    context = ProviderContext(
        node_name="generate_answer",
        llm_node="academic",
        user_query="question",
        current_user_message_index=None,
        state=state,
        messages=[],
        request_id="request-1",
        thread_id="thread-1",
        max_items_per_provider=6,
        max_content_chars_per_item=4000,
    )
    items = MemoryContextProvider().collect(context)
    resolved = resolve_context_policy(
        node_name="generate_answer",
        llm_node="academic",
        state=state,
    )

    filtered = filter_context_items_by_source_policy(
        items,
        injectable_sources=("memory",),
        exclude_message_source=True,
        source_policies={"memory": resolved.source_policies["memory"]},
        state=state,
        policy_mode="strict",
        target_node_name="generate_answer",
    )

    retained_buckets = {item.metadata["source_bucket"] for item in filtered.kept_items}
    assert "conversation_summary" in retained_buckets
    assert "episodic_memory_results" in retained_buckets
    assert "semantic_memory_results" in retained_buckets
    assert sum(item.token_estimate for item in filtered.kept_items) <= 1600


@pytest.mark.anyio
async def test_provider_dispatch_contains_ce_memory_profile_rules_and_updates_ledger(
    monkeypatch,
):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    mock_llm = MagicMock()
    mock_llm.model_name = "deepseek-v4-pro"
    mock_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content="CE answer"))
    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    monkeypatch.setattr(
        llm_module,
        "get_llm_call_max_retries",
        lambda node_name=None, default=0: 0,
    )

    events: list[dict] = []
    trace_token = set_trace_event_sink(events)
    influence_token = begin_influence_capture()
    try:
        result = await invoke_plain_llm_fail_fast(
            node_name="generate_answer",
            llm_node="academic",
            messages=[
                SystemMessage(content="Answer accurately."),
                HumanMessage(content="Explain binary search boundaries."),
            ],
            state=_memory_state(),
        )
        influences = end_influence_capture(influence_token)
        influence_token = None
    finally:
        if influence_token is not None:
            end_influence_capture(influence_token)
        reset_trace_event_sink(trace_token)

    assert result == "CE answer"
    provider_messages = mock_llm.ainvoke.await_args.args[0]
    provider_text = "\n".join(_message_text(message) for message in provider_messages)
    assert "<INJECTED_CONTEXT>" in provider_text
    assert "The learner is revisiting binary search." in provider_text
    assert "The learner missed the right-boundary case." in provider_text
    assert "The learner prefers worked examples." in provider_text
    assert "Second-year learner" in provider_text

    dispatched = [
        event
        for event in events
        if event.get("stage") == "context_injection.dispatched"
    ]
    source_types = [event["item"]["source_type"] for event in dispatched]
    assert set(source_types) == {"memory", "profile", "rules"}
    assert "right-boundary case" not in repr(dispatched)
    provider_influence = next(
        entry
        for entry in influences
        if entry["kind"] == "provider_bound_messages_metadata"
    )
    assert provider_influence["metadata"]["context_injection_source_counts"] == {
        "memory": 3,
        "profile": 1,
        "rules": 2,
    }

    ledger = new_session_context_memory_ledger("thread-1")
    record_keys = (
        "schema_version",
        "record_id",
        "dispatch_id",
        "request_id",
        "call_id",
        "attempt",
        "manifest_id",
        "thread_id",
        "item",
        "dispatched_at",
    )
    for event in dispatched:
        record = ContextInjectionRecordV1.model_validate(
            {key: event.get(key) for key in record_keys}
        )
        ledger = record_context_injection(ledger, record)

    assert ledger.request_count == 1
    assert ledger.source_stats["memory"].injection_count >= 3
    assert ledger.source_stats["profile"].injection_count == 1
    assert ledger.source_stats["rules"].injection_count >= 1
    assert ledger.retained_memory_tokens > 0


@pytest.mark.anyio
async def test_cross_thread_memory_is_filtered_but_required_rules_keep_call_safe(
    monkeypatch,
):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    state = _memory_state()
    state["conversation_summary"] = ""
    state["semantic_memory_results"] = []
    state["episodic_memory_results"] = [
        {
            "memory_id": "wrong-thread",
            "content": "This content must not reach the provider.",
            "thread_id": "thread-other",
            "score": 0.9,
        }
    ]
    state.pop("profile_summary")

    mock_llm = MagicMock()
    mock_llm.model_name = "deepseek-v4-pro"
    mock_llm.ainvoke = AsyncMock(
        return_value=SimpleNamespace(content="Rules-only answer")
    )
    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    monkeypatch.setattr(
        llm_module,
        "get_llm_call_max_retries",
        lambda node_name=None, default=0: 0,
    )

    events: list[dict] = []
    token = set_trace_event_sink(events)
    try:
        result = await invoke_plain_llm_fail_fast(
            node_name="generate_answer",
            llm_node="academic",
            messages=[HumanMessage(content="Current question")],
            state=state,
        )
    finally:
        reset_trace_event_sink(token)

    assert result == "Rules-only answer"
    provider_messages = mock_llm.ainvoke.await_args.args[0]
    provider_text = "\n".join(_message_text(message) for message in provider_messages)
    assert "This content must not reach the provider." not in provider_text
    dispatched_sources = {
        event["item"]["source_type"]
        for event in events
        if event.get("stage") == "context_injection.dispatched"
    }
    assert dispatched_sources == {"rules"}
    source_filter_events = [
        event for event in events if event.get("stage") == "context_source_filter"
    ]
    assert source_filter_events
    assert source_filter_events[-1]["source_drop_reasons"]["thread_mismatch"] == 1


@pytest.mark.anyio
async def test_explicit_ignore_blocks_same_thread_memory_in_production_path(
    monkeypatch,
):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    state = _memory_state()
    state["memory_use_policy"] = "ignore"
    state.pop("profile_summary")

    mock_llm = MagicMock()
    mock_llm.model_name = "deepseek-v4-pro"
    mock_llm.ainvoke = AsyncMock(
        return_value=SimpleNamespace(content="Rules-only answer")
    )
    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    monkeypatch.setattr(
        llm_module,
        "get_llm_call_max_retries",
        lambda node_name=None, default=0: 0,
    )

    events: list[dict] = []
    token = set_trace_event_sink(events)
    try:
        result = await invoke_plain_llm_fail_fast(
            node_name="generate_answer",
            llm_node="academic",
            messages=[HumanMessage(content="Answer only the current question.")],
            state=state,
        )
    finally:
        reset_trace_event_sink(token)

    assert result == "Rules-only answer"
    provider_messages = mock_llm.ainvoke.await_args.args[0]
    provider_text = "\n".join(_message_text(message) for message in provider_messages)
    assert "The learner is revisiting binary search." not in provider_text
    assert "right-boundary case" not in provider_text
    assert "worked examples" not in provider_text
    dispatched_sources = {
        event["item"]["source_type"]
        for event in events
        if event.get("stage") == "context_injection.dispatched"
    }
    assert dispatched_sources == {"rules"}


@pytest.mark.anyio
async def test_generate_answer_returns_provider_text_without_legacy_memory_footer(
    monkeypatch,
):
    from src.graph import academic as academic_module

    monkeypatch.setattr(
        academic_module,
        "invoke_plain_llm_fail_fast",
        AsyncMock(return_value="Authoritative provider answer"),
    )
    monkeypatch.setattr(academic_module, "_record_trace", AsyncMock())

    result = await academic_module.generate_answer(
        {
            "messages": [HumanMessage(content="Explain binary search.")],
            "context": [],
            "thread_id": "thread-1",
            "request_id": "request-1",
            "memory_use_policy": "use",
            "episodic_memory_results": [
                {
                    "memory_id": "m1",
                    "content": "Prior weak point.",
                    "score": 0.9,
                }
            ],
        }
    )

    assert result["messages"][0].content == "Authoritative provider answer"
    invoke_messages = academic_module.invoke_plain_llm_fail_fast.await_args.kwargs[
        "messages"
    ]
    assert "Prior weak point." not in "\n".join(
        _message_text(message) for message in invoke_messages
    )
