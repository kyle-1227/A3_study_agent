"""Tests for the bounded Context Influence Ledger and capture boundary."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.context_engineering.influence import (
    INFLUENCE_ENTRY_LIMIT,
    INFLUENCE_ID_PREFIX,
    build_influence_entry,
    build_influence_update,
    influence_status_payload,
    merge_context_influence_ledger,
    stable_influence_id,
)
from src.context_engineering.influence_runtime import (
    begin_influence_capture,
    end_influence_capture,
    record_llm_input_influences,
    wrap_context_influence_node,
)
from src.graph.state import initial_request_reset_transient_state
from src.observability.a3_trace import reset_trace_event_sink, set_trace_event_sink


def _state(**overrides):
    state = {
        "request_id": "request-1",
        "thread_id": "thread-1",
        "subject": "machine_learning",
    }
    state.update(overrides)
    return state


def test_stable_influence_ids_are_deterministic_sha256_ids():
    values = {
        "request_id": "request-1",
        "thread_id": "thread-1",
        "kind": "planner_output",
        "source_node": "mindmap_planner",
        "target_stage": "downstream",
        "content_fingerprint": "abc",
        "identity_metadata": {"workflow": "mindmap", "iteration": 1},
    }

    first = stable_influence_id(**values)
    second = stable_influence_id(**values)

    assert first == second
    assert first.startswith(f"{INFLUENCE_ID_PREFIX}:")
    assert len(first.rsplit(":", maxsplit=1)[-1]) == 32


def test_ledger_merge_is_idempotent_and_bounded_by_count():
    entries = [
        build_influence_entry(
            state=_state(),
            kind="planner_output",
            source_node="mindmap_planner",
            preview=f"outline {index}",
            fingerprint_source=f"outline {index}",
            metadata={"workflow": "mindmap", "iteration": index},
        )
        for index in range(INFLUENCE_ENTRY_LIMIT + 20)
    ]
    update = build_influence_update(state=_state(), entries=entries)

    once = merge_context_influence_ledger({}, update)
    twice = merge_context_influence_ledger(once, update)

    assert 0 < len(once["entries_by_id"]) <= INFLUENCE_ENTRY_LIMIT
    assert once["ordered_ids"] == twice["ordered_ids"]
    assert twice["total_recorded"] == len(entries)
    assert len(twice["entries_by_id"]) == len(once["entries_by_id"])
    assert "context_influence_ledger_trimmed" in twice["diagnostics"]


def test_ledger_sanitizes_secrets_and_does_not_store_full_prompt():
    token = begin_influence_capture()
    try:
        record_llm_input_influences(
            node_name="mindmap_agent",
            llm_node="mindmap",
            messages=[
                {
                    "role": "user",
                    "content": "private prompt api_key=sk-abcdefghijklmnopqrstuvwxyz",
                }
            ],
            state=_state(),
            manifest={
                "manifest_id": "llm_input_manifest:v1:one",
                "provider": "provider",
                "model": "model",
                "context_apply_applied": True,
                "context_injection_items": [
                    {"source_type": "memory"},
                    {"source_type": "memory"},
                    {"source_type": "profile"},
                ],
            },
            schema_name="MindmapDraft",
            output_mode="native_json_schema_pydantic",
        )
        entries = end_influence_capture(token)
    except BaseException:
        end_influence_capture(token)
        raise

    serialized = repr(entries)
    assert len(entries) == 3
    assert "private prompt" not in serialized
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in serialized
    assert all(not entry["preview"] for entry in entries)
    provider_entry = next(
        entry
        for entry in entries
        if entry["kind"] == "provider_bound_messages_metadata"
    )
    assert provider_entry["metadata"]["context_injection_source_counts"] == {
        "memory": 2,
        "profile": 1,
    }


def test_corrupt_or_incompatible_ledger_degrades_safely():
    incompatible = merge_context_influence_ledger(
        {},
        {"schema_version": 999, "entries": [{"raw_prompt": "do not keep"}]},
    )
    corrupt_status = influence_status_payload(
        {
            "schema_version": 1,
            "entries_by_id": {"bad": "not-an-entry"},
            "ordered_ids": ["bad"],
        }
    )

    assert incompatible["entries_by_id"] == {}
    assert incompatible["diagnostics"] == [
        "context_influence_schema_version_incompatible"
    ]
    assert corrupt_status["entry_count"] == 0


def test_normal_request_reset_preserves_context_influence_ledger():
    reset = initial_request_reset_transient_state()

    assert "context_influence_ledger" not in reset


async def test_graph_node_wrapper_captures_query_and_emits_safe_trace():
    async def node(_state):
        return {"intent": "academic"}

    trace_events: list[dict] = []
    trace_token = set_trace_event_sink(trace_events)
    try:
        result = await wrap_context_influence_node("supervisor", node)(
            {
                **_state(),
                "messages": [HumanMessage(content="build a machine learning plan")],
            }
        )
    finally:
        reset_trace_event_sink(trace_token)

    update = result["context_influence_ledger"]
    assert len(update["entries"]) == 1
    assert update["entries"][0]["kind"] == "original_user_query"
    capture_event = next(
        event
        for event in trace_events
        if event.get("stage") == "context_influence.captured"
    )
    assert capture_event["entry_count"] == 1
    assert "build a machine learning plan" not in repr(capture_event)
