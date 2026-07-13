from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from src.context_engineering.schema import ContextItem
from src.context_engineering.session_memory import (
    ContextInjectionRecordV1,
    apply_context_memory_compaction,
    build_injection_descriptor,
    clear_session_context_memory,
    new_session_context_memory_ledger,
    record_context_injection,
)
from src.graph.state import (
    SESSION_CONTEXT_MEMORY_LEDGER_CLEAR,
    initial_request_reset_transient_state,
    session_context_memory_ledger_reducer,
)


def _item(item_id: str, content: str, *, tokens: int = 10) -> ContextItem:
    return ContextItem(
        id=item_id,
        source_type="memory",
        title="",
        content=content,
        token_estimate=tokens,
        estimated=True,
        tokenizer_mode="estimated_mixed_v1",
        priority=50,
        scope="session",
        lifetime="session",
        compressible=True,
        can_drop=True,
        disclosure_level="summary",
    )


def _record(item: ContextItem, *, record_id: str, request_id: str = "request-1"):
    descriptor = build_injection_descriptor(item)
    assert descriptor is not None
    return ContextInjectionRecordV1(
        record_id=record_id,
        dispatch_id=f"dispatch-{record_id}",
        request_id=request_id,
        call_id="call-1",
        attempt=1,
        manifest_id="manifest-1",
        thread_id="thread-1",
        item=descriptor,
        dispatched_at=datetime(2026, 7, 13, tzinfo=timezone.utc),
    )


def test_repeat_injection_increases_lifetime_but_not_retained() -> None:
    ledger = new_session_context_memory_ledger("thread-1")
    ledger = record_context_injection(
        ledger, _record(_item("memory:a", "same", tokens=7), record_id="record-1")
    )
    ledger = record_context_injection(
        ledger, _record(_item("memory:b", "same", tokens=7), record_id="record-2")
    )

    assert ledger.retained_memory_tokens == 7
    assert ledger.lifetime_injected_tokens == 14
    assert ledger.lifetime_unique_tokens == 7
    assert ledger.injection_count == 2
    assert ledger.repeat_injection_count == 1
    assert ledger.memory_summary.active_item_count == 2
    assert ledger.memory_summary.active_unique_content_count == 1


def test_new_logical_item_version_replaces_retained_content() -> None:
    ledger = new_session_context_memory_ledger("thread-1")
    ledger = record_context_injection(
        ledger, _record(_item("memory:a", "old", tokens=5), record_id="record-1")
    )
    ledger = record_context_injection(
        ledger, _record(_item("memory:a", "new", tokens=9), record_id="record-2")
    )

    assert ledger.retained_memory_tokens == 9
    assert ledger.lifetime_injected_tokens == 14
    assert ledger.lifetime_unique_tokens == 14
    assert len(ledger.active_items) == 1


def test_record_replay_is_idempotent_and_request_count_is_unique() -> None:
    ledger = new_session_context_memory_ledger("thread-1")
    record = _record(_item("memory:a", "same"), record_id="record-1")
    once = record_context_injection(ledger, record)
    twice = record_context_injection(once, record)

    assert twice == once
    assert twice.request_count == 1
    assert twice.injection_count == 1


def test_descriptor_and_ledger_do_not_store_content() -> None:
    secret = "private transcript body"
    ledger = new_session_context_memory_ledger("thread-1")
    ledger = record_context_injection(
        ledger, _record(_item("memory:a", secret), record_id="record-1")
    )

    assert secret not in ledger.model_dump_json()


def test_compaction_reduces_retained_without_changing_lifetime() -> None:
    ledger = new_session_context_memory_ledger("thread-1")
    ledger = record_context_injection(
        ledger, _record(_item("memory:a", "one", tokens=5), record_id="record-1")
    )
    ledger = record_context_injection(
        ledger, _record(_item("memory:b", "two", tokens=8), record_id="record-2")
    )
    compacted = apply_context_memory_compaction(
        ledger,
        boundary_id="boundary-1",
        retained_logical_item_ids=["memory:b"],
        compacted_at=datetime(2026, 7, 13, 1, tzinfo=timezone.utc),
        before_tokens=13,
        after_tokens=8,
    )

    assert compacted.retained_memory_tokens == 8
    assert compacted.lifetime_injected_tokens == 13
    assert compacted.lifetime_unique_tokens == 13
    assert compacted.compaction.status == "compacted"


def test_compaction_cannot_silently_clear_all_memory() -> None:
    ledger = record_context_injection(
        new_session_context_memory_ledger("thread-1"),
        _record(_item("memory:a", "one"), record_id="record-1"),
    )
    with pytest.raises(ValueError, match="cannot clear all"):
        apply_context_memory_compaction(
            ledger,
            boundary_id="boundary-1",
            retained_logical_item_ids=[],
            compacted_at=datetime(2026, 7, 13, 1, tzinfo=timezone.utc),
            before_tokens=10,
            after_tokens=0,
        )


def test_only_explicit_clear_resets_lifetime_totals() -> None:
    ledger = record_context_injection(
        new_session_context_memory_ledger("thread-1"),
        _record(_item("memory:a", "one"), record_id="record-1"),
    )
    cleared = clear_session_context_memory(ledger)

    assert cleared.thread_id == ledger.thread_id
    assert cleared.retained_memory_tokens == 0
    assert cleared.lifetime_injected_tokens == 0
    assert cleared.injection_count == 0


def test_message_items_are_not_session_memory_descriptors() -> None:
    item = _item("message:a", "hello").model_copy(update={"source_type": "message"})
    assert build_injection_descriptor(item) is None


def test_state_reducer_persists_dispatch_idempotently_across_requests() -> None:
    record = _record(_item("memory:a", "same"), record_id="record-1")
    mutation = {
        "operation": "record_dispatch",
        "record": record.model_dump(mode="json"),
    }
    first = session_context_memory_ledger_reducer({}, mutation)
    replayed = session_context_memory_ledger_reducer(first, mutation)
    second_request = session_context_memory_ledger_reducer(
        replayed,
        {
            "operation": "record_dispatch",
            "record": _record(
                _item("memory:a", "same"),
                record_id="record-2",
                request_id="request-2",
            ).model_dump(mode="json"),
        },
    )

    assert replayed == first
    assert second_request["request_count"] == 2
    assert second_request["lifetime_injected_tokens"] == 20
    assert (
        "session_context_memory_ledger" not in initial_request_reset_transient_state()
    )


def test_state_reducer_only_clears_with_explicit_sentinel() -> None:
    record = _record(_item("memory:a", "same"), record_id="record-1")
    ledger = session_context_memory_ledger_reducer(
        {}, {"operation": "record_dispatch", "record": record.model_dump(mode="json")}
    )

    assert (
        session_context_memory_ledger_reducer(
            ledger, SESSION_CONTEXT_MEMORY_LEDGER_CLEAR
        )
        == {}
    )


@pytest.mark.anyio
async def test_dispatch_trace_persists_ledger_and_v3_together() -> None:
    from app import _update_session_context_memory_from_trace

    graph = AsyncMock()
    state_context = {"thread_id": "thread-1"}
    record = _record(_item("memory:a", "same"), record_id="record-1")
    event = {
        "stage": "context_injection.dispatched",
        **record.model_dump(mode="json"),
    }

    window = await _update_session_context_memory_from_trace(
        graph,
        {"configurable": {"thread_id": "thread-1"}},
        thread_id="thread-1",
        event=event,
        state_context=state_context,
    )

    graph.aupdate_state.assert_awaited_once()
    _config, values = graph.aupdate_state.await_args.args
    assert values["session_context_memory_ledger"]["lifetime_injected_tokens"] == 10
    assert values["thread_context_window_v3"]["retained_memory_tokens"] == 10
    assert values["thread_context_window_v3"]["updating"] is False
    assert window == values["thread_context_window_v3"]
    assert (
        state_context["session_context_memory_ledger"]
        == values["session_context_memory_ledger"]
    )
