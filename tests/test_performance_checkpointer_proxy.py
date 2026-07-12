from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.observability.checkpointer_proxy import (
    ObservableCheckpointer,
    observe_checkpointer,
)
from src.observability.performance_runtime import (
    build_performance_report,
    performance_request_recorder,
)


@pytest.mark.anyio
async def test_checkpointer_proxy_records_database_parent_and_checkpoint_child():
    delegate = MemorySaver()
    proxy = ObservableCheckpointer(delegate)
    config = {"configurable": {"thread_id": "thread-1"}}

    with performance_request_recorder(
        request_id="request-1",
        thread_id="thread-1",
        max_spans=32,
    ) as recorder:
        assert await proxy.aget_tuple(config) is None

    report = build_performance_report(recorder)
    database = next(
        item for item in recorder.events if item.operation_type == "database"
    )
    checkpoint = next(
        item for item in recorder.events if item.operation_type == "checkpoint"
    )

    assert database.parent_span_id == recorder.root_span_id
    assert checkpoint.parent_span_id == database.span_id
    assert report.coverage.invalid_parent_count == 0
    assert report.database_total_ms >= report.checkpoint_ms
    assert report.database_non_checkpoint_ms >= 0


def test_checkpointer_proxy_is_idempotent_and_transparent_without_recorder():
    delegate = MemorySaver()
    proxy = observe_checkpointer(delegate)

    assert isinstance(proxy, ObservableCheckpointer)
    assert observe_checkpointer(proxy) is proxy
    assert proxy.serde is delegate.serde
    assert proxy.get_tuple({"configurable": {"thread_id": "thread-1"}}) is None
