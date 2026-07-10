"""Tests for read-only pipeline continuity collection."""

from __future__ import annotations

from copy import deepcopy

from src.context_engineering.influence import (
    build_influence_entry,
    build_influence_update,
    merge_context_influence_ledger,
)
from src.context_engineering.providers.base import ProviderContext
from src.context_engineering.providers.pipeline_provider import PipelineContextProvider


def _context(state: dict, *, request_id: str = "request-2") -> ProviderContext:
    return ProviderContext(
        node_name="mindmap_agent",
        llm_node="mindmap",
        user_query="query",
        current_user_message_index=None,
        state=state,
        messages=[],
        request_id=request_id,
        thread_id="thread-1",
        max_items_per_provider=10,
        max_content_chars_per_item=4000,
    )


def _ledger() -> dict:
    state = {"request_id": "request-1", "thread_id": "thread-1"}
    entries = [
        build_influence_entry(
            state=state,
            kind="planner_output",
            source_node="review_doc_planner",
            title="Review outline",
            preview="A compact approved outline.",
            metadata={"workflow": "review_doc", "iteration": 1},
            priority=80,
        ),
        build_influence_entry(
            state=state,
            kind="local_evidence",
            source_node="rag_retrieve",
            title="Raw retrieval metadata",
            preview="Must not be injected through pipeline.",
            injectable=False,
        ),
    ]
    return merge_context_influence_ledger(
        {}, build_influence_update(state=state, entries=entries)
    )


def test_pipeline_provider_collects_only_injectable_entries_and_is_read_only():
    state = {"context_influence_ledger": _ledger()}
    before = deepcopy(state)

    items = PipelineContextProvider().collect(_context(state))

    assert state == before
    assert len(items) == 1
    assert items[0].source_type == "pipeline"
    assert items[0].content == "A compact approved outline."
    assert items[0].metadata["source_node"] == "review_doc_planner"
    assert items[0].metadata["purpose"] == "pipeline_continuity"
    assert items[0].relevance_score == 0.68


def test_pipeline_provider_current_request_outranks_thread_continuity():
    state = {"context_influence_ledger": _ledger()}

    prior = PipelineContextProvider().collect(_context(state, request_id="request-2"))
    current = PipelineContextProvider().collect(_context(state, request_id="request-1"))

    assert current[0].relevance_score == 0.9
    assert prior[0].relevance_score == 0.68


def test_pipeline_provider_skips_corrupt_workspace_without_mutation():
    state = {
        "context_influence_ledger": {
            "schema_version": 1,
            "entries_by_id": {"bad": "raw body"},
            "ordered_ids": ["bad"],
        }
    }
    before = deepcopy(state)

    assert PipelineContextProvider().collect(_context(state)) == []
    assert state == before


def test_pipeline_provider_revalidates_kind_and_redacts_corrupt_preview():
    ledger = _ledger()
    injectable_id = next(
        influence_id
        for influence_id, entry in ledger["entries_by_id"].items()
        if entry["injectable"]
    )
    ledger["entries_by_id"][injectable_id]["preview"] = (
        "safe summary api_key=sk-abcdefghijklmnopqrstuvwxyz"
    )
    items = PipelineContextProvider().collect(
        _context({"context_influence_ledger": ledger})
    )

    assert len(items) == 1
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in items[0].content
    assert "[REDACTED]" in items[0].content

    ledger["entries_by_id"][injectable_id]["kind"] = "local_evidence"
    assert (
        PipelineContextProvider().collect(
            _context({"context_influence_ledger": ledger})
        )
        == []
    )
