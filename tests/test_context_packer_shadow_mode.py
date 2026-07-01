"""Shadow mode tests for Phase 3A ContextPacker."""

from __future__ import annotations

import logging

from src.context_engineering.packing import policies as packing_policies
from src.context_engineering.packing import trace as packing_trace
from src.context_engineering.packing.trace import emit_context_packing_shadow
from src.context_engineering.schema import ContextItem
from src.observability.a3_trace import reset_trace_event_sink, set_trace_event_sink


def _item(
    token_estimate: int = 10,
    *,
    source_type: str = "message",
    can_drop: bool = True,
) -> ContextItem:
    return ContextItem(
        id="item-1",
        source_type=source_type,
        title="query",
        content="content",
        token_estimate=token_estimate,
        estimated=True,
        tokenizer_mode="estimated_mixed",
        priority=80,
        relevance_score=None,
        recency_score=None,
        confidence=None,
        scope="turn",
        lifetime="turn",
        compressible=True,
        can_drop=can_drop,
        disclosure_level="snippet",
        metadata={},
    )


def _fake_settings(*, enabled: bool = True, apply_to_llm: bool = False):
    def fake_get_setting(key: str, default=None):
        if key == "context_engineering":
            return {
                "enabled": True,
                "strict": True,
                "packer": {
                    "enabled": enabled,
                    "shadow_mode": True,
                    "apply_to_llm": apply_to_llm,
                    "strategy": "priority_budget",
                    "max_context_block_tokens": 1000,
                    "trace_selected_items": 10,
                    "trace_dropped_items": 10,
                    "enabled_nodes": [],
                    "enabled_sources": ["message"],
                },
            }
        return default

    return fake_get_setting


def test_emit_context_packing_shadow_enabled_false_sends_no_events(monkeypatch):
    monkeypatch.setattr(
        packing_trace, "get_packing_policy", lambda **_: _disabled_policy()
    )
    sink: list[dict] = []
    token = set_trace_event_sink(sink)
    try:
        result = emit_context_packing_shadow(
            logging.getLogger("test_context_packer_shadow_mode"),
            node_name="node",
            llm_node="llm",
            items=[_item()],
            state={"request_id": "r1", "thread_id": "t1"},
        )
    finally:
        reset_trace_event_sink(token)

    assert result is None
    assert sink == []


def test_packer_enabled_false_minimal_config_sends_no_events(monkeypatch):
    def fake_get_setting(key: str, default=None):
        if key == "context_engineering":
            return {"enabled": True, "packer": {"enabled": False}}
        return default

    monkeypatch.setattr(packing_policies, "get_setting", fake_get_setting)
    sink: list[dict] = []
    token = set_trace_event_sink(sink)
    try:
        result = emit_context_packing_shadow(
            logging.getLogger("test_context_packer_shadow_mode"),
            node_name="node",
            llm_node="llm",
            items=[_item()],
            state={"request_id": "r1", "thread_id": "t1"},
        )
    finally:
        reset_trace_event_sink(token)

    assert result is None
    assert sink == []


def test_emit_context_packing_shadow_apply_to_llm_error_does_not_escape(monkeypatch):
    monkeypatch.setattr(
        packing_trace,
        "get_packing_policy",
        lambda **_: _enabled_policy(apply_to_llm=True),
    )
    sink: list[dict] = []
    token = set_trace_event_sink(sink)
    try:
        result = emit_context_packing_shadow(
            logging.getLogger("test_context_packer_shadow_mode"),
            node_name="node",
            llm_node="llm",
            items=[_item()],
            state={"request_id": "r1", "thread_id": "t1"},
        )
    finally:
        reset_trace_event_sink(token)

    assert result is None
    assert sink[0]["stage"] == "context_packing_error"
    assert sink[0]["reason"] == "apply_to_llm_unsupported"


def test_emit_context_packing_shadow_packer_error_does_not_escape(monkeypatch):
    monkeypatch.setattr(
        packing_trace, "get_packing_policy", lambda **_: _enabled_policy()
    )
    sink: list[dict] = []
    token = set_trace_event_sink(sink)
    try:
        result = emit_context_packing_shadow(
            logging.getLogger("test_context_packer_shadow_mode"),
            node_name="node",
            llm_node="llm",
            items=[_item(token_estimate=1)],
            state={"request_id": "r1", "thread_id": "t1"},
        )
    finally:
        reset_trace_event_sink(token)

    assert result is not None
    stages = [event["stage"] for event in sink]
    assert "context_packing_plan" in stages
    assert "context_packed" in stages
    assert all("rendered_context" not in repr(event).lower() for event in sink)


def test_emit_context_packing_shadow_render_budget_error_is_trace_only(monkeypatch):
    monkeypatch.setattr(
        packing_trace,
        "get_packing_policy",
        lambda **_: _enabled_policy(max_context_block_tokens=2),
    )
    sink: list[dict] = []
    token = set_trace_event_sink(sink)
    try:
        result = emit_context_packing_shadow(
            logging.getLogger("test_context_packer_shadow_mode"),
            node_name="node",
            llm_node="llm",
            items=[_item(token_estimate=1)],
            state={"request_id": "r1", "thread_id": "t1"},
        )
    finally:
        reset_trace_event_sink(token)

    assert result is None
    assert sink[-1]["stage"] == "context_packing_error"
    assert sink[-1]["reason"] == "rendered_context_over_budget"


def test_emit_context_packing_shadow_required_source_disabled_is_trace_only(
    monkeypatch,
):
    monkeypatch.setattr(
        packing_trace,
        "get_packing_policy",
        lambda **_: _enabled_policy(enabled_sources=("message",)),
    )
    sink: list[dict] = []
    token = set_trace_event_sink(sink)
    try:
        result = emit_context_packing_shadow(
            logging.getLogger("test_context_packer_shadow_mode"),
            node_name="node",
            llm_node="llm",
            items=[_item(source_type="memory", can_drop=False)],
            state={"request_id": "r1", "thread_id": "t1"},
        )
    finally:
        reset_trace_event_sink(token)

    assert result is None
    assert sink[-1]["stage"] == "context_packing_error"
    assert sink[-1]["reason"] == "required_source_disabled"
    assert all(event.get("reason") != "source_disabled" for event in sink)


def _enabled_policy(
    *,
    apply_to_llm: bool = False,
    max_context_block_tokens: int = 1000,
    enabled_sources: tuple[str, ...] = ("message",),
):
    from src.context_engineering.packing.policies import PackingPolicy

    return PackingPolicy(
        enabled=True,
        shadow_mode=True,
        apply_to_llm=apply_to_llm,
        strategy="priority_budget",
        max_context_block_tokens=max_context_block_tokens,
        trace_selected_items=10,
        trace_dropped_items=10,
        enabled_nodes=(),
        enabled_sources=enabled_sources,
    )


def _disabled_policy():
    from src.context_engineering.packing.policies import PackingPolicy

    return PackingPolicy(
        enabled=False,
        shadow_mode=True,
        apply_to_llm=False,
        strategy="priority_budget",
        max_context_block_tokens=1000,
        trace_selected_items=10,
        trace_dropped_items=10,
        enabled_nodes=(),
        enabled_sources=("message",),
    )
