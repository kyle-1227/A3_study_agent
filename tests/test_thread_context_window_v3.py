from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.context_engineering.schema import ContextConfigError, ContextItem
from src.context_engineering.session_memory import (
    ContextInjectionRecordV1,
    apply_context_memory_compaction,
    build_injection_descriptor,
    new_session_context_memory_ledger,
    record_context_injection,
)
from src.context_engineering.thread_window_v3 import build_thread_context_window_v3


def _record(item_id: str, content: str, tokens: int) -> ContextInjectionRecordV1:
    item = ContextItem(
        id=item_id,
        source_type="memory",
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
    descriptor = build_injection_descriptor(item)
    assert descriptor is not None
    return ContextInjectionRecordV1(
        record_id=f"record-{item_id}",
        dispatch_id=f"dispatch-{item_id}",
        request_id="request-1",
        call_id="call-1",
        attempt=1,
        manifest_id="manifest-1",
        thread_id="thread-1",
        item=descriptor,
        dispatched_at=datetime(2026, 7, 13, tzinfo=timezone.utc),
    )


def test_v3_uses_retained_memory_ratio_and_contains_no_prediction_fields() -> None:
    ledger = record_context_injection(
        new_session_context_memory_ledger("thread-1"),
        _record("memory:a", "retained", 10_000),
    )
    window = build_thread_context_window_v3(ledger)
    payload = window.model_dump(mode="json")

    assert window.context_window_limit_tokens == 1_000_000
    assert window.retained_memory_tokens == 10_000
    assert window.retained_ratio == 0.01
    assert window.lifetime_injected_tokens == 10_000
    forbidden = {
        "thread_baseline",
        "next_call_estimate",
        "target_node",
        "output_reserve",
        "projected_peak",
        "projected_growth",
        "headroom",
    }
    assert forbidden.isdisjoint(payload)


def test_v3_compaction_lowers_retained_but_preserves_lifetime() -> None:
    ledger = new_session_context_memory_ledger("thread-1")
    ledger = record_context_injection(ledger, _record("memory:a", "one", 5))
    ledger = record_context_injection(ledger, _record("memory:b", "two", 8))
    before = build_thread_context_window_v3(ledger)
    compacted = apply_context_memory_compaction(
        ledger,
        boundary_id="boundary-1",
        retained_logical_item_ids=["memory:b"],
        compacted_at=datetime(2026, 7, 13, 1, tzinfo=timezone.utc),
        before_tokens=13,
        after_tokens=8,
    )
    after = build_thread_context_window_v3(compacted)

    assert before.retained_memory_tokens == 13
    assert after.retained_memory_tokens == 8
    assert after.lifetime_injected_tokens == before.lifetime_injected_tokens
    assert after.compaction.status == "compacted"


def test_v3_requires_explicit_session_window_model(monkeypatch) -> None:
    from src.context_engineering import thread_window_v3 as module

    monkeypatch.setattr(
        module,
        "get_context_engineering_config",
        lambda: {"enabled": True, "strict": True, "model_limits": {}},
    )
    with pytest.raises(ContextConfigError) as exc_info:
        build_thread_context_window_v3(new_session_context_memory_ledger("thread-1"))

    assert exc_info.value.reason == "session_memory_config_missing"
