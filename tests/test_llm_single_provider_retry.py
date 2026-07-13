"""Tests for bounded retries that never change provider or model."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage

from src.context_engineering.input_manifest import build_llm_input_manifest
from src.observability.a3_trace import reset_trace_event_sink, set_trace_event_sink


def _manifest(
    *,
    node_name: str = "test_node",
    provider: str = "deepseek_official",
    model: str = "test-model",
) -> dict:
    return build_llm_input_manifest(
        node_name=node_name,
        llm_node=node_name,
        provider=provider,
        model=model,
        messages=[HumanMessage(content="test prompt")],
        state={"request_id": "request-1", "thread_id": "thread-1"},
        call_purpose="test_llm_call",
    )


@pytest.mark.anyio
async def test_transport_retry_reuses_same_operation(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_with_provider_transport_retry

    monkeypatch.setattr(
        llm_module, "_provider_transport_max_retries", lambda node_name=None: 2
    )
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
            if calls == 1:
                raise TimeoutError("provider timed out")
            return "ok"

        result, retry_count = await invoke_with_provider_transport_retry(
            operation,
            node_name="evidence_judge",
            llm_node="evidence_judge",
            provider="deepseek_official",
            model="test-model",
            llm_input_manifest=_manifest(node_name="evidence_judge"),
            state={"request_id": "request-1", "thread_id": "thread-1"},
        )
    finally:
        reset_trace_event_sink(token)

    assert result == "ok"
    assert retry_count == 1
    assert calls == 2
    retry_events = [
        event for event in events if event["stage"].startswith("provider_transport")
    ]
    assert [event["stage"] for event in retry_events] == [
        "provider_transport_error",
        "provider_transport_retry_attempt",
    ]
    assert all(event["provider"] == "deepseek_official" for event in retry_events)
    assert all(event["model"] == "test-model" for event in retry_events)


@pytest.mark.anyio
async def test_transport_retry_does_not_retry_programming_error(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_with_provider_transport_retry

    monkeypatch.setattr(
        llm_module, "_provider_transport_max_retries", lambda node_name=None: 3
    )
    monkeypatch.setattr(llm_module.asyncio, "sleep", AsyncMock())
    operation = AsyncMock(side_effect=ValueError("bad schema"))

    with pytest.raises(ValueError, match="bad schema"):
        await invoke_with_provider_transport_retry(
            operation,
            node_name="supervisor",
            llm_node="supervisor",
            provider="deepseek_official",
            model="test-model",
            llm_input_manifest=_manifest(node_name="supervisor"),
            state={},
        )

    operation.assert_awaited_once()
    llm_module.asyncio.sleep.assert_not_called()


@pytest.mark.anyio
async def test_transport_retry_exhaustion_is_a_typed_failure(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_with_provider_transport_retry

    monkeypatch.setattr(
        llm_module, "_provider_transport_max_retries", lambda node_name=None: 1
    )
    monkeypatch.setattr(
        llm_module, "_provider_transport_delay_seconds", lambda _attempt: 0
    )
    monkeypatch.setattr(llm_module.asyncio, "sleep", AsyncMock())

    class Response:
        status_code = 429

    error = RuntimeError("rate limited")
    error.response = Response()
    operation = AsyncMock(side_effect=error)
    events: list[dict] = []
    token = set_trace_event_sink(events)
    try:
        with pytest.raises(RuntimeError, match="rate limited"):
            await invoke_with_provider_transport_retry(
                operation,
                node_name="query_rewrite",
                llm_node="query_rewrite",
                provider="deepseek_official",
                model="test-model",
                llm_input_manifest=_manifest(node_name="query_rewrite"),
                state={},
            )
    finally:
        reset_trace_event_sink(token)

    assert operation.await_count == 2
    final_events = [
        event for event in events if event["stage"] == "final_failure_after_retries"
    ]
    assert len(final_events) == 1
    assert final_events[0]["retry_count"] == 1
    assert final_events[0]["status_code"] == 429


@pytest.mark.anyio
async def test_invalid_manifest_blocks_provider_dispatch():
    from src.context_engineering.input_manifest import LLMInputManifestError
    from src.graph.llm import invoke_with_provider_transport_retry

    operation = AsyncMock(return_value="must not run")
    with pytest.raises(LLMInputManifestError):
        await invoke_with_provider_transport_retry(
            operation,
            node_name="supervisor",
            llm_node="supervisor",
            provider="deepseek_official",
            model="test-model",
            llm_input_manifest={},
            state={},
        )

    operation.assert_not_called()


def test_cross_model_fallback_runtime_symbols_are_removed():
    source = Path("src/graph/llm.py").read_text(encoding="utf-8")
    env_example = Path(".env.example").read_text(encoding="utf-8")

    for symbol in (
        "get_fallback_llm",
        "invoke_with_fallback",
        "async_invoke_with_fallback",
        "FALLBACK_MODEL",
        "FALLBACK_API_KEY",
        "FALLBACK_BASE_URL",
    ):
        assert symbol not in source
        assert symbol not in env_example
