"""Unit tests for LLM fallback/resilience mechanism.

Tests cover: invoke_with_fallback helper, async_invoke_with_fallback helper,
per-node fallback behavior, and tracing attributes on failover events.
All tests mock LLM invocations — no real API calls required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.context_engineering.input_manifest import build_llm_input_manifest
from src.observability.a3_trace import reset_trace_event_sink, set_trace_event_sink


def _test_manifest(
    *,
    node_name: str = "test_node",
    llm_node: str = "test_node",
    provider: str = "test_provider",
    model: str = "test-model",
    state: dict | None = None,
) -> dict:
    return build_llm_input_manifest(
        node_name=node_name,
        llm_node=llm_node,
        provider=provider,
        model=model,
        messages=[HumanMessage(content="test prompt")],
        state=state or {"request_id": "r-test", "thread_id": "t-test"},
        call_purpose="test_llm_call",
    )


# ===========================================================================
# TestInvokeWithFallback — sync core helper (kept for backward compat)
# ===========================================================================


class TestInvokeWithFallback:
    """Unit tests for invoke_with_fallback()."""

    def test_returns_primary_response_on_success(self):
        """Primary succeeds → return its response directly."""
        from src.graph.llm import invoke_with_fallback

        primary = MagicMock()
        primary.invoke.return_value = "primary-ok"

        result = invoke_with_fallback(primary, ["msg"])
        assert result == "primary-ok"

    def test_sets_fallback_used_false_on_success(self):
        """Span should record llm.fallback_used=False when primary succeeds."""
        from src.graph.llm import invoke_with_fallback

        primary = MagicMock()
        primary.invoke.return_value = "ok"
        span = MagicMock()

        invoke_with_fallback(primary, ["msg"], span=span)
        span.set_attribute.assert_called_with("llm.fallback_used", False)

    def test_falls_back_on_timeout_error(self):
        """TimeoutError on primary → fallback response returned."""
        from src.graph.llm import invoke_with_fallback

        primary = MagicMock()
        primary.invoke.side_effect = TimeoutError("timed out")
        fallback = MagicMock()
        fallback.invoke.return_value = "fallback-ok"

        result = invoke_with_fallback(primary, ["msg"], fallback=fallback)
        assert result == "fallback-ok"

    def test_falls_back_on_connection_error(self):
        """ConnectionError on primary → fallback response returned."""
        from src.graph.llm import invoke_with_fallback

        primary = MagicMock()
        primary.invoke.side_effect = ConnectionError("refused")
        fallback = MagicMock()
        fallback.invoke.return_value = "fallback-ok"

        result = invoke_with_fallback(primary, ["msg"], fallback=fallback)
        assert result == "fallback-ok"

    def test_sets_fallback_used_true_on_failover(self):
        """Span should record llm.fallback_used=True on failover."""
        from src.graph.llm import invoke_with_fallback

        primary = MagicMock()
        primary.invoke.side_effect = TimeoutError("timed out")
        fallback = MagicMock()
        fallback.invoke.return_value = "ok"
        span = MagicMock()

        invoke_with_fallback(primary, ["msg"], fallback=fallback, span=span)
        span.set_attribute.assert_any_call("llm.fallback_used", True)

    def test_records_fallback_model_on_span(self):
        """Span should record the fallback model name."""
        from src.graph.llm import invoke_with_fallback

        primary = MagicMock()
        primary.invoke.side_effect = TimeoutError("timed out")
        fallback = MagicMock()
        fallback.model_name = "deepseek-lite"
        fallback.invoke.return_value = "ok"
        span = MagicMock()

        invoke_with_fallback(primary, ["msg"], fallback=fallback, span=span)
        span.set_attribute.assert_any_call("llm.fallback_model", "deepseek-lite")

    def test_records_fallback_event_on_span(self):
        """Span should have an llm.fallback_triggered event with error details."""
        from src.graph.llm import invoke_with_fallback

        primary = MagicMock()
        primary.invoke.side_effect = TimeoutError("timed out")
        fallback = MagicMock()
        fallback.model_name = "deepseek-lite"
        fallback.invoke.return_value = "ok"
        span = MagicMock()

        invoke_with_fallback(primary, ["msg"], fallback=fallback, span=span)

        span.add_event.assert_called_once()
        call_args = span.add_event.call_args
        assert call_args[0][0] == "llm.fallback_triggered"
        assert call_args[0][1]["error_type"] == "TimeoutError"

    def test_propagates_error_when_no_fallback(self):
        """No fallback configured + primary fails → error propagates."""
        from src.graph.llm import invoke_with_fallback

        primary = MagicMock()
        primary.invoke.side_effect = TimeoutError("timed out")

        with pytest.raises(TimeoutError):
            invoke_with_fallback(primary, ["msg"])

    def test_propagates_fallback_error(self):
        """Both primary and fallback fail → fallback error propagates."""
        from src.graph.llm import invoke_with_fallback

        primary = MagicMock()
        primary.invoke.side_effect = TimeoutError("primary failed")
        fallback = MagicMock()
        fallback.invoke.side_effect = ConnectionError("fallback also failed")

        with pytest.raises(ConnectionError):
            invoke_with_fallback(primary, ["msg"], fallback=fallback)

    def test_does_not_catch_value_error(self):
        """Non-recoverable errors (ValueError) should NOT trigger fallback."""
        from src.graph.llm import invoke_with_fallback

        primary = MagicMock()
        primary.invoke.side_effect = ValueError("bad input")
        fallback = MagicMock()

        with pytest.raises(ValueError):
            invoke_with_fallback(primary, ["msg"], fallback=fallback)

        fallback.invoke.assert_not_called()

    def test_does_not_catch_key_error(self):
        """KeyError is a programming bug, not an API error — should propagate."""
        from src.graph.llm import invoke_with_fallback

        primary = MagicMock()
        primary.invoke.side_effect = KeyError("missing")
        fallback = MagicMock()

        with pytest.raises(KeyError):
            invoke_with_fallback(primary, ["msg"], fallback=fallback)

        fallback.invoke.assert_not_called()


# ===========================================================================
# TestAsyncInvokeWithFallback — async core helper
# ===========================================================================


class TestAsyncInvokeWithFallback:
    """Unit tests for async_invoke_with_fallback()."""

    async def test_returns_primary_response_on_success(self):
        from src.graph.llm import async_invoke_with_fallback

        primary = MagicMock()
        primary.ainvoke = AsyncMock(return_value="primary-ok")

        result = await async_invoke_with_fallback(primary, ["msg"])
        assert result == "primary-ok"

    async def test_sets_fallback_used_false_on_success(self):
        from src.graph.llm import async_invoke_with_fallback

        primary = MagicMock()
        primary.ainvoke = AsyncMock(return_value="ok")
        span = MagicMock()

        await async_invoke_with_fallback(primary, ["msg"], span=span)
        span.set_attribute.assert_called_with("llm.fallback_used", False)

    async def test_falls_back_on_timeout_error(self):
        from src.graph.llm import async_invoke_with_fallback

        primary = MagicMock()
        primary.ainvoke = AsyncMock(side_effect=TimeoutError("timed out"))
        fallback = MagicMock()
        fallback.ainvoke = AsyncMock(return_value="fallback-ok")

        result = await async_invoke_with_fallback(primary, ["msg"], fallback=fallback)
        assert result == "fallback-ok"

    async def test_falls_back_on_connection_error(self):
        from src.graph.llm import async_invoke_with_fallback

        primary = MagicMock()
        primary.ainvoke = AsyncMock(side_effect=ConnectionError("refused"))
        fallback = MagicMock()
        fallback.ainvoke = AsyncMock(return_value="fallback-ok")

        result = await async_invoke_with_fallback(primary, ["msg"], fallback=fallback)
        assert result == "fallback-ok"

    async def test_propagates_error_when_no_fallback(self):
        from src.graph.llm import async_invoke_with_fallback

        primary = MagicMock()
        primary.ainvoke = AsyncMock(side_effect=TimeoutError("timed out"))

        with pytest.raises(TimeoutError):
            await async_invoke_with_fallback(primary, ["msg"])

    async def test_does_not_catch_value_error(self):
        from src.graph.llm import async_invoke_with_fallback

        primary = MagicMock()
        primary.ainvoke = AsyncMock(side_effect=ValueError("bad input"))
        fallback = MagicMock()

        with pytest.raises(ValueError):
            await async_invoke_with_fallback(primary, ["msg"], fallback=fallback)

        fallback.ainvoke.assert_not_called()

    async def test_retries_primary_before_fallback(self, monkeypatch):
        from src.graph import llm as llm_module
        from src.graph.llm import async_invoke_with_fallback

        monkeypatch.setattr(
            llm_module, "get_llm_call_max_retries", lambda node_name=None, default=2: 2
        )
        primary = MagicMock()
        primary.ainvoke = AsyncMock(
            side_effect=[
                TimeoutError("first"),
                TimeoutError("second"),
                "primary-ok",
            ]
        )
        fallback = MagicMock()
        fallback.ainvoke = AsyncMock(return_value="fallback-ok")

        result = await async_invoke_with_fallback(primary, ["msg"], fallback=fallback)

        assert result == "primary-ok"
        assert primary.ainvoke.await_count == 3
        fallback.ainvoke.assert_not_called()

    async def test_falls_back_after_primary_retry_budget(self, monkeypatch):
        from src.graph import llm as llm_module
        from src.graph.llm import async_invoke_with_fallback

        monkeypatch.setattr(
            llm_module, "get_llm_call_max_retries", lambda node_name=None, default=2: 2
        )
        primary = MagicMock()
        primary.ainvoke = AsyncMock(
            side_effect=[
                TimeoutError("first"),
                TimeoutError("second"),
                TimeoutError("third"),
            ]
        )
        fallback = MagicMock()
        fallback.ainvoke = AsyncMock(return_value="fallback-ok")

        result = await async_invoke_with_fallback(primary, ["msg"], fallback=fallback)

        assert result == "fallback-ok"
        assert primary.ainvoke.await_count == 3
        fallback.ainvoke.assert_awaited_once()


# ===========================================================================
# TestProviderTransportRetry - same provider request, no business fallback
# ===========================================================================


class TestProviderTransportRetry:
    """Transport retry should not change model, prompt, schema, or fallback flags."""

    @pytest.mark.anyio
    async def test_retries_timeout_and_returns_success(self, monkeypatch):
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
                provider="openrouter",
                model="test-model",
                llm_input_manifest=_test_manifest(
                    node_name="evidence_judge",
                    llm_node="evidence_judge",
                    provider="openrouter",
                    state={"request_id": "r1", "thread_id": "t1"},
                ),
                state={"request_id": "r1"},
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
        assert all(event["fallback_used"] is False for event in retry_events)

    @pytest.mark.anyio
    async def test_does_not_retry_programming_errors(self, monkeypatch):
        from src.graph import llm as llm_module
        from src.graph.llm import invoke_with_provider_transport_retry

        monkeypatch.setattr(
            llm_module, "_provider_transport_max_retries", lambda node_name=None: 3
        )
        monkeypatch.setattr(llm_module.asyncio, "sleep", AsyncMock())

        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            raise ValueError("bad schema")

        with pytest.raises(ValueError):
            await invoke_with_provider_transport_retry(
                operation,
                node_name="supervisor",
                llm_node="supervisor",
                provider="openrouter",
                model="test-model",
                llm_input_manifest=_test_manifest(
                    node_name="supervisor",
                    llm_node="supervisor",
                    provider="openrouter",
                ),
                state={},
            )

        assert calls == 1
        llm_module.asyncio.sleep.assert_not_called()

    @pytest.mark.anyio
    async def test_retries_429_and_emits_final_failure(self, monkeypatch):
        from src.graph import llm as llm_module
        from src.graph.llm import invoke_with_provider_transport_retry

        monkeypatch.setattr(
            llm_module, "_provider_transport_max_retries", lambda node_name=None: 2
        )
        monkeypatch.setattr(
            llm_module, "_provider_transport_delay_seconds", lambda _attempt: 0
        )
        monkeypatch.setattr(llm_module.asyncio, "sleep", AsyncMock())

        class Response:
            status_code = 429
            text = "rate limited"

        events: list[dict] = []
        token = set_trace_event_sink(events)
        try:

            async def operation():
                exc = RuntimeError("rate limited")
                exc.response = Response()
                raise exc

            with pytest.raises(RuntimeError):
                await invoke_with_provider_transport_retry(
                    operation,
                    node_name="query_rewrite",
                    llm_node="query_rewrite",
                    provider="openrouter",
                    model="test-model",
                    llm_input_manifest=_test_manifest(
                        node_name="query_rewrite",
                        llm_node="query_rewrite",
                        provider="openrouter",
                        state={"request_id": "r2", "thread_id": "t2"},
                    ),
                    state={"request_id": "r2"},
                )
        finally:
            reset_trace_event_sink(token)

        stages = [event["stage"] for event in events]
        assert stages.count("provider_transport_error") == 2
        assert stages.count("provider_transport_retry_attempt") == 2
        assert stages[-1] == "final_failure_after_retries"
        assert events[-1]["retry_count"] == 2
        assert events[-1]["status_code"] == 429
        assert events[-1]["fallback_used"] is False

    @pytest.mark.anyio
    async def test_does_not_retry_http_400_invalid_request(self, monkeypatch):
        from src.graph import llm as llm_module
        from src.graph.llm import invoke_with_provider_transport_retry

        monkeypatch.setattr(
            llm_module, "_provider_transport_max_retries", lambda node_name=None: 2
        )
        monkeypatch.setattr(llm_module.asyncio, "sleep", AsyncMock())

        class Response:
            status_code = 400
            text = "invalid strict schema"

        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            exc = RuntimeError("invalid request")
            exc.response = Response()
            raise exc

        with pytest.raises(RuntimeError):
            await invoke_with_provider_transport_retry(
                operation,
                node_name="supervisor",
                llm_node="supervisor",
                provider="deepseek_official",
                model="deepseek-v4-pro",
                llm_input_manifest=_test_manifest(
                    node_name="supervisor",
                    llm_node="supervisor",
                    provider="deepseek_official",
                    model="deepseek-v4-pro",
                ),
                state={},
            )

        assert calls == 1
        llm_module.asyncio.sleep.assert_not_called()

    @pytest.mark.anyio
    async def test_missing_manifest_fails_before_provider_call(self):
        from src.context_engineering.input_manifest import LLMInputManifestError
        from src.graph.llm import invoke_with_provider_transport_retry

        operation = AsyncMock(return_value="should-not-run")

        with pytest.raises(LLMInputManifestError):
            await invoke_with_provider_transport_retry(
                operation,
                node_name="supervisor",
                llm_node="supervisor",
                provider="deepseek_official",
                model="deepseek-v4-pro",
                llm_input_manifest={},
                state={},
            )

        operation.assert_not_called()


class TestPlainLLMRetry:
    @pytest.mark.anyio
    async def test_retries_empty_output_then_succeeds(self, monkeypatch):
        from src.graph import llm as llm_module
        from src.graph.llm import invoke_plain_llm_fail_fast

        monkeypatch.setattr(
            llm_module, "get_llm_call_max_retries", lambda node_name=None, default=2: 2
        )
        mock_llm = MagicMock()
        mock_llm.model_name = "deepseek-v4-pro"
        mock_llm.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(content="   "),
                AIMessage(content=""),
                AIMessage(content="usable answer"),
            ]
        )
        monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)

        result = await invoke_plain_llm_fail_fast(
            node_name="plain_retry",
            llm_node="plain_retry",
            messages=[HumanMessage(content="question")],
            state={},
        )

        assert result == "usable answer"
        assert mock_llm.ainvoke.await_count == 3


# ===========================================================================
# TestGenerateAnswerFallback — academic node
# ===========================================================================


class TestGenerateAnswerFallback:
    """generate_answer must use fail-fast plain invoke, not provider fallback."""

    @patch("src.graph.academic.invoke_plain_llm_fail_fast")
    async def test_propagates_plain_invoke_timeout(self, mock_invoke_plain):
        mock_invoke_plain.side_effect = TimeoutError("primary timed out")

        state = {
            "messages": [HumanMessage(content="test")],
            "context": [],
        }

        from src.graph.academic import generate_answer

        with pytest.raises(TimeoutError):
            await generate_answer(state)

    @patch("src.graph.academic.invoke_plain_llm_fail_fast")
    async def test_returns_plain_response_when_healthy(self, mock_invoke_plain):
        mock_invoke_plain.return_value = "primary answer"

        state = {
            "messages": [HumanMessage(content="test")],
            "context": [],
        }

        from src.graph.academic import generate_answer

        result = await generate_answer(state)

        assert "primary answer" in result["messages"][0].content
        assert "fallback_model" not in mock_invoke_plain.await_args.kwargs


# ===========================================================================
# TestEmotionalResponseFallback — emotional node
# ===========================================================================


class TestEmotionalResponseFallback:
    """emotional_response must use fail-fast plain invoke, not provider fallback."""

    @patch("src.graph.emotional.invoke_plain_llm_fail_fast")
    async def test_propagates_plain_invoke_timeout(self, mock_invoke_plain):
        mock_invoke_plain.side_effect = TimeoutError("primary timed out")

        state = {
            "messages": [HumanMessage(content="test")],
            "thread_id": "thread-1",
            "request_id": "request-1",
        }

        from src.graph.emotional import emotional_response

        with pytest.raises(TimeoutError):
            await emotional_response(state)

    @patch("src.graph.emotional.invoke_plain_llm_fail_fast")
    async def test_returns_plain_response_when_healthy(self, mock_invoke_plain):
        mock_invoke_plain.return_value = "primary comfort"

        state = {
            "messages": [HumanMessage(content="test")],
            "thread_id": "thread-1",
            "request_id": "request-1",
        }

        from src.graph.emotional import emotional_response

        result = await emotional_response(state)

        assert "primary comfort" in result["messages"][0].content
        assert "fallback_model" not in mock_invoke_plain.await_args.kwargs


# ===========================================================================
# TestFallbackTracing — OTel span attributes on failover
# ===========================================================================


class TestFallbackTracing:
    """Graph plain LLM nodes should not emit fallback span metadata."""

    @patch("src.graph.academic.invoke_plain_llm_fail_fast")
    async def test_span_does_not_record_fallback_attributes_on_failure(
        self,
        mock_invoke_plain,
        in_memory_exporter,
    ):
        """Plain LLM failure should fail fast without fallback metadata."""
        mock_invoke_plain.side_effect = TimeoutError("timed out")

        state = {
            "messages": [HumanMessage(content="test")],
            "context": [],
        }

        from src.graph.academic import generate_answer

        with pytest.raises(TimeoutError):
            await generate_answer(state)

        spans = in_memory_exporter.get_finished_spans()
        llm_spans = [s for s in spans if s.name.startswith("llm.invoke")]
        assert len(llm_spans) >= 1

        llm_span = llm_spans[0]
        attrs = dict(llm_span.attributes)
        assert "llm.fallback_used" not in attrs
        assert "llm.fallback_model" not in attrs

        event_names = [e.name for e in llm_span.events]
        assert "llm.fallback_triggered" not in event_names

    @patch("src.graph.academic.invoke_plain_llm_fail_fast")
    async def test_span_records_plain_success_without_fallback_metadata(
        self,
        mock_invoke_plain,
        in_memory_exporter,
    ):
        """Plain LLM success should not add fallback metadata."""
        mock_invoke_plain.return_value = "primary ok"

        state = {
            "messages": [HumanMessage(content="test")],
            "context": [],
        }

        from src.graph.academic import generate_answer

        await generate_answer(state)

        spans = in_memory_exporter.get_finished_spans()
        llm_spans = [s for s in spans if s.name.startswith("llm.invoke")]
        assert len(llm_spans) >= 1

        attrs = dict(llm_spans[0].attributes)
        assert "llm.fallback_used" not in attrs
        event_names = [e.name for e in llm_spans[0].events]
        assert "llm.fallback_triggered" not in event_names
