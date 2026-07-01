"""Context usage tests for plain LLM calls."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.context_engineering.schema import ContextConfigError
from src.observability.a3_trace import reset_trace_event_sink, set_trace_event_sink


@pytest.mark.anyio
async def test_plain_llm_emits_context_usage_before_call(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    mock_llm = MagicMock()
    mock_llm.model_name = "deepseek-v4-pro"
    mock_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content="usable answer"))
    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    monkeypatch.setattr(
        llm_module, "get_llm_call_max_retries", lambda node_name=None, default=0: 0
    )

    sink: list[dict] = []
    token = set_trace_event_sink(sink)
    try:
        result = await invoke_plain_llm_fail_fast(
            node_name="generate_answer",
            llm_node="generate_answer",
            messages=[{"role": "user", "content": "question"}],
            state={"request_id": "r1", "thread_id": "t1"},
        )
    finally:
        reset_trace_event_sink(token)

    assert result == "usable answer"
    stages = [event.get("stage") for event in sink]
    assert stages.index("context_usage") < stages.index("plain_llm_output")
    assert mock_llm.ainvoke.await_count == 1


@pytest.mark.anyio
async def test_plain_llm_strict_missing_model_window_fails_before_call(monkeypatch):
    from src.context_engineering import budget
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    def fake_graph_setting(key: str, default=None):
        values = {
            "llm.missing_window.model": "unknown-model",
            "llm.missing_window.provider": "deepseek_official",
            "llm.missing_window.temperature": 0.7,
            "llm_outputs.strict_missing.max_raw_chars": 12000,
        }
        return values.get(key, default)

    def fake_budget_setting(key: str, default=None):
        if key == "context_engineering":
            return {
                "enabled": True,
                "strict": True,
                "tokenizer": {"mode": "estimated_mixed", "estimated": True},
                "model_limits": {},
                "thresholds": {
                    "warning_ratio": 0.70,
                    "critical_ratio": 0.85,
                    "compact_ratio": 0.90,
                },
                "default_reserved_output_tokens": 16000,
            }
        return default

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content="should not run"))
    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    monkeypatch.setattr(llm_module, "get_setting", fake_graph_setting)
    monkeypatch.setattr(budget, "get_setting", fake_budget_setting)

    with pytest.raises(ContextConfigError, match="model_window_unknown"):
        await invoke_plain_llm_fail_fast(
            node_name="strict_missing",
            llm_node="missing_window",
            messages=[{"role": "user", "content": "question"}],
            state={"request_id": "r1", "thread_id": "t1"},
        )

    mock_llm.ainvoke.assert_not_called()
