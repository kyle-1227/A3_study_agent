"""Plain LLM integration tests for ContextPacker shadow mode."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.context_engineering.packing.orchestrator import ContextPreparedMessages
from src.context_engineering.schema import ContextItem


def _item() -> ContextItem:
    return ContextItem(
        id="item-1",
        source_type="message",
        title="query",
        content="content",
        token_estimate=10,
        estimated=True,
        tokenizer_mode="estimated_mixed",
        priority=80,
        relevance_score=None,
        recency_score=None,
        confidence=None,
        scope="turn",
        lifetime="turn",
        compressible=True,
        can_drop=True,
        disclosure_level="snippet",
        metadata={},
    )


@pytest.mark.anyio
async def test_plain_llm_runs_packing_shadow_without_modifying_messages(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    prepare_calls: list[dict] = []
    mock_llm = MagicMock()
    mock_llm.model_name = "deepseek-v4-pro"
    mock_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content="answer"))
    prepared_messages = list(messages)

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    monkeypatch.setattr(
        llm_module,
        "get_llm_call_max_retries",
        lambda node_name=None, default=0: 0,
    )

    def _prepare_messages(_logger, **kwargs):
        prepare_calls.append(kwargs)
        return ContextPreparedMessages(
            messages_for_llm=prepared_messages,
            original_messages=list(kwargs["messages"]),
            trace_call_id="trace-1",
            next_trace_seq=0,
            context_apply_applied=False,
        )

    monkeypatch.setattr(
        llm_module,
        "prepare_messages_with_context_policy",
        _prepare_messages,
    )

    result = await invoke_plain_llm_fail_fast(
        node_name="generate_answer",
        llm_node="generate_answer",
        messages=messages,
        state={"request_id": "r1", "thread_id": "t1"},
    )

    assert result == "answer"
    assert prepare_calls[0]["node_name"] == "generate_answer"
    assert prepare_calls[0]["messages"] == messages
    mock_llm.ainvoke.assert_awaited_once_with(prepared_messages)
    assert messages == [{"role": "user", "content": "question"}]
