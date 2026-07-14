from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from src.llm import structured_output as so


class DemoStructuredOutput(BaseModel):
    title: str
    count: int


@pytest.fixture(autouse=True)
def _configured_context_usage_model(monkeypatch):
    monkeypatch.setattr(so, "_provider", lambda _node: "deepseek_official")
    monkeypatch.setattr(so, "_model", lambda _node: "deepseek-v4-pro")


@pytest.mark.asyncio
async def test_structured_llm_retries_parse_and_validation_then_succeeds(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content="not json"),
            AIMessage(content='{"title": "missing count"}'),
            AIMessage(content='{"title": "ok", "count": 2}'),
        ]
    )
    monkeypatch.setattr(so, "get_node_llm", lambda _node: mock_llm)
    monkeypatch.setattr(so, "get_llm_call_max_retries", lambda _node: 2)

    result = await so.invoke_structured_llm(
        node_name="unit_structured_retry",
        llm_node="unit_structured_retry",
        schema=DemoStructuredOutput,
        messages=[HumanMessage(content="make json")],
        output_mode="prompt_json_pydantic",
    )

    assert result.success is True
    assert result.retry_count == 2
    assert result.parsed == DemoStructuredOutput(title="ok", count=2)
    assert mock_llm.ainvoke.await_count == 3
    assert [attempt.success for attempt in result.attempts] == [False, False, True]


@pytest.mark.asyncio
async def test_structured_llm_raises_after_retry_budget(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content="not json"),
            AIMessage(content="still not json"),
            AIMessage(content='{"title": "missing count"}'),
        ]
    )
    monkeypatch.setattr(so, "get_node_llm", lambda _node: mock_llm)
    monkeypatch.setattr(so, "get_llm_call_max_retries", lambda _node: 2)

    with pytest.raises(so.StructuredOutputError) as exc_info:
        await so.invoke_structured_llm(
            node_name="unit_structured_retry",
            llm_node="unit_structured_retry",
            schema=DemoStructuredOutput,
            messages=[HumanMessage(content="make json")],
            output_mode="prompt_json_pydantic",
        )

    result = exc_info.value.result
    assert result.success is False
    assert result.retry_count == 2
    assert len(result.attempts) == 3
    assert mock_llm.ainvoke.await_count == 3
    assert result.attempts[-1].failure_phase == "validation_error"


@pytest.mark.asyncio
async def test_sensitive_trace_redacts_private_input_and_invalid_output(monkeypatch):
    private_input = "private-input-answer-canary-413"
    private_output = "private-output-answer-canary-729"
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=f'{{"title": "{private_output}"}}')
    )
    trace_events: list[tuple[str, dict]] = []
    monkeypatch.setattr(so, "get_node_llm", lambda _node: mock_llm)
    monkeypatch.setattr(so, "get_llm_call_max_retries", lambda _node: 0)
    monkeypatch.setattr(
        so,
        "emit_a3_trace",
        lambda _logger, stage, payload, **_kwargs: trace_events.append(
            (stage, payload)
        ),
    )

    with pytest.raises(so.StructuredOutputError):
        await so.invoke_structured_llm(
            node_name="unit_structured_retry",
            llm_node="unit_structured_retry",
            schema=DemoStructuredOutput,
            messages=[HumanMessage(content=private_input)],
            output_mode="prompt_json_pydantic",
            sensitive_trace=True,
        )

    output_event = next(
        payload for stage, payload in trace_events if stage == "structured_llm_output"
    )
    serialized = json.dumps(output_event, ensure_ascii=False)
    assert private_input not in serialized
    assert private_output not in serialized
    assert output_event["sensitive_content_redacted"] is True
    assert output_event["raw_output_chars"] > 0


@pytest.mark.asyncio
async def test_invalid_output_mode_does_not_retry_or_call_llm(monkeypatch):
    mock_get_llm = MagicMock()
    monkeypatch.setattr(so, "get_node_llm", mock_get_llm)
    monkeypatch.setattr(so, "get_llm_call_max_retries", lambda _node: 2)

    with pytest.raises(so.StructuredOutputError) as exc_info:
        await so.invoke_structured_llm(
            node_name="unit_structured_retry",
            llm_node="unit_structured_retry",
            schema=DemoStructuredOutput,
            messages=[HumanMessage(content="make json")],
            output_mode="not_a_mode",
        )

    assert len(exc_info.value.result.attempts) == 1
    assert exc_info.value.result.retry_count == 0
    mock_get_llm.assert_not_called()


@pytest.mark.asyncio
async def test_unimplemented_output_mode_does_not_retry_llm_call(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock()
    monkeypatch.setattr(so, "get_node_llm", lambda _node: mock_llm)
    monkeypatch.setattr(so, "get_llm_call_max_retries", lambda _node: 2)

    with pytest.raises(so.StructuredOutputError) as exc_info:
        await so.invoke_structured_llm(
            node_name="unit_structured_retry",
            llm_node="unit_structured_retry",
            schema=DemoStructuredOutput,
            messages=[HumanMessage(content="make json")],
            output_mode="constrained_decoding",
        )

    assert len(exc_info.value.result.attempts) == 1
    assert exc_info.value.result.retry_count == 0
    mock_llm.ainvoke.assert_not_called()
