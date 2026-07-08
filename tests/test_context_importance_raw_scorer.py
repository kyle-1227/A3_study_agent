"""Raw importance scorer tests for Phase 3B-2A."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.context_engineering.packing.importance import (
    ContextImportanceError,
    parse_importance_scorer_output,
)


def _mock_llm(
    content: str = '{"scores":[{"item_id":"i1","score":0.8,"reason_code":"useful_memory"}]}',
):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content=content))
    return mock_llm


@pytest.mark.anyio
async def test_raw_scorer_uses_manifest_guard_without_plain_llm(monkeypatch):
    from src.graph import llm as llm_module

    mock_llm = _mock_llm()
    traces: list[str] = []

    async def fail_plain(*_args, **_kwargs):
        raise AssertionError("raw scorer must not call invoke_plain_llm_fail_fast")

    transport_kwargs: dict = {}

    async def fake_transport_retry(operation, **kwargs):
        transport_kwargs.update(kwargs)
        return await operation(), 0

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    monkeypatch.setattr(llm_module, "invoke_plain_llm_fail_fast", fail_plain)
    monkeypatch.setattr(
        llm_module,
        "invoke_with_provider_transport_retry",
        fake_transport_retry,
    )
    monkeypatch.setattr(
        llm_module,
        "emit_a3_trace",
        lambda _logger, stage, _payload, **_kwargs: traces.append(stage),
    )

    scores = await llm_module.invoke_context_importance_scorer_raw(
        llm_node="importance_scorer",
        scorer_messages=[{"role": "user", "content": "internal prompt"}],
        timeout_seconds=1,
    )

    assert scores.scores[0].item_id == "i1"
    mock_llm.ainvoke.assert_awaited_once()
    assert transport_kwargs["llm_input_manifest"]["call_purpose"] == (
        "context_importance_scoring"
    )
    assert "plain_llm_output" not in traces
    assert "context_usage" not in traces
    assert "llm_input_manifest.built" in traces


def test_parse_importance_scorer_output_is_strict():
    with pytest.raises(ContextImportanceError) as exc_info:
        parse_importance_scorer_output(
            '{"scores":[{"item_id":"i1","score":0.8,'
            '"reason_code":"useful_memory","extra":"drift"}]}'
        )

    assert exc_info.value.reason == "context_importance_schema_invalid"


def test_parse_importance_scorer_rejects_unknown_reason_code():
    with pytest.raises(ContextImportanceError) as exc_info:
        parse_importance_scorer_output(
            '{"scores":[{"item_id":"i1","score":0.8,"reason_code":"freeform_reason"}]}'
        )

    assert exc_info.value.reason == "context_importance_schema_invalid"


@pytest.mark.anyio
async def test_raw_scorer_timeout_raises_safe_error(monkeypatch):
    from src.graph import llm as llm_module

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=TimeoutError("api_key=sk-secret"))
    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)

    with pytest.raises(ContextImportanceError) as exc_info:
        await llm_module.invoke_context_importance_scorer_raw(
            llm_node="importance_scorer",
            scorer_messages=[{"role": "user", "content": "internal prompt"}],
            timeout_seconds=1,
        )

    assert exc_info.value.reason == "context_importance_scorer_timed_out"
    assert "api_key" not in exc_info.value.warning.lower()
