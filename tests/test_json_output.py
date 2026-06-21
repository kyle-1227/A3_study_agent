"""Tests for strict JSON output parsing helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from src.graph import json_output as json_output_module
from src.graph.json_output import (
    JSONOutputError,
    ainvoke_strict_json,
    extract_json_object,
    validate_json_schema,
)


class DemoArtifact(BaseModel):
    title: str
    count: int


def test_extract_json_object_from_plain_json():
    assert extract_json_object('{"title": "demo", "count": 2}') == {"title": "demo", "count": 2}


def test_extract_json_object_from_code_fence():
    text = '```json\n{"title": "demo", "count": 2}\n```'
    assert extract_json_object(text) == {"title": "demo", "count": 2}


def test_extract_json_object_from_wrapped_text():
    text = 'Here is the result:\n{"title": "demo", "count": 2}\nDone.'
    assert extract_json_object(text) == {"title": "demo", "count": 2}


def test_validate_json_schema_accepts_valid_data():
    result = validate_json_schema({"title": "demo", "count": 2}, DemoArtifact)
    assert result.title == "demo"
    assert result.count == 2


def test_validate_json_schema_rejects_missing_fields():
    with pytest.raises(JSONOutputError) as exc_info:
        validate_json_schema({"title": "demo"}, DemoArtifact)
    assert "DemoArtifact validation failed" in str(exc_info.value)


def test_extract_json_object_rejects_missing_json():
    with pytest.raises(JSONOutputError) as exc_info:
        extract_json_object("no structured payload")
    assert "No JSON object start found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ainvoke_strict_json_retries_parse_failure_then_succeeds(monkeypatch):
    monkeypatch.setattr(json_output_module, "get_llm_call_max_retries", lambda _node: 2)
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content="no structured payload"),
            AIMessage(content='{"title": "demo", "count": 2}'),
        ]
    )

    result = await ainvoke_strict_json(
        llm,
        [HumanMessage(content="return json")],
        schema=DemoArtifact,
        node_name="json_retry",
    )

    assert result == DemoArtifact(title="demo", count=2)
    assert llm.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_ainvoke_strict_json_retries_validation_failure(monkeypatch):
    monkeypatch.setattr(json_output_module, "get_llm_call_max_retries", lambda _node: 2)
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content='{"title": "missing count"}'),
            AIMessage(content='{"title": "demo", "count": 2}'),
        ]
    )

    result = await ainvoke_strict_json(
        llm,
        [HumanMessage(content="return json")],
        schema=DemoArtifact,
        node_name="json_retry",
    )

    assert result.count == 2
    assert llm.ainvoke.await_count == 2
