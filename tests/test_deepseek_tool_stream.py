"""DeepSeek strict tool-call stream protocol tests."""

from __future__ import annotations

import json

import pytest

from src.llm.deepseek_tool_stream import (
    DeepSeekToolStreamProtocolError,
    consume_deepseek_tool_stream,
)


async def _lines(items: list[str]):
    for item in items:
        yield item


def _data(payload: dict) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False)


@pytest.mark.anyio
async def test_consumer_assembles_one_tool_and_streams_only_arguments() -> None:
    deltas: list[str] = []
    result = await consume_deepseek_tool_stream(
        _lines(
            [
                _data(
                    {
                        "choices": [
                            {
                                "finish_reason": None,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {
                                                "name": "qa_agent_QAResponse",
                                                "arguments": '{"answer":"你',
                                            },
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                ),
                _data(
                    {
                        "choices": [
                            {
                                "finish_reason": "tool_calls",
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {
                                                "arguments": '好","grounding_status":"general_knowledge"}',
                                            },
                                        }
                                    ]
                                },
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 4,
                            "total_tokens": 14,
                        },
                    }
                ),
                "data: [DONE]",
            ]
        ),
        expected_tool_name="qa_agent_QAResponse",
        on_arguments_delta=deltas.append,
    )

    assert result.arguments == (
        '{"answer":"你好","grounding_status":"general_knowledge"}'
    )
    assert deltas == [
        '{"answer":"你',
        '好","grounding_status":"general_knowledge"}',
    ]
    assert result.finish_reason == "tool_calls"
    assert result.usage["total_tokens"] == 14


@pytest.mark.anyio
async def test_consumer_rejects_missing_done_marker() -> None:
    with pytest.raises(DeepSeekToolStreamProtocolError, match="DONE"):
        await consume_deepseek_tool_stream(
            _lines(
                [
                    _data(
                        {
                            "choices": [
                                {
                                    "finish_reason": "tool_calls",
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "function": {
                                                    "name": "qa_agent_QAResponse",
                                                    "arguments": "{}",
                                                },
                                            }
                                        ]
                                    },
                                }
                            ]
                        }
                    )
                ]
            ),
            expected_tool_name="qa_agent_QAResponse",
            on_arguments_delta=lambda _fragment: None,
        )


@pytest.mark.anyio
async def test_consumer_rejects_second_tool_index() -> None:
    with pytest.raises(DeepSeekToolStreamProtocolError, match="index 0"):
        await consume_deepseek_tool_stream(
            _lines(
                [
                    _data(
                        {
                            "choices": [
                                {
                                    "finish_reason": None,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 1,
                                                "function": {
                                                    "name": "qa_agent_QAResponse",
                                                    "arguments": "{}",
                                                },
                                            }
                                        ]
                                    },
                                }
                            ]
                        }
                    ),
                    "data: [DONE]",
                ]
            ),
            expected_tool_name="qa_agent_QAResponse",
            on_arguments_delta=lambda _fragment: None,
        )
