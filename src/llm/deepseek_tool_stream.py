"""Strict parsing for DeepSeek streamed tool-call deltas."""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass
from typing import Any


class DeepSeekToolStreamProtocolError(RuntimeError):
    """Raised when a streamed DeepSeek envelope violates the strict contract."""


@dataclass(frozen=True)
class DeepSeekToolStreamResult:
    tool_name: str
    arguments: str
    finish_reason: str
    usage: dict[str, int]


async def consume_deepseek_tool_stream(
    lines: AsyncIterable[str],
    *,
    expected_tool_name: str,
    on_arguments_delta: Callable[[str], None],
) -> DeepSeekToolStreamResult:
    """Consume provider SSE lines and expose only function-argument fragments."""

    if not expected_tool_name:
        raise DeepSeekToolStreamProtocolError("expected_tool_name is required")
    tool_name = ""
    argument_parts: list[str] = []
    finish_reason = ""
    usage: dict[str, int] = {}
    done_seen = False

    async for raw_line in lines:
        line = str(raw_line).strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        raw_data = line[5:].strip()
        if raw_data == "[DONE]":
            done_seen = True
            break
        try:
            envelope = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            raise DeepSeekToolStreamProtocolError(
                "DeepSeek stream contained invalid JSON"
            ) from exc
        if not isinstance(envelope, dict):
            raise DeepSeekToolStreamProtocolError(
                "DeepSeek stream envelope must be an object"
            )
        _merge_usage(usage, envelope.get("usage"))
        choices = envelope.get("choices")
        if not isinstance(choices, list):
            raise DeepSeekToolStreamProtocolError(
                "DeepSeek stream choices must be a list"
            )
        if not choices:
            continue
        if len(choices) != 1 or not isinstance(choices[0], dict):
            raise DeepSeekToolStreamProtocolError(
                "DeepSeek strict tool stream requires exactly one choice"
            )
        choice = choices[0]
        raw_finish_reason = choice.get("finish_reason")
        if raw_finish_reason is not None:
            finish_reason = str(raw_finish_reason)
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        tool_calls = delta.get("tool_calls")
        if tool_calls is None:
            continue
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            raise DeepSeekToolStreamProtocolError(
                "DeepSeek strict tool stream requires one tool-call delta"
            )
        tool_call = tool_calls[0]
        if not isinstance(tool_call, dict):
            raise DeepSeekToolStreamProtocolError("tool-call delta must be an object")
        index = tool_call.get("index", 0)
        if isinstance(index, bool) or not isinstance(index, int) or index != 0:
            raise DeepSeekToolStreamProtocolError(
                "DeepSeek strict tool stream requires tool index 0"
            )
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        name_delta = function.get("name")
        if name_delta is not None:
            tool_name += str(name_delta)
            if not expected_tool_name.startswith(tool_name):
                raise DeepSeekToolStreamProtocolError(
                    "DeepSeek stream returned an unexpected tool name"
                )
        arguments_delta = function.get("arguments")
        if arguments_delta is not None:
            fragment = str(arguments_delta)
            argument_parts.append(fragment)
            on_arguments_delta(fragment)

    if not done_seen:
        raise DeepSeekToolStreamProtocolError(
            "DeepSeek stream ended before the [DONE] marker"
        )
    if tool_name != expected_tool_name:
        raise DeepSeekToolStreamProtocolError(
            "DeepSeek stream did not complete the expected tool name"
        )
    arguments = "".join(argument_parts)
    if not arguments:
        raise DeepSeekToolStreamProtocolError(
            "DeepSeek stream returned empty tool arguments"
        )
    if not finish_reason:
        raise DeepSeekToolStreamProtocolError(
            "DeepSeek stream did not provide a finish reason"
        )
    return DeepSeekToolStreamResult(
        tool_name=tool_name,
        arguments=arguments,
        finish_reason=finish_reason,
        usage=usage,
    )


def _merge_usage(target: dict[str, int], value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise DeepSeekToolStreamProtocolError("DeepSeek stream usage must be an object")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = value.get(key)
        if item is None:
            continue
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise DeepSeekToolStreamProtocolError(
                f"DeepSeek stream usage.{key} must be a non-negative integer"
            )
        target[key] = item


__all__ = [
    "DeepSeekToolStreamProtocolError",
    "DeepSeekToolStreamResult",
    "consume_deepseek_tool_stream",
]
