"""Strict JSON generation helpers for resource agents.

This module avoids provider-specific structured-output parameters such as
``response_format``. It asks the chat model for normal text, extracts the first
complete JSON object, and validates it with a Pydantic schema.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ValidationError

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class JSONOutputError(ValueError):
    """Raised when a model response cannot be parsed into the target schema."""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content or "")


def _preview(text: str, limit: int = 800) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "...<truncated>"


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object from model text."""
    start = text.find("{")
    if start < 0:
        raise JSONOutputError("No JSON object start found in model output")

    in_string = False
    escaped = False
    depth = 0

    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                raw_json = text[start : idx + 1]
                try:
                    value = json.loads(raw_json)
                except json.JSONDecodeError as exc:
                    raise JSONOutputError(f"Invalid JSON syntax: {exc}") from exc
                if not isinstance(value, dict):
                    raise JSONOutputError("Extracted JSON value is not an object")
                return value

    raise JSONOutputError("No complete JSON object found in model output")


def validate_json_schema(data: dict[str, Any], schema: type[SchemaT]) -> SchemaT:
    """Validate JSON data with Pydantic v2 or v1."""
    try:
        if hasattr(schema, "model_validate"):
            return schema.model_validate(data)
        return schema.parse_obj(data)
    except ValidationError as exc:
        raise JSONOutputError(f"{schema.__name__} validation failed: {exc}") from exc


async def ainvoke_strict_json(
    llm,
    messages: list[BaseMessage],
    *,
    schema: type[SchemaT],
    node_name: str,
    span=None,
) -> SchemaT:
    """Invoke a normal chat model and parse its response as strict JSON."""
    response = await llm.ainvoke(messages)
    if span is not None:
        span.set_attribute("llm.fallback_used", False)

    text = _content_to_text(getattr(response, "content", response))
    try:
        data = extract_json_object(text)
        return validate_json_schema(data, schema)
    except JSONOutputError as exc:
        raise JSONOutputError(
            f"{node_name} failed to produce valid {schema.__name__}: {exc}; "
            f"raw_output={_preview(text)}"
        ) from exc
