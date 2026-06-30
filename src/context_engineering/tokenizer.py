"""Estimated token counting for the Context Engineering Kernel."""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import BaseModel

from src.config import get_setting
from src.context_engineering.schema import ContextConfigError, TokenCount

_TOKENIZER_MODE = "estimated_mixed"


def count_text_tokens(text: str) -> TokenCount:
    """Estimate text tokens with explicit estimated metadata."""
    mode, estimated = _tokenizer_settings()
    if mode != _TOKENIZER_MODE:
        raise ContextConfigError(
            "tokenizer_mode_unsupported",
            f"context_engineering.tokenizer.mode must be {_TOKENIZER_MODE}",
        )
    if estimated is not True:
        raise ContextConfigError(
            "tokenizer_estimated_invalid",
            "context_engineering.tokenizer.estimated must be true for estimated_mixed",
        )
    if not text:
        return TokenCount(value=0, estimated=True, method=mode)

    cjk_chars = 0
    other_chars = 0
    for char in text:
        if _is_cjk_char(char):
            cjk_chars += 1
        else:
            other_chars += 1
    value = math.ceil(cjk_chars / 1.5) + math.ceil(other_chars / 3.5)
    return TokenCount(value=max(value, 1), estimated=True, method=mode)


def count_messages_tokens(messages: list[Any]) -> TokenCount:
    """Estimate tokens for a list of message-like objects."""
    mode, _estimated = _tokenizer_settings()
    total = 0
    for message in messages or []:
        total += count_text_tokens(message_content_to_text(message)).value
    return TokenCount(value=total, estimated=True, method=mode)


def count_schema_chars(schema: type[BaseModel]) -> int:
    """Return schema manifest size without exposing schema text."""
    try:
        schema_json = json.dumps(
            schema.model_json_schema(), ensure_ascii=False, sort_keys=True
        )
    except Exception as exc:
        raise ContextConfigError(
            "schema_size_unavailable",
            "structured output schema size could not be calculated",
        ) from exc
    return len(schema_json)


def message_content_to_text(value: Any) -> str:
    """Convert message-like content to text for counting only."""
    content = getattr(value, "content", value)
    if isinstance(value, dict):
        content = value.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _tokenizer_settings() -> tuple[str, bool]:
    raw = get_setting("context_engineering")
    if not isinstance(raw, dict):
        raise ContextConfigError(
            "context_engineering_missing",
            "context_engineering config is required",
        )
    if raw.get("enabled") is False:
        raise ContextConfigError(
            "context_engineering_disabled",
            "context engineering telemetry is disabled",
        )
    strict = raw.get("strict")
    if not isinstance(strict, bool):
        raise ContextConfigError(
            "context_engineering_strict_invalid",
            "context_engineering.strict must be a boolean",
        )

    tokenizer = raw.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise ContextConfigError(
            "tokenizer_config_missing",
            "context_engineering.tokenizer config is required",
        )
    mode = tokenizer.get("mode")
    estimated = tokenizer.get("estimated")
    if not isinstance(mode, str) or not mode.strip():
        raise ContextConfigError(
            "tokenizer_mode_missing",
            "context_engineering.tokenizer.mode is required",
        )
    if not isinstance(estimated, bool):
        raise ContextConfigError(
            "tokenizer_estimated_missing",
            "context_engineering.tokenizer.estimated is required",
        )
    return mode.strip(), estimated


def _is_cjk_char(char: str) -> bool:
    return (
        "\u4e00" <= char <= "\u9fff"
        or "\u3400" <= char <= "\u4dbf"
        or "\u3000" <= char <= "\u303f"
    )
