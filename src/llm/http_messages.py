"""OpenAI-compatible HTTP message normalization helpers."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage


class _InvalidHttpMessageFormatError(ValueError):
    """Raised before provider calls when HTTP chat messages are malformed."""


def _message_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(value)


def to_openai_message(message: Any) -> dict[str, Any]:
    """Normalize LangChain or dict messages to OpenAI-compatible role/content."""
    if isinstance(message, dict):
        role = str(message.get("role") or "").strip()
        if not role and message.get("type"):
            role = {
                "system": "system",
                "human": "user",
                "ai": "assistant",
                "assistant": "assistant",
                "user": "user",
                "tool": "tool",
            }.get(str(message.get("type") or "").strip().lower(), "")
        result = {
            "role": role or "user",
            "content": _message_text(message.get("content", "")),
        }
        if result["role"] == "tool" and message.get("tool_call_id"):
            result["tool_call_id"] = str(message.get("tool_call_id"))
        return result

    message_type = str(getattr(message, "type", "") or "").strip().lower()
    role = {
        "system": "system",
        "human": "user",
        "ai": "assistant",
        "assistant": "assistant",
        "user": "user",
        "tool": "tool",
    }.get(message_type, "")
    if not role:
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            role = "user"
    return {
        "role": role,
        "content": _message_text(getattr(message, "content", message)),
    }


def normalize_openai_messages(messages: list[Any]) -> list[dict[str, Any]]:
    return [to_openai_message(message) for message in messages or []]


def validate_openai_messages(messages: list[dict[str, Any]]) -> None:
    allowed_roles = {"system", "user", "assistant", "tool"}
    if not isinstance(messages, list) or not messages:
        raise _InvalidHttpMessageFormatError("invalid_http_message_format: messages must be a non-empty list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise _InvalidHttpMessageFormatError(
                f"invalid_http_message_format: messages[{index}] must be an object"
            )
        role = message.get("role")
        if role not in allowed_roles:
            raise _InvalidHttpMessageFormatError(
                f"invalid_http_message_format: messages[{index}].role={role!r}"
            )
        if "content" not in message:
            raise _InvalidHttpMessageFormatError(
                f"invalid_http_message_format: messages[{index}].content missing"
            )
        content = message.get("content")
        if not isinstance(content, (str, list)):
            raise _InvalidHttpMessageFormatError(
                f"invalid_http_message_format: messages[{index}].content must be str or list"
            )


def preview_openai_messages(
    messages: list[dict[str, Any]],
    *,
    max_preview_chars: int = 80,
    max_messages: int = 8,
) -> list[dict[str, Any]]:
    """Return a compact prompt-safe preview; never include full message content."""
    previews: list[dict[str, Any]] = []
    for index, message in enumerate((messages or [])[:max_messages]):
        content = _message_text(message.get("content", ""))
        if 0 < len(content) <= max_preview_chars:
            preview = "[short message omitted]"
        else:
            preview = content[:max_preview_chars]
        previews.append(
            {
                "index": index,
                "role": message.get("role", ""),
                "content_chars": len(content),
                "content_preview": preview + ("..." if len(content) > max_preview_chars else ""),
            }
        )
    return previews
