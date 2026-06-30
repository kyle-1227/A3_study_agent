"""Context provider for already-built chat messages."""

from __future__ import annotations

import hashlib
from typing import Any

from src.context_engineering.itemizer import make_context_item
from src.context_engineering.providers.base import ProviderContext
from src.context_engineering.schema import ContextItem
from src.context_engineering.tokenizer import message_content_to_text


class MessageContextProvider:
    """Objectize current and recent messages without changing prompt assembly."""

    name = "message_provider"
    source_type = "message"

    def collect(self, context: ProviderContext) -> list[ContextItem]:
        if context.max_items_per_provider <= 0:
            return []
        items: list[ContextItem] = []
        messages = list(context.messages or [])
        current_query, current_index = _current_query(context, messages)
        if current_query:
            items.append(
                make_context_item(
                    source_type="message",
                    title="current_user_query",
                    content=current_query,
                    priority=100,
                    scope="turn",
                    lifetime="turn",
                    compressible=False,
                    can_drop=False,
                    disclosure_level="full",
                    metadata={
                        "role": "user",
                        "message_index": current_index,
                        "kind": "current_user_query",
                        "request_id": context.request_id or "",
                        "thread_id": context.thread_id or "",
                        "content_hash": _content_hash(current_query),
                    },
                    max_content_chars=context.max_content_chars_per_item,
                )
            )
            if len(items) >= context.max_items_per_provider:
                return items

        for index, message in enumerate(messages):
            if len(items) >= context.max_items_per_provider:
                break
            content = message_content_to_text(message)
            if not content.strip():
                continue
            if current_index is not None and index == current_index:
                continue
            role = _message_role(message)
            items.append(
                make_context_item(
                    source_type="message",
                    title=f"recent_{role}_message_{index}",
                    content=content,
                    priority=80 if role == "user" else 70,
                    scope="session",
                    lifetime="session",
                    compressible=True,
                    can_drop=True,
                    disclosure_level="snippet",
                    metadata={
                        "role": role,
                        "message_index": index,
                        "kind": "recent_message",
                    },
                    max_content_chars=context.max_content_chars_per_item,
                )
            )
        return items


def _current_query(
    context: ProviderContext,
    messages: list[Any],
) -> tuple[str, int | None]:
    if context.user_query and context.user_query.strip():
        current_query = context.user_query.strip()
        current_index = _valid_message_index(
            context.current_user_message_index,
            messages,
        )
        if current_index is None:
            current_index = _matching_user_message_index(current_query, messages)
        return current_query, current_index
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if _message_role(message) == "user":
            return message_content_to_text(message).strip(), index
    return "", None


def _valid_message_index(index: int | None, messages: list[Any]) -> int | None:
    if index is None:
        return None
    if isinstance(index, bool) or not isinstance(index, int):
        return None
    if index < 0 or index >= len(messages):
        return None
    return index


def _matching_user_message_index(query: str, messages: list[Any]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if _message_role(message) != "user":
            continue
        if message_content_to_text(message).strip() == query:
            return index
    return None


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        role = message.get("role")
        return str(role or "message").strip().lower() or "message"
    class_name = type(message).__name__.lower()
    if "human" in class_name:
        return "user"
    if "ai" in class_name or "assistant" in class_name:
        return "assistant"
    if "system" in class_name:
        return "system"
    return "message"
