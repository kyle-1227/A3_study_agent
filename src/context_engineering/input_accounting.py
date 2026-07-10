"""Pure, content-free accounting for exact provider-bound messages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from langchain_core.messages import BaseMessage

from src.context_engineering.tokenizer import estimate_text_tokens_mixed

TOKENIZER_MODE = "estimated_mixed"


@dataclass(frozen=True)
class AccountedMessage:
    index: int
    role: str
    char_count: int
    estimated_tokens: int
    fingerprint: str
    contains_injected_context: bool
    contains_capability_context: bool


@dataclass(frozen=True)
class LLMInputAccounting:
    message_count: int
    prompt_chars: int
    input_estimated_tokens: int
    message_fingerprint: str
    tokenizer_mode: str
    messages: tuple[AccountedMessage, ...]


def build_llm_input_accounting(messages: list[Any]) -> LLMInputAccounting:
    """Count and fingerprint each message exactly once without retaining content."""
    accounted: list[AccountedMessage] = []
    identity: list[dict[str, Any]] = []
    for index, message in enumerate(messages or []):
        role = message_role(message)
        content = message_content(message)
        fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()
        item = AccountedMessage(
            index=index,
            role=role,
            char_count=len(content),
            estimated_tokens=estimate_text_tokens_mixed(content),
            fingerprint=fingerprint,
            contains_injected_context="<INJECTED_CONTEXT>" in content,
            contains_capability_context="<CAPABILITY_CONTEXT>" in content,
        )
        accounted.append(item)
        identity.append(
            {
                "index": index,
                "role": role,
                "char_count": item.char_count,
                "estimated_tokens": item.estimated_tokens,
                "fingerprint": fingerprint,
            }
        )
    message_fingerprint = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return LLMInputAccounting(
        message_count=len(accounted),
        prompt_chars=sum(item.char_count for item in accounted),
        input_estimated_tokens=sum(item.estimated_tokens for item in accounted),
        message_fingerprint=message_fingerprint,
        tokenizer_mode=TOKENIZER_MODE,
        messages=tuple(accounted),
    )


def message_role(message: Any) -> str:
    if isinstance(message, Mapping):
        return str(message.get("role") or message.get("type") or "unknown").strip()
    if isinstance(message, BaseMessage):
        return str(getattr(message, "type", "unknown") or "unknown").strip()
    return "unknown"


def message_content(message: Any) -> str:
    if isinstance(message, Mapping):
        return content_text(message.get("content"))
    if isinstance(message, BaseMessage):
        return content_text(getattr(message, "content", ""))
    return str(message or "")


def content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)
