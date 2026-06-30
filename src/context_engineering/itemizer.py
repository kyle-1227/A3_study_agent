"""Helpers for building safe ContextItem candidates."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.context_engineering.schema import (
    ContextDisclosureLevel,
    ContextItem,
    ContextLifetime,
    ContextScope,
    ContextSourceType,
    is_sensitive_metadata_key,
    sanitize_error_message,
)
from src.context_engineering.tokenizer import estimate_text_tokens_mixed

_TOKENIZER_MODE = "estimated_mixed"
_MAX_METADATA_STRING_CHARS = 300
_MAX_METADATA_LIST_ITEMS = 20
_MAX_METADATA_DEPTH = 2


def estimate_item_tokens(content: str) -> int:
    """Return a pure estimated_mixed token estimate for item content."""
    return estimate_text_tokens_mixed(content or "")


def sanitize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Return metadata with exact sensitive keys removed and values bounded."""
    safe: dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        normalized_key = str(key or "").strip()
        if not normalized_key or is_sensitive_metadata_key(normalized_key):
            continue
        safe[normalized_key] = _sanitize_metadata_value(value, depth=0)
    return safe


def stable_item_id(
    *,
    source_type: ContextSourceType,
    title: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Build a stable id from safe item identity fields."""
    safe_metadata = sanitize_metadata(metadata)
    raw = json.dumps(
        {
            "source_type": source_type,
            "title": str(title or ""),
            "metadata": safe_metadata,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{source_type}:{digest}"


def make_context_item(
    *,
    source_type: ContextSourceType,
    title: str,
    content: str,
    priority: int,
    scope: ContextScope,
    lifetime: ContextLifetime,
    compressible: bool,
    can_drop: bool,
    disclosure_level: ContextDisclosureLevel,
    metadata: dict[str, Any] | None = None,
    relevance_score: float | None = None,
    recency_score: float | None = None,
    confidence: float | None = None,
    item_id: str | None = None,
    max_content_chars: int | None = None,
) -> ContextItem:
    """Build a ContextItem while enforcing content and metadata safety."""
    safe_metadata = sanitize_metadata(metadata)
    safe_content = str(content or "")
    original_chars = len(safe_content)
    if max_content_chars is not None:
        if isinstance(max_content_chars, bool) or not isinstance(
            max_content_chars, int
        ):
            raise ValueError("max_content_chars must be an integer")
        if max_content_chars < 0:
            raise ValueError("max_content_chars must be >= 0")
        if original_chars > max_content_chars:
            safe_content = safe_content[:max_content_chars]
            safe_metadata["content_truncated"] = True
            safe_metadata["original_content_chars"] = original_chars

    return ContextItem(
        id=item_id
        or stable_item_id(
            source_type=source_type,
            title=title,
            metadata=safe_metadata,
        ),
        source_type=source_type,
        title=str(title or ""),
        content=safe_content,
        token_estimate=estimate_item_tokens(safe_content),
        estimated=True,
        tokenizer_mode=_TOKENIZER_MODE,
        priority=priority,
        relevance_score=relevance_score,
        recency_score=recency_score,
        confidence=confidence,
        scope=scope,
        lifetime=lifetime,
        compressible=compressible,
        can_drop=can_drop,
        disclosure_level=disclosure_level,
        metadata=safe_metadata,
    )


def _sanitize_metadata_value(value: Any, *, depth: int) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return sanitize_error_message(value, max_chars=_MAX_METADATA_STRING_CHARS)
    if depth >= _MAX_METADATA_DEPTH:
        return sanitize_error_message(value, max_chars=_MAX_METADATA_STRING_CHARS)
    if isinstance(value, list):
        return [
            _sanitize_metadata_value(item, depth=depth + 1)
            for item in value[:_MAX_METADATA_LIST_ITEMS]
        ]
    if isinstance(value, tuple):
        return [
            _sanitize_metadata_value(item, depth=depth + 1)
            for item in value[:_MAX_METADATA_LIST_ITEMS]
        ]
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key or "").strip()
            if not normalized_key or is_sensitive_metadata_key(normalized_key):
                continue
            safe[normalized_key] = _sanitize_metadata_value(
                item,
                depth=depth + 1,
            )
        return safe
    return sanitize_error_message(value, max_chars=_MAX_METADATA_STRING_CHARS)
