"""Render selected ContextItems into an internal context block."""

from __future__ import annotations

from src.context_engineering.schema import ContextItem, sanitize_error_message

_MAX_TITLE_CHARS = 120


def render_selected_context(items: list[ContextItem]) -> str:
    """Render selected items for internal shadow inspection only."""
    lines = ["<CONTEXT_PACK>"]
    for item in items:
        title = sanitize_error_message(item.title, max_chars=_MAX_TITLE_CHARS)
        content = _redact_content(item.content)
        lines.append(f"[{item.source_type}] {title}")
        if content:
            lines.append(content)
        lines.append("")
    lines.append("</CONTEXT_PACK>")
    return "\n".join(lines)


def _redact_content(content: object) -> str:
    text = str(content or "")
    if not text:
        return ""
    return sanitize_error_message(text, max_chars=max(len(text), 1))
