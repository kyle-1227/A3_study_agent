"""DuckDuckGo Web Search tool for online fallback retrieval.

Provides a lazy-singleton search tool and a convenience ``search()``
function that normalises DuckDuckGo output to a unified schema
(``content``, ``title``, ``url``) consumed by graph nodes.
"""

from __future__ import annotations

import re
import time
from typing import Any

from langchain_community.tools import DuckDuckGoSearchResults

_search_tool: DuckDuckGoSearchResults | None = None

_EMPTY_OR_ERROR_PHRASES = (
    "no good duckduckgo search result was found",
    "no results",
    "error",
    "rate limit",
)

_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)(cookie\s*[:=]\s*)[^;\n]+"),
    re.compile(r"(?i)(x-api-key\s*[:=]\s*)[^\s,;]+"),
)


def get_search_tool() -> DuckDuckGoSearchResults:
    """Lazy singleton — avoids repeated instantiation across graph nodes."""
    global _search_tool
    if _search_tool is None:
        _search_tool = DuckDuckGoSearchResults(
            max_results=3,
            output_format="list",
        )
    return _search_tool


# TEMP A3_TRACE: remove after diagnostics validation.
def sanitize_error_message(message: Any, max_chars: int = 500) -> str:
    """Redact common secrets and truncate diagnostic error text."""
    text = str(message or "").replace("\n", " ").strip()
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(r"\1[REDACTED]", text)
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def _looks_like_empty_or_error_text(text: str) -> bool:
    lowered = text.strip().lower()
    return any(phrase in lowered for phrase in _EMPTY_OR_ERROR_PHRASES)


def _normalize_result(item: dict) -> dict:
    return {
        "content": item.get("snippet", item.get("content", "")),
        "title": item.get("title", ""),
        "url": item.get("link", item.get("url", "")),
    }


def search_with_diagnostics(query: str) -> dict[str, Any]:
    """Execute a web search and return normalized results plus diagnostics."""
    provider = "duckduckgo"
    started = time.perf_counter()
    try:
        raw = get_search_tool().invoke(query)
    except Exception as exc:
        return {
            "provider": provider,
            "query": query,
            "ok": False,
            "results": [],
            "result_count": 0,
            "error_type": type(exc).__name__,
            "error_message": sanitize_error_message(exc),
            "raw_type": "",
            "raw_count": None,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    if isinstance(raw, list):
        results = [_normalize_result(item) for item in raw if isinstance(item, dict)]
        return {
            "provider": provider,
            "query": query,
            "ok": True,
            "results": results,
            "result_count": len(results),
            "error_type": "",
            "error_message": "",
            "raw_type": "list",
            "raw_count": len(raw),
            "elapsed_ms": elapsed_ms,
        }

    if isinstance(raw, str):
        if not raw.strip() or _looks_like_empty_or_error_text(raw):
            lowered = raw.lower()
            is_error = "error" in lowered or "rate limit" in lowered
            return {
                "provider": provider,
                "query": query,
                "ok": not is_error,
                "results": [],
                "result_count": 0,
                "error_type": "SearchProviderMessage" if is_error else "",
                "error_message": sanitize_error_message(raw) if is_error else "",
                "raw_type": "str_empty_or_error",
                "raw_count": None,
                "elapsed_ms": elapsed_ms,
            }
        result = {"content": raw, "title": "", "url": ""}
        return {
            "provider": provider,
            "query": query,
            "ok": True,
            "results": [result],
            "result_count": 1,
            "error_type": "",
            "error_message": "",
            "raw_type": "str",
            "raw_count": None,
            "elapsed_ms": elapsed_ms,
        }

    return {
        "provider": provider,
        "query": query,
        "ok": False,
        "results": [],
        "result_count": 0,
        "error_type": "UnexpectedSearchResultType",
        "error_message": sanitize_error_message(f"Unexpected raw result type: {type(raw).__name__}"),
        "raw_type": type(raw).__name__,
        "raw_count": None,
        "elapsed_ms": elapsed_ms,
    }


def search(query: str) -> list[dict]:
    """Execute a web search and return normalised results.

    DuckDuckGo returns keys ``snippet``, ``title``, ``link``.
    This function remaps them to ``content``, ``title``, ``url``
    so that downstream consumers stay provider-agnostic.

    Args:
        query: The search query string.

    Returns:
        A list of dicts with keys ``content``, ``title``, ``url``.
        Returns an empty list on any failure.
    """
    return search_with_diagnostics(query).get("results", [])
