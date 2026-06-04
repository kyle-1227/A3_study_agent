"""Tavily Web Search tool with provider diagnostics.

The public interfaces stay provider-agnostic:
``search_with_diagnostics()`` returns rich diagnostics and ``search()`` returns
only normalized result dictionaries. Tavily is the only provider; failures
return empty results and diagnostics instead of falling back.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

import httpx

from src.config import get_setting

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)(cookie\s*[:=]\s*)[^;\n]+"),
    re.compile(r"(?i)(x-api-key\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"tvly-[A-Za-z0-9_-]+"),
)


# TEMP A3_TRACE: remove after diagnostics validation.
def sanitize_error_message(message: Any, max_chars: int = 500) -> str:
    """Redact common secrets and truncate diagnostic error text."""
    text = str(message or "").replace("\n", " ").strip()
    for pattern in _SECRET_PATTERNS:
        if pattern.pattern.startswith("tvly-"):
            text = pattern.sub("tvly-[REDACTED]", text)
        else:
            text = pattern.sub(r"\1[REDACTED]", text)
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def _setting(path: str, default: Any) -> Any:
    return get_setting(f"web_search.{path}", default)


def _tavily_api_key_env() -> str:
    return str(_setting("tavily.api_key_env", "TAVILY_API_KEY") or "TAVILY_API_KEY")


def _tavily_api_key() -> str:
    return os.getenv(_tavily_api_key_env(), "").strip()


def _timeout_seconds(timeout_seconds: float | None) -> float:
    if timeout_seconds is not None:
        return max(1.0, float(timeout_seconds))
    try:
        return max(1.0, float(_setting("timeout_seconds", 8)))
    except (TypeError, ValueError):
        return 8.0


def _max_results(max_results: int | None) -> int:
    if max_results is not None:
        return max(1, int(max_results))
    try:
        return max(1, int(_setting("tavily.max_results", 5)))
    except (TypeError, ValueError):
        return 5


def _empty_diagnostics(
    *,
    query: str,
    original_user_query: str,
    subject: str,
    role: str,
    purpose: str,
    ok: bool = False,
    error_type: str = "",
    error_message: str = "",
    status_code: int | None = None,
    elapsed_ms: float | None = None,
) -> dict[str, Any]:
    return {
        "provider": "tavily",
        "query": query,
        "original_user_query": original_user_query,
        "subject": subject,
        "role": role,
        "purpose": purpose,
        "ok": ok,
        "results": [],
        "result_count": 0,
        "raw_type": "",
        "raw_count": 0,
        "elapsed_ms": elapsed_ms,
        "response_time": None,
        "usage_credits": None,
        "error_type": error_type,
        "error_message": sanitize_error_message(error_message),
        "status_code": status_code,
    }


def _request_payload(query: str, max_results: int) -> dict[str, Any]:
    return {
        "query": query,
        "search_depth": _setting("tavily.search_depth", "basic"),
        "max_results": max_results,
        "include_answer": bool(_setting("tavily.include_answer", False)),
        "include_raw_content": bool(_setting("tavily.include_raw_content", False)),
        "include_images": bool(_setting("tavily.include_images", False)),
        "include_favicon": bool(_setting("tavily.include_favicon", True)),
        "topic": _setting("tavily.topic", "general"),
        "include_usage": bool(_setting("tavily.include_usage", True)),
    }


def _normalize_tavily_result(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": item.get("content", "") or "",
        "title": item.get("title", "") or "",
        "url": item.get("url", "") or "",
        "score": item.get("score"),
        "raw_content": item.get("raw_content"),
        "favicon": item.get("favicon", "") or "",
        "raw": item,
    }


def _usage_credits(raw: dict[str, Any]) -> Any:
    usage = raw.get("usage")
    if isinstance(usage, dict):
        return usage.get("credits")
    return None


def search_with_diagnostics(
    query: str,
    *,
    original_user_query: str = "",
    subject: str = "",
    role: str = "",
    purpose: str = "",
    max_results: int | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Execute Tavily Search and return normalized results plus diagnostics."""
    query = str(query or "").strip()
    original_user_query = str(original_user_query or "")
    subject = str(subject or "")
    role = str(role or "")
    purpose = str(purpose or "")
    timeout = _timeout_seconds(timeout_seconds)
    result_limit = _max_results(max_results)
    started = time.perf_counter()

    api_key = _tavily_api_key()
    if not api_key:
        return _empty_diagnostics(
            query=query,
            original_user_query=original_user_query,
            subject=subject,
            role=role,
            purpose=purpose,
            error_type="MissingApiKey",
            error_message=f"{_tavily_api_key_env()} is not configured",
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                TAVILY_SEARCH_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=_request_payload(query, result_limit),
            )
            status_code = response.status_code
            response.raise_for_status()
            raw = response.json()
    except httpx.TimeoutException:
        return _empty_diagnostics(
            query=query,
            original_user_query=original_user_query,
            subject=subject,
            role=role,
            purpose=purpose,
            error_type="TimeoutError",
            error_message=f"tavily search exceeded {timeout}s",
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        )
    except httpx.HTTPStatusError as exc:
        return _empty_diagnostics(
            query=query,
            original_user_query=original_user_query,
            subject=subject,
            role=role,
            purpose=purpose,
            error_type=type(exc).__name__,
            error_message=exc.response.text or str(exc),
            status_code=exc.response.status_code,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        )
    except Exception as exc:
        return _empty_diagnostics(
            query=query,
            original_user_query=original_user_query,
            subject=subject,
            role=role,
            purpose=purpose,
            error_type=type(exc).__name__,
            error_message=str(exc),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    results_raw = raw.get("results") if isinstance(raw, dict) else []
    results = [
        _normalize_tavily_result(item)
        for item in (results_raw or [])
        if isinstance(item, dict)
    ]
    return {
        "provider": "tavily",
        "query": query,
        "original_user_query": original_user_query,
        "subject": subject,
        "role": role,
        "purpose": purpose,
        "ok": True,
        "results": results,
        "result_count": len(results),
        "raw_type": type(raw).__name__,
        "raw_count": len(results_raw or []) if isinstance(results_raw, list) else 0,
        "elapsed_ms": elapsed_ms,
        "response_time": raw.get("response_time") if isinstance(raw, dict) else None,
        "usage_credits": _usage_credits(raw) if isinstance(raw, dict) else None,
        "error_type": "",
        "error_message": "",
        "status_code": status_code,
    }


def search(query: str) -> list[dict]:
    """Execute Tavily Search and return normalized results only."""
    return search_with_diagnostics(query).get("results", [])
