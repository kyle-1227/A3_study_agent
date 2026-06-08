from __future__ import annotations

import httpx

from src.tools.search_tool import sanitize_error_message, search, search_with_diagnostics


class _FakeClient:
    def __init__(self, *, response: httpx.Response | None = None, exc: Exception | None = None, **kwargs):
        self.response = response
        self.exc = exc
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, *args, **kwargs):
        if self.exc:
            raise self.exc
        return self.response


def _patch_client(monkeypatch, *, response: httpx.Response | None = None, exc: Exception | None = None):
    monkeypatch.setattr(
        "src.tools.search_tool.httpx.Client",
        lambda **kwargs: _FakeClient(response=response, exc=exc, **kwargs),
    )


def test_search_with_diagnostics_reports_missing_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    diagnostics = search_with_diagnostics("query", original_user_query="original")

    assert diagnostics["provider"] == "tavily"
    assert diagnostics["ok"] is False
    assert diagnostics["error_type"] == "MissingApiKey"
    assert diagnostics["results"] == []
    assert diagnostics["original_user_query"] == "original"


def test_search_with_diagnostics_normalizes_tavily_results(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    response = httpx.Response(
        200,
        json={
            "results": [
                {
                    "title": "Title",
                    "url": "https://example.com",
                    "content": "body",
                    "score": 0.82,
                    "raw_content": None,
                    "favicon": "https://example.com/favicon.ico",
                }
            ],
            "response_time": 1.2,
            "usage": {"credits": 1},
        },
    )
    _patch_client(monkeypatch, response=response)

    diagnostics = search_with_diagnostics(
        "query",
        original_user_query="original",
        subject="python",
        role="core_concept",
        purpose="repair",
    )

    assert diagnostics["ok"] is True
    assert diagnostics["provider"] == "tavily"
    assert diagnostics["result_count"] == 1
    assert diagnostics["response_time"] == 1.2
    assert diagnostics["usage_credits"] == 1
    assert diagnostics["status_code"] == 200
    assert diagnostics["results"][0]["title"] == "Title"
    assert diagnostics["results"][0]["url"] == "https://example.com"
    assert diagnostics["results"][0]["score"] == 0.82
    assert diagnostics["results"][0]["favicon"] == "https://example.com/favicon.ico"
    assert diagnostics["subject"] == "python"
    assert diagnostics["role"] == "core_concept"
    assert diagnostics["purpose"] == "repair"


def test_search_with_diagnostics_reports_http_error(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret-token")
    request = httpx.Request("POST", "https://api.tavily.com/search")
    response = httpx.Response(401, text="api_key=tvly-secret-token invalid", request=request)
    _patch_client(monkeypatch, response=response)

    diagnostics = search_with_diagnostics("query")

    assert diagnostics["ok"] is False
    assert diagnostics["error_type"] == "HTTPStatusError"
    assert diagnostics["status_code"] == 401
    assert "tvly-secret-token" not in diagnostics["error_message"]


def test_search_with_diagnostics_reports_timeout(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    _patch_client(monkeypatch, exc=httpx.TimeoutException("slow"))

    diagnostics = search_with_diagnostics("query", timeout_seconds=2)

    assert diagnostics["ok"] is False
    assert diagnostics["error_type"] == "TimeoutError"
    assert "2" in diagnostics["error_message"]


def test_search_keeps_result_list_interface(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    response = httpx.Response(
        200,
        json={"results": [{"title": "Title", "url": "https://example.com", "content": "body"}]},
    )
    _patch_client(monkeypatch, response=response)

    assert search("query") == [
        {
            "content": "body",
            "title": "Title",
            "url": "https://example.com",
            "score": None,
            "raw_content": None,
            "favicon": "",
            "raw": {"title": "Title", "url": "https://example.com", "content": "body"},
        }
    ]


def test_sanitize_error_message_redacts_common_secret_shapes():
    text = sanitize_error_message(
        "api_key=sk-test Authorization: Bearer abc.def Cookie: session=private tvly-secret"
    )

    assert "sk-test" not in text
    assert "abc.def" not in text
    assert "session=private" not in text
    assert "tvly-secret" not in text
