from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.tools.search_tool import sanitize_error_message, search, search_with_diagnostics


def _mock_tool(raw=None, exc: Exception | None = None):
    tool = MagicMock()
    if exc is not None:
        tool.invoke.side_effect = exc
    else:
        tool.invoke.return_value = raw
    return tool


def test_search_with_diagnostics_normalizes_list_results():
    raw = [{"snippet": "body", "title": "Title", "link": "https://example.com"}]
    with patch("src.tools.search_tool.get_search_tool", return_value=_mock_tool(raw)):
        diagnostics = search_with_diagnostics("query")

    assert diagnostics["ok"] is True
    assert diagnostics["raw_type"] == "list"
    assert diagnostics["raw_count"] == 1
    assert diagnostics["result_count"] == 1
    assert diagnostics["results"][0] == {
        "content": "body",
        "title": "Title",
        "url": "https://example.com",
    }


def test_search_with_diagnostics_wraps_normal_string_result():
    with patch("src.tools.search_tool.get_search_tool", return_value=_mock_tool("short answer")):
        diagnostics = search_with_diagnostics("query")

    assert diagnostics["ok"] is True
    assert diagnostics["raw_type"] == "str"
    assert diagnostics["result_count"] == 1
    assert diagnostics["results"][0]["content"] == "short answer"


def test_search_with_diagnostics_does_not_wrap_empty_error_string():
    with patch(
        "src.tools.search_tool.get_search_tool",
        return_value=_mock_tool("No good DuckDuckGo Search Result was found"),
    ):
        diagnostics = search_with_diagnostics("query")

    assert diagnostics["raw_type"] == "str_empty_or_error"
    assert diagnostics["result_count"] == 0
    assert diagnostics["results"] == []


def test_search_with_diagnostics_reports_exception_safely():
    exc = RuntimeError("Authorization: Bearer secret-token Cookie: abc=123 request payload...")
    with patch("src.tools.search_tool.get_search_tool", return_value=_mock_tool(exc=exc)):
        diagnostics = search_with_diagnostics("query")

    assert diagnostics["ok"] is False
    assert diagnostics["error_type"] == "RuntimeError"
    assert "secret-token" not in diagnostics["error_message"]
    assert "abc=123" not in diagnostics["error_message"]


def test_search_with_diagnostics_unknown_type():
    with patch("src.tools.search_tool.get_search_tool", return_value=_mock_tool({"unexpected"})):
        diagnostics = search_with_diagnostics("query")

    assert diagnostics["ok"] is False
    assert diagnostics["error_type"] == "UnexpectedSearchResultType"
    assert diagnostics["results"] == []


def test_search_keeps_legacy_list_interface():
    raw = [{"content": "body", "title": "Title", "url": "https://example.com"}]
    with patch("src.tools.search_tool.get_search_tool", return_value=_mock_tool(raw)):
        assert search("query") == raw


def test_sanitize_error_message_redacts_common_secret_shapes():
    text = sanitize_error_message(
        "api_key=sk-test Authorization: Bearer abc.def Cookie: session=private"
    )

    assert "sk-test" not in text
    assert "abc.def" not in text
    assert "session=private" not in text
