"""Unit tests for tool wrappers (search_tool, rag_tool)."""

from __future__ import annotations

import httpx

from src.tools.search_tool import search


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


class TestSearchFunction:

    def test_returns_normalized_results(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        _patch_client(
            monkeypatch,
            response=httpx.Response(
                200,
                json={
                    "results": [
                        {"content": "content1", "title": "title1", "url": "url1"},
                        {"content": "content2", "title": "title2", "url": "url2"},
                    ]
                },
            ),
        )

        results = search("test query")

        assert len(results) == 2
        assert results[0]["content"] == "content1"
        assert results[0]["title"] == "title1"
        assert results[0]["url"] == "url1"

    def test_returns_empty_without_api_key(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        results = search("test query")

        assert results == []

    def test_returns_empty_on_exception(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        _patch_client(monkeypatch, exc=RuntimeError("API error"))

        results = search("test query")

        assert results == []

    def test_handles_empty_results(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        _patch_client(monkeypatch, response=httpx.Response(200, json={"results": []}))

        results = search("test")

        assert results == []


class TestPrompts:
    """Verify prompt templates are well-formed and contain expected placeholders."""

    def test_supervisor_prompt_not_empty(self):
        from src.config import load_prompt
        prompt = load_prompt("supervisor_system")
        assert len(prompt) > 100
        assert "academic" in prompt
        assert "planning" in prompt
        assert "emotional" in prompt

    def test_academic_prompts_have_placeholders(self):
        from src.config import load_prompt
        answer_prompt = load_prompt("academic_answer")
        system_prompt = load_prompt("academic_system")
        assert "{retrieved_context}" in answer_prompt
        assert "{search_context}" in answer_prompt
        assert "{question}" in answer_prompt
        assert "{resource_offer_instruction}" in answer_prompt
        assert len(system_prompt) > 50

    def test_planner_prompts_have_placeholders(self):
        from src.config import load_prompt
        generate_prompt = load_prompt("planner_generate")
        system_prompt = load_prompt("planner_system")
        assert "{user_request}" in generate_prompt
        assert "{planning_context}" in generate_prompt
        assert len(system_prompt) > 50

    def test_emotional_prompt_not_empty(self):
        from src.config import load_prompt
        prompt = load_prompt("emotional_system")
        assert len(prompt) > 50
        assert "学业发展导师" in prompt
