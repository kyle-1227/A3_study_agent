"""Unit tests for SubGraph B — Study Planner nodes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.planner import search_policy


class TestSearchPolicy:

    @patch("src.graph.planner.web_search_fn")
    async def test_returns_search_results(self, mock_search):
        mock_search.return_value = [
            {"content": "2026年高考6月7日", "title": "高考时间", "url": "https://example.com"},
        ]

        state = {"messages": [HumanMessage(content="帮我做复习计划")]}
        result = await search_policy(state)

        assert "search_results" in result
        assert len(result["search_results"]) == 1
        mock_search.assert_called_once()

    @patch("src.graph.planner.web_search_fn", side_effect=Exception("timeout"))
    async def test_returns_empty_on_exception(self, mock_search):
        state = {"messages": [HumanMessage(content="test")]}
        result = await search_policy(state)

        assert result["search_results"] == []

    @patch("src.graph.planner.web_search_fn")
    async def test_query_contains_current_year(self, mock_search):
        mock_search.return_value = []
        from datetime import datetime

        state = {"messages": [HumanMessage(content="test")]}
        await search_policy(state)

        call_args = mock_search.call_args[0][0]
        assert str(datetime.now().year) in call_args


