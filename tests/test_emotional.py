"""Unit tests for the Emotional Response node."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.emotional import emotional_response


class TestEmotionalResponse:

    @patch("src.graph.emotional.invoke_plain_llm_fail_fast")
    async def test_returns_ai_message(self, mock_invoke_plain, mock_llm_response):
        mock_invoke_plain.return_value = "同学你好，感到焦虑是很正常的..."

        state = {"messages": [HumanMessage(content="我好焦虑")]}
        result = await emotional_response(state)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert len(result["messages"][0].content) > 0

    @patch("src.graph.emotional.invoke_plain_llm_fail_fast")
    async def test_passes_full_history(self, mock_invoke_plain, mock_llm_response):
        mock_invoke_plain.return_value = "response"

        msgs = [
            HumanMessage(content="你好"),
            AIMessage(content="你好！"),
            HumanMessage(content="我压力好大"),
        ]
        state = {"messages": msgs}
        await emotional_response(state)

        call_args = mock_invoke_plain.await_args.kwargs["messages"]
        # First message is SystemMessage (prompt), then the 3 history messages
        assert len(call_args) == 4

    @patch("src.graph.emotional.invoke_plain_llm_fail_fast")
    async def test_system_prompt_included(self, mock_invoke_plain, mock_llm_response):
        from langchain_core.messages import SystemMessage

        mock_invoke_plain.return_value = "response"

        state = {"messages": [HumanMessage(content="test")]}
        await emotional_response(state)

        call_args = mock_invoke_plain.await_args.kwargs["messages"]
        assert isinstance(call_args[0], SystemMessage)
        assert "学业发展导师" in call_args[0].content
