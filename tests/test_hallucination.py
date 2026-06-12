"""Unit tests for hallucination evaluation and retry routing."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.academic import (
    MAX_RETRIES,
    HallucinationEvaluation,
    evaluate_hallucination,
    generate_answer,
    should_retry_or_end,
)


class TestHallucinationEvalSchema:
    def test_faithful_evaluation(self):
        evaluation = HallucinationEvaluation(is_faithful=True, reason="grounded")
        assert evaluation.is_faithful is True
        assert evaluation.reason == "grounded"

    def test_unfaithful_evaluation(self):
        evaluation = HallucinationEvaluation(is_faithful=False, reason="fabricated")
        assert evaluation.is_faithful is False


def _structured_result(is_faithful: bool, reason: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(
        parsed=HallucinationEvaluation(is_faithful=is_faithful, reason=reason),
        raw_output='{"is_faithful": true}',
    )


class TestEvaluateHallucinationNode:
    @patch("src.graph.academic.invoke_structured_llm", new_callable=AsyncMock)
    async def test_faithful_answer_not_flagged(self, mock_invoke):
        mock_invoke.return_value = _structured_result(True, "Good")

        result = await evaluate_hallucination({
            "messages": [HumanMessage(content="question"), AIMessage(content="answer")],
            "context": [{"content": "context"}],
            "retry_count": 0,
        })

        assert result == {"hallucination_detected": False}
        mock_invoke.assert_awaited_once()

    @patch("src.graph.academic.invoke_structured_llm", new_callable=AsyncMock)
    async def test_unfaithful_answer_detected(self, mock_invoke):
        mock_invoke.return_value = _structured_result(False, "Fabricated")

        result = await evaluate_hallucination({
            "messages": [HumanMessage(content="q"), AIMessage(content="bad")],
            "context": [],
            "retry_count": 0,
        })

        assert result["hallucination_detected"] is True
        assert result["retry_count"] == 1
        assert result["hallucination_reason"] == "Fabricated"

    @patch("src.graph.academic.invoke_structured_llm", new_callable=AsyncMock)
    async def test_increments_retry_count(self, mock_invoke):
        mock_invoke.return_value = _structured_result(False, "Off-topic")

        result = await evaluate_hallucination({
            "messages": [HumanMessage(content="q"), AIMessage(content="bad")],
            "context": [],
            "retry_count": 1,
        })

        assert result["retry_count"] == 2

    @patch("src.graph.academic.invoke_structured_llm", new_callable=AsyncMock)
    async def test_failure_raises_in_fail_fast_mode(self, mock_invoke):
        mock_invoke.side_effect = RuntimeError("hallucination evaluator failed")

        with pytest.raises(RuntimeError, match="hallucination evaluator failed"):
            await evaluate_hallucination({
                "messages": [HumanMessage(content="q"), AIMessage(content="a")],
                "context": [],
                "retry_count": 0,
            })

    @patch("src.graph.academic.invoke_structured_llm", new_callable=AsyncMock)
    async def test_extracts_last_human_message_as_question(self, mock_invoke):
        mock_invoke.return_value = _structured_result(True, "OK")

        await evaluate_hallucination({
            "messages": [
                HumanMessage(content="the real question"),
                AIMessage(content="first attempt"),
                AIMessage(content="second attempt"),
            ],
            "context": [],
            "retry_count": 1,
        })

        messages = mock_invoke.await_args.kwargs["messages"]
        prompt_text = messages[-1].content
        assert "the real question" in prompt_text
        assert "second attempt" in prompt_text


class TestShouldRetryOrEnd:
    def test_routes_end_when_valid(self):
        assert should_retry_or_end({"hallucination_detected": False, "retry_count": 0}) == "end"

    def test_routes_retry_first_attempt(self):
        assert should_retry_or_end({"hallucination_detected": True, "retry_count": 1}) == "retry"

    def test_routes_retry_at_max(self):
        assert should_retry_or_end({"hallucination_detected": True, "retry_count": MAX_RETRIES}) == "retry"

    def test_routes_end_past_max(self):
        assert should_retry_or_end({"hallucination_detected": True, "retry_count": MAX_RETRIES + 1}) == "end"

    def test_defaults_end_when_no_flag(self):
        assert should_retry_or_end({"retry_count": 0}) == "end"


class TestGenerateAnswerRetryCompat:
    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    async def test_uses_last_human_message_not_last_message(
        self, mock_get_llm, mock_get_fallback, mock_llm_response,
    ):
        mock_llm = type("LLM", (), {})()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response("new answer"))
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = None

        await generate_answer({
            "messages": [
                HumanMessage(content="What is the discriminant?"),
                AIMessage(content="Previous bad answer"),
            ],
            "context": [],
        })

        call_args = mock_llm.ainvoke.call_args[0][0]
        human_msgs = [m for m in call_args if isinstance(m, HumanMessage)]
        prompt_text = " ".join(m.content for m in human_msgs)
        assert "What is the discriminant?" in prompt_text
        assert "Previous bad answer" not in prompt_text
