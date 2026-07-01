"""Fail-fast tests for assessment error classification."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.assessment.classifier import classify_error
from src.assessment.errors import ErrorClassificationFailed
from src.assessment.types import ErrorClassificationStrict, QuizAttemptResult


def _quiz_result(*, is_correct: bool = False) -> QuizAttemptResult:
    return QuizAttemptResult(
        user_id="user-1",
        subject="python",
        topic="list comprehension",
        question="What does [x * 2 for x in xs] do?",
        user_answer="It filters odd numbers",
        correct_answer="It doubles every item",
        is_correct=is_correct,
        knowledge_points=["list comprehension", "iteration"],
    )


@pytest.mark.anyio
async def test_invalid_structured_output_raises(monkeypatch):
    import src.assessment.classifier as classifier

    monkeypatch.setattr(
        classifier,
        "invoke_structured_llm",
        AsyncMock(
            return_value=SimpleNamespace(
                success=False,
                parsed=None,
                failure_phase="validation_error",
                error_type="ValidationError",
                error_message="missing required field api_key=sk-secret-value",
            )
        ),
    )

    with pytest.raises(ErrorClassificationFailed) as exc_info:
        await classify_error(_quiz_result())

    message = str(exc_info.value)
    assert "list comprehension" in message
    assert "validation_error" in message
    assert "ValidationError" in message
    assert "sk-secret-value" not in message


@pytest.mark.anyio
async def test_llm_exception_raises(monkeypatch):
    import src.assessment.classifier as classifier

    monkeypatch.setattr(
        classifier,
        "invoke_structured_llm",
        AsyncMock(
            side_effect=RuntimeError(
                "provider exploded Authorization: Bearer secret-token"
            )
        ),
    )

    with pytest.raises(ErrorClassificationFailed) as exc_info:
        await classify_error(_quiz_result())

    message = str(exc_info.value)
    assert "llm_call_failed" in message
    assert "RuntimeError" in message
    assert "secret-token" not in message


@pytest.mark.anyio
async def test_correct_answer_branch_does_not_call_llm(monkeypatch):
    import src.assessment.classifier as classifier

    mock_invoke = AsyncMock()
    monkeypatch.setattr(classifier, "invoke_structured_llm", mock_invoke)

    result = await classify_error(_quiz_result(is_correct=True))

    mock_invoke.assert_not_awaited()
    assert result.confidence == 1.0
    assert result.concept_gap == ""
    assert result.quiz_knowledge_points == ["list comprehension", "iteration"]


def test_llm_schema_requires_classification_fields():
    with pytest.raises(Exception):
        ErrorClassificationStrict.model_validate({"error_type": "concept"})
