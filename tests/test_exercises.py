"""Tests for fail-fast leveled-exercise resource generation nodes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.exercises import (
    ExerciseArtifact,
    ExerciseItem,
    exercise_agent,
    exercise_output,
    exercise_reviewer,
    should_rewrite_exercise,
    stable_exercise_question_id,
)


def _exercise_item(level: str) -> ExerciseItem:
    return ExerciseItem(
        level=level,
        question=f"{level} question",
        answer=f"{level} answer",
        explanation=f"{level} explanation",
        pitfall=f"{level} pitfall",
        tags=["machine learning"],
    )


def _artifact() -> ExerciseArtifact:
    return ExerciseArtifact(
        title="Machine Learning Exercises",
        items=[
            _exercise_item("foundation"),
            _exercise_item("intermediate"),
            _exercise_item("application"),
            _exercise_item("self_check"),
        ],
    )


@pytest.mark.anyio
async def test_exercise_agent_generates_structured_items():
    artifact = _artifact()
    with patch(
        "src.graph.exercises.invoke_structured_llm",
        return_value=SimpleNamespace(parsed=artifact),
    ):
        result = await exercise_agent(
            {
                "messages": [HumanMessage(content="Create machine learning exercises")],
                "context": [{"content": "Course notes", "source": "ml.md"}],
                "exercise_outline": "foundation, intermediate, application, self_check",
                "exercise_round": 0,
            }
        )

    assert result["exercise_artifact"]["title"] == "Machine Learning Exercises"
    assert result["exercise_round"] == 1
    assert {item["level"] for item in result["exercise_items"]} == {
        "foundation",
        "intermediate",
        "application",
        "self_check",
    }
    assert all(
        item["question_id"].startswith("question:v1:")
        for item in result["exercise_items"]
    )
    assert len({item["question_id"] for item in result["exercise_items"]}) == 4


def test_stable_exercise_question_id_is_order_independent_for_tags():
    first = stable_exercise_question_id(
        level="basic",
        question="What is overfitting?",
        tags=["machine learning", "generalization"],
    )
    second = stable_exercise_question_id(
        level="basic",
        question="What is overfitting?",
        tags=["generalization", "machine learning"],
    )

    assert first == second
    assert first.startswith("question:v1:")


def test_stable_exercise_question_id_rejects_incomplete_identity():
    with pytest.raises(ValueError, match="requires level, question, and tags"):
        stable_exercise_question_id(level="basic", question="", tags=["ml"])


@pytest.mark.anyio
async def test_exercise_agent_empty_outline_raises():
    with pytest.raises(ValueError, match="outline"):
        await exercise_agent({"exercise_outline": ""})


@pytest.mark.anyio
async def test_exercise_reviewer_rejects_incomplete_items():
    result = await exercise_reviewer(
        {
            "messages": [HumanMessage(content="Create exercises")],
            "exercise_outline": "foundation, intermediate, application, self_check",
            "exercise_items": [
                {
                    "level": "foundation",
                    "question": "What is overfitting?",
                    "answer": "Weak generalization.",
                    "explanation": "",
                    "pitfall": "Do not only inspect training loss.",
                }
            ],
        }
    )

    assert result["exercise_review_verdict"] == "reject"
    assert "too low" in result["exercise_review_reason"]


@pytest.mark.anyio
async def test_exercise_reviewer_rejects_missing_required_field():
    items = []
    for level in ["foundation", "intermediate", "application", "self_check"]:
        item = _exercise_item(level)
        items.append(item.model_dump() if hasattr(item, "model_dump") else item.dict())
    items[2]["explanation"] = ""

    result = await exercise_reviewer(
        {
            "messages": [HumanMessage(content="Create exercises")],
            "exercise_outline": "foundation, intermediate, application, self_check",
            "exercise_items": items,
        }
    )

    assert result["exercise_review_verdict"] == "reject"
    assert "missing explanation" in result["exercise_review_reason"]


@pytest.mark.anyio
async def test_exercise_output_renders_markdown():
    items = []
    for level in ["foundation", "intermediate", "application", "self_check"]:
        item = _exercise_item(level)
        items.append(item.model_dump() if hasattr(item, "model_dump") else item.dict())

    result = await exercise_output(
        {
            "exercise_items": items,
            "exercise_artifact": {"title": "Machine Learning Exercises"},
            "exercise_review_verdict": "approve",
            "exercise_review_reason": "approved",
        }
    )

    content = result["messages"][0].content
    assert result["exercise_artifact"]["title"] == "Machine Learning Exercises"
    assert isinstance(result["messages"][0], AIMessage)
    for text in [
        "foundation",
        "intermediate",
        "application",
        "self_check",
        "answer",
        "explanation",
    ]:
        assert text in content


@pytest.mark.anyio
async def test_exercise_output_empty_artifact_raises():
    with pytest.raises(ValueError, match="exercise items"):
        await exercise_output({})


def test_should_rewrite_exercise_caps_retry_rounds():
    assert (
        should_rewrite_exercise(
            {"exercise_review_verdict": "approve", "exercise_round": 1}
        )
        == "output"
    )
    assert (
        should_rewrite_exercise(
            {"exercise_review_verdict": "reject", "exercise_round": 1}
        )
        == "rewrite"
    )
    with pytest.raises(RuntimeError, match="max rounds"):
        should_rewrite_exercise(
            {"exercise_review_verdict": "reject", "exercise_round": 3}
        )
