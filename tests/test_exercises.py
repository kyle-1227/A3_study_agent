"""Tests for collaborative leveled-exercise resource generation nodes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from src.graph.exercises import (
    ExerciseArtifact,
    ExerciseItem,
    exercise_agent,
    exercise_output,
    exercise_reviewer,
    should_rewrite_exercise,
)


def _exercise_item(level: str) -> ExerciseItem:
    return ExerciseItem(
        level=level,
        question=f"{level}题干",
        answer=f"{level}答案",
        explanation=f"{level}解析",
        pitfall=f"{level}易错提醒",
        tags=["过拟合"],
    )


@patch("src.graph.exercises.get_fallback_llm")
@patch("src.graph.exercises.get_node_llm")
async def test_exercise_agent_generates_structured_items(
    mock_get_llm,
    mock_get_fallback,
):
    structured = MagicMock()
    structured.ainvoke = AsyncMock(
        return_value=ExerciseArtifact(
            title="过拟合分层练习题",
            items=[
                _exercise_item("基础题"),
                _exercise_item("进阶题"),
                _exercise_item("应用题"),
                _exercise_item("自我检查题"),
            ],
        )
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    mock_get_llm.return_value = llm

    fallback_llm = MagicMock()
    fallback_llm.with_structured_output.return_value = MagicMock()
    mock_get_fallback.return_value = fallback_llm

    result = await exercise_agent({
        "messages": [HumanMessage(content="请生成机器学习过拟合的分层练习题")],
        "keypoints": ["机器学习", "过拟合"],
        "context": [{"content": "过拟合是模型泛化能力不足", "source": "ml.md"}],
        "exercise_outline": "基础题：概念；进阶题：判断；应用题：案例；自我检查题：复盘",
        "exercise_round": 0,
    })

    assert result["exercise_artifact"]["title"] == "过拟合分层练习题"
    assert result["exercise_round"] == 1
    assert {item["level"] for item in result["exercise_items"]} == {"基础题", "进阶题", "应用题", "自我检查题"}


async def test_exercise_reviewer_rejects_incomplete_items():
    result = await exercise_reviewer({
        "messages": [HumanMessage(content="请生成过拟合分层练习题")],
        "exercise_outline": "基础题、进阶题、应用题、自我检查题",
        "exercise_items": [
            {
                "level": "基础题",
                "question": "什么是过拟合？",
                "answer": "泛化能力不足。",
                "explanation": "",
                "pitfall": "不要只看训练集。",
            }
        ],
    })

    assert result["exercise_review_verdict"] == "reject"
    assert "至少" in result["exercise_review_reason"] or "缺少" in result["exercise_review_reason"]


async def test_exercise_reviewer_rejects_missing_level():
    items = []
    for level in ["基础题", "进阶题", "应用题", "基础题"]:
        item = _exercise_item(level)
        items.append(item.model_dump() if hasattr(item, "model_dump") else item.dict())

    result = await exercise_reviewer({
        "messages": [HumanMessage(content="请生成过拟合分层练习题")],
        "exercise_outline": "基础题、进阶题、应用题、自我检查题",
        "exercise_items": items,
    })

    assert result["exercise_review_verdict"] == "reject"
    assert "自我检查题" in result["exercise_review_reason"]


async def test_exercise_output_renders_markdown():
    items = []
    for level in ["基础题", "进阶题", "应用题", "自我检查题"]:
        item = _exercise_item(level)
        items.append(item.model_dump() if hasattr(item, "model_dump") else item.dict())

    result = await exercise_output({
        "exercise_items": items,
        "exercise_artifact": {"title": "过拟合分层练习题"},
        "exercise_review_verdict": "approve",
        "exercise_review_reason": "通过",
    })

    content = result["messages"][0].content
    assert result["exercise_artifact"]["title"] == "过拟合分层练习题"
    assert isinstance(result["messages"][0], AIMessage)
    for text in ["基础题", "进阶题", "应用题", "自我检查题", "答案", "解析", "易错提醒"]:
        assert text in content


def test_should_rewrite_exercise_caps_retry_rounds():
    assert should_rewrite_exercise({"exercise_review_verdict": "approve", "exercise_round": 1}) == "output"
    assert should_rewrite_exercise({"exercise_review_verdict": "reject", "exercise_round": 1}) == "rewrite"
    assert should_rewrite_exercise({"exercise_review_verdict": "reject", "exercise_round": 3}) == "output"
