"""Tests for fail-fast Markdown review document resource nodes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from src.config import load_settings
from src.graph.review_doc import (
    ReviewDocApprovalError,
    ReviewDocGenerationError,
    ReviewDocReviewVerdict,
    _context_for_subject,
    review_doc_agent,
    review_doc_output,
    review_doc_planner,
    review_doc_reviewer,
    should_rewrite_review_doc,
)
from src.llm.structured_output import StructuredLLMResult, StructuredOutputError


VALID_REVIEW_DOC_MARKDOWN = """# Python函数复习

## 复习目标
理解函数定义、参数和返回值。

## 核心知识点
- def 定义函数
- return 返回结果

## 易错点
- 不要混淆打印和返回值

## 自测清单
- 能写出一个带参数的函数
"""


def _failed_result() -> StructuredLLMResult:
    return StructuredLLMResult(
        success=False,
        parsed=None,
        node_name="review_doc_reviewer",
        llm_node="review_doc",
        schema_name="ReviewDocReviewVerdict",
        provider="test",
        model="test",
        output_mode="deepseek_tool_call_strict",
        failure_phase="business_validation_error",
        error_type="BusinessValidationError",
        error_message="invalid verdict",
    )


def _multi_document_state(**overrides: object) -> dict:
    python_markdown = VALID_REVIEW_DOC_MARKDOWN.replace(
        "# Python函数复习", "# Python 复习资料", 1
    )
    computer_markdown = VALID_REVIEW_DOC_MARKDOWN.replace(
        "# Python函数复习", "# 计算机科学导论 复习资料", 1
    )
    documents = [
        {
            "subject": "python",
            "title": "Python 复习资料",
            "markdown": python_markdown,
        },
        {
            "subject": "computer",
            "title": "计算机科学导论 复习资料",
            "markdown": computer_markdown,
        },
    ]
    state: dict[str, Any] = {
        "messages": [HumanMessage(content="分别生成 Python 和计算机导论复习资料")],
        "review_doc_markdown": "\n\n---\n\n".join(
            document["markdown"] for document in documents
        ),
        "review_doc_markdowns": documents,
        "review_doc_review_verdict": "approve",
        "review_doc_review_reason": "两份文档均通过。",
    }
    state.update(overrides)
    return state


def test_review_doc_verdict_is_strict() -> None:
    with pytest.raises(ValidationError):
        ReviewDocReviewVerdict.model_validate(
            {"verdict": "approve", "reason": "ok", "unexpected": True}
        )

    with pytest.raises(ValidationError):
        ReviewDocReviewVerdict.model_validate({"verdict": "approve", "reason": 123})


def test_review_doc_runtime_configuration_is_explicit() -> None:
    settings = load_settings(reload=True)
    config = settings["llm"]["review_doc"]

    assert config["provider"] == "deepseek_official"
    assert config["model"] == "deepseek-v4-pro"
    assert config["temperature"] == 0.2
    assert config["timeout_seconds"] == 90
    assert config["max_generation_rounds"] == 3


def test_review_doc_legacy_fallback_symbols_are_removed() -> None:
    source = Path("src/graph/review_doc.py").read_text(encoding="utf-8")

    for forbidden in (
        "fallback_used",
        "fallback_reason",
        "quality_warning",
        "_fallback_review_doc_markdown",
        "_ensure_markdown_title",
    ):
        assert forbidden not in source


def test_subject_context_does_not_fall_back_to_other_subjects() -> None:
    context = [{"retrieval_subject": "computer", "content": "computer notes"}]

    assert _context_for_subject(context, "python") == []


@pytest.mark.anyio
async def test_review_doc_planner_empty_outline_raises():
    with patch("src.graph.review_doc.invoke_plain_llm_fail_fast", return_value="  "):
        with pytest.raises(ValueError, match="empty outline"):
            await review_doc_planner(
                {"messages": [HumanMessage(content="make a review doc")]}
            )


@pytest.mark.anyio
async def test_review_doc_agent_empty_markdown_raises():
    with patch("src.graph.review_doc.invoke_plain_llm_fail_fast", return_value="  "):
        with pytest.raises(ReviewDocGenerationError, match="empty Markdown"):
            await review_doc_agent({"review_doc_outline": "outline"})


@pytest.mark.anyio
async def test_review_doc_agent_rejects_multi_document_title_mismatch():
    with (
        patch(
            "src.graph.review_doc.invoke_plain_llm_fail_fast",
            return_value=VALID_REVIEW_DOC_MARKDOWN,
        ),
        pytest.raises(ReviewDocGenerationError, match="title mismatch"),
    ):
        await review_doc_agent(
            {
                "messages": [HumanMessage(content="分别生成两科复习资料")],
                "review_doc_outline": "复习目标、核心知识点、易错点、自测清单",
                "retrieval_plan": [
                    {"subject": "python"},
                    {"subject": "computer"},
                ],
                "context": [],
            }
        )


@pytest.mark.anyio
async def test_review_doc_reviewer_local_rejects_without_llm():
    result = await review_doc_reviewer(
        {"review_doc_markdown": "# Title\n\n## Only one section"}
    )
    assert result["review_doc_review_verdict"] == "reject"
    assert "Markdown missing required sections" in result["review_doc_review_reason"]


@pytest.mark.anyio
async def test_review_doc_reviewer_structured_rejects():
    verdict = ReviewDocReviewVerdict(verdict="reject", reason="Needs citations")
    with patch(
        "src.graph.review_doc.invoke_structured_llm",
        return_value=SimpleNamespace(success=True, parsed=verdict),
    ):
        result = await review_doc_reviewer(
            {"review_doc_markdown": VALID_REVIEW_DOC_MARKDOWN}
        )
    assert result["review_doc_review_verdict"] == "reject"
    assert "Needs citations" in result["review_doc_revision_notes"]


@pytest.mark.anyio
async def test_review_doc_reviewer_exception_propagates():
    with patch(
        "src.graph.review_doc.invoke_structured_llm",
        side_effect=RuntimeError("review failed"),
    ):
        with pytest.raises(RuntimeError, match="review failed"):
            await review_doc_reviewer(
                {"review_doc_markdown": VALID_REVIEW_DOC_MARKDOWN}
            )


@pytest.mark.anyio
async def test_review_doc_reviewer_rejects_failed_structured_result():
    failed_result = _failed_result()
    with (
        patch(
            "src.graph.review_doc.invoke_structured_llm",
            return_value=failed_result,
        ),
        pytest.raises(StructuredOutputError) as exc_info,
    ):
        await review_doc_reviewer({"review_doc_markdown": VALID_REVIEW_DOC_MARKDOWN})

    assert exc_info.value.result is failed_result


@pytest.mark.anyio
async def test_legacy_fallback_marker_cannot_skip_real_reviewer():
    verdict = ReviewDocReviewVerdict(verdict="reject", reason="Evidence is weak")
    with patch(
        "src.graph.review_doc.invoke_structured_llm",
        return_value=SimpleNamespace(success=True, parsed=verdict),
    ) as reviewer:
        result = await review_doc_reviewer(
            {
                "review_doc_markdown": VALID_REVIEW_DOC_MARKDOWN,
                "review_doc_artifact": {"fallback_used": True},
            }
        )

    reviewer.assert_awaited_once()
    assert result["review_doc_review_verdict"] == "reject"


@pytest.mark.anyio
async def test_review_doc_output_empty_markdown_raises():
    with pytest.raises(ReviewDocApprovalError, match="markdown"):
        await review_doc_output({})


@pytest.mark.anyio
async def test_review_doc_output_creates_artifact():
    with patch(
        "src.graph.review_doc.create_markdown_artifact",
        return_value={
            "title": "Python函数复习",
            "filename": "title.md",
            "markdown_url": "/title.md",
        },
    ):
        result = await review_doc_output(
            {
                "review_doc_markdown": VALID_REVIEW_DOC_MARKDOWN,
                "review_doc_review_verdict": "approve",
                "review_doc_review_reason": "approved",
            }
        )

    assert result["review_doc_artifact"]["filename"] == "title.md"
    assert isinstance(result["messages"][0], AIMessage)


@pytest.mark.anyio
@pytest.mark.parametrize("verdict", ["", "reject", "unexpected"])
async def test_review_doc_output_requires_explicit_approve(verdict: str):
    writer = Mock()
    with (
        patch("src.graph.review_doc.create_markdown_artifact", writer),
        pytest.raises(ReviewDocApprovalError, match="approve verdict"),
    ):
        await review_doc_output(
            {
                "review_doc_markdown": VALID_REVIEW_DOC_MARKDOWN,
                "review_doc_review_verdict": verdict,
            }
        )

    writer.assert_not_called()


@pytest.mark.anyio
async def test_review_doc_output_rejects_empty_multi_document_before_io():
    state = _multi_document_state()
    state["review_doc_markdowns"][1]["markdown"] = ""
    writer = Mock()
    with (
        patch("src.graph.review_doc.create_markdown_artifact", writer),
        pytest.raises(ReviewDocApprovalError, match="local quality check"),
    ):
        await review_doc_output(state)

    writer.assert_not_called()


@pytest.mark.anyio
async def test_review_doc_output_propagates_artifact_failure():
    artifact_error = OSError("artifact storage unavailable")
    with (
        patch(
            "src.graph.review_doc.create_markdown_artifact",
            side_effect=artifact_error,
        ),
        pytest.raises(OSError) as exc_info,
    ):
        await review_doc_output(
            {
                "review_doc_markdown": VALID_REVIEW_DOC_MARKDOWN,
                "review_doc_review_verdict": "approve",
            }
        )

    assert exc_info.value is artifact_error


@pytest.mark.anyio
async def test_review_doc_output_creates_every_valid_multi_document():
    with patch(
        "src.graph.review_doc.create_markdown_artifact",
        side_effect=[
            {"artifact_id": "python-doc", "markdown_url": "/python.md"},
            {"artifact_id": "computer-doc", "markdown_url": "/computer.md"},
        ],
    ):
        result = await review_doc_output(_multi_document_state())

    assert [artifact["artifact_id"] for artifact in result["review_doc_artifacts"]] == [
        "python-doc",
        "computer-doc",
    ]
    assert result["review_doc_artifact"]["document_count"] == 2
    assert "quality_warning" not in result["review_doc_artifact"]


def test_should_rewrite_review_doc_max_rounds_rejected_raises():
    assert (
        should_rewrite_review_doc(
            {"review_doc_review_verdict": "approve", "review_doc_round": 1}
        )
        == "output"
    )
    assert (
        should_rewrite_review_doc(
            {"review_doc_review_verdict": "reject", "review_doc_round": 1}
        )
        == "rewrite"
    )
    with pytest.raises(RuntimeError, match="max rounds"):
        should_rewrite_review_doc(
            {"review_doc_review_verdict": "reject", "review_doc_round": 3}
        )

    with pytest.raises(ReviewDocApprovalError, match="explicit approve or reject"):
        should_rewrite_review_doc({"review_doc_review_verdict": ""})
