"""Tests for fail-fast Markdown review document resource nodes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.review_doc import (
    ReviewDocReviewVerdict,
    review_doc_agent,
    review_doc_output,
    review_doc_planner,
    review_doc_reviewer,
    should_rewrite_review_doc,
)


@pytest.mark.anyio
async def test_review_doc_planner_empty_outline_raises():
    with patch("src.graph.review_doc.invoke_plain_llm_fail_fast", return_value="  "):
        with pytest.raises(ValueError, match="empty outline"):
            await review_doc_planner({"messages": [HumanMessage(content="make a review doc")]})


@pytest.mark.anyio
async def test_review_doc_agent_empty_markdown_raises():
    with patch("src.graph.review_doc.invoke_plain_llm_fail_fast", return_value="  "):
        with pytest.raises(ValueError, match="empty Markdown"):
            await review_doc_agent({"review_doc_outline": "outline"})


@pytest.mark.anyio
async def test_review_doc_reviewer_local_rejects_without_llm():
    result = await review_doc_reviewer({"review_doc_markdown": "# Title\n\n## Only one section"})
    assert result["review_doc_review_verdict"] == "reject"
    assert "three H2" in result["review_doc_review_reason"]


@pytest.mark.anyio
async def test_review_doc_reviewer_structured_rejects():
    markdown = "# Title\n\n## A\ntext\n\n## B\ntext\n\n## C\ntext"
    verdict = ReviewDocReviewVerdict(verdict="reject", reason="Needs citations")
    with patch("src.graph.review_doc.invoke_structured_llm", return_value=SimpleNamespace(parsed=verdict)):
        result = await review_doc_reviewer({"review_doc_markdown": markdown})
    assert result["review_doc_review_verdict"] == "reject"
    assert "Needs citations" in result["review_doc_revision_notes"]


@pytest.mark.anyio
async def test_review_doc_reviewer_exception_propagates():
    markdown = "# Title\n\n## A\ntext\n\n## B\ntext\n\n## C\ntext"
    with patch("src.graph.review_doc.invoke_structured_llm", side_effect=RuntimeError("review failed")):
        with pytest.raises(RuntimeError, match="review failed"):
            await review_doc_reviewer({"review_doc_markdown": markdown})


@pytest.mark.anyio
async def test_review_doc_output_empty_markdown_raises():
    with pytest.raises(ValueError, match="markdown"):
        await review_doc_output({})


@pytest.mark.anyio
async def test_review_doc_output_creates_artifact():
    markdown = "# Title\n\n## A\ntext\n\n## B\ntext\n\n## C\ntext"
    with patch(
        "src.graph.review_doc.create_markdown_artifact",
        return_value={"title": "Title", "filename": "title.md", "markdown_url": "/title.md"},
    ):
        result = await review_doc_output({"review_doc_markdown": markdown})

    assert result["review_doc_artifact"]["filename"] == "title.md"
    assert isinstance(result["messages"][0], AIMessage)


def test_should_rewrite_review_doc_max_rounds_rejected_raises():
    assert should_rewrite_review_doc({"review_doc_review_verdict": "approve", "review_doc_round": 1}) == "output"
    assert should_rewrite_review_doc({"review_doc_review_verdict": "reject", "review_doc_round": 1}) == "rewrite"
    with pytest.raises(RuntimeError, match="max rounds"):
        should_rewrite_review_doc({"review_doc_review_verdict": "reject", "review_doc_round": 3})
