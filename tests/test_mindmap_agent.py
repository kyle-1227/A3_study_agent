"""Tests for fail-fast mindmap resource generation nodes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.mindmap import (
    MindmapArtifact,
    MindmapNode,
    mindmap_agent,
    mindmap_output,
    mindmap_reviewer,
    should_rewrite_mindmap,
)


def _mindmap_artifact() -> MindmapArtifact:
    return MindmapArtifact(
        title="Machine Learning Mindmap",
        tree=MindmapNode(
            title="Machine Learning",
            children=[
                MindmapNode(title="Supervised learning", children=[MindmapNode(title="Regression")]),
                MindmapNode(title="Unsupervised learning", children=[MindmapNode(title="Clustering")]),
                MindmapNode(title="Model evaluation", children=[MindmapNode(title="Validation set")]),
                MindmapNode(title="Practice", children=[MindmapNode(title="Mini project")]),
            ],
        ),
    )


@pytest.mark.anyio
async def test_mindmap_agent_generates_json_tree_from_outline():
    artifact = _mindmap_artifact()
    with patch("src.graph.mindmap.invoke_structured_llm", return_value=SimpleNamespace(parsed=artifact)):
        result = await mindmap_agent(
            {
                "messages": [HumanMessage(content="Create a machine learning mindmap")],
                "context": [{"content": "Machine learning course notes", "source": "ml.md"}],
                "mindmap_outline": "Supervised, unsupervised, evaluation, practice",
                "mindmap_round": 0,
            }
        )

    assert result["mindmap_tree"]["title"] == "Machine Learning"
    assert result["mindmap_round"] == 1
    assert "mindmap_artifact" not in result


@pytest.mark.anyio
async def test_mindmap_agent_empty_outline_raises():
    with pytest.raises(ValueError, match="outline"):
        await mindmap_agent({"mindmap_outline": ""})


@pytest.mark.anyio
async def test_mindmap_reviewer_rejects_generic_template_tree():
    result = await mindmap_reviewer(
        {
            "messages": [HumanMessage(content="Summarize machine learning")],
            "mindmap_outline": "Supervised, unsupervised, loss function, evaluation",
            "mindmap_tree": {
                "title": "Machine Learning",
                "children": [
                    {"title": "Core concepts", "children": []},
                    {"title": "Relationships", "children": []},
                    {"title": "Pitfalls", "children": []},
                    {"title": "Practice", "children": []},
                ],
            },
        }
    )

    assert result["mindmap_review_verdict"] == "reject"
    assert "too few nodes" in result["mindmap_review_reason"]


@pytest.mark.anyio
async def test_mindmap_output_generates_artifact():
    with patch(
        "src.graph.mindmap.create_xmind_artifact",
        return_value={
            "artifact_id": "a1",
            "filename": "mindmap.xmind",
            "path": "/tmp/mindmap.xmind",
            "xmind_url": "/artifacts/mindmaps/a1/mindmap.xmind",
        },
    ):
        result = await mindmap_output(
            {
                "mindmap_tree": {
                    "title": "Machine Learning",
                    "children": [{"title": "Regularization", "children": [{"title": "L2"}]}],
                },
                "mindmap_review_verdict": "approve",
                "mindmap_review_reason": "approved",
            }
        )

    assert result["mindmap_artifact"]["tree"]["title"] == "Machine Learning"
    assert result["mindmap_artifact"]["xmind_url"].endswith(".xmind")
    assert isinstance(result["messages"][0], AIMessage)


@pytest.mark.anyio
async def test_mindmap_output_empty_tree_raises():
    with pytest.raises(ValueError, match="mindmap tree"):
        await mindmap_output({})


def test_should_rewrite_mindmap_caps_retry_rounds():
    assert should_rewrite_mindmap({"mindmap_review_verdict": "approve", "mindmap_round": 1}) == "output"
    assert should_rewrite_mindmap({"mindmap_review_verdict": "reject", "mindmap_round": 1}) == "rewrite"
    with pytest.raises(RuntimeError, match="max rounds"):
        should_rewrite_mindmap({"mindmap_review_verdict": "reject", "mindmap_round": 3})
