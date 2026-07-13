"""Tests for fail-fast mindmap resource generation nodes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.mindmap import (
    MindmapArtifact,
    MindmapNode,
    MindmapReviewVerdict,
    mindmap_agent,
    mindmap_output,
    mindmap_reviewer,
    should_rewrite_mindmap,
)
from src.llm.structured_output import StructuredLLMResult, StructuredOutputError


def _mindmap_artifact() -> MindmapArtifact:
    return MindmapArtifact(
        title="Machine Learning Mindmap",
        tree=MindmapNode(
            title="Machine Learning",
            children=[
                MindmapNode(
                    title="Supervised learning",
                    children=[MindmapNode(title="Regression")],
                ),
                MindmapNode(
                    title="Unsupervised learning",
                    children=[MindmapNode(title="Clustering")],
                ),
                MindmapNode(
                    title="Model evaluation",
                    children=[MindmapNode(title="Validation set")],
                ),
                MindmapNode(
                    title="Practice", children=[MindmapNode(title="Mini project")]
                ),
            ],
        ),
    )


def _agent_state() -> dict:
    return {
        "messages": [HumanMessage(content="Create a machine learning mindmap")],
        "context": [{"content": "Machine learning course notes", "source": "ml.md"}],
        "mindmap_outline": "Supervised, unsupervised, evaluation, practice",
        "mindmap_round": 0,
    }


def _failed_structured_result(
    *,
    node_name: str = "mindmap_agent",
    schema_name: str = "MindmapArtifact",
    failure_phase: str,
) -> StructuredLLMResult:
    return StructuredLLMResult(
        success=False,
        parsed=None,
        node_name=node_name,
        llm_node="mindmap",
        schema_name=schema_name,
        provider="test",
        model="test",
        output_mode="native_json_schema_pydantic",
        failure_phase=failure_phase,
        error_type="TestStructuredFailure",
        error_message="structured output failed",
    )


@pytest.mark.anyio
async def test_mindmap_agent_generates_json_tree_from_outline():
    artifact = _mindmap_artifact()
    with patch(
        "src.graph.mindmap.invoke_structured_llm",
        return_value=SimpleNamespace(success=True, parsed=artifact),
    ):
        result = await mindmap_agent(_agent_state())

    assert result["mindmap_tree"]["title"] == "Machine Learning"
    assert result["mindmap_round"] == 1
    assert "mindmap_artifact" not in result


@pytest.mark.anyio
@pytest.mark.parametrize(
    "failure_phase",
    [
        "provider_http_error",
        "parsing_error",
        "validation_error",
        "business_validation_error",
    ],
)
async def test_mindmap_agent_propagates_structured_output_error(
    failure_phase: str,
):
    structured_error = StructuredOutputError(
        _failed_structured_result(failure_phase=failure_phase)
    )
    with (
        patch(
            "src.graph.mindmap.invoke_structured_llm",
            side_effect=structured_error,
        ),
        pytest.raises(StructuredOutputError) as exc_info,
    ):
        await mindmap_agent(_agent_state())

    assert exc_info.value is structured_error


@pytest.mark.anyio
async def test_mindmap_agent_rejects_non_raising_failed_structured_result():
    failed_result = _failed_structured_result(failure_phase="validation_error")
    with (
        patch(
            "src.graph.mindmap.invoke_structured_llm",
            return_value=failed_result,
        ),
        pytest.raises(StructuredOutputError) as exc_info,
    ):
        await mindmap_agent(_agent_state())

    assert exc_info.value.result is failed_result


@pytest.mark.anyio
async def test_mindmap_agent_propagates_transport_exception():
    transport_error = ConnectionError("mindmap provider transport failed")
    with (
        patch(
            "src.graph.mindmap.invoke_structured_llm",
            side_effect=transport_error,
        ),
        pytest.raises(ConnectionError) as exc_info,
    ):
        await mindmap_agent(_agent_state())

    assert exc_info.value is transport_error


@pytest.mark.anyio
async def test_mindmap_reviewer_uses_strict_structured_verdict():
    verdict = MindmapReviewVerdict(verdict="approve", reason="Structure is complete.")
    with patch(
        "src.graph.mindmap.invoke_structured_llm",
        return_value=SimpleNamespace(success=True, parsed=verdict),
    ):
        result = await mindmap_reviewer(
            {
                "messages": [HumanMessage(content="Review the mindmap")],
                "mindmap_outline": "Machine learning topics",
                "mindmap_tree": _mindmap_artifact().tree.model_dump(),
            }
        )

    assert result == {
        "mindmap_review_verdict": "approve",
        "mindmap_review_reason": "Structure is complete.",
        "mindmap_revision_notes": "",
    }


@pytest.mark.anyio
async def test_mindmap_agent_empty_outline_raises():
    with pytest.raises(ValueError, match="outline"):
        await mindmap_agent({"mindmap_outline": ""})


@pytest.mark.anyio
async def test_mindmap_reviewer_rejects_non_raising_failed_structured_result():
    failed_result = _failed_structured_result(
        node_name="mindmap_reviewer",
        schema_name="MindmapReviewVerdict",
        failure_phase="business_validation_error",
    )
    with (
        patch(
            "src.graph.mindmap.invoke_structured_llm",
            return_value=failed_result,
        ),
        pytest.raises(StructuredOutputError) as exc_info,
    ):
        await mindmap_reviewer(
            {
                "messages": [HumanMessage(content="Review the mindmap")],
                "mindmap_outline": "Machine learning topics",
                "mindmap_tree": _mindmap_artifact().tree.model_dump(),
            }
        )

    assert exc_info.value.result is failed_result


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
                    "children": [
                        {"title": "Regularization", "children": [{"title": "L2"}]}
                    ],
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


@pytest.mark.anyio
async def test_mindmap_output_missing_title_raises_without_default_artifact():
    with pytest.raises(ValueError, match="title"):
        await mindmap_output(
            {
                "mindmap_tree": {
                    "children": [{"title": "Regularization", "children": []}]
                }
            }
        )


def test_should_rewrite_mindmap_caps_retry_rounds():
    assert (
        should_rewrite_mindmap(
            {"mindmap_review_verdict": "approve", "mindmap_round": 1}
        )
        == "output"
    )
    assert (
        should_rewrite_mindmap({"mindmap_review_verdict": "reject", "mindmap_round": 1})
        == "rewrite"
    )
    with pytest.raises(RuntimeError, match="max rounds"):
        should_rewrite_mindmap({"mindmap_review_verdict": "reject", "mindmap_round": 3})
