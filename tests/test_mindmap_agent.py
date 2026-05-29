"""Tests for collaborative mindmap resource generation nodes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from src.graph.mindmap import (
    MindmapArtifact,
    MindmapNode,
    mindmap_agent,
    mindmap_output,
    mindmap_reviewer,
    should_rewrite_mindmap,
)


@patch("src.graph.mindmap.get_fallback_llm")
@patch("src.graph.mindmap.get_node_llm")
async def test_mindmap_agent_generates_json_tree_from_outline(
    mock_get_llm,
    mock_get_fallback,
):
    structured = MagicMock()
    structured.ainvoke = AsyncMock(
        return_value=MindmapArtifact(
            title="过拟合思维导图",
            tree=MindmapNode(
                title="过拟合",
                children=[
                    MindmapNode(
                        title="现象识别",
                        children=[MindmapNode(title="训练误差低"), MindmapNode(title="验证误差高")],
                    ),
                    MindmapNode(
                        title="缓解方法",
                        children=[MindmapNode(title="正则化"), MindmapNode(title="交叉验证")],
                    ),
                    MindmapNode(
                        title="模型评估",
                        children=[MindmapNode(title="泛化误差"), MindmapNode(title="验证集")],
                    ),
                    MindmapNode(
                        title="实践检查",
                        children=[MindmapNode(title="学习曲线"), MindmapNode(title="数据增强")],
                    ),
                ],
            ),
        )
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    mock_get_llm.return_value = llm

    fallback_llm = MagicMock()
    fallback_llm.with_structured_output.return_value = MagicMock()
    mock_get_fallback.return_value = fallback_llm

    result = await mindmap_agent({
        "messages": [HumanMessage(content="给我生成机器学习过拟合的思维导图")],
        "keypoints": ["机器学习", "过拟合"],
        "context": [{"content": "过拟合是模型泛化能力不足", "source": "ml.md"}],
        "mindmap_outline": "1. 现象识别：训练误差低、验证误差高\n2. 缓解方法：正则化、交叉验证",
        "mindmap_round": 0,
    })

    assert result["mindmap_tree"]["title"] == "过拟合"
    assert result["mindmap_round"] == 1
    assert "mindmap_artifact" not in result


async def test_mindmap_reviewer_rejects_generic_template_tree():
    result = await mindmap_reviewer({
        "messages": [HumanMessage(content="总结机器学习的知识点")],
        "mindmap_outline": "监督学习、无监督学习、损失函数、模型评估",
        "mindmap_tree": {
            "title": "机器学习",
            "children": [
                {"title": "核心概念", "children": []},
                {"title": "关系层级", "children": []},
                {"title": "易错点", "children": []},
                {"title": "实践案例", "children": []},
            ],
        },
    })

    assert result["mindmap_review_verdict"] == "reject"
    assert "节点数" in result["mindmap_review_reason"] or "模板" in result["mindmap_review_reason"]


@patch("src.graph.mindmap.create_xmind_artifact")
async def test_mindmap_output_generates_artifact(mock_create_artifact):
    mock_create_artifact.return_value = {
        "artifact_id": "a1",
        "filename": "mindmap.xmind",
        "path": "/tmp/mindmap.xmind",
        "xmind_url": "/artifacts/mindmaps/a1/mindmap.xmind",
    }

    result = await mindmap_output({
        "mindmap_tree": {
            "title": "过拟合",
            "children": [{"title": "正则化", "children": [{"title": "L2 正则"}]}],
        },
        "mindmap_review_verdict": "approve",
        "mindmap_review_reason": "通过",
    })

    assert result["mindmap_artifact"]["tree"]["title"] == "过拟合"
    assert result["mindmap_artifact"]["xmind_url"].endswith(".xmind")
    assert isinstance(result["messages"][0], AIMessage)


def test_should_rewrite_mindmap_caps_retry_rounds():
    assert should_rewrite_mindmap({"mindmap_review_verdict": "approve", "mindmap_round": 1}) == "output"
    assert should_rewrite_mindmap({"mindmap_review_verdict": "reject", "mindmap_round": 1}) == "rewrite"
    assert should_rewrite_mindmap({"mindmap_review_verdict": "reject", "mindmap_round": 3}) == "output"
