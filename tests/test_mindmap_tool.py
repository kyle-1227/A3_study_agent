"""Tests for mindmap tree normalization and artifact export."""

from __future__ import annotations

import zipfile

from src.tools.mindmap_tool import create_xmind_artifact, normalize_mindmap_tree


def _linear_tree(depth: int) -> dict:
    node: dict = {"title": f"Level {depth}", "children": []}
    for level in range(depth - 1, 0, -1):
        node = {"title": f"Level {level}", "children": [node]}
    return node


def test_normalize_tree_fills_empty_titles_and_limits_depth():
    tree = {
        "title": "",
        "children": [
            {
                "title": "A",
                "children": [
                    {
                        "title": "B",
                        "children": [{"title": "C", "children": [{"title": "D"}]}],
                    },
                ],
            },
        ],
    }

    normalized = normalize_mindmap_tree(tree, max_depth=3, max_nodes=10)

    assert normalized["title"] == "未命名知识点"
    assert normalized["children"][0]["children"][0]["children"] == []


def test_normalize_tree_allows_seven_levels_and_truncates_the_eighth():
    normalized = normalize_mindmap_tree(_linear_tree(8))

    node = normalized
    for level in range(1, 8):
        assert node["title"] == f"Level {level}"
        children = node["children"]
        if level == 7:
            assert children == []
        else:
            assert len(children) == 1
            node = children[0]


def test_normalize_tree_limits_node_count():
    tree = {"title": "root", "children": [{"title": str(i)} for i in range(10)]}

    normalized = normalize_mindmap_tree(tree, max_depth=5, max_nodes=4)

    assert normalized["title"] == "root"
    assert len(normalized["children"]) == 3


def test_create_xmind_artifact_writes_zip(tmp_path, monkeypatch):
    monkeypatch.setenv("MINDMAP_ARTIFACT_DIR", str(tmp_path))
    tree = {"title": "机器学习", "children": [{"title": "过拟合", "children": []}]}

    artifact = create_xmind_artifact(tree, title="机器学习导图")

    path = tmp_path / artifact["artifact_id"] / artifact["filename"]
    assert path.exists()
    assert artifact["xmind_url"].endswith(artifact["filename"])
    with zipfile.ZipFile(path) as zf:
        assert "content.xml" in zf.namelist()
        content = zf.read("content.xml").decode("utf-8")
        assert "机器学习" in content
