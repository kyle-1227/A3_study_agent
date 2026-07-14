from __future__ import annotations

from pathlib import Path

from src.graph import academic
from src.graph.builder import build_graph


REPO_ROOT = Path(__file__).resolve().parents[1]
RETIRED_NODE_NAMES = (
    "curriculum_planner",
    "assessment_result_handler",
    "adaptive_practice_responder",
    "recommendation_provider",
)


def test_retired_learning_node_callables_are_removed() -> None:
    for node_name in RETIRED_NODE_NAMES:
        assert not hasattr(academic, node_name), node_name


def test_retired_learning_nodes_are_absent_from_served_graph() -> None:
    node_ids = set(build_graph().nodes)
    assert node_ids.isdisjoint(RETIRED_NODE_NAMES)


def test_retired_learning_nodes_have_no_runtime_configuration_or_ui_mapping() -> None:
    runtime_surfaces = (
        REPO_ROOT / "config" / "settings.yaml",
        REPO_ROOT / "frontend" / "components" / "explain-panel.tsx",
    )
    for path in runtime_surfaces:
        content = path.read_text(encoding="utf-8")
        for node_name in RETIRED_NODE_NAMES:
            assert node_name not in content, f"{node_name} remains in {path}"
