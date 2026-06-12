"""Offline integration smoke tests for A3 Study Agent."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

from langchain_core.messages import HumanMessage

from src.graph.builder import build_graph
from src.graph.supervisor import route_by_intent, supervisor_node

project_root = Path(__file__).resolve().parent.parent


def test_graph_compiles_offline():
    graph = build_graph()
    compiled = graph.compile()
    assert compiled is not None


def test_planning_intent_no_longer_valid():
    """Planning is no longer a valid intent — routes to unknown."""
    assert route_by_intent({"intent": "planning"}) == "unknown"


@patch("src.graph.supervisor.invoke_structured_llm", new_callable=AsyncMock)
async def test_study_plan_request_stays_academic_with_resource_type(mock_invoke):
    """academic intent with requested_resource_type=study_plan stays academic."""
    from src.graph.supervisor import SupervisorOutput

    mock_invoke.return_value = type("Result", (), {
        "parsed": SupervisorOutput(
            intent="academic",
            keywords=["learning plan"],
            confidence=0.9,
            subject_candidates=[],
            requested_resource_type="study_plan",
        ),
        "raw_output": "{}",
    })()

    result = await supervisor_node({
        "messages": [HumanMessage(content="Help me make a learning plan")],
    })

    assert result["intent"] == "academic"
    assert result["requested_resource_type"] == "study_plan"


def test_no_hardcoded_secrets():
    secret_patterns = [
        re.compile(r"sk-[a-zA-Z0-9]{20,}"),
        re.compile(r"(OPENAI_API_KEY|OPENROUTER_API_KEY|TAVILY_API_KEY)\s*=\s*[\"'][a-zA-Z0-9]"),
    ]

    violations: list[str] = []
    for path in [project_root / "app.py", *list((project_root / "src").rglob("*.py"))]:
        content = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in secret_patterns:
            if pattern.findall(content):
                violations.append(str(path.relative_to(project_root)))

    assert not violations


def test_gitignore_covers_local_artifacts():
    content = (project_root / ".gitignore").read_text(encoding="utf-8", errors="ignore")
    assert ".env" in content
    assert "chroma_store" in content


def test_env_file_not_tracked_by_default():
    assert (project_root / ".env.example").exists()
