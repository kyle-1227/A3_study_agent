"""Unit tests for app.py — CORS, lifespan graph, and endpoint wiring."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestCORSConfiguration:
    """Verify CORS origins come from environment, not hardcoded wildcard."""

    def test_no_hardcoded_wildcard_origins(self):
        """app.py must not contain allow_origins=['*']."""
        content = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        assert 'allow_origins=["*"]' not in content
        assert "allow_origins=['*']" not in content

    def test_cors_reads_from_env(self):
        """ALLOWED_ORIGINS env var should control CORS origins."""
        content = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        assert "ALLOWED_ORIGINS" in content

    def test_cors_default_is_localhost(self):
        """Default CORS origin should be http://localhost:3000."""
        content = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        assert "http://localhost:3000" in content


class TestNoGlobalGraph:
    """Verify graph is stored on app.state, not as a module global."""

    def test_no_global_graph_variable(self):
        """app.py must not have a module-level 'graph = None' or 'global graph'."""
        content = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        # Should not have module-level graph = None
        lines = content.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped == "graph = None":
                pytest.fail("Found module-level 'graph = None' in app.py")
            if stripped == "global graph":
                pytest.fail("Found 'global graph' in app.py")

    def test_graph_stored_on_app_state(self):
        """Lifespan should store graph on app.state."""
        content = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        assert "app.state.graph" in content

    def test_generate_sse_accepts_graph_param(self):
        """generate_sse should accept graph as a parameter."""
        from app import generate_sse
        import inspect

        sig = inspect.signature(generate_sse)
        assert "graph" in sig.parameters


class TestPyprojectToml:
    """Verify pyproject.toml has required sections."""

    def test_pyproject_exists(self):
        assert (PROJECT_ROOT / "pyproject.toml").is_file()

    def test_has_project_section(self):
        content = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "[project]" in content

    def test_has_dependencies(self):
        content = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "dependencies" in content
        assert "langchain" in content
        assert "fastapi" in content

    def test_has_dev_dependencies(self):
        content = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "[project.optional-dependencies]" in content
        assert "pytest" in content

    def test_has_pytest_config(self):
        content = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "[tool.pytest.ini_options]" in content
        assert 'asyncio_mode = "auto"' in content


class TestEnvExample:
    """Verify .env.example has ALLOWED_ORIGINS."""

    def test_allowed_origins_in_env_example(self):
        content = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        assert "ALLOWED_ORIGINS" in content


class TestInputValidation:
    """Verify Pydantic max_length constraints on request schemas (SEC-01)."""

    def test_chat_request_rejects_oversized_query(self):
        from pydantic import ValidationError
        from src.schemas import ChatRequest

        with pytest.raises(ValidationError):
            ChatRequest(query="x" * 5000)

    def test_chat_request_accepts_normal_query(self):
        from src.schemas import ChatRequest

        req = ChatRequest(query="正常长度的问题")
        assert req.query == "正常长度的问题"

    def test_resume_request_rejects_oversized_plan(self):
        from pydantic import ValidationError
        from src.schemas import ResumeRequest

        with pytest.raises(ValidationError):
            ResumeRequest(thread_id="t-1", edited_plan="x" * 20000)

    def test_resume_request_accepts_normal_plan(self):
        from src.schemas import ResumeRequest

        req = ResumeRequest(thread_id="t-1", edited_plan="## 正常计划")
        assert req.edited_plan == "## 正常计划"


class TestMindmapArtifacts:
    """Verify mindmap artifact download route is safely wired."""

    def test_download_route_returns_xmind(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient
        from app import app

        monkeypatch.setenv("MINDMAP_ARTIFACT_DIR", str(tmp_path))
        artifact_dir = tmp_path / "a1"
        artifact_dir.mkdir()
        artifact_file = artifact_dir / "mindmap.xmind"
        artifact_file.write_bytes(b"fake-xmind")

        with TestClient(app) as client:
            response = client.get("/artifacts/mindmaps/a1/mindmap.xmind")

        assert response.status_code == 200
        assert response.content == b"fake-xmind"

    def test_download_route_rejects_missing_file(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient
        from app import app

        monkeypatch.setenv("MINDMAP_ARTIFACT_DIR", str(tmp_path))

        with TestClient(app) as client:
            response = client.get("/artifacts/mindmaps/a1/missing.xmind")

        assert response.status_code == 404
