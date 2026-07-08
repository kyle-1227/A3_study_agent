"""Unit tests for app.py: CORS, lifespan graph, and endpoint wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

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

        req = ChatRequest(query="normal length question")
        assert req.query == "normal length question"

    def test_resume_request_rejects_oversized_plan(self):
        from pydantic import ValidationError
        from src.schemas import ResumeRequest

        with pytest.raises(ValidationError):
            ResumeRequest(thread_id="t-1", edited_plan="x" * 20000)

    def test_resume_request_accepts_normal_plan(self):
        from src.schemas import ResumeRequest

        req = ResumeRequest(thread_id="t-1", edited_plan="## Normal plan")
        assert req.edited_plan == "## Normal plan"

    def test_resume_request_accepts_memory_use_choice(self):
        from src.schemas import ResumeRequest

        req = ResumeRequest(thread_id="t-1", memory_use_choice="use")
        assert req.memory_use_choice == "use"


class TestResourceFinalPayloadCore:
    """Verify final resource payload shaping."""

    def test_evidence_controlled_stop_is_evidence_summary_payload(self):
        from app import _resource_final_payload

        payload = _resource_final_payload(
            {
                "evidence_controlled_stop": True,
                "final_response_type": "evidence_summary",
                "requested_resource_type": "study_plan",
                "evidence_controlled_stop_reason": "evidence_insufficient",
                "plan": "## Evidence summary\nCurrent evidence is insufficient.",
                "study_plan_artifact": {},
            }
        )

        assert payload is not None
        assert payload["type"] == "resource_final"
        assert payload["resource_type"] == "evidence_summary"
        assert payload["controlled_stop"] is True
        assert payload["controlled_stop_reason"] == "evidence_insufficient"
        assert "Current evidence is insufficient" in payload["answer"]
        assert "study_plan" not in payload

    def test_multi_resource_bundle_payload(self):
        from app import _resource_final_payload

        payload = _resource_final_payload(
            {
                "requested_resource_type": "mindmap",
                "requested_resource_types": ["mindmap", "quiz"],
                "resource_generation_status": "partial_success",
                "resource_bundle_artifact": {
                    "type": "resource_bundle",
                    "status": "partial_success",
                    "resources": [{"resource_type": "mindmap", "title": "Mock Map"}],
                    "errors": [
                        {
                            "resource_type": "quiz",
                            "error_message_sanitized": "quiz failed",
                        }
                    ],
                },
                "messages": [type("Msg", (), {"content": "bundle summary"})()],
            }
        )

        assert payload is not None
        assert payload["type"] == "resource_final"
        assert payload["resource_type"] == "bundle"
        assert payload["resource_generation_status"] == "partial_success"
        assert payload["resources"] == [
            {"resource_type": "mindmap", "title": "Mock Map"}
        ]
        assert payload["errors"][0]["resource_type"] == "quiz"

    def test_resource_bundle_payload_uses_bundle_message_when_last_message_missing(
        self,
    ):
        from app import _resource_final_payload

        payload = _resource_final_payload(
            {
                "requested_resource_type": "mindmap",
                "requested_resource_types": ["mindmap", "quiz"],
                "resource_generation_status": "success",
                "resource_bundle_artifact": {
                    "type": "resource_bundle",
                    "status": "success",
                    "message": "# 已生成多类学习资源",
                    "resources": [{"resource_type": "mindmap", "title": "Mock Map"}],
                    "errors": [],
                },
                "messages": [],
            }
        )

        assert payload is not None
        assert payload["resource_type"] == "bundle"
        assert payload["answer"] == "# 已生成多类学习资源"
        assert "multi_resource_summary" not in payload


class TestDevMemoryClear:
    """Verify development-only persistent memory clearing."""

    @pytest.mark.anyio
    async def test_clear_persistent_memory_for_thread_updates_memory_fields(
        self, monkeypatch
    ):
        from app import clear_persistent_memory_for_thread
        from src.graph.state import (
            DICT_CLEAR,
            GENERATED_ARTIFACTS_CLEAR,
            MEMORY_CLEAR,
            TASK_WORKSPACE_CLEAR,
            WORKSPACE_EVENTS_CLEAR,
            LLM_INPUT_MANIFESTS_CLEAR,
        )

        graph = AsyncMock()
        monkeypatch.delenv("APP_ENV", raising=False)
        monkeypatch.delenv("A3_ENV", raising=False)

        with patch("app.get_setting", return_value=True):
            result = await clear_persistent_memory_for_thread(graph, "thread-1")

        graph.aupdate_state.assert_awaited_once()
        _, values = graph.aupdate_state.await_args.args
        assert values["conversation_summary"] == ""
        assert values["evidence_summary_memory"] is MEMORY_CLEAR
        assert values["evidence_gap_memory"] is MEMORY_CLEAR
        assert values["episodic_memory_results"] == []
        assert values["semantic_memory_results"] == []
        assert values["task_workspace"] is TASK_WORKSPACE_CLEAR
        assert values["workspace_events"] is WORKSPACE_EVENTS_CLEAR
        assert values["resource_artifacts_by_type"] is DICT_CLEAR
        assert values["last_generated_artifacts"] is GENERATED_ARTIFACTS_CLEAR
        assert values["llm_input_manifest"] == {}
        assert values["llm_input_manifests"] is LLM_INPUT_MANIFESTS_CLEAR
        assert values["thread_context_ledger"] is DICT_CLEAR
        assert values["background_context_window"] == {}
        assert values["context_continuity"] == {}
        assert result == {
            "ok": True,
            "thread_id": "thread-1",
            "cleared_fields": [
                "conversation_summary",
                "evidence_summary_memory",
                "evidence_gap_memory",
                "episodic_memory_results",
                "semantic_memory_results",
                "task_workspace",
                "workspace_events",
                "resource_artifacts_by_type",
                "last_generated_artifacts",
                "llm_input_manifest",
                "llm_input_manifests",
                "thread_context_ledger",
                "background_context_window",
                "context_continuity",
            ],
        }

    @pytest.mark.anyio
    async def test_clear_persistent_memory_for_thread_rejects_production(
        self, monkeypatch
    ):
        from app import clear_persistent_memory_for_thread

        graph = AsyncMock()
        monkeypatch.setenv("APP_ENV", "production")

        with patch("app.get_setting", return_value=True):
            with pytest.raises(Exception) as exc_info:
                await clear_persistent_memory_for_thread(graph, "thread-1")

        assert getattr(exc_info.value, "status_code", None) == 403
        graph.aupdate_state.assert_not_called()

    @pytest.mark.anyio
    async def test_clear_persistent_memory_for_thread_rejects_a3_production(
        self, monkeypatch
    ):
        from app import clear_persistent_memory_for_thread

        graph = AsyncMock()
        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.setenv("A3_ENV", "prod")

        with patch("app.get_setting", return_value=True):
            with pytest.raises(Exception) as exc_info:
                await clear_persistent_memory_for_thread(graph, "thread-1")

        assert getattr(exc_info.value, "status_code", None) == 403
        graph.aupdate_state.assert_not_called()

    @pytest.mark.anyio
    async def test_clear_persistent_memory_for_thread_rejects_disabled_config(
        self, monkeypatch
    ):
        from app import clear_persistent_memory_for_thread

        graph = AsyncMock()
        monkeypatch.delenv("APP_ENV", raising=False)
        monkeypatch.delenv("A3_ENV", raising=False)

        with patch("app.get_setting", return_value=False):
            with pytest.raises(Exception) as exc_info:
                await clear_persistent_memory_for_thread(graph, "thread-1")

        assert getattr(exc_info.value, "status_code", None) == 403
        graph.aupdate_state.assert_not_called()

    def test_clear_thread_memory_endpoint_returns_helper_result(self):
        from fastapi.testclient import TestClient
        from app import app

        helper_result = {
            "ok": True,
            "thread_id": "thread-1",
            "cleared_fields": [
                "conversation_summary",
                "evidence_summary_memory",
                "evidence_gap_memory",
            ],
        }

        with patch(
            "app.clear_persistent_memory_for_thread",
            new_callable=AsyncMock,
            return_value=helper_result,
        ):
            with TestClient(app) as client:
                response = client.post("/dev/threads/thread-1/memory/clear")

        assert response.status_code == 200
        assert response.json() == helper_result


def test_context_window_status_includes_workspace_counts():
    from app import _context_window_status

    _request_window, thread_window = _context_window_status(
        {
            "task_workspace": {
                "schema_version": 1,
                "workspace_id": "workspace:v1:one",
                "active_subject": "math",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "evidence_summaries": [{"evidence_id": "e1"}],
                "coverage_gaps": [{"gap_id": "g1"}],
                "artifacts_by_id": {
                    "artifact:v1:one": {"artifact_id": "artifact:v1:one"}
                },
            }
        }
    )

    assert thread_window["workspace_present"] is True
    assert thread_window["workspace_active_subject"] == "math"
    assert thread_window["workspace_evidence_summary_count"] == 1
    assert thread_window["workspace_gap_count"] == 1
    assert thread_window["workspace_artifact_count"] == 1
    assert thread_window["workspace_updated_at"] == "2026-01-01T00:00:00+00:00"


def test_new_request_status_values_preserve_thread_workspace_counts():
    from app import _context_window_status, _new_request_status_values

    values = _new_request_status_values(
        {
            "request_context_window": {
                "current_request_id": "old-request",
                "current_node": "mindmap_agent",
                "last_event_count": 12,
            },
            "context_usage_history": [{"node_name": "mindmap_agent"}],
            "task_workspace": {
                "schema_version": 1,
                "workspace_id": "workspace:v1:ml",
                "active_subject": "machine_learning",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "evidence_summaries": [{"evidence_id": "e1"}],
                "coverage_gaps": [],
                "artifacts_by_id": {
                    "artifact:v1:one": {"artifact_id": "artifact:v1:one"}
                },
            },
        },
        {
            "request_context_window": {
                "current_request_id": "new-request",
                "current_node": "",
                "last_event_count": 0,
            },
            "context_usage_history": [],
            "request_id": "new-request",
            "thread_id": "thread-1",
        },
    )

    request_window, thread_window = _context_window_status(values)

    assert request_window["current_request_id"] == "new-request"
    assert request_window["last_event_count"] == 0
    assert thread_window["context_usage_history_count"] == 1
    assert thread_window["workspace_present"] is True
    assert thread_window["workspace_active_subject"] == "machine_learning"
    assert thread_window["workspace_evidence_summary_count"] == 1
    assert thread_window["workspace_artifact_count"] == 1


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


class TestResourceFinalPayloadArtifacts:
    def test_plain_answer_without_artifacts_has_no_resource_payload(self):
        from app import _resource_final_payload

        payload = _resource_final_payload(
            {
                "requested_resource_type": "",
                "messages": [
                    SimpleNamespace(content="Python list 和 tuple 的区别是...")
                ],
            }
        )

        assert payload is None

    def test_multi_resource_payload_includes_all_available_artifacts(self):
        from app import _resource_final_payload

        final_state = {
            "requested_resource_type": "review_doc",
            "requested_resource_types": [
                "review_doc",
                "mindmap",
                "quiz",
                "code_practice",
                "video_script",
                "video_animation",
                "study_plan",
            ],
            "resource_generation_status": "success",
            "resource_bundle_artifact": {
                "type": "resource_bundle",
                "status": "success",
                "message": "bundle summary",
                "resources": [
                    {"resource_type": "review_doc", "status": "success"},
                    {"resource_type": "mindmap", "status": "success"},
                    {"resource_type": "quiz", "status": "success"},
                    {"resource_type": "code_practice", "status": "success"},
                    {"resource_type": "video_script", "status": "success"},
                    {"resource_type": "video_animation", "status": "success"},
                    {"resource_type": "study_plan", "status": "success"},
                ],
                "errors": [],
            },
            "messages": [SimpleNamespace(content="bundle summary")],
            "review_doc_artifact": {
                "title": "Python 复习资料",
                "filename": "python.md",
                "docx_filename": "python.docx",
                "markdown_url": "/artifacts/review-docs/r1/python.md",
                "docx_url": "/artifacts/review-docs/r1/python.docx",
                "markdown": "# Python 复习资料",
            },
            "review_doc_artifacts": [
                {
                    "title": "Python 复习资料",
                    "filename": "python.md",
                    "docx_filename": "python.docx",
                    "markdown_url": "/artifacts/review-docs/r1/python.md",
                    "docx_url": "/artifacts/review-docs/r1/python.docx",
                    "markdown": "# Python 复习资料",
                }
            ],
            "mindmap_artifact": {
                "title": "Python 思维导图",
                "tree": {"title": "Python", "children": []},
                "xmind_url": "/artifacts/mindmaps/m1/python.xmind",
            },
            "mindmap_tree": {"title": "Python", "children": []},
            "exercise_items": [{"question": "Q1"}],
            "exercise_artifact": {
                "title": "Python 练习题",
                "markdown_url": "/artifacts/exercises/e1/python.md",
                "docx_url": "/artifacts/exercises/e1/python.docx",
            },
            "code_practice_artifact": {
                "title": "Python 代码题",
                "markdown_url": "/artifacts/code-practice/c1/python.md",
                "docx_url": "/artifacts/code-practice/c1/python.docx",
                "source_url": "/artifacts/code-practice/c1/main.py",
            },
            "video_script_artifact": {
                "title": "Python 教学脚本",
                "markdown_url": "/artifacts/video-scripts/v1/script.md",
                "docx_url": "/artifacts/video-scripts/v1/script.docx",
                "srt_url": "/artifacts/video-scripts/v1/script.srt",
            },
            "video_animation_artifact": {
                "title": "Python 教学动画",
                "html_url": "/artifacts/video-animations/a1/preview.html",
                "json_url": "/artifacts/video-animations/a1/timeline.json",
                "srt_url": "/artifacts/video-animations/a1/captions.srt",
            },
            "study_plan_artifact": {
                "title": "Python Study Plan",
            },
            "study_plan_markdown": "# Python Study Plan",
            "study_plan_document_artifact": {
                "title": "Python Study Plan",
                "filename": "python-plan.md",
                "docx_filename": "python-plan.docx",
                "markdown_url": "/artifacts/review-docs/s1/python-plan.md",
                "docx_url": "/artifacts/review-docs/s1/python-plan.docx",
            },
        }

        payload = _resource_final_payload(final_state)

        assert payload is not None
        assert payload["resource_type"] == "bundle"
        assert payload["answer"] == "bundle summary"
        assert payload["resource_bundle"]["type"] == "resource_bundle"
        assert [item["resource_type"] for item in payload["resources"]] == [
            "review_doc",
            "mindmap",
            "quiz",
            "code_practice",
            "video_script",
            "video_animation",
            "study_plan",
        ]
        assert payload["errors"] == []
        assert payload["review_doc_artifacts"]
        assert payload["mindmap"]["title"] == "Python 思维导图"
        assert payload["exercise_artifact"]["title"] == "Python 练习题"
        assert payload["code_practice_artifact"]["source_url"].endswith("main.py")
        assert payload["video_script_artifact"]["srt_url"].endswith("script.srt")
        assert payload["video_animation_artifact"]["html_url"].endswith("preview.html")
        assert (
            payload["study_plan"]["markdown_url"]
            == "/artifacts/review-docs/s1/python-plan.md"
        )
        assert payload["study_plan"]["markdown"] == "# Python Study Plan"
        assert "multi_resource_results" not in payload
        assert "multi_resource_summary" not in payload

    def test_single_resource_payloads_still_work(self):
        from app import _resource_final_payload

        review_doc_payload = _resource_final_payload(
            {
                "requested_resource_type": "review_doc",
                "messages": [SimpleNamespace(content="# Python 复习资料")],
                "review_doc_artifacts": [
                    {
                        "title": "Python 复习资料",
                        "filename": "python.md",
                        "markdown_url": "/artifacts/review-docs/r1/python.md",
                    }
                ],
            }
        )
        quiz_payload = _resource_final_payload(
            {
                "requested_resource_type": "quiz",
                "messages": [SimpleNamespace(content="练习题正文")],
                "exercise_items": [{"question": "Q1"}],
                "exercise_artifact": {"title": "Python 练习题"},
            }
        )
        mindmap_payload = _resource_final_payload(
            {
                "requested_resource_type": "mindmap",
                "messages": [SimpleNamespace(content="mindmap")],
                "mindmap_artifact": {
                    "title": "Python 思维导图",
                    "tree": {"title": "Python", "children": []},
                    "xmind_url": "/artifacts/mindmaps/m1/python.xmind",
                },
            }
        )

        assert review_doc_payload and review_doc_payload["review_doc_artifacts"]
        assert (
            quiz_payload
            and quiz_payload["exercise_artifact"]["title"] == "Python 练习题"
        )
        assert (
            mindmap_payload and mindmap_payload["mindmap"]["title"] == "Python 思维导图"
        )
