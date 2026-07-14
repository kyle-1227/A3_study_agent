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

    def test_generate_stream_drafts_accepts_graph_param(self):
        """generate_stream_drafts should accept graph as a parameter."""
        from app import generate_stream_drafts
        import inspect

        sig = inspect.signature(generate_stream_drafts)
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
            ChatRequest(
                query="x" * 5000,
                request_id="00000000-0000-4000-8000-000000000001",
            )

    def test_chat_request_accepts_normal_query(self):
        from src.schemas import ChatRequest

        req = ChatRequest(
            query="normal length question",
            request_id="00000000-0000-4000-8000-000000000001",
        )
        assert req.query == "normal length question"

    def test_resume_request_rejects_oversized_plan(self):
        from pydantic import ValidationError
        from src.schemas import ResumeRequest

        with pytest.raises(ValidationError):
            ResumeRequest(
                thread_id="t-1",
                request_id="00000000-0000-4000-8000-000000000001",
                edited_plan="x" * 20000,
            )

    def test_resume_request_accepts_normal_plan(self):
        from src.schemas import ResumeRequest

        req = ResumeRequest(
            thread_id="t-1",
            request_id="00000000-0000-4000-8000-000000000001",
            edited_plan="## Normal plan",
        )
        assert req.edited_plan == "## Normal plan"

    def test_resume_request_accepts_memory_use_choice(self):
        from src.schemas import ResumeRequest

        req = ResumeRequest(
            thread_id="t-1",
            request_id="00000000-0000-4000-8000-000000000001",
            memory_use_choice="use",
        )
        assert req.memory_use_choice == "use"

    def test_resume_request_accepts_profile_completion(self):
        from src.schemas import ResumeRequest

        req = ResumeRequest(
            thread_id="t-1",
            request_id="00000000-0000-4000-8000-000000000001",
            profile_completion={
                "learning_goal": "Master ML basics",
                "current_foundation": "Python",
                "daily_study_time": "2 hours",
            },
        )

        assert req.profile_completion is not None
        assert req.profile_completion.learning_goal == "Master ML basics"

    def test_request_id_is_required_and_must_be_uuid(self):
        from pydantic import ValidationError
        from src.schemas import ChatRequest

        with pytest.raises(ValidationError):
            ChatRequest(query="missing id")
        with pytest.raises(ValidationError):
            ChatRequest(query="bad id", request_id="not-a-uuid")


class TestDevMemoryClear:
    """Verify development-only persistent memory clearing."""

    @pytest.mark.anyio
    async def test_clear_persistent_memory_for_thread_updates_memory_fields(
        self, monkeypatch
    ):
        from app import clear_persistent_memory_for_thread
        from src.graph.state import (
            ACTIVITY_TIMELINE_CLEAR,
            CONTEXT_USAGE_REPORTS_CLEAR,
            DICT_CLEAR,
            GENERATED_ARTIFACTS_CLEAR,
            MEMORY_CLEAR,
            TASK_WORKSPACE_CLEAR,
            WORKSPACE_EVENTS_CLEAR,
            LLM_INPUT_MANIFESTS_CLEAR,
            CONTEXT_INFLUENCE_LEDGER_CLEAR,
            SESSION_CONTEXT_MEMORY_LEDGER_CLEAR,
        )

        graph = AsyncMock()
        monkeypatch.delenv("APP_ENV", raising=False)
        monkeypatch.delenv("A3_ENV", raising=False)

        with patch("app.get_setting", return_value=True):
            result = await clear_persistent_memory_for_thread(graph, "thread-1")

        graph.aupdate_state.assert_awaited_once()
        _, values = graph.aupdate_state.await_args.args
        assert values["conversation_summary"] == ""
        assert values["conversation_summary_v2"] == {}
        assert values["compact_boundary"] == {}
        assert values["compaction_result"] == {}
        assert values["last_provider_dispatch"] == {}
        assert values["evidence_summary_memory"] is MEMORY_CLEAR
        assert values["evidence_gap_memory"] is MEMORY_CLEAR
        assert values["episodic_memory_results"] == []
        assert values["semantic_memory_results"] == []
        assert values["task_workspace"] is TASK_WORKSPACE_CLEAR
        assert values["workspace_events"] is WORKSPACE_EVENTS_CLEAR
        assert values["resource_artifacts_by_type"] is DICT_CLEAR
        assert values["last_generated_artifacts"] is GENERATED_ARTIFACTS_CLEAR
        assert values["last_resource_final_payload"] is DICT_CLEAR
        assert values["last_qa_response"] == {}
        assert values["llm_input_manifest"] == {}
        assert values["llm_input_manifests"] is LLM_INPUT_MANIFESTS_CLEAR
        assert values["thread_context_ledger"] is DICT_CLEAR
        assert (
            values["session_context_memory_ledger"]
            is SESSION_CONTEXT_MEMORY_LEDGER_CLEAR
        )
        assert values["thread_context_window_v3"] == {}
        assert values["background_context_window"] == {}
        assert values["context_continuity"] == {}
        assert values["context_influence_ledger"] is CONTEXT_INFLUENCE_LEDGER_CLEAR
        assert values["context_usage_report"] == {}
        assert values["context_usage_reports"] is CONTEXT_USAGE_REPORTS_CLEAR
        assert values["activity_timeline"] is ACTIVITY_TIMELINE_CLEAR
        assert result == {
            "ok": True,
            "thread_id": "thread-1",
            "cleared_fields": [
                "conversation_summary",
                "conversation_summary_v2",
                "compact_boundary",
                "compaction_result",
                "last_provider_dispatch",
                "evidence_summary_memory",
                "evidence_gap_memory",
                "episodic_memory_results",
                "semantic_memory_results",
                "task_workspace",
                "workspace_events",
                "resource_artifacts_by_type",
                "last_generated_artifacts",
                "last_resource_final_payload",
                "last_qa_response",
                "llm_input_manifest",
                "llm_input_manifests",
                "thread_context_ledger",
                "session_context_memory_ledger",
                "thread_context_window_v3",
                "background_context_window",
                "context_continuity",
                "context_influence_ledger",
                "context_usage_report",
                "context_usage_reports",
                "activity_timeline",
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

        with (
            patch(
                "app.clear_persistent_memory_for_thread",
                new_callable=AsyncMock,
                return_value=helper_result,
            ),
            patch("app.checkpointer_enabled", return_value=False),
        ):
            with TestClient(app) as client:
                response = client.post("/dev/threads/thread-1/memory/clear")

        assert response.status_code == 200
        assert response.json() == helper_result


def test_context_window_status_includes_workspace_counts():
    from app import _context_window_status
    from src.context_engineering.influence import (
        build_influence_entry,
        build_influence_update,
        merge_context_influence_ledger,
    )

    influence_state = {"request_id": "request-1", "thread_id": "thread-1"}
    influence_ledger = merge_context_influence_ledger(
        {},
        build_influence_update(
            state=influence_state,
            entries=[
                build_influence_entry(
                    state=influence_state,
                    kind="planner_output",
                    source_node="mindmap_planner",
                    preview="Compact outline",
                )
            ],
        ),
    )

    _request_window, thread_window = _context_window_status(
        {
            "context_influence_ledger": influence_ledger,
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
            },
        }
    )

    assert thread_window["workspace_present"] is True
    assert thread_window["workspace_active_subject"] == "math"
    assert thread_window["workspace_evidence_summary_count"] == 1
    assert thread_window["workspace_gap_count"] == 1
    assert thread_window["workspace_artifact_count"] == 1
    assert thread_window["workspace_updated_at"] == "2026-01-01T00:00:00+00:00"
    assert thread_window["context_influence_entry_count"] == 1
    assert thread_window["context_influence_ledger"]["present"] is True


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

        with patch("app.checkpointer_enabled", return_value=False):
            with TestClient(app) as client:
                response = client.get("/artifacts/mindmaps/a1/mindmap.xmind")

        assert response.status_code == 200
        assert response.content == b"fake-xmind"

    def test_download_route_rejects_missing_file(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient
        from app import app

        monkeypatch.setenv("MINDMAP_ARTIFACT_DIR", str(tmp_path))

        with patch("app.checkpointer_enabled", return_value=False):
            with TestClient(app) as client:
                response = client.get("/artifacts/mindmaps/a1/missing.xmind")

        assert response.status_code == 404


class TestResourceFinalV3Projection:
    @staticmethod
    def _payload() -> dict:
        from src.graph.resource_final_v3 import (
            ResourceFinalV3ResourceValidation,
            ResourceFinalV3Validation,
            build_resource_final_v3,
            build_resource_final_v3_resource,
        )

        resource_validation = ResourceFinalV3ResourceValidation(
            schema_version="resource_validation_v1",
            resource_type="mindmap",
            valid=True,
            terminal_status="success",
            renderable_count=1,
            downloadable_count=1,
            verified_local_count=1,
            remote_unverified_count=0,
            failure_reason="",
            warnings=(),
        )
        resource = build_resource_final_v3_resource(
            thread_id="thread-1",
            request_id="request-1",
            kind="mindmap",
            status="success",
            title="Machine learning map",
            summary="Mindmap ready",
            payload={
                "mindmap": {
                    "title": "Machine learning map",
                    "tree": {"title": "Machine learning"},
                    "xmind_url": "/artifacts/map.xmind",
                }
            },
            artifact_refs={"xmind_url": "/artifacts/map.xmind"},
            validation=resource_validation,
        )
        final = build_resource_final_v3(
            thread_id="thread-1",
            request_id="request-1",
            terminal_status="success",
            resources=(resource,),
            recommendations=(),
            blocked_resources=(),
            errors=(),
            validation=ResourceFinalV3Validation(
                schema_version="resource_final_validation_v3",
                resource_count=1,
                success_count=1,
                partial_success_count=0,
                failed_count=0,
                blocked_count=0,
                renderable_count=1,
                downloadable_count=1,
            ),
            summary="Resource bundle ready",
        )
        return final.model_dump(mode="json")

    def test_v3_is_authoritative_over_legacy_projection(self):
        from app import _resource_final_payload

        payload = self._payload()
        projected = _resource_final_payload(
            {
                "thread_id": "thread-1",
                "request_id": "request-1",
                "resource_final_v3": payload,
                "requested_resource_type": "study_plan",
                "study_plan_artifact": {"title": "legacy must not win"},
            }
        )
        assert projected == payload

    @pytest.mark.parametrize(
        "invalid_v3",
        [None, [], "resource_final_v3", {"type": "resource_final"}],
    )
    def test_present_invalid_v3_never_falls_back_to_legacy(self, invalid_v3):
        from app import _resource_final_payload

        with pytest.raises((TypeError, ValueError)):
            _resource_final_payload(
                {
                    "resource_final_v3": invalid_v3,
                    "requested_resource_type": "study_plan",
                    "study_plan_artifact": {"title": "legacy must not win"},
                }
            )

    @pytest.mark.parametrize(
        ("identity_field", "identity_value"),
        [("thread_id", "thread-other"), ("request_id", "request-other")],
    )
    def test_v3_identity_must_match_runtime_state(
        self,
        identity_field,
        identity_value,
    ):
        from app import _resource_final_payload

        state = {
            "thread_id": "thread-1",
            "request_id": "request-1",
            "resource_final_v3": self._payload(),
        }
        state[identity_field] = identity_value
        with pytest.raises(ValueError, match=identity_field):
            _resource_final_payload(state)


class TestThreadStatusProfileCompletion:
    """_thread_status_from_snapshot must derive profile_completion_request
    from pending task interrupts when checkpoint values are empty."""

    def test_derives_profile_completion_from_task_interrupt(self):
        from app import _thread_status_from_snapshot

        request_payload = {
            "title": "Need profile before study plan",
            "fields": [
                {
                    "key": "learning_goal",
                    "label": "Learning goal",
                    "required": True,
                    "max_chars": 400,
                },
                {
                    "key": "current_foundation",
                    "label": "Current foundation",
                    "required": True,
                    "max_chars": 400,
                },
                {
                    "key": "daily_study_time",
                    "label": "Daily study time",
                    "required": True,
                    "max_chars": 200,
                },
            ],
        }
        interrupt_obj = SimpleNamespace(
            value={
                "type": "profile_completion_required",
                "profile_completion_request": request_payload,
                "resume_available": True,
            }
        )
        task = SimpleNamespace(interrupts=[interrupt_obj])
        snapshot = SimpleNamespace(
            # No pending_interrupt_type or profile_completion_request in values
            values={
                "run_status": "running",
                "schema_version": "run_control_v1",
                "pending_interrupt_type": "",
                "profile_completion_request": {},
                "current_node": "",
                "last_completed_node": "",
                "stopped_at": "",
                "stop_reason": "",
            },
            tasks=[task],
            next=(),
        )

        response = _thread_status_from_snapshot("t-1", snapshot)

        assert response.resume_available is True
        assert response.pending_interrupt_type == "profile_completion_required"
        assert response.profile_completion_request
        assert response.profile_completion_request["title"] == request_payload["title"]
        assert len(response.profile_completion_request["fields"]) == 3

    def test_status_exposes_last_qa_response_additively(self):
        from app import _thread_status_from_snapshot
        from src.graph.qa import QAResponse, QASuggestion, build_qa_final_payload

        qa_payload = build_qa_final_payload(
            response=QAResponse(
                answer="Stored answer",
                uncertainty_note="",
                grounding_status="general_knowledge",
                suggestions=[
                    QASuggestion(
                        label="Continue",
                        action="continue_qa",
                        resource_type="",
                    )
                ],
            ),
            qa_scope="general",
            thread_id="t-1",
            request_id="r-1",
        )
        snapshot = SimpleNamespace(
            values={
                "schema_version": "run_control_v1",
                "run_status": "completed",
                "stop_requested": False,
                "stop_reason": "",
                "stop_requested_at": "",
                "current_node": "",
                "last_completed_node": "qa_agent",
                "resume_available": False,
                "stopped_at": "",
                "pending_interrupt_type": "",
                "last_qa_response": qa_payload,
            },
            tasks=[],
            next=(),
        )

        response = _thread_status_from_snapshot("t-1", snapshot)

        assert response.last_qa_response["qa_id"] == qa_payload["qa_id"]
        assert response.thread_context_window["last_qa_response_present"] is True
        assert response.thread_context_window["last_qa_scope"] == "general"

    def test_profile_completion_sanitizer_consistent_with_sse(self):
        """_safe_profile_completion_request produces identical output for
        both SSE emission and status response paths."""
        from app import _safe_profile_completion_request

        interrupt_value = {
            "type": "profile_completion_required",
            "profile_completion_request": {
                "title": "Need profile",
                "fields": [
                    {"key": "learning_goal", "label": "Goal", "required": True},
                ],
            },
            "resume_available": True,
        }

        first = _safe_profile_completion_request(interrupt_value)
        second = _safe_profile_completion_request(dict(interrupt_value))

        assert first == second
        assert first["title"] == "Need profile"
        assert first["fields"][0]["key"] == "learning_goal"
