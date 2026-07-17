"""Unit tests for PostgreSQL checkpointer integration.

Tests cover: checkpointer lifecycle, graph compilation with checkpointer,
thread_id config generation, and the SSE streaming with config.
All tests mock the PostgreSQL connection — no real database required.
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.graph.builder import get_compiled_graph


# ===========================================================================
# TestGetCompiledGraph — checkpointer parameter
# ===========================================================================


class TestGetCompiledGraphWithCheckpointer:
    """Tests that get_compiled_graph correctly accepts a checkpointer."""

    def test_compiles_without_checkpointer(self, learning_guidance_runtime):
        """Default behavior: compile without checkpointer (backward-compatible)."""
        compiled = get_compiled_graph(learning_guidance_runtime)
        assert compiled is not None
        assert hasattr(compiled, "ainvoke")

    def test_compiles_with_checkpointer(self, learning_guidance_runtime):
        """When a checkpointer is provided, it should be wired into the graph."""
        saver = MemorySaver()
        compiled = get_compiled_graph(
            learning_guidance_runtime,
            checkpointer=saver,
        )
        assert compiled is not None
        assert hasattr(compiled, "ainvoke")

    def test_compiled_graph_has_checkpointer(self, learning_guidance_runtime):
        """The compiled graph should reference the checkpointer."""
        saver = MemorySaver()
        compiled = get_compiled_graph(
            learning_guidance_runtime,
            checkpointer=saver,
        )
        assert compiled.checkpointer is saver

    def test_compiled_graph_none_checkpointer(self, learning_guidance_runtime):
        """When checkpointer=None (default), graph.checkpointer should be None."""
        compiled = get_compiled_graph(learning_guidance_runtime)
        assert compiled.checkpointer is None


# ===========================================================================
# TestCheckpointerModule — lifecycle management
# ===========================================================================


class TestCheckpointerModule:
    """Tests for src/database/checkpointer.py functions."""

    def test_get_db_uri_from_env(self):
        """get_db_uri() should read DB_URI from environment."""
        from src.database.checkpointer import get_db_uri

        with patch.dict(os.environ, {"DB_URI": "postgresql://u:p@localhost:5432/db"}):
            assert get_db_uri() == "postgresql://u:p@localhost:5432/db"

    def test_get_db_uri_returns_none_when_missing(self):
        """get_db_uri() should return None when DB_URI is not set."""
        from src.database.checkpointer import get_db_uri

        with patch.dict(os.environ, {}, clear=True):
            assert get_db_uri() is None

    def test_make_thread_config_generates_uuid(self):
        """make_thread_config() with no arg should generate a UUID thread_id."""
        from src.database.checkpointer import make_thread_config

        config = make_thread_config()
        thread_id = config["configurable"]["thread_id"]
        # Should be a valid UUID string
        parsed = uuid.UUID(thread_id)
        assert str(parsed) == thread_id

    def test_make_thread_config_uses_provided_id(self):
        """make_thread_config(thread_id) should use the given ID."""
        from src.database.checkpointer import make_thread_config

        config = make_thread_config("my-session-123")
        assert config["configurable"]["thread_id"] == "my-session-123"

    def test_make_thread_config_structure(self):
        """Config should have the exact structure LangGraph expects."""
        from src.database.checkpointer import make_thread_config

        config = make_thread_config("test")
        assert "configurable" in config
        assert "thread_id" in config["configurable"]
        assert config["recursion_limit"] == 96

    @pytest.mark.parametrize("value", [None, True, 0, 257, "96"])
    def test_graph_recursion_limit_rejects_missing_or_invalid_values(
        self,
        value,
    ):
        from src.database import checkpointer as checkpointer_module

        with patch.object(checkpointer_module, "get_setting", return_value=value):
            with pytest.raises(
                ValueError,
                match="graph.execution_recursion_limit",
            ):
                checkpointer_module.graph_recursion_limit()


class TestPostgresPoolContract:
    """Tests for strict pool configuration and saver lifecycle."""

    @staticmethod
    def _valid_settings() -> dict[str, object]:
        return {
            "min_size": 1,
            "max_size": 8,
            "timeout_seconds": 10,
            "max_waiting": 64,
            "max_lifetime_seconds": 1800,
            "max_idle_seconds": 300,
            "reconnect_timeout_seconds": 30,
            "num_workers": 2,
        }

    def test_pool_config_requires_exact_valid_settings(self, monkeypatch):
        from src.database import checkpointer as checkpointer_module

        settings = self._valid_settings()
        monkeypatch.setattr(
            checkpointer_module,
            "get_setting",
            lambda key: settings if key == "checkpointer.postgres_pool" else None,
        )

        config = checkpointer_module.postgres_pool_config()

        assert config == checkpointer_module.PostgresPoolConfig(
            min_size=1,
            max_size=8,
            timeout_seconds=10.0,
            max_waiting=64,
            max_lifetime_seconds=1800.0,
            max_idle_seconds=300.0,
            reconnect_timeout_seconds=30.0,
            num_workers=2,
        )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("min_size", 0),
            ("max_size", True),
            ("timeout_seconds", 0),
            ("max_waiting", 0),
            ("max_lifetime_seconds", 300),
            ("max_idle_seconds", float("inf")),
            ("reconnect_timeout_seconds", -1),
            ("num_workers", 17),
        ],
    )
    def test_pool_config_rejects_invalid_values(self, monkeypatch, field, value):
        from src.database import checkpointer as checkpointer_module

        settings = self._valid_settings()
        settings[field] = value
        monkeypatch.setattr(
            checkpointer_module,
            "get_setting",
            lambda key: settings if key == "checkpointer.postgres_pool" else None,
        )

        with pytest.raises(ValueError, match="checkpointer.postgres_pool"):
            checkpointer_module.postgres_pool_config()

    @pytest.mark.anyio
    async def test_open_checkpointer_uses_health_checked_pool(self, monkeypatch):
        from src.database import checkpointer as checkpointer_module

        events: list[str] = []

        class FakePool:
            last: "FakePool | None" = None

            def __init__(self, conninfo, **kwargs):
                self.conninfo = conninfo
                self.kwargs = kwargs
                FakePool.last = self

            @staticmethod
            async def check_connection(_connection):
                return None

            async def __aenter__(self):
                events.append("pool_enter")
                return self

            async def __aexit__(self, *_args):
                events.append("pool_exit")
                return False

            async def wait(self, *, timeout):
                events.append(f"pool_wait:{timeout:g}")

        class FakeSaver:
            def __init__(self, pool):
                assert pool is FakePool.last
                events.append("saver_init")

            async def setup(self):
                events.append("saver_setup")

        monkeypatch.setattr(checkpointer_module, "AsyncConnectionPool", FakePool)
        monkeypatch.setattr(checkpointer_module, "AsyncPostgresSaver", FakeSaver)
        monkeypatch.setattr(
            checkpointer_module,
            "postgres_pool_config",
            lambda: checkpointer_module.PostgresPoolConfig(**self._valid_settings()),
        )

        async with checkpointer_module.open_postgres_checkpointer(
            "postgresql://user:pass@localhost:5432/a3"
        ) as saver:
            assert isinstance(saver, FakeSaver)

        pool = FakePool.last
        assert pool is not None
        assert pool.kwargs["open"] is False
        assert pool.kwargs["check"] is FakePool.check_connection
        assert pool.kwargs["kwargs"]["autocommit"] is True
        assert pool.kwargs["kwargs"]["prepare_threshold"] == 0
        assert events == [
            "pool_enter",
            "pool_wait:10",
            "saver_init",
            "saver_setup",
            "pool_exit",
        ]


class TestAppLifespanCheckpointer:
    """Tests app lifespan checkpointer selection without real database access."""

    @pytest.mark.anyio
    async def test_postgres_checkpointer_requires_db_uri(self, monkeypatch):
        import app as app_module

        fake_app = SimpleNamespace(state=SimpleNamespace())
        monkeypatch.setattr(app_module, "checkpointer_enabled", lambda: True)
        monkeypatch.setattr(app_module, "checkpointer_type", lambda: "postgres")
        monkeypatch.setattr(app_module, "get_db_uri", lambda: None)
        get_graph = MagicMock()
        monkeypatch.setattr(
            app_module,
            "get_compiled_resource_evidence_parent_child_graph",
            get_graph,
        )

        with pytest.raises(RuntimeError, match="requires DB_URI"):
            async with app_module.lifespan(fake_app):
                pass

        get_graph.assert_not_called()

    @pytest.mark.anyio
    async def test_postgres_checkpointer_setup_failure_does_not_fallback(
        self,
        monkeypatch,
    ):
        import app as app_module

        @asynccontextmanager
        async def failing_checkpointer(_db_uri):
            raise RuntimeError("setup failed")
            yield

        fake_app = SimpleNamespace(state=SimpleNamespace())
        monkeypatch.setattr(app_module, "checkpointer_enabled", lambda: True)
        monkeypatch.setattr(app_module, "checkpointer_type", lambda: "postgres")
        monkeypatch.setattr(
            app_module,
            "get_db_uri",
            lambda: "postgresql://user:pass@localhost:5432/a3",
        )
        monkeypatch.setattr(
            app_module,
            "open_postgres_checkpointer",
            failing_checkpointer,
        )
        get_graph = MagicMock()
        monkeypatch.setattr(
            app_module,
            "get_compiled_resource_evidence_parent_child_graph",
            get_graph,
        )

        with pytest.raises(RuntimeError, match="setup failed"):
            async with app_module.lifespan(fake_app):
                pass

        get_graph.assert_not_called()


# ===========================================================================
# TestChatRequestModel — thread_id field
# ===========================================================================


class TestChatRequestWithThreadId:
    """Tests that the ChatRequest model accepts an optional thread_id."""

    def test_request_without_thread_id(self):
        """ChatRequest should work without thread_id (backward-compatible)."""
        from src.schemas import ChatRequest

        req = ChatRequest(
            query="hello",
            request_id="00000000-0000-4000-8000-000000000001",
        )
        assert req.query == "hello"
        assert req.thread_id is None

    def test_request_with_thread_id(self):
        """ChatRequest should accept an optional thread_id."""
        from src.schemas import ChatRequest

        req = ChatRequest(
            query="hello",
            request_id="00000000-0000-4000-8000-000000000001",
            thread_id="abc-123",
        )
        assert req.thread_id == "abc-123"


# ===========================================================================
# TestSSEWithConfig — streaming with thread config
# ===========================================================================


class AsyncIteratorMock:
    """Helper to create an async iterator from a list."""

    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


class TestSSEWithConfig:
    """Tests that the SSE generator passes config to graph.astream_events."""

    @staticmethod
    def _make_mock_graph(events=None):
        """Create a mock graph with astream_events and aget_state."""
        from types import SimpleNamespace

        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock(events or []),
        )
        mock_graph.aupdate_state = AsyncMock()
        mock_graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(next=(), tasks=[]),
        )
        return mock_graph

    @pytest.mark.anyio
    async def test_generate_stream_drafts_passes_config(self):
        """generate_stream_drafts should pass thread config to astream_events."""
        from app import generate_stream_drafts

        mock_graph = self._make_mock_graph()

        async for _ in generate_stream_drafts(
            "hello", mock_graph, thread_id="test-thread"
        ):
            pass

        call_args = mock_graph.astream_events.call_args
        config = call_args.kwargs.get("config")
        assert config is not None
        assert config["configurable"]["thread_id"] == "test-thread"

    @pytest.mark.anyio
    async def test_generate_stream_drafts_auto_generates_thread_id(self):
        """When no thread_id is provided, one should be auto-generated."""
        from app import generate_stream_drafts

        mock_graph = self._make_mock_graph()

        async for _ in generate_stream_drafts("hello", mock_graph):
            pass

        call_args = mock_graph.astream_events.call_args
        config = call_args.kwargs.get("config")
        assert config is not None
        thread_id = config["configurable"]["thread_id"]
        # Should be a valid UUID
        uuid.UUID(thread_id)
