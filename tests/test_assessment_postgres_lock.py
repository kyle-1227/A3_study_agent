"""PostgreSQL advisory-lock adapter tests without a live database."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.database import assessment_lock as lock_module
from src.database.assessment_lock import PostgresAssessmentExecutionLock


class _Cursor:
    def __init__(self, calls: list[tuple[str, tuple[int, ...]]]) -> None:
        self._calls = calls

    async def __aenter__(self) -> _Cursor:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def execute(self, statement: str, parameters: tuple[int, ...]) -> None:
        self._calls.append((statement, parameters))


class _Connection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[int, ...]]] = []
        self.closed = False

    def cursor(self) -> _Cursor:
        return _Cursor(self.calls)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.anyio
async def test_postgres_lock_is_parameterized_and_connection_scoped(
    monkeypatch: pytest.MonkeyPatch,
):
    connection = _Connection()
    connect_calls: list[tuple[str, bool]] = []

    async def connect(conninfo: str, *, autocommit: bool):
        connect_calls.append((conninfo, autocommit))
        return connection

    monkeypatch.setattr(
        lock_module.psycopg,
        "AsyncConnection",
        SimpleNamespace(connect=connect),
    )
    execution_lock = PostgresAssessmentExecutionLock("postgresql://db.example/a3")

    async with execution_lock.hold("thread-assessment-lock-1"):
        assert connection.closed is False

    assert connect_calls == [("postgresql://db.example/a3", True)]
    assert len(connection.calls) == 1
    statement, parameters = connection.calls[0]
    assert statement == "SELECT pg_advisory_lock(%s)"
    assert len(parameters) == 1
    assert isinstance(parameters[0], int)
    assert connection.closed is True
    assert "db.example" not in statement


def test_postgres_lock_requires_explicit_connection_and_thread_identity():
    with pytest.raises(ValueError, match="connection URI"):
        PostgresAssessmentExecutionLock("")

    execution_lock = PostgresAssessmentExecutionLock("postgresql://db.example/a3")

    async def enter_blank_thread() -> None:
        async with execution_lock.hold("  "):
            raise AssertionError("blank thread lock must not be entered")

    with pytest.raises(ValueError, match="thread_id"):
        asyncio.run(enter_blank_thread())
