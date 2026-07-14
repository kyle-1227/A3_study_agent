"""PostgreSQL session lock for cross-process assessment execution."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import psycopg

_LOCK_DOMAIN = b"a3-study-agent:assessment-execution:v1\0"


class PostgresAssessmentExecutionLock:
    """Serialize one thread's assessment dispatches across app workers."""

    def __init__(self, conninfo: str) -> None:
        if not isinstance(conninfo, str) or not conninfo.strip():
            raise ValueError("PostgreSQL assessment lock requires a connection URI")
        self._conninfo = conninfo

    @asynccontextmanager
    async def hold(self, thread_id: str) -> AsyncIterator[None]:
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError("PostgreSQL assessment lock requires thread_id")
        lock_key = _assessment_advisory_lock_key(thread_id)
        connection = await psycopg.AsyncConnection.connect(
            self._conninfo,
            autocommit=True,
        )
        try:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT pg_advisory_lock(%s)",
                    (lock_key,),
                )
            yield
        finally:
            await connection.close()


def _assessment_advisory_lock_key(thread_id: str) -> int:
    digest = hashlib.sha256(
        _LOCK_DOMAIN + thread_id.encode("utf-8"),
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)


__all__ = ["PostgresAssessmentExecutionLock"]
