"""Opt-in real PostgreSQL assessment-lock concurrency test."""

from __future__ import annotations

import asyncio
import os

import pytest

from src.database.assessment_lock import PostgresAssessmentExecutionLock


@pytest.mark.asyncio
async def test_real_postgres_lock_serializes_two_independent_instances():
    conninfo = os.getenv("A3_TEST_POSTGRES_URI")
    if not conninfo:
        pytest.skip("A3_TEST_POSTGRES_URI is not configured")

    first_lock = PostgresAssessmentExecutionLock(conninfo)
    second_lock = PostgresAssessmentExecutionLock(conninfo)
    first_acquired = asyncio.Event()
    release_first = asyncio.Event()
    second_attempted = asyncio.Event()
    second_acquired = asyncio.Event()

    async def first_worker() -> None:
        async with first_lock.hold("thread-assessment-postgres-integration-1"):
            first_acquired.set()
            await release_first.wait()

    async def second_worker() -> None:
        second_attempted.set()
        async with second_lock.hold("thread-assessment-postgres-integration-1"):
            second_acquired.set()

    first_task = asyncio.create_task(first_worker())
    await asyncio.wait_for(first_acquired.wait(), timeout=10)
    second_task = asyncio.create_task(second_worker())
    await asyncio.wait_for(second_attempted.wait(), timeout=10)
    await asyncio.sleep(0.05)
    assert second_acquired.is_set() is False

    release_first.set()
    await asyncio.wait_for(first_task, timeout=10)
    await asyncio.wait_for(second_task, timeout=10)
    assert second_acquired.is_set() is True
