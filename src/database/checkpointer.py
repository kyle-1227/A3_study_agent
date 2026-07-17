"""PostgreSQL checkpointer lifecycle management for LangGraph state persistence.

Uses langgraph-checkpoint-postgres AsyncPostgresSaver to persist conversation
state across sessions, keyed by thread_id.
"""

from __future__ import annotations

import math
import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from src.config import get_setting

logger = logging.getLogger(__name__)

_MAX_GRAPH_RECURSION_LIMIT = 256
_POSTGRES_POOL_KEYS = frozenset(
    {
        "min_size",
        "max_size",
        "timeout_seconds",
        "max_waiting",
        "max_lifetime_seconds",
        "max_idle_seconds",
        "reconnect_timeout_seconds",
        "num_workers",
    }
)


@dataclass(frozen=True, slots=True)
class PostgresPoolConfig:
    """Strict reconnecting pool settings for the production checkpointer."""

    min_size: int
    max_size: int
    timeout_seconds: float
    max_waiting: int
    max_lifetime_seconds: float
    max_idle_seconds: float
    reconnect_timeout_seconds: float
    num_workers: int


def get_db_uri() -> str | None:
    """Read the PostgreSQL connection URI from environment.

    Normalizes SQLAlchemy-style schemes (e.g. ``postgresql+asyncpg://``)
    to plain ``postgresql://`` as required by psycopg.

    Returns:
        The DB_URI string, or None if not configured.
    """
    uri = os.getenv("DB_URI")
    if uri and uri.startswith("postgresql+"):
        uri = "postgresql" + uri[uri.index("://") :]
    return uri


def _pool_int(
    raw: dict[str, object],
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = raw.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise ValueError(
            f"checkpointer.postgres_pool.{name} must be an integer "
            f"between {minimum} and {maximum}"
        )
    return value


def _pool_float(
    raw: dict[str, object],
    name: str,
    *,
    maximum: float,
) -> float:
    value = raw.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
        or value > maximum
    ):
        raise ValueError(
            f"checkpointer.postgres_pool.{name} must be a positive number "
            f"not greater than {maximum:g}"
        )
    return float(value)


def postgres_pool_config() -> PostgresPoolConfig:
    """Load the exact reconnecting-pool contract without silent defaults."""

    raw = get_setting("checkpointer.postgres_pool")
    if not isinstance(raw, dict) or set(raw) != _POSTGRES_POOL_KEYS:
        raise ValueError(
            "checkpointer.postgres_pool must define exactly "
            + ", ".join(sorted(_POSTGRES_POOL_KEYS))
        )
    min_size = _pool_int(raw, "min_size", minimum=1, maximum=32)
    max_size = _pool_int(raw, "max_size", minimum=1, maximum=64)
    if max_size < min_size:
        raise ValueError(
            "checkpointer.postgres_pool.max_size must be greater than or "
            "equal to min_size"
        )
    max_idle_seconds = _pool_float(
        raw,
        "max_idle_seconds",
        maximum=86_400,
    )
    max_lifetime_seconds = _pool_float(
        raw,
        "max_lifetime_seconds",
        maximum=86_400,
    )
    if max_lifetime_seconds <= max_idle_seconds:
        raise ValueError(
            "checkpointer.postgres_pool.max_lifetime_seconds must be greater "
            "than max_idle_seconds"
        )
    return PostgresPoolConfig(
        min_size=min_size,
        max_size=max_size,
        timeout_seconds=_pool_float(raw, "timeout_seconds", maximum=120),
        max_waiting=_pool_int(raw, "max_waiting", minimum=1, maximum=1024),
        max_lifetime_seconds=max_lifetime_seconds,
        max_idle_seconds=max_idle_seconds,
        reconnect_timeout_seconds=_pool_float(
            raw,
            "reconnect_timeout_seconds",
            maximum=300,
        ),
        num_workers=_pool_int(raw, "num_workers", minimum=1, maximum=16),
    )


@asynccontextmanager
async def open_postgres_checkpointer(
    db_uri: str,
) -> AsyncIterator[AsyncPostgresSaver]:
    """Yield a saver backed by a health-checked, reconnecting connection pool."""

    if not isinstance(db_uri, str) or not db_uri or db_uri != db_uri.strip():
        raise ValueError("PostgreSQL checkpointer requires a non-empty DB_URI")
    config = postgres_pool_config()
    pool: AsyncConnectionPool[AsyncConnection[dict[str, Any]]]
    pool = AsyncConnectionPool(
        conninfo=db_uri,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        min_size=config.min_size,
        max_size=config.max_size,
        open=False,
        check=AsyncConnectionPool.check_connection,
        timeout=config.timeout_seconds,
        max_waiting=config.max_waiting,
        max_lifetime=config.max_lifetime_seconds,
        max_idle=config.max_idle_seconds,
        reconnect_timeout=config.reconnect_timeout_seconds,
        num_workers=config.num_workers,
    )
    async with pool:
        await pool.wait(timeout=config.timeout_seconds)
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        yield checkpointer


def _env_bool(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def checkpointer_enabled() -> bool:
    """Return whether LangGraph checkpointer support should be enabled."""
    env_value = _env_bool("CHECKPOINTER_ENABLED")
    if env_value is not None:
        return env_value
    return bool(get_setting("checkpointer.enabled", True))


def checkpointer_type() -> str:
    """Return configured checkpointer type."""
    value = os.getenv("CHECKPOINTER_TYPE") or get_setting("checkpointer.type", "memory")
    return str(value or "memory").strip().lower()


def graph_recursion_limit() -> int:
    """Return the explicit bounded LangGraph superstep ceiling."""

    value = get_setting("graph.execution_recursion_limit")
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > _MAX_GRAPH_RECURSION_LIMIT
    ):
        raise ValueError(
            "graph.execution_recursion_limit must be an integer between 1 and 256"
        )
    return value


def make_thread_config(thread_id: str | None = None) -> dict:
    """Build the LangGraph config dict with a thread_id.

    Args:
        thread_id: An explicit session identifier. If None, a new UUID is generated.

    Returns:
        Config dict containing the thread identity and explicit recursion limit.
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": graph_recursion_limit(),
    }
