"""PostgreSQL checkpointer lifecycle management for LangGraph state persistence.

Uses langgraph-checkpoint-postgres AsyncPostgresSaver to persist conversation
state across sessions, keyed by thread_id.
"""

from __future__ import annotations

import logging
import os
import uuid

from src.config import get_setting

logger = logging.getLogger(__name__)

_MAX_GRAPH_RECURSION_LIMIT = 256


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
