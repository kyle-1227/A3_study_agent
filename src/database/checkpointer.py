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


def get_db_uri() -> str | None:
    """Read the PostgreSQL connection URI from environment.

    Normalizes SQLAlchemy-style schemes (e.g. ``postgresql+asyncpg://``)
    to plain ``postgresql://`` as required by psycopg.

    Returns:
        The DB_URI string, or None if not configured.
    """
    uri = os.getenv("DB_URI")
    if uri and uri.startswith("postgresql+"):
        uri = "postgresql" + uri[uri.index("://"):]
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


def make_thread_config(thread_id: str | None = None) -> dict:
    """Build the LangGraph config dict with a thread_id.

    Args:
        thread_id: An explicit session identifier. If None, a new UUID is generated.

    Returns:
        Config dict in the format ``{"configurable": {"thread_id": "..."}}``.
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    return {"configurable": {"thread_id": thread_id}}
