"""Shared strict identity contracts for durable user and anonymous ownership."""

from __future__ import annotations

import hashlib
import re


USER_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9:._/-]{0,159}$"
ANONYMOUS_THREAD_OWNER_PREFIX = "anonymous-thread:v1:"
_IDENTITY_PATTERN = re.compile(USER_ID_PATTERN)
_RESERVED_USER_IDS = frozenset({"unknown"})


class UserIdentityError(ValueError):
    """Content-safe typed rejection of a durable identity."""

    def __init__(self, *, code: str) -> None:
        self.code = code
        super().__init__(f"{code}: durable identity is invalid")


def validate_user_id(value: object) -> str:
    """Return one normalized public user ID or fail without repair."""

    return _validate_identity(value, field_name="user_id", user_scope=True)


def validate_thread_id(value: object) -> str:
    """Return one normalized thread ID suitable for anonymous ownership."""

    return _validate_identity(value, field_name="thread_id", user_scope=False)


def compile_anonymous_thread_owner_id(thread_id: object) -> str:
    """Project a thread into a stable owner namespace disjoint from user IDs."""

    validated = validate_thread_id(thread_id)
    digest = hashlib.sha256(validated.encode("utf-8")).hexdigest()
    return f"{ANONYMOUS_THREAD_OWNER_PREFIX}{digest}"


def _validate_identity(
    value: object,
    *,
    field_name: str,
    user_scope: bool,
) -> str:
    if not isinstance(value, str):
        raise UserIdentityError(code=f"{field_name}_type_invalid")
    folded = value.casefold()
    if (
        value != value.strip()
        or not _IDENTITY_PATTERN.fullmatch(value)
        or folded in _RESERVED_USER_IDS
        or (user_scope and folded.startswith(ANONYMOUS_THREAD_OWNER_PREFIX))
    ):
        raise UserIdentityError(code=f"{field_name}_value_invalid")
    return value


__all__ = [
    "ANONYMOUS_THREAD_OWNER_PREFIX",
    "USER_ID_PATTERN",
    "UserIdentityError",
    "compile_anonymous_thread_owner_id",
    "validate_thread_id",
    "validate_user_id",
]
