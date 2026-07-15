"""Fail-closed tests for the read-only profile storage adapter surface."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

from src.profile.schema import UserProfile
from src.profile.storage import ProfileStorageReadError, SQLiteProfileStore


def _profile_payload(user_id: str) -> dict[str, object]:
    return UserProfile(user_id=user_id).model_dump(mode="json")


def _create_profile_table(path: Path, *, profile_column: bool = True) -> None:
    statement = (
        "CREATE TABLE profiles (user_id TEXT PRIMARY KEY, profile_json TEXT NOT NULL)"
        if profile_column
        else "CREATE TABLE profiles (user_id TEXT PRIMARY KEY, payload TEXT)"
    )
    with sqlite3.connect(path) as db:
        db.execute(statement)


def _insert_raw_profile(path: Path, *, user_id: str, raw: str) -> None:
    with sqlite3.connect(path) as db:
        db.execute(
            "INSERT INTO profiles (user_id, profile_json) VALUES (?, ?)",
            (user_id, raw),
        )


@pytest.mark.asyncio
async def test_profile_load_strict_round_trip_does_not_initialize_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteProfileStore(tmp_path / "profiles.sqlite")
    profile = UserProfile(user_id="user-1")
    await store.save(profile)

    async def forbidden_init() -> None:
        raise AssertionError("strict load called _ensure_init")

    monkeypatch.setattr(store, "_ensure_init", forbidden_init)
    loaded = await store.load_strict("user-1")

    assert loaded == profile


@pytest.mark.asyncio
async def test_profile_load_strict_missing_database_is_not_created(
    tmp_path: Path,
) -> None:
    database = tmp_path / "missing.sqlite"
    store = SQLiteProfileStore(database)

    with pytest.raises(ProfileStorageReadError) as error:
        await store.load_strict("user-1")

    assert error.value.code == "profile_database_missing"
    assert not database.exists()


@pytest.mark.asyncio
async def test_profile_load_strict_distinguishes_missing_table_and_schema(
    tmp_path: Path,
) -> None:
    empty_database = tmp_path / "empty.sqlite"
    sqlite3.connect(empty_database).close()
    with pytest.raises(ProfileStorageReadError) as missing:
        await SQLiteProfileStore(empty_database).load_strict("user-1")
    assert missing.value.code == "profile_table_missing"

    invalid_database = tmp_path / "invalid.sqlite"
    _create_profile_table(invalid_database, profile_column=False)
    with pytest.raises(ProfileStorageReadError) as invalid:
        await SQLiteProfileStore(invalid_database).load_strict("user-1")
    assert invalid.value.code == "profile_table_schema_invalid"


@pytest.mark.asyncio
async def test_profile_load_strict_returns_none_only_for_absent_user(
    tmp_path: Path,
) -> None:
    database = tmp_path / "profiles.sqlite"
    _create_profile_table(database)

    assert await SQLiteProfileStore(database).load_strict("absent-user") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    (
        "{malformed-sensitive-marker",
        '{"user_id":"user-1","user_id":"duplicate-sensitive-marker"}',
        '{"value":NaN,"marker":"non-finite-sensitive-marker"}',
        '{"value":1e400,"marker":"overflow-sensitive-marker"}',
    ),
)
async def test_profile_load_strict_rejects_invalid_json_without_leaking_content(
    tmp_path: Path,
    raw: str,
) -> None:
    database = tmp_path / "profiles.sqlite"
    _create_profile_table(database)
    _insert_raw_profile(database, user_id="user-1", raw=raw)

    with pytest.raises(ProfileStorageReadError) as error:
        await SQLiteProfileStore(database).load_strict("user-1")

    assert error.value.code == "profile_record_json_invalid"
    assert "sensitive-marker" not in str(error.value)
    assert error.value.__cause__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ("non_object", "missing", "extra", "wrong_type"))
async def test_profile_load_strict_rejects_incomplete_or_drifted_schema(
    tmp_path: Path,
    mutation: str,
) -> None:
    payload: object = _profile_payload("user-1")
    if mutation == "non_object":
        payload = []
    else:
        assert isinstance(payload, dict)
        if mutation == "missing":
            del payload["behavior"]
        elif mutation == "extra":
            learning_style = payload["learning_style"]
            assert isinstance(learning_style, dict)
            learning_style["schema_drift"] = True
        elif mutation == "wrong_type":
            payload["goals"] = {}
        else:
            raise AssertionError(f"unhandled mutation: {mutation}")

    database = tmp_path / f"{mutation}.sqlite"
    _create_profile_table(database)
    _insert_raw_profile(
        database,
        user_id="user-1",
        raw=json.dumps(payload, ensure_ascii=False),
    )

    with pytest.raises(ProfileStorageReadError) as error:
        await SQLiteProfileStore(database).load_strict("user-1")

    assert error.value.code == "profile_record_schema_invalid"


@pytest.mark.asyncio
async def test_profile_load_strict_rejects_payload_identity_mismatch(
    tmp_path: Path,
) -> None:
    database = tmp_path / "profiles.sqlite"
    _create_profile_table(database)
    _insert_raw_profile(
        database,
        user_id="user-1",
        raw=json.dumps(_profile_payload("other-user")),
    )

    with pytest.raises(ProfileStorageReadError) as error:
        await SQLiteProfileStore(database).load_strict("user-1")

    assert error.value.code == "profile_record_identity_mismatch"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_id",
    ("", " user-1", "user-1 ", "unknown", "anonymous-thread:v1:forged"),
)
async def test_profile_load_strict_validates_identity_before_file_access(
    tmp_path: Path,
    user_id: str,
) -> None:
    database = tmp_path / "missing.sqlite"

    with pytest.raises(ValueError):
        await SQLiteProfileStore(database).load_strict(user_id)

    assert not database.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation_kind", ["tuple_list_field", "tuple_in_extra"])
async def test_profile_mutation_rejects_non_json_native_python_shapes(
    tmp_path: Path,
    mutation_kind: str,
) -> None:
    database = tmp_path / "profiles.sqlite"
    store = SQLiteProfileStore(database)
    await store.save(UserProfile(user_id="user-1"))
    before = await store.load_strict("user-1")

    def mutate(current: UserProfile | None) -> UserProfile:
        assert current is not None
        if mutation_kind == "tuple_list_field":
            object.__setattr__(current, "goals", ())
        else:
            current.extra["invalid"] = ("not", "json")
        return current

    with pytest.raises(ValueError):
        await store.mutate_strict("user-1", mutate)

    assert await SQLiteProfileStore(database).load_strict("user-1") == before


@pytest.mark.asyncio
async def test_profile_mutation_returns_the_canonical_persisted_instance(
    tmp_path: Path,
) -> None:
    database = tmp_path / "profiles.sqlite"
    store = SQLiteProfileStore(database)
    await store.save(UserProfile(user_id="user-1"))
    candidate: UserProfile | None = None

    def mutate(current: UserProfile | None) -> UserProfile:
        nonlocal candidate
        assert current is not None
        candidate = current.model_copy(deep=True)
        candidate.tags.append("verified")
        return candidate

    persisted = await store.mutate_strict("user-1", mutate)
    reopened = await SQLiteProfileStore(database).load_strict("user-1")

    assert candidate is not None
    assert persisted is not candidate
    assert persisted == reopened
    assert persisted.tags == ["verified"]
