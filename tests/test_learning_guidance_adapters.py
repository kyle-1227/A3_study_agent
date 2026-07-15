"""Real SQLite adapter tests for strict profile and history projections."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.learning_guidance.adapters.history import (
    HistoryAdapterError,
    HistorySnapshotAdapterV1,
)
from src.learning_guidance.adapters.profile import (
    ProfileAdapterError,
    ProfileSnapshotAdapterV1,
    profile_goal_fingerprint,
)
from src.learning_guidance.knowledge_graph import KnowledgeGraphV1
from src.memory.schema import EpisodicMemoryRecord
from src.memory.storage import MemoryStorageReadError, SQLiteMemoryStore
from src.profile.schema import Goal, SkillEntry, UserProfile
from src.profile.storage import SQLiteProfileStore


def _knowledge_graph() -> KnowledgeGraphV1:
    return KnowledgeGraphV1.model_validate(
        {
            "schema_version": "knowledge_graph_v1",
            "data_version": "test-v1",
            "subjects": [
                {
                    "subject_id": "math",
                    "title": "Mathematics",
                    "topics": [
                        {
                            "topic_id": "math.algebra",
                            "title": "Algebra",
                            "difficulty": 0.5,
                            "estimated_hours": 2.0,
                            "prerequisite_topic_ids": [],
                            "knowledge_points": ["Equations"],
                            "resources": [
                                {
                                    "resource_id": "math.algebra.review",
                                    "resource_type": "review_doc",
                                    "title": "Algebra review",
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )


def _profile_binding(*, preferences: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "learning_guidance_profile_v1",
        "skills": [
            {
                "signal_id": "skill.math.algebra",
                "subject": "math",
                "topic_id": "math.algebra",
            }
        ],
        "goals": [
            {
                "signal_id": "goal.math.algebra",
                "subject": "math",
                "topic_id": "math.algebra",
                "goal_fingerprint": profile_goal_fingerprint(
                    Goal(goal="Learn algebra", importance=0.9, progress=0.2)
                ),
            }
        ],
        "preferences": preferences,
    }


def _profile(*, binding: object) -> UserProfile:
    profile = UserProfile(
        user_id="user-1",
        skills={"math.algebra": SkillEntry(level=0.3, confidence=0.8)},
        goals=[Goal(goal="Learn algebra", importance=0.9, progress=0.2)],
        extra={"learning_guidance_v1": binding},
    )
    profile.learning_style.prefer_visual = 0.8
    return profile


@pytest.mark.asyncio
async def test_profile_adapter_projects_only_explicit_bound_signals(
    tmp_path: Path,
) -> None:
    store = SQLiteProfileStore(tmp_path / "profiles.sqlite")
    await store.save(
        _profile(
            binding=_profile_binding(
                preferences=[
                    {
                        "signal_id": "preference.math.algebra.visual",
                        "subject": "math",
                        "topic_id": "math.algebra",
                        "dimension": "prefer_visual",
                        "strength": 0.8,
                    }
                ]
            )
        )
    )
    adapter = ProfileSnapshotAdapterV1(
        store=store,
        knowledge_graph=_knowledge_graph(),
    )

    snapshot = await adapter.load("user-1")

    assert snapshot is not None
    assert snapshot.skills[0].topic_id == "math.algebra"
    assert snapshot.goals[0].topic_id == "math.algebra"
    assert snapshot.preferences[0].strength == 0.8


@pytest.mark.asyncio
async def test_profile_adapter_does_not_project_default_preferences(
    tmp_path: Path,
) -> None:
    store = SQLiteProfileStore(tmp_path / "profiles.sqlite")
    await store.save(_profile(binding=_profile_binding(preferences=[])))
    adapter = ProfileSnapshotAdapterV1(
        store=store,
        knowledge_graph=_knowledge_graph(),
    )

    snapshot = await adapter.load("user-1")

    assert snapshot is not None
    assert snapshot.preferences == ()


@pytest.mark.asyncio
async def test_profile_adapter_returns_unavailable_without_v1_marker(
    tmp_path: Path,
) -> None:
    store = SQLiteProfileStore(tmp_path / "profiles.sqlite")
    await store.save(UserProfile(user_id="user-1"))
    adapter = ProfileSnapshotAdapterV1(
        store=store,
        knowledge_graph=_knowledge_graph(),
    )

    assert await adapter.load("user-1") is None


@pytest.mark.asyncio
async def test_profile_adapter_rejects_explicit_null_v1_marker(
    tmp_path: Path,
) -> None:
    store = SQLiteProfileStore(tmp_path / "profiles.sqlite")
    await store.save(_profile(binding=None))
    adapter = ProfileSnapshotAdapterV1(
        store=store,
        knowledge_graph=_knowledge_graph(),
    )

    with pytest.raises(ProfileAdapterError) as error:
        await adapter.load("user-1")

    assert error.value.code == "profile_binding_schema_invalid"


@pytest.mark.asyncio
async def test_profile_goal_binding_survives_goal_reordering(tmp_path: Path) -> None:
    binding = _profile_binding(preferences=[])
    profile = _profile(binding=binding)
    profile.goals.insert(
        0,
        Goal(goal="Learn geometry", importance=0.4, progress=0.1),
    )
    store = SQLiteProfileStore(tmp_path / "profiles.sqlite")
    await store.save(profile)
    adapter = ProfileSnapshotAdapterV1(
        store=store,
        knowledge_graph=_knowledge_graph(),
    )

    snapshot = await adapter.load("user-1")

    assert snapshot is not None
    assert tuple(goal.goal for goal in snapshot.goals) == ("Learn algebra",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("schema", "profile_binding_schema_invalid"),
        ("topic", "profile_binding_topic_invalid"),
        ("skill_alias", "profile_skill_binding_invalid"),
        ("goal_fingerprint", "profile_goal_binding_invalid"),
        ("preference", "profile_preference_binding_invalid"),
    ),
)
async def test_profile_adapter_fails_fast_on_declared_invalid_binding(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    binding = _profile_binding(
        preferences=[
            {
                "signal_id": "preference.math.algebra.visual",
                "subject": "math",
                "topic_id": "math.algebra",
                "dimension": "prefer_visual",
                "strength": 0.8,
            }
        ]
    )
    profile = _profile(binding=binding)
    if mutation == "schema":
        binding["unexpected"] = True
    elif mutation == "topic":
        binding["skills"][0]["topic_id"] = "math.unknown"
    elif mutation == "skill_alias":
        profile.skills = {"algebra": SkillEntry(level=0.3, confidence=0.8)}
    elif mutation == "goal_fingerprint":
        binding["goals"][0]["goal_fingerprint"] = "0" * 64
    elif mutation == "preference":
        binding["preferences"][0]["strength"] = 0.7
    else:
        raise AssertionError(f"unhandled mutation: {mutation}")
    store = SQLiteProfileStore(tmp_path / f"{mutation}.sqlite")
    await store.save(profile)
    adapter = ProfileSnapshotAdapterV1(
        store=store,
        knowledge_graph=_knowledge_graph(),
    )

    with pytest.raises(ProfileAdapterError) as error:
        await adapter.load("user-1")

    assert error.value.code == expected_code


def _history_marker(**changes: object) -> dict[str, object]:
    marker: dict[str, object] = {
        "schema_version": "learning_guidance_history_event_v1",
        "topic_id": "math.algebra",
        "event_type": "assessment",
        "observed_at": "2026-07-15T10:00:00+00:00",
        "outcome_score": 0.4,
    }
    marker.update(changes)
    return marker


async def _save_memory(
    store: SQLiteMemoryStore,
    *,
    memory_id: str,
    metadata: dict[str, object],
    content: str = "content is not an adapter input",
) -> None:
    await store.save_episodic(
        EpisodicMemoryRecord(
            memory_id=memory_id,
            user_id="user-1",
            memory_type="quiz_attempt",
            content=content,
            subject="math",
            metadata=metadata,
            created_at="2026-07-15T10:00:00+00:00",
        )
    )


@pytest.mark.asyncio
async def test_history_adapter_projects_only_strict_tagged_metadata(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    await _save_memory(
        store,
        memory_id="memory-tagged",
        metadata={"learning_guidance_v1": _history_marker()},
    )
    await _save_memory(
        store,
        memory_id="memory-legacy",
        metadata={},
        content="assessment math.algebra outcome 1.0",
    )
    adapter = HistorySnapshotAdapterV1(
        store=store,
        knowledge_graph=_knowledge_graph(),
        history_limit=10,
    )

    snapshot = await adapter.load("user-1", "math")

    assert snapshot is not None
    assert tuple(event.history_id for event in snapshot.events) == ("memory-tagged",)
    assert snapshot.events[0].outcome_score == 0.4


@pytest.mark.asyncio
async def test_history_adapter_returns_unavailable_without_valid_markers(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    await _save_memory(store, memory_id="memory-legacy", metadata={})
    adapter = HistorySnapshotAdapterV1(
        store=store,
        knowledge_graph=_knowledge_graph(),
        history_limit=10,
    )

    assert await adapter.load("user-1", "math") is None


@pytest.mark.asyncio
async def test_history_adapter_rejects_explicit_null_v1_marker(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    await _save_memory(
        store,
        memory_id="memory-null",
        metadata={"learning_guidance_v1": None},
    )
    adapter = HistorySnapshotAdapterV1(
        store=store,
        knowledge_graph=_knowledge_graph(),
        history_limit=10,
    )

    with pytest.raises(HistoryAdapterError) as error:
        await adapter.load("user-1", "math")

    assert error.value.code == "history_binding_schema_invalid"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("marker", "expected_code"),
    (
        (
            _history_marker(unexpected=True),
            "history_binding_schema_invalid",
        ),
        (
            _history_marker(observed_at="2026-07-15T10:00:00"),
            "history_binding_schema_invalid",
        ),
        (
            _history_marker(topic_id="math.unknown"),
            "history_binding_topic_invalid",
        ),
    ),
)
async def test_history_adapter_fails_fast_on_declared_invalid_marker(
    tmp_path: Path,
    marker: dict[str, object],
    expected_code: str,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    await _save_memory(
        store,
        memory_id="memory-invalid",
        metadata={"learning_guidance_v1": marker},
    )
    adapter = HistorySnapshotAdapterV1(
        store=store,
        knowledge_graph=_knowledge_graph(),
        history_limit=10,
    )

    with pytest.raises(HistoryAdapterError) as error:
        await adapter.load("user-1", "math")

    assert error.value.code == expected_code


@pytest.mark.asyncio
async def test_history_adapter_propagates_strict_store_failures(
    tmp_path: Path,
) -> None:
    adapter = HistorySnapshotAdapterV1(
        store=SQLiteMemoryStore(tmp_path / "missing.sqlite"),
        knowledge_graph=_knowledge_graph(),
        history_limit=10,
    )

    with pytest.raises(MemoryStorageReadError) as error:
        await adapter.load("user-1", "math")

    assert error.value.code == "memory_database_missing"
