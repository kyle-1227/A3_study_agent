"""Strict create-once and evolution tests for the guidance profile writer."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import ValidationError
import pytest

from src.learning_guidance.adapters.profile import (
    LearningGuidanceProfileBindingV1,
    ProfileSnapshotAdapterV1,
)
from src.learning_guidance.knowledge_graph import KnowledgeGraphV1
from src.learning_guidance.profile_writer import (
    CompiledLearningGuidanceProfileWriteV1,
    LearningGuidanceProfileWriteRequestV1,
    LearningGuidanceProfileWriterV1,
    PROFILE_BINDING_EXTRA_KEY,
    PROFILE_SIGNAL_ID_PREFIX,
    PROFILE_WRITE_RECEIPT_EXTRA_KEY,
    ProfileGoalWriteV1,
    ProfilePreferenceWriteV1,
    ProfileSkillWriteV1,
    ProfileWriterError,
    build_profile_write_source_v1,
    compile_profile_write_request_v1,
    profile_write_source_for_request_v1,
    stable_profile_write_request_hash,
)
from src.profile.schema import ExtractedProfileInfo, Goal, SkillEntry, UserProfile
from src.profile.storage import SQLiteProfileStore


def _knowledge_graph() -> KnowledgeGraphV1:
    return KnowledgeGraphV1.model_validate(
        {
            "schema_version": "knowledge_graph_v1",
            "data_version": "profile-writer-tests-v1",
            "subjects": [
                {
                    "subject_id": "math",
                    "title": "Mathematics",
                    "topics": [
                        {
                            "topic_id": "math.algebra",
                            "title": "Algebra",
                            "difficulty": 0.4,
                            "estimated_hours": 8.0,
                            "prerequisite_topic_ids": [],
                            "knowledge_points": ["Equations"],
                            "resources": [
                                {
                                    "resource_id": "math-algebra-review",
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


def _profile(*, level: float = 0.3, importance: float = 0.9) -> UserProfile:
    profile = UserProfile(
        user_id="user-1",
        skills={
            "math.algebra": SkillEntry(
                level=level,
                confidence=0.8,
                last_observed="2026-07-15T10:00:00+00:00",
                evidence_count=1,
            )
        },
        goals=[
            Goal(
                goal="Learn algebra",
                importance=importance,
                progress=0.2,
                created_at="2026-07-15T10:00:00+00:00",
            )
        ],
        extra={"nickname": "learner"},
    )
    profile.learning_style.prefer_visual = 0.8
    return profile


def _request(
    *,
    request_id: str = "profile-request-1",
    level: float = 0.3,
    importance: float = 0.9,
    topic_id: str = "math.algebra",
) -> LearningGuidanceProfileWriteRequestV1:
    return LearningGuidanceProfileWriteRequestV1(
        schema_version="learning_guidance_profile_write_request_v1",
        request_id=request_id,
        user_id="user-1",
        skills=[
            ProfileSkillWriteV1(
                subject="math",
                topic_id=topic_id,
                level=level,
                confidence=0.8,
            ),
        ],
        goals=[
            ProfileGoalWriteV1(
                subject="math",
                topic_id=topic_id,
                goal="Learn algebra",
                importance=importance,
                progress=0.2,
            ),
        ],
        preferences=[
            ProfilePreferenceWriteV1(
                subject="math",
                topic_id=topic_id,
                dimension="prefer_visual",
                strength=0.8,
            ),
        ],
    )


@pytest.mark.asyncio
async def test_create_once_reopens_through_the_strict_reader(tmp_path: Path) -> None:
    database = tmp_path / "profiles.sqlite"
    writer = LearningGuidanceProfileWriterV1(
        store=SQLiteProfileStore(database),
        knowledge_graph=_knowledge_graph(),
    )
    request = _request()

    result = await writer.create_once(
        request,
        base_profile=_profile(),
        source=profile_write_source_for_request_v1(request),
    )

    assert result.status == "created"
    assert result.request_hash == stable_profile_write_request_hash(request)
    assert all(
        signal.signal_id.startswith(PROFILE_SIGNAL_ID_PREFIX)
        for signal in (
            *result.binding.skills,
            *result.binding.goals,
            *result.binding.preferences,
        )
    )
    reopened_store = SQLiteProfileStore(database)
    reopened = await reopened_store.load_strict("user-1")
    assert reopened is not None
    assert PROFILE_BINDING_EXTRA_KEY in reopened.extra
    assert PROFILE_WRITE_RECEIPT_EXTRA_KEY in reopened.extra

    snapshot = await ProfileSnapshotAdapterV1(
        store=reopened_store,
        knowledge_graph=_knowledge_graph(),
    ).load("user-1")
    assert snapshot is not None
    assert snapshot.user_id == "user-1"
    assert snapshot.skills[0].topic_id == "math.algebra"
    assert snapshot.goals[0].goal == "Learn algebra"
    assert snapshot.preferences[0].strength == 0.8


@pytest.mark.asyncio
async def test_identical_replay_is_a_no_write_after_sqlite_reopen(
    tmp_path: Path,
) -> None:
    database = tmp_path / "profiles.sqlite"
    request = _request()
    first_writer = LearningGuidanceProfileWriterV1(
        store=SQLiteProfileStore(database),
        knowledge_graph=_knowledge_graph(),
    )
    await first_writer.create_once(
        request,
        base_profile=_profile(),
        source=profile_write_source_for_request_v1(request),
    )
    before = await SQLiteProfileStore(database).load_strict("user-1")

    replay = await LearningGuidanceProfileWriterV1(
        store=SQLiteProfileStore(database),
        knowledge_graph=_knowledge_graph(),
    ).create_once(
        request,
        base_profile=_profile(),
        source=profile_write_source_for_request_v1(request),
    )
    after = await SQLiteProfileStore(database).load_strict("user-1")

    assert replay.status == "replayed"
    assert after == before


@pytest.mark.asyncio
async def test_concurrent_identical_create_is_one_create_and_one_replay(
    tmp_path: Path,
) -> None:
    database = tmp_path / "profiles.sqlite"
    store = SQLiteProfileStore(database)
    await store.save(_profile())
    writer = LearningGuidanceProfileWriterV1(
        store=store,
        knowledge_graph=_knowledge_graph(),
    )
    request = _request()

    results = await asyncio.gather(
        writer.create_once(
            request,
            base_profile=_profile(),
            source=profile_write_source_for_request_v1(request),
        ),
        writer.create_once(
            request,
            base_profile=_profile(),
            source=profile_write_source_for_request_v1(request),
        ),
    )

    assert sorted(result.status for result in results) == ["created", "replayed"]
    snapshot = await ProfileSnapshotAdapterV1(
        store=SQLiteProfileStore(database),
        knowledge_graph=_knowledge_graph(),
    ).load("user-1")
    assert snapshot is not None


@pytest.mark.asyncio
async def test_onboard_v2_applies_command_fields_to_an_unbound_existing_profile(
    tmp_path: Path,
) -> None:
    database = tmp_path / "profiles.sqlite"
    store = SQLiteProfileStore(database)
    existing = _profile()
    existing.dislikes = ["historical dislike"]
    existing.tags = ["history-tag"]
    existing.behavior.total_sessions = 7
    existing.extra.update(
        {
            "nickname": "historical nickname",
            "grade": "historical grade",
            "history_marker": {"turns": 11},
        }
    )
    await store.save(existing)
    before = await store.load_strict("user-1")
    assert before is not None

    request = _request(request_id="onboard-request-1")
    base_profile = _profile()
    base_profile.dislikes = ["rote repetition"]
    base_profile.extra.update(
        {
            "nickname": "onboard nickname",
            "grade": "grade-10",
        }
    )
    source = build_profile_write_source_v1(
        source_kind="onboard_v2",
        payload={
            "schema_version": "onboard_v2",
            "profile": request.model_dump(mode="json"),
            "nickname": "onboard nickname",
            "grade": "grade-10",
            "dislikes": ["rote repetition"],
        },
    )

    result = await LearningGuidanceProfileWriterV1(
        store=store,
        knowledge_graph=_knowledge_graph(),
    ).create_once(
        request,
        base_profile=base_profile,
        source=source,
    )
    reopened = await SQLiteProfileStore(database).load_strict("user-1")

    assert result.status == "created"
    assert reopened == result.profile
    assert reopened is not None
    assert reopened.extra["nickname"] == "onboard nickname"
    assert reopened.extra["grade"] == "grade-10"
    assert reopened.dislikes == ["rote repetition"]
    assert reopened.extra["history_marker"] == {"turns": 11}
    assert reopened.skills == before.skills
    assert reopened.goals == before.goals
    assert reopened.learning_style == before.learning_style
    assert reopened.behavior == before.behavior
    assert reopened.agent_observations == before.agent_observations
    assert reopened.tags == before.tags
    assert reopened.created_at == before.created_at


@pytest.mark.asyncio
async def test_signal_ids_ignore_request_identity_but_request_hash_does_not(
    tmp_path: Path,
) -> None:
    first_request = _request(request_id="profile-request-1")
    second_request = _request(request_id="profile-request-2")
    first = await LearningGuidanceProfileWriterV1(
        store=SQLiteProfileStore(tmp_path / "first.sqlite"),
        knowledge_graph=_knowledge_graph(),
    ).create_once(
        first_request,
        base_profile=_profile(),
        source=profile_write_source_for_request_v1(first_request),
    )
    second = await LearningGuidanceProfileWriterV1(
        store=SQLiteProfileStore(tmp_path / "second.sqlite"),
        knowledge_graph=_knowledge_graph(),
    ).create_once(
        second_request,
        base_profile=_profile(),
        source=profile_write_source_for_request_v1(second_request),
    )

    assert first.request_hash != second.request_hash
    assert tuple(
        signal.signal_id
        for signal in (
            *first.binding.skills,
            *first.binding.goals,
            *first.binding.preferences,
        )
    ) == tuple(
        signal.signal_id
        for signal in (
            *second.binding.skills,
            *second.binding.goals,
            *second.binding.preferences,
        )
    )


@pytest.mark.asyncio
async def test_request_drift_conflicts_without_changing_the_row(tmp_path: Path) -> None:
    database = tmp_path / "profiles.sqlite"
    writer = LearningGuidanceProfileWriterV1(
        store=SQLiteProfileStore(database),
        knowledge_graph=_knowledge_graph(),
    )
    initial_request = _request()
    await writer.create_once(
        initial_request,
        base_profile=_profile(),
        source=profile_write_source_for_request_v1(initial_request),
    )
    before = await SQLiteProfileStore(database).load_strict("user-1")

    with pytest.raises(ProfileWriterError) as error:
        await writer.create_once(
            _request(level=0.4),
            base_profile=_profile(level=0.4),
            source=profile_write_source_for_request_v1(_request(level=0.4)),
        )

    assert error.value.code == "profile_write_request_conflict"
    assert await SQLiteProfileStore(database).load_strict("user-1") == before


@pytest.mark.asyncio
async def test_unknown_topic_fails_before_creating_storage(tmp_path: Path) -> None:
    database = tmp_path / "profiles.sqlite"
    writer = LearningGuidanceProfileWriterV1(
        store=SQLiteProfileStore(database),
        knowledge_graph=_knowledge_graph(),
    )

    with pytest.raises(ProfileWriterError) as error:
        await writer.create_once(
            _request(topic_id="math.unknown"),
            base_profile=_profile(),
            source=profile_write_source_for_request_v1(
                _request(topic_id="math.unknown")
            ),
        )

    assert error.value.code == "profile_write_topic_invalid"
    assert not database.exists()


def test_request_contract_rejects_schema_drift_and_unnormalized_text() -> None:
    payload = _request().model_dump(mode="json")
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        LearningGuidanceProfileWriteRequestV1.model_validate(payload)

    payload = _request().model_dump(mode="json")
    payload["user_id"] = "unknown"
    with pytest.raises(ValidationError):
        LearningGuidanceProfileWriteRequestV1.model_validate(payload, strict=True)


@pytest.mark.parametrize("field_name", ["goals", "preferences"])
def test_request_contract_rejects_same_subject_topic_drift(field_name: str) -> None:
    payload = _request().model_dump(mode="json")
    payload[field_name][0]["topic_id"] = "math.geometry"

    with pytest.raises(ValidationError):
        LearningGuidanceProfileWriteRequestV1.model_validate(payload, strict=True)


def test_preference_dimension_must_cover_every_bound_topic() -> None:
    payload = _request().model_dump(mode="json")
    payload["skills"].append(
        {
            "subject": "math",
            "topic_id": "math.geometry",
            "level": 0.2,
            "confidence": 0.7,
        }
    )
    payload["goals"].append(
        {
            "subject": "math",
            "topic_id": "math.geometry",
            "goal": "Learn geometry",
            "importance": 0.8,
            "progress": 0.1,
        }
    )

    with pytest.raises(ValidationError):
        LearningGuidanceProfileWriteRequestV1.model_validate(payload, strict=True)


def test_goal_text_must_be_globally_unique_across_topics() -> None:
    payload = _request().model_dump(mode="json")
    payload["skills"].append(
        {
            "subject": "math",
            "topic_id": "math.geometry",
            "level": 0.2,
            "confidence": 0.7,
        }
    )
    payload["goals"].append(
        {
            "subject": "math",
            "topic_id": "math.geometry",
            "goal": "Learn algebra",
            "importance": 0.8,
            "progress": 0.1,
        }
    )
    payload["preferences"].append(
        {
            "subject": "math",
            "topic_id": "math.geometry",
            "dimension": "prefer_visual",
            "strength": 0.8,
        }
    )

    with pytest.raises(ValidationError):
        LearningGuidanceProfileWriteRequestV1.model_validate(payload, strict=True)


@pytest.mark.parametrize(
    "invalid_sequence",
    [
        (
            ProfileSkillWriteV1(
                subject="math",
                topic_id="math.algebra",
                level=0.3,
                confidence=0.8,
            ),
        ),
        {
            ProfileSkillWriteV1(
                subject="math",
                topic_id="math.algebra",
                level=0.3,
                confidence=0.8,
            )
        },
        (item for item in []),
        "not-a-list",
    ],
    ids=["tuple", "set", "generator", "string"],
)
def test_public_request_rejects_non_list_sequences(invalid_sequence: object) -> None:
    payload = _request().model_dump(mode="python")
    payload["skills"] = invalid_sequence

    with pytest.raises(ValidationError):
        LearningGuidanceProfileWriteRequestV1.model_validate(payload, strict=True)


def test_public_request_accepts_json_lists_and_compiles_deeply_immutable() -> None:
    request = LearningGuidanceProfileWriteRequestV1.model_validate_json(
        _request().model_dump_json(),
        strict=True,
    )

    compiled = compile_profile_write_request_v1(request)

    assert isinstance(compiled, CompiledLearningGuidanceProfileWriteV1)
    assert isinstance(compiled.skills, tuple)
    assert isinstance(compiled.goals, tuple)
    assert isinstance(compiled.preferences, tuple)
    with pytest.raises(AttributeError):
        compiled.skills = ()  # type: ignore[misc]
    with pytest.raises(ValidationError):
        compiled.skills[0].subject = "physics"  # type: ignore[misc]


def test_compiler_revalidates_a_mutated_existing_model_instance() -> None:
    request = _request()
    request.skills.append("invalid")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        compile_profile_write_request_v1(request)

    payload = _request().model_dump(mode="json")
    payload["user_id"] = " user-1"
    with pytest.raises(ValidationError):
        LearningGuidanceProfileWriteRequestV1.model_validate(payload)


@pytest.mark.asyncio
async def test_evolution_refreshes_only_existing_binding_slots(
    tmp_path: Path,
) -> None:
    database = tmp_path / "profiles.sqlite"
    writer = LearningGuidanceProfileWriterV1(
        store=SQLiteProfileStore(database),
        knowledge_graph=_knowledge_graph(),
    )
    create_request = _request()
    created = await writer.create_once(
        create_request,
        base_profile=_profile(),
        source=profile_write_source_for_request_v1(create_request),
    )
    original_goal_fingerprint = created.binding.goals[0].goal_fingerprint
    original_signal_ids = tuple(
        signal.signal_id
        for signal in (
            *created.binding.skills,
            *created.binding.goals,
            *created.binding.preferences,
        )
    )
    original_receipt = created.profile.extra[PROFILE_WRITE_RECEIPT_EXTRA_KEY]

    evolved = await writer.evolve_existing_binding(
        user_id="user-1",
        extracted=ExtractedProfileInfo(
            skills_observed={"free-text-skill": 0.6},
            goals_observed=[
                {"goal": "Learn algebra", "importance": 0.5},
                {"goal": "New free goal", "importance": 0.7},
            ],
            style_signals={"prefer_visual": 0.2},
        ),
    )

    assert "free-text-skill" in evolved.profile.skills
    assert any(goal.goal == "New free goal" for goal in evolved.profile.goals)
    assert evolved.profile.extra[PROFILE_WRITE_RECEIPT_EXTRA_KEY] == original_receipt
    snapshot = await ProfileSnapshotAdapterV1(
        store=SQLiteProfileStore(database),
        knowledge_graph=_knowledge_graph(),
    ).load("user-1")
    assert snapshot is not None
    assert tuple(signal.topic_id for signal in snapshot.skills) == ("math.algebra",)
    assert tuple(signal.goal for signal in snapshot.goals) == ("Learn algebra",)
    assert len(snapshot.preferences) == 1
    assert (
        snapshot.preferences[0].strength == evolved.profile.learning_style.prefer_visual
    )

    binding = LearningGuidanceProfileBindingV1.model_validate(
        evolved.profile.extra[PROFILE_BINDING_EXTRA_KEY]
    )
    assert binding.goals[0].goal_fingerprint != original_goal_fingerprint
    assert (
        tuple(
            signal.signal_id
            for signal in (*binding.skills, *binding.goals, *binding.preferences)
        )
        == original_signal_ids
    )

    before_replay = await SQLiteProfileStore(database).load_strict("user-1")
    replay_request = _request()
    replay = await writer.create_once(
        replay_request,
        base_profile=evolved.profile,
        source=profile_write_source_for_request_v1(replay_request),
    )
    assert replay.status == "replayed"
    assert await SQLiteProfileStore(database).load_strict("user-1") == before_replay


@pytest.mark.asyncio
async def test_evolution_requires_an_existing_writer_binding(tmp_path: Path) -> None:
    database = tmp_path / "profiles.sqlite"
    store = SQLiteProfileStore(database)
    await store.save(_profile())
    before = await SQLiteProfileStore(database).load_strict("user-1")
    writer = LearningGuidanceProfileWriterV1(
        store=SQLiteProfileStore(database),
        knowledge_graph=_knowledge_graph(),
    )

    with pytest.raises(ProfileWriterError) as error:
        await writer.evolve_existing_binding(
            user_id="user-1",
            extracted=ExtractedProfileInfo(
                skills_observed={"unbound": 0.4},
            ),
        )

    assert error.value.code == "profile_write_binding_missing"
    assert await SQLiteProfileStore(database).load_strict("user-1") == before
