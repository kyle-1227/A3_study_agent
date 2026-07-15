"""Strict assessment-to-guidance history persistence tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError
import pytest

from src.analytics.growth_analyzer import analyze_growth
from src.assessment.attempt_contracts import (
    AssessmentAttemptV1,
    AssessmentLearningGuidanceBindingV1,
    build_assessment_final_v1,
    stable_assessment_attempt_hash,
)
from src.assessment.attempt_journal import AssessmentAttemptRecordV1
from src.assessment.identity import stable_adaptive_practice_question_id
from src.config.learning_guidance_config import load_learning_guidance_config
from src.graph.learning_guidance import (
    make_resource_recommendation_node,
    resource_recommendation_output_from_state,
)
from src.learning_guidance.adapters.history import (
    HistoryAdapterError,
    HistorySnapshotAdapterV1,
)
from src.learning_guidance.factory import build_learning_guidance_runtime
from src.learning_guidance.history_contract import (
    LEARNING_GUIDANCE_HISTORY_ID_PREFIX,
)
from src.learning_guidance.history_writer import (
    LearningGuidanceHistoryWriterError,
    LearningGuidanceHistoryWriterV1,
)
from src.learning_guidance.knowledge_graph import KnowledgeGraphV1
from src.learning_guidance.profile_writer import (
    LearningGuidanceProfileWriteRequestV1,
    LearningGuidanceProfileWriterV1,
    ProfileGoalWriteV1,
    ProfilePreferenceWriteV1,
    ProfileSkillWriteV1,
    profile_write_source_for_request_v1,
)
from src.memory.schema import EpisodicMemoryRecord
from src.memory.storage import SQLiteMemoryStore
from src.profile.schema import Goal, SkillEntry, UserProfile
from src.profile.storage import SQLiteProfileStore


THREAD_ID = "thread-1"
USER_ID = "learner-1"
REQUEST_ID = "11111111-1111-4111-8111-111111111111"
GENERATION_REQUEST_ID = "22222222-2222-4222-8222-222222222222"
RESOURCE_ID = f"resource:v3:{'a' * 64}"
QUESTION_ID = f"question:v1:{'b' * 64}"
COMMITTED_AT = datetime(2026, 7, 15, 9, 30, tzinfo=timezone.utc)


def _knowledge_graph() -> KnowledgeGraphV1:
    return KnowledgeGraphV1.model_validate(
        {
            "schema_version": "knowledge_graph_v1",
            "data_version": "writer-test-v1",
            "subjects": [
                {
                    "subject_id": "math",
                    "title": "Mathematics",
                    "topics": [
                        {
                            "topic_id": "math.algebra",
                            "title": "Algebra",
                            "difficulty": 0.4,
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
                        },
                        {
                            "topic_id": "math.geometry",
                            "title": "Geometry",
                            "difficulty": 0.5,
                            "estimated_hours": 2.0,
                            "prerequisite_topic_ids": ["math.algebra"],
                            "knowledge_points": ["Triangles"],
                            "resources": [
                                {
                                    "resource_id": "math.geometry.review",
                                    "resource_type": "review_doc",
                                    "title": "Geometry review",
                                }
                            ],
                        },
                    ],
                }
            ],
        }
    )


def _binding(*, topic_id: str = "math.algebra") -> AssessmentLearningGuidanceBindingV1:
    return AssessmentLearningGuidanceBindingV1(
        schema_version="assessment_learning_guidance_binding_v1",
        user_id=USER_ID,
        subject="math",
        topic_id=topic_id,
        resource_type="quiz",
        generation_request_id=GENERATION_REQUEST_ID,
        assignment_contract_version="resource_evidence_assignment_v1",
        assignment_fingerprint="c" * 64,
    )


def _record(*, is_correct: bool = True) -> AssessmentAttemptRecordV1:
    attempt = AssessmentAttemptV1(
        schema_version="assessment_attempt_v1",
        request_id=REQUEST_ID,
        resource_id=RESOURCE_ID,
        question_id=QUESTION_ID,
        answer="private submitted answer",
        time_spent_seconds=12.0,
    )
    adaptive_question = "A new review question."
    adaptive_tags = ("algebra",)
    final = build_assessment_final_v1(
        thread_id=THREAD_ID,
        attempt=attempt,
        is_correct=is_correct,
        error_classification=(
            None
            if is_correct
            else {
                "schema_version": "assessment_error_classification_v1",
                "error_type": "concept",
                "concept_gap": "A bounded public diagnosis.",
                "suggestion": "Review the prerequisite concept.",
                "confidence": 0.8,
            }
        ),
        adaptive_tasks=(
            ()
            if is_correct
            else (
                {
                    "schema_version": "adaptive_practice_task_v1",
                    "question_id": stable_adaptive_practice_question_id(
                        task_type="review",
                        question=adaptive_question,
                        tags=adaptive_tags,
                        difficulty=0.3,
                    ),
                    "task_type": "review",
                    "question": adaptive_question,
                    "answer": "A complete answer.",
                    "explanation": "A complete explanation.",
                    "reason": "Targets the diagnosed gap.",
                    "tags": adaptive_tags,
                    "difficulty": 0.3,
                },
            )
        ),
    )
    return AssessmentAttemptRecordV1(
        schema_version="assessment_attempt_record_v1",
        request_id=REQUEST_ID,
        request_hash=stable_assessment_attempt_hash(
            thread_id=THREAD_ID,
            attempt=attempt,
        ),
        status="completed",
        started_at=datetime(2026, 7, 15, 9, 29, tzinfo=timezone.utc),
        committed_at=COMMITTED_AT,
        final=final,
        error_code="",
        failure_stage="",
        exception_type="",
    )


@pytest.mark.asyncio
async def test_writer_reader_round_trip_is_reopen_safe_and_idempotent(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.sqlite"
    writer = LearningGuidanceHistoryWriterV1(
        store=SQLiteMemoryStore(database),
        knowledge_graph=_knowledge_graph(),
    )

    inserted = await writer.write_assessment_once(
        binding=_binding(),
        record=_record(),
    )
    replayed = await LearningGuidanceHistoryWriterV1(
        store=SQLiteMemoryStore(database),
        knowledge_graph=_knowledge_graph(),
    ).write_assessment_once(binding=_binding(), record=_record())
    snapshot = await HistorySnapshotAdapterV1(
        store=SQLiteMemoryStore(database),
        knowledge_graph=_knowledge_graph(),
        history_limit=10,
    ).load(USER_ID, "math")

    assert inserted.status == "inserted"
    assert replayed.status == "replayed"
    assert inserted.history_id == replayed.history_id
    assert inserted.history_id.startswith(LEARNING_GUIDANCE_HISTORY_ID_PREFIX)
    assert snapshot is not None
    assert len(snapshot.events) == 1
    assert snapshot.events[0].history_id == inserted.history_id
    assert snapshot.events[0].observed_at == COMMITTED_AT
    assert snapshot.events[0].outcome_score == 1.0


@pytest.mark.asyncio
async def test_incorrect_assessment_records_zero_without_private_answer_text(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    result = await LearningGuidanceHistoryWriterV1(
        store=store,
        knowledge_graph=_knowledge_graph(),
    ).write_assessment_once(binding=_binding(), record=_record(is_correct=False))

    records = await store.get_episodic_by_ids([result.history_id])

    assert len(records) == 1
    record = records[0]
    assert record.metadata["learning_guidance_v1"]["outcome_score"] == 0.0
    serialized = record.model_dump_json()
    assert "private submitted answer" not in serialized
    assert "A complete answer." not in serialized
    assert "assessment_request_hash" not in serialized
    assert "assessment-attempt:v1:" not in serialized
    assert record.embedding is None

    growth = await analyze_growth(
        USER_ID,
        subject="math",
        days=3650,
        store=store,
    )
    assert growth.total_events == 1
    assert growth.overall_accuracy == 0.0
    assert growth.series[0].topic == "math.algebra"


@pytest.mark.asyncio
async def test_same_assessment_source_with_drift_is_a_conflict(tmp_path: Path) -> None:
    writer = LearningGuidanceHistoryWriterV1(
        store=SQLiteMemoryStore(tmp_path / "memory.sqlite"),
        knowledge_graph=_knowledge_graph(),
    )
    await writer.write_assessment_once(binding=_binding(), record=_record())

    with pytest.raises(LearningGuidanceHistoryWriterError) as error:
        await writer.write_assessment_once(
            binding=_binding(topic_id="math.geometry"),
            record=_record(),
        )

    assert error.value.code == "learning_guidance_history_event_conflict"


@pytest.mark.asyncio
async def test_writer_rejects_topic_outside_the_bound_subject(tmp_path: Path) -> None:
    writer = LearningGuidanceHistoryWriterV1(
        store=SQLiteMemoryStore(tmp_path / "memory.sqlite"),
        knowledge_graph=_knowledge_graph(),
    )

    with pytest.raises(LearningGuidanceHistoryWriterError) as error:
        await writer.write_assessment_once(
            binding=_binding(topic_id="math.unknown"),
            record=_record(),
        )

    assert error.value.code == "assessment_history_topic_invalid"


def test_assessment_binding_rejects_reserved_user_identity() -> None:
    payload = _binding().model_dump(mode="json")
    payload["user_id"] = "unknown"

    with pytest.raises(ValidationError):
        AssessmentLearningGuidanceBindingV1.model_validate(payload, strict=True)


@pytest.mark.asyncio
async def test_writer_revalidates_a_forged_binding_instance(tmp_path: Path) -> None:
    binding = _binding()
    object.__setattr__(binding, "user_id", "unknown")
    writer = LearningGuidanceHistoryWriterV1(
        store=SQLiteMemoryStore(tmp_path / "memory.sqlite"),
        knowledge_graph=_knowledge_graph(),
    )

    with pytest.raises(LearningGuidanceHistoryWriterError) as error:
        await writer.write_assessment_once(binding=binding, record=_record())

    assert error.value.code == "assessment_history_source_invalid"


@pytest.mark.asyncio
async def test_writer_rejects_mutated_nested_python_collection_shape(
    tmp_path: Path,
) -> None:
    record = _record(is_correct=False)
    assert record.final is not None
    object.__setattr__(
        record.final, "adaptive_tasks", list(record.final.adaptive_tasks)
    )
    writer = LearningGuidanceHistoryWriterV1(
        store=SQLiteMemoryStore(tmp_path / "memory.sqlite"),
        knowledge_graph=_knowledge_graph(),
    )

    with pytest.raises(LearningGuidanceHistoryWriterError) as error:
        await writer.write_assessment_once(binding=_binding(), record=record)

    assert error.value.code == "assessment_history_source_invalid"


@pytest.mark.asyncio
async def test_declared_history_id_without_marker_fails_closed(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    await store.save_episodic(
        EpisodicMemoryRecord(
            memory_id=f"{LEARNING_GUIDANCE_HISTORY_ID_PREFIX}{'d' * 64}",
            user_id=USER_ID,
            memory_type="quiz_attempt",
            content="bounded invalid fact",
            subject="math",
            metadata={"unrelated": "marker"},
        )
    )

    with pytest.raises(HistoryAdapterError) as error:
        await HistorySnapshotAdapterV1(
            store=store,
            knowledge_graph=_knowledge_graph(),
            history_limit=10,
        ).load(USER_ID, "math")

    assert error.value.code == "history_binding_schema_invalid"


@pytest.mark.asyncio
async def test_ordinary_rows_do_not_consume_the_guidance_history_limit(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    writer = LearningGuidanceHistoryWriterV1(
        store=store,
        knowledge_graph=_knowledge_graph(),
    )
    expected = await writer.write_assessment_once(
        binding=_binding(),
        record=_record(),
    )
    for index in range(501):
        await store.save_episodic(
            EpisodicMemoryRecord(
                memory_id=(
                    f"{LEARNING_GUIDANCE_HISTORY_ID_PREFIX.upper()}ordinary-{index:03d}"
                ),
                user_id=USER_ID,
                memory_type="key_conversation",
                content=f"ordinary row {index}",
                subject="math",
                created_at=(f"2026-07-16T{index // 60:02d}:{index % 60:02d}:00+00:00"),
            )
        )

    snapshot = await HistorySnapshotAdapterV1(
        store=SQLiteMemoryStore(store.db_path),
        knowledge_graph=_knowledge_graph(),
        history_limit=1,
    ).load(USER_ID, "math")

    assert snapshot is not None
    assert tuple(event.history_id for event in snapshot.events) == (
        expected.history_id,
    )
    growth = await analyze_growth(
        USER_ID,
        subject="math",
        days=3650,
        store=SQLiteMemoryStore(store.db_path),
    )
    assert growth.total_events == 1
    assert growth.overall_accuracy == 1.0
    assert (
        sum(
            point.event_count
            for series in growth.series
            for point in series.data_points
        )
        == 1
    )


@pytest.mark.asyncio
async def test_recommendation_becomes_available_only_after_durable_assessment(
    tmp_path: Path,
) -> None:
    """Exercise profile/history writers through reopened production adapters."""

    profile_path = tmp_path / "profile.sqlite"
    memory_path = tmp_path / "memory.sqlite"
    await SQLiteMemoryStore(memory_path).initialize()
    knowledge_graph = _knowledge_graph()
    observed_at = "2026-07-15T09:00:00+00:00"
    profile = UserProfile(
        user_id=USER_ID,
        skills={
            "math.algebra": SkillEntry(
                level=0.3,
                confidence=0.8,
                last_observed=observed_at,
                evidence_count=1,
            )
        },
        goals=[
            Goal(
                goal="Master algebra",
                importance=0.9,
                progress=0.2,
                created_at=observed_at,
            )
        ],
    )
    profile.learning_style.prefer_theory = 0.8
    profile_request = LearningGuidanceProfileWriteRequestV1(
        schema_version="learning_guidance_profile_write_request_v1",
        request_id="profile-request-e2e",
        user_id=USER_ID,
        skills=[
            ProfileSkillWriteV1(
                subject="math",
                topic_id="math.algebra",
                level=0.3,
                confidence=0.8,
            )
        ],
        goals=[
            ProfileGoalWriteV1(
                subject="math",
                topic_id="math.algebra",
                goal="Master algebra",
                importance=0.9,
                progress=0.2,
            )
        ],
        preferences=[
            ProfilePreferenceWriteV1(
                subject="math",
                topic_id="math.algebra",
                dimension="prefer_theory",
                strength=0.8,
            )
        ],
    )
    await LearningGuidanceProfileWriterV1(
        store=SQLiteProfileStore(profile_path),
        knowledge_graph=knowledge_graph,
    ).create_once(
        profile_request,
        base_profile=profile,
        source=profile_write_source_for_request_v1(profile_request),
    )
    config = load_learning_guidance_config(
        Path(__file__).resolve().parents[1] / "config" / "learning_guidance.yaml"
    )

    def reopened_runtime():
        return build_learning_guidance_runtime(
            config=config,
            knowledge_graph=knowledge_graph,
            profile_db_path=profile_path,
            memory_db_path=memory_path,
            clock=lambda: datetime(2026, 7, 16, 9, 30, tzinfo=timezone.utc),
        )

    state = {
        "request_id": "recommendation-request-e2e",
        "user_id": USER_ID,
        "subject": "math",
        "subject_candidates": ["math"],
        "workspace_continuation_applied": False,
    }
    first_update = await make_resource_recommendation_node(
        reopened_runtime(),
        mode="explicit_request",
    )(state)  # type: ignore[arg-type]
    first = resource_recommendation_output_from_state(
        {**state, **first_update},
        expected_mode="explicit_request",
    )
    assert first.status == "unavailable"
    assert first.unavailable_reason == "history_unavailable"

    history_result = await LearningGuidanceHistoryWriterV1(
        store=SQLiteMemoryStore(memory_path),
        knowledge_graph=knowledge_graph,
    ).write_assessment_once(binding=_binding(), record=_record(is_correct=False))

    second_update = await make_resource_recommendation_node(
        reopened_runtime(),
        mode="explicit_request",
    )(state)  # type: ignore[arg-type]
    second = resource_recommendation_output_from_state(
        {**state, **second_update},
        expected_mode="explicit_request",
    )
    assert second.status == "available"
    assert second.unavailable_reason is None
    assert second.batch is not None
    assert second.batch.items[0].topic_id == "math.algebra"
    assert second.batch.items[0].history_ids == (history_result.history_id,)
