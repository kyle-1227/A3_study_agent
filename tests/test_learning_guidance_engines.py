"""Deterministic topic-local tests for path and recommendation engines."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.config.learning_guidance_config import load_learning_guidance_config
from src.learning_guidance.adapters.path import LearnerPathEngineV1, PathEngineError
from src.learning_guidance.adapters.recommendation import (
    RecommendationEngineError,
    ResourceRecommendationEngineV1,
)
from src.learning_guidance.contracts import (
    LearnerGoalSignalV1,
    LearnerHistoryEventV1,
    LearnerHistorySnapshotV1,
    LearnerPathEngineRequestV1,
    LearnerPreferenceSignalV1,
    LearnerProfileSnapshotV1,
    LearnerSkillSignalV1,
    RecommendationResourceContextV1,
    ResourceRecommendationEngineRequestV1,
)
from src.learning_guidance.knowledge_graph import KnowledgeGraphV1


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)


def _topic(
    topic_id: str,
    *,
    prerequisites: list[str],
    resources: list[tuple[str, str]],
) -> dict[str, object]:
    return {
        "topic_id": topic_id,
        "title": topic_id,
        "difficulty": 0.5,
        "estimated_hours": 2.0,
        "prerequisite_topic_ids": prerequisites,
        "knowledge_points": [f"point:{topic_id}"],
        "resources": [
            {
                "resource_id": resource_id,
                "resource_type": resource_type,
                "title": f"title:{resource_id}",
            }
            for resource_id, resource_type in resources
        ],
    }


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
                        _topic(
                            "math.foundations",
                            prerequisites=[],
                            resources=[("math.foundations.review", "review_doc")],
                        ),
                        _topic(
                            "math.algebra",
                            prerequisites=["math.foundations"],
                            resources=[
                                ("math.algebra.review", "review_doc"),
                                ("math.algebra.quiz", "quiz"),
                            ],
                        ),
                        _topic(
                            "math.calculus",
                            prerequisites=["math.algebra"],
                            resources=[("math.calculus.review", "review_doc")],
                        ),
                    ],
                }
            ],
        }
    )


def _skill(topic_id: str, level: float, confidence: float) -> LearnerSkillSignalV1:
    return LearnerSkillSignalV1(
        signal_id=f"skill:{topic_id}",
        subject="math",
        topic_id=topic_id,
        level=level,
        confidence=confidence,
    )


def _goal(
    topic_id: str,
    *,
    importance: float,
    progress: float,
) -> LearnerGoalSignalV1:
    return LearnerGoalSignalV1(
        signal_id=f"goal:{topic_id}",
        subject="math",
        topic_id=topic_id,
        goal=f"Learn {topic_id}",
        importance=importance,
        progress=progress,
    )


def _preference(
    topic_id: str,
    dimension: str,
    strength: float,
) -> LearnerPreferenceSignalV1:
    return LearnerPreferenceSignalV1.model_validate(
        {
            "signal_id": f"preference:{topic_id}:{dimension}",
            "subject": "math",
            "topic_id": topic_id,
            "dimension": dimension,
            "strength": strength,
        }
    )


def _event(
    topic_id: str,
    *,
    days_ago: int,
    outcome: float,
) -> LearnerHistoryEventV1:
    return LearnerHistoryEventV1(
        history_id=f"history:{topic_id}:{days_ago}",
        subject="math",
        topic_id=topic_id,
        event_type="assessment",
        observed_at=NOW - timedelta(days=days_ago),
        outcome_score=outcome,
    )


def _profile(*, include_preferences: bool = True) -> LearnerProfileSnapshotV1:
    preferences = (
        tuple(
            preference
            for topic_id in (
                "math.foundations",
                "math.algebra",
                "math.calculus",
            )
            for preference in (
                _preference(topic_id, "prefer_theory", 0.7),
                _preference(topic_id, "prefer_practice", 0.8),
            )
        )
        if include_preferences
        else ()
    )
    return LearnerProfileSnapshotV1(
        schema_version="learner_profile_snapshot_v1",
        user_id="user-1",
        skills=(
            _skill("math.foundations", 0.9, 0.9),
            _skill("math.algebra", 0.2, 0.8),
            _skill("math.calculus", 0.5, 0.8),
        ),
        goals=(
            _goal("math.foundations", importance=0.2, progress=0.9),
            _goal("math.algebra", importance=0.9, progress=0.2),
            _goal("math.calculus", importance=0.6, progress=0.1),
        ),
        preferences=preferences,
    )


def _history(*, future: bool = False) -> LearnerHistorySnapshotV1:
    algebra = _event("math.algebra", days_ago=2, outcome=0.4)
    if future:
        algebra = algebra.model_copy(update={"observed_at": NOW + timedelta(days=1)})
    return LearnerHistorySnapshotV1(
        schema_version="learner_history_snapshot_v1",
        user_id="user-1",
        subject="math",
        events=(
            _event("math.foundations", days_ago=1, outcome=0.9),
            algebra,
            _event("math.calculus", days_ago=10, outcome=0.7),
        ),
    )


def _path_request() -> LearnerPathEngineRequestV1:
    return LearnerPathEngineRequestV1(
        schema_version="learner_path_engine_request_v1",
        request_id="request-1",
        user_id="user-1",
        subject="math",
        requested_resource_types=("review_doc", "quiz"),
        profile=_profile(),
        history=_history(),
    )


@pytest.mark.asyncio
async def test_path_engine_uses_thresholds_topology_and_topic_local_evidence() -> None:
    config = load_learning_guidance_config(
        PROJECT_ROOT / "config" / "learning_guidance.yaml"
    )
    engine = LearnerPathEngineV1(
        knowledge_graph=_knowledge_graph(),
        policy=config.path_policy,
        clock=lambda: NOW,
    )

    plan = await engine.plan(_path_request())

    assert tuple(step.topic_id for step in plan.steps) == (
        "math.foundations",
        "math.algebra",
        "math.calculus",
    )
    assert tuple(step.status for step in plan.steps) == ("skip", "repeat", "blocked")
    assert plan.steps[1].recommended_resource_types == ("review_doc", "quiz")
    assert plan.steps[0].recommended_resource_types == ()
    for step in plan.steps:
        assert all(step.topic_id in signal_id for signal_id in step.profile_signal_ids)
        assert all(step.topic_id in history_id for history_id in step.history_ids)


@pytest.mark.asyncio
async def test_path_engine_rejects_future_history() -> None:
    config = load_learning_guidance_config(
        PROJECT_ROOT / "config" / "learning_guidance.yaml"
    )
    engine = LearnerPathEngineV1(
        knowledge_graph=_knowledge_graph(),
        policy=config.path_policy,
        clock=lambda: NOW,
    )
    request = _path_request().model_copy(update={"history": _history(future=True)})

    with pytest.raises(PathEngineError) as error:
        await engine.plan(request)

    assert error.value.code == "path_history_timestamp_invalid"


def _recommendation_request(
    *,
    mode: str,
    include_preferences: bool = True,
) -> ResourceRecommendationEngineRequestV1:
    generated = (
        (
            RecommendationResourceContextV1(
                resource_id="generated-review-1",
                resource_type="review_doc",
                subject="math",
                topic_id="math.algebra",
                title="Generated algebra review",
            ),
        )
        if mode == "automatic_after_generation"
        else ()
    )
    return ResourceRecommendationEngineRequestV1.model_validate(
        {
            "schema_version": "resource_recommendation_engine_request_v1",
            "request_id": "request-1",
            "mode": mode,
            "user_id": "user-1",
            "subject": "math",
            "profile": _profile(include_preferences=include_preferences),
            "history": _history(),
            "generated_resources": generated,
        }
    )


@pytest.mark.asyncio
async def test_automatic_recommendation_targets_only_real_generated_resource() -> None:
    config = load_learning_guidance_config(
        PROJECT_ROOT / "config" / "learning_guidance.yaml"
    )
    engine = ResourceRecommendationEngineV1(
        knowledge_graph=_knowledge_graph(),
        policy=config.recommendation_policy,
        clock=lambda: NOW,
    )

    result = await engine.recommend(
        _recommendation_request(mode="automatic_after_generation")
    )

    assert result.status == "available"
    assert result.batch is not None
    assert len(result.batch.items) == 1
    item = result.batch.items[0]
    assert item.resource_id == "generated-review-1"
    assert item.resource_type == "review_doc"
    assert item.subject == "math"
    assert item.topic_id == "math.algebra"
    assert item.title == "Generated algebra review"
    assert item.source_resource_ids == ("generated-review-1",)
    assert item.score_factors.weakness == pytest.approx(0.8)
    assert item.score_factors.forgetting == pytest.approx(2 / 30)
    assert item.score_factors.preference == 0.7
    assert item.score_factors.goal == pytest.approx(0.72)


@pytest.mark.asyncio
async def test_explicit_recommendation_uses_only_curated_catalog_resources() -> None:
    config = load_learning_guidance_config(
        PROJECT_ROOT / "config" / "learning_guidance.yaml"
    )
    engine = ResourceRecommendationEngineV1(
        knowledge_graph=_knowledge_graph(),
        policy=config.recommendation_policy,
        clock=lambda: NOW,
    )

    result = await engine.recommend(_recommendation_request(mode="explicit_request"))

    assert result.status == "available"
    assert result.batch is not None
    assert all(not item.source_resource_ids for item in result.batch.items)
    assert {item.resource_id for item in result.batch.items}.issubset(
        {
            "math.foundations.review",
            "math.algebra.review",
            "math.algebra.quiz",
            "math.calculus.review",
        }
    )
    expected_order = tuple(
        item.resource_id
        for item in sorted(
            result.batch.items,
            key=lambda item: (-item.score_factors.combined, item.resource_id),
        )
    )
    assert tuple(item.resource_id for item in result.batch.items) == expected_order


@pytest.mark.asyncio
async def test_recommendation_without_real_preference_is_explicitly_unavailable() -> (
    None
):
    config = load_learning_guidance_config(
        PROJECT_ROOT / "config" / "learning_guidance.yaml"
    )
    engine = ResourceRecommendationEngineV1(
        knowledge_graph=_knowledge_graph(),
        policy=config.recommendation_policy,
        clock=lambda: NOW,
    )

    result = await engine.recommend(
        _recommendation_request(
            mode="automatic_after_generation",
            include_preferences=False,
        )
    )

    assert result.status == "unavailable"
    assert result.unavailable_reason == "no_eligible_candidates"
    assert result.batch is None


@pytest.mark.asyncio
async def test_recommendation_rejects_unknown_generated_topic() -> None:
    config = load_learning_guidance_config(
        PROJECT_ROOT / "config" / "learning_guidance.yaml"
    )
    engine = ResourceRecommendationEngineV1(
        knowledge_graph=_knowledge_graph(),
        policy=config.recommendation_policy,
        clock=lambda: NOW,
    )
    request = _recommendation_request(mode="automatic_after_generation")
    invalid_context = request.generated_resources[0].model_copy(
        update={"topic_id": "math.unknown"}
    )
    request = request.model_copy(update={"generated_resources": (invalid_context,)})

    with pytest.raises(RecommendationEngineError) as error:
        await engine.recommend(request)

    assert error.value.code == "recommendation_resource_binding_invalid"
