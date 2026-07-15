"""Production factory tests for real learning-guidance adapter composition."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from src.config.learning_guidance_config import (
    LearningGuidanceConfigV1,
    load_learning_guidance_config,
)
from src.learning_guidance.contracts import (
    LearnerPathEngineRequestV1,
    RecommendationResourceContextV1,
    ResourceRecommendationEngineRequestV1,
)
from src.learning_guidance.adapters.profile import profile_goal_fingerprint
from src.learning_guidance.factory import (
    build_learning_guidance_runtime,
    load_learning_guidance_runtime,
    resolve_knowledge_graph_path,
)
from src.learning_guidance.knowledge_graph import (
    KnowledgeGraphPathError,
    KnowledgeGraphV1,
)
from src.memory.schema import EpisodicMemoryRecord
from src.memory.storage import SQLiteMemoryStore
from src.profile.schema import Goal, SkillEntry, UserProfile
from src.profile.storage import SQLiteProfileStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)


def _config() -> LearningGuidanceConfigV1:
    return load_learning_guidance_config(
        PROJECT_ROOT / "config" / "learning_guidance.yaml"
    )


def _knowledge_graph(*, data_version: str = "test-v1") -> KnowledgeGraphV1:
    return KnowledgeGraphV1.model_validate(
        {
            "schema_version": "knowledge_graph_v1",
            "data_version": data_version,
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


@pytest.mark.asyncio
async def test_factory_composes_four_real_adapters_end_to_end(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.sqlite"
    memory_path = tmp_path / "memory.sqlite"
    algebra_goal = Goal(goal="Learn algebra", importance=0.9, progress=0.2)
    profile = UserProfile(
        user_id="user-1",
        skills={"math.algebra": SkillEntry(level=0.3, confidence=0.8)},
        goals=[algebra_goal],
        extra={
            "learning_guidance_v1": {
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
                        "goal_fingerprint": profile_goal_fingerprint(algebra_goal),
                    }
                ],
                "preferences": [
                    {
                        "signal_id": "preference.math.algebra.theory",
                        "subject": "math",
                        "topic_id": "math.algebra",
                        "dimension": "prefer_theory",
                        "strength": 0.8,
                    }
                ],
            }
        },
    )
    profile.learning_style.prefer_theory = 0.8
    await SQLiteProfileStore(profile_path).save(profile)
    await SQLiteMemoryStore(memory_path).save_episodic(
        EpisodicMemoryRecord(
            memory_id="memory-1",
            user_id="user-1",
            memory_type="quiz_attempt",
            content="private body",
            subject="math",
            metadata={
                "learning_guidance_v1": {
                    "schema_version": "learning_guidance_history_event_v1",
                    "topic_id": "math.algebra",
                    "event_type": "assessment",
                    "observed_at": "2026-07-14T12:00:00+00:00",
                    "outcome_score": 0.4,
                }
            },
            created_at="2026-07-14T12:00:00+00:00",
        )
    )
    runtime = build_learning_guidance_runtime(
        config=_config(),
        knowledge_graph=_knowledge_graph(),
        profile_db_path=profile_path,
        memory_db_path=memory_path,
        clock=lambda: NOW,
    )

    profile_snapshot = await runtime.load_profile("user-1")
    history_snapshot = await runtime.load_history("user-1", "math")

    assert profile_snapshot is not None
    assert history_snapshot is not None
    plan = await runtime.plan_learning_path(
        LearnerPathEngineRequestV1(
            schema_version="learner_path_engine_request_v1",
            request_id="request-1",
            user_id="user-1",
            subject="math",
            requested_resource_types=("review_doc",),
            profile=profile_snapshot,
            history=history_snapshot,
        )
    )
    assert plan.steps[0].recommended_resource_types == ("review_doc",)
    result = await runtime.recommend_resources(
        ResourceRecommendationEngineRequestV1(
            schema_version="resource_recommendation_engine_request_v1",
            request_id="request-1",
            mode="automatic_after_generation",
            user_id="user-1",
            subject="math",
            profile=profile_snapshot,
            history=history_snapshot,
            generated_resources=(
                RecommendationResourceContextV1(
                    resource_id="generated-1",
                    resource_type="review_doc",
                    subject="math",
                    topic_id="math.algebra",
                    title="Generated review",
                ),
            ),
        )
    )
    assert result.status == "available"
    assert result.batch is not None
    item = result.batch.items[0]
    assert (
        item.resource_id,
        item.resource_type,
        item.subject,
        item.topic_id,
        item.title,
    ) == (
        "generated-1",
        "review_doc",
        "math",
        "math.algebra",
        "Generated review",
    )


def test_runtime_fingerprint_is_path_independent_and_content_sensitive(
    tmp_path: Path,
) -> None:
    first = build_learning_guidance_runtime(
        config=_config(),
        knowledge_graph=_knowledge_graph(),
        profile_db_path=tmp_path / "first-profile.sqlite",
        memory_db_path=tmp_path / "first-memory.sqlite",
        clock=lambda: NOW,
    )
    second = build_learning_guidance_runtime(
        config=_config(),
        knowledge_graph=_knowledge_graph(),
        profile_db_path=tmp_path / "other-profile.sqlite",
        memory_db_path=tmp_path / "other-memory.sqlite",
        clock=lambda: NOW,
    )
    changed_graph = build_learning_guidance_runtime(
        config=_config(),
        knowledge_graph=_knowledge_graph(data_version="test-v2"),
        profile_db_path=tmp_path / "first-profile.sqlite",
        memory_db_path=tmp_path / "first-memory.sqlite",
        clock=lambda: NOW,
    )
    changed_payload = _config().model_dump(mode="json")
    changed_payload["history_limit"] = 199
    changed_policy = build_learning_guidance_runtime(
        config=LearningGuidanceConfigV1.model_validate(changed_payload),
        knowledge_graph=_knowledge_graph(),
        profile_db_path=tmp_path / "first-profile.sqlite",
        memory_db_path=tmp_path / "first-memory.sqlite",
        clock=lambda: NOW,
    )

    assert first.runtime_fingerprint == second.runtime_fingerprint
    assert first.runtime_fingerprint != changed_graph.runtime_fingerprint
    assert first.runtime_fingerprint != changed_policy.runtime_fingerprint


def test_missing_knowledge_graph_blocks_before_store_creation(tmp_path: Path) -> None:
    config_path = tmp_path / "learning-guidance.yaml"
    config_path.write_text(
        yaml.safe_dump(
            _config().model_dump(mode="json"),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    profile_path = tmp_path / "profile.sqlite"
    memory_path = tmp_path / "memory.sqlite"

    with pytest.raises(KnowledgeGraphPathError):
        load_learning_guidance_runtime(
            config_path=config_path,
            project_root=tmp_path,
            profile_db_path=profile_path,
            memory_db_path=memory_path,
            clock=lambda: NOW,
        )

    assert not profile_path.exists()
    assert not memory_path.exists()


def test_knowledge_graph_path_cannot_escape_explicit_project_root(
    tmp_path: Path,
) -> None:
    payload = _config().model_dump(mode="json")
    payload["knowledge_graph_path"] = "../outside.yaml"
    config = LearningGuidanceConfigV1.model_validate(payload)

    with pytest.raises(ValueError, match="inside project_root"):
        resolve_knowledge_graph_path(config=config, project_root=tmp_path)
