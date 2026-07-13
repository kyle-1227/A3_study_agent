from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.graph.learning_guidance import (
    LEARNER_PATH_OUTPUT_STATE_KEY,
    RECOMMENDATION_RESOURCE_CONTEXT_STATE_KEY,
    RESOURCE_RECOMMENDATION_OUTPUT_STATE_KEY,
    make_learner_path_planner_node,
    make_resource_recommendation_node,
)
from src.graph.state import initial_request_reset_transient_state
from src.learning_guidance.contracts import (
    LearnerGoalSignalV1,
    LearnerHistoryEventV1,
    LearnerHistorySnapshotV1,
    LearnerPathPlanV1,
    LearnerPathStepV1,
    LearnerPreferenceSignalV1,
    LearnerProfileSnapshotV1,
    LearnerSkillSignalV1,
    RecommendationMode,
    RecommendationResourceContextV1,
    RecommendationScoreFactorsV1,
    RecommendationScoreWeightsV1,
    ResourceRecommendationBatchV1,
    ResourceRecommendationItemV1,
)
from src.learning_guidance.runtime import (
    LearningGuidanceContractError,
    LearningGuidanceRuntime,
)


NOW = datetime(2026, 7, 13, 8, 0, tzinfo=timezone.utc)


def _profile(user_id: str = "learner-1") -> LearnerProfileSnapshotV1:
    return LearnerProfileSnapshotV1(
        schema_version="learner_profile_snapshot_v1",
        user_id=user_id,
        skills=(
            LearnerSkillSignalV1(
                signal_id="skill-python",
                subject="python",
                topic_id="python-basics",
                level=0.4,
                confidence=0.9,
            ),
        ),
        goals=(
            LearnerGoalSignalV1(
                signal_id="goal-python",
                subject="python",
                goal="掌握 Python 基础",
                importance=0.8,
                progress=0.2,
            ),
        ),
        preferences=(
            LearnerPreferenceSignalV1(
                signal_id="preference-practice",
                dimension="prefer_practice",
                strength=0.9,
            ),
        ),
    )


def _history(user_id: str = "learner-1") -> LearnerHistorySnapshotV1:
    return LearnerHistorySnapshotV1(
        schema_version="learner_history_snapshot_v1",
        user_id=user_id,
        subject="python",
        events=(
            LearnerHistoryEventV1(
                history_id="history-python-1",
                subject="python",
                topic_id="python-basics",
                event_type="assessment",
                observed_at=NOW,
                outcome_score=0.45,
            ),
        ),
    )


def _path_plan(user_id: str = "learner-1") -> LearnerPathPlanV1:
    return LearnerPathPlanV1(
        schema_version="learner_path_plan_v1",
        user_id=user_id,
        subject="python",
        generated_at=NOW,
        steps=(
            LearnerPathStepV1(
                step_id="path-step-python-basics",
                position=1,
                topic_id="python-basics",
                subject="python",
                title="Python 基础巩固",
                status="reinforce",
                estimated_hours=2.0,
                reason="画像与最近测评均表明该主题需要巩固",
                recommended_resource_types=("quiz",),
                profile_signal_ids=("skill-python", "goal-python"),
                history_ids=("history-python-1",),
            ),
        ),
        summary="先巩固 Python 基础，再进入后续主题。",
    )


def _score_factors() -> RecommendationScoreFactorsV1:
    return RecommendationScoreFactorsV1(
        weakness=0.8,
        forgetting=0.6,
        preference=0.4,
        goal=0.2,
        combined=0.5,
        weights=RecommendationScoreWeightsV1(
            weakness=0.25,
            forgetting=0.25,
            preference=0.25,
            goal=0.25,
        ),
    )


def _recommendation_batch(
    mode: RecommendationMode,
    *,
    user_id: str = "learner-1",
) -> ResourceRecommendationBatchV1:
    source_resource_ids = (
        ("generated-quiz-1",) if mode == "automatic_after_generation" else ()
    )
    return ResourceRecommendationBatchV1(
        schema_version="resource_recommendation_batch_v1",
        mode=mode,
        user_id=user_id,
        subject="python",
        generated_at=NOW,
        items=(
            ResourceRecommendationItemV1(
                recommendation_id=f"recommendation-{mode}",
                resource_id="recommended-quiz-2",
                resource_type="quiz",
                subject="python",
                topic_id="python-basics",
                title="Python 基础强化练习",
                rank=1,
                score_factors=_score_factors(),
                reason="由明确画像信号与最近测评记录共同支持",
                profile_signal_ids=(
                    "skill-python",
                    "goal-python",
                    "preference-practice",
                ),
                history_ids=("history-python-1",),
                source_resource_ids=source_resource_ids,
            ),
        ),
        summary="推荐一组与当前薄弱主题直接相关的练习。",
    )


class RuntimeStub:
    def __init__(
        self,
        *,
        profile: LearnerProfileSnapshotV1 | None = None,
        history: LearnerHistorySnapshotV1 | None = None,
    ) -> None:
        self.profile = _profile() if profile is None else profile
        self.history = _history() if history is None else history
        self.profile_result: LearnerProfileSnapshotV1 | None = self.profile
        self.history_result: LearnerHistorySnapshotV1 | None = self.history
        self.profile_calls: list[str] = []
        self.history_calls: list[tuple[str, str]] = []
        self.path_requests = []
        self.recommendation_requests = []
        self.path_error: Exception | None = None
        self.recommendation_error: Exception | None = None

    async def load_profile(self, user_id: str):
        self.profile_calls.append(user_id)
        return self.profile_result

    async def load_history(self, user_id: str, subject: str):
        self.history_calls.append((user_id, subject))
        return self.history_result

    async def plan_learning_path(self, request):
        self.path_requests.append(request)
        if self.path_error is not None:
            raise self.path_error
        return _path_plan(user_id=request.user_id)

    async def recommend_resources(self, request):
        self.recommendation_requests.append(request)
        if self.recommendation_error is not None:
            raise self.recommendation_error
        return _recommendation_batch(request.mode, user_id=request.user_id)

    def runtime(self) -> LearningGuidanceRuntime:
        return LearningGuidanceRuntime(
            load_profile=self.load_profile,
            load_history=self.load_history,
            plan_learning_path=self.plan_learning_path,
            recommend_resources=self.recommend_resources,
        )


def _state(**updates):
    state = {
        "request_id": "request-1",
        "thread_id": "thread-must-not-be-used-as-user",
        "user_id": "learner-1",
        "subject": "python",
    }
    state.update(updates)
    return state


def test_contracts_forbid_schema_drift_and_default_score_mismatch() -> None:
    payload = {
        "signal_id": "skill-python",
        "subject": "python",
        "topic_id": "python-basics",
        "level": 0.4,
        "confidence": 0.9,
        "legacy_score": 0.5,
    }
    with pytest.raises(ValidationError, match="extra_forbidden"):
        LearnerSkillSignalV1.model_validate(payload)

    with pytest.raises(ValidationError, match="combined recommendation score"):
        RecommendationScoreFactorsV1(
            weakness=0.8,
            forgetting=0.6,
            preference=0.4,
            goal=0.2,
            combined=0.7,
            weights=RecommendationScoreWeightsV1(
                weakness=0.25,
                forgetting=0.25,
                preference=0.25,
                goal=0.25,
            ),
        )


def test_new_request_reset_has_explicit_empty_user_id() -> None:
    assert initial_request_reset_transient_state()["user_id"] == ""


def test_app_user_id_state_value_never_uses_thread_identity() -> None:
    from app import _explicit_user_id_state_value

    assert _explicit_user_id_state_value("real-user-7") == "real-user-7"
    assert _explicit_user_id_state_value(None) == ""


@pytest.mark.anyio
async def test_learner_path_requires_explicit_user_id_without_thread_fallback() -> None:
    stub = RuntimeStub()
    node = make_learner_path_planner_node(stub.runtime())

    update = await node(_state(user_id=""))

    output = update[LEARNER_PATH_OUTPUT_STATE_KEY]
    assert output["status"] == "unavailable"
    assert output["unavailable_reason"] == "missing_user_id"
    assert output["user_id"] is None
    assert stub.profile_calls == []
    assert stub.history_calls == []


@pytest.mark.anyio
async def test_learner_path_passes_real_user_id_and_strict_context() -> None:
    stub = RuntimeStub()
    node = make_learner_path_planner_node(stub.runtime())

    update = await node(_state())

    output = update[LEARNER_PATH_OUTPUT_STATE_KEY]
    assert output["status"] == "available"
    assert output["user_id"] == "learner-1"
    assert stub.profile_calls == ["learner-1"]
    assert stub.history_calls == [("learner-1", "python")]
    assert stub.path_requests[0].user_id == "learner-1"
    assert stub.path_requests[0].user_id != "thread-must-not-be-used-as-user"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("state_updates", "missing_dependency", "expected_reason"),
    [
        ({"subject": ""}, None, "missing_subject"),
        ({}, "profile", "profile_unavailable"),
        ({}, "history", "history_unavailable"),
    ],
)
async def test_learner_path_returns_explicit_unavailable_status(
    state_updates,
    missing_dependency,
    expected_reason,
) -> None:
    stub = RuntimeStub()
    if missing_dependency == "profile":
        stub.profile_result = None
    elif missing_dependency == "history":
        stub.history_result = None
    node = make_learner_path_planner_node(stub.runtime())

    update = await node(_state(**state_updates))

    output = update[LEARNER_PATH_OUTPUT_STATE_KEY]
    assert output["status"] == "unavailable"
    assert output["unavailable_reason"] == expected_reason
    assert stub.path_requests == []


@pytest.mark.anyio
async def test_learner_path_does_not_swallow_engine_errors() -> None:
    stub = RuntimeStub()
    stub.path_error = RuntimeError("planner failed")
    node = make_learner_path_planner_node(stub.runtime())

    with pytest.raises(RuntimeError, match="planner failed"):
        await node(_state())


@pytest.mark.anyio
@pytest.mark.parametrize(
    "mode",
    ["automatic_after_generation", "explicit_request"],
)
async def test_resource_recommendation_supports_both_explicit_modes(
    mode: RecommendationMode,
) -> None:
    stub = RuntimeStub()
    node = make_resource_recommendation_node(stub.runtime(), mode=mode)
    state = _state()
    if mode == "automatic_after_generation":
        state[RECOMMENDATION_RESOURCE_CONTEXT_STATE_KEY] = [
            RecommendationResourceContextV1(
                resource_id="generated-quiz-1",
                resource_type="quiz",
                subject="python",
            )
        ]

    update = await node(state)

    output = update[RESOURCE_RECOMMENDATION_OUTPUT_STATE_KEY]
    assert output["status"] == "available"
    assert output["mode"] == mode
    request = stub.recommendation_requests[0]
    assert request.user_id == "learner-1"
    assert request.mode == mode
    if mode == "automatic_after_generation":
        assert request.generated_resources[0].resource_id == "generated-quiz-1"
    else:
        assert request.generated_resources == ()


@pytest.mark.anyio
async def test_automatic_recommendation_requires_real_generated_resource() -> None:
    stub = RuntimeStub()
    node = make_resource_recommendation_node(
        stub.runtime(),
        mode="automatic_after_generation",
    )

    update = await node(_state())

    output = update[RESOURCE_RECOMMENDATION_OUTPUT_STATE_KEY]
    assert output["status"] == "unavailable"
    assert output["unavailable_reason"] == "generated_resources_unavailable"
    assert stub.profile_calls == []
    assert stub.recommendation_requests == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("state_updates", "missing_dependency", "expected_reason"),
    [
        ({"subject": ""}, None, "missing_subject"),
        ({}, "profile", "profile_unavailable"),
        ({}, "history", "history_unavailable"),
    ],
)
async def test_resource_recommendation_returns_explicit_unavailable_status(
    state_updates,
    missing_dependency,
    expected_reason,
) -> None:
    stub = RuntimeStub()
    if missing_dependency == "profile":
        stub.profile_result = None
    elif missing_dependency == "history":
        stub.history_result = None
    node = make_resource_recommendation_node(
        stub.runtime(),
        mode="explicit_request",
    )

    update = await node(_state(**state_updates))

    output = update[RESOURCE_RECOMMENDATION_OUTPUT_STATE_KEY]
    assert output["status"] == "unavailable"
    assert output["unavailable_reason"] == expected_reason
    assert stub.recommendation_requests == []


@pytest.mark.anyio
async def test_recommendation_does_not_swallow_engine_errors() -> None:
    stub = RuntimeStub()
    stub.recommendation_error = RuntimeError("recommendation failed")
    node = make_resource_recommendation_node(
        stub.runtime(),
        mode="explicit_request",
    )

    with pytest.raises(RuntimeError, match="recommendation failed"):
        await node(_state())


@pytest.mark.anyio
async def test_recommendation_rejects_unbound_engine_evidence() -> None:
    stub = RuntimeStub()

    async def unbound_recommendation(request):
        batch = _recommendation_batch(request.mode)
        item = batch.items[0].model_copy(update={"history_ids": ("unknown-history",)})
        return batch.model_copy(update={"items": (item,)})

    runtime = LearningGuidanceRuntime(
        load_profile=stub.load_profile,
        load_history=stub.load_history,
        plan_learning_path=stub.plan_learning_path,
        recommend_resources=unbound_recommendation,
    )
    node = make_resource_recommendation_node(runtime, mode="explicit_request")

    with pytest.raises(
        LearningGuidanceContractError,
        match="unknown_recommendation_history_evidence",
    ):
        await node(_state())
