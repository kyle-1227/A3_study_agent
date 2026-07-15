from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json

import pytest
from pydantic import ValidationError

from src.graph.learning_guidance import (
    LEARNER_PATH_OUTPUT_STATE_KEY,
    LEARNER_PATH_PROVIDER_PROJECTION_STATE_KEY,
    RECOMMENDATION_RESOURCE_CONTEXT_STATE_KEY,
    RESOURCE_RECOMMENDATION_OUTPUT_STATE_KEY,
    learner_path_provider_projection_for_runtime_from_state,
    learner_path_provider_projection_from_state,
    make_learner_path_planner_node,
    make_resource_recommendation_node,
    resource_recommendation_output_for_runtime_from_state,
    resource_recommendation_output_from_state,
)
from src.graph.state import initial_request_reset_transient_state
from src.learning_guidance.contracts import (
    LearnerGoalSignalV1,
    LearnerHistoryEventV1,
    LearnerHistorySnapshotV1,
    LearnerPathEngineRequestV1,
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
    ResourceRecommendationEngineRequestV1,
    ResourceRecommendationEngineResultV1,
    ResourceRecommendationItemV1,
)
from src.learning_guidance.knowledge_graph import KnowledgeGraphV1
from src.learning_guidance.runtime import (
    LearningGuidanceContractError,
    LearningGuidanceRuntime,
)


NOW = datetime(2026, 7, 13, 8, 0, tzinfo=timezone.utc)


def _knowledge_graph() -> KnowledgeGraphV1:
    return KnowledgeGraphV1.model_validate(
        {
            "schema_version": "knowledge_graph_v1",
            "data_version": "test-v1",
            "subjects": [
                {
                    "subject_id": "python",
                    "title": "Python",
                    "topics": [
                        {
                            "topic_id": "python-basics",
                            "title": "Python basics",
                            "difficulty": 0.4,
                            "estimated_hours": 2.0,
                            "prerequisite_topic_ids": [],
                            "knowledge_points": ["Python syntax"],
                            "resources": [
                                {
                                    "resource_id": "python-basics-quiz",
                                    "resource_type": "quiz",
                                    "title": "Python basics quiz",
                                }
                            ],
                        },
                        {
                            "topic_id": "python-practice",
                            "title": "Python practice",
                            "difficulty": 0.5,
                            "estimated_hours": 1.0,
                            "prerequisite_topic_ids": ["python-basics"],
                            "knowledge_points": ["Python practice"],
                            "resources": [
                                {
                                    "resource_id": "python-practice-quiz",
                                    "resource_type": "quiz",
                                    "title": "Python practice quiz",
                                }
                            ],
                        },
                    ],
                }
            ],
        }
    )


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
            LearnerSkillSignalV1(
                signal_id="skill-python-practice",
                subject="python",
                topic_id="python-practice",
                level=0.4,
                confidence=0.7,
            ),
        ),
        goals=(
            LearnerGoalSignalV1(
                signal_id="goal-python",
                subject="python",
                topic_id="python-basics",
                goal="掌握 Python 基础",
                importance=0.8,
                progress=0.2,
            ),
            LearnerGoalSignalV1(
                signal_id="goal-python-practice",
                subject="python",
                topic_id="python-practice",
                goal="Practice Python basics",
                importance=0.6,
                progress=0.1,
            ),
        ),
        preferences=(
            LearnerPreferenceSignalV1(
                signal_id="preference-practice",
                subject="python",
                topic_id="python-basics",
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
            LearnerHistoryEventV1(
                history_id="history-python-practice-1",
                subject="python",
                topic_id="python-practice",
                event_type="practice",
                observed_at=NOW,
                outcome_score=0.55,
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
                title="Python basics",
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
    automatic = mode == "automatic_after_generation"
    resource_id = "generated-quiz-1" if automatic else "python-basics-quiz"
    title = "Generated Python quiz" if automatic else "Python basics quiz"
    return ResourceRecommendationBatchV1(
        schema_version="resource_recommendation_batch_v1",
        mode=mode,
        user_id=user_id,
        subject="python",
        generated_at=NOW,
        items=(
            ResourceRecommendationItemV1(
                recommendation_id=f"recommendation-{mode}",
                resource_id=resource_id,
                resource_type="quiz",
                subject="python",
                topic_id="python-basics",
                title=title,
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


def _engine_result(
    request: ResourceRecommendationEngineRequestV1,
    batch: ResourceRecommendationBatchV1,
) -> ResourceRecommendationEngineResultV1:
    return ResourceRecommendationEngineResultV1(
        schema_version="resource_recommendation_engine_result_v1",
        request_id=request.request_id,
        mode=request.mode,
        user_id=request.user_id,
        subject=request.subject,
        status="available",
        unavailable_reason=None,
        batch=batch,
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
        self.path_requests: list[LearnerPathEngineRequestV1] = []
        self.recommendation_requests: list[ResourceRecommendationEngineRequestV1] = []
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
        batch = _recommendation_batch(request.mode, user_id=request.user_id)
        return _engine_result(request, batch)

    def runtime(
        self,
        *,
        provider_projection_max_steps: int = 50,
        provider_projection_max_chars: int = 65_536,
    ) -> LearningGuidanceRuntime:
        return LearningGuidanceRuntime(
            runtime_fingerprint="1" * 64,
            knowledge_graph=_knowledge_graph(),
            provider_projection_max_steps=provider_projection_max_steps,
            provider_projection_max_chars=provider_projection_max_chars,
            load_profile=self.load_profile,
            load_history=self.load_history,
            plan_learning_path=self.plan_learning_path,
            recommend_resources=self.recommend_resources,
        )


def _retrieval_plan(*subjects: str) -> list[dict[str, object]]:
    return [
        {
            "subject": subject,
            "role": "core_concept",
            "local_retrieval_query": f"{subject} course concepts",
            "web_research_seed_query": f"{subject} official tutorial",
            "purpose": "support the requested learning resource",
            "priority": 1.0,
            "_parent_child_priority_explicit": True,
        }
        for subject in subjects
    ]


def _state(**updates):
    state = {
        "request_id": "request-1",
        "thread_id": "thread-must-not-be-used-as-user",
        "user_id": "learner-1",
        "subject": "python",
        "evidence_requested_subjects": ["python"],
        "retrieval_plan": _retrieval_plan("python"),
        "requested_resource_type": "quiz",
        "requested_resource_types": ["quiz"],
        RECOMMENDATION_RESOURCE_CONTEXT_STATE_KEY: [],
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

    batch_payload = _recommendation_batch("automatic_after_generation").model_dump(
        mode="python"
    )
    batch_payload["items"][0]["title"] = "x" * 241
    with pytest.raises(ValidationError, match="at most 240 characters"):
        ResourceRecommendationBatchV1.model_validate(batch_payload)


def test_new_request_reset_has_explicit_empty_user_id() -> None:
    reset = initial_request_reset_transient_state()
    assert reset["user_id"] == ""
    assert reset[LEARNER_PATH_PROVIDER_PROJECTION_STATE_KEY] == {}


def test_learning_guidance_runtime_requires_explicit_stable_fingerprint() -> None:
    stub = RuntimeStub()

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        LearningGuidanceRuntime(
            runtime_fingerprint="not-a-digest",
            knowledge_graph=_knowledge_graph(),
            provider_projection_max_steps=50,
            provider_projection_max_chars=65_536,
            load_profile=stub.load_profile,
            load_history=stub.load_history,
            plan_learning_path=stub.plan_learning_path,
            recommend_resources=stub.recommend_resources,
        )


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
    runtime = stub.runtime()
    node = make_learner_path_planner_node(runtime)

    update = await node(_state())

    output = update[LEARNER_PATH_OUTPUT_STATE_KEY]
    assert output["status"] == "available"
    assert output["user_id"] == "learner-1"
    assert stub.profile_calls == ["learner-1"]
    assert stub.history_calls == [("learner-1", "python")]
    assert stub.path_requests[0].user_id == "learner-1"
    assert stub.path_requests[0].user_id != "thread-must-not-be-used-as-user"

    projection = update[LEARNER_PATH_PROVIDER_PROJECTION_STATE_KEY]
    encoded = json.dumps(
        projection,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert len(encoded) <= runtime.provider_projection_max_chars
    assert projection["status"] == "available"
    assert projection["steps"][0]["step_id"] == "path-step-python-basics"
    for forbidden_field in (
        "request_id",
        "user_id",
        "profile_signal_ids",
        "history_ids",
        "runtime_fingerprint",
        "provider_projection_policy_fingerprint",
    ):
        assert f'"{forbidden_field}"' not in encoded
    assert "learner-1" not in encoded
    assert "history-python-1" not in encoded


@pytest.mark.anyio
@pytest.mark.parametrize(
    "retrieval_subjects",
    [
        ("python", "math"),
        ("math",),
    ],
)
async def test_learner_path_rejects_unsupported_retrieval_subject_scope_before_io(
    retrieval_subjects: tuple[str, ...],
) -> None:
    stub = RuntimeStub()
    node = make_learner_path_planner_node(stub.runtime())

    update = await node(_state(retrieval_plan=_retrieval_plan(*retrieval_subjects)))

    output = update[LEARNER_PATH_OUTPUT_STATE_KEY]
    projection = update[LEARNER_PATH_PROVIDER_PROJECTION_STATE_KEY]
    assert output["status"] == "unavailable"
    assert output["unavailable_reason"] == "unsupported_subject_scope"
    assert projection["status"] == "unavailable"
    assert projection["unavailable_reason"] == "unsupported_subject_scope"
    assert stub.profile_calls == []
    assert stub.history_calls == []
    assert stub.path_requests == []


@pytest.mark.anyio
async def test_learner_path_provider_projection_length_limit_fails_without_truncation() -> (
    None
):
    stub = RuntimeStub()
    node = make_learner_path_planner_node(stub.runtime(provider_projection_max_chars=1))

    with pytest.raises(LearningGuidanceContractError) as error:
        await node(_state())

    assert error.value.code == "learner_path_provider_projection_too_large"
    assert len(stub.path_requests) == 1


@pytest.mark.anyio
async def test_learner_path_provider_projection_step_limit_fails_without_truncation() -> (
    None
):
    stub = RuntimeStub()
    first_step = _path_plan().steps[0]
    two_step_plan = LearnerPathPlanV1(
        schema_version="learner_path_plan_v1",
        user_id="learner-1",
        subject="python",
        generated_at=NOW,
        steps=(
            first_step,
            LearnerPathStepV1(
                step_id="path-step-python-practice",
                position=2,
                topic_id="python-practice",
                subject="python",
                title="Python practice",
                status="ready",
                estimated_hours=1.0,
                reason="在基础巩固后完成一轮练习。",
                recommended_resource_types=(),
                profile_signal_ids=(
                    "skill-python-practice",
                    "goal-python-practice",
                ),
                history_ids=("history-python-practice-1",),
            ),
        ),
        summary="先巩固基础，再完成练习。",
    )

    async def plan_two_steps(request):
        stub.path_requests.append(request)
        return two_step_plan

    runtime = replace(
        stub.runtime(provider_projection_max_steps=1),
        plan_learning_path=plan_two_steps,
    )
    node = make_learner_path_planner_node(runtime)

    with pytest.raises(LearningGuidanceContractError) as error:
        await node(_state())

    assert error.value.code == "learner_path_provider_projection_too_large"
    assert len(stub.path_requests) == 1


@pytest.mark.anyio
async def test_learner_path_rejects_engine_topic_metadata_tamper() -> None:
    stub = RuntimeStub()
    payload = _path_plan().model_dump(mode="python")
    payload["steps"][0]["title"] = "Tampered topic title"
    tampered_plan = LearnerPathPlanV1.model_validate(payload)

    async def plan_with_tampered_metadata(request):
        stub.path_requests.append(request)
        return tampered_plan

    runtime = replace(
        stub.runtime(),
        plan_learning_path=plan_with_tampered_metadata,
    )
    node = make_learner_path_planner_node(runtime)

    with pytest.raises(
        LearningGuidanceContractError,
        match="path_topic_metadata_mismatch",
    ):
        await node(_state())


@pytest.mark.anyio
async def test_learner_path_provider_projection_rejects_tamper_and_stale_state() -> (
    None
):
    stub = RuntimeStub()
    node = make_learner_path_planner_node(stub.runtime())
    state = _state()
    update = await node(state)

    parsed = learner_path_provider_projection_from_state(
        {**state, **update},
    )
    assert parsed.status == "available"

    tampered_projection = {
        **update[LEARNER_PATH_PROVIDER_PROJECTION_STATE_KEY],
        "summary": "tampered summary",
    }
    with pytest.raises(LearningGuidanceContractError) as tamper_error:
        learner_path_provider_projection_from_state(
            {
                **state,
                **update,
                LEARNER_PATH_PROVIDER_PROJECTION_STATE_KEY: tampered_projection,
            },
        )
    assert tamper_error.value.code == "learner_path_provider_projection_mismatch"

    with pytest.raises(LearningGuidanceContractError) as stale_error:
        learner_path_provider_projection_from_state(
            {**state, **update, "request_id": "stale-request"},
        )
    assert stale_error.value.code == "learner_path_request_mismatch"

    with pytest.raises(LearningGuidanceContractError) as scope_error:
        learner_path_provider_projection_from_state(
            {
                **state,
                **update,
                "retrieval_plan": _retrieval_plan("python", "math"),
            },
        )
    assert scope_error.value.code == "learner_path_scope_mismatch"

    tampered_policy_output = {
        **update[LEARNER_PATH_OUTPUT_STATE_KEY],
        "provider_projection_max_steps": 49,
    }
    with pytest.raises(LearningGuidanceContractError) as policy_binding_error:
        learner_path_provider_projection_from_state(
            {
                **state,
                **update,
                LEARNER_PATH_OUTPUT_STATE_KEY: tampered_policy_output,
            }
        )
    assert policy_binding_error.value.code == "invalid_learner_path_planner_output"


@pytest.mark.anyio
async def test_learner_path_provider_projection_forbids_identity_schema_drift() -> None:
    stub = RuntimeStub()
    node = make_learner_path_planner_node(stub.runtime())
    state = _state()
    update = await node(state)
    leaked_projection = {
        **update[LEARNER_PATH_PROVIDER_PROJECTION_STATE_KEY],
        "request_id": state["request_id"],
    }

    with pytest.raises(LearningGuidanceContractError) as error:
        learner_path_provider_projection_from_state(
            {
                **state,
                **update,
                LEARNER_PATH_PROVIDER_PROJECTION_STATE_KEY: leaked_projection,
            },
        )

    assert error.value.code == "invalid_learner_path_provider_projection"


@pytest.mark.anyio
async def test_learner_path_checkpoint_rejects_changed_guidance_runtime() -> None:
    stub = RuntimeStub()
    runtime = stub.runtime()
    node = make_learner_path_planner_node(runtime)
    state = _state()
    update = await node(state)

    with pytest.raises(LearningGuidanceContractError) as runtime_error:
        learner_path_provider_projection_for_runtime_from_state(
            {**state, **update},
            runtime=replace(runtime, runtime_fingerprint="2" * 64),
        )
    assert runtime_error.value.code == "learning_guidance_runtime_mismatch"

    with pytest.raises(LearningGuidanceContractError) as policy_error:
        learner_path_provider_projection_for_runtime_from_state(
            {**state, **update},
            runtime=replace(runtime, provider_projection_max_steps=49),
        )
    assert policy_error.value.code == ("learning_guidance_projection_policy_mismatch")


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
                topic_id="python-basics",
                title="Generated Python quiz",
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
        generated = request.generated_resources[0]
        assert (
            generated.resource_id,
            generated.resource_type,
            generated.subject,
            generated.topic_id,
            generated.title,
        ) == (
            "generated-quiz-1",
            "quiz",
            "python",
            "python-basics",
            "Generated Python quiz",
        )
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
async def test_automatic_recommendation_requires_explicit_empty_context_key() -> None:
    stub = RuntimeStub()
    node = make_resource_recommendation_node(
        stub.runtime(),
        mode="automatic_after_generation",
    )
    state = _state()
    del state[RECOMMENDATION_RESOURCE_CONTEXT_STATE_KEY]

    with pytest.raises(
        LearningGuidanceContractError,
        match="missing_recommendation_resource_context",
    ):
        await node(state)


@pytest.mark.anyio
async def test_learner_path_rejects_whitespace_only_identity() -> None:
    stub = RuntimeStub()
    node = make_learner_path_planner_node(stub.runtime())

    with pytest.raises(LearningGuidanceContractError, match="invalid_user_id"):
        await node(_state(user_id="   "))


@pytest.mark.anyio
async def test_automatic_recommendation_rejects_multi_subject_scope_explicitly() -> (
    None
):
    stub = RuntimeStub()
    node = make_resource_recommendation_node(
        stub.runtime(),
        mode="automatic_after_generation",
    )
    state = _state(evidence_requested_subjects=["python", "math"])
    state[RECOMMENDATION_RESOURCE_CONTEXT_STATE_KEY] = [
        RecommendationResourceContextV1(
            resource_id="generated-quiz-1",
            resource_type="quiz",
            subject="python",
            topic_id="python-basics",
            title="Generated Python quiz",
        )
    ]

    update = await node(state)

    output = update[RESOURCE_RECOMMENDATION_OUTPUT_STATE_KEY]
    assert output["status"] == "unavailable"
    assert output["unavailable_reason"] == "unsupported_subject_scope"
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
        payload = _recommendation_batch(request.mode).model_dump(mode="python")
        payload["items"][0]["history_ids"] = ("unknown-history",)
        return _engine_result(
            request,
            ResourceRecommendationBatchV1.model_validate(payload),
        )

    runtime = LearningGuidanceRuntime(
        runtime_fingerprint="2" * 64,
        knowledge_graph=_knowledge_graph(),
        provider_projection_max_steps=50,
        provider_projection_max_chars=65_536,
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


@pytest.mark.anyio
async def test_explicit_recommendation_rejects_non_catalog_engine_target() -> None:
    stub = RuntimeStub()

    async def invented_recommendation(request):
        payload = _recommendation_batch(request.mode).model_dump(mode="python")
        payload["items"][0]["resource_id"] = "invented-quiz"
        payload["items"][0]["title"] = "Invented quiz"
        return _engine_result(
            request,
            ResourceRecommendationBatchV1.model_validate(payload),
        )

    runtime = replace(
        stub.runtime(),
        recommend_resources=invented_recommendation,
    )
    node = make_resource_recommendation_node(runtime, mode="explicit_request")

    with pytest.raises(
        LearningGuidanceContractError,
        match="recommendation_catalog_binding_mismatch",
    ):
        await node(_state())


@pytest.mark.anyio
async def test_automatic_recommendation_rejects_unbound_target_resource() -> None:
    stub = RuntimeStub()

    async def unbound_target_recommendation(request):
        payload = _recommendation_batch(request.mode).model_dump(mode="python")
        payload["items"][0]["resource_id"] = "catalog-resource-not-generated"
        return _engine_result(
            request,
            ResourceRecommendationBatchV1.model_validate(payload),
        )

    runtime = replace(
        stub.runtime(),
        recommend_resources=unbound_target_recommendation,
    )
    node = make_resource_recommendation_node(
        runtime,
        mode="automatic_after_generation",
    )
    state = _state()
    state[RECOMMENDATION_RESOURCE_CONTEXT_STATE_KEY] = [
        RecommendationResourceContextV1(
            resource_id="generated-quiz-1",
            resource_type="quiz",
            subject="python",
            topic_id="python-basics",
            title="Generated Python quiz",
        )
    ]

    with pytest.raises(
        LearningGuidanceContractError,
        match="unknown_recommendation_target_resource",
    ):
        await node(state)


@pytest.mark.anyio
async def test_automatic_recommendation_rejects_tampered_target_title() -> None:
    stub = RuntimeStub()

    async def tampered_title_recommendation(request):
        payload = _recommendation_batch(request.mode).model_dump(mode="python")
        payload["items"][0]["title"] = "Tampered title"
        return _engine_result(
            request,
            ResourceRecommendationBatchV1.model_validate(payload),
        )

    runtime = replace(
        stub.runtime(),
        recommend_resources=tampered_title_recommendation,
    )
    node = make_resource_recommendation_node(
        runtime,
        mode="automatic_after_generation",
    )
    state = _state()
    state[RECOMMENDATION_RESOURCE_CONTEXT_STATE_KEY] = [
        RecommendationResourceContextV1(
            resource_id="generated-quiz-1",
            resource_type="quiz",
            subject="python",
            topic_id="python-basics",
            title="Generated Python quiz",
        )
    ]

    with pytest.raises(
        LearningGuidanceContractError,
        match="recommendation_target_binding_mismatch",
    ):
        await node(state)


@pytest.mark.anyio
async def test_recommendation_checkpoint_projection_rejects_stale_request() -> None:
    stub = RuntimeStub()
    node = make_resource_recommendation_node(
        stub.runtime(),
        mode="explicit_request",
    )
    state = _state()
    update = await node(state)

    with pytest.raises(
        LearningGuidanceContractError,
        match="recommendation_request_mismatch",
    ):
        resource_recommendation_output_from_state(
            {
                **state,
                **update,
                "request_id": "stale-request",
            },
            expected_mode="explicit_request",
        )


@pytest.mark.anyio
async def test_recommendation_checkpoint_projection_rejects_json_coercion() -> None:
    stub = RuntimeStub()
    node = make_resource_recommendation_node(
        stub.runtime(),
        mode="explicit_request",
    )
    state = _state()
    update = await node(state)
    drifted = update[RESOURCE_RECOMMENDATION_OUTPUT_STATE_KEY]
    drifted["batch"]["items"][0]["rank"] = "1"

    with pytest.raises(
        LearningGuidanceContractError,
        match="invalid_resource_recommendation_output",
    ):
        resource_recommendation_output_from_state(
            {**state, RESOURCE_RECOMMENDATION_OUTPUT_STATE_KEY: drifted},
            expected_mode="explicit_request",
        )


@pytest.mark.anyio
async def test_automatic_recommendation_replay_rejects_tampered_title() -> None:
    stub = RuntimeStub()
    runtime = stub.runtime()
    node = make_resource_recommendation_node(
        runtime,
        mode="automatic_after_generation",
    )
    state = _state()
    state[RECOMMENDATION_RESOURCE_CONTEXT_STATE_KEY] = [
        RecommendationResourceContextV1(
            resource_id="generated-quiz-1",
            resource_type="quiz",
            subject="python",
            topic_id="python-basics",
            title="Generated Python quiz",
        )
    ]
    update = await node(state)
    update[RESOURCE_RECOMMENDATION_OUTPUT_STATE_KEY]["batch"]["items"][0]["title"] = (
        "Tampered replay title"
    )

    with pytest.raises(
        LearningGuidanceContractError,
        match="recommendation_target_binding_mismatch",
    ):
        resource_recommendation_output_from_state(
            {**state, **update},
            expected_mode="automatic_after_generation",
        )


@pytest.mark.anyio
async def test_explicit_recommendation_replay_rejects_catalog_title_tamper() -> None:
    stub = RuntimeStub()
    runtime = stub.runtime()
    node = make_resource_recommendation_node(runtime, mode="explicit_request")
    state = _state()
    update = await node(state)
    update[RESOURCE_RECOMMENDATION_OUTPUT_STATE_KEY]["batch"]["items"][0]["title"] = (
        "Tampered catalog title"
    )

    with pytest.raises(
        LearningGuidanceContractError,
        match="recommendation_catalog_binding_mismatch",
    ):
        resource_recommendation_output_for_runtime_from_state(
            {**state, **update},
            expected_mode="explicit_request",
            runtime=runtime,
        )
