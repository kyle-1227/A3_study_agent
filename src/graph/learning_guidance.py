"""Strict graph nodes for learner paths and resource recommendations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping

from src.graph.state import LearningState
from src.learning_guidance.contracts import (
    LearnerHistorySnapshotV1,
    LearnerPathEngineRequestV1,
    LearnerPathPlanV1,
    LearnerPathPlannerOutputV1,
    LearnerPathUnavailableReason,
    LearnerProfileSnapshotV1,
    RecommendationMode,
    RecommendationResourceContextV1,
    RecommendationUnavailableReason,
    ResourceRecommendationBatchV1,
    ResourceRecommendationEngineRequestV1,
    ResourceRecommendationOutputV1,
)
from src.learning_guidance.runtime import (
    LearningGuidanceContractError,
    LearningGuidanceRuntime,
)


LEARNER_PATH_OUTPUT_STATE_KEY = "learner_path_planner_output"
RESOURCE_RECOMMENDATION_OUTPUT_STATE_KEY = "resource_recommendation_output"
RECOMMENDATION_RESOURCE_CONTEXT_STATE_KEY = "recommendation_resource_context"


def _required_state_text(state: Mapping[str, object], field_name: str) -> str:
    value = state.get(field_name)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise LearningGuidanceContractError(
            code=f"invalid_{field_name}",
            reason=f"{field_name} must be an explicit normalized non-blank string",
        )
    return value


def _optional_state_text(
    state: Mapping[str, object],
    field_name: str,
) -> str | None:
    value = state.get(field_name)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise LearningGuidanceContractError(
            code=f"invalid_{field_name}",
            reason=f"{field_name} must be a string when present",
        )
    if not value.strip():
        return None
    if value != value.strip():
        raise LearningGuidanceContractError(
            code=f"invalid_{field_name}",
            reason=f"{field_name} must already be normalized",
        )
    return value


def _path_unavailable(
    *,
    request_id: str,
    reason: LearnerPathUnavailableReason,
    user_id: str | None,
    subject: str | None,
) -> dict[str, object]:
    output = LearnerPathPlannerOutputV1(
        schema_version="learner_path_planner_output_v1",
        request_id=request_id,
        status="unavailable",
        unavailable_reason=reason,
        user_id=user_id,
        subject=subject,
        plan=None,
    )
    return {LEARNER_PATH_OUTPUT_STATE_KEY: output.model_dump(mode="json")}


def _recommendation_unavailable(
    *,
    request_id: str,
    mode: RecommendationMode,
    reason: RecommendationUnavailableReason,
    user_id: str | None,
    subject: str | None,
) -> dict[str, object]:
    output = ResourceRecommendationOutputV1(
        schema_version="resource_recommendation_output_v1",
        request_id=request_id,
        mode=mode,
        status="unavailable",
        unavailable_reason=reason,
        user_id=user_id,
        subject=subject,
        batch=None,
    )
    return {RESOURCE_RECOMMENDATION_OUTPUT_STATE_KEY: output.model_dump(mode="json")}


def _validate_profile_binding(
    profile: LearnerProfileSnapshotV1,
    *,
    user_id: str,
) -> None:
    if profile.user_id != user_id:
        raise LearningGuidanceContractError(
            code="profile_identity_mismatch",
            reason="profile user_id differs from the explicit request user_id",
        )


def _validate_history_binding(
    history: LearnerHistorySnapshotV1,
    *,
    user_id: str,
    subject: str,
) -> None:
    if history.user_id != user_id:
        raise LearningGuidanceContractError(
            code="history_identity_mismatch",
            reason="history user_id differs from the explicit request user_id",
        )
    if history.subject != subject:
        raise LearningGuidanceContractError(
            code="history_subject_mismatch",
            reason="history subject differs from the explicit request subject",
        )


def _subject_profile_signal_ids(
    profile: LearnerProfileSnapshotV1,
    subject: str,
) -> frozenset[str]:
    return frozenset(
        [signal.signal_id for signal in profile.skills if signal.subject == subject]
        + [signal.signal_id for signal in profile.goals if signal.subject == subject]
        + [signal.signal_id for signal in profile.preferences]
    )


def _validate_plan_binding(
    plan: LearnerPathPlanV1,
    *,
    request: LearnerPathEngineRequestV1,
) -> None:
    if plan.user_id != request.user_id or plan.subject != request.subject:
        raise LearningGuidanceContractError(
            code="learner_path_identity_mismatch",
            reason="learner path identity differs from its engine request",
        )
    allowed_profile_ids = _subject_profile_signal_ids(
        request.profile,
        request.subject,
    )
    allowed_history_ids = request.history.history_ids()
    for step in plan.steps:
        if not set(step.profile_signal_ids).issubset(allowed_profile_ids):
            raise LearningGuidanceContractError(
                code="unknown_path_profile_evidence",
                reason="learner path references profile signals outside its request",
            )
        if not set(step.history_ids).issubset(allowed_history_ids):
            raise LearningGuidanceContractError(
                code="unknown_path_history_evidence",
                reason="learner path references history outside its request",
            )


def _recommendation_resources(
    state: Mapping[str, object],
) -> tuple[RecommendationResourceContextV1, ...]:
    raw = state.get(RECOMMENDATION_RESOURCE_CONTEXT_STATE_KEY)
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise LearningGuidanceContractError(
            code="invalid_recommendation_resource_context",
            reason="recommendation resource context must be a sequence",
        )
    resources: list[RecommendationResourceContextV1] = []
    for item in raw:
        try:
            resources.append(RecommendationResourceContextV1.model_validate(item))
        except (TypeError, ValueError) as exc:
            raise LearningGuidanceContractError(
                code="invalid_recommendation_resource_context",
                reason="recommendation resource context violates its strict schema",
            ) from exc
    return tuple(resources)


def _validate_recommendation_binding(
    batch: ResourceRecommendationBatchV1,
    *,
    request: ResourceRecommendationEngineRequestV1,
) -> None:
    if (
        batch.user_id != request.user_id
        or batch.subject != request.subject
        or batch.mode != request.mode
    ):
        raise LearningGuidanceContractError(
            code="recommendation_identity_mismatch",
            reason="recommendation batch binding differs from its engine request",
        )
    allowed_profile_ids = _subject_profile_signal_ids(
        request.profile,
        request.subject,
    )
    allowed_history_ids = request.history.history_ids()
    allowed_resource_ids = frozenset(
        resource.resource_id for resource in request.generated_resources
    )
    for item in batch.items:
        if not set(item.profile_signal_ids).issubset(allowed_profile_ids):
            raise LearningGuidanceContractError(
                code="unknown_recommendation_profile_evidence",
                reason="recommendation references profile signals outside its request",
            )
        if not set(item.history_ids).issubset(allowed_history_ids):
            raise LearningGuidanceContractError(
                code="unknown_recommendation_history_evidence",
                reason="recommendation references history outside its request",
            )
        if not set(item.source_resource_ids).issubset(allowed_resource_ids):
            raise LearningGuidanceContractError(
                code="unknown_recommendation_resource_evidence",
                reason="recommendation references generated resources outside its request",
            )


async def _load_available_context(
    runtime: LearningGuidanceRuntime,
    *,
    user_id: str,
    subject: str,
) -> tuple[LearnerProfileSnapshotV1 | None, LearnerHistorySnapshotV1 | None]:
    profile = await runtime.load_profile(user_id)
    if profile is None:
        return None, None
    if not isinstance(profile, LearnerProfileSnapshotV1):
        raise LearningGuidanceContractError(
            code="invalid_profile_snapshot_type",
            reason="profile loader must return LearnerProfileSnapshotV1",
        )
    _validate_profile_binding(profile, user_id=user_id)
    if not profile.supports_subject(subject):
        return None, None

    history = await runtime.load_history(user_id, subject)
    if history is None:
        return profile, None
    if not isinstance(history, LearnerHistorySnapshotV1):
        raise LearningGuidanceContractError(
            code="invalid_history_snapshot_type",
            reason="history loader must return LearnerHistorySnapshotV1",
        )
    _validate_history_binding(history, user_id=user_id, subject=subject)
    return profile, history


def make_learner_path_planner_node(
    runtime: LearningGuidanceRuntime,
) -> Callable[[LearningState], Awaitable[dict[str, object]]]:
    """Create a strict learner path node with no identity or score fallback."""

    if not isinstance(runtime, LearningGuidanceRuntime):
        raise TypeError("runtime must be LearningGuidanceRuntime")

    async def learner_path_planner(state: LearningState) -> dict[str, object]:
        request_id = _required_state_text(state, "request_id")
        user_id = _optional_state_text(state, "user_id")
        subject = _optional_state_text(state, "subject")
        if user_id is None:
            return _path_unavailable(
                request_id=request_id,
                reason="missing_user_id",
                user_id=None,
                subject=subject,
            )
        if subject is None:
            return _path_unavailable(
                request_id=request_id,
                reason="missing_subject",
                user_id=user_id,
                subject=None,
            )

        profile, history = await _load_available_context(
            runtime,
            user_id=user_id,
            subject=subject,
        )
        if profile is None:
            return _path_unavailable(
                request_id=request_id,
                reason="profile_unavailable",
                user_id=user_id,
                subject=subject,
            )
        if history is None:
            return _path_unavailable(
                request_id=request_id,
                reason="history_unavailable",
                user_id=user_id,
                subject=subject,
            )

        request = LearnerPathEngineRequestV1(
            schema_version="learner_path_engine_request_v1",
            request_id=request_id,
            user_id=user_id,
            subject=subject,
            profile=profile,
            history=history,
        )
        plan = await runtime.plan_learning_path(request)
        if not isinstance(plan, LearnerPathPlanV1):
            raise LearningGuidanceContractError(
                code="invalid_learner_path_type",
                reason="learner path engine must return LearnerPathPlanV1",
            )
        _validate_plan_binding(plan, request=request)
        output = LearnerPathPlannerOutputV1(
            schema_version="learner_path_planner_output_v1",
            request_id=request_id,
            status="available",
            unavailable_reason=None,
            user_id=user_id,
            subject=subject,
            plan=plan,
        )
        return {LEARNER_PATH_OUTPUT_STATE_KEY: output.model_dump(mode="json")}

    return learner_path_planner


def make_resource_recommendation_node(
    runtime: LearningGuidanceRuntime,
    *,
    mode: RecommendationMode,
) -> Callable[[LearningState], Awaitable[dict[str, object]]]:
    """Create one recommendation entry mode with a fixed, explicit contract."""

    if not isinstance(runtime, LearningGuidanceRuntime):
        raise TypeError("runtime must be LearningGuidanceRuntime")
    if mode not in {"automatic_after_generation", "explicit_request"}:
        raise ValueError("mode must be an explicit RecommendationMode")

    async def resource_recommendation(state: LearningState) -> dict[str, object]:
        request_id = _required_state_text(state, "request_id")
        user_id = _optional_state_text(state, "user_id")
        subject = _optional_state_text(state, "subject")
        if user_id is None:
            return _recommendation_unavailable(
                request_id=request_id,
                mode=mode,
                reason="missing_user_id",
                user_id=None,
                subject=subject,
            )
        if subject is None:
            return _recommendation_unavailable(
                request_id=request_id,
                mode=mode,
                reason="missing_subject",
                user_id=user_id,
                subject=None,
            )

        generated_resources = (
            _recommendation_resources(state)
            if mode == "automatic_after_generation"
            else ()
        )
        if mode == "automatic_after_generation" and not generated_resources:
            return _recommendation_unavailable(
                request_id=request_id,
                mode=mode,
                reason="generated_resources_unavailable",
                user_id=user_id,
                subject=subject,
            )

        profile, history = await _load_available_context(
            runtime,
            user_id=user_id,
            subject=subject,
        )
        if profile is None:
            return _recommendation_unavailable(
                request_id=request_id,
                mode=mode,
                reason="profile_unavailable",
                user_id=user_id,
                subject=subject,
            )
        if history is None:
            return _recommendation_unavailable(
                request_id=request_id,
                mode=mode,
                reason="history_unavailable",
                user_id=user_id,
                subject=subject,
            )

        request = ResourceRecommendationEngineRequestV1(
            schema_version="resource_recommendation_engine_request_v1",
            request_id=request_id,
            mode=mode,
            user_id=user_id,
            subject=subject,
            profile=profile,
            history=history,
            generated_resources=generated_resources,
        )
        batch = await runtime.recommend_resources(request)
        if not isinstance(batch, ResourceRecommendationBatchV1):
            raise LearningGuidanceContractError(
                code="invalid_recommendation_batch_type",
                reason=(
                    "recommendation engine must return ResourceRecommendationBatchV1"
                ),
            )
        _validate_recommendation_binding(batch, request=request)
        output = ResourceRecommendationOutputV1(
            schema_version="resource_recommendation_output_v1",
            request_id=request_id,
            mode=mode,
            status="available",
            unavailable_reason=None,
            user_id=user_id,
            subject=subject,
            batch=batch,
        )
        return {
            RESOURCE_RECOMMENDATION_OUTPUT_STATE_KEY: output.model_dump(mode="json")
        }

    return resource_recommendation


__all__ = [
    "LEARNER_PATH_OUTPUT_STATE_KEY",
    "RECOMMENDATION_RESOURCE_CONTEXT_STATE_KEY",
    "RESOURCE_RECOMMENDATION_OUTPUT_STATE_KEY",
    "make_learner_path_planner_node",
    "make_resource_recommendation_node",
]
