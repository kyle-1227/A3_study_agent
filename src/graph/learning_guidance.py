"""Strict graph nodes for learner paths and resource recommendations."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import TypeVar

from src.graph.resource_final_v3 import ResourceFinalV3Recommendation
from src.graph.state import LearningState
from src.learning_guidance.contracts import (
    LEARNER_PATH_PROVIDER_MAX_CHARS,
    LEARNER_PATH_PROVIDER_MAX_STEPS,
    LearnerHistorySnapshotV1,
    LearnerPathEngineRequestV1,
    LearnerPathPlanV1,
    LearnerPathPlannerOutputV1,
    LearnerPathProviderProjectionV1,
    LearnerPathProviderStepV1,
    LearnerPathUnavailableReason,
    LearnerProfileSnapshotV1,
    RecommendationMode,
    RecommendationResourceContextV1,
    RecommendationUnavailableReason,
    ResourceRecommendationBatchV1,
    ResourceRecommendationEngineRequestV1,
    ResourceRecommendationEngineResultV1,
    ResourceRecommendationOutputV1,
)
from src.learning_guidance.recommendation_final import build_recommendation_final_v1
from src.learning_guidance.runtime import (
    LearningGuidanceContractError,
    LearningGuidanceRuntime,
)
from src.resource_contracts import normalize_requested_resource_types


LEARNER_PATH_OUTPUT_STATE_KEY = "learner_path_planner_output"
LEARNER_PATH_PROVIDER_PROJECTION_STATE_KEY = "learner_path_provider_projection"
RESOURCE_RECOMMENDATION_OUTPUT_STATE_KEY = "resource_recommendation_output"
RECOMMENDATION_RESOURCE_CONTEXT_STATE_KEY = "recommendation_resource_context"
RECOMMENDATION_FINAL_OUTPUT_FIELD = "recommendation_final_v1"

_StrictStateModelT = TypeVar(
    "_StrictStateModelT",
    LearnerPathPlannerOutputV1,
    LearnerPathProviderProjectionV1,
    ResourceRecommendationOutputV1,
)


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
    if field_name not in state:
        raise LearningGuidanceContractError(
            code=f"missing_{field_name}",
            reason=f"{field_name} must be explicitly present in graph state",
        )
    value = state[field_name]
    if value == "":
        return None
    if not isinstance(value, str):
        raise LearningGuidanceContractError(
            code=f"invalid_{field_name}",
            reason=f"{field_name} must be a string when present",
        )
    if not value.strip():
        raise LearningGuidanceContractError(
            code=f"invalid_{field_name}",
            reason=(
                f"{field_name} must use an explicit empty string or a normalized "
                "non-blank value"
            ),
        )
    if value != value.strip():
        raise LearningGuidanceContractError(
            code=f"invalid_{field_name}",
            reason=f"{field_name} must already be normalized",
        )
    return value


def _retrieval_plan_subjects(state: Mapping[str, object]) -> tuple[str, ...]:
    raw_plan = state.get("retrieval_plan")
    if not isinstance(raw_plan, (list, tuple)) or not raw_plan:
        raise LearningGuidanceContractError(
            code="invalid_learner_path_subject_scope",
            reason="learner path planning requires a non-empty retrieval plan",
        )
    subjects: list[str] = []
    for item in raw_plan:
        if not isinstance(item, Mapping):
            raise LearningGuidanceContractError(
                code="invalid_learner_path_subject_scope",
                reason="retrieval plan entries must be strict objects",
            )
        subject = item.get("subject")
        if (
            not isinstance(subject, str)
            or not subject.strip()
            or subject != subject.strip()
        ):
            raise LearningGuidanceContractError(
                code="invalid_learner_path_subject_scope",
                reason=("retrieval plan subjects must be normalized non-blank strings"),
            )
        if subject not in subjects:
            subjects.append(subject)
    return tuple(subjects)


def _learner_path_scope_reason(
    state: Mapping[str, object],
    *,
    subject: str,
) -> LearnerPathUnavailableReason | None:
    subjects = _retrieval_plan_subjects(state)
    if len(subjects) != 1 or subjects[0] != subject:
        return "unsupported_subject_scope"
    return None


def _provider_projection_for_output(
    output: LearnerPathPlannerOutputV1,
    *,
    max_steps: int,
    max_chars: int,
) -> LearnerPathProviderProjectionV1:
    if (
        isinstance(max_steps, bool)
        or not 1 <= max_steps <= LEARNER_PATH_PROVIDER_MAX_STEPS
    ):
        raise ValueError("max_steps must be within the provider projection contract")
    if (
        isinstance(max_chars, bool)
        or not 1 <= max_chars <= LEARNER_PATH_PROVIDER_MAX_CHARS
    ):
        raise ValueError("max_chars must be within the provider projection contract")
    if (
        output.provider_projection_max_steps != max_steps
        or output.provider_projection_max_chars != max_chars
    ):
        raise LearningGuidanceContractError(
            code="learner_path_provider_projection_policy_mismatch",
            reason="projection limits differ from the learner path output binding",
        )

    if output.status == "available":
        if output.plan is None:
            raise AssertionError("available learner path output requires a plan")
        if len(output.plan.steps) > max_steps:
            raise LearningGuidanceContractError(
                code="learner_path_provider_projection_too_large",
                reason="learner path exceeds the configured provider step limit",
            )
        projection = LearnerPathProviderProjectionV1(
            schema_version="learner_path_provider_projection_v1",
            status="available",
            unavailable_reason=None,
            subject=output.subject,
            summary=output.plan.summary,
            steps=tuple(
                LearnerPathProviderStepV1(
                    step_id=step.step_id,
                    position=step.position,
                    topic_id=step.topic_id,
                    title=step.title,
                    status=step.status,
                    estimated_hours=step.estimated_hours,
                    reason=step.reason,
                    recommended_resource_types=step.recommended_resource_types,
                )
                for step in output.plan.steps
            ),
        )
    else:
        projection = LearnerPathProviderProjectionV1(
            schema_version="learner_path_provider_projection_v1",
            status="unavailable",
            unavailable_reason=output.unavailable_reason,
            subject=output.subject,
            summary=None,
            steps=(),
        )

    encoded = json.dumps(
        projection.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(encoded) > max_chars:
        raise LearningGuidanceContractError(
            code="learner_path_provider_projection_too_large",
            reason="learner path exceeds the configured provider character limit",
        )
    return projection


def _learner_path_state_update(
    *,
    output: LearnerPathPlannerOutputV1,
    runtime: LearningGuidanceRuntime,
) -> dict[str, object]:
    projection = _provider_projection_for_output(
        output,
        max_steps=runtime.provider_projection_max_steps,
        max_chars=runtime.provider_projection_max_chars,
    )
    return {
        LEARNER_PATH_OUTPUT_STATE_KEY: output.model_dump(mode="json"),
        LEARNER_PATH_PROVIDER_PROJECTION_STATE_KEY: projection.model_dump(mode="json"),
    }


def _path_unavailable(
    *,
    runtime: LearningGuidanceRuntime,
    request_id: str,
    reason: LearnerPathUnavailableReason,
    user_id: str | None,
    subject: str | None,
) -> dict[str, object]:
    output = LearnerPathPlannerOutputV1(
        schema_version="learner_path_planner_output_v1",
        runtime_fingerprint=runtime.runtime_fingerprint,
        provider_projection_policy_fingerprint=(
            runtime.provider_projection_policy_fingerprint
        ),
        provider_projection_max_steps=runtime.provider_projection_max_steps,
        provider_projection_max_chars=runtime.provider_projection_max_chars,
        request_id=request_id,
        status="unavailable",
        unavailable_reason=reason,
        user_id=user_id,
        subject=subject,
        plan=None,
    )
    return _learner_path_state_update(output=output, runtime=runtime)


def _recommendation_unavailable(
    *,
    runtime: LearningGuidanceRuntime,
    request_id: str,
    mode: RecommendationMode,
    reason: RecommendationUnavailableReason,
    user_id: str | None,
    subject: str | None,
) -> dict[str, object]:
    output = ResourceRecommendationOutputV1(
        schema_version="resource_recommendation_output_v1",
        runtime_fingerprint=runtime.runtime_fingerprint,
        provider_projection_policy_fingerprint=(
            runtime.provider_projection_policy_fingerprint
        ),
        provider_projection_max_steps=runtime.provider_projection_max_steps,
        provider_projection_max_chars=runtime.provider_projection_max_chars,
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
        + [
            signal.signal_id
            for signal in profile.preferences
            if signal.subject == subject
        ]
    )


def _profile_signal_bindings(
    profile: LearnerProfileSnapshotV1,
) -> dict[str, tuple[str, str, str]]:
    bindings: dict[str, tuple[str, str, str]] = {}
    for signal in profile.skills:
        bindings[signal.signal_id] = ("skill", signal.subject, signal.topic_id)
    for signal in profile.goals:
        bindings[signal.signal_id] = ("goal", signal.subject, signal.topic_id)
    for signal in profile.preferences:
        bindings[signal.signal_id] = (
            "preference",
            signal.subject,
            signal.topic_id,
        )
    return bindings


def _history_bindings(
    history: LearnerHistorySnapshotV1,
) -> dict[str, tuple[str, str]]:
    return {
        event.history_id: (event.subject, event.topic_id) for event in history.events
    }


def _validate_plan_binding(
    plan: LearnerPathPlanV1,
    *,
    request: LearnerPathEngineRequestV1,
    runtime: LearningGuidanceRuntime,
) -> None:
    if plan.user_id != request.user_id or plan.subject != request.subject:
        raise LearningGuidanceContractError(
            code="learner_path_identity_mismatch",
            reason="learner path identity differs from its engine request",
        )
    subject_node = runtime.knowledge_graph.subject(request.subject)
    if subject_node is None:
        raise LearningGuidanceContractError(
            code="path_subject_not_in_knowledge_graph",
            reason="learner path subject is absent from the curated graph",
        )
    topics_by_id = {topic.topic_id: topic for topic in subject_node.topics}
    topic_positions = {
        topic.topic_id: position for position, topic in enumerate(subject_node.topics)
    }
    step_topic_ids = tuple(step.topic_id for step in plan.steps)
    if len(step_topic_ids) != len(set(step_topic_ids)):
        raise LearningGuidanceContractError(
            code="duplicate_path_topic",
            reason="learner path must contain each curated topic at most once",
        )
    step_topic_positions: list[int] = []
    for step in plan.steps:
        topic = topics_by_id.get(step.topic_id)
        if topic is None:
            raise LearningGuidanceContractError(
                code="path_topic_not_in_knowledge_graph",
                reason="learner path step references an unknown curated topic",
            )
        if step.title != topic.title or step.estimated_hours != topic.estimated_hours:
            raise LearningGuidanceContractError(
                code="path_topic_metadata_mismatch",
                reason=(
                    "learner path title and estimated hours must match the curated "
                    "topic"
                ),
            )
        step_topic_positions.append(topic_positions[step.topic_id])
    if step_topic_positions != sorted(step_topic_positions):
        raise LearningGuidanceContractError(
            code="path_topic_order_mismatch",
            reason="learner path topics must preserve curated topological order",
        )
    profile_bindings = _profile_signal_bindings(request.profile)
    history_bindings = _history_bindings(request.history)
    for step in plan.steps:
        profile_refs = tuple(
            profile_bindings.get(signal_id) for signal_id in step.profile_signal_ids
        )
        if any(binding is None for binding in profile_refs):
            raise LearningGuidanceContractError(
                code="unknown_path_profile_evidence",
                reason="learner path references profile signals outside its request",
            )
        if any(
            binding is not None
            and (binding[1], binding[2]) != (step.subject, step.topic_id)
            for binding in profile_refs
        ):
            raise LearningGuidanceContractError(
                code="off_topic_path_profile_evidence",
                reason="learner path profile evidence must match its exact topic",
            )
        history_refs = tuple(
            history_bindings.get(history_id) for history_id in step.history_ids
        )
        if any(binding is None for binding in history_refs):
            raise LearningGuidanceContractError(
                code="unknown_path_history_evidence",
                reason="learner path references history outside its request",
            )
        if any(binding != (step.subject, step.topic_id) for binding in history_refs):
            raise LearningGuidanceContractError(
                code="off_topic_path_history_evidence",
                reason="learner path history evidence must match its exact topic",
            )

    assigned_resource_types = tuple(
        resource_type
        for step in plan.steps
        for resource_type in step.recommended_resource_types
    )
    if len(assigned_resource_types) != len(set(assigned_resource_types)):
        raise LearningGuidanceContractError(
            code="duplicate_path_resource_binding",
            reason="each requested resource type must bind to exactly one topic",
        )
    if set(assigned_resource_types) != set(request.requested_resource_types):
        raise LearningGuidanceContractError(
            code="incomplete_path_resource_binding",
            reason="learner path must bind every requested resource type exactly once",
        )


def _recommendation_resources(
    state: Mapping[str, object],
) -> tuple[RecommendationResourceContextV1, ...]:
    raw = state.get(RECOMMENDATION_RESOURCE_CONTEXT_STATE_KEY)
    if raw is None:
        raise LearningGuidanceContractError(
            code="missing_recommendation_resource_context",
            reason=(
                "automatic recommendation requires an explicit resource context list"
            ),
        )
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


def automatic_recommendation_scope_reason(
    state: Mapping[str, object],
    *,
    subject: str,
) -> RecommendationUnavailableReason | None:
    """Reject multi-subject or mismatched generated-resource recommendation scope."""

    raw = state.get("evidence_requested_subjects")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise LearningGuidanceContractError(
            code="invalid_evidence_requested_subjects",
            reason="automatic recommendation requires explicit evidence subjects",
        )
    subjects: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip() or item != item.strip():
            raise LearningGuidanceContractError(
                code="invalid_evidence_requested_subjects",
                reason="evidence subjects must be normalized non-blank strings",
            )
        subjects.append(item)
    if len(subjects) != len(set(subjects)):
        raise LearningGuidanceContractError(
            code="invalid_evidence_requested_subjects",
            reason="evidence subjects must be unique",
        )
    if len(subjects) != 1 or subjects[0] != subject:
        return "unsupported_subject_scope"
    return None


def explicit_recommendation_scope_reason(
    state: Mapping[str, object],
    *,
    subject: str | None,
) -> RecommendationUnavailableReason | None:
    """Validate the current-message subject scope for explicit recommendations."""

    raw = state.get("subject_candidates")
    if not isinstance(raw, (list, tuple)):
        raise LearningGuidanceContractError(
            code="invalid_recommendation_subject_candidates",
            reason="explicit recommendation requires an explicit subject candidate list",
        )
    subjects: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip() or item != item.strip():
            raise LearningGuidanceContractError(
                code="invalid_recommendation_subject_candidates",
                reason="recommendation subjects must be normalized non-blank strings",
            )
        if item in subjects:
            raise LearningGuidanceContractError(
                code="invalid_recommendation_subject_candidates",
                reason="recommendation subjects must be unique",
            )
        subjects.append(item)
    continuation_applied = state.get("workspace_continuation_applied")
    if not isinstance(continuation_applied, bool):
        raise LearningGuidanceContractError(
            code="invalid_recommendation_workspace_continuation",
            reason="workspace_continuation_applied must be an explicit boolean",
        )
    if not subjects:
        if continuation_applied:
            continuation = state.get("workspace_continuation")
            if not isinstance(continuation, Mapping):
                raise LearningGuidanceContractError(
                    code="invalid_recommendation_workspace_continuation",
                    reason="applied workspace continuation requires strict context",
                )
            continuation_subject = continuation.get("normalized_subject")
            if (
                continuation.get("continuation_applied") is not True
                or not isinstance(continuation_subject, str)
                or not continuation_subject.strip()
                or continuation_subject != continuation_subject.strip()
                or subject != continuation_subject
            ):
                raise LearningGuidanceContractError(
                    code="recommendation_subject_scope_mismatch",
                    reason=(
                        "selected subject must match the applied workspace continuation"
                    ),
                )
            return None
        if subject is not None:
            raise LearningGuidanceContractError(
                code="recommendation_subject_scope_mismatch",
                reason="a missing explicit subject scope cannot carry a selected subject",
            )
        return "missing_subject"
    if continuation_applied:
        raise LearningGuidanceContractError(
            code="invalid_recommendation_workspace_continuation",
            reason="explicit subject candidates may not also inherit workspace scope",
        )
    if subject != subjects[0]:
        raise LearningGuidanceContractError(
            code="recommendation_subject_scope_mismatch",
            reason="selected subject must match the first explicit subject candidate",
        )
    if len(subjects) != 1:
        return "unsupported_subject_scope"
    return None


def _validate_explicit_recommendation_catalog(
    batch: ResourceRecommendationBatchV1,
    *,
    runtime: LearningGuidanceRuntime,
) -> None:
    subject_node = runtime.knowledge_graph.subject(batch.subject)
    if subject_node is None:
        raise LearningGuidanceContractError(
            code="recommendation_subject_not_in_knowledge_graph",
            reason="explicit recommendation subject is absent from the catalog",
        )
    catalog_by_id = {
        resource.resource_id: (
            resource.resource_type,
            batch.subject,
            topic.topic_id,
            resource.title,
        )
        for topic in subject_node.topics
        for resource in topic.resources
    }
    for item in batch.items:
        if catalog_by_id.get(item.resource_id) != (
            item.resource_type,
            item.subject,
            item.topic_id,
            item.title,
        ):
            raise LearningGuidanceContractError(
                code="recommendation_catalog_binding_mismatch",
                reason=(
                    "explicit recommendation must exactly match one curated "
                    "catalog resource"
                ),
            )


def _validate_recommendation_binding(
    batch: ResourceRecommendationBatchV1,
    *,
    request: ResourceRecommendationEngineRequestV1,
    runtime: LearningGuidanceRuntime,
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
    profile_bindings = _profile_signal_bindings(request.profile)
    history_bindings = _history_bindings(request.history)
    resources_by_id = {
        resource.resource_id: resource for resource in request.generated_resources
    }
    allowed_resource_ids = frozenset(resources_by_id)
    if request.mode == "explicit_request":
        _validate_explicit_recommendation_catalog(batch, runtime=runtime)
    for item in batch.items:
        profile_refs = tuple(
            profile_bindings.get(signal_id) for signal_id in item.profile_signal_ids
        )
        if any(binding is None for binding in profile_refs):
            raise LearningGuidanceContractError(
                code="unknown_recommendation_profile_evidence",
                reason="recommendation references profile signals outside its request",
            )
        if any(
            binding is not None
            and (binding[1], binding[2]) != (item.subject, item.topic_id)
            for binding in profile_refs
        ):
            raise LearningGuidanceContractError(
                code="off_topic_recommendation_profile_evidence",
                reason="recommendation profile evidence must match its exact topic",
            )
        referenced_kinds = {
            binding[0] for binding in profile_refs if binding is not None
        }
        if referenced_kinds != {"skill", "goal", "preference"}:
            raise LearningGuidanceContractError(
                code="incomplete_recommendation_profile_evidence",
                reason=("recommendation must cite skill, goal, and preference signals"),
            )
        history_refs = tuple(
            history_bindings.get(history_id) for history_id in item.history_ids
        )
        if any(binding is None for binding in history_refs):
            raise LearningGuidanceContractError(
                code="unknown_recommendation_history_evidence",
                reason="recommendation references history outside its request",
            )
        if any(binding != (item.subject, item.topic_id) for binding in history_refs):
            raise LearningGuidanceContractError(
                code="off_topic_recommendation_history_evidence",
                reason="recommendation history evidence must match its exact topic",
            )
        if not set(item.source_resource_ids).issubset(allowed_resource_ids):
            raise LearningGuidanceContractError(
                code="unknown_recommendation_resource_evidence",
                reason="recommendation references generated resources outside its request",
            )
        if request.mode == "automatic_after_generation" and (
            item.resource_id not in allowed_resource_ids
            or item.resource_id not in item.source_resource_ids
        ):
            raise LearningGuidanceContractError(
                code="unknown_recommendation_target_resource",
                reason=(
                    "automatic recommendation must target and cite a verified "
                    "generated resource"
                ),
            )
        if request.mode == "automatic_after_generation":
            target = resources_by_id[item.resource_id]
            if (
                target.resource_type != item.resource_type
                or target.subject != item.subject
                or target.topic_id != item.topic_id
                or target.title != item.title
            ):
                raise LearningGuidanceContractError(
                    code="recommendation_target_binding_mismatch",
                    reason=(
                        "automatic recommendation target must match the verified "
                        "resource type, subject, topic, and title"
                    ),
                )
            if any(
                resources_by_id[source_id].topic_id != item.topic_id
                for source_id in item.source_resource_ids
            ):
                raise LearningGuidanceContractError(
                    code="off_topic_recommendation_resource_evidence",
                    reason="generated resource evidence must match the target topic",
                )


def _strict_json_state_model(
    raw: object,
    *,
    field_name: str,
    model_type: type[_StrictStateModelT],
    max_serialized_chars: int | None,
) -> _StrictStateModelT:
    if not isinstance(raw, Mapping):
        raise LearningGuidanceContractError(
            code=f"invalid_{field_name}",
            reason=f"{field_name} must be a strict JSON object",
        )
    try:
        encoded = json.dumps(
            dict(raw),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if max_serialized_chars is not None and len(encoded) > max_serialized_chars:
            raise LearningGuidanceContractError(
                code=f"{field_name}_too_large",
                reason=f"{field_name} exceeds its serialized character limit",
            )
        # Pydantic strict models reject their own JSON datetime/tuple transport
        # forms. Decode those JSON-native forms, then require byte-for-byte
        # canonical equality so no scalar coercion or repair can be accepted.
        parsed = model_type.model_validate_json(encoded, strict=False)
        canonical = json.dumps(
            parsed.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if canonical != encoded:
            raise ValueError(f"{field_name} is not canonical JSON")
        return parsed
    except (TypeError, ValueError) as exc:
        raise LearningGuidanceContractError(
            code=f"invalid_{field_name}",
            reason=f"{field_name} violates its strict schema",
        ) from exc


def learner_path_output_from_state(
    state: Mapping[str, object],
) -> LearnerPathPlannerOutputV1:
    """Revalidate the checkpoint-safe learner path output without coercion."""

    parsed = _strict_json_state_model(
        state.get(LEARNER_PATH_OUTPUT_STATE_KEY),
        field_name=LEARNER_PATH_OUTPUT_STATE_KEY,
        model_type=LearnerPathPlannerOutputV1,
        max_serialized_chars=None,
    )
    if not isinstance(parsed, LearnerPathPlannerOutputV1):
        raise AssertionError("learner path parser returned the wrong model type")
    expected_request_id = _required_state_text(state, "request_id")
    expected_user_id = _optional_state_text(state, "user_id")
    expected_subject = _optional_state_text(state, "subject")
    if parsed.request_id != expected_request_id:
        raise LearningGuidanceContractError(
            code="learner_path_request_mismatch",
            reason="learner path output request_id differs from graph state",
        )
    if parsed.user_id != expected_user_id:
        raise LearningGuidanceContractError(
            code="learner_path_user_mismatch",
            reason="learner path output user_id differs from graph state",
        )
    if parsed.subject != expected_subject:
        raise LearningGuidanceContractError(
            code="learner_path_subject_mismatch",
            reason="learner path output subject differs from graph state",
        )
    if expected_subject is None:
        if not (
            parsed.status == "unavailable"
            and parsed.unavailable_reason == "missing_subject"
        ):
            raise LearningGuidanceContractError(
                code="learner_path_scope_mismatch",
                reason="learner path output must reflect the missing subject",
            )
        return parsed

    scope_reason = _learner_path_scope_reason(state, subject=expected_subject)
    if scope_reason is not None:
        if not (
            parsed.status == "unavailable" and parsed.unavailable_reason == scope_reason
        ):
            raise LearningGuidanceContractError(
                code="learner_path_scope_mismatch",
                reason=(
                    "learner path output does not reflect the retrieval plan scope"
                ),
            )
        return parsed

    if expected_user_id is None and not (
        parsed.status == "unavailable"
        and parsed.unavailable_reason == "missing_user_id"
    ):
        raise LearningGuidanceContractError(
            code="learner_path_user_mismatch",
            reason="learner path output must reflect the missing user_id",
        )
    if parsed.unavailable_reason == "unsupported_subject_scope":
        raise LearningGuidanceContractError(
            code="learner_path_scope_mismatch",
            reason=(
                "learner path output reports an unsupported scope for a single "
                "matching subject"
            ),
        )
    return parsed


def _validate_guidance_output_runtime_binding(
    output: LearnerPathPlannerOutputV1 | ResourceRecommendationOutputV1,
    *,
    runtime: LearningGuidanceRuntime,
) -> None:
    if output.runtime_fingerprint != runtime.runtime_fingerprint:
        raise LearningGuidanceContractError(
            code="learning_guidance_runtime_mismatch",
            reason="checkpoint guidance output belongs to a different runtime",
        )
    if (
        output.provider_projection_policy_fingerprint
        != runtime.provider_projection_policy_fingerprint
        or output.provider_projection_max_steps != runtime.provider_projection_max_steps
        or output.provider_projection_max_chars != runtime.provider_projection_max_chars
    ):
        raise LearningGuidanceContractError(
            code="learning_guidance_projection_policy_mismatch",
            reason="checkpoint guidance output uses a different projection policy",
        )


def _learner_path_provider_projection_for_output_from_state(
    state: Mapping[str, object],
    *,
    output: LearnerPathPlannerOutputV1,
) -> LearnerPathProviderProjectionV1:
    max_steps = output.provider_projection_max_steps
    max_chars = output.provider_projection_max_chars
    projection = _strict_json_state_model(
        state.get(LEARNER_PATH_PROVIDER_PROJECTION_STATE_KEY),
        field_name=LEARNER_PATH_PROVIDER_PROJECTION_STATE_KEY,
        model_type=LearnerPathProviderProjectionV1,
        max_serialized_chars=max_chars,
    )
    expected = _provider_projection_for_output(
        output,
        max_steps=max_steps,
        max_chars=max_chars,
    )
    if projection != expected:
        raise LearningGuidanceContractError(
            code="learner_path_provider_projection_mismatch",
            reason="provider projection differs from the validated learner path output",
        )
    return projection


def learner_path_provider_projection_from_state(
    state: Mapping[str, object],
) -> LearnerPathProviderProjectionV1:
    """Return the exact projection under its self-validated checkpoint policy."""

    output = learner_path_output_from_state(state)
    return _learner_path_provider_projection_for_output_from_state(
        state,
        output=output,
    )


def learner_path_provider_projection_for_runtime_from_state(
    state: Mapping[str, object],
    *,
    runtime: LearningGuidanceRuntime,
) -> LearnerPathProviderProjectionV1:
    """Return a projection only when its checkpoint runtime is still current."""

    output = learner_path_output_from_state(state)
    _validate_guidance_output_runtime_binding(output, runtime=runtime)
    return _learner_path_provider_projection_for_output_from_state(
        state,
        output=output,
    )


def resource_recommendation_output_from_state(
    state: Mapping[str, object],
    *,
    expected_mode: RecommendationMode,
) -> ResourceRecommendationOutputV1:
    """Revalidate one recommendation result and its fixed graph entry mode."""

    parsed = _strict_json_state_model(
        state.get(RESOURCE_RECOMMENDATION_OUTPUT_STATE_KEY),
        field_name=RESOURCE_RECOMMENDATION_OUTPUT_STATE_KEY,
        model_type=ResourceRecommendationOutputV1,
        max_serialized_chars=None,
    )
    if not isinstance(parsed, ResourceRecommendationOutputV1):
        raise AssertionError("recommendation parser returned the wrong model type")
    if parsed.mode != expected_mode:
        raise LearningGuidanceContractError(
            code="recommendation_mode_mismatch",
            reason="recommendation output mode differs from the graph entry mode",
        )
    expected_request_id = _required_state_text(state, "request_id")
    expected_user_id = _optional_state_text(state, "user_id")
    expected_subject = _optional_state_text(state, "subject")
    if parsed.request_id != expected_request_id:
        raise LearningGuidanceContractError(
            code="recommendation_request_mismatch",
            reason="recommendation output request_id differs from graph state",
        )
    if parsed.user_id != expected_user_id:
        raise LearningGuidanceContractError(
            code="recommendation_user_mismatch",
            reason="recommendation output user_id differs from graph state",
        )
    if parsed.subject != expected_subject:
        raise LearningGuidanceContractError(
            code="recommendation_subject_mismatch",
            reason="recommendation output subject differs from graph state",
        )
    if expected_mode == "explicit_request":
        if expected_user_id is None:
            if not (
                parsed.status == "unavailable"
                and parsed.unavailable_reason == "missing_user_id"
            ):
                raise LearningGuidanceContractError(
                    code="recommendation_user_mismatch",
                    reason="recommendation output must reflect the missing user_id",
                )
            return parsed
        scope_reason = explicit_recommendation_scope_reason(
            state,
            subject=expected_subject,
        )
        if scope_reason is not None:
            if not (
                parsed.status == "unavailable"
                and parsed.unavailable_reason == scope_reason
            ):
                raise LearningGuidanceContractError(
                    code="recommendation_subject_scope_mismatch",
                    reason=(
                        "explicit recommendation output does not reflect the current "
                        "subject scope"
                    ),
                )
            return parsed
        if parsed.unavailable_reason in {
            "missing_subject",
            "unsupported_subject_scope",
        }:
            raise LearningGuidanceContractError(
                code="recommendation_subject_scope_mismatch",
                reason="recommendation output reports a subject scope not in graph state",
            )
    if parsed.status == "available" and parsed.mode == "automatic_after_generation":
        if parsed.batch is None:
            raise AssertionError("available recommendation output requires a batch")
        resources_by_id = {
            item.resource_id: item for item in _recommendation_resources(state)
        }
        allowed_resource_ids = frozenset(resources_by_id)
        for item in parsed.batch.items:
            if not set(item.source_resource_ids).issubset(allowed_resource_ids):
                raise LearningGuidanceContractError(
                    code="unknown_recommendation_resource_evidence",
                    reason=(
                        "recommendation output references resources outside the "
                        "verified bundle context"
                    ),
                )
            if (
                item.resource_id not in allowed_resource_ids
                or item.resource_id not in item.source_resource_ids
            ):
                raise LearningGuidanceContractError(
                    code="unknown_recommendation_target_resource",
                    reason=(
                        "automatic recommendation target is outside the verified "
                        "bundle context"
                    ),
                )
            target = resources_by_id[item.resource_id]
            if (
                target.resource_type != item.resource_type
                or target.subject != item.subject
                or target.topic_id != item.topic_id
                or target.title != item.title
            ):
                raise LearningGuidanceContractError(
                    code="recommendation_target_binding_mismatch",
                    reason=(
                        "recommendation target differs from the verified resource "
                        "type, subject, topic, or title"
                    ),
                )
            if any(
                resources_by_id[source_id].topic_id != item.topic_id
                for source_id in item.source_resource_ids
            ):
                raise LearningGuidanceContractError(
                    code="off_topic_recommendation_resource_evidence",
                    reason="recommendation source resources must match its topic",
                )
    return parsed


def resource_recommendation_output_for_runtime_from_state(
    state: Mapping[str, object],
    *,
    expected_mode: RecommendationMode,
    runtime: LearningGuidanceRuntime,
) -> ResourceRecommendationOutputV1:
    """Return a recommendation only when its checkpoint runtime is current."""

    output = resource_recommendation_output_from_state(
        state,
        expected_mode=expected_mode,
    )
    _validate_guidance_output_runtime_binding(output, runtime=runtime)
    if (
        output.status == "available"
        and output.mode == "explicit_request"
        and output.batch is not None
    ):
        _validate_explicit_recommendation_catalog(output.batch, runtime=runtime)
    return output


def resource_final_recommendations(
    output: ResourceRecommendationOutputV1,
) -> tuple[ResourceFinalV3Recommendation, ...]:
    """Project a validated recommendation batch into Resource Final V3."""

    if output.status == "unavailable":
        return ()
    if output.batch is None:
        raise AssertionError("available recommendation output requires a batch")
    if output.mode == "automatic_after_generation":
        trigger = "automatic"
    elif output.mode == "explicit_request":
        trigger = "explicit_request"
    else:
        raise AssertionError("validated recommendation output has an unknown mode")
    return tuple(
        ResourceFinalV3Recommendation(
            recommendation_id=item.recommendation_id,
            resource_id=item.resource_id,
            resource_type=item.resource_type,
            trigger=trigger,
            rank=item.rank,
            title=item.title,
            reason=item.reason,
        )
        for item in output.batch.items
    )


_RECOMMENDATION_UNAVAILABLE_MESSAGES: dict[
    RecommendationUnavailableReason,
    str,
] = {
    "missing_user_id": "缺少明确的用户身份",
    "missing_subject": "缺少明确的学习科目",
    "profile_unavailable": "学习画像不可用",
    "history_unavailable": "学习历史不可用",
    "generated_resources_unavailable": "没有真实生成且可推荐的资源",
    "no_eligible_candidates": "没有满足严格证据和评分门槛的推荐候选",
    "unsupported_subject_scope": "资源跨多个科目或与当前主科目不一致",
}


def recommendation_public_status_message(
    output: ResourceRecommendationOutputV1,
) -> str:
    """Return a bounded public status without inventing a neutral score."""

    if output.status == "available":
        if output.batch is None:
            raise AssertionError("available recommendation output requires a batch")
        return f"个性化推荐已生成（{len(output.batch.items)} 项）。"
    if output.unavailable_reason is None:
        raise AssertionError("unavailable recommendation output requires a reason")
    reason = _RECOMMENDATION_UNAVAILABLE_MESSAGES[output.unavailable_reason]
    return f"个性化推荐暂不可用 [{output.unavailable_reason}]：{reason}。"


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
        if subject is None:
            return _path_unavailable(
                runtime=runtime,
                request_id=request_id,
                reason="missing_subject",
                user_id=user_id,
                subject=None,
            )
        scope_reason = _learner_path_scope_reason(state, subject=subject)
        if scope_reason is not None:
            return _path_unavailable(
                runtime=runtime,
                request_id=request_id,
                reason=scope_reason,
                user_id=user_id,
                subject=subject,
            )
        if user_id is None:
            return _path_unavailable(
                runtime=runtime,
                request_id=request_id,
                reason="missing_user_id",
                user_id=None,
                subject=subject,
            )

        profile, history = await _load_available_context(
            runtime,
            user_id=user_id,
            subject=subject,
        )
        if profile is None:
            return _path_unavailable(
                runtime=runtime,
                request_id=request_id,
                reason="profile_unavailable",
                user_id=user_id,
                subject=subject,
            )
        if history is None:
            return _path_unavailable(
                runtime=runtime,
                request_id=request_id,
                reason="history_unavailable",
                user_id=user_id,
                subject=subject,
            )

        requested_resource_types = tuple(
            normalize_requested_resource_types(
                state.get("requested_resource_types") or [],
                state.get("requested_resource_type") or "",
            )
        )
        if not requested_resource_types:
            raise LearningGuidanceContractError(
                code="missing_path_resource_types",
                reason="learner path planning requires explicit resource types",
            )

        request = LearnerPathEngineRequestV1(
            schema_version="learner_path_engine_request_v1",
            request_id=request_id,
            user_id=user_id,
            subject=subject,
            requested_resource_types=requested_resource_types,
            profile=profile,
            history=history,
        )
        plan = await runtime.plan_learning_path(request)
        if not isinstance(plan, LearnerPathPlanV1):
            raise LearningGuidanceContractError(
                code="invalid_learner_path_type",
                reason="learner path engine must return LearnerPathPlanV1",
            )
        _validate_plan_binding(
            plan,
            request=request,
            runtime=runtime,
        )
        output = LearnerPathPlannerOutputV1(
            schema_version="learner_path_planner_output_v1",
            runtime_fingerprint=runtime.runtime_fingerprint,
            provider_projection_policy_fingerprint=(
                runtime.provider_projection_policy_fingerprint
            ),
            provider_projection_max_steps=runtime.provider_projection_max_steps,
            provider_projection_max_chars=runtime.provider_projection_max_chars,
            request_id=request_id,
            status="available",
            unavailable_reason=None,
            user_id=user_id,
            subject=subject,
            plan=plan,
        )
        return _learner_path_state_update(output=output, runtime=runtime)

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
                runtime=runtime,
                request_id=request_id,
                mode=mode,
                reason="missing_user_id",
                user_id=None,
                subject=subject,
            )
        if mode == "explicit_request":
            scope_reason = explicit_recommendation_scope_reason(
                state,
                subject=subject,
            )
            if scope_reason is not None:
                return _recommendation_unavailable(
                    runtime=runtime,
                    request_id=request_id,
                    mode=mode,
                    reason=scope_reason,
                    user_id=user_id,
                    subject=subject,
                )
        if subject is None:
            return _recommendation_unavailable(
                runtime=runtime,
                request_id=request_id,
                mode=mode,
                reason="missing_subject",
                user_id=user_id,
                subject=None,
            )

        if mode == "automatic_after_generation":
            scope_reason = automatic_recommendation_scope_reason(
                state,
                subject=subject,
            )
            if scope_reason is not None:
                return _recommendation_unavailable(
                    runtime=runtime,
                    request_id=request_id,
                    mode=mode,
                    reason=scope_reason,
                    user_id=user_id,
                    subject=subject,
                )
            generated_resources = _recommendation_resources(state)
        else:
            generated_resources = ()
        if mode == "automatic_after_generation" and not generated_resources:
            return _recommendation_unavailable(
                runtime=runtime,
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
                runtime=runtime,
                request_id=request_id,
                mode=mode,
                reason="profile_unavailable",
                user_id=user_id,
                subject=subject,
            )
        if history is None:
            return _recommendation_unavailable(
                runtime=runtime,
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
        result = await runtime.recommend_resources(request)
        if not isinstance(result, ResourceRecommendationEngineResultV1):
            raise LearningGuidanceContractError(
                code="invalid_recommendation_result_type",
                reason=(
                    "recommendation engine must return "
                    "ResourceRecommendationEngineResultV1"
                ),
            )
        if (
            result.request_id != request.request_id
            or result.mode != request.mode
            or result.user_id != request.user_id
            or result.subject != request.subject
        ):
            raise LearningGuidanceContractError(
                code="recommendation_engine_result_mismatch",
                reason="recommendation engine result differs from its request identity",
            )
        if result.status == "unavailable":
            return _recommendation_unavailable(
                runtime=runtime,
                request_id=request_id,
                mode=mode,
                reason="no_eligible_candidates",
                user_id=user_id,
                subject=subject,
            )
        batch = result.batch
        if batch is None:
            raise AssertionError("available recommendation engine result has no batch")
        _validate_recommendation_binding(
            batch,
            request=request,
            runtime=runtime,
        )
        output = ResourceRecommendationOutputV1(
            schema_version="resource_recommendation_output_v1",
            runtime_fingerprint=runtime.runtime_fingerprint,
            provider_projection_policy_fingerprint=(
                runtime.provider_projection_policy_fingerprint
            ),
            provider_projection_max_steps=runtime.provider_projection_max_steps,
            provider_projection_max_chars=runtime.provider_projection_max_chars,
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


def make_recommendation_final_output_node(
    runtime: LearningGuidanceRuntime,
) -> Callable[[LearningState], dict[str, object]]:
    """Create the strict recommendation-only terminal projection node."""

    if not isinstance(runtime, LearningGuidanceRuntime):
        raise TypeError("runtime must be LearningGuidanceRuntime")

    def recommendation_final_output(state: LearningState) -> dict[str, object]:
        if _required_state_text(state, "response_mode") != "recommendation":
            raise LearningGuidanceContractError(
                code="recommendation_final_response_mode_mismatch",
                reason="recommendation final requires response_mode=recommendation",
            )
        request_id = _required_state_text(state, "request_id")
        thread_id = _required_state_text(state, "thread_id")
        user_id = _optional_state_text(state, "user_id")
        output = resource_recommendation_output_for_runtime_from_state(
            state,
            expected_mode="explicit_request",
            runtime=runtime,
        )
        final = build_recommendation_final_v1(
            thread_id=thread_id,
            request_id=request_id,
            output=output,
            knowledge_graph=runtime.knowledge_graph,
            expected_user_id=user_id,
            expected_runtime_fingerprint=runtime.runtime_fingerprint,
        )
        return {
            RECOMMENDATION_FINAL_OUTPUT_FIELD: final.model_dump(mode="json"),
            "final_response_type": "recommendation_final",
        }

    return recommendation_final_output


__all__ = [
    "LEARNER_PATH_OUTPUT_STATE_KEY",
    "LEARNER_PATH_PROVIDER_PROJECTION_STATE_KEY",
    "RECOMMENDATION_RESOURCE_CONTEXT_STATE_KEY",
    "RECOMMENDATION_FINAL_OUTPUT_FIELD",
    "RESOURCE_RECOMMENDATION_OUTPUT_STATE_KEY",
    "automatic_recommendation_scope_reason",
    "explicit_recommendation_scope_reason",
    "learner_path_output_from_state",
    "learner_path_provider_projection_for_runtime_from_state",
    "learner_path_provider_projection_from_state",
    "make_learner_path_planner_node",
    "make_recommendation_final_output_node",
    "make_resource_recommendation_node",
    "recommendation_public_status_message",
    "resource_final_recommendations",
    "resource_recommendation_output_for_runtime_from_state",
    "resource_recommendation_output_from_state",
]
