"""Strict recommendation-only public terminal tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.learning_guidance.contracts import (
    RecommendationMode,
    RecommendationUnavailableReason,
    ResourceRecommendationBatchV1,
    ResourceRecommendationItemV1,
    ResourceRecommendationOutputV1,
    RecommendationScoreFactorsV1,
    RecommendationScoreWeightsV1,
    build_learner_path_provider_policy_fingerprint,
)
from src.learning_guidance.knowledge_graph import (
    CatalogResourceV1,
    KnowledgeGraphV1,
    KnowledgeSubjectV1,
    KnowledgeTopicV1,
)
from src.learning_guidance.recommendation_final import (
    RecommendationFinalContractError,
    RecommendationFinalV1,
    build_recommendation_final_v1,
    validate_recommendation_final_v1,
)
from src.resource_contracts import ResourceType


REQUEST_ID = "00000000-0000-4000-8000-000000000001"
THREAD_ID = "thread-recommendation-1"
USER_ID = "learner-1"
RUNTIME_FINGERPRINT = "a" * 64


def _knowledge_graph(*, include_unrelated_wide_entry: bool = False) -> KnowledgeGraphV1:
    topics = [
        KnowledgeTopicV1(
            topic_id="python.basics",
            title="Python basics",
            difficulty=0.2,
            estimated_hours=2.0,
            prerequisite_topic_ids=(),
            knowledge_points=("variables",),
            resources=(
                CatalogResourceV1(
                    resource_id="python.basics.notes",
                    resource_type="review_doc",
                    title="Python basics notes",
                ),
            ),
        ),
        KnowledgeTopicV1(
            topic_id="python.loops",
            title="Python loops",
            difficulty=0.4,
            estimated_hours=3.0,
            prerequisite_topic_ids=("python.basics",),
            knowledge_points=("for loops", "while loops"),
            resources=(
                CatalogResourceV1(
                    resource_id="python.loops.quiz",
                    resource_type="quiz",
                    title="Python loops quiz",
                ),
            ),
        ),
    ]
    if include_unrelated_wide_entry:
        topics.append(
            KnowledgeTopicV1(
                topic_id="python." + "x" * 170,
                title="Wide but valid catalog topic",
                difficulty=0.6,
                estimated_hours=1.0,
                prerequisite_topic_ids=("python.loops",),
                knowledge_points=("boundary coverage",),
                resources=(
                    CatalogResourceV1(
                        resource_id="python.boundary.resource",
                        resource_type="quiz",
                        title="T" * 300,
                    ),
                ),
            )
        )
    return KnowledgeGraphV1(
        schema_version="knowledge_graph_v1",
        data_version="2026-07-15",
        subjects=(
            KnowledgeSubjectV1(
                subject_id="python",
                title="Python",
                topics=tuple(topics),
            ),
        ),
    )


def _item(
    *,
    resource_id: str = "python.loops.quiz",
    resource_type: ResourceType = "quiz",
    topic_id: str = "python.loops",
    title: str = "Python loops quiz",
    mode: RecommendationMode = "explicit_request",
    reason: str = "The learner needs more practice with loops.",
) -> ResourceRecommendationItemV1:
    return ResourceRecommendationItemV1(
        recommendation_id="recommendation-python-loops-1",
        resource_id=resource_id,
        resource_type=resource_type,
        subject="python",
        topic_id=topic_id,
        title=title,
        rank=1,
        score_factors=RecommendationScoreFactorsV1(
            weakness=0.8,
            forgetting=0.6,
            preference=0.7,
            goal=0.9,
            combined=0.75,
            weights=RecommendationScoreWeightsV1(
                weakness=0.25,
                forgetting=0.25,
                preference=0.25,
                goal=0.25,
            ),
        ),
        reason=reason,
        profile_signal_ids=("skill-python-loops", "goal-python", "prefer-practice"),
        history_ids=("history-python-loops-1",),
        source_resource_ids=("generated-quiz-1",)
        if mode == "automatic_after_generation"
        else (),
    )


def _available_output(
    *,
    resource_id: str = "python.loops.quiz",
    resource_type: ResourceType = "quiz",
    topic_id: str = "python.loops",
    title: str = "Python loops quiz",
    mode: RecommendationMode = "explicit_request",
    request_id: str = REQUEST_ID,
    reason: str = "The learner needs more practice with loops.",
) -> ResourceRecommendationOutputV1:
    max_steps = 50
    max_chars = 65_536
    return ResourceRecommendationOutputV1(
        schema_version="resource_recommendation_output_v1",
        runtime_fingerprint=RUNTIME_FINGERPRINT,
        provider_projection_policy_fingerprint=(
            build_learner_path_provider_policy_fingerprint(
                max_steps=max_steps,
                max_chars=max_chars,
            )
        ),
        provider_projection_max_steps=max_steps,
        provider_projection_max_chars=max_chars,
        request_id=request_id,
        mode=mode,
        status="available",
        unavailable_reason=None,
        user_id=USER_ID,
        subject="python",
        batch=ResourceRecommendationBatchV1(
            schema_version="resource_recommendation_batch_v1",
            mode=mode,
            user_id=USER_ID,
            subject="python",
            generated_at=datetime(2026, 7, 15, 8, 0, tzinfo=timezone.utc),
            items=(
                _item(
                    resource_id=resource_id,
                    resource_type=resource_type,
                    topic_id=topic_id,
                    title=title,
                    mode=mode,
                    reason=reason,
                ),
            ),
            summary="One personalized recommendation is available.",
        ),
    )


def _unavailable_output(
    *,
    reason: RecommendationUnavailableReason = "profile_unavailable",
    user_id: str | None = USER_ID,
    subject: str | None = "python",
) -> ResourceRecommendationOutputV1:
    max_steps = 50
    max_chars = 65_536
    return ResourceRecommendationOutputV1(
        schema_version="resource_recommendation_output_v1",
        runtime_fingerprint=RUNTIME_FINGERPRINT,
        provider_projection_policy_fingerprint=(
            build_learner_path_provider_policy_fingerprint(
                max_steps=max_steps,
                max_chars=max_chars,
            )
        ),
        provider_projection_max_steps=max_steps,
        provider_projection_max_chars=max_chars,
        request_id=REQUEST_ID,
        mode="explicit_request",
        status="unavailable",
        unavailable_reason=reason,
        user_id=user_id,
        subject=subject,
        batch=None,
    )


def _build_available() -> RecommendationFinalV1:
    return build_recommendation_final_v1(
        thread_id=THREAD_ID,
        request_id=REQUEST_ID,
        output=_available_output(),
        knowledge_graph=_knowledge_graph(),
        expected_user_id=USER_ID,
        expected_runtime_fingerprint=RUNTIME_FINGERPRINT,
    )


def test_available_final_is_stable_and_bound_to_curated_candidates() -> None:
    first = _build_available()
    second = _build_available()

    assert first == second
    assert first.schema_version == "recommendation_final_v1"
    assert first.type == "recommendation_final"
    assert first.terminal_status == "available"
    assert first.user_id == USER_ID
    assert first.learning_guidance_runtime_fingerprint == RUNTIME_FINGERPRINT
    assert first.recommendation_final_id.startswith("recommendation-final:v1:")
    assert first.payload_hash.startswith("recommendation-final-payload:v1:")
    assert first.candidate_snapshot is not None
    assert first.candidate_snapshot.source_fingerprint == (
        _knowledge_graph().artifact_fingerprint
    )
    assert first.candidate_snapshot.candidate_count == 2
    assert first.candidate_snapshot.targets[0].resource_id == "python.loops.quiz"
    assert first.recommendations[0].score == 0.75
    assert first.summary == "Personalized recommendations available: 1."

    public_payload = first.model_dump(mode="json")
    assert "profile_signal_ids" not in str(public_payload)
    assert "history_ids" not in str(public_payload)
    assert validate_recommendation_final_v1(public_payload) == first
    revalidated_instance = validate_recommendation_final_v1(first)
    assert revalidated_instance == first
    assert revalidated_instance is not first
    assert isinstance(first.recommendations, tuple)
    assert isinstance(first.candidate_snapshot.targets, tuple)


def test_unavailable_final_is_authoritative_without_fake_candidates() -> None:
    final = build_recommendation_final_v1(
        thread_id=THREAD_ID,
        request_id=REQUEST_ID,
        output=_unavailable_output(),
        knowledge_graph=_knowledge_graph(),
        expected_user_id=USER_ID,
        expected_runtime_fingerprint=RUNTIME_FINGERPRINT,
    )

    assert final.terminal_status == "unavailable"
    assert final.unavailable_reason == "profile_unavailable"
    assert final.user_id == "learner-1"
    assert final.recommendations == ()
    assert final.candidate_snapshot is None
    assert final.generated_at is None

    missing_user = build_recommendation_final_v1(
        thread_id=THREAD_ID,
        request_id=REQUEST_ID,
        output=_unavailable_output(reason="missing_user_id", user_id=None),
        knowledge_graph=_knowledge_graph(),
        expected_user_id=None,
        expected_runtime_fingerprint=RUNTIME_FINGERPRINT,
    )
    assert missing_user.user_id is None
    assert missing_user.unavailable_reason == "missing_user_id"

    no_candidates = build_recommendation_final_v1(
        thread_id=THREAD_ID,
        request_id=REQUEST_ID,
        output=_unavailable_output(reason="no_eligible_candidates"),
        knowledge_graph=_knowledge_graph(),
        expected_user_id=USER_ID,
        expected_runtime_fingerprint=RUNTIME_FINGERPRINT,
    )
    assert no_candidates.unavailable_reason == "no_eligible_candidates"
    assert no_candidates.summary == (
        "Personalized recommendations are unavailable because no catalog "
        "candidate met the strict evidence and score thresholds."
    )
    assert no_candidates.recommendations == ()
    assert no_candidates.candidate_snapshot is None


def test_builder_rejects_non_explicit_mode_and_request_drift() -> None:
    with pytest.raises(
        RecommendationFinalContractError,
        match="recommendation_final_mode_mismatch",
    ):
        build_recommendation_final_v1(
            thread_id=THREAD_ID,
            request_id=REQUEST_ID,
            output=_available_output(mode="automatic_after_generation"),
            knowledge_graph=_knowledge_graph(),
            expected_user_id=USER_ID,
            expected_runtime_fingerprint=RUNTIME_FINGERPRINT,
        )

    with pytest.raises(
        RecommendationFinalContractError,
        match="recommendation_final_request_mismatch",
    ):
        build_recommendation_final_v1(
            thread_id=THREAD_ID,
            request_id="00000000-0000-4000-8000-000000000002",
            output=_available_output(),
            knowledge_graph=_knowledge_graph(),
            expected_user_id=USER_ID,
            expected_runtime_fingerprint=RUNTIME_FINGERPRINT,
        )


def test_builder_rejects_authenticated_user_and_runtime_drift() -> None:
    with pytest.raises(
        RecommendationFinalContractError,
        match="recommendation_final_user_mismatch",
    ):
        build_recommendation_final_v1(
            thread_id=THREAD_ID,
            request_id=REQUEST_ID,
            output=_available_output(),
            knowledge_graph=_knowledge_graph(),
            expected_user_id="other-learner",
            expected_runtime_fingerprint=RUNTIME_FINGERPRINT,
        )

    with pytest.raises(
        RecommendationFinalContractError,
        match="recommendation_final_runtime_mismatch",
    ):
        build_recommendation_final_v1(
            thread_id=THREAD_ID,
            request_id=REQUEST_ID,
            output=_available_output(),
            knowledge_graph=_knowledge_graph(),
            expected_user_id=USER_ID,
            expected_runtime_fingerprint="b" * 64,
        )


def test_builder_rejects_automatic_only_unavailable_reason() -> None:
    with pytest.raises(
        RecommendationFinalContractError,
        match="recommendation_final_unavailable_reason_mismatch",
    ):
        build_recommendation_final_v1(
            thread_id=THREAD_ID,
            request_id=REQUEST_ID,
            output=_unavailable_output(reason="generated_resources_unavailable"),
            knowledge_graph=_knowledge_graph(),
            expected_user_id=USER_ID,
            expected_runtime_fingerprint=RUNTIME_FINGERPRINT,
        )


@pytest.mark.parametrize(
    ("request_id", "expected_error"),
    [
        ("not-a-uuid", "request_id must be a UUID"),
        ("AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA", "canonical UUID text"),
    ],
)
def test_final_requires_canonical_stream_request_id(
    request_id: str,
    expected_error: str,
) -> None:
    with pytest.raises(ValidationError, match=expected_error):
        build_recommendation_final_v1(
            thread_id=THREAD_ID,
            request_id=request_id,
            output=_available_output(request_id=request_id),
            knowledge_graph=_knowledge_graph(),
            expected_user_id=USER_ID,
            expected_runtime_fingerprint=RUNTIME_FINGERPRINT,
        )


def test_builder_rejects_unknown_or_drifted_catalog_target() -> None:
    with pytest.raises(
        RecommendationFinalContractError,
        match="recommendation_final_unknown_candidate",
    ):
        build_recommendation_final_v1(
            thread_id=THREAD_ID,
            request_id=REQUEST_ID,
            output=_available_output(resource_id="python.loops.missing"),
            knowledge_graph=_knowledge_graph(),
            expected_user_id=USER_ID,
            expected_runtime_fingerprint=RUNTIME_FINGERPRINT,
        )

    with pytest.raises(
        RecommendationFinalContractError,
        match="recommendation_final_candidate_mismatch",
    ):
        build_recommendation_final_v1(
            thread_id=THREAD_ID,
            request_id=REQUEST_ID,
            output=_available_output(title="Drifted title"),
            knowledge_graph=_knowledge_graph(),
            expected_user_id=USER_ID,
            expected_runtime_fingerprint=RUNTIME_FINGERPRINT,
        )


def test_builder_rejects_private_evidence_ids_in_public_text() -> None:
    with pytest.raises(
        RecommendationFinalContractError,
        match="recommendation_final_private_evidence_exposure",
    ):
        build_recommendation_final_v1(
            thread_id=THREAD_ID,
            request_id=REQUEST_ID,
            output=_available_output(
                reason="Review history-python-loops-1 before continuing."
            ),
            knowledge_graph=_knowledge_graph(),
            expected_user_id=USER_ID,
            expected_runtime_fingerprint=RUNTIME_FINGERPRINT,
        )


def test_unrelated_wide_catalog_entry_does_not_break_selected_projection() -> None:
    final = build_recommendation_final_v1(
        thread_id=THREAD_ID,
        request_id=REQUEST_ID,
        output=_available_output(),
        knowledge_graph=_knowledge_graph(include_unrelated_wide_entry=True),
        expected_user_id=USER_ID,
        expected_runtime_fingerprint=RUNTIME_FINGERPRINT,
    )

    assert final.candidate_snapshot is not None
    assert final.candidate_snapshot.candidate_count == 3
    assert len(final.candidate_snapshot.targets) == 1


def test_persisted_final_rejects_extra_fields_coercion_and_tampering() -> None:
    payload = _build_available().model_dump(mode="json")

    with pytest.raises(ValidationError, match="extra_forbidden"):
        validate_recommendation_final_v1({**payload, "legacy": True})

    coercion = _build_available().model_dump(mode="json")
    coercion["recommendations"][0]["rank"] = "1"
    with pytest.raises(ValidationError, match="int_type"):
        validate_recommendation_final_v1(coercion)

    tampered_hash = _build_available().model_dump(mode="json")
    tampered_hash["summary"] = "A different summary."
    with pytest.raises(ValidationError, match="payload_hash"):
        validate_recommendation_final_v1(tampered_hash)

    tampered_id = _build_available().model_dump(mode="json")
    tampered_id["recommendation_final_id"] = "recommendation-final:v1:" + "0" * 64
    with pytest.raises(ValidationError, match="recommendation_final_id"):
        validate_recommendation_final_v1(tampered_id)

    tampered_snapshot = _build_available().model_dump(mode="json")
    tampered_snapshot["candidate_snapshot"]["source_data_version"] = "other"
    with pytest.raises(ValidationError, match="snapshot_id"):
        validate_recommendation_final_v1(tampered_snapshot)

    forced_instance_mutation = _build_available()
    object.__setattr__(forced_instance_mutation, "summary", "Forced mutation.")
    with pytest.raises(ValidationError, match="payload_hash"):
        validate_recommendation_final_v1(forced_instance_mutation)


def test_public_validator_rejects_python_tuple_normalization() -> None:
    top_level = _build_available().model_dump(mode="json")
    top_level["recommendations"] = tuple(top_level["recommendations"])
    with pytest.raises(TypeError, match="JSON array"):
        validate_recommendation_final_v1(top_level)

    nested = _build_available().model_dump(mode="json")
    nested["candidate_snapshot"]["targets"] = tuple(
        nested["candidate_snapshot"]["targets"]
    )
    with pytest.raises(TypeError, match="JSON array"):
        validate_recommendation_final_v1(nested)


def test_python_model_contract_rejects_list_coercion() -> None:
    payload = _build_available().model_dump(mode="python")
    payload["recommendations"] = list(payload["recommendations"])

    with pytest.raises(ValidationError, match="tuple_type"):
        RecommendationFinalV1.model_validate(payload, strict=True)
