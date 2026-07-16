from __future__ import annotations

from pathlib import Path

import pytest

from src.config.rag_rollout_config import load_rag_rollout_config
from src.rag.rollout_control import (
    RolloutControlError,
    RolloutObservation,
    RolloutStage,
    evaluate_rollout_stage,
    route_generation,
    rollout_stage_from_config,
)


def _stage(*, enabled: bool, fraction: float) -> RolloutStage:
    return RolloutStage(
        stage_id="single-subject",
        activation_enabled=enabled,
        candidate_fraction=fraction,
        eligible_subjects=("math",),
        minimum_evaluable_requests=100,
        minimum_observation_seconds=3600,
        maximum_candidate_error_rate=0.01,
        maximum_recall_regression=0.02,
        maximum_p95_latency_ratio=1.25,
        maximum_context_token_ratio=1.35,
    )


def _observation(**overrides: object) -> RolloutObservation:
    values: dict[str, object] = {
        "stage_id": "single-subject",
        "evaluable_requests": 100,
        "observation_seconds": 3600,
        "candidate_error_rate": 0.0,
        "worst_subject_recall_delta": 0.0,
        "p95_latency_ratio": 1.0,
        "context_token_ratio": 1.0,
        "orphan_child_count": 0,
        "parent_hydration_failure_count": 0,
        "generation_mismatch_count": 0,
    }
    values.update(overrides)
    return RolloutObservation.model_validate(values)


def test_disabled_stage_always_routes_primary() -> None:
    route = route_generation(
        request_id="request-1",
        subject="math",
        primary_generation_id="primary",
        candidate_generation_id="candidate",
        stage=_stage(enabled=False, fraction=1.0),
    )
    assert route.route_kind == "primary"
    assert route.generation_id == "primary"


def test_candidate_route_requires_explicit_generation() -> None:
    with pytest.raises(RolloutControlError, match="candidate generation"):
        route_generation(
            request_id="request-1",
            subject="math",
            primary_generation_id="primary",
            candidate_generation_id=None,
            stage=_stage(enabled=True, fraction=1.0),
        )


def test_unknown_subject_does_not_receive_candidate() -> None:
    route = route_generation(
        request_id="request-1",
        subject="unknown",
        primary_generation_id="primary",
        candidate_generation_id="candidate",
        stage=_stage(enabled=True, fraction=1.0),
    )
    assert route.route_kind == "primary"


def test_insufficient_observation_holds() -> None:
    decision = evaluate_rollout_stage(
        stage=_stage(enabled=True, fraction=0.05),
        observation=_observation(evaluable_requests=99),
    )
    assert decision.action == "hold"
    assert decision.reason_codes == ("insufficient_evaluable_requests",)


def test_integrity_failure_requires_explicit_rollback() -> None:
    decision = evaluate_rollout_stage(
        stage=_stage(enabled=True, fraction=0.05),
        observation=_observation(orphan_child_count=1),
    )
    assert decision.action == "rollback_required"
    assert decision.reason_codes == ("orphan_children_detected",)


def test_healthy_stage_continues() -> None:
    decision = evaluate_rollout_stage(
        stage=_stage(enabled=True, fraction=0.05),
        observation=_observation(),
    )
    assert decision.action == "continue"
    assert decision.reason_codes == ()


def test_strict_rollout_config_maps_without_threshold_defaults() -> None:
    config = load_rag_rollout_config(Path("config/rag/rollout.yaml"))

    stage = rollout_stage_from_config(
        rollout_config=config,
        stage_id="single_subject",
    )

    assert stage.activation_enabled is True
    assert stage.candidate_fraction == 0.05
    assert stage.minimum_evaluable_requests == 200
    assert stage.maximum_context_token_ratio == 1.35
