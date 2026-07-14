"""Strict policy tests for production learning-guidance composition."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable

import pytest
import yaml
from pydantic import ValidationError

from src.config._rag_config import RagConfigValidationError
from src.config.learning_guidance_config import (
    LearnerPathPolicyV1,
    LearningGuidanceConfigV1,
    RecommendationWeightsV1,
    ResourcePreferenceBindingV1,
    ResourceRecommendationPolicyV1,
    load_learning_guidance_config,
)
from src.resource_contracts import RESOURCE_TYPE_ORDER


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _payload() -> dict[str, object]:
    return yaml.safe_load(
        (PROJECT_ROOT / "config" / "learning_guidance.yaml").read_text(encoding="utf-8")
    )


def _write_yaml(path: Path, payload: object) -> Path:
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_checked_in_learning_guidance_policy_is_strict_and_complete() -> None:
    config = load_learning_guidance_config(
        PROJECT_ROOT / "config" / "learning_guidance.yaml"
    )

    assert config.schema_version == "learning_guidance_config_v1"
    assert config.adapter_version == "learning_guidance_adapters_v1"
    assert config.knowledge_graph_path == Path("data/knowledge_graph.yaml")
    assert (
        tuple(
            binding.resource_type
            for binding in config.recommendation_policy.resource_preferences
        )
        == RESOURCE_TYPE_ORDER
    )
    assert len(config.policy_fingerprint) == 64


def test_learning_guidance_policy_has_no_defaults_and_is_frozen() -> None:
    model_types = (
        LearningGuidanceConfigV1,
        LearnerPathPolicyV1,
        RecommendationWeightsV1,
        ResourcePreferenceBindingV1,
        ResourceRecommendationPolicyV1,
    )
    for model_type in model_types:
        assert model_type.model_config["extra"] == "forbid"
        assert model_type.model_config["strict"] is True
        assert model_type.model_config["frozen"] is True
        assert all(field.is_required() for field in model_type.model_fields.values())

    config = LearningGuidanceConfigV1.model_validate(_payload())
    with pytest.raises(ValidationError):
        config.history_limit = 10


@pytest.mark.parametrize(
    ("mutate", "location", "error_type"),
    (
        (
            lambda payload: payload.__setitem__("history_limit", "200"),
            "history_limit",
            "int_type",
        ),
        (
            lambda payload: payload.__setitem__("unexpected", True),
            "unexpected",
            "extra_forbidden",
        ),
    ),
)
def test_learning_guidance_policy_rejects_coercion_and_schema_drift(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], None],
    location: str,
    error_type: str,
) -> None:
    payload = _payload()
    mutate(payload)

    with pytest.raises(RagConfigValidationError) as error:
        load_learning_guidance_config(_write_yaml(tmp_path / "policy.yaml", payload))

    assert (location, error_type) in error.value.validation_errors


def test_learning_guidance_policy_rejects_invalid_business_constraints(
    tmp_path: Path,
) -> None:
    payload = _payload()
    recommendation = payload["recommendation_policy"]
    assert isinstance(recommendation, dict)
    weights = recommendation["weights"]
    assert isinstance(weights, dict)
    weights["goal"] = 0.5
    with pytest.raises(RagConfigValidationError):
        load_learning_guidance_config(_write_yaml(tmp_path / "weights.yaml", payload))

    payload = _payload()
    path_policy = payload["path_policy"]
    assert isinstance(path_policy, dict)
    path_policy["reinforce_level"] = path_policy["mastery_level"]
    with pytest.raises(RagConfigValidationError):
        load_learning_guidance_config(
            _write_yaml(tmp_path / "thresholds.yaml", payload)
        )

    payload = _payload()
    recommendation = payload["recommendation_policy"]
    assert isinstance(recommendation, dict)
    bindings = recommendation["resource_preferences"]
    assert isinstance(bindings, list)
    bindings[0], bindings[1] = bindings[1], bindings[0]
    with pytest.raises(RagConfigValidationError):
        load_learning_guidance_config(_write_yaml(tmp_path / "order.yaml", payload))


def test_learning_guidance_policy_fingerprint_is_stable_and_sensitive() -> None:
    first = LearningGuidanceConfigV1.model_validate(_payload())
    second = LearningGuidanceConfigV1.model_validate(deepcopy(_payload()))
    changed_payload = _payload()
    changed_payload["history_limit"] = 201
    changed = LearningGuidanceConfigV1.model_validate(changed_payload)

    assert first.policy_fingerprint == second.policy_fingerprint
    assert first.policy_fingerprint != changed.policy_fingerprint
