"""Contract and topology tests for the curated KnowledgeGraphV1 artifact."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.learning_guidance.knowledge_graph import (
    KnowledgeGraphPathError,
    KnowledgeGraphV1,
    KnowledgeGraphValidationError,
    KnowledgeGraphYamlError,
    KnowledgeGraphYamlRootError,
    load_knowledge_graph,
)


def _resource(resource_id: str, resource_type: str = "review_doc") -> dict[str, object]:
    return {
        "resource_id": resource_id,
        "resource_type": resource_type,
        "title": f"Resource {resource_id}",
    }


def _topic(
    topic_id: str,
    *,
    prerequisites: list[str],
    resource_id: str,
) -> dict[str, object]:
    return {
        "topic_id": topic_id,
        "title": f"Topic {topic_id}",
        "difficulty": 0.4,
        "estimated_hours": 2.0,
        "prerequisite_topic_ids": prerequisites,
        "knowledge_points": [f"Knowledge point {topic_id}"],
        "resources": [_resource(resource_id)],
    }


def _payload() -> dict[str, object]:
    return {
        "schema_version": "knowledge_graph_v1",
        "data_version": "2026.07.15",
        "subjects": [
            {
                "subject_id": "math",
                "title": "Mathematics",
                "topics": [
                    _topic(
                        "math.foundations",
                        prerequisites=[],
                        resource_id="math.foundations.review",
                    ),
                    _topic(
                        "math.algebra",
                        prerequisites=["math.foundations"],
                        resource_id="math.algebra.review",
                    ),
                ],
            }
        ],
    }


def _write_yaml(path: Path, payload: object) -> Path:
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _validation_types(error: KnowledgeGraphValidationError) -> set[str]:
    return {error_type for _, error_type in error.validation_errors}


def test_knowledge_graph_loads_exact_inventory_and_stable_fingerprint(
    tmp_path: Path,
) -> None:
    first = load_knowledge_graph(_write_yaml(tmp_path / "first.yaml", _payload()))
    second = load_knowledge_graph(_write_yaml(tmp_path / "second.yaml", _payload()))

    assert first == second
    assert first.artifact_fingerprint == second.artifact_fingerprint
    assert len(first.artifact_fingerprint) == 64
    assert first.subject("math") is not None
    assert first.subject("Math") is None
    assert first.topic("math.algebra") is not None
    assert first.topic("algebra") is None

    changed_payload = _payload()
    changed_payload["data_version"] = "2026.07.16"
    changed = KnowledgeGraphV1.model_validate(changed_payload)
    assert first.artifact_fingerprint != changed.artifact_fingerprint


def test_knowledge_graph_loader_has_typed_path_yaml_and_root_failures(
    tmp_path: Path,
) -> None:
    with pytest.raises(KnowledgeGraphPathError):
        load_knowledge_graph(tmp_path / "missing.yaml")

    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("subjects: [", encoding="utf-8")
    with pytest.raises(KnowledgeGraphYamlError):
        load_knowledge_graph(malformed)

    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(KnowledgeGraphYamlRootError):
        load_knowledge_graph(scalar)


@pytest.mark.parametrize(
    ("field", "value", "expected_type"),
    (
        ("schema_version", "knowledge_graph_v2", "literal_error"),
        ("data_version", " 2026.07.15", "value_error"),
        ("unexpected", True, "extra_forbidden"),
    ),
)
def test_knowledge_graph_rejects_schema_drift_and_unnormalized_text(
    tmp_path: Path,
    field: str,
    value: object,
    expected_type: str,
) -> None:
    payload = _payload()
    payload[field] = value

    with pytest.raises(KnowledgeGraphValidationError) as error:
        load_knowledge_graph(_write_yaml(tmp_path / "invalid.yaml", payload))

    assert expected_type in _validation_types(error.value)


@pytest.mark.parametrize("invalid_value", ("0.4", float("nan"), float("inf")))
def test_knowledge_graph_rejects_numeric_coercion_and_non_finite_values(
    tmp_path: Path,
    invalid_value: object,
) -> None:
    payload = _payload()
    subjects = payload["subjects"]
    assert isinstance(subjects, list)
    topic = subjects[0]["topics"][0]
    topic["difficulty"] = invalid_value

    with pytest.raises(KnowledgeGraphValidationError):
        load_knowledge_graph(_write_yaml(tmp_path / "numeric.yaml", payload))


@pytest.mark.parametrize(
    ("mutation", "expected_type"),
    (
        ("duplicate_subject", "knowledge_graph_duplicate_subject_id"),
        ("duplicate_topic", "knowledge_graph_duplicate_topic_id"),
        ("duplicate_resource", "knowledge_graph_duplicate_resource_id"),
        ("self_prerequisite", "knowledge_graph_self_prerequisite"),
        ("unknown_prerequisite", "knowledge_graph_unknown_prerequisite"),
        ("cross_subject_prerequisite", "knowledge_graph_cross_subject_prerequisite"),
        ("cycle", "knowledge_graph_cycle"),
        ("non_topological_order", "knowledge_graph_topological_order"),
    ),
)
def test_knowledge_graph_rejects_invalid_global_inventory_and_topology(
    tmp_path: Path,
    mutation: str,
    expected_type: str,
) -> None:
    payload = _payload()
    subjects = payload["subjects"]
    assert isinstance(subjects, list)
    math = subjects[0]
    assert isinstance(math, dict)
    topics = math["topics"]
    assert isinstance(topics, list)

    if mutation == "duplicate_subject":
        subjects.append(deepcopy(math))
    elif mutation == "duplicate_topic":
        duplicate = deepcopy(topics[0])
        duplicate["resources"] = [_resource("math.duplicate.review")]
        topics.append(duplicate)
    elif mutation == "duplicate_resource":
        topics[1]["resources"] = deepcopy(topics[0]["resources"])
    elif mutation == "self_prerequisite":
        topics[0]["prerequisite_topic_ids"] = ["math.foundations"]
    elif mutation == "unknown_prerequisite":
        topics[0]["prerequisite_topic_ids"] = ["math.unknown"]
    elif mutation == "cross_subject_prerequisite":
        subjects.append(
            {
                "subject_id": "physics",
                "title": "Physics",
                "topics": [
                    _topic(
                        "physics.motion",
                        prerequisites=["math.foundations"],
                        resource_id="physics.motion.review",
                    )
                ],
            }
        )
    elif mutation == "cycle":
        topics[0]["prerequisite_topic_ids"] = ["math.algebra"]
    elif mutation == "non_topological_order":
        topics[0], topics[1] = topics[1], topics[0]
    else:
        raise AssertionError(f"unhandled test mutation: {mutation}")

    with pytest.raises(KnowledgeGraphValidationError) as error:
        load_knowledge_graph(_write_yaml(tmp_path / f"{mutation}.yaml", payload))

    assert expected_type in _validation_types(error.value)


def test_knowledge_graph_rejects_duplicate_lists_and_resource_aliases(
    tmp_path: Path,
) -> None:
    payload = _payload()
    subjects = payload["subjects"]
    assert isinstance(subjects, list)
    topic = subjects[0]["topics"][0]
    topic["knowledge_points"] = ["same", "same"]
    with pytest.raises(KnowledgeGraphValidationError):
        load_knowledge_graph(_write_yaml(tmp_path / "knowledge-points.yaml", payload))

    payload = _payload()
    subjects = payload["subjects"]
    assert isinstance(subjects, list)
    topic = subjects[0]["topics"][1]
    topic["prerequisite_topic_ids"] = ["math.foundations", "math.foundations"]
    with pytest.raises(KnowledgeGraphValidationError):
        load_knowledge_graph(_write_yaml(tmp_path / "prerequisites.yaml", payload))

    payload = _payload()
    subjects = payload["subjects"]
    assert isinstance(subjects, list)
    resource = subjects[0]["topics"][0]["resources"][0]
    resource["resource_type"] = "review"
    with pytest.raises(KnowledgeGraphValidationError):
        load_knowledge_graph(_write_yaml(tmp_path / "alias.yaml", payload))
