"""Focused validation for the private six-case smoke authoring draft."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re

import pytest
from pydantic import ValidationError
import yaml  # type: ignore[import-untyped]

from src.config.learning_guidance_config import load_learning_guidance_config
from src.evaluation.evidence_rollout.contracts import (
    EvidenceEvaluationDatasetContentV1,
    EvidenceEvaluationDatasetV1,
)
from src.graph.evidence import EvidenceCandidate
from src.learning_guidance.factory import resolve_knowledge_graph_path
from src.learning_guidance.knowledge_graph import load_knowledge_graph
from src.resource_contracts import RESOURCE_TYPE_ORDER


ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = ROOT / "config" / "evaluation" / "private_authoring"
DRAFT_PATH = PRIVATE_ROOT / "evidence_rollout_smoke_dataset.authoring.json"
PACKET_PATH = PRIVATE_ROOT / "evidence_rollout_smoke_reviewer_packet.yaml"


def _draft() -> EvidenceEvaluationDatasetContentV1:
    return EvidenceEvaluationDatasetContentV1.model_validate_json(
        DRAFT_PATH.read_bytes()
    )


def _packet() -> dict[str, object]:
    value = yaml.safe_load(PACKET_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return value


def _list(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


def test_smoke_draft_is_unsealed_and_cannot_authorize_activation() -> None:
    draft = _draft()
    packet = _packet()
    raw = json.loads(DRAFT_PATH.read_text(encoding="utf-8"))

    assert draft.dataset_id.endswith("_draft_v1")
    assert "dataset_fingerprint" not in raw
    with pytest.raises(ValidationError, match="dataset_fingerprint"):
        EvidenceEvaluationDatasetV1.model_validate_json(DRAFT_PATH.read_bytes())
    assert packet["packet_status"] == "draft_unapproved"
    assert packet["smoke_only"] is True
    assert packet["activation_eligible"] is False
    assert packet["sealing_allowed"] is False
    assert packet["human_semantic_review_status"] == "not_started"
    assert packet["first_domain_review_status"] == "pending"
    assert packet["second_independent_review_status"] == "pending"
    blockers = set(_list(packet["seal_blockers"]))
    assert {
        "schema_v1_missing_kg_identity_binding",
        "schema_v1_missing_initial_evidence_identity",
        "smoke_case_count_below_activation_gate_resolution",
        "human_domain_review_missing",
        "independent_review_missing",
    } == blockers


def test_smoke_inventory_covers_required_partitions_subjects_resources_and_routes() -> (
    None
):
    draft = _draft()
    packet = _packet()
    stats = _dict(packet["statistics"])
    simple = [
        case
        for case in draft.cases
        if len(case.subjects) == 1 and len(case.resource_types) == 1
    ]
    multi = [
        case
        for case in draft.cases
        if len(case.subjects) > 1 or len(case.resource_types) > 1
    ]
    route_counts: Counter[str] = Counter()
    for case in draft.cases:
        for target in case.targets:
            route_counts[
                {
                    ("parent_child",): "parent_child_only",
                    ("parent_child", "web"): "parent_child_and_web",
                    ("web",): "web_only",
                }[tuple(target.required_sources)]
            ] += 1

    assert len(draft.cases) == stats["case_count"] == 6
    assert len(simple) == stats["simple_case_count"] == 4
    assert len(multi) == stats["multi_case_count"] == 2
    assert sum(len(case.resource_types) > 1 for case in draft.cases) == 2
    assert sum(len(case.subjects) > 1 for case in draft.cases) == 1
    assert sum(case.initial_evidence_sufficient for case in draft.cases) == 3
    assert sum(not case.initial_evidence_sufficient for case in draft.cases) == 3
    assert {subject for case in draft.cases for subject in case.subjects} == {
        "big_data",
        "computer",
        "machine_learning",
        "math",
        "python",
    }
    assert {
        target.resource_type for case in draft.cases for target in case.targets
    } == set(RESOURCE_TYPE_ORDER)
    assert route_counts == stats["target_route_counts"]
    assert sum(len(case.targets) for case in draft.cases) == stats["target_count"] == 8
    assert (
        sum(len(case.requirements) for case in draft.cases)
        == stats["requirement_count"]
        == 12
    )
    expected_weight = stats["requirement_weight_total"]
    assert isinstance(expected_weight, (int, float)) and not isinstance(
        expected_weight, bool
    )
    assert sum(
        requirement.weight for case in draft.cases for requirement in case.requirements
    ) == pytest.approx(float(expected_weight))


def test_sidecar_kg_bindings_are_exact_but_remain_non_authoritative() -> None:
    draft = _draft()
    packet = _packet()
    guidance = load_learning_guidance_config(ROOT / "config" / "learning_guidance.yaml")
    kg_path = resolve_knowledge_graph_path(config=guidance, project_root=ROOT)
    kg = load_knowledge_graph(kg_path)
    kg_packet = _dict(packet["knowledge_graph"])
    assert kg_packet["artifact_fingerprint"] == kg.artifact_fingerprint
    assert kg_packet["data_version"] == kg.data_version
    assert kg_packet["binding_status"] == "sidecar_only_blocked"
    route_contract = _dict(packet["route_contract"])
    production_sources = set(
        EvidenceCandidate.model_json_schema()["properties"]["source_type"]["enum"]
    )
    assert {
        route_contract["parent_child_production_source_type"],
        route_contract["web_production_source_type"],
    } == production_sources

    case_by_id = {case.case_id: case for case in draft.cases}
    packet_cases = _list(packet["case_bindings"])
    assert [_dict(item)["case_id"] for item in packet_cases] == list(case_by_id)
    for raw_case in packet_cases:
        case_packet = _dict(raw_case)
        case = case_by_id[str(case_packet["case_id"])]
        target_by_id = {target.target_id: target for target in case.targets}
        for raw_target in _list(case_packet["targets"]):
            binding = _dict(raw_target)
            target = target_by_id[str(binding["target_id"])]
            topic = kg.topic(str(binding["topic_id"]))
            assert topic is not None
            subject = kg.subject(str(binding["subject_id"]))
            assert subject is not None
            assert topic.topic_id in {item.topic_id for item in subject.topics}
            assert target.subject == subject.subject_id
            assert binding["expected_sources"] == target.required_sources
            catalog_ids = [str(item) for item in _list(binding["catalog_resource_ids"])]
            topic_inventory = [resource.resource_id for resource in topic.resources]
            assert catalog_ids
            assert catalog_ids == [
                item for item in topic_inventory if item in set(catalog_ids)
            ]


def test_schema_and_business_negatives_reject_duplicate_gap_and_missing_partition() -> (
    None
):
    original = json.loads(DRAFT_PATH.read_text(encoding="utf-8"))

    duplicate = deepcopy(original)
    duplicate["cases"][1]["case_id"] = duplicate["cases"][0]["case_id"]
    with pytest.raises(ValidationError, match="cases must not repeat"):
        EvidenceEvaluationDatasetContentV1.model_validate(duplicate)

    uncovered = deepcopy(original)
    multi_case = uncovered["cases"][4]
    multi_case["requirements"] = [multi_case["requirements"][0]]
    with pytest.raises(ValidationError, match="every target must have"):
        EvidenceEvaluationDatasetContentV1.model_validate(uncovered)

    no_multi = deepcopy(original)
    no_multi["cases"] = no_multi["cases"][:4]
    with pytest.raises(ValidationError, match="simple, multi"):
        EvidenceEvaluationDatasetContentV1.model_validate(no_multi)


def test_private_smoke_artifacts_are_utf8_secret_free_and_report_isolated() -> None:
    texts = [
        path.read_bytes().decode("utf-8", errors="strict")
        for path in (DRAFT_PATH, PACKET_PATH)
    ]
    for text in texts:
        assert all(ord(character) < 128 for character in text)
        assert not any(
            marker in text for marker in ("\ufffd", "\u9225", "\u951f", "\u7039")
        )
        for pattern in (
            r"sk-[A-Za-z0-9_-]{12,}",
            r"Bearer\s+[A-Za-z0-9._-]{12,}",
            r"postgres(?:ql)?://\S+",
            r"(?i)(?:api[_-]?key|token|secret)\s*[:=]\s*[A-Za-z0-9_-]{12,}",
        ):
            assert re.search(pattern, text) is None

    draft = _draft()
    private_content = {case.query for case in draft.cases} | {
        requirement.criterion
        for case in draft.cases
        for requirement in case.requirements
    }
    public_report_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "src" / "evaluation" / "evidence_rollout" / "report.py",
            ROOT / "docs" / "reports" / "evidence_rollout_evaluation_status.md",
            ROOT / "config" / "evaluation" / "evidence_rollout.yaml",
        )
    )
    assert all(value not in public_report_text for value in private_content)
