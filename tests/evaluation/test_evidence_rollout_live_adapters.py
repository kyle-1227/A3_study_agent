from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.evaluation.evidence_rollout.contracts import (
    EvidenceEvaluationCaseSpecV2,
    EvidenceEvaluationDatasetContentV2,
    EvidenceEvaluationRuntimeBindingV2,
    EvidenceInitialEvidenceIdentityV2,
    EvidenceRequirementGoldV2,
    EvidenceResourceSubjectTargetV2,
    EvidenceVariantDefinitionV2,
    canonical_sha256,
    dataset_case_bindings,
    seal_evidence_evaluation_dataset,
)
from src.evaluation.evidence_rollout.live_adapters import (
    EvidenceVariantServedComposition,
    _compile_no_planning_state,
    _observation,
    build_served_evidence_live_adapters,
)
from src.evaluation.evidence_rollout.runner import LiveEvidenceVariantExecutor
from src.evaluation.evidence_rollout.contracts import (
    load_evidence_rollout_execution_config,
)
from src.rag.parent_child.evidence_evaluation import Variant
from tests.test_evidence_orchestration_graph import _runtime


def _target(*, target_id: str, resource_type: str) -> EvidenceResourceSubjectTargetV2:
    payload = {
        "schema_version": "evidence_resource_subject_target_v2",
        "target_id": target_id,
        "subject": "math",
        "resource_type": resource_type,
        "topic_id": "functions",
        "catalog_resource_ids": ["functions_quiz"],
        "required_sources": ["parent_child"],
    }
    return EvidenceResourceSubjectTargetV2.model_validate(
        {**payload, "target_fingerprint": canonical_sha256(payload)}
    )


def _initial(*, fixture_id: str, state: str) -> EvidenceInitialEvidenceIdentityV2:
    payload = {
        "schema_version": "evidence_initial_evidence_identity_v2",
        "state": state,
        "fixture_id": fixture_id,
        "source_inventory": ["parent_child"],
    }
    return EvidenceInitialEvidenceIdentityV2.model_validate(
        {**payload, "fixture_fingerprint": canonical_sha256(payload)}
    )


def _case(
    *,
    case_id: str,
    targets: list[EvidenceResourceSubjectTargetV2],
    initially_sufficient: bool,
) -> EvidenceEvaluationCaseSpecV2:
    requirements = [
        EvidenceRequirementGoldV2(
            schema_version="evidence_requirement_gold_v2",
            requirement_id=f"gold_{target.target_id}",
            target_id=target.target_id,
            criterion=f"Support {target.target_id} with exact evidence.",
            weight=1.0,
        )
        for target in targets
    ]
    payload = {
        "schema_version": "evidence_evaluation_case_spec_v2",
        "case_id": case_id,
        "query": "Create exact learning resources about mathematical functions.",
        "subjects": ["math"],
        "resource_types": [target.resource_type for target in targets],
        "initial_evidence": _initial(
            fixture_id=f"{case_id}_fixture",
            state="sufficient" if initially_sufficient else "insufficient",
        ).model_dump(mode="json"),
        "targets": [target.model_dump(mode="json") for target in targets],
        "requirements": [item.model_dump(mode="json") for item in requirements],
    }
    return EvidenceEvaluationCaseSpecV2.model_validate(
        {**payload, "case_fingerprint": canonical_sha256(payload)}
    )


def _dataset():
    runtime = _runtime()
    simple = _case(
        case_id="live_simple",
        targets=[_target(target_id="simple_quiz", resource_type="quiz")],
        initially_sufficient=True,
    )
    multi = _case(
        case_id="live_multi",
        targets=[
            _target(target_id="multi_quiz", resource_type="quiz"),
            _target(target_id="multi_review", resource_type="review_doc"),
        ],
        initially_sufficient=False,
    )
    content = EvidenceEvaluationDatasetContentV2(
        schema_version="evidence_evaluation_dataset_v2",
        dataset_id="live_adapter_test_v2",
        knowledge_graph_data_version=(
            runtime.learning_guidance.knowledge_graph.data_version
        ),
        knowledge_graph_artifact_fingerprint=(
            runtime.learning_guidance.knowledge_graph.artifact_fingerprint
        ),
        cases=[simple, multi],
    )
    return runtime, seal_evidence_evaluation_dataset(content)


def test_builds_complete_ordered_v2_adapter_matrix() -> None:
    runtime, dataset = _dataset()

    adapters = build_served_evidence_live_adapters(
        dataset=dataset,
        knowledge_graph=runtime.learning_guidance.knowledge_graph,
        runtime=runtime,
        generation_manifest_fingerprint="a" * 64,
    )

    assert [adapter.identity.variant for adapter in adapters] == [
        "P0",
        "PG",
        "PR",
        "PGR",
    ]
    assert [
        (
            adapter.identity.resource_planning_enabled,
            adapter.identity.bounded_repair_enabled,
        )
        for adapter in adapters
    ] == [(False, False), (True, False), (False, True), (True, True)]
    assert all(
        adapter.identity.declared_cases == dataset_case_bindings(dataset)
        for adapter in adapters
    )
    assert len({adapter.identity.adapter_fingerprint for adapter in adapters}) == 4
    executor = LiveEvidenceVariantExecutor(
        dataset=dataset,
        knowledge_graph=runtime.learning_guidance.knowledge_graph,
        execution_config=load_evidence_rollout_execution_config(
            Path(__file__).resolve().parents[2]
            / "config"
            / "evaluation"
            / "evidence_rollout.yaml"
        ),
        adapters=adapters,
    )
    assert executor.missing_variants == ()


@pytest.mark.parametrize(
    ("variant", "planning", "repair", "expected_planner", "expected_repair"),
    [
        ("P0", False, False, 0, 0),
        ("PG", True, False, 1, 0),
        ("PR", False, True, 0, 1),
        ("PGR", True, True, 1, 1),
    ],
)
def test_composition_executes_exact_factorial_nodes(
    monkeypatch,
    variant: Variant,
    planning: bool,
    repair: bool,
    expected_planner: int,
    expected_repair: int,
) -> None:
    import src.evaluation.evidence_rollout.live_adapters as module

    runtime, dataset = _dataset()
    case = dataset.cases[0]
    calls = {"path": 0, "planner": 0, "judge": 0, "repair": 0}

    async def path_node(_state):
        calls["path"] += 1
        return {}

    async def planner_node(_state):
        calls["planner"] += 1
        return _compile_no_planning_state(case=case, runtime=runtime)

    async def source_node(_state):
        return {}

    async def judge_node(_state):
        calls["judge"] += 1
        if calls["judge"] == 1:
            return {
                "evidence_orchestration_route": "repair",
                "resource_evidence_readiness": [
                    {
                        "resource_type": "quiz",
                        "readiness_state": "blocked_insufficient_evidence",
                    }
                ],
            }
        return {
            "evidence_orchestration_route": "terminal",
            "evidence_terminal_status": "blocked_insufficient_evidence",
            "evidence_terminal_reason_code": "supplement_round_budget_exhausted",
        }

    def repair_node(_state):
        calls["repair"] += 1
        return {"evidence_current_round": 1}

    async def hydrate_node(_state):
        return {"evidence_hydration_count": 1}

    monkeypatch.setattr(
        module,
        "make_learner_path_planner_node",
        lambda _runtime: path_node,
    )
    monkeypatch.setattr(
        module,
        "make_resource_evidence_planner_node",
        lambda _runtime: planner_node,
    )
    monkeypatch.setattr(
        module,
        "make_retrieval_round_router_node",
        lambda _runtime: lambda _state: {},
    )
    monkeypatch.setattr(
        module,
        "make_local_rag_search_batch_node",
        lambda _runtime: source_node,
    )
    monkeypatch.setattr(
        module,
        "make_web_research_search_batch_node",
        lambda _runtime: source_node,
    )
    monkeypatch.setattr(
        module,
        "make_retrieval_round_merge_node",
        lambda _runtime: lambda _state: {},
    )
    monkeypatch.setattr(
        module,
        "make_requirement_evidence_judge_node",
        lambda _runtime: judge_node,
    )
    monkeypatch.setattr(
        module,
        "make_evidence_repair_planner_node",
        lambda _runtime: repair_node,
    )
    monkeypatch.setattr(
        module,
        "make_terminal_parent_hydration_node",
        lambda _runtime: hydrate_node,
    )
    monkeypatch.setattr(
        module,
        "make_resource_evidence_assignment_node",
        lambda _runtime: lambda _state: {},
    )

    composition = EvidenceVariantServedComposition(
        runtime=runtime,
        variant=variant,
        resource_planning_enabled=planning,
        bounded_repair_enabled=repair,
    )
    final_state = asyncio.run(composition.execute(case))

    assert calls["path"] == expected_planner
    assert calls["planner"] == expected_planner
    assert calls["repair"] == expected_repair
    assert calls["judge"] == (2 if repair else 1)
    assert final_state["evidence_orchestration_route"] == "terminal"
    if not repair:
        assert final_state["evidence_terminal_reason_code"] == (
            "repair_disabled_by_variant"
        )


def test_variant_factors_are_not_aliasable() -> None:
    runtime, _dataset_value = _dataset()

    with pytest.raises(ValueError, match="canonical semantics"):
        EvidenceVariantServedComposition(
            runtime=runtime,
            variant="P0",
            resource_planning_enabled=True,
            bounded_repair_enabled=False,
        )


def test_no_planning_compiler_preserves_authored_topic_binding() -> None:
    runtime, dataset = _dataset()

    state = _compile_no_planning_state(case=dataset.cases[0], runtime=runtime)

    assert state["evidence_requested_subjects"] == ["math"]
    assert state["evidence_requested_resource_types"] == ["quiz"]
    assert {item["topic_id"] for item in state["evidence_requirements"]} == {
        "functions"
    }
    assert state["evidence_all_tasks"]


def test_observation_projects_only_strict_content_free_terminal_state() -> None:
    runtime, dataset = _dataset()
    case = dataset.cases[0]
    composition = EvidenceVariantServedComposition(
        runtime=runtime,
        variant="P0",
        resource_planning_enabled=False,
        bounded_repair_enabled=False,
    )
    state = {
        **_compile_no_planning_state(case=case, runtime=runtime),
        "evidence_source_outcomes": [
            {"source_type": "local_rag"},
            {"source_type": "web"},
        ],
        "evidence_ledger": [],
        "resource_evidence_readiness": [
            {
                "resource_type": "quiz",
                "readiness_state": "blocked_insufficient_evidence",
            }
        ],
        "resource_evidence_assignments": [],
        "evidence_terminal_status": "blocked_insufficient_evidence",
        "evidence_terminal_reason_code": "repair_disabled_by_variant",
    }
    definition = EvidenceVariantDefinitionV2(
        variant="P0",
        resource_planning_enabled=False,
        bounded_repair_enabled=False,
    )
    binding = EvidenceEvaluationRuntimeBindingV2(
        schema_version="evidence_evaluation_runtime_binding_v2",
        run_id="live_adapter_test_run",
        execution_mode="live",
        dataset_id=dataset.dataset_id,
        dataset_fingerprint=dataset.dataset_fingerprint,
        knowledge_graph_data_version=dataset.knowledge_graph_data_version,
        knowledge_graph_artifact_fingerprint=(
            dataset.knowledge_graph_artifact_fingerprint
        ),
        case_bindings=dataset_case_bindings(dataset),
        execution_config_fingerprint="1" * 64,
        benchmark_config_fingerprint="2" * 64,
        rollout_config_fingerprint="3" * 64,
        runtime_fingerprint=runtime.orchestration_fingerprint,
        generation_id=runtime.parent_child.generation_id,
        generation_manifest_fingerprint="4" * 64,
        executor_fingerprint="5" * 64,
    )

    observation = _observation(
        case=case,
        binding=binding,
        definition=definition,
        composition=composition,
        state=state,
        latency_ms=1.0,
    )

    assert observation.case_binding == dataset_case_bindings(dataset)[0]
    assert observation.required_gap_count == 1
    assert observation.premature_stop is True
    assert observation.over_search is True
    assert observation.repeated_query_count == 1
    assert observation.source_route_true_positive == 1
    assert observation.source_route_false_positive == 1
    assert observation.output_fingerprint
