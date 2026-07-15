"""Production node compositions for the strict P0/PG/PR/PGR live matrix."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import time
from typing import Literal

from langchain_core.messages import HumanMessage

from src.config.evidence_orchestration_contracts import (
    EvidenceLedgerEntry,
    EvidenceRequirement,
    EvidenceRequirementDraft,
    EvidenceRequirementDraftBatch,
    EvidenceSourceType,
    ResourceEvidenceAssignment,
    RetrievalTask,
    build_retrieval_task,
    compile_evidence_requirement_batch,
    validate_requirement_inventory,
    validate_retrieval_tasks,
)
from src.evaluation.evidence_rollout.contracts import (
    EvidenceCaseBindingIdentityV2,
    EvidenceEvaluationCaseSpecV2,
    EvidenceEvaluationDatasetV2,
    EvidenceEvaluationRuntimeBindingV2,
    EvidenceLiveAdapterIdentityV2,
    EvidenceVariantAttemptV2,
    EvidenceVariantDefinitionV2,
    EvidenceVariantObservationV2,
    case_binding_identity,
    canonical_sha256,
    dataset_case_bindings,
    model_fingerprint,
    query_fingerprint,
)
from src.graph.evidence_orchestration import (
    EvidenceOrchestrationRuntime,
    make_evidence_repair_planner_node,
    make_local_rag_search_batch_node,
    make_requirement_evidence_judge_node,
    make_resource_evidence_assignment_node,
    make_resource_evidence_planner_node,
    make_retrieval_round_merge_node,
    make_retrieval_round_router_node,
    make_terminal_parent_hydration_node,
    make_web_research_search_batch_node,
    route_after_requirement_evidence_judge,
)
from src.graph.learning_guidance import make_learner_path_planner_node
from src.graph.state import LearningState, initial_request_reset_transient_state
from src.learning_guidance.knowledge_graph import KnowledgeGraphV1
from src.rag.parent_child.evidence_evaluation import Variant
from src.resource_contracts import ResourceType


_VARIANT_FACTORS: dict[Variant, tuple[bool, bool]] = {
    "P0": (False, False),
    "PG": (True, False),
    "PR": (False, True),
    "PGR": (True, True),
}
_SOURCE_ROUTE: dict[EvidenceSourceType, Literal["parent_child", "web"]] = {
    "local_rag": "parent_child",
    "web": "web",
}
_ADAPTER_ALGORITHM = "served_evidence_variant_composition_v1"


class LiveEvidenceAdapterError(RuntimeError):
    """One live adapter violated a strict production composition invariant."""

    def __init__(self, *, code: str, reason: str) -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"{code}: {reason}")


def _target_by_pair(
    case: EvidenceEvaluationCaseSpecV2,
) -> dict[tuple[ResourceType, str], object]:
    return {(target.resource_type, target.subject): target for target in case.targets}


def _source_policy_sources(source_policy: str) -> tuple[EvidenceSourceType, ...]:
    if source_policy in {"local_only", "local_then_web_on_gap"}:
        return ("local_rag",)
    if source_policy == "web_only":
        return ("web",)
    if source_policy == "local_and_web":
        return ("local_rag", "web")
    raise LiveEvidenceAdapterError(
        code="unsupported_evidence_source_policy",
        reason="resource profile contains an unknown source policy",
    )


def _task_priority(
    requirement: EvidenceRequirement,
    runtime: EvidenceOrchestrationRuntime,
) -> Literal["high", "medium", "low"]:
    if requirement.criticality == "required":
        return runtime.policy.required_task_priority
    return runtime.policy.supporting_task_priority


def _compile_no_planning_state(
    *,
    case: EvidenceEvaluationCaseSpecV2,
    runtime: EvidenceOrchestrationRuntime,
) -> dict[str, object]:
    """Compile profile/KG-bound requirements without an LLM planning call."""

    drafts: list[EvidenceRequirementDraft] = []
    for target in case.targets:
        profile = runtime.profiles.profile_for(target.resource_type)
        for need in profile.needs:
            drafts.append(
                EvidenceRequirementDraft(
                    resource_type=target.resource_type,
                    subject=target.subject,
                    topic_id=target.topic_id,
                    profile_need_id=need.need_id,
                    evidence_kind=need.evidence_kind,
                    scope=need.scope,
                    criticality=need.criticality,
                    source_policy=need.source_policy,
                    acceptance_criteria=need.acceptance_criteria,
                    query_intent=case.query,
                )
            )
    batch = EvidenceRequirementDraftBatch(
        schema_version="evidence_requirement_draft_batch_v1",
        requirements=drafts,
    )
    requirements = compile_evidence_requirement_batch(batch)
    validate_requirement_inventory(
        requested_resource_types=tuple(case.resource_types),
        requested_subjects=tuple(case.subjects),
        canonical_subjects=set(runtime.parent_child.available_subjects),
        requirements=requirements,
        profiles=runtime.profiles,
        config=runtime.policy,
    )

    ordered = tuple(
        requirement
        for _index, requirement in sorted(
            enumerate(requirements),
            key=lambda pair: (
                0 if pair[1].criticality == "required" else 1,
                pair[0],
            ),
        )
    )
    required_actions = sum(
        len(_source_policy_sources(item.source_policy))
        for item in ordered
        if item.criticality == "required"
    )
    if required_actions > runtime.policy.max_total_search_tasks:
        raise LiveEvidenceAdapterError(
            code="required_initial_search_budget_exceeded",
            reason="required no-planning actions exceed the explicit task budget",
        )
    tasks: list[RetrievalTask] = []
    for requirement in ordered:
        for source_type in _source_policy_sources(requirement.source_policy):
            if len(tasks) >= runtime.policy.max_search_tasks_per_round:
                break
            tasks.append(
                build_retrieval_task(
                    requirement=requirement,
                    source_type=source_type,
                    query=requirement.query_intent,
                    purpose=requirement.acceptance_criteria,
                    priority=_task_priority(requirement, runtime),
                    round_index=0,
                    result_limit=runtime.policy.max_results_per_task,
                )
            )
        if len(tasks) >= runtime.policy.max_search_tasks_per_round:
            break
    if not tasks:
        raise LiveEvidenceAdapterError(
            code="empty_initial_task_plan",
            reason="no-planning composition produced no retrieval task",
        )
    validate_retrieval_tasks(
        tasks=tasks,
        requirements=requirements,
        config=runtime.policy,
        round_index=0,
        existing_total_search_tasks=0,
        prior_retrieval_signatures=set(),
        local_then_web_gap_requirement_ids=set(),
    )
    return {
        "evidence_orchestration_fingerprint": runtime.orchestration_fingerprint,
        "evidence_requested_resource_types": list(case.resource_types),
        "evidence_requested_subjects": list(case.subjects),
        "evidence_requirements": [
            item.model_dump(mode="json") for item in requirements
        ],
        "evidence_current_round": 0,
        "evidence_current_tasks": [item.model_dump(mode="json") for item in tasks],
        "evidence_all_tasks": [item.model_dump(mode="json") for item in tasks],
        "evidence_retrieval_signatures": [item.retrieval_signature for item in tasks],
        "evidence_candidate_records": [],
        "evidence_ledger": [],
        "evidence_coverage": {},
        "evidence_previous_coverage": {},
        "evidence_source_outcomes": [],
        "evidence_parent_child_rounds": [],
        "evidence_repair_plans": [],
        "evidence_consecutive_no_progress_rounds": 0,
        "evidence_orchestration_route": "retrieve",
        "evidence_terminal_status": "",
        "evidence_terminal_reason_code": "",
        "evidence_hydration_count": 0,
        "resource_evidence_readiness": [],
        "resource_evidence_assignments": [],
        "blocked_resource_types": [],
        "ready_resource_types": [],
    }


def _initial_case_state(
    *,
    case: EvidenceEvaluationCaseSpecV2,
    variant: Variant,
) -> LearningState:
    state: dict[str, object] = initial_request_reset_transient_state()
    state.update(
        {
            "messages": [HumanMessage(content=case.query)],
            "request_id": f"evaluation:{case.case_id}:{variant}",
            "session_id": f"evaluation:{case.case_id}",
            "thread_id": f"evaluation:{case.case_id}",
            "user_id": f"evaluation:{case.case_id}",
            "response_mode": "resource",
            "subject": case.subjects[0] if len(case.subjects) == 1 else "",
            "primary_subject": case.subjects[0] if len(case.subjects) == 1 else "",
            "requested_resource_type": case.resource_types[0],
            "requested_resource_types": list(case.resource_types),
            "retrieval_plan": [
                {
                    "subject": subject,
                    "role": f"evaluation_{subject}",
                    "local_retrieval_query": case.query,
                    "web_research_query": case.query,
                    "purpose": "Evaluate evidence support for the authored request.",
                    "priority": 1.0,
                    "_parent_child_priority_explicit": True,
                }
                for subject in case.subjects
            ],
            "learning_goal": case.query,
        }
    )
    return state  # type: ignore[return-value]


def _merge_state(state: LearningState, update: Mapping[str, object]) -> LearningState:
    merged = dict(state)
    merged.update(update)
    return merged  # type: ignore[return-value]


def _validate_planned_target_topics(
    *,
    state: LearningState,
    case: EvidenceEvaluationCaseSpecV2,
) -> None:
    targets = _target_by_pair(case)
    requirements = tuple(
        EvidenceRequirement.model_validate(item)
        for item in (state.get("evidence_requirements") or [])
    )
    for requirement in requirements:
        target = targets.get((requirement.resource_type, requirement.subject))
        if target is None or getattr(target, "topic_id", None) != requirement.topic_id:
            raise LiveEvidenceAdapterError(
                code="planned_target_binding_mismatch",
                reason="planned requirement differs from the authored target topic",
            )


def _force_no_repair_terminal(state: LearningState) -> LearningState:
    if route_after_requirement_evidence_judge(state) != "repair":
        raise LiveEvidenceAdapterError(
            code="no_repair_terminal_without_gap",
            reason="no-repair stop is valid only after an explicit repair route",
        )
    readiness = state.get("resource_evidence_readiness")
    if not isinstance(readiness, list) or not readiness:
        raise LiveEvidenceAdapterError(
            code="missing_no_repair_readiness",
            reason="no-repair terminal requires explicit readiness rows",
        )
    ready_count = sum(
        isinstance(item, dict) and item.get("readiness_state") == "ready"
        for item in readiness
    )
    status = (
        "partial_resources_ready" if ready_count else "blocked_insufficient_evidence"
    )
    return _merge_state(
        state,
        {
            "evidence_orchestration_route": "terminal",
            "evidence_terminal_status": status,
            "evidence_terminal_reason_code": "repair_disabled_by_variant",
            "evidence_orchestration_status": "terminal",
        },
    )


@dataclass(frozen=True, slots=True)
class EvidenceVariantServedComposition:
    """Execute the real candidate nodes under one explicit factorial policy."""

    runtime: EvidenceOrchestrationRuntime
    variant: Variant
    resource_planning_enabled: bool
    bounded_repair_enabled: bool

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, EvidenceOrchestrationRuntime):
            raise TypeError("runtime must be EvidenceOrchestrationRuntime")
        expected = _VARIANT_FACTORS.get(self.variant)
        if expected != (
            self.resource_planning_enabled,
            self.bounded_repair_enabled,
        ):
            raise ValueError("variant factors do not match canonical semantics")

    @property
    def composition_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "schema_version": _ADAPTER_ALGORITHM,
                "variant": self.variant,
                "resource_planning_enabled": self.resource_planning_enabled,
                "bounded_repair_enabled": self.bounded_repair_enabled,
                "orchestration_fingerprint": self.runtime.orchestration_fingerprint,
                "nodes": [
                    "learner_path_planner"
                    if self.resource_planning_enabled
                    else "profile_kg_no_planning_compiler",
                    "resource_evidence_planner"
                    if self.resource_planning_enabled
                    else "profile_kg_no_planning_compiler",
                    "retrieval_round_router",
                    "local_rag_search_batch",
                    "web_research_search_batch",
                    "retrieval_round_merge",
                    "requirement_evidence_judge",
                    "evidence_repair_planner"
                    if self.bounded_repair_enabled
                    else "explicit_no_repair_terminal",
                    "terminal_parent_hydration",
                    "resource_evidence_assignment",
                ],
            }
        )

    async def execute(self, case: EvidenceEvaluationCaseSpecV2) -> LearningState:
        if not isinstance(case, EvidenceEvaluationCaseSpecV2):
            raise TypeError("case must be EvidenceEvaluationCaseSpecV2")
        state = _initial_case_state(case=case, variant=self.variant)
        if self.resource_planning_enabled:
            path_update = await make_learner_path_planner_node(
                self.runtime.learning_guidance
            )(state)
            state = _merge_state(state, path_update)
            plan_update = await make_resource_evidence_planner_node(self.runtime)(state)
        else:
            plan_update = _compile_no_planning_state(case=case, runtime=self.runtime)
        state = _merge_state(state, plan_update)
        _validate_planned_target_topics(state=state, case=case)

        while True:
            state = _merge_state(
                state,
                make_retrieval_round_router_node(self.runtime)(state),
            )
            local_update, web_update = await asyncio.gather(
                make_local_rag_search_batch_node(self.runtime)(state),
                make_web_research_search_batch_node(self.runtime)(state),
            )
            state = _merge_state(state, local_update)
            state = _merge_state(state, web_update)
            state = _merge_state(
                state,
                make_retrieval_round_merge_node(self.runtime)(state),
            )
            state = _merge_state(
                state,
                await make_requirement_evidence_judge_node(self.runtime)(state),
            )
            route = route_after_requirement_evidence_judge(state)
            if route == "terminal":
                break
            if not self.bounded_repair_enabled:
                state = _force_no_repair_terminal(state)
                break
            state = _merge_state(
                state,
                make_evidence_repair_planner_node(self.runtime)(state),
            )

        state = _merge_state(
            state,
            await make_terminal_parent_hydration_node(self.runtime)(state),
        )
        state = _merge_state(
            state,
            make_resource_evidence_assignment_node(self.runtime)(state),
        )
        return state


def _safe_failure_code(error: BaseException) -> str:
    value = getattr(error, "code", None)
    if (
        isinstance(value, str)
        and value
        and len(value) <= 200
        and value[0].isalnum()
        and all(character.isalnum() or character in "._:-" for character in value)
    ):
        return value
    return "live_variant_execution_failed"


def _safe_failure_type(error: BaseException) -> str:
    value = type(error).__name__
    if value and len(value) <= 200 and value.replace("_", "").isalnum():
        return value
    return "UnexpectedException"


def _attempted_routes(
    tasks: Sequence[RetrievalTask],
) -> set[tuple[ResourceType, str, Literal["parent_child", "web"]]]:
    return {
        (task.resource_type, task.subject, _SOURCE_ROUTE[task.source_type])
        for task in tasks
    }


def _expected_routes(
    case: EvidenceEvaluationCaseSpecV2,
) -> set[tuple[ResourceType, str, Literal["parent_child", "web"]]]:
    return {
        (target.resource_type, target.subject, source)
        for target in case.targets
        for source in target.required_sources
    }


def _assigned_pairs(
    assignments: Sequence[ResourceEvidenceAssignment],
) -> set[tuple[ResourceType, str]]:
    return {
        (assignment.resource_type, subject)
        for assignment in assignments
        for subject in assignment.subjects
    }


def _observation(
    *,
    case: EvidenceEvaluationCaseSpecV2,
    binding: EvidenceEvaluationRuntimeBindingV2,
    definition: EvidenceVariantDefinitionV2,
    composition: EvidenceVariantServedComposition,
    state: LearningState,
    latency_ms: float,
) -> EvidenceVariantObservationV2:
    if not math.isfinite(latency_ms) or latency_ms <= 0:
        raise LiveEvidenceAdapterError(
            code="invalid_live_variant_latency",
            reason="measured live latency must be finite and positive",
        )
    tasks = tuple(
        RetrievalTask.model_validate(item)
        for item in (state.get("evidence_all_tasks") or [])
    )
    assignments = tuple(
        ResourceEvidenceAssignment.model_validate(item)
        for item in (state.get("resource_evidence_assignments") or [])
    )
    assigned_pairs = _assigned_pairs(assignments)
    expected_pairs = {(target.resource_type, target.subject) for target in case.targets}
    target_by_id = {target.target_id: target for target in case.targets}
    weighted_total = sum(item.weight for item in case.requirements)
    weighted_covered = sum(
        item.weight
        for item in case.requirements
        if (
            target_by_id[item.target_id].resource_type,
            target_by_id[item.target_id].subject,
        )
        in assigned_pairs
    )
    expected_routes = _expected_routes(case)
    attempted_routes = _attempted_routes(tasks)
    source_outcomes = tuple(state.get("evidence_source_outcomes") or [])
    local_required = any(task.source_type == "local_rag" for task in tasks)
    web_required = any(task.source_type == "web" for task in tasks)
    local_status: Literal["ok", "not_required", "failed"] = (
        "ok" if local_required else "not_required"
    )
    web_status: Literal["ok", "not_required", "failed"] = (
        "ok" if web_required else "not_required"
    )
    if local_required and not any(
        isinstance(item, dict) and item.get("source_type") == "local_rag"
        for item in source_outcomes
    ):
        local_status = "failed"
    if web_required and not any(
        isinstance(item, dict) and item.get("source_type") == "web"
        for item in source_outcomes
    ):
        web_status = "failed"
    query_slots = [
        (task.source_type, task.subject, task.query_fingerprint) for task in tasks
    ]
    repeated_query_count = sum(
        count - 1 for count in Counter(query_slots).values() if count > 1
    )
    ledger = tuple(
        item
        for item in (
            EvidenceLedgerEntry.model_validate(raw)
            for raw in (state.get("evidence_ledger") or [])
        )
        if item.accepted
    )
    correct_evidence_count = sum(
        (
            item.resource_type,
            item.subject,
            _SOURCE_ROUTE[item.source_type],
        )
        in expected_routes
        for item in ledger
    )
    required_gap_count = sum(
        isinstance(item, dict)
        and item.get("readiness_state") == "blocked_insufficient_evidence"
        for item in (state.get("resource_evidence_readiness") or [])
    )
    round_index = state.get("evidence_current_round")
    if isinstance(round_index, bool) or not isinstance(round_index, int):
        raise LiveEvidenceAdapterError(
            code="invalid_live_round_index",
            reason="terminal state lacks an explicit integer round index",
        )
    bounded = (
        round_index <= composition.runtime.policy.max_supplement_rounds
        and len(tasks) <= composition.runtime.policy.max_total_search_tasks
    )
    terminal_reason = state.get("evidence_terminal_reason_code")
    premature_stop = bool(
        required_gap_count
        and terminal_reason == "repair_disabled_by_variant"
        and not composition.bounded_repair_enabled
    )
    output_projection = {
        "schema_version": "served_evidence_variant_output_v1",
        "case_binding": case_binding_identity(case).model_dump(mode="json"),
        "variant": definition.variant,
        "composition_fingerprint": composition.composition_fingerprint,
        "terminal_status": state.get("evidence_terminal_status"),
        "terminal_reason_code": terminal_reason,
        "coverage": state.get("evidence_coverage"),
        "readiness": state.get("resource_evidence_readiness"),
        "assignments": [item.model_dump(mode="json") for item in assignments],
        "accepted_evidence": [
            {
                "evidence_id": item.evidence_id,
                "requirement_id": item.requirement_id,
                "source_type": item.source_type,
                "subject": item.subject,
            }
            for item in ledger
        ],
    }
    return EvidenceVariantObservationV2(
        schema_version="evidence_variant_observation_v2",
        case_id=case.case_id,
        variant=definition.variant,
        query_fingerprint=query_fingerprint(case.query),
        dataset_fingerprint=binding.dataset_fingerprint,
        knowledge_graph_data_version=binding.knowledge_graph_data_version,
        knowledge_graph_artifact_fingerprint=(
            binding.knowledge_graph_artifact_fingerprint
        ),
        case_binding=case_binding_identity(case),
        execution_config_fingerprint=binding.execution_config_fingerprint,
        benchmark_config_fingerprint=binding.benchmark_config_fingerprint,
        rollout_config_fingerprint=binding.rollout_config_fingerprint,
        runtime_fingerprint=binding.runtime_fingerprint,
        generation_id=binding.generation_id,
        generation_manifest_fingerprint=(binding.generation_manifest_fingerprint),
        executor_fingerprint=binding.executor_fingerprint,
        variant_definition_fingerprint=model_fingerprint(definition),
        output_fingerprint=canonical_sha256(output_projection),
        provider_status="ok",
        parent_child_status=local_status,
        web_status=web_status,
        bounded=bounded,
        forced_stop_marked_sufficient=bool(
            required_gap_count and state.get("evidence_terminal_status") == "sufficient"
        ),
        silent_resource_omission=not {item[0] for item in expected_pairs}.issubset(
            {item[0] for item in assigned_pairs}
        ),
        silent_subject_omission=not {item[1] for item in expected_pairs}.issubset(
            {item[1] for item in assigned_pairs}
        ),
        repeated_query_count=repeated_query_count,
        weighted_covered=weighted_covered,
        weighted_total=weighted_total,
        required_gap_count=required_gap_count,
        selected_evidence_count=len(ledger),
        correct_evidence_count=correct_evidence_count,
        premature_stop=premature_stop,
        over_search=case.initial_evidence.state == "sufficient" and bool(tasks),
        source_route_true_positive=len(expected_routes & attempted_routes),
        source_route_false_positive=len(attempted_routes - expected_routes),
        source_route_false_negative=len(expected_routes - attempted_routes),
        expected_resource_subject_count=len(expected_pairs),
        assigned_resource_subject_count=len(assigned_pairs),
        correct_resource_subject_count=len(expected_pairs & assigned_pairs),
        retrieval_cost_units=0.25 + float(len(tasks)),
        latency_ms=latency_ms,
    )


class ServedEvidenceLiveVariantAdapter:
    """V2 adapter around one exact real-node served composition."""

    def __init__(
        self,
        *,
        dataset: EvidenceEvaluationDatasetV2,
        knowledge_graph: KnowledgeGraphV1,
        generation_manifest_fingerprint: str,
        composition: EvidenceVariantServedComposition,
    ) -> None:
        if not isinstance(dataset, EvidenceEvaluationDatasetV2):
            raise TypeError("dataset must be EvidenceEvaluationDatasetV2")
        if not isinstance(knowledge_graph, KnowledgeGraphV1):
            raise TypeError("knowledge_graph must be KnowledgeGraphV1")
        if knowledge_graph.data_version != dataset.knowledge_graph_data_version:
            raise ValueError("dataset knowledge graph data_version mismatch")
        if (
            knowledge_graph.artifact_fingerprint
            != dataset.knowledge_graph_artifact_fingerprint
        ):
            raise ValueError("dataset knowledge graph artifact mismatch")
        if (
            not isinstance(generation_manifest_fingerprint, str)
            or len(generation_manifest_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in generation_manifest_fingerprint
            )
        ):
            raise ValueError("generation_manifest_fingerprint must be SHA-256")
        if not isinstance(composition, EvidenceVariantServedComposition):
            raise TypeError("composition must be EvidenceVariantServedComposition")
        self._composition = composition
        adapter_fingerprint = canonical_sha256(
            {
                "schema_version": "served_evidence_live_adapter_v2",
                "dataset_id": dataset.dataset_id,
                "dataset_fingerprint": dataset.dataset_fingerprint,
                "knowledge_graph_data_version": dataset.knowledge_graph_data_version,
                "knowledge_graph_artifact_fingerprint": (
                    dataset.knowledge_graph_artifact_fingerprint
                ),
                "generation_id": composition.runtime.parent_child.generation_id,
                "generation_manifest_fingerprint": generation_manifest_fingerprint,
                "composition_fingerprint": composition.composition_fingerprint,
                "declared_cases": [
                    item.model_dump(mode="json")
                    for item in dataset_case_bindings(dataset)
                ],
            }
        )
        self._identity = EvidenceLiveAdapterIdentityV2(
            schema_version="evidence_live_adapter_identity_v2",
            variant=composition.variant,
            resource_planning_enabled=composition.resource_planning_enabled,
            bounded_repair_enabled=composition.bounded_repair_enabled,
            adapter_fingerprint=adapter_fingerprint,
            dataset_id=dataset.dataset_id,
            dataset_fingerprint=dataset.dataset_fingerprint,
            knowledge_graph_data_version=dataset.knowledge_graph_data_version,
            knowledge_graph_artifact_fingerprint=(
                dataset.knowledge_graph_artifact_fingerprint
            ),
            declared_cases=dataset_case_bindings(dataset),
        )

    @property
    def identity(self) -> EvidenceLiveAdapterIdentityV2:
        return self._identity

    async def execute(
        self,
        *,
        case: EvidenceEvaluationCaseSpecV2,
        binding: EvidenceEvaluationRuntimeBindingV2,
    ) -> EvidenceVariantAttemptV2:
        expected_case: EvidenceCaseBindingIdentityV2 | None = next(
            (
                item
                for item in self._identity.declared_cases
                if item.case_id == case.case_id
            ),
            None,
        )
        if expected_case is None or expected_case != case_binding_identity(case):
            raise LiveEvidenceAdapterError(
                code="live_adapter_case_binding_mismatch",
                reason="case does not match the adapter's exact declared inventory",
            )
        if (
            binding.runtime_fingerprint
            != self._composition.runtime.orchestration_fingerprint
            or binding.generation_id
            != self._composition.runtime.parent_child.generation_id
        ):
            raise LiveEvidenceAdapterError(
                code="live_adapter_runtime_binding_mismatch",
                reason="runtime binding differs from the served composition",
            )
        definition = EvidenceVariantDefinitionV2(
            variant=self._composition.variant,
            resource_planning_enabled=(self._composition.resource_planning_enabled),
            bounded_repair_enabled=self._composition.bounded_repair_enabled,
        )
        started_ns = time.perf_counter_ns()
        try:
            state = await self._composition.execute(case)
            latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
            observation = _observation(
                case=case,
                binding=binding,
                definition=definition,
                composition=self._composition,
                state=state,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            return EvidenceVariantAttemptV2(
                schema_version="evidence_variant_attempt_v2",
                case_id=case.case_id,
                variant=self._composition.variant,
                status="failed",
                observation=None,
                failure_reason_code=_safe_failure_code(exc),
                failure_type=_safe_failure_type(exc),
            )
        return EvidenceVariantAttemptV2(
            schema_version="evidence_variant_attempt_v2",
            case_id=case.case_id,
            variant=self._composition.variant,
            status="success",
            observation=observation,
            failure_reason_code=None,
            failure_type=None,
        )


def build_served_evidence_live_adapters(
    *,
    dataset: EvidenceEvaluationDatasetV2,
    knowledge_graph: KnowledgeGraphV1,
    runtime: EvidenceOrchestrationRuntime,
    generation_manifest_fingerprint: str,
) -> tuple[ServedEvidenceLiveVariantAdapter, ...]:
    """Build the complete canonical matrix with no inferred or missing adapter."""

    return tuple(
        ServedEvidenceLiveVariantAdapter(
            dataset=dataset,
            knowledge_graph=knowledge_graph,
            generation_manifest_fingerprint=generation_manifest_fingerprint,
            composition=EvidenceVariantServedComposition(
                runtime=runtime,
                variant=variant,
                resource_planning_enabled=factors[0],
                bounded_repair_enabled=factors[1],
            ),
        )
        for variant, factors in _VARIANT_FACTORS.items()
    )


__all__ = [
    "EvidenceVariantServedComposition",
    "LiveEvidenceAdapterError",
    "ServedEvidenceLiveVariantAdapter",
    "build_served_evidence_live_adapters",
]
