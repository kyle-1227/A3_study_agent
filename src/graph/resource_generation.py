"""Parallel learning-resource generation orchestration.

The graph-level resource node uses LangGraph dynamic fan-out/fan-in while each
worker reuses the existing resource-generation node functions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from uuid import uuid4

from langchain_core.messages import AIMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Send
from pydantic import ValidationError

from src.config.evidence_orchestration_contracts import (
    RESOURCE_EVIDENCE_CONTRACT_VERSION,
    ResourceEvidenceAssignment,
)
from src.context_engineering.influence import merge_context_influence_ledger
from src.context_engineering.influence_runtime import (
    begin_influence_capture,
    build_node_output_influences,
    combine_influence_updates,
    emit_influence_capture_trace,
    end_influence_capture,
    influence_entries_for_scope,
)
from src.context_engineering.workspace import (
    build_workspace_artifact_update,
    workspace_trace_payload,
)
from src.graph.assessment_quiz import validate_assessment_quiz_runtime_binding_v1
from src.graph.code_practice import (
    code_practice_agent,
    code_practice_output,
    code_practice_planner,
    code_practice_reviewer,
    code_practice_rewrite,
    should_rewrite_code_practice,
)
from src.graph.exercises import (
    exercise_agent,
    exercise_output,
    exercise_planner,
    exercise_reviewer,
    exercise_rewrite,
    should_rewrite_exercise,
)
from src.graph.mindmap import (
    mindmap_agent,
    mindmap_output,
    mindmap_planner,
    mindmap_reviewer,
    mindmap_rewrite,
    should_rewrite_mindmap,
)
from src.graph.learning_guidance import (
    automatic_recommendation_scope_reason,
    recommendation_public_status_message,
    resource_final_recommendations,
    resource_recommendation_output_for_runtime_from_state,
)
from src.graph.review_doc import (
    review_doc_agent,
    review_doc_output,
    review_doc_planner,
    review_doc_reviewer,
    review_doc_rewrite,
    should_rewrite_review_doc,
)
from src.graph.resource_contracts import (
    RESOURCE_ALIASES as _RESOURCE_ALIASES,
    RESOURCE_TYPE_ORDER as _RESOURCE_TYPE_ORDER,
    SUPPORTED_RESOURCE_TYPES as _SUPPORTED_RESOURCE_TYPES,
    normalize_requested_resource_types,
    normalize_resource_type,
)
from src.graph.resource_final_runtime import build_resource_final_v3_from_bundle
from src.graph.resource_final_v3 import (
    ResourceFinalV3,
    ResourceFinalV3Recommendation,
    ResourceFinalV3TerminalStatus,
)
from src.graph.resource_validation import (
    ResourceValidationResultV1,
    validate_renderable_resource_result,
)
from src.graph.state import LearningState, RESOURCE_RESULTS_CLEAR
from src.learning_guidance.contracts import RecommendationResourceContextV1
from src.learning_guidance.runtime import (
    LearningGuidanceContractError,
    LearningGuidanceRuntime,
)
from src.graph.study_plan import (
    route_after_study_plan_consensus,
    study_plan_agent,
    study_plan_consensus,
    study_plan_emotional_intel,
    missing_profile_fields_for_resource,
    study_plan_output,
    study_plan_planner,
    study_plan_reviewer_academic,
    study_plan_reviewer_emotional,
    study_plan_rewrite,
)
from src.graph.video_animation import (
    should_rewrite_video_animation,
    video_animation_agent,
    video_animation_output,
    video_animation_planner,
    video_animation_reviewer,
    video_animation_rewrite,
)
from src.graph.video_script import (
    should_rewrite_video_script,
    video_script_agent,
    video_script_output,
    video_script_planner,
    video_script_reviewer,
    video_script_rewrite,
)
from src.observability.a3_trace import emit_a3_trace
from src.observability.node_registry import get_node_runtime_metadata
from src.observability.performance_runtime import performance_span
from src.tools.search_tool import sanitize_error_message
from src.tracing import traced_node

logger = logging.getLogger(__name__)

# Compatibility exports for existing callers; canonical ownership lives in
# ``resource_contracts`` so planners no longer import the generation runtime.
RESOURCE_ALIASES = _RESOURCE_ALIASES
RESOURCE_TYPE_ORDER = _RESOURCE_TYPE_ORDER
SUPPORTED_RESOURCE_TYPES = _SUPPORTED_RESOURCE_TYPES

RESOURCE_OUTPUT_STATE_KEYS: dict[str, tuple[str, ...]] = {
    "mindmap": (
        "mindmap_outline",
        "mindmap_tree",
        "mindmap_artifact",
        "mindmap_review_verdict",
        "mindmap_review_reason",
        "mindmap_revision_notes",
        "mindmap_round",
    ),
    "quiz": (
        "exercise_outline",
        "exercise_items",
        "exercise_artifact",
        "exercise_resource_v3",
        "exercise_review_verdict",
        "exercise_review_reason",
        "exercise_revision_notes",
        "exercise_round",
    ),
    "review_doc": (
        "review_doc_outline",
        "review_doc_markdown",
        "review_doc_markdowns",
        "review_doc_artifact",
        "review_doc_artifacts",
        "review_doc_review_verdict",
        "review_doc_review_reason",
        "review_doc_revision_notes",
        "review_doc_round",
    ),
    "study_plan": (
        "study_plan_emotional_intel",
        "study_plan_emotional_profile",
        "study_plan_outline",
        "study_plan_artifact",
        "study_plan_markdown",
        "study_plan_round",
        "study_plan_academic_verdict",
        "study_plan_academic_reason",
        "study_plan_emotional_verdict",
        "study_plan_emotional_reason",
        "study_plan_consensus",
        "study_plan_revision_notes",
        "study_plan_document_artifact",
    ),
    "code_practice": (
        "code_practice_outline",
        "code_practice_markdown",
        "code_practice_artifact",
        "code_practice_review_verdict",
        "code_practice_review_reason",
        "code_practice_revision_notes",
        "code_practice_round",
    ),
    "video_script": (
        "video_script_outline",
        "video_script_markdown",
        "video_script_srt",
        "video_script_artifact",
        "video_script_review_verdict",
        "video_script_review_reason",
        "video_script_revision_notes",
        "video_script_round",
    ),
    "video_animation": (
        "video_animation_spec",
        "video_animation_html",
        "video_animation_artifact",
        "video_animation_review_verdict",
        "video_animation_review_reason",
        "video_animation_revision_notes",
        "video_animation_round",
        "video_animation_render_log",
    ),
}


@dataclass(frozen=True)
class ResourceResultContract:
    artifact_key: str
    artifacts_key: str = ""
    embedded_artifact_keys: tuple[tuple[str, str], ...] = ()
    title_state_key: str = ""


RESOURCE_RESULT_CONTRACTS: dict[str, ResourceResultContract] = {
    "mindmap": ResourceResultContract(
        artifact_key="mindmap_artifact",
        title_state_key="mindmap_tree",
    ),
    "quiz": ResourceResultContract(artifact_key="exercise_artifact"),
    "review_doc": ResourceResultContract(
        artifact_key="review_doc_artifact",
        artifacts_key="review_doc_artifacts",
    ),
    "study_plan": ResourceResultContract(
        artifact_key="study_plan_artifact",
        embedded_artifact_keys=(("document", "study_plan_document_artifact"),),
    ),
    "code_practice": ResourceResultContract(artifact_key="code_practice_artifact"),
    "video_script": ResourceResultContract(artifact_key="video_script_artifact"),
    "video_animation": ResourceResultContract(artifact_key="video_animation_artifact"),
}

if set(RESOURCE_RESULT_CONTRACTS) != set(SUPPORTED_RESOURCE_TYPES):
    raise RuntimeError(
        "resource result contracts do not cover supported resource types"
    )


def _candidate_resource_assignments_from_state(
    state: LearningState,
    *,
    required: bool,
) -> tuple[ResourceEvidenceAssignment, ...] | None:
    marker = state.get("resource_evidence_contract_version")
    raw_scope = state.get("evidence_requested_resource_types")
    scope_is_empty = raw_scope is None or raw_scope == [] or raw_scope == ()
    if marker in (None, ""):
        if not scope_is_empty:
            raise LearningGuidanceContractError(
                code="missing_resource_evidence_contract_version",
                reason=(
                    "evidence-scoped resource flow requires an explicit contract "
                    "version"
                ),
            )
        if required:
            raise LearningGuidanceContractError(
                code="missing_candidate_evidence_scope",
                reason=(
                    "candidate resource flow requires explicit evidence resource scope"
                ),
            )
        return None
    if marker != RESOURCE_EVIDENCE_CONTRACT_VERSION:
        raise LearningGuidanceContractError(
            code="invalid_resource_evidence_contract_version",
            reason="resource evidence contract version is unknown",
        )
    if scope_is_empty:
        raise LearningGuidanceContractError(
            code="missing_candidate_evidence_scope",
            reason="versioned candidate flow requires non-empty evidence scope",
        )
    if not isinstance(raw_scope, (list, tuple)):
        raise LearningGuidanceContractError(
            code="invalid_candidate_evidence_scope",
            reason="candidate evidence resource scope must be a sequence",
        )
    scope = tuple(raw_scope)
    if any(
        not isinstance(item, str) or item not in SUPPORTED_RESOURCE_TYPES
        for item in scope
    ) or len(scope) != len(set(scope)):
        raise LearningGuidanceContractError(
            code="invalid_candidate_evidence_scope",
            reason=(
                "candidate evidence resource scope must contain unique canonical "
                "resource types"
            ),
        )
    raw_assignments = state.get("resource_evidence_assignments")
    if not isinstance(raw_assignments, (list, tuple)):
        raise LearningGuidanceContractError(
            code="invalid_resource_evidence_assignments",
            reason="candidate assignments must be a sequence",
        )
    try:
        assignments = tuple(
            ResourceEvidenceAssignment.model_validate(item) for item in raw_assignments
        )
    except (TypeError, ValidationError, ValueError):
        raise LearningGuidanceContractError(
            code="invalid_resource_evidence_assignments",
            reason="candidate assignment violates its strict fingerprinted contract",
        ) from None
    resource_types = tuple(item.resource_type for item in assignments)
    if len(resource_types) != len(set(resource_types)):
        raise LearningGuidanceContractError(
            code="duplicate_resource_evidence_assignment",
            reason="candidate assignments must contain one row per resource type",
        )
    return assignments


def _fallback_delivery_timeout_seconds(state: LearningState) -> float:
    raw_timeout = state.get("resource_fallback_delivery_max_seconds")
    if (
        isinstance(raw_timeout, bool)
        or not isinstance(raw_timeout, (int, float))
        or not math.isfinite(float(raw_timeout))
        or raw_timeout <= 0
    ):
        raise LearningGuidanceContractError(
            code="invalid_fallback_delivery_timeout",
            reason="fallback worker requires a positive finite delivery timeout",
        )
    return float(raw_timeout)


def _resource_plan_from_state(state: LearningState) -> list[dict]:
    assignments = _candidate_resource_assignments_from_state(
        state,
        required=False,
    )
    if assignments is not None:
        raw_resources = state.get("requested_resource_types")
        if not isinstance(raw_resources, (list, tuple)):
            raise LearningGuidanceContractError(
                code="invalid_candidate_ready_resources",
                reason="candidate ready resources must be an explicit sequence",
            )
        resources = list(raw_resources)
        if any(
            not isinstance(item, str) or item not in SUPPORTED_RESOURCE_TYPES
            for item in resources
        ) or len(resources) != len(set(resources)):
            raise LearningGuidanceContractError(
                code="invalid_candidate_ready_resources",
                reason=(
                    "candidate ready resources must use unique exact canonical types"
                ),
            )
        singular = state.get("requested_resource_type")
        expected_singular = resources[0] if resources else ""
        if singular != expected_singular:
            raise LearningGuidanceContractError(
                code="candidate_ready_resource_shadow_mismatch",
                reason=(
                    "candidate requested_resource_type must match the exact ready "
                    "resource list"
                ),
            )
        assignment_by_resource = {item.resource_type: item for item in assignments}
        if set(assignment_by_resource) != set(resources):
            raise LearningGuidanceContractError(
                code="resource_assignment_inventory_mismatch",
                reason=(
                    "candidate assignments must exactly match ready resource types"
                ),
            )
        tasks: list[dict[str, object]] = []
        fallback_timeout_seconds: float | None = None
        for resource_type in resources:
            assignment = assignment_by_resource[resource_type]
            task: dict[str, object] = {
                "task_id": f"resource:{resource_type}",
                "resource_type": resource_type,
                "subjects": list(assignment.subjects),
                "topic_ids": list(assignment.topic_ids),
                "delivery_mode": assignment.delivery_mode,
                "status": "pending",
            }
            if assignment.delivery_mode == "fallback":
                if fallback_timeout_seconds is None:
                    fallback_timeout_seconds = _fallback_delivery_timeout_seconds(state)
                task["fallback_delivery_timeout_seconds"] = fallback_timeout_seconds
            tasks.append(task)
        return tasks
    resources = normalize_requested_resource_types(
        state.get("requested_resource_types") or [],
        state.get("requested_resource_type") or "",
    )
    return [
        {
            "task_id": f"resource:{resource_type}",
            "resource_type": resource_type,
            "status": "pending",
        }
        for resource_type in resources
    ]


def _resource_assignment_for_worker(
    state: LearningState,
    resource_type: str,
) -> tuple[ResourceEvidenceAssignment, list[dict]] | None:
    """Return the strict candidate assignment and its resource-scoped context."""

    parsed_assignments = _candidate_resource_assignments_from_state(
        state,
        required=False,
    )
    if parsed_assignments is None:
        return None
    assignments = tuple(
        item for item in parsed_assignments if item.resource_type == resource_type
    )
    if len(assignments) != 1:
        raise LearningGuidanceContractError(
            code="missing_worker_resource_assignment",
            reason="resource-aware worker requires one exact evidence assignment",
        )
    assignment = assignments[0]
    evidence_ids = frozenset(assignment.evidence_ids)
    scoped_context: list[dict] = []
    for item in state.get("context") or []:
        if not isinstance(item, dict):
            raise LearningGuidanceContractError(
                code="invalid_resource_evidence_context",
                reason="resource-aware context items must be mappings",
            )
        if "evidence_ids" in item:
            raw_evidence_ids = item["evidence_ids"]
            if not isinstance(raw_evidence_ids, (list, tuple)):
                raise LearningGuidanceContractError(
                    code="invalid_resource_evidence_context",
                    reason="context evidence_ids must be a sequence",
                )
            item_evidence_ids = tuple(raw_evidence_ids)
        elif "evidence_id" in item:
            item_evidence_ids = (item["evidence_id"],)
        else:
            continue
        if (
            not item_evidence_ids
            or any(
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
                for value in item_evidence_ids
            )
            or len(item_evidence_ids) != len(set(item_evidence_ids))
        ):
            raise LearningGuidanceContractError(
                code="invalid_resource_evidence_context",
                reason="context evidence ids must be unique normalized strings",
            )
        if frozenset(item_evidence_ids) & evidence_ids:
            scoped_context.append(item)
    if not scoped_context:
        raise ValueError(
            "resource-aware worker assignment resolved to an empty approved context"
        )
    return assignment, scoped_context


_FALLBACK_EVIDENCE_SCOPE_CONSTRAINT = (
    "Use only claims explicitly supported by the accepted evidence below. "
    "Omit unsupported extensions, examples, or prerequisites."
)
_FALLBACK_EVIDENCE_SCOPE_WARNING = "evidence_scope_limited"


def _constrain_fallback_context(context: list[dict]) -> list[dict]:
    """Keep accepted evidence intact while surfacing the fallback scope boundary."""

    constrained: list[dict] = []
    for item in context:
        updated = dict(item)
        content = updated.get("content")
        if isinstance(content, str) and content.strip():
            updated["content"] = f"{_FALLBACK_EVIDENCE_SCOPE_CONSTRAINT}\n\n{content}"
        constrained.append(updated)
    return constrained


def _fallback_delivery_timeout_from_task(task: dict) -> float:
    raw_timeout = task.get("fallback_delivery_timeout_seconds")
    if (
        isinstance(raw_timeout, bool)
        or not isinstance(raw_timeout, (int, float))
        or not math.isfinite(float(raw_timeout))
        or raw_timeout <= 0
    ):
        raise LearningGuidanceContractError(
            code="invalid_fallback_delivery_timeout",
            reason="fallback resource task requires a positive finite timeout",
        )
    return float(raw_timeout)


def _mark_fallback_result_partial_success(result: dict) -> dict:
    """Revalidate an already-renderable result before exposing limited delivery."""

    raw_validation = result.get("validation")
    if not isinstance(raw_validation, dict):
        raise LearningGuidanceContractError(
            code="invalid_fallback_validation",
            reason="fallback delivery requires a serialized resource validation",
        )
    validation = ResourceValidationResultV1.model_validate_json(
        json.dumps(raw_validation, ensure_ascii=False)
    )
    if not validation.valid:
        raise LearningGuidanceContractError(
            code="invalid_fallback_validation",
            reason="fallback delivery requires a successful normal resource validation",
        )
    warnings = tuple(
        dict.fromkeys((_FALLBACK_EVIDENCE_SCOPE_WARNING, *validation.warnings))
    )[:24]
    limited_validation = ResourceValidationResultV1.model_validate(
        {
            **validation.model_dump(mode="python"),
            "terminal_status": "partial_success",
            "warnings": warnings,
        }
    )
    limited_result = dict(result)
    limited_result["status"] = "partial_success"
    limited_result["validation"] = limited_validation.model_dump(mode="json")
    return limited_result


def _debug_base(state: LearningState, tasks: list[dict]) -> dict:
    run_id = str(state.get("request_id") or uuid4())
    resource_types = [task["resource_type"] for task in tasks]
    return {
        "run_id": run_id,
        "status": "running" if tasks else "skipped",
        "selected_resource_types": resource_types,
        "success_count": 0,
        "failed_count": 0,
        "partial_success": False,
        "developer_warnings": [],
        "stages": [
            {
                "stage": "resource_generation.orchestrator.start",
                "status": "success" if tasks else "skipped",
                "selected_resource_types": resource_types,
                "task_count": len(tasks),
            }
        ],
    }


@traced_node
async def resource_orchestrator(state: LearningState) -> dict:
    """Plan resource worker tasks after Evidence Judge V2 has approved context."""
    tasks = _resource_plan_from_state(state)
    resource_types = [task["resource_type"] for task in tasks]
    debug = _debug_base(state, tasks)
    debug["stages"].append(
        {
            "stage": "resource_generation.orchestrator.success",
            "status": "success" if tasks else "skipped",
            "task_count": len(tasks),
            "selected_resource_types": resource_types,
        }
    )
    emit_a3_trace(
        logger,
        "resource_generation.orchestrator.success",
        {
            "task_count": len(tasks),
            "selected_resource_types": resource_types,
            "status": debug["status"],
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    plan = {
        "tasks": tasks,
        "blocked_resource_types": list(state.get("blocked_resource_types") or []),
        "requested_resource_types": list(
            state.get("evidence_requested_resource_types") or resource_types
        ),
    }
    return {
        "requested_resource_type": resource_types[0] if resource_types else "",
        "requested_resource_types": resource_types,
        "resource_generation_plan": plan,
        "resource_branch_results": RESOURCE_RESULTS_CLEAR,
        "resource_bundle_artifact": {},
        "resource_generation_debug": debug,
        "resource_generation_status": "running" if tasks else "skipped",
    }


def dispatch_resource_workers(state: LearningState) -> list[Send]:
    """Dynamically fan out one worker per planned resource task."""
    tasks = (state.get("resource_generation_plan") or {}).get("tasks") or []
    if not tasks:
        return [Send("resource_bundle_output", dict(state))]
    return [
        Send("resource_worker", {**dict(state), "resource_task": task})
        for task in tasks
    ]


def dispatch_resource_workers_to_recommendation_aggregator(
    state: LearningState,
) -> list[Send]:
    """Candidate-only fan-out whose empty path still reaches recommendation."""

    tasks = (state.get("resource_generation_plan") or {}).get("tasks") or []
    if not tasks:
        return [Send("resource_bundle_aggregator", dict(state))]
    return [
        Send("resource_worker", {**dict(state), "resource_task": task})
        for task in tasks
    ]


@traced_node
async def resource_preflight_router(state: LearningState) -> dict:
    """Normalize requested resources before graph-level resource preflight gates."""
    resource_types = normalize_requested_resource_types(
        state.get("requested_resource_types") or [],
        state.get("requested_resource_type") or "",
    )
    emit_a3_trace(
        logger,
        "resource_generation.preflight.checked",
        {
            "requested_resource_types": resource_types,
            "requires_profile_completion_gate": "study_plan" in resource_types,
            "resource_count": len(resource_types),
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "requested_resource_type": resource_types[0] if resource_types else "",
        "requested_resource_types": resource_types,
        "resource_generation_status": "preflight" if resource_types else "skipped",
    }


def route_after_resource_preflight(state: LearningState) -> str:
    """Route study-plan requests through the graph-level profile checkpoint gate."""
    resource_types = normalize_requested_resource_types(
        state.get("requested_resource_types") or [],
        state.get("requested_resource_type") or "",
    )
    return (
        "study_plan_profile_gate_main"
        if "study_plan" in resource_types
        else "resource_orchestrator"
    )


def _merge_node_output(local_state: dict, output: dict | None) -> str:
    if not output:
        return ""
    message_content = ""
    for message in output.get("messages") or []:
        if isinstance(message, AIMessage):
            message_content = str(message.content or "")
        elif hasattr(message, "content"):
            message_content = str(getattr(message, "content") or "")
    for key, value in output.items():
        if key != "messages":
            local_state[key] = value
    return message_content


async def _run_resource_subnode(
    local_state: dict,
    *,
    resource_type: str,
    subnode: str,
    func: Callable[[dict], Awaitable[dict | None]],
) -> dict | None:
    start = time.perf_counter()
    capture_token = begin_influence_capture()
    emit_a3_trace(
        logger,
        "resource_subnode.start",
        {
            "resource_type": resource_type,
            "subnode": subnode,
            "elapsed_ms": 0,
            "status": "start",
            "error_type": "",
        },
        state=local_state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    try:
        metadata = get_node_runtime_metadata(subnode)
        render_span = (
            performance_span(
                "render",
                f"render.{subnode}",
                attributes={"resource_type": resource_type},
            )
            if metadata is not None and metadata.role == "output"
            else nullcontext()
        )
        with render_span:
            output = await func(local_state)
    except GraphInterrupt:
        end_influence_capture(capture_token)
        emit_a3_trace(
            logger,
            "resource_subnode.end",
            {
                "resource_type": resource_type,
                "subnode": subnode,
                "elapsed_ms": int((time.perf_counter() - start) * 1000),
                "status": "interrupted",
                "error_type": "GraphInterrupt",
            },
            state=local_state,
            env_flag="LOG_GENERATION_SUMMARY",
        )
        raise
    except Exception as exc:
        end_influence_capture(capture_token)
        emit_a3_trace(
            logger,
            "resource_subnode.end",
            {
                "resource_type": resource_type,
                "subnode": subnode,
                "elapsed_ms": int((time.perf_counter() - start) * 1000),
                "status": "failed",
                "error_type": type(exc).__name__,
            },
            state=local_state,
            env_flag="LOG_GENERATION_SUMMARY",
        )
        raise
    captured = end_influence_capture(capture_token)
    captured.extend(
        build_node_output_influences(
            node_name=subnode,
            output=output,
            state=local_state,
        )
    )
    if captured:
        influence_update = combine_influence_updates(
            state=local_state,
            updates=(),
            entries=captured,
        )
        local_state["context_influence_ledger"] = merge_context_influence_ledger(
            local_state.get("context_influence_ledger") or {},
            influence_update,
        )
        emit_influence_capture_trace(
            node_name=subnode,
            entries=captured,
            state=local_state,
        )
    emit_a3_trace(
        logger,
        "resource_subnode.end",
        {
            "resource_type": resource_type,
            "subnode": subnode,
            "elapsed_ms": int((time.perf_counter() - start) * 1000),
            "status": "success",
            "error_type": "",
        },
        state=local_state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return output


async def _merge_resource_subnode(
    local_state: dict,
    *,
    resource_type: str,
    subnode: str,
    func: Callable[[dict], Awaitable[dict | None]],
) -> str:
    return _merge_node_output(
        local_state,
        await _run_resource_subnode(
            local_state,
            resource_type=resource_type,
            subnode=subnode,
            func=func,
        ),
    )


async def _merge_resource_sequence(
    local_state: dict,
    *,
    resource_type: str,
    steps: tuple[tuple[str, Callable[[dict], Awaitable[dict | None]]], ...],
) -> str:
    message_content = ""
    for subnode, func in steps:
        message_content = await _merge_resource_subnode(
            local_state,
            resource_type=resource_type,
            subnode=subnode,
            func=func,
        )
    return message_content


async def _run_mindmap_resource(local_state: dict) -> str:
    await _merge_resource_sequence(
        local_state,
        resource_type="mindmap",
        steps=(
            ("mindmap_planner", mindmap_planner),
            ("mindmap_agent", mindmap_agent),
            ("mindmap_reviewer", mindmap_reviewer),
        ),
    )
    while should_rewrite_mindmap(local_state) == "rewrite":
        await _merge_resource_sequence(
            local_state,
            resource_type="mindmap",
            steps=(
                ("mindmap_rewrite", mindmap_rewrite),
                ("mindmap_agent", mindmap_agent),
                ("mindmap_reviewer", mindmap_reviewer),
            ),
        )
    return await _merge_resource_subnode(
        local_state,
        resource_type="mindmap",
        subnode="mindmap_output",
        func=mindmap_output,
    )


async def _run_quiz_resource(local_state: dict) -> str:
    await _merge_resource_sequence(
        local_state,
        resource_type="quiz",
        steps=(
            ("exercise_planner", exercise_planner),
            ("exercise_agent", exercise_agent),
            ("exercise_reviewer", exercise_reviewer),
        ),
    )
    while should_rewrite_exercise(local_state) == "rewrite":
        await _merge_resource_sequence(
            local_state,
            resource_type="quiz",
            steps=(
                ("exercise_rewrite", exercise_rewrite),
                ("exercise_agent", exercise_agent),
                ("exercise_reviewer", exercise_reviewer),
            ),
        )
    return await _merge_resource_subnode(
        local_state,
        resource_type="quiz",
        subnode="exercise_output",
        func=exercise_output,
    )


async def _run_review_doc_resource(local_state: dict) -> str:
    await _merge_resource_sequence(
        local_state,
        resource_type="review_doc",
        steps=(
            ("review_doc_planner", review_doc_planner),
            ("review_doc_agent", review_doc_agent),
            ("review_doc_reviewer", review_doc_reviewer),
        ),
    )
    while should_rewrite_review_doc(local_state) == "rewrite":
        await _merge_resource_sequence(
            local_state,
            resource_type="review_doc",
            steps=(
                ("review_doc_rewrite", review_doc_rewrite),
                ("review_doc_agent", review_doc_agent),
                ("review_doc_reviewer", review_doc_reviewer),
            ),
        )
    return await _merge_resource_subnode(
        local_state,
        resource_type="review_doc",
        subnode="review_doc_output",
        func=review_doc_output,
    )


async def _run_study_plan_resource(local_state: dict) -> str:
    await _merge_resource_sequence(
        local_state,
        resource_type="study_plan",
        steps=(
            ("study_plan_emotional_intel", study_plan_emotional_intel),
            ("study_plan_planner", study_plan_planner),
        ),
    )
    while True:
        await _merge_resource_subnode(
            local_state,
            resource_type="study_plan",
            subnode="study_plan_agent",
            func=study_plan_agent,
        )
        academic_update, emotional_update = await asyncio.gather(
            _run_resource_subnode(
                local_state,
                resource_type="study_plan",
                subnode="study_plan_reviewer_academic",
                func=study_plan_reviewer_academic,
            ),
            _run_resource_subnode(
                local_state,
                resource_type="study_plan",
                subnode="study_plan_reviewer_emotional",
                func=study_plan_reviewer_emotional,
            ),
        )
        _merge_node_output(local_state, academic_update)
        _merge_node_output(local_state, emotional_update)
        await _merge_resource_subnode(
            local_state,
            resource_type="study_plan",
            subnode="study_plan_consensus",
            func=study_plan_consensus,
        )
        if route_after_study_plan_consensus(local_state) == "output":
            return await _merge_resource_subnode(
                local_state,
                resource_type="study_plan",
                subnode="study_plan_output",
                func=study_plan_output,
            )
        await _merge_resource_subnode(
            local_state,
            resource_type="study_plan",
            subnode="study_plan_rewrite",
            func=study_plan_rewrite,
        )


async def _run_code_practice_resource(local_state: dict) -> str:
    await _merge_resource_sequence(
        local_state,
        resource_type="code_practice",
        steps=(
            ("code_practice_planner", code_practice_planner),
            ("code_practice_agent", code_practice_agent),
            ("code_practice_reviewer", code_practice_reviewer),
        ),
    )
    while should_rewrite_code_practice(local_state) == "rewrite":
        await _merge_resource_sequence(
            local_state,
            resource_type="code_practice",
            steps=(
                ("code_practice_rewrite", code_practice_rewrite),
                ("code_practice_agent", code_practice_agent),
                ("code_practice_reviewer", code_practice_reviewer),
            ),
        )
    return await _merge_resource_subnode(
        local_state,
        resource_type="code_practice",
        subnode="code_practice_output",
        func=code_practice_output,
    )


async def _run_video_script_resource(local_state: dict) -> str:
    await _merge_resource_sequence(
        local_state,
        resource_type="video_script",
        steps=(
            ("video_script_planner", video_script_planner),
            ("video_script_agent", video_script_agent),
            ("video_script_reviewer", video_script_reviewer),
        ),
    )
    while should_rewrite_video_script(local_state) == "rewrite":
        await _merge_resource_sequence(
            local_state,
            resource_type="video_script",
            steps=(
                ("video_script_rewrite", video_script_rewrite),
                ("video_script_agent", video_script_agent),
                ("video_script_reviewer", video_script_reviewer),
            ),
        )
    return await _merge_resource_subnode(
        local_state,
        resource_type="video_script",
        subnode="video_script_output",
        func=video_script_output,
    )


async def _run_video_animation_resource(local_state: dict) -> str:
    await _merge_resource_sequence(
        local_state,
        resource_type="video_animation",
        steps=(
            ("video_animation_planner", video_animation_planner),
            ("video_animation_agent", video_animation_agent),
            ("video_animation_reviewer", video_animation_reviewer),
        ),
    )
    while should_rewrite_video_animation(local_state) == "rewrite":
        await _merge_resource_sequence(
            local_state,
            resource_type="video_animation",
            steps=(
                ("video_animation_rewrite", video_animation_rewrite),
                ("video_animation_agent", video_animation_agent),
                ("video_animation_reviewer", video_animation_reviewer),
            ),
        )
    return await _merge_resource_subnode(
        local_state,
        resource_type="video_animation",
        subnode="video_animation_output",
        func=video_animation_output,
    )


RESOURCE_RUNNERS = {
    "mindmap": _run_mindmap_resource,
    "quiz": _run_quiz_resource,
    "review_doc": _run_review_doc_resource,
    "code_practice": _run_code_practice_resource,
    "video_script": _run_video_script_resource,
    "video_animation": _run_video_animation_resource,
    "study_plan": _run_study_plan_resource,
}


def _assert_study_plan_profile_complete(local_state: dict) -> None:
    """Fail fast if the graph-level profile gate was bypassed for study plans."""
    requirement_status = missing_profile_fields_for_resource(
        local_state,
        "study_plan",
    )
    missing_required = requirement_status.get("missing_required_fields") or []
    if not missing_required:
        return
    missing_keys = [
        str(field.get("key") or "")
        for field in missing_required
        if isinstance(field, dict) and str(field.get("key") or "").strip()
    ]
    raise RuntimeError(
        "study_plan_profile_missing_after_preflight: " + ",".join(sorted(missing_keys))
    )


def _state_updates_for_resource(resource_type: str, local_state: dict) -> dict:
    return {
        key: local_state.get(key)
        for key in RESOURCE_OUTPUT_STATE_KEYS.get(resource_type, ())
        if key in local_state
    }


def _result_contract(resource_type: str) -> ResourceResultContract:
    contract = RESOURCE_RESULT_CONTRACTS.get(resource_type)
    if contract is None:
        raise ValueError(f"resource result contract is not registered: {resource_type}")
    return contract


def _primary_artifact(contract: ResourceResultContract, local_state: dict) -> dict:
    artifact = dict(local_state.get(contract.artifact_key) or {})
    for output_key, state_key in contract.embedded_artifact_keys:
        artifact[output_key] = dict(local_state.get(state_key) or {})
    return artifact


def _artifact_collection(
    contract: ResourceResultContract,
    local_state: dict,
) -> list[dict]:
    if not contract.artifacts_key:
        return []
    return [
        dict(item)
        for item in (local_state.get(contract.artifacts_key) or [])
        if isinstance(item, dict)
    ]


def _resource_title(
    resource_type: str,
    contract: ResourceResultContract,
    artifact: dict,
    local_state: dict,
) -> str:
    title = str(artifact.get("title") or "").strip()
    if not title and contract.title_state_key:
        title_state = local_state.get(contract.title_state_key)
        if isinstance(title_state, dict):
            title = str(title_state.get("title") or "").strip()
    return title or resource_type


def _success_result(
    resource_type: str, local_state: dict, message_content: str, elapsed_ms: int
) -> dict:
    contract = _result_contract(resource_type)
    artifact = _primary_artifact(contract, local_state)
    artifacts = _artifact_collection(contract, local_state)
    state_updates = _state_updates_for_resource(resource_type, local_state)
    validation = validate_renderable_resource_result(
        resource_type,
        artifact,
        artifacts,
        state_updates,
    )
    influence_entries = influence_entries_for_scope(
        local_state.get("context_influence_ledger") or {},
        request_id=str(local_state.get("request_id") or ""),
        workflow=resource_type,
    )
    if not validation.valid:
        return {
            "resource_type": resource_type,
            "status": "failed",
            "title": resource_type,
            "artifact": {},
            "artifacts": [],
            "state_updates": {},
            "message_content": "",
            "message_preview": "",
            "error_type": "ResourceValidationError",
            "error_code": validation.failure_reason,
            "error_message_sanitized": validation.failure_reason,
            "elapsed_ms": elapsed_ms,
            "context_influence_entries": influence_entries,
            "validation": validation.model_dump(mode="json"),
        }
    if resource_type == "quiz":
        validate_assessment_quiz_runtime_binding_v1(
            thread_id=local_state.get("thread_id"),
            exercise_items=local_state.get("exercise_items"),
            exercise_artifact=local_state.get("exercise_artifact"),
            exercise_resource_v3=local_state.get("exercise_resource_v3"),
            assessment_checkpoint_resources=local_state.get(
                "assessment_checkpoint_resources"
            ),
        )
    return {
        "resource_type": resource_type,
        "status": validation.terminal_status,
        "title": _resource_title(resource_type, contract, artifact, local_state),
        "artifact": artifact,
        "artifacts": artifacts,
        "state_updates": state_updates,
        "message_content": message_content,
        "message_preview": message_content[:500],
        "error_type": None,
        "error_message_sanitized": None,
        "elapsed_ms": elapsed_ms,
        "context_influence_entries": influence_entries,
        "validation": validation.model_dump(mode="json"),
    }


def _failed_result(resource_type: str, exc: BaseException, elapsed_ms: int) -> dict:
    error_type = type(exc).__name__
    error_message_sanitized = sanitize_error_message(str(exc), max_chars=1200)
    if not error_message_sanitized:
        # Standard exceptions such as TimeoutError may have no message. Preserve
        # the typed failure without inventing provider details or emitting an
        # invalid blank ResourceFinalV3 error.
        error_message_sanitized = f"Resource generation failed with {error_type}."
    return {
        "resource_type": resource_type,
        "status": "failed",
        "title": resource_type,
        "artifact": {},
        "artifacts": [],
        "state_updates": {},
        "message_content": "",
        "message_preview": "",
        "error_type": error_type,
        "error_code": f"{resource_type}.generation_failed",
        "error_message_sanitized": error_message_sanitized,
        "elapsed_ms": elapsed_ms,
        "validation": None,
    }


def _count_mindmap_nodes(tree: Any) -> int:
    """Count mindmap nodes without assuming a perfect tree shape."""
    try:
        if isinstance(tree, dict):
            children = tree.get("children") or []
            return 1 + _count_mindmap_nodes(children)
        if isinstance(tree, list):
            return sum(_count_mindmap_nodes(child) for child in tree)
    except Exception:
        return 0
    return 0


@traced_node
async def resource_worker(state: LearningState) -> dict:
    """Generate exactly one resource branch and return a reducer-safe result."""
    task = dict(state.get("resource_task") or {})
    resource_type = normalize_resource_type(task.get("resource_type"))
    private_checkpoint_update: dict | None = None
    start = time.perf_counter()
    emit_a3_trace(
        logger,
        "resource_generation.worker.start",
        {"resource_type": resource_type, "task_id": task.get("task_id", "")},
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    try:
        if resource_type not in RESOURCE_RUNNERS:
            raise ValueError(f"unsupported resource_type: {task.get('resource_type')}")
        local_state = dict(state)
        local_state["requested_resource_type"] = resource_type
        local_state["requested_resource_types"] = [resource_type]
        local_state["resource_delivery_mode"] = "strict"
        local_state["resource_evidence_scope_constraint"] = ""
        assignment_result = _resource_assignment_for_worker(
            state,
            resource_type,
        )
        if assignment_result is not None:
            assignment, scoped_context = assignment_result
            if task.get("resource_type") != resource_type:
                raise LearningGuidanceContractError(
                    code="noncanonical_resource_task_type",
                    reason="candidate worker task must use the exact canonical type",
                )
            task_subjects = tuple(task.get("subjects") or ())
            task_topic_ids = tuple(task.get("topic_ids") or ())
            if (
                task_subjects != assignment.subjects
                or task_topic_ids != assignment.topic_ids
            ):
                raise LearningGuidanceContractError(
                    code="resource_task_assignment_mismatch",
                    reason="resource task topic binding differs from its assignment",
                )
            task_delivery_mode = task.get("delivery_mode")
            if task_delivery_mode not in (None, assignment.delivery_mode):
                raise LearningGuidanceContractError(
                    code="resource_task_delivery_mode_mismatch",
                    reason="resource task delivery mode differs from its assignment",
                )
            local_state["resource_evidence_assignment"] = assignment.model_dump(
                mode="json"
            )
            local_state["resource_delivery_mode"] = assignment.delivery_mode
            if assignment.delivery_mode == "fallback":
                local_state["resource_evidence_scope_constraint"] = (
                    _FALLBACK_EVIDENCE_SCOPE_CONSTRAINT
                )
                local_state["context"] = _constrain_fallback_context(scoped_context)
            else:
                local_state["context"] = scoped_context
            local_state["graded_evidence"] = scoped_context
        if resource_type == "study_plan":
            _assert_study_plan_profile_complete(local_state)
        if assignment_result is not None and assignment.delivery_mode == "fallback":
            message_content = await asyncio.wait_for(
                RESOURCE_RUNNERS[resource_type](local_state),
                timeout=_fallback_delivery_timeout_from_task(task),
            )
        else:
            message_content = await RESOURCE_RUNNERS[resource_type](local_state)
        result = _success_result(
            resource_type,
            local_state,
            message_content,
            int((time.perf_counter() - start) * 1000),
        )
        if (
            assignment_result is not None
            and assignment.delivery_mode == "fallback"
            and result["status"] != "failed"
        ):
            result = _mark_fallback_result_partial_success(result)
        if assignment_result is not None:
            result["subjects"] = list(task_subjects)
            result["topic_ids"] = list(task_topic_ids)
        if resource_type == "quiz" and result["status"] in {
            "success",
            "partial_success",
        }:
            raw_checkpoint = local_state.get("assessment_checkpoint_resources")
            if not isinstance(raw_checkpoint, dict) or not raw_checkpoint:
                raise ValueError(
                    "successful quiz did not produce assessment checkpoint state"
                )
            private_checkpoint_update = raw_checkpoint
        emit_a3_trace(
            logger,
            (
                "resource_generation.worker.success"
                if result["status"] in {"success", "partial_success"}
                else "resource_generation.worker.failed"
            ),
            {
                "resource_type": resource_type,
                "elapsed_ms": result["elapsed_ms"],
                "message_chars": len(message_content),
                "terminal_status": result["status"],
                "delivery_mode": local_state["resource_delivery_mode"],
                "renderable_count": (
                    (result.get("validation") or {}).get("renderable_count", 0)
                ),
            },
            state=state,
            env_flag="LOG_GENERATION_SUMMARY",
        )
    except GraphInterrupt:
        emit_a3_trace(
            logger,
            "resource_generation.worker.interrupted",
            {
                "resource_type": resource_type,
                "elapsed_ms": int((time.perf_counter() - start) * 1000),
            },
            state=state,
            env_flag="LOG_GENERATION_SUMMARY",
        )
        raise
    except Exception as exc:
        logger.exception("resource_worker failed for resource_type=%s", resource_type)
        result = _failed_result(
            resource_type or str(task.get("resource_type") or "unknown"),
            exc,
            int((time.perf_counter() - start) * 1000),
        )
        emit_a3_trace(
            logger,
            "resource_generation.worker.failed",
            {
                "resource_type": result["resource_type"],
                "elapsed_ms": result["elapsed_ms"],
                "error_type": result["error_type"],
                "error_message_sanitized": result["error_message_sanitized"],
            },
            state=state,
            env_flag="LOG_GENERATION_SUMMARY",
        )
    if task.get("subjects") is not None:
        result["subjects"] = list(task.get("subjects") or [])
    if task.get("topic_ids") is not None:
        result["topic_ids"] = list(task.get("topic_ids") or [])
    worker_update: dict[str, Any] = {"resource_branch_results": [result]}
    if private_checkpoint_update is not None:
        worker_update["assessment_checkpoint_resources"] = private_checkpoint_update
    return worker_update


def _resource_display_name(resource_type: str) -> str:
    return {
        "review_doc": "复习资料",
        "mindmap": "思维导图",
        "quiz": "练习题",
        "code_practice": "代码实操案例",
        "video_script": "教学视频 / 动画脚本",
        "video_animation": "教学动画 / MP4 视频",
        "study_plan": "学习计划",
    }.get(resource_type, resource_type)


def _yes_no(value: bool) -> str:
    return "是" if value else "否"


def _bundle_display_order(result: dict) -> int:
    order = {
        "review_doc": 0,
        "mindmap": 1,
        "quiz": 2,
        "code_practice": 3,
        "video_script": 4,
        "video_animation": 5,
        "study_plan": 6,
    }
    return order.get(str(result.get("resource_type") or ""), len(order))


def _compose_resource_section(result: dict) -> list[str]:
    resource_type = str(result.get("resource_type") or "")
    title = str(result.get("title") or _resource_display_name(resource_type))
    metrics = _resource_metrics(result)
    lines = [f"### {_resource_display_name(resource_type)}"]

    if resource_type == "review_doc":
        lines.extend(
            [
                "已生成 Markdown / Word 版本。",
                f"- 标题：{title}",
                f"- 文档数量：{metrics.get('artifact_count', 0)}",
                f"- Markdown 字数：{metrics.get('markdown_chars', 0)}",
            ]
        )
    elif resource_type == "mindmap":
        lines.extend(
            [
                "已生成 XMind 版本。",
                f"- 标题：{title}",
                f"- 节点数量：{metrics.get('node_count', 0)}",
            ]
        )
    elif resource_type == "quiz":
        lines.extend(
            [
                "已生成 Markdown / Word 版本。",
                f"- 标题：{title}",
                f"- 题目数量：{metrics.get('item_count', 0)}",
            ]
        )
    elif resource_type == "code_practice":
        lines.extend(
            [
                "已生成 Markdown / Word / Python 源码版本。",
                f"- 标题：{title}",
                f"- 包含 Python 源码：{_yes_no(bool(metrics.get('has_python')))}",
            ]
        )
    elif resource_type == "video_script":
        lines.extend(
            [
                "已生成 Markdown / Word / SRT 字幕版本。",
                f"- 标题：{title}",
                f"- SRT 字符数：{metrics.get('srt_chars', 0)}",
            ]
        )
    elif resource_type == "video_animation":
        generated_formats = "HTML 预览 / JSON / SRT"
        if bool(metrics.get("render_success")):
            generated_formats += " / MP4"
        lines.extend(
            [
                f"已生成 {generated_formats} 版本。",
                f"- 标题：{title}",
                f"- MP4 渲染成功：{_yes_no(bool(metrics.get('render_success')))}",
            ]
        )
    elif resource_type == "study_plan":
        lines.extend(
            [
                "已生成 Markdown / Word 学习计划文档。",
                f"- 标题：{title}",
                f"- 是否包含文档：{_yes_no(bool(metrics.get('has_document')))}",
            ]
        )
    else:
        lines.extend([f"- 标题：{title}"])
    return lines


def _compose_bundle_message(
    status: str,
    successes: list[dict],
    failures: list[dict],
) -> str:
    if len(successes) == 1 and not failures:
        content = str(successes[0].get("message_content") or "").strip()
        if content:
            return content

    lines = [
        "# 已生成多类学习资源",
        "",
        "本次已根据你的请求生成以下学习资源，可分别查看或下载：",
        "",
    ]

    if successes:
        lines.extend(["## 已生成", ""])
        for result in sorted(successes, key=_bundle_display_order):
            lines.extend(_compose_resource_section(result))
            lines.append("")

    if failures:
        lines.extend(["## 未完成", ""])
        for result in sorted(failures, key=_bundle_display_order):
            resource_type = result.get("resource_type") or "unknown"
            reason = (
                result.get("error_message_sanitized")
                or result.get("error_type")
                or "unknown error"
            )
            lines.append(f"- {resource_type}: {reason}")
        lines.append("")

    if status == "failed":
        lines.append("所有请求的学习资源都生成失败，请稍后重试或缩小资源范围。")
    elif status == "partial_success":
        lines.append("部分资源已生成，失败的资源可以稍后单独重试。")

    return "\n".join(lines).strip()


def _append_blocked_resource_message(
    message: str,
    *,
    status: str,
    blocked_resources: list[dict],
) -> str:
    """Append explicit evidence blocking without changing legacy bundle wording."""

    if not blocked_resources:
        return message
    lines = (
        [message, "", "## 证据不足，未生成", ""]
        if message
        else [
            "# 证据不足，资源生成已停止",
            "",
        ]
    )
    for item in blocked_resources:
        resource_type = item.get("resource_type") or "unknown"
        lines.append(f"- {resource_type}: 所需证据尚未满足，已明确阻断生成。")
    lines.extend(
        [
            "",
            (
                "所有请求的资源都因证据不足而停止，未生成降级内容。"
                if status == "blocked_insufficient_evidence"
                else "其余资源保持阻断，不会输出降级内容。"
            ),
        ]
    )
    return "\n".join(lines).strip()


def _resource_metrics(result: dict) -> dict:
    resource_type = result.get("resource_type")
    raw_artifact = result.get("artifact")
    artifact: dict[str, Any] = raw_artifact if isinstance(raw_artifact, dict) else {}
    raw_artifacts = result.get("artifacts")
    artifacts: list[Any] = raw_artifacts if isinstance(raw_artifacts, list) else []
    raw_state_updates = result.get("state_updates")
    state_updates: dict[str, Any] = (
        raw_state_updates if isinstance(raw_state_updates, dict) else {}
    )

    if resource_type == "review_doc":
        markdown = (
            state_updates.get("review_doc_markdown") or artifact.get("markdown") or ""
        )
        return {
            "artifact_count": len(artifacts) or (1 if artifact else 0),
            "markdown_chars": len(str(markdown)),
            "has_markdown": bool(
                artifact.get("markdown_url") or artifact.get("markdown")
            ),
            "has_docx": bool(artifact.get("docx_url") or artifact.get("docx_filename")),
        }

    if resource_type == "mindmap":
        tree = state_updates.get("mindmap_tree") or artifact.get("tree") or {}
        return {
            "has_xmind": bool(artifact.get("xmind_url")),
            "node_count": _count_mindmap_nodes(tree),
            "has_png": bool(artifact.get("png_url")),
            "has_svg": bool(artifact.get("svg_url")),
        }

    if resource_type == "quiz":
        exercise_items = state_updates.get("exercise_items")
        if not isinstance(exercise_items, list):
            exercise_items = []
        return {
            "item_count": len(exercise_items),
            "has_markdown": bool(
                artifact.get("markdown_url") or artifact.get("markdown")
            ),
            "has_docx": bool(artifact.get("docx_url") or artifact.get("docx_filename")),
            "has_pdf": bool(artifact.get("pdf_url") or artifact.get("pdf_filename")),
        }

    if resource_type == "code_practice":
        markdown = (
            state_updates.get("code_practice_markdown")
            or artifact.get("markdown")
            or ""
        )
        return {
            "markdown_chars": len(str(markdown)),
            "has_markdown": bool(artifact.get("markdown_url") or markdown),
            "has_docx": bool(artifact.get("docx_url") or artifact.get("docx_filename")),
            "has_python": bool(
                artifact.get("python_url") or artifact.get("python_filename")
            ),
        }

    if resource_type == "video_script":
        markdown = (
            state_updates.get("video_script_markdown") or artifact.get("markdown") or ""
        )
        srt = state_updates.get("video_script_srt") or artifact.get("srt") or ""
        return {
            "markdown_chars": len(str(markdown)),
            "srt_chars": len(str(srt)),
            "has_markdown": bool(artifact.get("markdown_url") or markdown),
            "has_docx": bool(artifact.get("docx_url") or artifact.get("docx_filename")),
            "has_srt": bool(
                artifact.get("srt_url") or artifact.get("srt_filename") or srt
            ),
        }

    if resource_type == "video_animation":
        return {
            "duration_seconds": artifact.get("duration_seconds", 0),
            "render_success": bool(artifact.get("render_success")),
            "mp4_available": bool(
                artifact.get("mp4_available") or artifact.get("mp4_url")
            ),
            "has_html": bool(artifact.get("html_url") or artifact.get("html_filename")),
            "has_json": bool(artifact.get("json_url") or artifact.get("json_filename")),
            "has_srt": bool(artifact.get("srt_url") or artifact.get("srt_filename")),
            "has_mp4": bool(artifact.get("mp4_url") and artifact.get("render_success")),
        }

    if resource_type == "study_plan":
        raw_document = artifact.get("document")
        document: dict[str, Any] = (
            raw_document if isinstance(raw_document, dict) else {}
        )
        markdown = (
            state_updates.get("study_plan_markdown")
            or document.get("markdown")
            or artifact.get("markdown")
            or ""
        )
        return {
            "has_markdown": bool(
                document.get("markdown_url") or document.get("markdown") or markdown
            ),
            "has_docx": bool(document.get("docx_url") or document.get("docx_filename")),
            "has_document": bool(document),
        }

    return {}


def _resource_summary(result: dict) -> dict:
    return {
        "resource_type": result.get("resource_type"),
        "subjects": result.get("subjects") or [],
        "topic_ids": result.get("topic_ids") or [],
        "status": result.get("status"),
        "title": result.get("title"),
        "artifact": result.get("artifact") or {},
        "artifacts": result.get("artifacts") or [],
        "message_preview": result.get("message_preview") or "",
        "error_type": result.get("error_type"),
        "error_message_sanitized": result.get("error_message_sanitized"),
        "elapsed_ms": result.get("elapsed_ms"),
        "metrics": _resource_metrics(result),
        "validation": result.get("validation"),
    }


@dataclass(frozen=True, slots=True)
class _ResourceBundleInputs:
    requested: tuple[str, ...]
    results: tuple[dict[str, Any], ...]
    complete_successes: tuple[dict[str, Any], ...]
    partial_successes: tuple[dict[str, Any], ...]
    renderable_results: tuple[dict[str, Any], ...]
    failures: tuple[dict[str, Any], ...]
    blocked_resources: tuple[dict[str, Any], ...]
    status: str
    message: str


def _collect_resource_bundle_inputs(state: LearningState) -> _ResourceBundleInputs:
    evidence_requested = list(state.get("evidence_requested_resource_types") or [])
    requested = (
        evidence_requested
        if evidence_requested
        else normalize_requested_resource_types(
            state.get("requested_resource_types") or [],
            state.get("requested_resource_type") or "",
        )
    )
    results = tuple(
        result
        for result in state.get("resource_branch_results") or []
        if isinstance(result, dict) and result.get("resource_type")
    )
    complete_successes = tuple(
        result for result in results if result.get("status") == "success"
    )
    partial_successes = tuple(
        result for result in results if result.get("status") == "partial_success"
    )
    renderable_results = (*complete_successes, *partial_successes)
    failures = tuple(result for result in results if result.get("status") == "failed")
    readiness = tuple(
        item
        for item in (state.get("resource_evidence_readiness") or [])
        if isinstance(item, dict)
    )
    blocked_resources = tuple(
        {
            "resource_type": item.get("resource_type"),
            "status": "blocked_insufficient_evidence",
            "blocked_requirement_ids": item.get("blocked_requirement_ids"),
            "reason_code": item.get("reason_code"),
        }
        for item in readiness
        if item.get("readiness_state") == "blocked_insufficient_evidence"
    )

    if partial_successes or (renderable_results and (failures or blocked_resources)):
        status = "partial_success"
    elif complete_successes:
        status = "success"
    elif blocked_resources and not failures:
        status = "blocked_insufficient_evidence"
    elif requested or results:
        status = "failed"
    else:
        status = "skipped"

    message = _append_blocked_resource_message(
        _compose_bundle_message(
            status,
            list(renderable_results),
            list(failures),
        ),
        status=status,
        blocked_resources=list(blocked_resources),
    )
    return _ResourceBundleInputs(
        requested=tuple(requested),
        results=results,
        complete_successes=complete_successes,
        partial_successes=partial_successes,
        renderable_results=renderable_results,
        failures=failures,
        blocked_resources=blocked_resources,
        status=status,
        message=message,
    )


def _resource_final_terminal_status(status: str) -> ResourceFinalV3TerminalStatus:
    if status == "success":
        return "success"
    if status == "partial_success":
        return "partial_success"
    if status == "failed":
        return "failed"
    if status == "blocked_insufficient_evidence":
        return "controlled_stop"
    raise RuntimeError(f"resource bundle has unsupported terminal status {status!r}")


def _build_resource_final_for_bundle(
    state: LearningState,
    *,
    inputs: _ResourceBundleInputs,
    recommendations: tuple[ResourceFinalV3Recommendation, ...],
    summary: str,
) -> ResourceFinalV3:
    if inputs.status == "skipped":
        raise RuntimeError("a skipped resource bundle cannot produce Resource Final V3")
    thread_id = state.get("thread_id")
    request_id = state.get("request_id")
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise RuntimeError("resource bundle requires a non-blank thread_id")
    if not isinstance(request_id, str) or not request_id.strip():
        raise RuntimeError("resource bundle requires a non-blank request_id")
    return build_resource_final_v3_from_bundle(
        thread_id=thread_id,
        request_id=request_id,
        requested_resource_types=inputs.requested,
        terminal_status=_resource_final_terminal_status(inputs.status),
        branch_results=inputs.results,
        blocked_resources=inputs.blocked_resources,
        recommendations=recommendations,
        summary=summary,
    )


def _recommendation_context_for_bundle(
    state: LearningState,
    *,
    inputs: _ResourceBundleInputs,
) -> tuple[RecommendationResourceContextV1, ...]:
    if inputs.status == "skipped":
        raise RuntimeError("candidate resource flow reached a skipped bundle")
    subject = state.get("subject")
    if (
        not isinstance(subject, str)
        or not subject.strip()
        or subject != subject.strip()
    ):
        raise LearningGuidanceContractError(
            code="invalid_recommendation_resource_subject",
            reason="candidate recommendation context requires an exact subject",
        )
    assignments = _candidate_resource_assignments_from_state(
        state,
        required=True,
    )
    if assignments is None:
        raise AssertionError("required candidate assignments parser returned None")
    if automatic_recommendation_scope_reason(state, subject=subject) is not None:
        return ()
    assignment_by_resource = {item.resource_type: item for item in assignments}
    provisional = _build_resource_final_for_bundle(
        state,
        inputs=inputs,
        recommendations=(),
        summary=inputs.message,
    )
    results_by_type = {
        str(result.get("resource_type")): result for result in inputs.renderable_results
    }
    contexts: list[RecommendationResourceContextV1] = []
    for resource in provisional.resources:
        result = results_by_type.get(resource.kind)
        if result is None:
            raise LearningGuidanceContractError(
                code="missing_recommendation_resource_binding",
                reason="verified resource lacks its branch topic binding",
            )
        assignment = assignment_by_resource.get(resource.kind)
        if assignment is None:
            raise LearningGuidanceContractError(
                code="missing_recommendation_resource_assignment",
                reason="verified resource lacks its original evidence assignment",
            )
        subjects = tuple(result.get("subjects") or ())
        topic_ids = tuple(result.get("topic_ids") or ())
        if subjects != assignment.subjects or topic_ids != assignment.topic_ids:
            raise LearningGuidanceContractError(
                code="resource_branch_assignment_mismatch",
                reason="resource branch topic binding differs from its assignment",
            )
        if assignment.subjects != (subject,) or len(assignment.topic_ids) != 1:
            raise LearningGuidanceContractError(
                code="invalid_recommendation_resource_binding",
                reason=(
                    "automatic recommendation requires one exact subject/topic "
                    "binding per resource"
                ),
            )
        contexts.append(
            RecommendationResourceContextV1(
                resource_id=resource.resource_id,
                resource_type=resource.kind,
                subject=subject,
                topic_id=assignment.topic_ids[0],
                title=resource.title,
            )
        )
    return tuple(contexts)


def _validate_recommendation_context_for_bundle(
    state: LearningState,
    *,
    inputs: _ResourceBundleInputs,
) -> None:
    expected = _recommendation_context_for_bundle(state, inputs=inputs)
    raw = state.get("recommendation_resource_context")
    if not isinstance(raw, (list, tuple)):
        raise LearningGuidanceContractError(
            code="invalid_recommendation_resource_context",
            reason="candidate finalizer requires a recommendation context sequence",
        )
    try:
        actual = tuple(
            RecommendationResourceContextV1.model_validate(item) for item in raw
        )
    except (TypeError, ValueError) as exc:
        raise LearningGuidanceContractError(
            code="invalid_recommendation_resource_context",
            reason="candidate recommendation context violates its strict schema",
        ) from exc
    if actual != expected:
        raise LearningGuidanceContractError(
            code="recommendation_resource_context_mismatch",
            reason="recommendation context differs from verified bundle resources",
        )


@traced_node
async def resource_bundle_aggregator(state: LearningState) -> dict[str, object]:
    """Build only verified recommendation context; never emit a terminal result."""

    inputs = _collect_resource_bundle_inputs(state)
    contexts = _recommendation_context_for_bundle(state, inputs=inputs)
    return {
        "recommendation_resource_context": [
            item.model_dump(mode="json") for item in contexts
        ]
    }


async def _resource_bundle_output_impl(
    state: LearningState,
    *,
    recommendations: tuple[ResourceFinalV3Recommendation, ...],
    recommendation_status_message: str,
) -> dict:
    inputs = _collect_resource_bundle_inputs(state)
    requested = list(inputs.requested)
    results = list(inputs.results)
    complete_successes = list(inputs.complete_successes)
    partial_successes = list(inputs.partial_successes)
    renderable_results = list(inputs.renderable_results)
    failures = list(inputs.failures)
    blocked_resources = list(inputs.blocked_resources)
    status = inputs.status

    state_updates: dict = {}
    for result in renderable_results:
        state_updates.update(result.get("state_updates") or {})

    renderable_count = sum(
        int((result.get("validation") or {}).get("renderable_count") or 0)
        for result in renderable_results
    )
    downloadable_count = sum(
        int((result.get("validation") or {}).get("downloadable_count") or 0)
        for result in renderable_results
    )

    bundle = {
        "type": "resource_bundle",
        "status": status,
        "requested_resource_types": requested,
        "success_count": len(complete_successes),
        "partial_success_count": len(partial_successes),
        "failed_count": len(failures),
        "blocked_count": len(blocked_resources),
        "renderable_resource_count": len(renderable_results),
        "renderable_count": renderable_count,
        "downloadable_count": downloadable_count,
        "resources": [_resource_summary(result) for result in renderable_results],
        "errors": [_resource_summary(result) for result in failures],
        "blocked_resources": blocked_resources,
    }
    debug = dict(state.get("resource_generation_debug") or {})
    stages = list(debug.get("stages") or [])
    stages.append(
        {
            "stage": "resource_generation.bundle.complete",
            "status": status,
            "success_count": len(complete_successes),
            "partial_success_count": len(partial_successes),
            "failed_count": len(failures),
            "blocked_count": len(blocked_resources),
            "renderable_resource_count": len(renderable_results),
            "resource_count": len(results),
        }
    )
    debug.update(
        {
            "status": status,
            "success_count": len(complete_successes),
            "partial_success_count": len(partial_successes),
            "failed_count": len(failures),
            "partial_success": status == "partial_success",
            "branch_results": [_resource_summary(result) for result in results],
            "stages": stages,
        }
    )
    message = (
        f"{recommendation_status_message}\n\n{inputs.message}"
        if recommendation_status_message and inputs.message
        else recommendation_status_message or inputs.message
    )
    bundle["message"] = message
    resource_final_v3: dict[str, Any] = {}
    if status != "skipped":
        resource_final_v3 = _build_resource_final_for_bundle(
            state,
            inputs=inputs,
            recommendations=recommendations,
            summary=message,
        ).model_dump(mode="json")
    artifact_updates: dict[str, Any] = {}
    influence_entries = [
        entry
        for result in results
        for entry in (result.get("context_influence_entries") or [])
        if isinstance(entry, dict)
    ]
    influence_update = combine_influence_updates(
        state=state,
        updates=(),
        entries=influence_entries,
    )
    if renderable_results:
        try:
            workspace_successes = [
                {**result, "metrics": _resource_metrics(result)}
                for result in renderable_results
            ]
            artifact_updates = build_workspace_artifact_update(
                state,
                workspace_successes,
            )
            workspace_payload = workspace_trace_payload(
                artifact_updates.get("task_workspace") or {}
            )
            emit_a3_trace(
                logger,
                "task_workspace.update_planned",
                workspace_payload,
                state=state,
                env_flag="LOG_A3_TRACE",
            )
            emit_a3_trace(
                logger,
                "resource_artifacts.indexed",
                workspace_payload,
                state=state,
                env_flag="LOG_A3_TRACE",
            )
            emit_a3_trace(
                logger,
                "task_workspace.updated",
                workspace_payload,
                state=state,
                env_flag="LOG_A3_TRACE",
            )
            workspace_events = list(artifact_updates.get("workspace_events") or [])
            workspace_events.extend(
                [
                    {
                        "stage": "task_workspace.updated",
                        **workspace_payload,
                    }
                ]
            )
            artifact_updates["workspace_events"] = workspace_events
        except Exception as exc:
            failure_payload = {
                "thread_id": state.get("thread_id", ""),
                "request_id": state.get("request_id", ""),
                "updated_sources": ["resource_bundle_output"],
                "diagnostics": [sanitize_error_message(exc, max_chars=160)],
            }
            emit_a3_trace(
                logger,
                "task_workspace.update_failed",
                failure_payload,
                state=state,
                env_flag="LOG_A3_TRACE",
            )
            artifact_updates = {
                "workspace_events": [
                    {"stage": "task_workspace.update_failed", **failure_payload}
                ]
            }
    emit_a3_trace(
        logger,
        "resource_generation.bundle.complete",
        {
            "status": status,
            "requested_resource_types": requested,
            "success_count": len(complete_successes),
            "partial_success_count": len(partial_successes),
            "failed_count": len(failures),
            "renderable_resource_count": len(renderable_results),
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        **state_updates,
        **artifact_updates,
        "resource_bundle_artifact": bundle,
        "resource_final_v3": resource_final_v3,
        "resource_generation_debug": debug,
        "resource_generation_status": status,
        "context_influence_ledger": influence_update,
        "messages": [AIMessage(content=message)] if message else [],
    }


@traced_node
async def resource_bundle_output(state: LearningState) -> dict:
    """Finalize the currently served graph without candidate recommendations."""

    return await _resource_bundle_output_impl(
        state,
        recommendations=(),
        recommendation_status_message="",
    )


@traced_node
async def resource_bundle_output_with_recommendations(
    state: LearningState,
    *,
    runtime: LearningGuidanceRuntime,
) -> dict:
    """Finalize the candidate graph after one strict automatic recommendation."""

    inputs = _collect_resource_bundle_inputs(state)
    _validate_recommendation_context_for_bundle(state, inputs=inputs)
    output = resource_recommendation_output_for_runtime_from_state(
        state,
        expected_mode="automatic_after_generation",
        runtime=runtime,
    )
    return await _resource_bundle_output_impl(
        state,
        recommendations=resource_final_recommendations(output),
        recommendation_status_message=recommendation_public_status_message(output),
    )


def make_resource_bundle_output_with_recommendations_node(
    runtime: LearningGuidanceRuntime,
) -> Callable[[LearningState], Awaitable[dict]]:
    """Bind candidate finalization to the current guidance runtime."""

    if not isinstance(runtime, LearningGuidanceRuntime):
        raise TypeError("runtime must be LearningGuidanceRuntime")

    async def candidate_resource_bundle_output(state: LearningState) -> dict:
        return await resource_bundle_output_with_recommendations(
            state,
            runtime=runtime,
        )

    return candidate_resource_bundle_output
