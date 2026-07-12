"""Resource-aware, bounded evidence retrieval for the Parent-Child candidate graph."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
import logging
import time
from typing import Literal, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from src.config import load_prompt
from src.config.evidence_orchestration_config import (
    EvidenceOrchestrationConfig,
    ResourceEvidenceProfilesConfig,
)
from src.config.evidence_orchestration_contracts import (
    DuplicateRetrievalSignatureError,
    EvidenceBudgetExceededError,
    EvidenceLedgerEntry,
    EvidenceOrchestrationContractError,
    EvidenceRepairPlan,
    EvidenceRequirement,
    EvidenceRequirementDraftBatch,
    EvidenceSourceType,
    RequirementCoverageBatch,
    ResourceReadiness,
    RetrievalPriority,
    RetrievalTask,
    build_retrieval_task,
    compile_evidence_requirement_batch,
    derive_resource_evidence_assignments,
    derive_resource_readiness,
    make_evidence_id,
    make_repair_plan_signature,
    validate_evidence_ledger,
    validate_requirement_coverage,
    validate_requirement_inventory,
    validate_retrieval_tasks,
)
from src.graph.academic import execute_validated_web_research_tasks
from src.graph.academic import SearchQueryRewriteOutput
from src.graph.evidence import EvidenceCandidate
from src.graph.parent_child_nodes import (
    ParentChildGraphContractError,
    ParentChildGraphRuntime,
    make_parent_child_rag_node,
)
from src.graph.state import CONTEXT_CLEAR, LearningState
from src.graph.web_research import WebResearchTask, WebSourceSummaryBatch
from src.llm.structured_output import (
    get_llm_output_mode,
    get_max_raw_chars,
    invoke_structured_llm,
)
from src.observability.evidence_trace import (
    EVIDENCE_TRACE_SCHEMA_VERSION,
    emit_evidence_trace,
)
from src.rag.course_catalog import get_available_subjects_from_data
from src.rag.parent_child.handoff import LocalEvidenceRef, parent_context_items
from src.rag.parent_child.retrieval import MultiBranchHybridChildResult
from src.resource_contracts import (
    ResourceType,
    normalize_requested_resource_types,
)

logger = logging.getLogger(__name__)

_RESOURCE_TYPES_ADAPTER = TypeAdapter(tuple[ResourceType, ...])


class EvidenceOrchestrationRuntimeError(RuntimeError):
    """Candidate orchestration failed a typed runtime invariant."""

    def __init__(self, *, code: str, reason: str) -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"{code}: {reason}")


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _required_string_list(
    document: dict[str, object],
    field_name: str,
) -> list[str]:
    value = document.get(field_name)
    if not isinstance(value, list):
        raise ParentChildGraphContractError(
            f"hydrated evidence field {field_name!r} must be a list"
        )
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ParentChildGraphContractError(
                f"hydrated evidence field {field_name!r} must contain non-blank strings"
            )
        items.append(item)
    return items


def _required_numeric_field(
    document: dict[str, object],
    field_name: str,
) -> float:
    value = document.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParentChildGraphContractError(
            f"hydrated evidence field {field_name!r} must be numeric"
        )
    return float(value)


def _required_state_count(state: LearningState, field_name: str) -> int:
    value = state.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceOrchestrationRuntimeError(
            code="invalid_evidence_state_count",
            reason=f"{field_name} must be an explicit non-negative integer",
        )
    return value


@dataclass(frozen=True, slots=True)
class EvidenceOrchestrationRuntime:
    """All explicit dependencies for the joint Parent-Child/evidence candidate."""

    parent_child: ParentChildGraphRuntime
    policy: EvidenceOrchestrationConfig
    profiles: ResourceEvidenceProfilesConfig
    web_timeout_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.parent_child, ParentChildGraphRuntime):
            raise TypeError("parent_child must be ParentChildGraphRuntime")
        if not isinstance(self.policy, EvidenceOrchestrationConfig):
            raise TypeError("policy must be EvidenceOrchestrationConfig")
        if not isinstance(self.profiles, ResourceEvidenceProfilesConfig):
            raise TypeError("profiles must be ResourceEvidenceProfilesConfig")
        if self.web_timeout_seconds <= 0:
            raise ValueError("web_timeout_seconds must be positive")

    @property
    def profile_fingerprint(self) -> str:
        return _digest(self.profiles.model_dump(mode="json"))

    @property
    def orchestration_fingerprint(self) -> str:
        prompt_fingerprints = {
            name: _text_digest(load_prompt(name))
            for name in (
                "search_query_rewriter",
                "resource_evidence_planner",
                "requirement_evidence_judge",
                "web_source_summarizer",
            )
        }
        return _digest(
            {
                "schema_version": "resource_evidence_graph_runtime_v1",
                "parent_child_handoff_fingerprint": (
                    self.parent_child.graph_handoff_fingerprint
                ),
                "policy": self.policy.model_dump(mode="json"),
                "profile_fingerprint": self.profile_fingerprint,
                "prompt_fingerprints": prompt_fingerprints,
                "structured_schemas": {
                    "query_rewrite": SearchQueryRewriteOutput.model_json_schema(),
                    "requirement_plan": (
                        EvidenceRequirementDraftBatch.model_json_schema()
                    ),
                    "requirement_coverage": (
                        RequirementCoverageBatch.model_json_schema()
                    ),
                    "web_source_summary": WebSourceSummaryBatch.model_json_schema(),
                },
            }
        )

    @property
    def candidate_bundle_fingerprint(self) -> str:
        """Fingerprint index, policies, profiles, prompts, and schemas together."""

        return self.orchestration_fingerprint


class EvidenceCandidateRecord(BaseModel):
    """Strict internal candidate record; trace code never emits its content."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    round_index: int = Field(ge=0)
    task_id: str = Field(min_length=1)
    requirement_id: str = Field(min_length=1)
    resource_type: ResourceType
    subject: str = Field(min_length=1)
    source_type: Literal["local_rag", "web"]
    evidence_id: str = Field(min_length=1)
    candidate_ref: str = Field(min_length=1)
    candidate_snapshot_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_identity_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate: EvidenceCandidate
    original: dict[str, object]

    @model_validator(mode="after")
    def validate_binding(self) -> "EvidenceCandidateRecord":
        if self.candidate.evidence_id != self.evidence_id:
            raise ValueError("candidate evidence id must match the record")
        if self.candidate.source_type != self.source_type:
            raise ValueError("candidate source type must match the record")
        if self.candidate.subject != self.subject:
            raise ValueError("candidate subject must match the record")
        metadata = self.candidate.metadata
        if (
            metadata.get("task_id") != self.task_id
            or metadata.get("requirement_id") != self.requirement_id
            or metadata.get("candidate_ref") != self.candidate_ref
        ):
            raise ValueError("candidate metadata must preserve task binding")
        return self


class EvidenceSourceOutcome(BaseModel):
    """Per-task successful source outcome, including an explicit empty result."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    round_index: int = Field(ge=0)
    task_id: str = Field(min_length=1)
    requirement_id: str = Field(min_length=1)
    source_type: Literal["local_rag", "web"]
    status: Literal["completed", "empty"]
    candidate_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_status_count(self) -> "EvidenceSourceOutcome":
        if self.status == "empty" and self.candidate_count != 0:
            raise ValueError("empty source outcome cannot contain candidates")
        if self.status == "completed" and self.candidate_count == 0:
            raise ValueError("completed source outcome requires candidates")
        return self


class ParentChildRoundSnapshot(BaseModel):
    """One child-only retrieval snapshot retained until terminal hydration."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    round_index: int = Field(ge=0)
    retrieval_result: dict[str, object]
    local_refs: tuple[dict[str, object], ...]
    snapshot_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("local_refs", mode="before")
    @classmethod
    def freeze_local_refs(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_fingerprint(self) -> "ParentChildRoundSnapshot":
        expected = _digest(
            {
                "round_index": self.round_index,
                "retrieval_result": self.retrieval_result,
                "local_refs": self.local_refs,
            }
        )
        if self.snapshot_fingerprint != expected:
            raise ValueError("parent-child round snapshot fingerprint mismatch")
        return self


def _last_human_query(state: LearningState) -> str:
    for message in reversed(state.get("messages") or []):
        if isinstance(message, HumanMessage):
            value = str(message.content or "").strip()
        elif isinstance(message, dict) and message.get("type") == "human":
            value = str(message.get("content") or "").strip()
        else:
            continue
        if value:
            return value
    raise EvidenceOrchestrationRuntimeError(
        code="missing_user_query",
        reason="resource evidence planning requires a current human query",
    )


def _requested_resources(state: LearningState) -> tuple[ResourceType, ...]:
    raw_items = [
        str(item).strip()
        for item in (state.get("requested_resource_types") or [])
        if str(item).strip()
    ]
    singular = str(state.get("requested_resource_type") or "").strip()
    if singular:
        raw_items.append(singular)
    unique_raw = tuple(dict.fromkeys(raw_items))
    normalized = tuple(normalize_requested_resource_types(unique_raw))
    if not unique_raw or normalized != unique_raw:
        raise EvidenceOrchestrationRuntimeError(
            code="noncanonical_requested_resources",
            reason="candidate input must contain non-empty canonical resource types",
        )
    return _RESOURCE_TYPES_ADAPTER.validate_python(normalized)


def _requested_subjects(
    state: LearningState,
    runtime: EvidenceOrchestrationRuntime,
) -> tuple[str, ...]:
    plan = state.get("retrieval_plan") or []
    subjects = tuple(
        dict.fromkeys(
            str(item.get("subject") or "").strip()
            for item in plan
            if isinstance(item, dict) and str(item.get("subject") or "").strip()
        )
    )
    if not subjects:
        raise EvidenceOrchestrationRuntimeError(
            code="missing_subject_plan",
            reason="candidate requires a non-empty validated retrieval plan",
        )
    available = set(runtime.parent_child.available_subjects)
    catalog_subjects = set(get_available_subjects_from_data())
    if any(
        subject not in available or subject not in catalog_subjects
        for subject in subjects
    ):
        raise EvidenceOrchestrationRuntimeError(
            code="unavailable_subject",
            reason="candidate subject must exist in the pinned index and course catalog",
        )
    return subjects


def _render_prompt(name: str, values: dict[str, object]) -> str:
    template = load_prompt(name)
    return template.format(**values)


def _planner_business_validation(
    parsed: BaseModel,
    *,
    resources: tuple[ResourceType, ...],
    subjects: tuple[str, ...],
    runtime: EvidenceOrchestrationRuntime,
) -> str:
    if not isinstance(parsed, EvidenceRequirementDraftBatch):
        return "parsed result is not EvidenceRequirementDraftBatch"
    try:
        requirements = compile_evidence_requirement_batch(parsed)
        validate_requirement_inventory(
            requested_resource_types=resources,
            requested_subjects=subjects,
            canonical_subjects=set(runtime.parent_child.available_subjects),
            requirements=requirements,
            profiles=runtime.profiles,
            config=runtime.policy,
        )
    except (EvidenceOrchestrationContractError, ValueError) as exc:
        return str(exc)
    return ""


def _initial_sources(
    requirement: EvidenceRequirement,
) -> tuple[EvidenceSourceType, ...]:
    if requirement.source_policy in {"local_only", "local_then_web_on_gap"}:
        return ("local_rag",)
    if requirement.source_policy == "web_only":
        return ("web",)
    if requirement.source_policy == "local_and_web":
        return ("local_rag", "web")
    raise EvidenceOrchestrationRuntimeError(
        code="unknown_source_policy",
        reason="compiled requirement has an unsupported source policy",
    )


def _priority(
    requirement: EvidenceRequirement,
    runtime: EvidenceOrchestrationRuntime,
) -> RetrievalPriority:
    return (
        runtime.policy.required_task_priority
        if requirement.criticality == "required"
        else runtime.policy.supporting_task_priority
    )


def _build_initial_tasks(
    requirements: tuple[EvidenceRequirement, ...],
    runtime: EvidenceOrchestrationRuntime,
) -> tuple[RetrievalTask, ...]:
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
    required_action_count = sum(
        len(_initial_sources(requirement))
        for requirement in ordered
        if requirement.criticality == "required"
    )
    if required_action_count > runtime.policy.max_total_search_tasks:
        raise EvidenceBudgetExceededError(
            code="required_initial_search_budget_exceeded",
            reason="required profile needs cannot fit the total search-task budget",
        )

    tasks: list[RetrievalTask] = []
    for requirement in ordered:
        for source_type in _initial_sources(requirement):
            if len(tasks) >= runtime.policy.max_search_tasks_per_round:
                break
            tasks.append(
                build_retrieval_task(
                    requirement=requirement,
                    source_type=source_type,
                    query=requirement.query_intent,
                    purpose=requirement.acceptance_criteria,
                    priority=_priority(requirement, runtime),
                    round_index=0,
                    result_limit=runtime.policy.max_results_per_task,
                )
            )
        if len(tasks) >= runtime.policy.max_search_tasks_per_round:
            break
    if not tasks:
        raise EvidenceOrchestrationRuntimeError(
            code="empty_initial_task_plan",
            reason="validated requirements produced no initial retrieval tasks",
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
    return tuple(tasks)


def make_resource_evidence_planner_node(
    runtime: EvidenceOrchestrationRuntime,
) -> Callable[[LearningState], Awaitable[dict]]:
    """Create the strict resource/profile/subject evidence planner node."""

    async def resource_evidence_planner(state: LearningState) -> dict:
        resources = _requested_resources(state)
        subjects = _requested_subjects(state, runtime)
        question = _last_human_query(state)
        profiles = tuple(
            runtime.profiles.profile_for(resource).model_dump(mode="json")
            for resource in resources
        )
        prompt = _render_prompt(
            "resource_evidence_planner",
            {
                "question": question,
                "learning_goal": str(state.get("learning_goal") or "").strip(),
                "requested_resource_types_json": json.dumps(
                    resources, ensure_ascii=False
                ),
                "subjects_json": json.dumps(subjects, ensure_ascii=False),
                "retrieval_plan_json": json.dumps(
                    state.get("retrieval_plan") or [], ensure_ascii=False
                ),
                "profiles_json": json.dumps(profiles, ensure_ascii=False),
                "max_requirements": runtime.policy.max_requirements_per_request,
            },
        )
        try:
            structured = await invoke_structured_llm(
                node_name="resource_evidence_planner",
                llm_node="query_rewrite",
                schema=EvidenceRequirementDraftBatch,
                messages=[
                    SystemMessage(
                        content=(
                            "Return only the strict evidence requirement schema. "
                            "Configured profile fields are immutable input contracts."
                        )
                    ),
                    HumanMessage(content=prompt),
                ],
                output_mode=get_llm_output_mode("resource_evidence_planner"),
                fallback_modes=[],
                business_validator=lambda parsed: _planner_business_validation(
                    parsed,
                    resources=resources,
                    subjects=subjects,
                    runtime=runtime,
                ),
                state=state,
                max_raw_chars=get_max_raw_chars("resource_evidence_planner"),
            )
        except Exception as exc:
            emit_evidence_trace(
                logger,
                {
                    "schema_version": EVIDENCE_TRACE_SCHEMA_VERSION,
                    "stage": "evidence_orchestration.failed",
                    "status": "failed",
                    "round_index": 0,
                    "source": "orchestration",
                    "error_type": type(exc).__name__,
                    "reason_code": "requirement_planner_failed",
                    "budget_used_tasks": 0,
                    "budget_remaining_tasks": (runtime.policy.max_total_search_tasks),
                },
                state=state,
            )
            raise
        parsed = structured.parsed
        if not isinstance(parsed, EvidenceRequirementDraftBatch):
            raise TypeError("planner result is not EvidenceRequirementDraftBatch")
        requirements = compile_evidence_requirement_batch(parsed)
        validate_requirement_inventory(
            requested_resource_types=resources,
            requested_subjects=subjects,
            canonical_subjects=set(runtime.parent_child.available_subjects),
            requirements=requirements,
            profiles=runtime.profiles,
            config=runtime.policy,
        )
        tasks = _build_initial_tasks(requirements, runtime)
        emit_evidence_trace(
            logger,
            {
                "schema_version": EVIDENCE_TRACE_SCHEMA_VERSION,
                "stage": "evidence_orchestration.plan.accepted",
                "orchestration_fingerprint": runtime.orchestration_fingerprint,
                "profile_fingerprint": runtime.profile_fingerprint,
                "requirement_count": len(requirements),
                "resource_count": len(resources),
                "subject_count": len(subjects),
                "budget_max_rounds": runtime.policy.max_supplement_rounds,
                "budget_max_tasks": runtime.policy.max_total_search_tasks,
            },
            state=state,
        )
        return {
            "evidence_orchestration_fingerprint": runtime.orchestration_fingerprint,
            "evidence_requested_resource_types": list(resources),
            "evidence_requested_subjects": list(subjects),
            "evidence_requirements": [
                item.model_dump(mode="json") for item in requirements
            ],
            "evidence_current_round": 0,
            "evidence_current_tasks": [item.model_dump(mode="json") for item in tasks],
            "evidence_all_tasks": [item.model_dump(mode="json") for item in tasks],
            "evidence_retrieval_signatures": [
                item.retrieval_signature for item in tasks
            ],
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

    return resource_evidence_planner


def make_rag_generation_router_node(
    runtime: EvidenceOrchestrationRuntime,
) -> Callable[[LearningState], dict]:
    """Create an explicit candidate marker; served-primary selection stays external."""

    def rag_generation_router(state: LearningState) -> dict:
        del state
        return {
            "rag_generation_route": "resource_evidence_parent_child",
            "evidence_orchestration_fingerprint": runtime.orchestration_fingerprint,
        }

    return rag_generation_router


def make_retrieval_round_router_node(
    runtime: EvidenceOrchestrationRuntime,
) -> Callable[[LearningState], dict]:
    """Create a validator for one bounded round before static source fan-out."""

    def retrieval_round_router(state: LearningState) -> dict:
        round_index = _required_state_count(state, "evidence_current_round")
        tasks = tuple(
            RetrievalTask.model_validate(item)
            for item in (state.get("evidence_current_tasks") or [])
        )
        requirements = tuple(
            EvidenceRequirement.model_validate(item)
            for item in (state.get("evidence_requirements") or [])
        )
        all_tasks = tuple(
            RetrievalTask.model_validate(item)
            for item in (state.get("evidence_all_tasks") or [])
        )
        if not tasks or any(task.round_index != round_index for task in tasks):
            raise EvidenceOrchestrationRuntimeError(
                code="invalid_current_round_tasks",
                reason="retrieval router requires non-empty tasks for its exact round",
            )
        requirement_ids = {item.requirement_id for item in requirements}
        if any(task.requirement_id not in requirement_ids for task in tasks):
            raise EvidenceOrchestrationRuntimeError(
                code="unknown_round_requirement",
                reason="round task references an unknown requirement",
            )
        validate_retrieval_tasks(
            tasks=tasks,
            requirements=requirements,
            config=runtime.policy,
            round_index=round_index,
            existing_total_search_tasks=len(all_tasks) - len(tasks),
            prior_retrieval_signatures={
                task.retrieval_signature
                for task in all_tasks
                if task.round_index < round_index
            },
            local_then_web_gap_requirement_ids={
                item.requirement_id
                for item in requirements
                if round_index > 0 and item.source_policy == "local_then_web_on_gap"
            },
        )
        local_count = sum(task.source_type == "local_rag" for task in tasks)
        web_count = sum(task.source_type == "web" for task in tasks)
        used = len(all_tasks)
        emit_evidence_trace(
            logger,
            {
                "schema_version": EVIDENCE_TRACE_SCHEMA_VERSION,
                "stage": "evidence_orchestration.round.started",
                "round_index": round_index,
                "task_count": len(tasks),
                "local_task_count": local_count,
                "web_task_count": web_count,
                "budget_used_tasks": used,
                "budget_remaining_tasks": (
                    runtime.policy.max_total_search_tasks - used
                ),
            },
            state=state,
        )
        return {"evidence_orchestration_status": "retrieving"}

    return retrieval_round_router


def _query_batch_fingerprint(tasks: Sequence[RetrievalTask]) -> str:
    return _digest(
        [
            {
                "task_id": task.task_id,
                "source_type": task.source_type,
                "query_fingerprint": task.query_fingerprint,
            }
            for task in tasks
        ]
    )


def _candidate_record(
    *,
    candidate: EvidenceCandidate,
    original: dict[str, object],
    task: RetrievalTask,
) -> EvidenceCandidateRecord:
    candidate_ref = candidate.evidence_id
    source_identity = str(
        candidate.metadata.get("source_id")
        or candidate.metadata.get("canonical_url")
        or candidate_ref
    )
    source_identity_fingerprint = _text_digest(source_identity)
    content_fingerprint = _text_digest(candidate.content_preview)
    evidence_id = make_evidence_id(
        requirement_id=task.requirement_id,
        source_type=task.source_type,
        source_identity_fingerprint=source_identity_fingerprint,
        content_fingerprint=content_fingerprint,
    )
    metadata = {
        **candidate.metadata,
        "task_id": task.task_id,
        "requirement_id": task.requirement_id,
        "resource_type": task.resource_type,
        "round_index": task.round_index,
        "candidate_ref": candidate_ref,
        "query_fingerprint": task.query_fingerprint,
    }
    rebound = candidate.model_copy(
        update={
            "evidence_id": evidence_id,
            "role": task.requirement_id,
            "purpose": task.purpose,
            "metadata": metadata,
        }
    )
    snapshot = rebound.model_dump(mode="json")
    return EvidenceCandidateRecord(
        round_index=task.round_index,
        task_id=task.task_id,
        requirement_id=task.requirement_id,
        resource_type=task.resource_type,
        subject=task.subject,
        source_type=task.source_type,
        evidence_id=evidence_id,
        candidate_ref=candidate_ref,
        candidate_snapshot_fingerprint=_digest(snapshot),
        source_identity_fingerprint=source_identity_fingerprint,
        content_fingerprint=content_fingerprint,
        candidate=rebound,
        original={**original, "evidence_id": evidence_id},
    )


def _source_outcomes(
    tasks: Sequence[RetrievalTask],
    records: Sequence[EvidenceCandidateRecord],
) -> tuple[EvidenceSourceOutcome, ...]:
    counts = Counter(record.task_id for record in records)
    return tuple(
        EvidenceSourceOutcome(
            round_index=task.round_index,
            task_id=task.task_id,
            requirement_id=task.requirement_id,
            source_type=task.source_type,
            status="completed" if counts[task.task_id] else "empty",
            candidate_count=counts[task.task_id],
        )
        for task in tasks
    )


def _emit_source_trace(
    *,
    state: LearningState,
    source: Literal["local", "web"],
    round_index: int,
    tasks: Sequence[RetrievalTask],
    records: Sequence[EvidenceCandidateRecord],
    latency_ms: int,
) -> None:
    fingerprint = _query_batch_fingerprint(tasks)
    if records:
        event = {
            "schema_version": EVIDENCE_TRACE_SCHEMA_VERSION,
            "stage": "evidence_orchestration.source.completed",
            "round_index": round_index,
            "source": source,
            "status": "completed",
            "task_count": len(tasks),
            "query_batch_fingerprint": fingerprint,
            "candidate_count": len(records),
            "latency_ms": latency_ms,
        }
    else:
        event = {
            "schema_version": EVIDENCE_TRACE_SCHEMA_VERSION,
            "stage": "evidence_orchestration.source.empty",
            "round_index": round_index,
            "source": source,
            "status": "empty",
            "task_count": len(tasks),
            "query_batch_fingerprint": fingerprint,
            "latency_ms": latency_ms,
            "reason_code": ("no_tasks_assigned" if not tasks else "no_candidates"),
        }
    emit_evidence_trace(logger, event, state=state)


def make_local_rag_search_batch_node(
    runtime: EvidenceOrchestrationRuntime,
) -> Callable[[LearningState], Awaitable[dict]]:
    """Create a Parent-Child local batch node for validated current-round tasks."""

    parent_rag_node = make_parent_child_rag_node(runtime.parent_child)

    async def local_rag_search_batch(state: LearningState) -> dict:
        started = time.perf_counter()
        round_index = _required_state_count(state, "evidence_current_round")
        tasks = tuple(
            task
            for task in (
                RetrievalTask.model_validate(item)
                for item in (state.get("evidence_current_tasks") or [])
            )
            if task.source_type == "local_rag"
        )
        if not tasks:
            _emit_source_trace(
                state=state,
                source="local",
                round_index=round_index,
                tasks=tasks,
                records=(),
                latency_ms=0,
            )
            return {
                "evidence_local_batch": {
                    "round_index": round_index,
                    "records": [],
                    "outcomes": [],
                    "parent_child_round": {},
                }
            }

        task_by_id = {task.task_id: task for task in tasks}
        retrieval_plan = [
            {
                "subject": task.subject,
                "role": task.requirement_id,
                "local_retrieval_query": task.query,
                "web_research_seed_query": task.query,
                "purpose": task.task_id,
                "relation_to_goal": task.purpose,
                "priority": runtime.policy.retrieval_priority_weights.weight_for(
                    task.priority
                ),
                "retrieval_coverage_hint": task.purpose,
                "retrieval_coverage_goals": [task.purpose],
                "_parent_child_priority_explicit": True,
            }
            for task in tasks
        ]
        try:
            output = await parent_rag_node(
                {
                    **state,
                    "retrieval_plan": retrieval_plan,
                    "local_retrieval_query": tasks[0].query,
                }
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            emit_evidence_trace(
                logger,
                {
                    "schema_version": EVIDENCE_TRACE_SCHEMA_VERSION,
                    "stage": "evidence_orchestration.source.failed",
                    "round_index": round_index,
                    "source": "local",
                    "status": "failed",
                    "task_count": len(tasks),
                    "query_batch_fingerprint": _query_batch_fingerprint(tasks),
                    "latency_ms": latency_ms,
                    "reason_code": "local_retrieval_failed",
                    "error_type": type(exc).__name__,
                },
                state=state,
            )
            raise

        originals = output.get("local_evidence_originals") or {}
        records: list[EvidenceCandidateRecord] = []
        for raw_candidate in output.get("local_evidence_candidates") or []:
            candidate = EvidenceCandidate.model_validate(raw_candidate)
            task = task_by_id.get(candidate.purpose)
            if task is None:
                raise ParentChildGraphContractError(
                    "local candidate is not bound to a current retrieval task"
                )
            raw_original = originals.get(candidate.evidence_id)
            if not isinstance(raw_original, dict):
                raise ParentChildGraphContractError(
                    "local candidate original is absent from Parent-Child output"
                )
            records.append(
                _candidate_record(
                    candidate=candidate,
                    original=dict(raw_original),
                    task=task,
                )
            )
        snapshot_payload = {
            "round_index": round_index,
            "retrieval_result": output.get("parent_child_retrieval_result") or {},
            "local_refs": tuple(output.get("parent_child_local_refs") or []),
        }
        snapshot = ParentChildRoundSnapshot(
            **snapshot_payload,
            snapshot_fingerprint=_digest(snapshot_payload),
        )
        outcomes = _source_outcomes(tasks, records)
        latency_ms = int((time.perf_counter() - started) * 1000)
        _emit_source_trace(
            state=state,
            source="local",
            round_index=round_index,
            tasks=tasks,
            records=records,
            latency_ms=latency_ms,
        )
        return {
            "evidence_local_batch": {
                "round_index": round_index,
                "records": [item.model_dump(mode="json") for item in records],
                "outcomes": [item.model_dump(mode="json") for item in outcomes],
                "parent_child_round": snapshot.model_dump(mode="json"),
            }
        }

    return local_rag_search_batch


def make_web_research_search_batch_node(
    runtime: EvidenceOrchestrationRuntime,
) -> Callable[[LearningState], Awaitable[dict]]:
    """Create a direct-query Web batch node that never re-plans tasks."""

    async def web_research_search_batch(state: LearningState) -> dict:
        started = time.perf_counter()
        round_index = _required_state_count(state, "evidence_current_round")
        tasks = tuple(
            task
            for task in (
                RetrievalTask.model_validate(item)
                for item in (state.get("evidence_current_tasks") or [])
            )
            if task.source_type == "web"
        )
        if not tasks:
            _emit_source_trace(
                state=state,
                source="web",
                round_index=round_index,
                tasks=tasks,
                records=(),
                latency_ms=0,
            )
            return {
                "evidence_web_batch": {
                    "round_index": round_index,
                    "records": [],
                    "outcomes": [],
                }
            }
        web_tasks = [
            WebResearchTask(
                task_id=task.task_id,
                subject=task.subject,
                role=task.requirement_id,
                purpose=task.purpose,
                search_query=task.query,
                reason=task.purpose,
                priority=runtime.policy.retrieval_priority_weights.weight_for(
                    task.priority
                ),
            )
            for task in tasks
        ]
        try:
            output = await execute_validated_web_research_tasks(
                state=state,
                tasks=web_tasks,
                original_user_query=_last_human_query(state),
                timeout=runtime.web_timeout_seconds,
                max_results_per_task=runtime.policy.max_results_per_task,
                max_concurrent_tasks=runtime.policy.max_concurrent_tasks,
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            emit_evidence_trace(
                logger,
                {
                    "schema_version": EVIDENCE_TRACE_SCHEMA_VERSION,
                    "stage": "evidence_orchestration.source.failed",
                    "round_index": round_index,
                    "source": "web",
                    "status": "failed",
                    "task_count": len(tasks),
                    "query_batch_fingerprint": _query_batch_fingerprint(tasks),
                    "latency_ms": latency_ms,
                    "reason_code": "web_retrieval_failed",
                    "error_type": type(exc).__name__,
                },
                state=state,
            )
            raise

        task_by_id = {task.task_id: task for task in tasks}
        originals = output.get("originals") or {}
        records: list[EvidenceCandidateRecord] = []
        for raw_candidate in output.get("candidates") or []:
            candidate = EvidenceCandidate.model_validate(raw_candidate)
            task_id = str(candidate.metadata.get("task_id") or "")
            task = task_by_id.get(task_id)
            if task is None:
                raise EvidenceOrchestrationRuntimeError(
                    code="unbound_web_candidate",
                    reason="web candidate task id is not in the current batch",
                )
            raw_original = originals.get(candidate.evidence_id)
            if not isinstance(raw_original, dict):
                raise EvidenceOrchestrationRuntimeError(
                    code="missing_web_original",
                    reason="web candidate has no curated original record",
                )
            records.append(
                _candidate_record(
                    candidate=candidate,
                    original=dict(raw_original),
                    task=task,
                )
            )
        outcomes = _source_outcomes(tasks, records)
        latency_ms = int((time.perf_counter() - started) * 1000)
        _emit_source_trace(
            state=state,
            source="web",
            round_index=round_index,
            tasks=tasks,
            records=records,
            latency_ms=latency_ms,
        )
        return {
            "evidence_web_batch": {
                "round_index": round_index,
                "records": [item.model_dump(mode="json") for item in records],
                "outcomes": [item.model_dump(mode="json") for item in outcomes],
            }
        }

    return web_research_search_batch


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _merge_unique_models(
    existing: Sequence[_ModelT],
    additions: Sequence[_ModelT],
    *,
    identity: Callable[[_ModelT], object],
) -> tuple[_ModelT, ...]:
    merged: dict[object, _ModelT] = {}
    for item in (*existing, *additions):
        key = identity(item)
        previous = merged.get(key)
        if previous is not None and previous != item:
            raise EvidenceOrchestrationRuntimeError(
                code="resume_state_conflict",
                reason="checkpoint replay produced conflicting state for one identity",
            )
        merged[key] = item
    return tuple(merged.values())


def _merge_candidate_records(
    existing: Sequence[EvidenceCandidateRecord],
    additions: Sequence[EvidenceCandidateRecord],
) -> tuple[EvidenceCandidateRecord, ...]:
    merged: dict[str, EvidenceCandidateRecord] = {}
    for item in (*existing, *additions):
        previous = merged.get(item.evidence_id)
        if previous is None:
            merged[item.evidence_id] = item
            continue
        identity_fields = (
            "requirement_id",
            "resource_type",
            "subject",
            "source_type",
            "source_identity_fingerprint",
            "content_fingerprint",
        )
        if any(
            getattr(previous, field_name) != getattr(item, field_name)
            for field_name in identity_fields
        ):
            raise EvidenceOrchestrationRuntimeError(
                code="evidence_identity_collision",
                reason="one exact evidence id resolved to conflicting identity fields",
            )
    return tuple(merged.values())


def make_retrieval_round_merge_node(
    runtime: EvidenceOrchestrationRuntime,
) -> Callable[[LearningState], dict]:
    """Create the sole owner of cumulative candidate and round state."""

    def retrieval_round_merge(state: LearningState) -> dict:
        round_index = _required_state_count(state, "evidence_current_round")
        local_batch = state.get("evidence_local_batch") or {}
        web_batch = state.get("evidence_web_batch") or {}
        if (
            local_batch.get("round_index") != round_index
            or web_batch.get("round_index") != round_index
        ):
            raise EvidenceOrchestrationRuntimeError(
                code="round_barrier_mismatch",
                reason="local and web batches must match the active round",
            )
        existing_records = tuple(
            EvidenceCandidateRecord.model_validate(item)
            for item in (state.get("evidence_candidate_records") or [])
        )
        additions = tuple(
            EvidenceCandidateRecord.model_validate(item)
            for item in [
                *(local_batch.get("records") or []),
                *(web_batch.get("records") or []),
            ]
        )
        merged_records = _merge_candidate_records(existing_records, additions)
        if len(merged_records) > runtime.policy.max_ledger_entries:
            raise EvidenceBudgetExceededError(
                code="candidate_ledger_budget_exceeded",
                reason="retrieved candidates exceed max_ledger_entries",
            )

        existing_outcomes = tuple(
            EvidenceSourceOutcome.model_validate(item)
            for item in (state.get("evidence_source_outcomes") or [])
        )
        new_outcomes = tuple(
            EvidenceSourceOutcome.model_validate(item)
            for item in [
                *(local_batch.get("outcomes") or []),
                *(web_batch.get("outcomes") or []),
            ]
        )
        merged_outcomes = _merge_unique_models(
            existing_outcomes,
            new_outcomes,
            identity=lambda item: (item.round_index, item.task_id),
        )

        existing_rounds = tuple(
            ParentChildRoundSnapshot.model_validate(item)
            for item in (state.get("evidence_parent_child_rounds") or [])
        )
        new_round_payload = local_batch.get("parent_child_round") or {}
        new_rounds = (
            (ParentChildRoundSnapshot.model_validate(new_round_payload),)
            if new_round_payload
            else ()
        )
        merged_rounds = _merge_unique_models(
            existing_rounds,
            new_rounds,
            identity=lambda item: item.round_index,
        )
        deduplicated_count = (
            len(existing_records) + len(additions) - len(merged_records)
        )
        local_count = sum(
            record.round_index == round_index and record.source_type == "local_rag"
            for record in merged_records
        )
        web_count = sum(
            record.round_index == round_index and record.source_type == "web"
            for record in merged_records
        )
        emit_evidence_trace(
            logger,
            {
                "schema_version": EVIDENCE_TRACE_SCHEMA_VERSION,
                "stage": "evidence_orchestration.round.merged",
                "round_index": round_index,
                "local_candidate_count": local_count,
                "web_candidate_count": web_count,
                "deduplicated_count": deduplicated_count,
                "ledger_count": len(merged_records),
                "ledger_fingerprint": _digest(
                    [record.evidence_id for record in merged_records]
                ),
            },
            state=state,
        )
        return {
            "evidence_candidate_records": [
                item.model_dump(mode="json") for item in merged_records
            ],
            "evidence_source_outcomes": [
                item.model_dump(mode="json") for item in merged_outcomes
            ],
            "evidence_parent_child_rounds": [
                item.model_dump(mode="json") for item in merged_rounds
            ],
            "evidence_orchestration_status": "judging",
        }

    return retrieval_round_merge


def _ledger_entry(
    record: EvidenceCandidateRecord,
    *,
    accepted: bool,
) -> EvidenceLedgerEntry:
    return EvidenceLedgerEntry(
        round_index=record.round_index,
        task_id=record.task_id,
        requirement_id=record.requirement_id,
        evidence_id=record.evidence_id,
        resource_type=record.resource_type,
        subject=record.subject,
        source_type=record.source_type,
        candidate_ref=record.candidate_ref,
        candidate_snapshot_fingerprint=record.candidate_snapshot_fingerprint,
        source_identity_fingerprint=record.source_identity_fingerprint,
        content_fingerprint=record.content_fingerprint,
        accepted=accepted,
        rejection_reason_code="" if accepted else "not_selected_by_coverage_judge",
    )


def _judge_candidates_payload(
    records: Sequence[EvidenceCandidateRecord],
) -> list[dict[str, object]]:
    return [
        {
            "evidence_id": record.evidence_id,
            "requirement_id": record.requirement_id,
            "source_type": record.source_type,
            "subject": record.subject,
            "title": record.candidate.title,
            "content_preview": record.candidate.content_preview,
            "retrieval_score": (
                record.candidate.rerank_score
                if record.source_type == "local_rag"
                else record.candidate.tavily_score
            ),
        }
        for record in records
    ]


def _coverage_business_validation(
    parsed: BaseModel,
    *,
    round_index: int,
    requirements: tuple[EvidenceRequirement, ...],
    provisional_entries: tuple[EvidenceLedgerEntry, ...],
    attempted_tasks: tuple[RetrievalTask, ...],
    outcomes: tuple[EvidenceSourceOutcome, ...],
) -> str:
    if not isinstance(parsed, RequirementCoverageBatch):
        return "parsed result is not RequirementCoverageBatch"
    try:
        if parsed.round_index != round_index:
            raise EvidenceOrchestrationRuntimeError(
                code="coverage_round_mismatch",
                reason="coverage output round must match the active retrieval round",
            )
        validate_requirement_coverage(
            batch=parsed,
            requirements=requirements,
            entries=provisional_entries,
        )
        requirement_by_id = {
            requirement.requirement_id: requirement for requirement in requirements
        }
        prior_signatures = {
            (
                task.requirement_id,
                task.source_type,
                task.query_fingerprint,
            )
            for task in attempted_tasks
        }
        local_attempted = {
            outcome.requirement_id
            for outcome in outcomes
            if outcome.source_type == "local_rag"
        }
        for coverage in parsed.coverages:
            requirement = requirement_by_id[coverage.requirement_id]
            if (
                coverage.coverage_state != "complete"
                and requirement.source_policy == "local_then_web_on_gap"
            ):
                if coverage.requirement_id in local_attempted:
                    valid_staged_query = bool(
                        coverage.suggested_web_query
                        and not coverage.suggested_local_query
                    )
                    reason = (
                        "local_then_web_on_gap must move to a web query after "
                        "local completion"
                    )
                else:
                    valid_staged_query = bool(
                        coverage.suggested_local_query
                        and not coverage.suggested_web_query
                    )
                    reason = (
                        "local_then_web_on_gap must complete a local attempt "
                        "before web retrieval"
                    )
                if not valid_staged_query:
                    raise EvidenceOrchestrationRuntimeError(
                        code="staged_gap_query_invalid",
                        reason=reason,
                    )
            for source_type, query in (
                ("local_rag", coverage.suggested_local_query),
                ("web", coverage.suggested_web_query),
            ):
                if not query:
                    continue
                query_fingerprint = _text_digest(query)
                if (
                    coverage.requirement_id,
                    source_type,
                    query_fingerprint,
                ) in prior_signatures:
                    raise EvidenceOrchestrationRuntimeError(
                        code="repeated_gap_query",
                        reason="coverage repair query repeats a prior source-bound query",
                    )
    except (
        EvidenceOrchestrationContractError,
        EvidenceOrchestrationRuntimeError,
        ValueError,
    ) as exc:
        return str(exc)
    return ""


def _coverage_counts(batch: RequirementCoverageBatch) -> dict[str, int]:
    counts = Counter(item.coverage_state for item in batch.coverages)
    return {
        "complete": counts["complete"],
        "partial": counts["partial"],
        "missing": counts["missing"],
    }


def _previous_coverage_counts(
    state: LearningState,
    requirement_count: int,
) -> dict[str, int]:
    raw = state.get("evidence_coverage")
    if raw is None:
        raise EvidenceOrchestrationRuntimeError(
            code="missing_evidence_coverage_state",
            reason="evidence_coverage must be explicitly initialized",
        )
    if raw == {}:
        return {"complete": 0, "partial": 0, "missing": requirement_count}
    parsed = RequirementCoverageBatch.model_validate(raw)
    return _coverage_counts(parsed)


def make_requirement_evidence_judge_node(
    runtime: EvidenceOrchestrationRuntime,
) -> Callable[[LearningState], Awaitable[dict]]:
    """Create the per-requirement coverage judge and bounded route decision."""

    async def requirement_evidence_judge(state: LearningState) -> dict:
        round_index = _required_state_count(state, "evidence_current_round")
        requirements = tuple(
            EvidenceRequirement.model_validate(item)
            for item in (state.get("evidence_requirements") or [])
        )
        records = tuple(
            EvidenceCandidateRecord.model_validate(item)
            for item in (state.get("evidence_candidate_records") or [])
        )
        attempted_tasks = tuple(
            RetrievalTask.model_validate(item)
            for item in (state.get("evidence_all_tasks") or [])
        )
        outcomes = tuple(
            EvidenceSourceOutcome.model_validate(item)
            for item in (state.get("evidence_source_outcomes") or [])
        )
        if not requirements:
            raise EvidenceOrchestrationRuntimeError(
                code="missing_requirements_at_judge",
                reason="coverage judge requires the compiled requirement inventory",
            )
        provisional_entries = tuple(
            _ledger_entry(record, accepted=True) for record in records
        )
        prompt = _render_prompt(
            "requirement_evidence_judge",
            {
                "question": _last_human_query(state),
                "learning_goal": str(state.get("learning_goal") or "").strip(),
                "round_index": round_index,
                "requirements_json": json.dumps(
                    [item.model_dump(mode="json") for item in requirements],
                    ensure_ascii=False,
                ),
                "candidates_json": json.dumps(
                    _judge_candidates_payload(records),
                    ensure_ascii=False,
                ),
                "attempted_queries_json": json.dumps(
                    [
                        {
                            "requirement_id": task.requirement_id,
                            "source_type": task.source_type,
                            "query_fingerprint": task.query_fingerprint,
                        }
                        for task in attempted_tasks
                    ],
                    ensure_ascii=False,
                ),
            },
        )
        try:
            structured = await invoke_structured_llm(
                node_name="requirement_evidence_judge",
                llm_node="evidence_judge",
                schema=RequirementCoverageBatch,
                messages=[
                    SystemMessage(
                        content=(
                            "Judge every supplied requirement exactly once. "
                            "Never declare resource readiness or invent evidence ids."
                        )
                    ),
                    HumanMessage(content=prompt),
                ],
                output_mode=get_llm_output_mode("requirement_evidence_judge"),
                fallback_modes=[],
                business_validator=lambda parsed: _coverage_business_validation(
                    parsed,
                    round_index=round_index,
                    requirements=requirements,
                    provisional_entries=provisional_entries,
                    attempted_tasks=attempted_tasks,
                    outcomes=outcomes,
                ),
                state=state,
                max_raw_chars=get_max_raw_chars("requirement_evidence_judge"),
            )
        except Exception as exc:
            emit_evidence_trace(
                logger,
                {
                    "schema_version": EVIDENCE_TRACE_SCHEMA_VERSION,
                    "stage": "evidence_orchestration.failed",
                    "status": "failed",
                    "round_index": round_index,
                    "source": "judge",
                    "error_type": type(exc).__name__,
                    "reason_code": "coverage_judge_failed",
                    "budget_used_tasks": len(attempted_tasks),
                    "budget_remaining_tasks": (
                        runtime.policy.max_total_search_tasks - len(attempted_tasks)
                    ),
                },
                state=state,
            )
            raise
        parsed = structured.parsed
        if not isinstance(parsed, RequirementCoverageBatch):
            raise TypeError("judge result is not RequirementCoverageBatch")
        validation_error = _coverage_business_validation(
            parsed,
            round_index=round_index,
            requirements=requirements,
            provisional_entries=provisional_entries,
            attempted_tasks=attempted_tasks,
            outcomes=outcomes,
        )
        if validation_error:
            raise EvidenceOrchestrationRuntimeError(
                code="coverage_business_validation_failed",
                reason=validation_error,
            )

        accepted_ids = {
            evidence_id
            for coverage in parsed.coverages
            for evidence_id in coverage.evidence_ids
        }
        ledger = tuple(
            _ledger_entry(record, accepted=record.evidence_id in accepted_ids)
            for record in records
        )
        validate_evidence_ledger(
            entries=ledger,
            tasks=attempted_tasks,
            requirements=requirements,
            config=runtime.policy,
        )
        resources = _RESOURCE_TYPES_ADAPTER.validate_python(
            tuple(state.get("evidence_requested_resource_types") or [])
        )
        readiness = derive_resource_readiness(
            requested_resource_types=resources,
            requirements=requirements,
            batch=parsed,
        )
        counts = _coverage_counts(parsed)
        previous_counts = _previous_coverage_counts(state, len(requirements))
        previous_coverage = state.get("evidence_coverage")
        if not isinstance(previous_coverage, dict):
            raise EvidenceOrchestrationRuntimeError(
                code="invalid_evidence_coverage_state",
                reason="evidence_coverage must be an explicit mapping",
            )
        previous_ledger = tuple(
            EvidenceLedgerEntry.model_validate(item)
            for item in (state.get("evidence_ledger") or [])
        )
        previous_accepted_ids = _accepted_ids(previous_ledger)
        new_accepted_count = len(accepted_ids - previous_accepted_ids)
        progressed = (
            counts["missing"] < previous_counts["missing"]
            or counts["complete"] > previous_counts["complete"]
            or new_accepted_count > 0
        )
        previous_no_progress = _required_state_count(
            state,
            "evidence_consecutive_no_progress_rounds",
        )
        no_progress_rounds = (
            0 if round_index == 0 or progressed else previous_no_progress + 1
        )
        ready_count = sum(row.readiness_state == "ready" for row in readiness)
        all_ready = ready_count == len(readiness)
        if all_ready:
            route = "terminal"
            terminal_status = "sufficient"
            terminal_reason = "all_required_evidence_complete"
        elif round_index >= runtime.policy.max_supplement_rounds:
            route = "terminal"
            terminal_status = (
                "partial_resources_ready"
                if ready_count
                else "insufficient_empty_sources"
                if not records
                else "insufficient_max_rounds"
            )
            terminal_reason = "supplement_round_budget_exhausted"
        elif (
            round_index > 0
            and no_progress_rounds >= runtime.policy.max_consecutive_no_progress_rounds
        ):
            route = "terminal"
            terminal_status = (
                "partial_resources_ready"
                if ready_count
                else "insufficient_empty_sources"
                if not records
                else "insufficient_no_progress"
            )
            terminal_reason = "no_measurable_coverage_progress"
        else:
            route = "repair"
            terminal_status = ""
            terminal_reason = "required_evidence_gap"

        emit_evidence_trace(
            logger,
            {
                "schema_version": EVIDENCE_TRACE_SCHEMA_VERSION,
                "stage": "evidence_orchestration.coverage.judged",
                "round_index": round_index,
                "requirement_count": len(requirements),
                "complete_count": counts["complete"],
                "partial_count": counts["partial"],
                "missing_count": counts["missing"],
                "accepted_evidence_count": len(accepted_ids),
                "coverage_fingerprint": _digest(parsed.model_dump(mode="json")),
            },
            state=state,
        )
        emit_evidence_trace(
            logger,
            {
                "schema_version": EVIDENCE_TRACE_SCHEMA_VERSION,
                "stage": "evidence_orchestration.progress.evaluated",
                "round_index": round_index,
                "previous_complete_count": previous_counts["complete"],
                "current_complete_count": counts["complete"],
                "previous_partial_count": previous_counts["partial"],
                "current_partial_count": counts["partial"],
                "previous_missing_count": previous_counts["missing"],
                "current_missing_count": counts["missing"],
                "new_accepted_evidence_count": new_accepted_count,
                "progressed": progressed,
                "consecutive_no_progress_rounds": no_progress_rounds,
            },
            state=state,
        )
        if route == "terminal":
            emit_evidence_trace(
                logger,
                {
                    "schema_version": EVIDENCE_TRACE_SCHEMA_VERSION,
                    "stage": "evidence_orchestration.route.decided",
                    "round_index": round_index,
                    "status": "terminal",
                    "reason_code": terminal_reason,
                    "next_local_task_count": 0,
                    "next_web_task_count": 0,
                    "budget_remaining_rounds": (
                        runtime.policy.max_supplement_rounds - round_index
                    ),
                    "budget_remaining_tasks": (
                        runtime.policy.max_total_search_tasks - len(attempted_tasks)
                    ),
                },
                state=state,
            )
        return {
            "evidence_previous_coverage": previous_coverage,
            "evidence_coverage": parsed.model_dump(mode="json"),
            "evidence_ledger": [item.model_dump(mode="json") for item in ledger],
            "resource_evidence_readiness": [
                item.model_dump(mode="json") for item in readiness
            ],
            "evidence_consecutive_no_progress_rounds": no_progress_rounds,
            "evidence_orchestration_route": route,
            "evidence_terminal_status": terminal_status,
            "evidence_terminal_reason_code": terminal_reason,
            "evidence_orchestration_status": (
                "terminal" if route == "terminal" else "repairing"
            ),
        }

    return requirement_evidence_judge


def route_after_requirement_evidence_judge(state: LearningState) -> str:
    """Route only explicit, validated judge decisions."""

    route = str(state.get("evidence_orchestration_route") or "")
    if route not in {"repair", "terminal"}:
        raise EvidenceOrchestrationRuntimeError(
            code="invalid_evidence_route",
            reason="coverage judge must select repair or terminal",
        )
    return route


def _repair_source_queries(
    requirement: EvidenceRequirement,
    coverage,
    *,
    local_attempted: bool,
) -> tuple[tuple[Literal["local_rag", "web"], str], ...]:
    if requirement.source_policy == "local_only":
        return (("local_rag", coverage.suggested_local_query),)
    if requirement.source_policy == "web_only":
        return (("web", coverage.suggested_web_query),)
    if requirement.source_policy == "local_then_web_on_gap":
        return (
            (("web", coverage.suggested_web_query),)
            if local_attempted
            else (("local_rag", coverage.suggested_local_query),)
        )
    if requirement.source_policy == "local_and_web":
        return (
            ("local_rag", coverage.suggested_local_query),
            ("web", coverage.suggested_web_query),
        )
    raise EvidenceOrchestrationRuntimeError(
        code="unknown_repair_source_policy",
        reason="repair planner received an unsupported source policy",
    )


def make_evidence_repair_planner_node(
    runtime: EvidenceOrchestrationRuntime,
) -> Callable[[LearningState], dict]:
    """Create a deterministic planner for Judge-authored targeted gap queries."""

    def evidence_repair_planner(state: LearningState) -> dict:
        current_round = _required_state_count(state, "evidence_current_round")
        next_round = current_round + 1
        if next_round > runtime.policy.max_supplement_rounds:
            raise EvidenceBudgetExceededError(
                code="repair_round_budget_exceeded",
                reason="repair planner cannot schedule beyond supplement budget",
            )
        requirements = tuple(
            EvidenceRequirement.model_validate(item)
            for item in (state.get("evidence_requirements") or [])
        )
        requirement_by_id = {item.requirement_id: item for item in requirements}
        coverage = RequirementCoverageBatch.model_validate(
            state.get("evidence_coverage")
        )
        coverage_by_id = {item.requirement_id: item for item in coverage.coverages}
        readiness = tuple(
            ResourceReadiness.model_validate(item)
            for item in (state.get("resource_evidence_readiness") or [])
        )
        blocked_ids = tuple(
            dict.fromkeys(
                requirement_id
                for row in readiness
                for requirement_id in row.blocked_requirement_ids
            )
        )
        if not blocked_ids:
            raise EvidenceOrchestrationRuntimeError(
                code="repair_without_blocked_requirements",
                reason="repair route requires unresolved required evidence",
            )
        all_tasks = tuple(
            RetrievalTask.model_validate(item)
            for item in (state.get("evidence_all_tasks") or [])
        )
        prior_signatures = {task.retrieval_signature for task in all_tasks}
        source_outcomes = tuple(
            EvidenceSourceOutcome.model_validate(item)
            for item in (state.get("evidence_source_outcomes") or [])
        )
        local_attempted_ids = {
            item.requirement_id
            for item in source_outcomes
            if item.source_type == "local_rag" and item.status in {"completed", "empty"}
        }
        remaining_total = runtime.policy.max_total_search_tasks - len(all_tasks)
        task_capacity = min(
            runtime.policy.max_search_tasks_per_round,
            remaining_total,
        )
        if task_capacity <= 0:
            raise EvidenceBudgetExceededError(
                code="repair_task_budget_exhausted",
                reason="no total search-task budget remains for repair",
            )
        proposed: list[RetrievalTask] = []
        proposed_signatures: set[str] = set()
        duplicate_signature_count = 0
        for requirement_id in blocked_ids:
            requirement = requirement_by_id[requirement_id]
            row = coverage_by_id[requirement_id]
            for source_type, query in _repair_source_queries(
                requirement,
                row,
                local_attempted=requirement_id in local_attempted_ids,
            ):
                if len(proposed) >= task_capacity:
                    break
                if not query:
                    raise EvidenceOrchestrationRuntimeError(
                        code="missing_repair_query",
                        reason="coverage judge omitted a required source-specific repair query",
                    )
                task = build_retrieval_task(
                    requirement=requirement,
                    source_type=source_type,
                    query=query,
                    purpose=requirement.acceptance_criteria,
                    priority=runtime.policy.required_task_priority,
                    round_index=next_round,
                    result_limit=runtime.policy.max_results_per_task,
                )
                if (
                    task.retrieval_signature in prior_signatures
                    or task.retrieval_signature in proposed_signatures
                ):
                    duplicate_signature_count += 1
                    continue
                proposed.append(task)
                proposed_signatures.add(task.retrieval_signature)
            if len(proposed) >= task_capacity:
                break
        if not proposed:
            if duplicate_signature_count:
                raise DuplicateRetrievalSignatureError(
                    code="duplicate_retrieval_signature",
                    reason="repair queries repeat previously attempted signatures",
                )
            raise EvidenceOrchestrationRuntimeError(
                code="empty_repair_plan",
                reason="blocked requirements produced no repair task",
            )
        validate_retrieval_tasks(
            tasks=proposed,
            requirements=requirements,
            config=runtime.policy,
            round_index=next_round,
            existing_total_search_tasks=len(all_tasks),
            prior_retrieval_signatures=prior_signatures,
            local_then_web_gap_requirement_ids=set(blocked_ids),
        )
        target_ids = tuple(dict.fromkeys(task.requirement_id for task in proposed))
        plan = EvidenceRepairPlan(
            round_index=next_round,
            target_requirement_ids=target_ids,
            tasks=tuple(proposed),
            reason="targeted_required_gap_repair",
            plan_signature=make_repair_plan_signature(
                round_index=next_round,
                target_requirement_ids=target_ids,
                tasks=proposed,
            ),
        )
        repair_plans = [
            *(state.get("evidence_repair_plans") or []),
            plan.model_dump(mode="json"),
        ]
        updated_tasks = (*all_tasks, *proposed)
        local_count = sum(task.source_type == "local_rag" for task in proposed)
        web_count = sum(task.source_type == "web" for task in proposed)
        emit_evidence_trace(
            logger,
            {
                "schema_version": EVIDENCE_TRACE_SCHEMA_VERSION,
                "stage": "evidence_orchestration.route.decided",
                "round_index": current_round,
                "status": "repair",
                "reason_code": "targeted_required_gap_repair",
                "next_local_task_count": local_count,
                "next_web_task_count": web_count,
                "budget_remaining_rounds": (
                    runtime.policy.max_supplement_rounds - next_round
                ),
                "budget_remaining_tasks": (
                    runtime.policy.max_total_search_tasks - len(updated_tasks)
                ),
            },
            state=state,
        )
        return {
            "evidence_current_round": next_round,
            "evidence_current_tasks": [
                task.model_dump(mode="json") for task in proposed
            ],
            "evidence_all_tasks": [
                task.model_dump(mode="json") for task in updated_tasks
            ],
            "evidence_retrieval_signatures": [
                task.retrieval_signature for task in updated_tasks
            ],
            "evidence_repair_plans": repair_plans,
            "evidence_orchestration_route": "retrieve",
            "evidence_orchestration_status": "repair_planned",
        }

    return evidence_repair_planner


def _accepted_ids(ledger: Sequence[EvidenceLedgerEntry]) -> set[str]:
    return {item.evidence_id for item in ledger if item.accepted}


def _coverage_confidence(
    batch: RequirementCoverageBatch,
) -> dict[str, float]:
    return {
        evidence_id: coverage.confidence
        for coverage in batch.coverages
        for evidence_id in coverage.evidence_ids
    }


def make_terminal_parent_hydration_node(
    runtime: EvidenceOrchestrationRuntime,
) -> Callable[[LearningState], Awaitable[dict]]:
    """Create the one-shot terminal parent hydration and approved context node."""

    async def parent_child_parent_hydration(state: LearningState) -> dict:
        if _required_state_count(state, "evidence_hydration_count") != 0:
            raise EvidenceOrchestrationRuntimeError(
                code="parent_hydration_repeated",
                reason="parent bodies may be hydrated exactly once after terminal judging",
            )
        ledger = tuple(
            EvidenceLedgerEntry.model_validate(item)
            for item in (state.get("evidence_ledger") or [])
        )
        accepted_ids = _accepted_ids(ledger)
        all_records = tuple(
            EvidenceCandidateRecord.model_validate(item)
            for item in (state.get("evidence_candidate_records") or [])
        )
        records = tuple(
            record for record in all_records if record.evidence_id in accepted_ids
        )
        coverage = RequirementCoverageBatch.model_validate(
            state.get("evidence_coverage")
        )
        confidence_by_id = _coverage_confidence(coverage)
        snapshots = tuple(
            ParentChildRoundSnapshot.model_validate(item)
            for item in (state.get("evidence_parent_child_rounds") or [])
        )
        local_docs: list[dict[str, object]] = []
        hydrated_parent_ids: set[str] = set()
        for snapshot in snapshots:
            round_records = tuple(
                record
                for record in records
                if record.source_type == "local_rag"
                and record.round_index == snapshot.round_index
            )
            if not round_records:
                continue
            result = MultiBranchHybridChildResult.model_validate_json(
                json.dumps(
                    snapshot.retrieval_result,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            )
            if result.request.generation_id != runtime.parent_child.generation_id:
                raise ParentChildGraphContractError(
                    "stored orchestration round changed Parent-Child generation"
                )
            refs = tuple(
                LocalEvidenceRef.model_validate_json(
                    json.dumps(
                        item,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                )
                for item in snapshot.local_refs
            )
            ref_by_id = {item.evidence_id: item for item in refs}
            if len(ref_by_id) != len(refs):
                raise ParentChildGraphContractError(
                    "stored orchestration round has duplicate local refs"
                )
            selected_child_ids = tuple(
                dict.fromkeys(record.candidate_ref for record in round_records)
            )
            if any(child_id not in ref_by_id for child_id in selected_child_ids):
                raise ParentChildGraphContractError(
                    "accepted local evidence is absent from its retrieval snapshot"
                )
            contexts = await asyncio.to_thread(
                runtime.parent_child.retriever.hydrate_kept_multi,
                result,
                selected_child_ids,
            )
            returned_support = {
                child_id
                for context in contexts
                for child_id in context.supporting_child_ids
            }
            if returned_support != set(selected_child_ids):
                raise ParentChildGraphContractError(
                    "terminal hydration support differs from accepted child evidence"
                )
            record_by_child: dict[str, list[EvidenceCandidateRecord]] = {}
            for record in round_records:
                record_by_child.setdefault(record.candidate_ref, []).append(record)
            for item in parent_context_items(contexts):
                supporting_records = tuple(
                    record
                    for child_id in item.supporting_child_ids
                    for record in record_by_child.get(child_id, [])
                )
                evidence_ids = tuple(
                    dict.fromkeys(record.evidence_id for record in supporting_records)
                )
                resource_types = tuple(
                    dict.fromkeys(record.resource_type for record in supporting_records)
                )
                providers = tuple(
                    dict.fromkeys(
                        record.candidate.provider for record in supporting_records
                    )
                )
                if not evidence_ids:
                    raise ParentChildGraphContractError(
                        "hydrated parent has no accepted requirement-bound evidence"
                    )
                if len(providers) != 1 or not providers[0].strip():
                    raise ParentChildGraphContractError(
                        "hydrated parent must resolve to one explicit evidence provider"
                    )
                hydrated_parent_ids.add(item.parent_id)
                local_docs.append(
                    {
                        "type": "rag",
                        "source_type": "local_rag",
                        "provider": providers[0],
                        "evidence_id": evidence_ids[0],
                        "evidence_ids": list(evidence_ids),
                        "resource_types": list(resource_types),
                        "subject": item.subject,
                        "retrieval_subject": item.subject,
                        "source": item.source_relpath,
                        "content": item.content,
                        "page_content": item.content,
                        "parent_id": item.parent_id,
                        "generation_id": item.generation_id,
                        "policy_id": item.policy_id,
                        "page_start": item.page_start,
                        "page_end": item.page_end,
                        "supporting_child_ids": list(item.supporting_child_ids),
                        "expansion_mode": item.expansion_mode,
                        "window_spans": [list(span) for span in item.window_spans],
                        "evidence_score": max(
                            confidence_by_id[evidence_id]
                            for evidence_id in evidence_ids
                        ),
                        "score_source": "requirement_coverage_confidence_v1",
                    }
                )

        web_docs = []
        for record in records:
            if record.source_type != "web":
                continue
            web_docs.append(
                {
                    **record.original,
                    "type": "web_evidence",
                    "source_type": "web",
                    "evidence_id": record.evidence_id,
                    "evidence_ids": [record.evidence_id],
                    "resource_types": [record.resource_type],
                    "subject": record.subject,
                    "retrieval_subject": record.subject,
                    "evidence_score": confidence_by_id[record.evidence_id],
                    "score_source": "requirement_coverage_confidence_v1",
                }
            )
        combined = [*local_docs, *web_docs]
        deduped: dict[tuple[object, ...], dict[str, object]] = {}
        for doc in combined:
            key = (
                doc.get("source_type"),
                doc.get("parent_id") or doc.get("evidence_id"),
                _text_digest(str(doc.get("content") or "")),
            )
            previous = deduped.get(key)
            if previous is None:
                deduped[key] = doc
                continue
            previous_evidence_ids = _required_string_list(previous, "evidence_ids")
            current_evidence_ids = _required_string_list(doc, "evidence_ids")
            previous["evidence_ids"] = list(
                dict.fromkeys([*previous_evidence_ids, *current_evidence_ids])
            )
            previous_resource_types = _required_string_list(
                previous,
                "resource_types",
            )
            current_resource_types = _required_string_list(doc, "resource_types")
            previous["resource_types"] = list(
                dict.fromkeys([*previous_resource_types, *current_resource_types])
            )
            previous["evidence_score"] = max(
                _required_numeric_field(previous, "evidence_score"),
                _required_numeric_field(doc, "evidence_score"),
            )
        approved_context = sorted(
            deduped.values(),
            key=lambda item: (
                -_required_numeric_field(item, "evidence_score"),
                str(item.get("source_type") or ""),
                str(item.get("parent_id") or item.get("evidence_id") or ""),
            ),
        )
        return {
            "context": [*CONTEXT_CLEAR, *approved_context],
            "graded_evidence": approved_context,
            "evidence_hydration_count": 1,
            "parent_child_hydration": {
                "schema_version": "resource_evidence_parent_hydration_v1",
                "generation_id": runtime.parent_child.generation_id,
                "retrieval_fingerprint": (
                    runtime.parent_child.graph_handoff_fingerprint
                ),
                "round_count": len(snapshots),
                "parent_count": len(hydrated_parent_ids),
                "accepted_local_evidence_count": sum(
                    record.source_type == "local_rag" for record in records
                ),
            },
        }

    return parent_child_parent_hydration


def make_resource_evidence_assignment_node(
    runtime: EvidenceOrchestrationRuntime,
) -> Callable[[LearningState], dict]:
    """Create code-derived per-resource readiness and assignment output."""

    def resource_evidence_assignment(state: LearningState) -> dict:
        requirements = tuple(
            EvidenceRequirement.model_validate(item)
            for item in (state.get("evidence_requirements") or [])
        )
        coverage = RequirementCoverageBatch.model_validate(
            state.get("evidence_coverage")
        )
        ledger = tuple(
            EvidenceLedgerEntry.model_validate(item)
            for item in (state.get("evidence_ledger") or [])
        )
        resources = _RESOURCE_TYPES_ADAPTER.validate_python(
            tuple(state.get("evidence_requested_resource_types") or [])
        )
        readiness = derive_resource_readiness(
            requested_resource_types=resources,
            requirements=requirements,
            batch=coverage,
        )
        assignments = derive_resource_evidence_assignments(
            readiness=readiness,
            requirements=requirements,
            batch=coverage,
            entries=ledger,
        )
        assignment_by_resource = {item.resource_type: item for item in assignments}
        ready = tuple(
            row.resource_type for row in readiness if row.readiness_state == "ready"
        )
        blocked = tuple(
            row.resource_type
            for row in readiness
            if row.readiness_state == "blocked_insufficient_evidence"
        )
        for row in readiness:
            assignment = assignment_by_resource.get(row.resource_type)
            fingerprint = (
                assignment.assignment_fingerprint
                if assignment is not None
                else _digest(row.model_dump(mode="json"))
            )
            emit_evidence_trace(
                logger,
                {
                    "schema_version": EVIDENCE_TRACE_SCHEMA_VERSION,
                    "stage": "evidence_orchestration.resource.assigned",
                    "round_index": _required_state_count(
                        state,
                        "evidence_current_round",
                    ),
                    "resource_type": row.resource_type,
                    "status": (
                        "ready" if row.readiness_state == "ready" else "blocked"
                    ),
                    "requirement_count": len(row.required_requirement_ids),
                    "assigned_evidence_count": len(row.evidence_ids),
                    "missing_requirement_count": len(row.blocked_requirement_ids),
                    "assignment_fingerprint": fingerprint,
                },
                state=state,
            )
        if len(ready) == len(resources):
            terminal_status = "sufficient"
        elif ready:
            terminal_status = "partial_resources_ready"
        else:
            terminal_status_value = state.get("evidence_terminal_status")
            if (
                not isinstance(terminal_status_value, str)
                or not terminal_status_value.strip()
            ):
                raise EvidenceOrchestrationContractError(
                    code="missing_terminal_status",
                    reason=(
                        "terminal resource assignment requires an explicit terminal status"
                    ),
                )
            terminal_status = terminal_status_value
            if terminal_status not in {
                "insufficient_max_rounds",
                "insufficient_no_progress",
                "insufficient_empty_sources",
                "blocked_insufficient_evidence",
            }:
                raise EvidenceOrchestrationContractError(
                    code="invalid_blocked_terminal_status",
                    reason=(
                        "blocked resource assignment received an invalid terminal status"
                    ),
                )
        terminal_reason_code = state.get("evidence_terminal_reason_code")
        if (
            not isinstance(terminal_reason_code, str)
            or not terminal_reason_code.strip()
        ):
            raise EvidenceOrchestrationContractError(
                code="missing_terminal_reason_code",
                reason="terminal resource assignment requires an explicit reason code",
            )
        all_tasks = tuple(
            RetrievalTask.model_validate(item)
            for item in (state.get("evidence_all_tasks") or [])
        )
        if not all_tasks:
            raise EvidenceOrchestrationContractError(
                code="missing_terminal_task_inventory",
                reason=(
                    "terminal resource assignment requires attempted retrieval tasks"
                ),
            )
        emit_evidence_trace(
            logger,
            {
                "schema_version": EVIDENCE_TRACE_SCHEMA_VERSION,
                "stage": "evidence_orchestration.terminal",
                "orchestration_fingerprint": runtime.orchestration_fingerprint,
                "status": terminal_status,
                "rounds_completed": _required_state_count(
                    state,
                    "evidence_current_round",
                )
                + 1,
                "ready_resource_count": len(ready),
                "blocked_resource_count": len(blocked),
                "total_search_tasks": len(all_tasks),
                "ledger_count": len(ledger),
                "reason_code": terminal_reason_code,
            },
            state=state,
        )
        return {
            "resource_evidence_readiness": [
                item.model_dump(mode="json") for item in readiness
            ],
            "resource_evidence_assignments": [
                item.model_dump(mode="json") for item in assignments
            ],
            "ready_resource_types": list(ready),
            "blocked_resource_types": list(blocked),
            "requested_resource_type": ready[0] if ready else "",
            "requested_resource_types": list(ready),
            "resource_generation_status": (
                "preflight" if ready else "blocked_insufficient_evidence"
            ),
            "evidence_terminal_status": terminal_status,
            "evidence_orchestration_status": "complete",
        }

    return resource_evidence_assignment


__all__ = [
    "EvidenceCandidateRecord",
    "EvidenceOrchestrationRuntime",
    "EvidenceOrchestrationRuntimeError",
    "EvidenceSourceOutcome",
    "ParentChildRoundSnapshot",
    "make_evidence_repair_planner_node",
    "make_local_rag_search_batch_node",
    "make_rag_generation_router_node",
    "make_requirement_evidence_judge_node",
    "make_resource_evidence_assignment_node",
    "make_resource_evidence_planner_node",
    "make_retrieval_round_merge_node",
    "make_retrieval_round_router_node",
    "make_terminal_parent_hydration_node",
    "make_web_research_search_batch_node",
    "route_after_requirement_evidence_judge",
]
