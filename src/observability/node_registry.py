"""Canonical cross-cutting metadata for graph and logical subnodes.

The registry is presentation and capture metadata only. LangGraph remains the
source of truth for executable topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

InfluenceKind = str
NodeRole = Literal[
    "router",
    "retrieval",
    "judge",
    "planner",
    "agent",
    "reviewer",
    "consensus",
    "output",
    "interrupt",
    "middleware",
]


@dataclass(frozen=True)
class InfluenceCaptureRule:
    kind: InfluenceKind
    preview_fields: tuple[str, ...] = ()
    list_fields: tuple[str, ...] = ()
    priority: int = 50
    injectable: bool | None = None


@dataclass(frozen=True)
class NodeRuntimeMetadata:
    node_id: str
    label: str
    description: str
    role: NodeRole
    operation: str
    group: str
    stage_rank: int
    parent: str = ""
    workflow: str = ""
    iteration_field: str = ""
    order: int = 0
    visible: bool = True
    capture_current_user_query: bool = False
    capture_rules: tuple[InfluenceCaptureRule, ...] = ()
    activity_running: str = ""
    activity_completed: str = ""


def _rule(
    kind: str,
    *preview_fields: str,
    list_fields: tuple[str, ...] = (),
    priority: int = 50,
    injectable: bool | None = None,
) -> InfluenceCaptureRule:
    return InfluenceCaptureRule(
        kind=kind,
        preview_fields=tuple(preview_fields),
        list_fields=list_fields,
        priority=priority,
        injectable=injectable,
    )


_NODE_METADATA: dict[str, NodeRuntimeMetadata] = {}


def _register(metadata: NodeRuntimeMetadata) -> None:
    if metadata.node_id in _NODE_METADATA:
        raise ValueError(f"duplicate node metadata: {metadata.node_id}")
    _NODE_METADATA[metadata.node_id] = metadata


def _metadata(
    node_id: str,
    *,
    label: str,
    description: str,
    role: NodeRole,
    operation: str = "",
    group: str,
    stage_rank: int,
    parent: str = "",
    workflow: str = "",
    iteration_field: str = "",
    order: int = 0,
    visible: bool = True,
    capture_current_user_query: bool = False,
    capture_rules: tuple[InfluenceCaptureRule, ...] = (),
) -> NodeRuntimeMetadata:
    return NodeRuntimeMetadata(
        node_id=node_id,
        label=label,
        description=description,
        role=role,
        operation=operation or role,
        group=group,
        stage_rank=stage_rank,
        parent=parent,
        workflow=workflow,
        iteration_field=iteration_field,
        order=order,
        visible=visible,
        capture_current_user_query=capture_current_user_query,
        capture_rules=capture_rules,
        activity_running=f"Running {label}",
        activity_completed=f"Completed {label}",
    )


for item in (
    _metadata(
        "supervisor",
        label="Request routing",
        description="Classify response mode, subject, and requested capability.",
        role="router",
        group="routing",
        stage_rank=10,
        order=10,
        capture_current_user_query=True,
    ),
    _metadata(
        "episodic_memory_retriever",
        label="Memory retrieval",
        description="Select bounded episodic and semantic learning memory.",
        role="retrieval",
        group="context",
        stage_rank=20,
        order=20,
    ),
    _metadata(
        "memory_use_decider",
        label="Memory policy",
        description="Decide whether selected long-term memory may influence the request.",
        role="router",
        group="context",
        stage_rank=25,
        order=25,
    ),
    _metadata(
        "search_query_rewriter",
        label="Query rewrite",
        description="Build retrieval queries for the current request.",
        role="planner",
        group="retrieval",
        stage_rank=30,
        order=30,
        capture_rules=(
            _rule(
                "query_rewrite",
                "rewritten_query",
                "search_query",
                "query",
                "search_queries",
                priority=80,
            ),
        ),
    ),
    _metadata(
        "academic_router",
        label="Academic retrieval routing",
        description="Dispatch local and web retrieval branches.",
        role="router",
        group="retrieval",
        stage_rank=35,
        order=35,
    ),
    _metadata(
        "rag_retrieve",
        label="Local retrieval",
        description="Retrieve local course evidence candidates.",
        role="retrieval",
        group="retrieval",
        stage_rank=40,
        order=40,
        capture_rules=(
            _rule(
                "local_evidence",
                "local_evidence_count",
                list_fields=("local_evidence_candidates", "local_evidence"),
                priority=70,
                injectable=False,
            ),
        ),
    ),
    _metadata(
        "web_search",
        label="Web research",
        description="Collect web evidence candidates for judging.",
        role="retrieval",
        group="retrieval",
        stage_rank=40,
        order=45,
        capture_rules=(
            _rule(
                "web_evidence",
                "web_evidence_count",
                list_fields=("web_evidence_candidates", "web_evidence"),
                priority=70,
                injectable=False,
            ),
        ),
    ),
    _metadata(
        "evidence_judge",
        label="Evidence judge",
        description="Gate factual evidence and identify coverage gaps.",
        role="judge",
        group="evidence",
        stage_rank=50,
        order=50,
        capture_rules=(
            _rule(
                "evidence_judge",
                "evidence_judge_state",
                "evidence_answerability",
                "evidence_judge_output",
                priority=90,
            ),
            _rule(
                "coverage_gap",
                list_fields=("evidence_coverage_gaps",),
                priority=85,
            ),
            _rule("workspace", "task_workspace", priority=70),
        ),
    ),
    _metadata(
        "generate_answer",
        label="Academic answer",
        description="Generate an evidence-grounded academic answer.",
        role="agent",
        group="answer",
        stage_rank=70,
        order=70,
        capture_rules=(_rule("agent_output", "messages", priority=65),),
    ),
    _metadata(
        "evaluate_hallucination",
        label="Answer review",
        description="Review answer grounding and decide whether to retry.",
        role="reviewer",
        group="answer",
        stage_rank=80,
        order=80,
        capture_rules=(
            _rule(
                "reviewer_output",
                "hallucination_verdict",
                "hallucination_reason",
                "evaluation",
                priority=75,
            ),
        ),
    ),
    _metadata(
        "rewrite_query",
        label="Retry query rewrite",
        description="Rewrite a query after answer-grounding review.",
        role="planner",
        group="answer",
        stage_rank=85,
        order=85,
        capture_rules=(
            _rule("query_rewrite", "rewritten_query", "query", priority=80),
        ),
    ),
    _metadata(
        "episodic_memory_writer",
        label="Memory write",
        description="Persist a compact learning episode after a successful answer.",
        role="output",
        group="context",
        stage_rank=90,
        order=90,
    ),
    _metadata(
        "evidence_summary_output",
        label="Evidence summary",
        description="Return a controlled evidence-gap result.",
        role="output",
        group="evidence",
        stage_rank=90,
        order=90,
    ),
    _metadata(
        "resource_preflight_router",
        label="Resource preflight",
        description="Normalize requested resources and select preflight gates.",
        role="router",
        group="resources",
        stage_rank=55,
        order=55,
    ),
    _metadata(
        "study_plan_profile_gate_main",
        label="Study profile gate",
        description="Interrupt study-plan generation until required profile facts exist.",
        role="interrupt",
        group="resources",
        stage_rank=58,
        order=58,
        capture_rules=(
            _rule(
                "profile_completion",
                "profile_completion_request",
                priority=95,
            ),
            _rule("learner_profile", "learner_profile", priority=90),
        ),
    ),
    _metadata(
        "resource_orchestrator",
        label="Resource orchestrator",
        description="Build and dispatch the resource fan-out plan.",
        role="planner",
        group="resources",
        stage_rank=60,
        order=60,
        capture_rules=(
            _rule(
                "retrieval_plan",
                "resource_generation_plan",
                priority=75,
            ),
        ),
    ),
    _metadata(
        "resource_worker",
        label="Resource worker",
        description="Execute one resource workflow branch.",
        role="agent",
        group="resources",
        stage_rank=65,
        order=65,
    ),
    _metadata(
        "resource_bundle_output",
        label="Resource bundle",
        description="Fan in resource branches and index successful artifacts.",
        role="output",
        group="resources",
        stage_rank=95,
        order=95,
        capture_rules=(
            _rule("artifact", "resource_bundle_artifact", priority=75),
            _rule("workspace", "task_workspace", priority=70),
        ),
    ),
    _metadata(
        "emotional_response",
        label="Emotional support",
        description="Produce a supportive learning response.",
        role="agent",
        group="support",
        stage_rank=70,
        order=70,
        capture_rules=(_rule("agent_output", "messages", priority=60),),
    ),
    _metadata(
        "handle_unknown",
        label="Classification failure",
        description="Handle an invalid or unsupported routing state.",
        role="output",
        group="routing",
        stage_rank=90,
        order=90,
        capture_rules=(_rule("agent_output", "messages", priority=40),),
    ),
):
    _register(item)


_RESOURCE_WORKFLOWS: dict[str, tuple[str, ...]] = {
    "review_doc": (
        "review_doc_planner",
        "review_doc_agent",
        "review_doc_reviewer",
        "review_doc_rewrite",
        "review_doc_output",
    ),
    "mindmap": (
        "mindmap_planner",
        "mindmap_agent",
        "mindmap_reviewer",
        "mindmap_rewrite",
        "mindmap_output",
    ),
    "quiz": (
        "exercise_planner",
        "exercise_agent",
        "exercise_reviewer",
        "exercise_rewrite",
        "exercise_output",
    ),
    "code_practice": (
        "code_practice_planner",
        "code_practice_agent",
        "code_practice_reviewer",
        "code_practice_rewrite",
        "code_practice_output",
    ),
    "video_script": (
        "video_script_planner",
        "video_script_agent",
        "video_script_reviewer",
        "video_script_rewrite",
        "video_script_output",
    ),
    "video_animation": (
        "video_animation_planner",
        "video_animation_agent",
        "video_animation_reviewer",
        "video_animation_rewrite",
        "video_animation_output",
    ),
    "study_plan": (
        "study_plan_emotional_intel",
        "study_plan_planner",
        "study_plan_agent",
        "study_plan_reviewer_academic",
        "study_plan_reviewer_emotional",
        "study_plan_consensus",
        "study_plan_rewrite",
        "study_plan_output",
    ),
}


def _resource_role(node_id: str) -> NodeRole:
    if "consensus" in node_id:
        return "consensus"
    if "reviewer" in node_id:
        return "reviewer"
    if node_id.endswith("_planner") or node_id.endswith("_rewrite"):
        return "planner"
    if node_id.endswith("_output"):
        return "output"
    return "agent"


def _resource_capture_rule(role: NodeRole) -> tuple[InfluenceCaptureRule, ...]:
    if role == "planner":
        return (
            _rule(
                "planner_output",
                "outline",
                "plan",
                "revision_notes",
                "messages",
                priority=75,
            ),
        )
    if role == "reviewer":
        return (
            _rule(
                "reviewer_output",
                "verdict",
                "reason",
                "review_reason",
                "revision_notes",
                priority=80,
            ),
        )
    if role == "consensus":
        return (
            _rule(
                "consensus_output",
                "consensus",
                "revision_notes",
                "reason",
                priority=85,
            ),
        )
    if role == "agent":
        return (
            _rule(
                "agent_output",
                "summary",
                "title",
                "messages",
                priority=70,
            ),
        )
    if role == "output":
        return (_rule("artifact", "artifact", "messages", priority=70),)
    return ()


for workflow, node_ids in _RESOURCE_WORKFLOWS.items():
    for order, node_id in enumerate(node_ids, start=1):
        if node_id in _NODE_METADATA:
            continue
        role = _resource_role(node_id)
        _register(
            _metadata(
                node_id,
                label=node_id.replace("_", " ").title(),
                description=f"Logical {role} step for the {workflow} workflow.",
                role=role,
                operation=("rewrite" if node_id.endswith("_rewrite") else role),
                group="resource_workflow",
                parent="resource_worker",
                workflow=workflow,
                iteration_field=f"{workflow}_round",
                stage_rank={
                    "planner": 62,
                    "agent": 66,
                    "reviewer": 72,
                    "consensus": 76,
                    "output": 88,
                }.get(role, 65),
                order=order,
                capture_rules=_resource_capture_rule(role),
            )
        )


for item in (
    _metadata(
        "web_research_planner",
        label="Web research plan",
        description="Build structured web research tasks.",
        role="planner",
        group="retrieval",
        stage_rank=36,
        capture_rules=(_rule("retrieval_plan", "tasks", "queries", priority=80),),
    ),
    _metadata(
        "web_source_summarizer",
        label="Web source summary",
        description="Summarize one web source for later evidence judging.",
        role="agent",
        group="retrieval",
        stage_rank=44,
        capture_rules=(_rule("web_evidence", "summary", "title", injectable=False),),
    ),
    _metadata(
        "evidence_item_grader",
        label="Evidence grading",
        description="Grade one evidence candidate without bypassing the judge.",
        role="judge",
        group="evidence",
        stage_rank=48,
        capture_rules=(_rule("evidence_judge", "keep", "reason", injectable=False),),
    ),
    _metadata(
        "evidence_sufficiency_judge",
        label="Evidence sufficiency",
        description="Assess answerability and coverage gaps.",
        role="judge",
        group="evidence",
        stage_rank=49,
        capture_rules=(
            _rule("evidence_judge", "overall_evidence_state", "decision_summary"),
            _rule("coverage_gap", list_fields=("coverage_gaps",), priority=85),
        ),
    ),
):
    _register(item)


def get_node_runtime_metadata(node_id: str) -> NodeRuntimeMetadata | None:
    return _NODE_METADATA.get(str(node_id or "").strip())


def get_registered_node_metadata() -> tuple[NodeRuntimeMetadata, ...]:
    return tuple(
        sorted(
            _NODE_METADATA.values(),
            key=lambda item: (item.group, item.order, item.node_id),
        )
    )


def get_resource_workflow_nodes() -> dict[str, tuple[str, ...]]:
    return dict(_RESOURCE_WORKFLOWS)
