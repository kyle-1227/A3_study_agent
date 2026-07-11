"""Cross-module Phase 7 acceptance tests without external LLM calls."""

from __future__ import annotations

from typing import Any

import pytest

from app import _resource_final_payload
from src.context_engineering.packing.node_policy import SourceBudgetPolicy
from src.context_engineering.packing.source_policy import (
    filter_context_items_by_source_policy,
)
from src.context_engineering.schema import ContextItem
from src.graph.builder import route_after_evidence_judge
from src.graph.qa import (
    QAResponse,
    QASuggestion,
    build_qa_final_payload,
    validate_qa_response,
)
from src.graph.resource_generation import (
    resource_bundle_output,
    resource_orchestrator,
    resource_preflight_router,
    route_after_resource_preflight,
)
from src.graph.study_plan import study_plan_profile_gate_main
from src.graph.supervisor import (
    SupervisorOutput,
    route_after_supervisor,
    validate_supervisor_output,
)
from src.observability.activity import build_activity_event, merge_activity_timeline


@pytest.mark.anyio
async def test_profile_interrupt_resumes_once_then_fan_in_builds_one_bundle(
    monkeypatch,
):
    from src.graph import study_plan as study_plan_module

    state = {
        "thread_id": "phase7-thread",
        "request_id": "phase7-resource-request",
        "subject": "Machine Learning",
        "primary_subject": "Machine Learning",
        "requested_resource_types": ["study_plan", "mindmap"],
        "requested_resource_type": "study_plan",
        "learner_profile": {},
    }
    preflight = await resource_preflight_router(state)
    gated_state = {**state, **preflight}
    assert route_after_resource_preflight(gated_state) == (
        "study_plan_profile_gate_main"
    )

    interrupts: list[dict[str, Any]] = []

    def complete_profile(payload):
        interrupts.append(payload)
        return {
            "type": "profile_completion_required",
            "profile_completion": {
                "learning_goal": "Build machine learning foundations",
                "current_foundation": "Comfortable with Python",
                "daily_study_time": "90 minutes",
            },
        }

    monkeypatch.setattr(study_plan_module, "interrupt", complete_profile)

    profile_update = await study_plan_profile_gate_main(gated_state)
    completed_profile_state = {**gated_state, **profile_update}
    await study_plan_profile_gate_main(completed_profile_state)

    assert len(interrupts) == 1
    assert completed_profile_state["learner_profile"] == {
        "learning_goal": "Build machine learning foundations",
        "current_foundation": "Comfortable with Python",
        "daily_study_time": "90 minutes",
    }

    plan_update = await resource_orchestrator(completed_profile_state)
    branch_state = {
        **completed_profile_state,
        **plan_update,
        "resource_branch_results": [
            {
                "resource_type": "study_plan",
                "status": "success",
                "title": "Machine Learning Study Plan",
                "message_preview": "Study plan generated.",
                "artifact": {
                    "title": "Machine Learning Study Plan",
                    "filename": "study-plan.md",
                    "markdown_url": "/artifacts/study-plans/phase7/plan.md",
                },
                "state_updates": {
                    "study_plan_artifact": {
                        "title": "Machine Learning Study Plan",
                        "phases": [{"title": "Foundations"}],
                    },
                    "study_plan_document_artifact": {
                        "title": "Machine Learning Study Plan",
                        "filename": "study-plan.md",
                        "markdown_url": "/artifacts/study-plans/phase7/plan.md",
                    },
                },
                "elapsed_ms": 20,
            },
            {
                "resource_type": "mindmap",
                "status": "success",
                "title": "Machine Learning Map",
                "message_preview": "Mindmap generated.",
                "artifact": {
                    "title": "Machine Learning Map",
                    "filename": "mindmap.html",
                    "html_url": "/artifacts/mindmaps/phase7/map.html",
                },
                "state_updates": {
                    "mindmap_artifact": {
                        "title": "Machine Learning Map",
                        "tree": {
                            "title": "Machine Learning",
                            "children": [],
                        },
                    },
                    "mindmap_tree": {
                        "title": "Machine Learning",
                        "children": [],
                    },
                },
                "elapsed_ms": 15,
            },
        ],
    }
    bundle_update = await resource_bundle_output(branch_state)
    final_state = {**branch_state, **bundle_update}
    resource_final = _resource_final_payload(final_state)

    assert bundle_update["resource_generation_status"] == "success"
    assert bundle_update["resource_bundle_artifact"]["success_count"] == 2
    assert len(bundle_update["last_generated_artifacts"]) == 2
    assert resource_final is not None
    assert resource_final["type"] == "resource_final"
    assert resource_final["resource_type"] == "bundle"
    assert resource_final["resource"]["kind"] == "bundle"

    activities = [
        build_activity_event(
            thread_id=state["thread_id"],
            request_id=state["request_id"],
            sequence=index,
            kind="artifact",
            status="completed",
            activity_key=f"resource:{resource_type}",
            title=f"{resource_type} completed",
            now=f"2026-07-11T00:00:0{index}+00:00",
        ).model_dump(mode="json")
        for index, resource_type in enumerate(("study_plan", "mindmap"), start=1)
    ]
    timeline = merge_activity_timeline([], [*activities, activities[0]])
    assert len(timeline) == 2
    assert len({item["activity_id"] for item in timeline}) == 2


@pytest.mark.parametrize(
    (
        "intent",
        "qa_scope",
        "requires_live_verification",
        "supervisor_route",
        "grounding_status",
        "kept_evidence_count",
        "uncertainty_note",
    ),
    [
        (
            "unknown",
            "general",
            True,
            "qa",
            "not_live_verified",
            0,
            "This answer was not live-verified.",
        ),
        (
            "academic",
            "academic",
            False,
            "academic",
            "judged_evidence",
            1,
            "",
        ),
        (
            "unknown",
            "a3_agent",
            False,
            "qa",
            "capability_registry",
            0,
            "",
        ),
    ],
)
def test_qa_modes_share_strict_routing_grounding_and_final_contract(
    intent,
    qa_scope,
    requires_live_verification,
    supervisor_route,
    grounding_status,
    kept_evidence_count,
    uncertainty_note,
):
    supervisor = SupervisorOutput(
        intent=intent,
        response_mode="qa",
        qa_scope=qa_scope,
        requires_live_verification=requires_live_verification,
        keywords=["bounded"],
        confidence=0.9,
        subject_candidates=[],
        requested_resource_type="",
        requested_resource_types=[],
    )
    assert validate_supervisor_output(supervisor) == ""
    supervisor_state = supervisor.model_dump()
    assert route_after_supervisor(supervisor_state) == supervisor_route
    if qa_scope == "academic":
        assert route_after_evidence_judge(supervisor_state) == "qa"

    response = QAResponse(
        answer="A bounded answer.",
        uncertainty_note=uncertainty_note,
        grounding_status=grounding_status,
        suggestions=[QASuggestion(label="Continue", action="continue_qa")],
    )
    assert (
        validate_qa_response(
            response,
            qa_scope=qa_scope,
            kept_evidence_count=kept_evidence_count,
            requires_live_verification=requires_live_verification,
        )
        == ""
    )
    final = build_qa_final_payload(
        response=response,
        qa_scope=qa_scope,
        thread_id="phase7-thread",
        request_id=f"phase7-{qa_scope}",
    )
    assert final["type"] == "qa_final"
    assert final["qa_scope"] == qa_scope
    assert final["response"]["grounding_status"] == grounding_status


def _context_item(
    item_id: str,
    *,
    source_type: str,
    metadata: dict[str, Any],
    relevance_score: float = 0.1,
) -> ContextItem:
    return ContextItem(
        id=item_id,
        source_type=source_type,
        title=item_id,
        content="bounded context",
        token_estimate=4,
        estimated=True,
        tokenizer_mode="estimated_mixed",
        priority=10,
        relevance_score=relevance_score,
        recency_score=0.8,
        confidence=0.8,
        scope="session",
        lifetime="session",
        compressible=True,
        can_drop=True,
        disclosure_level="snippet",
        metadata=metadata,
    )


def test_broad_expands_business_selection_but_never_safety_boundaries():
    artifact_policy = SourceBudgetPolicy(
        source_type="artifact",
        min_priority=90,
        min_relevance_score=0.9,
        allowed_purposes=("different_purpose",),
        require_thread_match=True,
        require_subject_match=True,
        require_task_match=True,
        stale_policy="drop",
    )
    same_thread = _context_item(
        "same-thread",
        source_type="artifact",
        metadata={
            "thread_id": "phase7-thread",
            "subject": "different subject",
            "resource_type": "review_doc",
            "purpose": "artifact_reference",
            "stale": True,
        },
    )
    other_thread = same_thread.model_copy(
        update={
            "id": "other-thread",
            "metadata": {**same_thread.metadata, "thread_id": "other-thread"},
        }
    )
    kwargs = {
        "injectable_sources": ("artifact",),
        "exclude_message_source": True,
        "source_policies": {"artifact": artifact_policy},
        "state": {
            "thread_id": "phase7-thread",
            "subject": "Machine Learning",
            "requested_resource_type": "mindmap",
        },
    }

    strict = filter_context_items_by_source_policy(
        [same_thread, other_thread],
        policy_mode="strict",
        **kwargs,
    )
    broad = filter_context_items_by_source_policy(
        [same_thread, other_thread],
        policy_mode="broad",
        **kwargs,
    )

    assert strict.kept_items == []
    assert [item.id for item in broad.kept_items] == ["same-thread"]
    assert broad.source_drop_reasons["thread_mismatch"] == 1
    assert "broad_business_filters_bypassed" in broad.warnings

    unapproved = _context_item(
        "unapproved-evidence",
        source_type="evidence",
        metadata={"grounding_approved": False},
        relevance_score=0.9,
    )
    for mode in ("strict", "broad"):
        evidence_result = filter_context_items_by_source_policy(
            [unapproved],
            injectable_sources=("evidence",),
            exclude_message_source=True,
            source_policies={"evidence": SourceBudgetPolicy(source_type="evidence")},
            state={"thread_id": "phase7-thread"},
            policy_mode=mode,
        )
        assert evidence_result.kept_items == []
        assert evidence_result.source_drop_reasons == {"grounding_not_approved": 1}
