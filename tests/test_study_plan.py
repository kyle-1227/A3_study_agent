"""Tests for the Evidence-Judge-backed study plan resource agent."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from src.graph.study_plan import (
    StudyPlanArtifact,
    StudyPlanPhase,
    StudyPlanReviewVerdict,
    route_after_study_plan_consensus,
    study_plan_consensus,
    study_plan_output,
    study_plan_planner,
    study_plan_profile_gate,
    validate_study_plan_artifact,
)


def _valid_artifact() -> StudyPlanArtifact:
    return StudyPlanArtifact(
        title="Personalized Study Plan",
        learner_profile_summary="Learner needs a structured path.",
        overall_goal="Build a reliable foundation.",
        phases=[
            StudyPlanPhase(
                title="Foundation",
                duration="Week 1",
                goals=["Understand basics"],
                tasks=["Read core notes"],
                resources=["Judged evidence 1"],
                practice=["Small exercise"],
                checkpoints=["Explain the core idea"],
            ),
            StudyPlanPhase(
                title="Practice",
                duration="Week 2",
                goals=["Apply basics"],
                tasks=["Complete a project task"],
                resources=["Judged evidence 2"],
                practice=["Project drill"],
                checkpoints=["Review project output"],
            ),
        ],
        weekly_schedule=["Week 1: foundation", "Week 2: practice"],
        milestones=["Finish foundation", "Finish practice"],
        practice_tasks=["Exercise set", "Mini project"],
        risk_warnings=["Keep workload realistic"],
        evidence_usage=["Used judged course evidence"],
    )


@pytest.mark.anyio
async def test_study_plan_planner_empty_outline_raises():
    with patch("src.graph.study_plan.invoke_plain_llm_fail_fast", return_value="   "):
        with pytest.raises(ValueError, match="empty outline"):
            await study_plan_planner(
                {"messages": [HumanMessage(content="make a plan")]}
            )


def test_validate_study_plan_artifact_rejects_missing_evidence_usage():
    artifact = _valid_artifact()
    artifact.evidence_usage = []
    assert "evidence_usage" in validate_study_plan_artifact(artifact)


@pytest.mark.anyio
async def test_study_plan_consensus_max_rounds_rejected_raises():
    state = {
        "study_plan_round": 3,
        "study_plan_academic_verdict": "reject",
        "study_plan_academic_reason": "Missing evidence",
        "study_plan_emotional_verdict": "approve",
        "study_plan_emotional_reason": "Workload is acceptable",
    }
    with pytest.raises(RuntimeError, match="max rounds"):
        await study_plan_consensus(state)


def test_route_after_study_plan_consensus():
    assert route_after_study_plan_consensus({"study_plan_consensus": True}) == "output"
    assert (
        route_after_study_plan_consensus({"study_plan_consensus": False}) == "rewrite"
    )


@pytest.mark.anyio
async def test_study_plan_profile_gate_interrupts_when_profile_missing():
    resume_value = {
        "type": "profile_completion_required",
        "profile_completion": {
            "learning_goal": "Master ML basics",
            "current_foundation": "Knows Python",
            "daily_study_time": "2 hours",
            "deadline": "8 weeks",
            "preferred_learning_style": "Examples first",
            "weak_points": "Math notation",
        },
    }

    with patch(
        "src.graph.study_plan.interrupt", return_value=resume_value
    ) as mock_interrupt:
        result = await study_plan_profile_gate(
            {
                "request_id": "r1",
                "thread_id": "t1",
                "subject": "machine learning",
                "learning_goal": "ML basics",
            }
        )

    mock_interrupt.assert_called_once()
    interrupt_payload = mock_interrupt.call_args.args[0]
    assert interrupt_payload["type"] == "profile_completion_required"
    assert interrupt_payload["resume_available"] is True
    assert interrupt_payload["profile_completion_request"]["fields"]
    assert result["learner_profile"]["learning_goal"] == "Master ML basics"
    assert "学习目标: Master ML basics" in result["learner_profile_summary"]
    assert result["profile_summary"] == result["learner_profile_summary"]
    assert result["task_workspace"]["profile_requirements"]
    assert result["task_workspace"]["constraints"]


@pytest.mark.anyio
async def test_study_plan_profile_gate_skips_when_profile_present():
    with patch("src.graph.study_plan.interrupt") as mock_interrupt:
        result = await study_plan_profile_gate(
            {
                "profile_summary": "Goal: master ML basics",
                "thread_id": "t1",
                "request_id": "r1",
            }
        )

    mock_interrupt.assert_not_called()
    assert result == {"profile_completion_request": {}}


@pytest.mark.anyio
async def test_study_plan_output_empty_artifact_raises():
    with pytest.raises(ValueError, match="artifact"):
        await study_plan_output({})


@pytest.mark.anyio
async def test_study_plan_output_renders_markdown_and_artifact():
    artifact = _valid_artifact()
    with patch(
        "src.graph.study_plan.create_markdown_artifact",
        return_value={"title": "Personalized Study Plan", "filename": "study-plan.md"},
    ):
        result = await study_plan_output({"study_plan_artifact": artifact.model_dump()})

    assert "Personalized Study Plan" in result["study_plan_markdown"]
    assert result["study_plan_document_artifact"]["filename"] == "study-plan.md"
    assert result["messages"]


def test_review_verdict_schema_accepts_review_output():
    verdict = StudyPlanReviewVerdict(
        verdict="approve",
        reason="Plan is coherent.",
        revision_notes=[],
        risk_flags=[],
    )
    assert verdict.verdict == "approve"
