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
            await study_plan_planner({"messages": [HumanMessage(content="make a plan")]})


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
    assert route_after_study_plan_consensus({"study_plan_consensus": False}) == "rewrite"


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
