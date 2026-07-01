"""Tests for deterministic Evidence Judge fallback behavior."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from src.graph.evidence import EvidenceJudgeOutput
from src.llm.structured_output import StructuredLLMResult, StructuredOutputError


def _state_with_candidates() -> dict:
    return {
        "messages": [HumanMessage(content="帮我生成一份 Python 的复习资料、思维导图和练习题")],
        "requested_resource_type": "review_doc",
        "requested_resource_types": ["review_doc", "mindmap", "quiz"],
        "learning_goal": "Python learning resources",
        "local_evidence_candidates": [
            {
                "evidence_id": "local:python:0",
                "source_type": "local_rag",
                "subject": "python",
                "role": "core_concept",
                "purpose": "course review",
                "source": "python_course.md",
                "content_preview": "Python functions and data structures",
            }
        ],
        "local_evidence_originals": {
            "local:python:0": {
                "content": "Python functions and data structures course notes.",
                "source": "python_course.md",
            }
        },
        "web_evidence_candidates": [
            {
                "evidence_id": "web:python:0:0",
                "source_type": "web",
                "provider": "tavily",
                "subject": "python",
                "role": "supporting_context",
                "purpose": "resource_enrichment",
                "title": "Python docs",
                "url": "https://docs.python.org/3/tutorial/",
                "source": "https://docs.python.org/3/tutorial/",
                "content_preview": "Official Python tutorial",
            }
        ],
        "web_evidence_originals": {
            "web:python:0:0": {
                "title": "Python docs",
                "url": "https://docs.python.org/3/tutorial/",
                "content": "Official Python tutorial content.",
            }
        },
    }


@pytest.mark.anyio
async def test_evidence_judge_falls_back_on_business_validation_error(monkeypatch):
    import src.graph.academic as academic

    async def fake_invoke_structured_llm(**kwargs):
        result = StructuredLLMResult(
            success=False,
            parsed=None,
            node_name="evidence_judge",
            llm_node="evidence_judge",
            schema_name="EvidenceJudgeOutput",
            provider="test",
            model="test",
            output_mode="native_json_schema_pydantic",
            failure_phase="business_validation_error",
            error_type="BusinessValidationError",
            business_validation_error=(
                "missing evidence_id values: ['web:python:0:0']; "
                "expected 2 judged evidence items, got 0"
            ),
        )
        raise StructuredOutputError(result)

    monkeypatch.setattr(academic, "invoke_structured_llm", fake_invoke_structured_llm)

    result = await academic.evidence_judge(_state_with_candidates())

    judge_output = result["evidence_judge_output"]
    judged_ids = [item["evidence_id"] for item in judge_output["judged_evidence"]]
    assert result["evidence_judge_failed"] is True
    assert result["evidence_judge_state"] == "partially_sufficient"
    assert result["degraded_generation"] is True
    assert result["degraded_reason"] == "Evidence Judge validation failed; fallback evidence selection was used."
    assert result["evidence_coverage_gaps"] == []
    assert judged_ids == ["local:python:0", "web:python:0:0"]
    assert {item["evidence_id"] for item in result["context"]} == set(judged_ids)


@pytest.mark.anyio
async def test_evidence_judge_falls_back_when_judged_evidence_is_empty(monkeypatch):
    import src.graph.academic as academic

    async def fake_invoke_structured_llm(**kwargs):
        return StructuredLLMResult(
            success=True,
            parsed=EvidenceJudgeOutput(
                overall_evidence_state="insufficient",
                need_more_web_research=True,
                judged_evidence=[],
                coverage_gaps=[],
                decision_summary="The model omitted judged_evidence.",
            ),
            node_name="evidence_judge",
            llm_node="evidence_judge",
            schema_name="EvidenceJudgeOutput",
            provider="test",
            model="test",
            output_mode="native_json_schema_pydantic",
        )

    monkeypatch.setattr(academic, "invoke_structured_llm", fake_invoke_structured_llm)

    result = await academic.evidence_judge(_state_with_candidates())

    judge_output = result["evidence_judge_output"]
    assert result["evidence_judge_failed"] is True
    assert result["evidence_judge_state"] == "partially_sufficient"
    assert judge_output["need_more_web_research"] is False
    assert len(judge_output["judged_evidence"]) == 2
    assert len(result["context"]) == 2
