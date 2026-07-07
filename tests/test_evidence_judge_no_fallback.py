"""Evidence Judge failures must not synthesize fallback results."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from src.graph.evidence import EvidenceJudgeOutput
from src.llm.structured_output import StructuredLLMResult, StructuredOutputError


def _state_with_candidates() -> dict:
    return {
        "messages": [
            HumanMessage(
                content="Generate Python review notes, a mind map, and practice questions."
            )
        ],
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
async def test_evidence_judge_raises_on_business_validation_error_without_fallback(
    monkeypatch,
):
    import src.graph.academic as academic

    async def fake_invoke_structured_llm(**kwargs):
        result = StructuredLLMResult(
            success=False,
            parsed=None,
            node_name=kwargs["node_name"],
            llm_node="evidence_judge",
            schema_name=kwargs["schema"].__name__,
            provider="test",
            model="test",
            output_mode="native_json_schema_pydantic",
            failure_phase="business_validation_error",
            error_type="BusinessValidationError",
            business_validation_error=(
                "missing evidence_id values: ['web:python:0:0']; "
                "expected 2 judged evidence items, got 0"
            ),
            raw_output='{"judged_evidence": []}',
        )
        raise StructuredOutputError(result)

    monkeypatch.setattr(academic, "invoke_structured_llm", fake_invoke_structured_llm)

    with pytest.raises(RuntimeError) as exc_info:
        await academic.evidence_judge(_state_with_candidates())

    message = str(exc_info.value)
    assert "Evidence Judge failed without fallback" in message
    assert "BusinessValidationError" in message
    assert "web:python:0:0" in message


@pytest.mark.anyio
async def test_evidence_judge_raises_when_judged_evidence_is_empty_without_fallback(
    monkeypatch,
):
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
            node_name=kwargs["node_name"],
            llm_node="evidence_judge",
            schema_name=kwargs["schema"].__name__,
            provider="test",
            model="test",
            output_mode="native_json_schema_pydantic",
        )

    monkeypatch.setattr(academic, "invoke_structured_llm", fake_invoke_structured_llm)

    with pytest.raises(RuntimeError) as exc_info:
        await academic.evidence_judge(_state_with_candidates())

    message = str(exc_info.value)
    assert "Evidence Judge failed without fallback" in message
    assert "InvalidStructuredResult" in message
