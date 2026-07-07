from __future__ import annotations

import logging

import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from src.graph.academic import (
    _assert_no_silent_fallback,
    _build_evidence_item_grader_messages,
    _build_evidence_sufficiency_messages,
    _grade_evidence_items_with_llm,
    _judge_evidence_candidates_with_llm,
    _last_failed_execution_stage,
    _trace_parse_evidence_grade_raw,
    evidence_judge,
    validate_evidence_grade_batch_output,
    validate_evidence_sufficiency_output,
)
from src.graph.evidence import (
    EvidenceCandidate,
    EvidenceCoverageGap,
    EvidenceGradeBatch,
    EvidenceJudgeItem,
    EvidenceJudgeOutput,
    EvidenceSufficiencyOutput,
)
from src.llm.schema_drift import analyze_schema_drift_trace_only
from src.llm.schema_manifest import build_canonical_manifest, load_drift_guard_config
from src.llm.structured_output import StructuredLLMResult, StructuredOutputError, _build_reask_context


def _candidate(evidence_id: str = "local:python:0") -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        source_type="local_rag",
        provider="chroma_rag",
        subject="python",
        role="core_concept",
        title="Python notes",
        source="python.pdf",
        content_preview="Python functions and return values.",
    )


def _judge_item(
    evidence_id: str = "local:python:0",
    *,
    keep: bool = True,
    quality: str = "high",
    use_case: str = "core_evidence",
    evidence_score: float = 0.9,
    score_reason: str = "Directly supports the requested concept.",
    coverage_contribution: str = "Covers the requested concept.",
    reason: str = "Useful evidence.",
) -> EvidenceJudgeItem:
    return EvidenceJudgeItem(
        evidence_id=evidence_id,
        keep=keep,
        final_quality=quality,
        relevance="high" if keep else "low",
        authority="medium",
        usefulness="high" if keep else "low",
        risk="low",
        evidence_score=evidence_score,
        score_reason=score_reason,
        evidence_type="local_textbook_chunk",
        use_case=use_case,
        coverage_contribution=coverage_contribution,
        reason=reason if keep else reason or "Not useful.",
    )


def _state() -> dict:
    return {
        "messages": [HumanMessage(content="Explain Python functions")],
        "expanded_keypoints": ["Python functions"],
        "request_id": "req",
        "thread_id": "thread",
    }


def _structured_result(*, parsed, node_name: str, schema_name: str, output_mode: str = "native_json_schema_pydantic"):
    return StructuredLLMResult(
        success=True,
        parsed=parsed,
        node_name=node_name,
        llm_node="evidence_judge",
        schema_name=schema_name,
        provider="unit",
        model="unit",
        output_mode=output_mode,
        raw_output=parsed.model_dump_json() if parsed is not None else "",
    )


def _structured_error(
    *,
    node_name: str,
    schema_name: str,
    message: str,
    business_error: str = "",
    raw_output: str = "{}",
    extra_debug: dict | None = None,
):
    return StructuredOutputError(
        StructuredLLMResult(
            success=False,
            parsed=None,
            node_name=node_name,
            llm_node="evidence_judge",
            schema_name=schema_name,
            provider="unit",
            model="unit",
            output_mode="native_json_schema_pydantic",
            failure_phase="business_validation_error" if business_error else "validation_error",
            error_type="BusinessValidationError" if business_error else "ValidationError",
            error_message=message,
            business_validation_error=business_error,
            validation_error="" if business_error else message,
            raw_output=raw_output,
            extra_debug=extra_debug or {},
        )
    )


def test_evidence_grade_batch_schema_validates_and_caps_items():
    batch = EvidenceGradeBatch(judged_evidence=[_judge_item()])

    assert batch.judged_evidence[0].evidence_id == "local:python:0"
    assert batch.judged_evidence[0].evidence_score == pytest.approx(0.9)

    with pytest.raises(ValidationError):
        EvidenceGradeBatch(
            judged_evidence=[
                _judge_item(f"local:python:{index}")
                for index in range(9)
            ]
        )


def test_evidence_judge_item_requires_explicit_coverage_contribution():
    with pytest.raises(ValidationError, match="coverage_contribution"):
        EvidenceJudgeItem(
            evidence_id="local:python:0",
            keep=True,
            reason="Useful evidence.",
        )

    with pytest.raises(ValidationError, match="coverage_contribution"):
        EvidenceJudgeItem(
            evidence_id="local:python:0",
            keep=False,
            reason="Discarded evidence.",
        )

    with pytest.raises(ValidationError, match="coverage_contribution must not be empty"):
        _judge_item(coverage_contribution="")

    discarded = _judge_item(
        keep=False,
        coverage_contribution="",
        reason="Not relevant to the request.",
    )
    assert discarded.coverage_contribution == ""

    kept = _judge_item(coverage_contribution="Covers Python quiz practice.")
    assert kept.coverage_contribution == "Covers Python quiz practice."


def test_evidence_judge_item_requires_llm_evidence_score():
    with pytest.raises(ValidationError, match="evidence_score"):
        EvidenceJudgeItem(
            evidence_id="local:python:0",
            keep=True,
            final_quality="high",
            relevance="high",
            authority="medium",
            usefulness="high",
            risk="low",
            score_reason="Good support.",
            evidence_type="local_textbook_chunk",
            use_case="core_evidence",
            coverage_contribution="Covers Python functions.",
            reason="Useful evidence.",
        )

    with pytest.raises(ValidationError, match="less than or equal to 1"):
        _judge_item(evidence_score=1.5)

    with pytest.raises(ValidationError, match="score_reason"):
        _judge_item(score_reason="")


def test_evidence_descriptions_are_canonical_without_alias_lists():
    coverage_description = EvidenceJudgeItem.model_fields["coverage_contribution"].description or ""
    reason_description = EvidenceJudgeItem.model_fields["reason"].description or ""
    score_description = EvidenceJudgeItem.model_fields["evidence_score"].description or ""
    gap_description = EvidenceCoverageGap.model_fields["gap"].description or ""

    assert "what coverage the evidence contributes" in coverage_description
    assert "why the grading decision was made" in reason_description
    assert "supports the current user request" in score_description
    assert "missing or weak" in gap_description
    for forbidden_alias in ("coverage_reason", "support_reason", "topic", "followup_query"):
        assert forbidden_alias not in coverage_description
        assert forbidden_alias not in reason_description
        assert forbidden_alias not in gap_description


def test_evidence_alias_drift_is_reported_but_not_normalized():
    with pytest.raises(ValidationError):
        EvidenceGradeBatch.model_validate({
            "judged_evidence": [
                {
                    "evidence_id": "local:python:0",
                    "keep": True,
                    "final_quality": "high",
                    "relevance": "high",
                    "authority": "medium",
                    "usefulness": "high",
                    "risk": "low",
                    "evidence_score": 0.88,
                    "score_reason": "Direct support.",
                    "evidence_type": "local_textbook_chunk",
                    "use_case": "core_evidence",
                    "coverage_reason": "Covers functions.",
                    "reason": "Useful.",
                }
            ]
        })

    manifest = build_canonical_manifest(
        EvidenceGradeBatch,
        node_name="evidence_item_grader",
        output_mode="deepseek_tool_call_strict",
    )
    drift_guard = load_drift_guard_config("evidence_item_grader")
    report = analyze_schema_drift_trace_only(
        {
            "judged_evidence": [
                {
                    "evidence_id": "local:python:0",
                    "keep": True,
                    "final_quality": "high",
                    "relevance": "high",
                    "authority": "medium",
                    "usefulness": "high",
                    "risk": "low",
                    "evidence_score": 0.88,
                    "score_reason": "Direct support.",
                    "evidence_type": "local_textbook_chunk",
                    "use_case": "core_evidence",
                    "coverage_reason": "Covers functions.",
                    "reason": "Useful.",
                }
            ]
        },
        manifest=manifest,
        drift_guard=drift_guard,
        node_name="evidence_item_grader",
    )

    assert report.alias_hits_by_path["judged_evidence[0].coverage_reason"] == "coverage_contribution"
    assert "judged_evidence[0].coverage_contribution" in report.missing_required_by_path


def _coverage_gap(index: int = 0, *, query: str = "python function exercises") -> EvidenceCoverageGap:
    return EvidenceCoverageGap(
        subject="python",
        role="core_concept",
        gap=f"gap {index}",
        suggested_search_query=query,
        purpose="coverage_expansion",
        priority=0.5,
    )


def test_evidence_sufficiency_output_schema_validates_limits():
    output = EvidenceSufficiencyOutput(
        overall_evidence_state="sufficient",
        answerability="can_answer",
        need_more_local_rag=False,
        need_more_web_research=False,
        coverage_gaps=[],
        decision_summary="Enough evidence.",
    )

    assert output.answerability == "can_answer"

    output_with_max_gaps = EvidenceSufficiencyOutput(
        overall_evidence_state="partially_sufficient",
        answerability="can_answer_with_caveats",
        need_more_local_rag=False,
        need_more_web_research=True,
        coverage_gaps=[_coverage_gap(index) for index in range(10)],
        decision_summary="Needs broader coverage.",
    )

    assert len(output_with_max_gaps.coverage_gaps) == 10

    with pytest.raises(ValidationError):
        EvidenceSufficiencyOutput(
            overall_evidence_state="partially_sufficient",
            answerability="can_answer_with_caveats",
            need_more_local_rag=False,
            need_more_web_research=True,
            coverage_gaps=[_coverage_gap(index) for index in range(11)],
            decision_summary="Needs a bit more coverage.",
        )

    with pytest.raises(ValidationError):
        EvidenceSufficiencyOutput(
            overall_evidence_state="partially_sufficient",
            answerability="can_answer_with_caveats",
            need_more_local_rag=False,
            need_more_web_research=True,
            coverage_gaps=[],
            decision_summary="x" * 601,
        )


def test_sufficiency_gap_alias_drift_is_reported_but_not_normalized():
    with pytest.raises(ValidationError):
        EvidenceSufficiencyOutput.model_validate({
            "overall_evidence_state": "insufficient",
            "answerability": "cannot_answer",
            "need_more_local_rag": True,
            "need_more_web_research": True,
            "coverage_gaps": [
                {
                    "subject": "python",
                    "role": "core_concept",
                    "topic": "Missing practice exercises",
                    "query": "Python functions practice questions",
                    "purpose": "coverage_expansion",
                    "priority": 0.8,
                }
            ],
            "decision_summary": "Need more practice evidence.",
        })

    manifest = build_canonical_manifest(
        EvidenceSufficiencyOutput,
        node_name="evidence_sufficiency_judge",
        output_mode="deepseek_tool_call_strict",
    )
    drift_guard = load_drift_guard_config("evidence_sufficiency_judge")
    report = analyze_schema_drift_trace_only(
        {
            "overall_evidence_state": "insufficient",
            "answerability": "cannot_answer",
            "need_more_local_rag": True,
            "need_more_web_research": True,
            "coverage_gaps": [
                {
                    "subject": "python",
                    "role": "core_concept",
                    "topic": "Missing practice exercises",
                    "query": "Python functions practice questions",
                    "purpose": "coverage_expansion",
                    "priority": 0.8,
                }
            ],
            "decision_summary": "Need more practice evidence.",
        },
        manifest=manifest,
        drift_guard=drift_guard,
        node_name="evidence_sufficiency_judge",
    )

    assert report.alias_hits_by_path["coverage_gaps[0].topic"] == "gap"
    assert report.alias_hits_by_path["coverage_gaps[0].query"] == "suggested_search_query"
    assert "coverage_gaps[0].gap" in report.missing_required_by_path
    assert "coverage_gaps[0].suggested_search_query" in report.missing_required_by_path


def test_sufficiency_reask_prompt_contains_manifest_and_drift_report():
    result = StructuredLLMResult(
        success=False,
        parsed=None,
        node_name="evidence_sufficiency_judge",
        llm_node="evidence_judge",
        schema_name="EvidenceSufficiencyOutput",
        provider="unit",
        model="unit",
        output_mode="deepseek_tool_call_strict",
        failure_phase="validation_error",
        error_type="ValidationError",
        error_message="coverage_gaps.0.gap Field required",
        validation_error="coverage_gaps.0.gap Field required",
        raw_output=(
            '{"overall_evidence_state":"insufficient","answerability":"cannot_answer",'
            '"need_more_local_rag":true,"need_more_web_research":true,'
            '"coverage_gaps":[{"topic":"Missing practice","query":"Python practice"}],'
            '"decision_summary":"Need more evidence."}'
        ),
    )

    context = _build_reask_context(
        node_name="evidence_sufficiency_judge",
        schema_name="EvidenceSufficiencyOutput",
        schema=EvidenceSufficiencyOutput,
        result=result,
        attempt_number=1,
    )

    assert context is not None
    assert "Canonical schema manifest" in context.instruction
    assert "Schema drift report" in context.instruction
    assert "coverage_gaps[].gap" in context.instruction
    assert "coverage_gaps[0].topic" in context.instruction
    assert context.schema_drift_report["alias_hits_by_path"]["coverage_gaps[0].topic"] == "gap"


def test_grade_batch_validator_finds_missing_duplicate_and_unknown_ids():
    missing = EvidenceGradeBatch(judged_evidence=[_judge_item("local:python:0")])
    duplicate = EvidenceGradeBatch(
        judged_evidence=[
            _judge_item("local:python:0"),
            _judge_item("local:python:0"),
        ]
    )
    unknown = EvidenceGradeBatch(
        judged_evidence=[
            _judge_item("local:python:0"),
            _judge_item("local:python:extra"),
        ]
    )
    count_mismatch = EvidenceGradeBatch(judged_evidence=[_judge_item("local:python:0")])

    assert "missing" in validate_evidence_grade_batch_output(
        missing,
        expected_ids=["local:python:0", "local:python:1"],
    )
    assert "duplicate" in validate_evidence_grade_batch_output(
        duplicate,
        expected_ids=["local:python:0", "local:python:1"],
    )
    assert "unknown" in validate_evidence_grade_batch_output(
        unknown,
        expected_ids=["local:python:0", "local:python:1"],
    )
    assert "expected 2 judged evidence items, got 1" in validate_evidence_grade_batch_output(
        count_mismatch,
        expected_ids=["local:python:0", "local:python:1"],
    )


def test_grade_batch_validator_rejects_empty_reason_and_accepts_discarded_empty_coverage():
    empty_reason = EvidenceGradeBatch(judged_evidence=[
        _judge_item(reason=""),
    ])
    discarded_empty_coverage = EvidenceGradeBatch(judged_evidence=[
        _judge_item(
            keep=False,
            coverage_contribution="",
            reason="Not relevant to the request.",
        ),
    ])

    assert "reason must not be empty" in validate_evidence_grade_batch_output(
        empty_reason,
        expected_ids=["local:python:0"],
    )
    assert validate_evidence_grade_batch_output(
        discarded_empty_coverage,
        expected_ids=["local:python:0"],
    ) == ""


def test_trace_helpers_extract_failed_stage_and_raw_grade_counts():
    debug = {
        "stages": [
            {"stage": "first", "status": "failed", "error_type": "FirstError"},
            {"stage": "second", "status": "success"},
            {"stage": "last", "status": "failed", "error_type": "LastError"},
        ]
    }
    assert _last_failed_execution_stage(debug)["stage"] == "last"

    parsed = _trace_parse_evidence_grade_raw(
        '{"judged_evidence": ['
        '{"evidence_id": "local:python:0", "keep": true},'
        '{"evidence_id": "web:python:1", "keep": false}'
        ']}'
    )
    assert parsed["judged_ids"] == ["local:python:0", "web:python:1"]
    assert parsed["kept_count"] == 1
    assert parsed["rejected_count"] == 1


def test_evidence_prompts_include_json_requested_resource_types():
    candidate = _candidate()
    item_messages = _build_evidence_item_grader_messages(
        candidates=[candidate],
        original_user_query="给我一份python思维导图和练习题",
        learning_goal="Learn Python with a mindmap and quiz",
        requested_resource_type="mindmap",
        requested_resource_types=["mindmap", "quiz"],
        batch_index=0,
    )
    item_prompt = item_messages[-1]["content"]
    assert "- requested_resource_type: mindmap" in item_prompt
    assert '["mindmap", "quiz"]' in item_prompt
    assert "evidence_score is mandatory" in item_prompt
    assert '"evidence_score": 0.92' in item_prompt

    sufficiency_messages = _build_evidence_sufficiency_messages(
        candidates=[candidate],
        judged_items=[_judge_item()],
        original_user_query="给我一份python思维导图和练习题",
        learning_goal="Learn Python with a mindmap and quiz",
        requested_resource_type="mindmap",
        requested_resource_types=["mindmap", "quiz"],
        expanded_keypoints=["Python basics"],
    )
    assert '["mindmap", "quiz"]' in sufficiency_messages[-1]["content"]


def test_sufficiency_validator_rules():
    sufficient_zero_kept = EvidenceSufficiencyOutput(
        overall_evidence_state="sufficient",
        answerability="can_answer",
        need_more_local_rag=False,
        need_more_web_research=False,
        coverage_gaps=[],
        decision_summary="Enough evidence.",
    )
    sufficient_wrong_answerability = EvidenceSufficiencyOutput(
        overall_evidence_state="sufficient",
        answerability="can_answer_with_caveats",
        need_more_local_rag=False,
        need_more_web_research=False,
        coverage_gaps=[],
        decision_summary="Enough evidence.",
    )
    partial_cannot_answer = EvidenceSufficiencyOutput(
        overall_evidence_state="partially_sufficient",
        answerability="cannot_answer",
        need_more_local_rag=False,
        need_more_web_research=True,
        coverage_gaps=[],
        decision_summary="Partial evidence.",
    )
    partial_ok = EvidenceSufficiencyOutput(
        overall_evidence_state="partially_sufficient",
        answerability="can_answer_with_caveats",
        need_more_local_rag=False,
        need_more_web_research=True,
        coverage_gaps=[],
        decision_summary="Partial evidence.",
    )
    insufficient_can_answer = EvidenceSufficiencyOutput(
        overall_evidence_state="insufficient",
        answerability="can_answer",
        need_more_local_rag=True,
        need_more_web_research=False,
        coverage_gaps=[],
        decision_summary="Not enough evidence.",
    )
    insufficient_no_followup = EvidenceSufficiencyOutput(
        overall_evidence_state="insufficient",
        answerability="cannot_answer",
        need_more_local_rag=False,
        need_more_web_research=False,
        coverage_gaps=[],
        decision_summary="Not enough evidence.",
    )
    empty_gap_query = EvidenceSufficiencyOutput(
        overall_evidence_state="partially_sufficient",
        answerability="can_answer_with_caveats",
        need_more_local_rag=False,
        need_more_web_research=True,
        coverage_gaps=[_coverage_gap(query="")],
        decision_summary="Partial evidence.",
    )
    empty_summary = EvidenceSufficiencyOutput(
        overall_evidence_state="partially_sufficient",
        answerability="can_answer_with_caveats",
        need_more_local_rag=False,
        need_more_web_research=True,
        coverage_gaps=[],
        decision_summary="",
    )

    assert "kept_count=0" in validate_evidence_sufficiency_output(
        sufficient_zero_kept,
        kept_count=0,
    )
    assert "sufficient evidence must have answerability=can_answer" in validate_evidence_sufficiency_output(
        sufficient_wrong_answerability,
        kept_count=1,
    )
    assert "partially_sufficient evidence cannot have answerability=cannot_answer" in validate_evidence_sufficiency_output(
        partial_cannot_answer,
        kept_count=1,
    )
    assert validate_evidence_sufficiency_output(partial_ok, kept_count=1) == ""
    assert "insufficient evidence cannot have answerability=can_answer" in validate_evidence_sufficiency_output(
        insufficient_can_answer,
        kept_count=0,
    )
    assert "must request local RAG or web research" in validate_evidence_sufficiency_output(
        insufficient_no_followup,
        kept_count=0,
    )
    assert "suggested_search_query must not be empty" in validate_evidence_sufficiency_output(
        empty_gap_query,
        kept_count=1,
    )
    assert "decision_summary must not be empty" in validate_evidence_sufficiency_output(
        empty_summary,
        kept_count=1,
    )


@pytest.mark.asyncio
async def test_no_candidates_returns_explicit_insufficient_without_fallback(monkeypatch):
    async def fail_if_called(**_kwargs):
        raise AssertionError("No-candidate Evidence Judge path should not call LLM.")

    monkeypatch.setattr("src.graph.academic.invoke_structured_llm", fail_if_called)

    parsed, debug = await _judge_evidence_candidates_with_llm(
        state=_state(),
        candidates=[],
        original_user_query="Explain Python functions",
        learning_goal="Understand functions",
        requested_resource_type="answer",
        round_index=1,
    )

    assert parsed is not None
    assert parsed.overall_evidence_state == "insufficient"
    assert parsed.need_more_web_research is True
    assert parsed.judged_evidence == []
    assert debug["used_fallback"] is False
    assert debug["fallback_chain"] == []
    assert debug["stages"][-1]["action_taken"] == "assembled_empty_evidence_judge_output"


@pytest.mark.asyncio
async def test_sufficiency_failure_returns_failed_when_fallback_disabled(monkeypatch):
    candidate = _candidate()
    grade_batch = EvidenceGradeBatch(judged_evidence=[_judge_item()])

    async def fake_invoke_structured_llm(**kwargs):
        if kwargs["node_name"] == "evidence_item_grader":
            return _structured_result(
                parsed=grade_batch,
                node_name="evidence_item_grader",
                schema_name="EvidenceGradeBatch",
            )
        raise _structured_error(
            node_name="evidence_sufficiency_judge",
            schema_name="EvidenceSufficiencyOutput",
            message="bad sufficiency json",
        )

    monkeypatch.setattr("src.graph.academic.invoke_structured_llm", fake_invoke_structured_llm)

    parsed, debug = await _judge_evidence_candidates_with_llm(
        state=_state(),
        candidates=[candidate],
        original_user_query="Explain Python functions",
        learning_goal="Understand functions",
        requested_resource_type="answer",
        round_index=1,
    )

    assert parsed is None
    assert debug["used_fallback"] is False
    assert debug["status"] != "success"
    assert debug["fallback_chain"] == []
    assert any(
        stage.get("action_taken") == "return_failed_stage_to_v2_dispatcher"
        for stage in debug["stages"]
    )


@pytest.mark.asyncio
async def test_item_grader_failure_returns_failed_without_previous_fallback(monkeypatch):
    async def fake_invoke_structured_llm(**kwargs):
        raise _structured_error(
            node_name=kwargs["node_name"],
            schema_name=kwargs["schema"].__name__,
            message="grader failed",
            business_error="missing evidence_id values: ['local:python:0']",
        )

    monkeypatch.setattr("src.graph.academic.invoke_structured_llm", fake_invoke_structured_llm)

    parsed, debug = await _judge_evidence_candidates_with_llm(
        state=_state(),
        candidates=[_candidate()],
        original_user_query="Explain Python functions",
        learning_goal="Understand functions",
        requested_resource_type="answer",
        round_index=1,
    )

    assert parsed is None
    assert debug["evidence_judge_version"] == "v2"
    assert debug["status"] == "failed"
    assert debug["used_fallback"] is False
    failed_stage = _last_failed_execution_stage(debug)
    assert failed_stage["action_taken"] in {
        "return_failed_stage_for_v2_dispatcher",
        "return_failed_stage_to_v2_dispatcher",
    }
    assert failed_stage["validation_errors"]


@pytest.mark.asyncio
async def test_evidence_judge_returns_graded_evidence_for_ce_handoff(monkeypatch):
    parsed = EvidenceJudgeOutput(
        overall_evidence_state="sufficient",
        judged_evidence=[
            _judge_item(
                evidence_score=0.86,
                score_reason="Directly supports generating the review document.",
            )
        ],
        decision_summary="Evidence is sufficient for the requested resource.",
    )

    async def fake_judge(**_kwargs):
        return parsed, {"status": "success", "stages": []}

    monkeypatch.setattr(
        "src.graph.academic._judge_evidence_candidates_with_llm",
        fake_judge,
    )

    candidate = _candidate().model_dump(mode="json")
    result = await evidence_judge(
        {
            **_state(),
            "local_evidence_candidates": [candidate],
            "local_evidence_originals": {
                "local:python:0": {
                    "content": "Python functions use parameters and return values.",
                    "source": "python.pdf",
                }
            },
            "requested_resource_type": "review_doc",
            "requested_resource_types": ["review_doc"],
            "learning_goal": "Generate a review document about Python functions.",
        }
    )

    assert result["graded_evidence"] == result["context"]
    assert len(result["graded_evidence"]) == 1
    item = result["graded_evidence"][0]
    assert item["evidence_id"] == "local:python:0"
    assert item["content"] == "Python functions use parameters and return values."
    assert item["evidence_score"] == pytest.approx(0.86)
    assert item["relevance_score"] == pytest.approx(0.86)
    assert item["score_source"] == "evidence_item_grader"
    assert item["score_scale"] == "0-1"
    assert item["score_type"] == "task_relevance"
    assert item["score_reason"] == "Directly supports generating the review document."


@pytest.mark.asyncio
async def test_v2_disabled_returns_failed_without_previous_fallback(monkeypatch):
    monkeypatch.setattr("src.graph.academic._evidence_judge_v2_enabled", lambda: False)

    parsed, debug = await _judge_evidence_candidates_with_llm(
        state=_state(),
        candidates=[_candidate()],
        original_user_query="Explain Python functions",
        learning_goal="Understand functions",
        requested_resource_type="answer",
        round_index=1,
    )

    assert parsed is None
    assert debug["evidence_judge_version"] == "v2"
    assert debug["status"] == "failed"
    assert debug["used_fallback"] is False
    assert debug["stages"][0]["error_type"] == "EvidenceJudgeV2Disabled"


@pytest.mark.asyncio
async def test_business_validator_failure_is_exposed_in_stage_debug(monkeypatch):
    raw_output = (
        '{"judged_evidence": ['
        '{"evidence_id": "local:python:0", "keep": true, "reason": "Useful but invalid."},'
        '{"evidence_id": "web:python:1", "keep": false, "reason": "Discarded."}'
        ']}'
    )

    async def fake_invoke_structured_llm(**kwargs):
        raise _structured_error(
            node_name=kwargs["node_name"],
            schema_name=kwargs["schema"].__name__,
            message="business validation failed",
            business_error="missing evidence_id values: ['local:python:0']; duplicate evidence_id values: ['x']",
            raw_output=raw_output,
            extra_debug={
                "schema_manifest": {"schema_name": "EvidenceGradeBatch", "field_paths": ["judged_evidence[].coverage_contribution"]},
                "schema_drift_report": {
                    "alias_hits_by_path": {"judged_evidence[0].coverage_reason": "coverage_contribution"},
                    "missing_required_by_path": ["judged_evidence[0].coverage_contribution"],
                },
                "drift_guard_source": "default+evidence_item_grader",
                "drift_guard_config_validated": True,
                "manifest_injected": True,
                "manifest_truncated": False,
            },
        )

    monkeypatch.setattr("src.graph.academic.invoke_structured_llm", fake_invoke_structured_llm)

    judged, debug = await _grade_evidence_items_with_llm(
        state=_state(),
        candidates=[_candidate()],
        original_user_query="Explain Python functions",
        learning_goal="Understand functions",
        requested_resource_type="answer",
        round_index=1,
    )

    assert judged is None
    failed_stage = debug["stages"][0]
    assert failed_stage["status"] == "failed"
    assert failed_stage["validation_errors"]
    assert failed_stage["judged_ids"] == ["local:python:0", "web:python:1"]
    assert failed_stage["kept_count"] == 1
    assert failed_stage["rejected_count"] == 1
    assert failed_stage["unknown_ids"] == ["web:python:1"]
    assert failed_stage["schema_manifest"]["schema_name"] == "EvidenceGradeBatch"
    assert failed_stage["schema_drift_report"]["alias_hits_by_path"] == {
        "judged_evidence[0].coverage_reason": "coverage_contribution"
    }
    assert failed_stage["drift_guard_source"] == "default+evidence_item_grader"


@pytest.mark.asyncio
async def test_grade_evidence_items_duplicate_ids_fail_before_llm(monkeypatch):
    async def fail_if_called(**_kwargs):
        raise AssertionError("evidence_item_grader should not be called with duplicate ids")

    monkeypatch.setattr("src.graph.academic.invoke_structured_llm", fail_if_called)
    web_duplicate = EvidenceCandidate(
        evidence_id="local:python:source:chunk",
        source_type="web",
        provider="unit",
        subject="python",
        role="practice",
        content_preview="Useful web evidence.",
        metadata={"fetch_status": "success", "content_chars": 120, "covered_roles": ["practice"]},
    )

    judged, debug = await _grade_evidence_items_with_llm(
        state=_state(),
        candidates=[_candidate("local:python:source:chunk"), web_duplicate],
        original_user_query="Explain Python functions",
        learning_goal="Understand functions",
        requested_resource_type="answer",
        round_index=1,
    )

    assert judged is None
    failed_stage = debug["stages"][0]
    assert failed_stage["stage"] == "evidence_item_grader.precheck"
    assert failed_stage["error_type"] == "DuplicateEvidenceIdBeforeGrading"
    assert failed_stage["duplicate_evidence_ids_before_grading"] == ["local:python:source:chunk"]


@pytest.mark.asyncio
async def test_evidence_judge_duplicate_ids_fail_before_dispatch(monkeypatch):
    async def fail_if_called(**_kwargs):
        raise AssertionError("Evidence Judge dispatcher should not receive duplicate ids")

    monkeypatch.setattr("src.graph.academic._judge_evidence_candidates_with_llm", fail_if_called)

    duplicate = _candidate("local:python:source:chunk").model_dump(mode="json")
    with pytest.raises(RuntimeError, match="duplicate evidence_id values before evidence candidate build"):
        await evidence_judge(
            {
                **_state(),
                "local_evidence_candidates": [duplicate, duplicate],
                "web_evidence_candidates": [],
                "local_evidence_originals": {},
                "web_evidence_originals": {},
                "requested_resource_type": "mindmap",
            }
        )


@pytest.mark.asyncio
async def test_evidence_judge_runtime_error_includes_failed_stage_details(monkeypatch):
    async def fake_judge(**_kwargs):
        return None, {
            "status": "failed",
            "stages": [
                {
                    "stage": "evidence_item_grader.batch",
                    "node_name": "evidence_item_grader",
                    "status": "failed",
                    "error_type": "BusinessValidationError",
                    "error_message_sanitized": "business validation failed",
                    "validation_errors": ["coverage_contribution must not be empty"],
                    "retry_count": 2,
                    "raw_preview": '{"judged_evidence": []}',
                }
            ],
        }

    monkeypatch.setattr("src.graph.academic._judge_evidence_candidates_with_llm", fake_judge)

    with pytest.raises(RuntimeError) as exc_info:
        await evidence_judge(
            {
                **_state(),
                "local_evidence_candidates": [_candidate().model_dump(mode="json")],
                "web_evidence_candidates": [],
                "local_evidence_originals": {},
                "web_evidence_originals": {},
                "requested_resource_type": "mindmap",
                "requested_resource_types": ["mindmap", "quiz"],
            }
        )

    message = str(exc_info.value)
    assert "stage=evidence_item_grader.batch" in message
    assert "node=evidence_item_grader" in message
    assert "error_type=BusinessValidationError" in message
    assert "coverage_contribution must not be empty" in message
    assert "retry_count=2" in message
    assert "raw_preview=" in message


def test_silent_fallback_guard_raises_in_strict_mode(monkeypatch):
    monkeypatch.setattr("src.graph.academic._evidence_judge_v2_strict_observability", lambda: True)

    with pytest.raises(RuntimeError, match="observability violation"):
        _assert_no_silent_fallback({
            "evidence_judge_version": "v2",
            "status": "success",
            "used_fallback": False,
            "fallback_chain": [],
            "developer_warnings": [],
            "stages": [
                {
                    "stage": "evidence_sufficiency_judge",
                    "is_fallback": True,
                }
            ],
        })


def test_silent_fallback_guard_repairs_in_place_when_not_strict(monkeypatch, caplog):
    monkeypatch.setattr("src.graph.academic._evidence_judge_v2_strict_observability", lambda: False)
    debug = {
        "evidence_judge_version": "v2",
        "status": "success",
        "used_fallback": False,
        "fallback_chain": [
            {
                "from": "evidence_sufficiency_judge",
                "to": "legacy_unexpected_fallback",
                "reason": "StructuredOutputError",
            }
        ],
        "developer_warnings": [],
        "stages": [
            {
                "stage": "evidence_sufficiency_judge",
                "is_fallback": True,
            }
        ],
    }

    with caplog.at_level(logging.ERROR, logger="src.graph.academic"):
        _assert_no_silent_fallback(debug)

    assert "observability violation" in caplog.text
    assert debug["used_fallback"] is True
    assert debug["status"] == "fallback"
