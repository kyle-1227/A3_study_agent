from __future__ import annotations

import logging
import re

import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from src.graph.academic import (
    WEB_RESEARCH_V2_SKIP_REASON,
    _assert_no_silent_web_research_fallback,
    _dedupe_web_sources_by_canonical_url,
    _web_search_dual_source,
)
from src.graph.web_research import (
    WebResearchPlan,
    WebResearchTask,
    WebSourceSummary,
    WebSourceSummaryBatch,
    validate_web_research_plan,
    validate_web_source_summary_batch,
)
from src.llm.structured_output import StructuredLLMResult, StructuredOutputError


def _state(query: str = "我还想要一份大数据的练习题") -> dict:
    return {
        "messages": [HumanMessage(content=query)],
        "learning_goal": "练习大数据基础",
        "requested_resource_type": "quiz",
        "request_id": "test-request",
        "thread_id": "test-thread",
    }


def _branches() -> list[dict]:
    return [
        {
            "subject": "big_data",
            "role": "core_concept",
            "purpose": "exercise_material",
            "rag_query": "big data practice",
            "web_search_query": "big data exercises course",
            "priority": 0.9,
        }
    ]


def _task(**overrides) -> WebResearchTask:
    payload = {
        "task_id": "task-1",
        "subject": "big_data",
        "role": "core_concept",
        "purpose": "exercise_material",
        "search_query": "big data exercises course",
        "reason": "Need exercise material.",
        "priority": 0.8,
    }
    payload.update(overrides)
    return WebResearchTask(**payload)


def _summary(**overrides) -> WebSourceSummary:
    payload = {
        "source_id": "websrc:0",
        "keep": True,
        "summary": "Covers big data exercises.",
        "coverage_points": ["HDFS and MapReduce practice"],
        "reason": "Relevant exercise source.",
        "evidence_type": "tutorial",
        "use_case": "exercise_material",
        "relevance": "high",
        "usefulness": "high",
        "risk": "low",
    }
    payload.update(overrides)
    return WebSourceSummary(**payload)


def _structured_result(parsed, *, node_name: str, schema_name: str) -> StructuredLLMResult:
    return StructuredLLMResult(
        success=True,
        parsed=parsed,
        node_name=node_name,
        llm_node=node_name,
        schema_name=schema_name,
        provider="test",
        model="test",
        output_mode="deepseek_tool_call_strict",
        fallback_modes=[],
        raw_output=parsed.model_dump_json() if parsed is not None else "{}",
    )


def _structured_error(*, node_name: str, schema_name: str, message: str = "structured failure") -> StructuredOutputError:
    return StructuredOutputError(
        StructuredLLMResult(
            success=False,
            parsed=None,
            node_name=node_name,
            llm_node=node_name,
            schema_name=schema_name,
            provider="test",
            model="test",
            output_mode="deepseek_tool_call_strict",
            fallback_modes=[],
            raw_output="{bad",
            failure_phase="validation_error",
            error_type="ValidationError",
            error_message=message,
            validation_error=message,
        )
    )


def _source_ids_from_messages(messages: list[dict]) -> list[str]:
    content = str(messages[-1]["content"])
    return re.findall(r'"source_id":\s*"([^"]+)"', content)


def test_web_research_schema_limits_and_literal_rejection():
    tasks = [_task(task_id=f"task-{index}", search_query=f"query {index}") for index in range(7)]
    with pytest.raises(ValidationError):
        WebResearchPlan(tasks=tasks)

    summaries = [_summary(source_id=f"websrc:{index}") for index in range(13)]
    with pytest.raises(ValidationError):
        WebSourceSummaryBatch(summaries=summaries)

    with pytest.raises(ValidationError):
        WebSourceSummary(**{**_summary().model_dump(), "evidence_type": "made_up"})

    with pytest.raises(ValidationError):
        WebSourceSummary(**{**_summary().model_dump(), "url": "https://example.edu"})


def test_planner_validator_rejects_invalid_task_shape():
    plan = WebResearchPlan(tasks=[
        _task(task_id="", search_query="", role="", purpose="", reason="", priority=1.4),
        _task(task_id="task-2", subject="python", search_query="same query"),
        _task(task_id="task-3", search_query="same query"),
    ])

    error = validate_web_research_plan(
        plan,
        allowed_subjects=["big_data"],
        max_total_tasks=1,
        max_tasks_per_subject=1,
    )

    assert "task count 3 exceeds max_total_tasks 1" in error
    assert "task_id must not be empty" in error
    assert "search_query must not be empty" in error
    assert "subject 'python' is not in allowed subjects" in error
    assert "role must not be empty" in error
    assert "purpose must not be empty" in error
    assert "reason must not be empty" in error
    assert "priority must be between 0 and 1" in error
    assert "duplicate search_query" in error


def test_planner_validator_rejects_duplicate_ids_queries_and_subject_caps():
    plan = WebResearchPlan(tasks=[
        _task(task_id="dup", search_query="same query"),
        _task(task_id="dup", search_query="same query", priority=0.2),
    ])

    error = validate_web_research_plan(
        plan,
        allowed_subjects=["big_data"],
        max_total_tasks=6,
        max_tasks_per_subject=1,
    )

    assert "duplicate task_id values" in error
    assert "duplicate search_query values" in error
    assert "subject 'big_data' task count 2 exceeds max_tasks_per_subject 1" in error


def test_summarizer_validator_requires_each_source_exactly_once():
    batch = WebSourceSummaryBatch(summaries=[
        _summary(source_id="websrc:0"),
        _summary(source_id="websrc:0"),
        _summary(source_id="websrc:9"),
    ])

    error = validate_web_source_summary_batch(batch, expected_source_ids=["websrc:0", "websrc:1"])

    assert "missing source_id values: ['websrc:1']" in error
    assert "duplicate source_id values: ['websrc:0']" in error
    assert "unknown source_id values: ['websrc:9']" in error
    assert "expected 2 source summaries, got 3" in error


def test_summarizer_validator_rejects_empty_fields():
    batch = WebSourceSummaryBatch(summaries=[
        _summary(source_id="", reason="", summary="", coverage_points=[]),
    ])

    error = validate_web_source_summary_batch(batch, expected_source_ids=["websrc:0"])

    assert "source_id must not be empty" in error
    assert "missing source_id values: ['websrc:0']" in error
    assert "reason must not be empty" in error
    assert "summary must not be empty when keep=true" in error
    assert "coverage_points must contain at least one item when keep=true" in error


def test_canonical_url_dedup_keeps_higher_score_then_priority():
    sources = [
        {
            "task_id": "task-low",
            "task_priority": 0.9,
            "original_url": "https://www.example.com/course?utm_source=test",
            "tavily_score": 0.2,
        },
        {
            "task_id": "task:ml:0",
            "task_priority": 0.1,
            "original_url": "https://example.com/course",
            "tavily_score": 0.8,
        },
        {
            "task_id": "task-priority-low",
            "task_priority": 0.2,
            "original_url": "https://example.com/guide",
            "tavily_score": 0.5,
        },
        {
            "task_id": "task-priority-high",
            "task_priority": 0.9,
            "original_url": "https://www.example.com/guide/",
            "tavily_score": 0.5,
        },
    ]

    deduped, debug = _dedupe_web_sources_by_canonical_url(sources)

    assert debug["duplicate_url_count"] == 2
    assert {source["task_id"] for source in deduped} == {"task:ml:0", "task-priority-high"}
    assert deduped[0]["source_id"] == "websrc:task_ml_0:0"
    assert all(str(source["source_id"]).startswith("websrc:") for source in deduped)


@pytest.mark.asyncio
async def test_planner_failure_uses_legacy_fallback(monkeypatch):
    async def fake_invoke_structured_llm(**kwargs):
        raise _structured_error(
            node_name=kwargs["node_name"],
            schema_name=kwargs["schema"].__name__,
            message="planner failed",
        )

    async def fake_legacy(state, branches, branch_debug):
        return {
            "web_evidence_candidates": [{"evidence_id": "web:legacy:0"}],
            "web_evidence_originals": {"web:legacy:0": {"title": "legacy"}},
        }

    monkeypatch.setattr("src.graph.academic.invoke_structured_llm", fake_invoke_structured_llm)
    monkeypatch.setattr("src.graph.academic._web_search_dual_source_legacy", fake_legacy)

    result = await _web_search_dual_source(_state(), _branches(), {"mode": "dual_source_evidence"})
    debug = result["web_research_v2_debug"]

    assert result["web_evidence_candidates"] == [{"evidence_id": "web:legacy:0"}]
    assert debug["status"] == "fallback"
    assert debug["used_fallback"] is True
    assert any(item["to"] == "legacy_dual_source_web_search" for item in debug["fallback_chain"])


@pytest.mark.asyncio
async def test_summarizer_failure_uses_basic_doc_fallback(monkeypatch):
    plan = WebResearchPlan(tasks=[_task()])

    async def fake_invoke_structured_llm(**kwargs):
        if kwargs["node_name"] == "web_research_planner":
            return _structured_result(parsed=plan, node_name="web_research_planner", schema_name="WebResearchPlan")
        raise _structured_error(
            node_name="web_source_summarizer",
            schema_name="WebSourceSummaryBatch",
            message="summarizer failed",
        )

    def fake_web_search_fn(*args, **kwargs):
        return {
            "ok": True,
            "results": [
                {
                    "title": "Big Data Exercise Set",
                    "url": "https://example.edu/big-data/exercises",
                    "content": "Practice HDFS, MapReduce, and Spark basics.",
                    "score": 0.87,
                }
            ],
            "result_count": 1,
        }

    monkeypatch.setattr("src.graph.academic.invoke_structured_llm", fake_invoke_structured_llm)
    monkeypatch.setattr("src.graph.academic.web_search_fn", fake_web_search_fn)

    result = await _web_search_dual_source(_state(), _branches(), {"mode": "dual_source_evidence"})
    candidate = result["web_evidence_candidates"][0]
    original = next(iter(result["web_evidence_originals"].values()))
    debug = result["web_research_v2_debug"]

    assert debug["used_fallback"] is True
    assert candidate["metadata"]["source_id"].startswith("websrc:task-1:")
    assert candidate["metadata"]["task_id"] == "task-1"
    assert candidate["metadata"]["canonical_url"] == "https://example.edu/big-data/exercises"
    assert candidate["metadata"]["url"] == "https://example.edu/big-data/exercises"
    assert candidate["metadata"]["summary_source"] == "basic_tavily_fallback"
    assert candidate["metadata"]["source_summary_fallback_used"] is True
    assert candidate["metadata"]["web_research_v2_stage"] == "summarizer_fallback"
    assert original["summary_source"] == "basic_tavily_fallback"
    assert original["source_summary_fallback_used"] is True


@pytest.mark.asyncio
async def test_web_research_v2_records_search_result_judge_skipped(monkeypatch):
    plan = WebResearchPlan(tasks=[_task()])

    async def fake_invoke_structured_llm(**kwargs):
        if kwargs["node_name"] == "web_research_planner":
            return _structured_result(parsed=plan, node_name="web_research_planner", schema_name="WebResearchPlan")
        source_ids = _source_ids_from_messages(kwargs["messages"])
        batch = WebSourceSummaryBatch(summaries=[_summary(source_id=source_id) for source_id in source_ids])
        return _structured_result(
            parsed=batch,
            node_name="web_source_summarizer",
            schema_name="WebSourceSummaryBatch",
        )

    def fake_web_search_fn(*args, **kwargs):
        return [
            {
                "title": "Big Data Exercises",
                "url": "https://example.edu/big-data",
                "content": "Hands-on exercises for distributed systems.",
                "score": 0.91,
            }
        ]

    monkeypatch.setattr("src.graph.academic.invoke_structured_llm", fake_invoke_structured_llm)
    monkeypatch.setattr("src.graph.academic.web_search_fn", fake_web_search_fn)

    result = await _web_search_dual_source(_state(), _branches(), {"mode": "dual_source_evidence"})
    debug = result["web_research_v2_debug"]

    assert debug["search_result_judge_skipped"] is True
    assert debug["skip_reason"] == WEB_RESEARCH_V2_SKIP_REASON
    assert all(stage["skip_reason"] == WEB_RESEARCH_V2_SKIP_REASON for stage in debug["stages"] if "skip_reason" in stage)


@pytest.mark.asyncio
async def test_summarizer_rejects_all_sources_returns_empty_degraded(monkeypatch):
    plan = WebResearchPlan(tasks=[_task(task_id="task:ml:0")])
    warning = "All web sources were rejected by Web Source Summarizer; continuing with local evidence only."

    async def fake_invoke_structured_llm(**kwargs):
        if kwargs["node_name"] == "web_research_planner":
            return _structured_result(parsed=plan, node_name="web_research_planner", schema_name="WebResearchPlan")
        source_ids = _source_ids_from_messages(kwargs["messages"])
        batch = WebSourceSummaryBatch(summaries=[
            _summary(
                source_id=source_id,
                keep=False,
                summary="",
                coverage_points=[],
                reason="Not useful enough for this request.",
                use_case="discard",
                relevance="low",
                usefulness="low",
                risk="medium",
            )
            for source_id in source_ids
        ])
        return _structured_result(
            parsed=batch,
            node_name="web_source_summarizer",
            schema_name="WebSourceSummaryBatch",
        )

    def fake_web_search_fn(*args, **kwargs):
        return [
            {
                "title": "Rejected Big Data Page",
                "url": "https://example.edu/rejected-1",
                "content": "A generic page.",
                "score": 0.71,
            },
            {
                "title": "Rejected Big Data Blog",
                "url": "https://example.edu/rejected-2",
                "content": "Another generic page.",
                "score": 0.69,
            },
        ]

    monkeypatch.setattr("src.graph.academic.invoke_structured_llm", fake_invoke_structured_llm)
    monkeypatch.setattr("src.graph.academic.web_search_fn", fake_web_search_fn)

    result = await _web_search_dual_source(_state(), _branches(), {"mode": "dual_source_evidence"})
    debug = result["web_research_v2_debug"]
    final_stage = next(stage for stage in debug["stages"] if stage["stage"] == "web_research_v2.final")

    assert result["web_evidence_candidates"] == []
    assert result["web_evidence_originals"] == {}
    assert debug["status"] == "degraded"
    assert warning in debug["developer_warnings"]
    assert final_stage["summarizer_result_count"] == 2
    assert final_stage["kept_count"] == 0
    assert final_stage["rejected_count"] == 2
    assert final_stage["developer_warning"] == warning


@pytest.mark.asyncio
async def test_single_tavily_timeout_continues(monkeypatch):
    plan = WebResearchPlan(tasks=[
        _task(task_id="task-timeout", search_query="timeout query"),
        _task(task_id="task-ok", search_query="ok query"),
    ])

    async def fake_invoke_structured_llm(**kwargs):
        if kwargs["node_name"] == "web_research_planner":
            return _structured_result(parsed=plan, node_name="web_research_planner", schema_name="WebResearchPlan")
        source_ids = _source_ids_from_messages(kwargs["messages"])
        batch = WebSourceSummaryBatch(summaries=[_summary(source_id=source_id) for source_id in source_ids])
        return _structured_result(
            parsed=batch,
            node_name="web_source_summarizer",
            schema_name="WebSourceSummaryBatch",
        )

    def fake_web_search_fn(query, *args, **kwargs):
        if query == "timeout query":
            raise TimeoutError("boom")
        return [
            {
                "title": "Big Data Practice",
                "url": "https://example.edu/practice",
                "content": "Practice tasks.",
                "score": 0.7,
            }
        ]

    monkeypatch.setattr("src.graph.academic.invoke_structured_llm", fake_invoke_structured_llm)
    monkeypatch.setattr("src.graph.academic.web_search_fn", fake_web_search_fn)

    result = await _web_search_dual_source(_state(), _branches(), {"mode": "dual_source_evidence"})
    debug = result["web_research_v2_debug"]

    assert len(result["web_evidence_candidates"]) == 1
    assert any(stage["stage"] == "web_search_executor.task" and stage["status"] == "failed" for stage in debug["stages"])
    assert debug["status"] == "success"


@pytest.mark.asyncio
async def test_all_tasks_failed_returns_empty_degraded(monkeypatch):
    plan = WebResearchPlan(tasks=[_task()])

    async def fake_invoke_structured_llm(**kwargs):
        return _structured_result(parsed=plan, node_name="web_research_planner", schema_name="WebResearchPlan")

    def fake_web_search_fn(*args, **kwargs):
        raise TimeoutError("tavily down")

    monkeypatch.setattr("src.graph.academic.invoke_structured_llm", fake_invoke_structured_llm)
    monkeypatch.setattr("src.graph.academic.web_search_fn", fake_web_search_fn)

    result = await _web_search_dual_source(_state(), _branches(), {"mode": "dual_source_evidence"})
    debug = result["web_research_v2_debug"]

    assert result["web_evidence_candidates"] == []
    assert result["web_evidence_originals"] == {}
    assert debug["status"] == "degraded"
    assert "All web research tasks failed; continuing with local evidence only." in debug["developer_warnings"]


def test_web_research_silent_fallback_guard_raises_in_strict_mode(monkeypatch):
    monkeypatch.setattr("src.graph.academic._web_research_v2_strict_observability", lambda: True)

    with pytest.raises(RuntimeError, match="observability violation"):
        _assert_no_silent_web_research_fallback({
            "status": "success",
            "used_fallback": False,
            "fallback_chain": [],
            "developer_warnings": [],
            "stages": [{"stage": "web_source_summarizer.batch", "is_fallback": True}],
        })


def test_web_research_silent_fallback_guard_repairs_when_not_strict(monkeypatch, caplog):
    monkeypatch.setattr("src.graph.academic._web_research_v2_strict_observability", lambda: False)
    debug = {
        "status": "success",
        "used_fallback": False,
        "fallback_chain": [
            {
                "from": "web_source_summarizer",
                "to": "basic_tavily_fallback",
                "reason": "StructuredOutputError",
            }
        ],
        "developer_warnings": [],
        "stages": [{"stage": "web_source_summarizer.batch", "is_fallback": True}],
    }

    with caplog.at_level(logging.ERROR, logger="src.graph.academic"):
        _assert_no_silent_web_research_fallback(debug)

    assert "observability violation" in caplog.text
    assert debug["used_fallback"] is True
    assert debug["status"] == "fallback"
