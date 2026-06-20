from __future__ import annotations

import re

import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from src.config import load_prompt
from src.graph.academic import (
    WEB_RESEARCH_V2_EVIDENCE_BOUNDARY_REASON,
    WEB_RESEARCH_V2_LLM_SUMMARY_SOURCE,
    WEB_RESEARCH_V2_PLANNER_NODE,
    WEB_RESEARCH_V2_STAGE_CANDIDATE_BUILD,
    WEB_RESEARCH_V2_STAGE_COMPLETE,
    WEB_RESEARCH_V2_STAGE_CURATE,
    WEB_RESEARCH_V2_STAGE_DEDUPE,
    WEB_RESEARCH_V2_STAGE_FAILED,
    WEB_RESEARCH_V2_STAGE_FETCH_SOURCE,
    WEB_RESEARCH_V2_STAGE_FETCH_START,
    WEB_RESEARCH_V2_STAGE_PLAN_FAILED,
    WEB_RESEARCH_V2_STAGE_PLAN_START,
    WEB_RESEARCH_V2_STAGE_PLAN_SUCCESS,
    WEB_RESEARCH_V2_STAGE_SEARCH_START,
    WEB_RESEARCH_V2_STAGE_SEARCH_TASK,
    WEB_RESEARCH_V2_STAGE_START,
    WEB_RESEARCH_V2_STAGE_SUMMARIZE_FAILED,
    WEB_RESEARCH_V2_STAGE_SUMMARIZE_START,
    WEB_RESEARCH_V2_STAGE_SUMMARIZE_SUCCESS,
    WEB_RESEARCH_V2_SUMMARIZER_NODE,
    WEB_RESEARCH_V2_WARNING_ALL_SOURCES_REJECTED,
    WEB_RESEARCH_V2_WARNING_DISABLED,
    WEB_RESEARCH_V2_WARNING_PLANNER_FAILED_FAIL_FAST,
    WEB_RESEARCH_V2_WARNING_SUMMARIZER_FAILED,
    _build_web_research_planner_messages,
    _build_web_source_summarizer_messages,
    _dedupe_web_sources_by_canonical_url,
    _drop_deprecated_web_state_keys,
    _trace_parse_web_source_summary_raw,
    _web_research_branch_payload,
    _web_search_dual_source,
)
from src.graph.web_research import (
    WebFetchedSource,
    WebRawSource,
    WebResearchPlan,
    WebResearchTask,
    WebSourceSummary,
    WebSourceSummaryBatch,
    build_web_source_summarizer_input_dto,
    normalize_web_raw_source,
    validate_web_research_plan,
    validate_web_source_summary_batch,
)
from src.llm.structured_output import StructuredLLMResult, StructuredOutputError
from src.observability.a3_trace import reset_trace_event_sink, set_trace_event_sink


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
            "local_retrieval_query": "big data practice",
            "web_research_seed_query": "big data exercises course",
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
        "source_id": "websrc:task-1:abc123abc123",
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


def _search_result(*, content: str = "Practice HDFS, MapReduce, and Spark basics.") -> list[dict]:
    return [
        {
            "title": "Big Data Exercise Set",
            "url": "https://example.edu/big-data/exercises?utm_source=test",
            "content": content,
            "score": 0.87,
        }
    ]


async def _successful_llm(**kwargs):
    if kwargs["node_name"] == WEB_RESEARCH_V2_PLANNER_NODE:
        return _structured_result(
            parsed=WebResearchPlan(tasks=[_task()]),
            node_name=WEB_RESEARCH_V2_PLANNER_NODE,
            schema_name="WebResearchPlan",
        )
    source_ids = _source_ids_from_messages(kwargs["messages"])
    batch = WebSourceSummaryBatch(summaries=[_summary(source_id=source_id) for source_id in source_ids])
    return _structured_result(
        parsed=batch,
        node_name=WEB_RESEARCH_V2_SUMMARIZER_NODE,
        schema_name="WebSourceSummaryBatch",
    )


def test_web_research_schema_limits_and_literal_rejection():
    tasks = [_task(task_id=f"task-{index}", search_query=f"query {index}") for index in range(7)]
    with pytest.raises(ValidationError):
        WebResearchPlan(tasks=tasks)

    summaries = [_summary(source_id=f"websrc:task-{index}:abc") for index in range(13)]
    with pytest.raises(ValidationError):
        WebSourceSummaryBatch(summaries=summaries)

    for bad_value in ["practice_material", "mindmap", "practice_questions", "web_article"]:
        with pytest.raises(ValidationError):
            WebSourceSummary(**{**_summary().model_dump(), "use_case": bad_value})

    with pytest.raises(ValidationError):
        WebSourceSummary(**{**_summary().model_dump(), "url": "https://example.edu"})


def test_web_source_summary_requires_reason_and_rejects_task_metadata():
    keep_true_missing_reason = _summary().model_dump()
    keep_true_missing_reason.pop("reason")
    with pytest.raises(ValidationError):
        WebSourceSummary(**keep_true_missing_reason)

    keep_false_missing_reason = _summary(
        keep=False,
        summary="",
        coverage_points=[],
        use_case="discard",
        relevance="low",
        usefulness="low",
        risk="medium",
    ).model_dump()
    keep_false_missing_reason.pop("reason")
    with pytest.raises(ValidationError):
        WebSourceSummary(**keep_false_missing_reason)

    assert _summary(reason="Useful for the requested learning goal.").reason

    for forbidden in ["task_id", "subject", "role", "purpose"]:
        with pytest.raises(ValidationError):
            WebSourceSummary(**{**_summary().model_dump(), forbidden: "input-only metadata"})


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


def test_raw_source_normalization_and_stable_dedupe():
    raw = normalize_web_raw_source(
        _search_result()[0],
        task_id="task:ml:0",
        subject="big_data",
        role="core_concept",
        purpose="exercise_material",
        search_query="big data exercises",
        task_priority=0.8,
        provider="tavily",
        provider_rank=0,
        retrieved_at="2026-06-17T00:00:00+00:00",
    )

    assert isinstance(raw, WebRawSource)
    assert raw.canonical_url == "https://example.edu/big-data/exercises"
    assert raw.provider_score == 0.87

    deduped, debug = _dedupe_web_sources_by_canonical_url([raw.model_dump(mode="json")])

    assert debug["duplicate_url_count"] == 0
    assert re.fullmatch(r"websrc:task_ml_0:[0-9a-f]{12}", deduped[0]["source_id"])


def test_fetch_source_schema_preserves_fetch_status():
    fetched = WebFetchedSource.model_validate({
        **normalize_web_raw_source(
            _search_result()[0],
            task_id="task-1",
            subject="big_data",
            role="core_concept",
            purpose="exercise_material",
            search_query="big data exercises",
            task_priority=0.8,
            provider="tavily",
            provider_rank=0,
        ).model_dump(),
        "fetch_status": "success",
        "content_chars": 42,
        "content_preview": "Readable provider content",
    })

    assert fetched.fetch_status == "success"
    assert fetched.content_chars == 42


def test_deprecated_checkpoint_state_keys_are_dropped_before_v2_boundary():
    old_web_query = "web_" + "search_" + "query"
    old_debug = "web_research_" + "v2_debug"
    old_prefix_key = "web_" + "supple" + "ment_results"
    state = {
        old_web_query: "old query",
        old_debug: {"status": "old"},
        old_prefix_key: [{"title": "old"}],
        "web_research_seed_query": "new seed",
    }

    sanitized, dropped = _drop_deprecated_web_state_keys(state)

    assert old_web_query in dropped
    assert old_debug in dropped
    assert old_prefix_key in dropped
    assert old_web_query not in sanitized
    assert old_debug not in sanitized
    assert old_prefix_key not in sanitized
    assert sanitized["web_research_seed_query"] == "new seed"


def test_web_research_prompts_define_v2_boundaries_and_enum_mapping():
    planner_prompt = load_prompt("web_research_planner")
    summarizer_prompt = load_prompt("web_source_summarizer")

    assert "Forbidden output fields: rag_query, web_search_query" in planner_prompt
    assert "never output seed_search_query" in planner_prompt
    assert "Return only fields defined by WebResearchTask" in planner_prompt
    assert (
        "Per subject, output at most {max_tasks_per_subject} tasks total"
        in planner_prompt
    )
    assert "core_concept plus practice or exercise" in planner_prompt
    assert (
        "Do not add extra deep_dive, application, or application_context tasks"
        in planner_prompt
    )
    assert "Web Source Summarizer prepares structured source summaries" in summarizer_prompt
    assert "Evidence Judge V2 makes final sufficiency decisions" in summarizer_prompt
    assert "use_case=exercise_material" in summarizer_prompt
    assert "use_case=roadmap_reference" in summarizer_prompt
    assert "practice_material" in summarizer_prompt
    assert "reason is required" in summarizer_prompt
    assert "Forbidden fields: task_id, subject, role, purpose" in summarizer_prompt
    assert "Do not copy task metadata fields into summary items" in summarizer_prompt
    assert "WebSourceSummarizerInputDTO" in summarizer_prompt
    assert "source_id, source_text, source_context, and provider_score" in summarizer_prompt


def _keys_recursive(value) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for item in value.values():
            keys.update(_keys_recursive(item))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys.update(_keys_recursive(item))
        return keys
    return set()


def test_web_research_task_reason_description_is_task_semantic():
    description = WebResearchTask.model_fields["reason"].description or ""

    assert "search task" in description
    assert "expected to retrieve" in description
    assert "keep=true" not in description
    assert "rejected" not in description
    assert "summary item" not in description


def test_web_source_summarizer_input_dto_excludes_metadata_keys():
    source = {
        "source_id": "websrc:task-1:abc",
        "task_id": "task-1",
        "subject": "python",
        "role": "core_concept",
        "purpose": "Find Python practice material.",
        "search_query": "python practice questions",
        "url": "https://example.com/python",
        "domain": "example.com",
        "title": "Python Practice",
        "canonical_url": "https://example.com/python",
        "original_url": "https://example.com/python?utm=1",
        "metadata": {"debug": True},
        "provider_score": 0.88,
        "snippet": "Short practice snippet.",
        "content_preview": "Readable source content.",
    }

    dto = build_web_source_summarizer_input_dto(
        query="Give me Python exercises",
        learning_goal="Practice Python basics",
        requested_resource_type="quiz",
        requested_resource_types=["quiz", "exercise"],
        output_language="same_as_user_query",
        sources=[source],
    )
    dumped = dto.model_dump(mode="json")
    keys = _keys_recursive(dumped)

    forbidden_keys = {
        "task_id",
        "subject",
        "role",
        "purpose",
        "search_query",
        "url",
        "domain",
        "title",
        "canonical_url",
        "original_url",
        "metadata",
        "debug",
    }
    assert forbidden_keys.isdisjoint(keys)
    assert dumped["sources"][0]["source_id"] == "websrc:task-1:abc"
    assert "Readable source content" in dumped["sources"][0]["source_text"]
    assert "Python Practice" in dumped["sources"][0]["source_text"]
    assert "Practice Python basics" in dumped["sources"][0]["source_context"]
    assert dumped["sources"][0]["provider_score"] == 0.88


def test_web_source_summarizer_messages_use_dto_projection():
    source = WebFetchedSource(
        **WebRawSource(
            task_id="task-1",
            source_id="websrc:task-1:abc",
            original_url="https://example.com/python",
            canonical_url="https://example.com/python",
            title="Python Practice",
            domain="example.com",
            snippet="Short practice snippet.",
            provider="tavily",
            provider_score=0.9,
            subject="python",
            role="core_concept",
            purpose="Find Python practice material.",
            search_query="python practice questions",
        ).model_dump(),
        fetch_status="success",
        content_chars=24,
        content_preview="Readable source content.",
    ).model_dump(mode="json")
    messages = _build_web_source_summarizer_messages(
        state={**_state("Give me Python exercises"), "requested_resource_types": ["quiz", "exercise"]},
        sources=[source],
        original_user_query="Give me Python exercises",
    )
    content = messages[-1]["content"]

    assert '"sources":' in content
    assert '"source_text"' in content
    assert '"source_context"' in content
    assert '"task_id"' not in content
    assert '"subject"' not in content
    assert '"role"' not in content
    assert '"purpose"' not in content
    assert '"search_query"' not in content
    assert '"url"' not in content
    assert '"domain"' not in content
    assert '"title"' not in content


def test_web_source_summary_raw_trace_parser_counts_missing_reason_and_extra_fields():
    missing_reason = _trace_parse_web_source_summary_raw(
        '{"summaries": ['
        '{"source_id": "websrc:0", "keep": true},'
        '{"source_id": "websrc:1", "keep": false}'
        ']}'
    )
    assert missing_reason["returned_source_ids"] == ["websrc:0", "websrc:1"]
    assert missing_reason["kept_count"] == 1
    assert missing_reason["rejected_count"] == 1
    assert missing_reason["missing_required_reason_count"] == 2
    assert missing_reason["extra_field_count"] == 0

    extra_fields = _trace_parse_web_source_summary_raw(
        '{"summaries": ['
        '{"source_id": "websrc:0", "keep": true, "reason": "useful", '
        '"task_id": "task-1", "subject": "python", "role": "core", "purpose": "quiz"}'
        ']}'
    )
    assert extra_fields["missing_required_reason_count"] == 0
    assert extra_fields["extra_field_names"] == ["purpose", "role", "subject", "task_id"]
    assert extra_fields["extra_field_count"] == 4


def test_web_research_planner_input_uses_v2_friendly_branch_payload():
    branch_payload = _web_research_branch_payload(_branches(), original_user_query="current query")
    messages = _build_web_research_planner_messages(
        state=_state(),
        branches=_branches(),
        original_user_query="current query",
    )
    branch_item = branch_payload[0]

    assert messages[-1]["content"]
    assert "Per subject, output at most 2 tasks total" in messages[-1]["content"]
    assert "core_concept plus practice or exercise" in messages[-1]["content"]
    assert branch_item["seed_search_query"] == "big data exercises course"
    assert "local_branch_status" in branch_item
    assert "weak_reason" in branch_item
    assert "local_retrieval_query" not in branch_item
    assert "web_research_seed_query" not in branch_item
    assert "retrieval_coverage_goals" not in branch_item


def test_web_research_task_rejects_deprecated_output_fields():
    with pytest.raises(ValidationError):
        WebResearchPlan.model_validate({
            "tasks": [
                {
                    "task_id": "task-deprecated",
                    "subject": "big_data",
                    "role": "core_concept",
                    "purpose": "exercise_material",
                    "search_query": "big data exercises",
                    "web_research_seed_query": "deprecated field must fail",
                    "local_retrieval_query": "deprecated field must fail",
                    "retrieval_coverage_goals": ["deprecated field must fail"],
                    "reason": "Deprecated fields should be exposed as validation errors.",
                    "priority": 0.8,
                }
            ]
        })


@pytest.mark.asyncio
async def test_web_research_pipeline_success_stage_order_and_stable_candidate(monkeypatch):
    events: list[dict] = []

    monkeypatch.setattr("src.graph.academic.invoke_structured_llm", _successful_llm)
    monkeypatch.setattr("src.graph.academic.web_search_fn", lambda *args, **kwargs: _search_result())

    token = set_trace_event_sink(events)
    try:
        result = await _web_search_dual_source(_state(), _branches(), {"mode": "dual_source_evidence"})
    finally:
        reset_trace_event_sink(token)

    debug = result["web_research_debug"]
    stages = [stage["stage"] for stage in debug["stages"]]
    candidate = result["web_evidence_candidates"][0]

    assert result["web_research_outcome"] == "success"
    assert debug["status"] == "success"
    assert debug["used_fallback"] is False
    assert debug["fallback_chain"] == []
    assert stages == [
        WEB_RESEARCH_V2_STAGE_START,
        WEB_RESEARCH_V2_STAGE_PLAN_START,
        WEB_RESEARCH_V2_STAGE_PLAN_SUCCESS,
        WEB_RESEARCH_V2_STAGE_SEARCH_START,
        WEB_RESEARCH_V2_STAGE_SEARCH_TASK,
        WEB_RESEARCH_V2_STAGE_DEDUPE,
        WEB_RESEARCH_V2_STAGE_FETCH_START,
        WEB_RESEARCH_V2_STAGE_FETCH_SOURCE,
        WEB_RESEARCH_V2_STAGE_CURATE,
        WEB_RESEARCH_V2_STAGE_SUMMARIZE_START,
        WEB_RESEARCH_V2_STAGE_SUMMARIZE_SUCCESS,
        WEB_RESEARCH_V2_STAGE_CANDIDATE_BUILD,
        WEB_RESEARCH_V2_STAGE_COMPLETE,
    ]
    assert candidate["evidence_id"].startswith("web:websrc_task-1_")
    assert candidate["metadata"]["source_id"].startswith("websrc:task-1:")
    assert candidate["metadata"]["fetch_status"] == "success"
    assert candidate["metadata"]["summary_source"] == WEB_RESEARCH_V2_LLM_SUMMARY_SOURCE
    assert debug["evidence_boundary_reason"] == WEB_RESEARCH_V2_EVIDENCE_BOUNDARY_REASON
    assert any(event["stage"] == WEB_RESEARCH_V2_STAGE_COMPLETE for event in events)
    assert all("search_query" not in event for event in events if event["stage"] == WEB_RESEARCH_V2_STAGE_SEARCH_TASK)
    assert all("canonical_url" not in event for event in events if event["stage"] == WEB_RESEARCH_V2_STAGE_FETCH_SOURCE)


@pytest.mark.asyncio
async def test_planner_failure_fail_fast_raises_without_previous_fallback(monkeypatch):
    async def fake_invoke_structured_llm(**kwargs):
        raise _structured_error(
            node_name=kwargs["node_name"],
            schema_name=kwargs["schema"].__name__,
            message="planner failed",
        )

    monkeypatch.setattr("src.graph.academic.invoke_structured_llm", fake_invoke_structured_llm)

    with pytest.raises(RuntimeError, match=WEB_RESEARCH_V2_WARNING_PLANNER_FAILED_FAIL_FAST) as exc_info:
        await _web_search_dual_source(_state(), _branches(), {"mode": "dual_source_evidence"})

    debug = getattr(exc_info.value, "web_research_debug")

    assert debug["status"] == "failed"
    assert debug["web_research_outcome"] == "failed"
    assert debug["used_fallback"] is False
    assert WEB_RESEARCH_V2_WARNING_PLANNER_FAILED_FAIL_FAST in debug["developer_warnings"]
    assert any(stage["stage"] == WEB_RESEARCH_V2_STAGE_PLAN_FAILED for stage in debug["stages"])
    assert debug["stages"][-1]["stage"] == WEB_RESEARCH_V2_STAGE_FAILED


@pytest.mark.asyncio
async def test_web_research_v2_disabled_returns_skipped_without_previous_pipeline(monkeypatch):
    async def fake_invoke_structured_llm(**kwargs):
        raise AssertionError("LLM must not be called when Web Research V2 is disabled")

    monkeypatch.setattr("src.graph.academic.invoke_structured_llm", fake_invoke_structured_llm)
    monkeypatch.setattr("src.graph.academic._web_research_v2_enabled", lambda: False)

    result = await _web_search_dual_source(_state(), _branches(), {"mode": "dual_source_evidence"})
    debug = result["web_research_debug"]

    assert result["web_evidence_candidates"] == []
    assert result["web_evidence_originals"] == {}
    assert result["web_research_outcome"] == "skipped"
    assert debug["status"] == "skipped"
    assert debug["used_fallback"] is False
    assert WEB_RESEARCH_V2_WARNING_DISABLED in debug["developer_warnings"]


@pytest.mark.asyncio
async def test_summarizer_failure_raises_without_basic_fallback(monkeypatch):
    async def fake_invoke_structured_llm(**kwargs):
        if kwargs["node_name"] == WEB_RESEARCH_V2_PLANNER_NODE:
            return _structured_result(
                parsed=WebResearchPlan(tasks=[_task()]),
                node_name=WEB_RESEARCH_V2_PLANNER_NODE,
                schema_name="WebResearchPlan",
            )
        raise _structured_error(
            node_name=WEB_RESEARCH_V2_SUMMARIZER_NODE,
            schema_name="WebSourceSummaryBatch",
            message="summarizer failed",
        )

    monkeypatch.setattr("src.graph.academic.invoke_structured_llm", fake_invoke_structured_llm)
    monkeypatch.setattr("src.graph.academic.web_search_fn", lambda *args, **kwargs: _search_result())

    with pytest.raises(RuntimeError, match=WEB_RESEARCH_V2_WARNING_SUMMARIZER_FAILED) as exc_info:
        await _web_search_dual_source(_state(), _branches(), {"mode": "dual_source_evidence"})

    debug = getattr(exc_info.value, "web_research_debug")

    assert debug["status"] == "failed"
    assert debug["used_fallback"] is False
    assert debug["fallback_chain"] == []
    assert any(stage["stage"] == WEB_RESEARCH_V2_STAGE_SUMMARIZE_FAILED for stage in debug["stages"])


@pytest.mark.asyncio
async def test_summarizer_rejects_all_sources_raises(monkeypatch):
    async def fake_invoke_structured_llm(**kwargs):
        if kwargs["node_name"] == WEB_RESEARCH_V2_PLANNER_NODE:
            return _structured_result(
                parsed=WebResearchPlan(tasks=[_task(task_id="task:ml:0")]),
                node_name=WEB_RESEARCH_V2_PLANNER_NODE,
                schema_name="WebResearchPlan",
            )
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
            node_name=WEB_RESEARCH_V2_SUMMARIZER_NODE,
            schema_name="WebSourceSummaryBatch",
        )

    monkeypatch.setattr("src.graph.academic.invoke_structured_llm", fake_invoke_structured_llm)
    monkeypatch.setattr("src.graph.academic.web_search_fn", lambda *args, **kwargs: _search_result())

    with pytest.raises(RuntimeError, match=WEB_RESEARCH_V2_WARNING_ALL_SOURCES_REJECTED) as exc_info:
        await _web_search_dual_source(_state(), _branches(), {"mode": "dual_source_evidence"})

    debug = getattr(exc_info.value, "web_research_debug")
    final_stage = debug["stages"][-1]

    assert debug["status"] == "failed"
    assert debug["used_fallback"] is False
    assert final_stage["stage"] == WEB_RESEARCH_V2_STAGE_FAILED
    assert final_stage["summarizer_result_count"] == 1
    assert final_stage["summarizer_kept_count"] == 0
    assert WEB_RESEARCH_V2_WARNING_ALL_SOURCES_REJECTED in debug["developer_warnings"]


@pytest.mark.asyncio
async def test_search_task_timeout_raises_fail_fast(monkeypatch):
    monkeypatch.setattr("src.graph.academic.invoke_structured_llm", _successful_llm)

    def fake_web_search_fn(*args, **kwargs):
        raise TimeoutError("tavily down")

    monkeypatch.setattr("src.graph.academic.web_search_fn", fake_web_search_fn)

    with pytest.raises(RuntimeError, match="fallback is disabled"):
        await _web_search_dual_source(_state(), _branches(), {"mode": "dual_source_evidence"})


@pytest.mark.asyncio
async def test_fetch_failure_raises_fail_fast(monkeypatch):
    monkeypatch.setattr("src.graph.academic.invoke_structured_llm", _successful_llm)
    monkeypatch.setattr("src.graph.academic.web_search_fn", lambda *args, **kwargs: _search_result(content=""))

    with pytest.raises(RuntimeError, match="source fetch failed") as exc_info:
        await _web_search_dual_source(_state(), _branches(), {"mode": "dual_source_evidence"})

    debug = getattr(exc_info.value, "web_research_debug")

    assert debug["status"] == "failed"
    assert any(stage["stage"] == WEB_RESEARCH_V2_STAGE_FAILED for stage in debug["stages"])
