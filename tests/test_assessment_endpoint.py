"""SSE endpoint tests for strict checkpoint-bound assessment attempts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

import app as app_module
from src.assessment.attempt_contracts import (
    AdaptivePracticeBatchV1,
    AdaptivePracticeTaskV1,
    AssessmentAttemptV1,
    AssessmentCheckpointResourcesV2,
    AssessmentErrorClassificationV1,
    AssessmentQuestionRecordV1,
    AssessmentLearningGuidanceBindingV1,
    AssessmentResourceRecordV2,
    PrivateExerciseAnswerKeyV1,
    PublicExerciseCardV1,
)
from src.assessment.attempt_journal import (
    LocalAssessmentExecutionLock,
    assessment_attempt_journal_reducer,
)
from src.assessment.identity import (
    stable_adaptive_practice_question_id,
    stable_exercise_question_id,
)
from src.learning_guidance.adapters.history import HistorySnapshotAdapterV1
from src.learning_guidance.history_writer import (
    LearningGuidanceHistoryWriterError,
    LearningGuidanceHistoryWriterV1,
)
from src.learning_guidance.knowledge_graph import KnowledgeGraphV1
from src.memory.storage import SQLiteMemoryStore
from src.streaming.session import StreamSessionManager
from src.streaming.settings import StreamingRuntimeConfig

THREAD_ID = "thread-assessment-endpoint-1"
REQUEST_ID = "00000000-0000-4000-8000-000000000301"
RESOURCE_ID = f"resource:v3:{'a' * 64}"
OTHER_RESOURCE_ID = f"resource:v3:{'b' * 64}"
QUESTION = "What is 2 + 2?"
QUESTION_ID = stable_exercise_question_id(
    level="basic",
    question_type="free_text",
    question=QUESTION,
    choices=(),
    tags=("arithmetic",),
)
ADAPTIVE_QUESTION = "What is 1 + 2?"
ADAPTIVE_QUESTION_ID = stable_adaptive_practice_question_id(
    task_type="review",
    question=ADAPTIVE_QUESTION,
    tags=("arithmetic",),
    difficulty=0.2,
)


def _checkpoint(
    *,
    learning_guidance_binding: AssessmentLearningGuidanceBindingV1 | None = None,
) -> AssessmentCheckpointResourcesV2:
    return AssessmentCheckpointResourcesV2(
        schema_version="assessment_checkpoint_resources_v2",
        thread_id=THREAD_ID,
        resources=(
            AssessmentResourceRecordV2(
                schema_version="assessment_resource_record_v2",
                resource_id=RESOURCE_ID,
                learning_guidance_binding=learning_guidance_binding,
                questions=(
                    AssessmentQuestionRecordV1(
                        schema_version="assessment_question_record_v1",
                        card=PublicExerciseCardV1(
                            schema_version="exercise_card_v1",
                            question_id=QUESTION_ID,
                            question_type="free_text",
                            level="basic",
                            question=QUESTION,
                            choices=(),
                            tags=("arithmetic",),
                        ),
                        answer_key=PrivateExerciseAnswerKeyV1(
                            schema_version="exercise_answer_key_v1",
                            question_id=QUESTION_ID,
                            accepted_answers=("4",),
                            match_mode="exact",
                            answer_explanation="Two plus two equals four.",
                        ),
                    ),
                ),
            ),
        ),
    )


def _attempt(
    *,
    answer: str = "4",
    resource_id: str = RESOURCE_ID,
) -> AssessmentAttemptV1:
    return AssessmentAttemptV1(
        schema_version="assessment_attempt_v1",
        request_id=REQUEST_ID,
        resource_id=resource_id,
        question_id=QUESTION_ID,
        answer=answer,
        time_spent_seconds=7.5,
    )


def _classification() -> AssessmentErrorClassificationV1:
    return AssessmentErrorClassificationV1(
        schema_version="assessment_error_classification_v1",
        error_type="concept",
        concept_gap="The addition fact is not stable.",
        suggestion="Review number composition.",
        confidence=0.95,
    )


def _adaptive_batch() -> AdaptivePracticeBatchV1:
    return AdaptivePracticeBatchV1(
        schema_version="adaptive_practice_batch_v1",
        tasks=(
            AdaptivePracticeTaskV1(
                schema_version="adaptive_practice_task_v1",
                question_id=ADAPTIVE_QUESTION_ID,
                task_type="review",
                question=ADAPTIVE_QUESTION,
                answer="3",
                explanation="One plus two equals three.",
                reason="Review a simpler fact after the concept error.",
                tags=("arithmetic",),
                difficulty=0.2,
            ),
        ),
    )


class _CheckpointGraph:
    _a3_node_ids = frozenset({"resource_bundle_output"})

    def __init__(
        self,
        *,
        with_checkpoint: bool = True,
        learning_guidance_binding: AssessmentLearningGuidanceBindingV1 | None = None,
    ) -> None:
        self.values = (
            {
                "thread_id": THREAD_ID,
                "current_node": "",
                "last_completed_node": "resource_bundle_output",
                "assessment_checkpoint_resources": _checkpoint(
                    learning_guidance_binding=learning_guidance_binding,
                ).model_dump(mode="json"),
            }
            if with_checkpoint
            else {}
        )
        self.update_count = 0

    async def aget_state(self, _config):
        return SimpleNamespace(values=dict(self.values), tasks=(), next=())

    async def aupdate_state(self, _config, values, *, as_node: str):
        assert as_node == "resource_bundle_output"
        self.update_count += 1
        if "assessment_attempt_journal" in values:
            self.values["assessment_attempt_journal"] = (
                assessment_attempt_journal_reducer(
                    self.values.get("assessment_attempt_journal", {}),
                    values["assessment_attempt_journal"],
                )
            )


def _manager() -> StreamSessionManager:
    return StreamSessionManager(
        StreamingRuntimeConfig(
            retry_ms=1000,
            journal_max_events=20,
            journal_max_bytes=100_000,
            journal_ttl_seconds=60,
        )
    )


def _request(
    graph: _CheckpointGraph,
    service,
    *,
    history_writer: LearningGuidanceHistoryWriterV1 | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                graph=graph,
                assessment_attempt_service=service,
                learning_guidance_history_writer=history_writer,
            )
        )
    )


def _guidance_binding() -> AssessmentLearningGuidanceBindingV1:
    return AssessmentLearningGuidanceBindingV1(
        schema_version="assessment_learning_guidance_binding_v1",
        user_id="learner-1",
        subject="math",
        topic_id="math.algebra",
        resource_type="quiz",
        generation_request_id="00000000-0000-4000-8000-000000000302",
        assignment_contract_version="resource_evidence_assignment_v1",
        assignment_fingerprint="c" * 64,
    )


def _guidance_graph() -> KnowledgeGraphV1:
    return KnowledgeGraphV1.model_validate(
        {
            "schema_version": "knowledge_graph_v1",
            "data_version": "assessment-endpoint-test-v1",
            "subjects": [
                {
                    "subject_id": "math",
                    "title": "Mathematics",
                    "topics": [
                        {
                            "topic_id": "math.algebra",
                            "title": "Algebra",
                            "difficulty": 0.4,
                            "estimated_hours": 2.0,
                            "prerequisite_topic_ids": [],
                            "knowledge_points": ["Equations"],
                            "resources": [
                                {
                                    "resource_id": "math.algebra.review",
                                    "resource_type": "review_doc",
                                    "title": "Algebra review",
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )


async def _payloads(response) -> list[dict]:
    frames: list[str] = []
    async for chunk in response.body_iterator:
        frames.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
    return [
        json.loads(
            next(
                line.removeprefix("data: ")
                for line in frame.splitlines()
                if line.startswith("data: ")
            )
        )
        for frame in frames
    ]


def _service(
    monkeypatch: pytest.MonkeyPatch,
    graph: _CheckpointGraph,
) -> tuple[object, AsyncMock, AsyncMock]:
    classifier = AsyncMock(return_value=_classification())
    generator = AsyncMock(return_value=_adaptive_batch())
    monkeypatch.setattr(app_module, "classify_assessment_error_v1", classifier)
    monkeypatch.setattr(app_module, "generate_adaptive_practice_v1", generator)
    return (
        app_module._build_assessment_attempt_service(
            graph,
            execution_lock=LocalAssessmentExecutionLock(),
        ),
        classifier,
        generator,
    )


@pytest.mark.anyio
async def test_correct_attempt_streams_and_replays_one_authoritative_final(
    monkeypatch: pytest.MonkeyPatch,
):
    graph = _CheckpointGraph()
    service, classifier, generator = _service(monkeypatch, graph)
    monkeypatch.setattr(app_module, "stream_session_manager", _manager())
    request = _request(graph, service)

    first_response = await app_module.assessment_attempt_endpoint(
        THREAD_ID,
        _attempt(),
        request,
    )
    first = await _payloads(first_response)
    second_response = await app_module.assessment_attempt_endpoint(
        THREAD_ID,
        _attempt(),
        request,
    )
    second = await _payloads(second_response)

    assert [item["type"] for item in first] == [
        "stream_start",
        "activity_update",
        "assessment_final",
        "stream_done",
    ]
    assert [item["event_id"] for item in second] == [item["event_id"] for item in first]
    assert first[1]["data"]["kind"] == "assessment_history"
    assert first[1]["data"]["payload"]["status"] == "unavailable"
    assert (
        first[1]["data"]["payload"]["reason"] == "assessment_topic_binding_unavailable"
    )
    final = first[2]["data"]
    assert final["terminal_status"] == "correct"
    assert final["request_id"] == REQUEST_ID
    assert final["thread_id"] == THREAD_ID
    assert first[-1]["data"]["terminal_type"] == "assessment_final"
    assert graph.update_count == 2
    classifier.assert_not_awaited()
    generator.assert_not_awaited()
    private_keys = {
        "answer_key",
        "accepted_answers",
        "canonical_correct_answer",
        "answer_explanation",
    }
    assert private_keys.isdisjoint(final)


@pytest.mark.anyio
async def test_bound_assessment_persists_history_before_final_and_replays_after_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    graph = _CheckpointGraph(learning_guidance_binding=_guidance_binding())
    database = tmp_path / "memory.sqlite"
    writer = LearningGuidanceHistoryWriterV1(
        store=SQLiteMemoryStore(database),
        knowledge_graph=_guidance_graph(),
    )
    service, _, _ = _service(monkeypatch, graph)
    monkeypatch.setattr(app_module, "stream_session_manager", _manager())

    first_response = await app_module.assessment_attempt_endpoint(
        THREAD_ID,
        _attempt(),
        _request(graph, service, history_writer=writer),
    )
    first = await _payloads(first_response)

    assert [item["type"] for item in first] == [
        "stream_start",
        "activity_update",
        "assessment_final",
        "stream_done",
    ]
    assert first[1]["data"]["kind"] == "assessment_history"
    assert first[1]["data"]["payload"]["status"] == "inserted"
    assert first[2]["data"]["terminal_status"] == "correct"

    restarted_service, _, _ = _service(monkeypatch, graph)
    monkeypatch.setattr(app_module, "stream_session_manager", _manager())
    replay_response = await app_module.assessment_attempt_endpoint(
        THREAD_ID,
        _attempt(),
        _request(
            graph,
            restarted_service,
            history_writer=LearningGuidanceHistoryWriterV1(
                store=SQLiteMemoryStore(database),
                knowledge_graph=_guidance_graph(),
            ),
        ),
    )
    replay = await _payloads(replay_response)

    assert replay[1]["data"]["payload"]["status"] == "replayed"
    assert (
        replay[1]["data"]["payload"]["history_id"]
        == (first[1]["data"]["payload"]["history_id"])
    )
    snapshot = await HistorySnapshotAdapterV1(
        store=SQLiteMemoryStore(database),
        knowledge_graph=_guidance_graph(),
        history_limit=10,
    ).load("learner-1", "math")
    assert snapshot is not None
    assert len(snapshot.events) == 1
    assert snapshot.events[0].history_id == (first[1]["data"]["payload"]["history_id"])


@pytest.mark.anyio
async def test_transient_history_failure_recovers_with_same_request_without_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    graph = _CheckpointGraph(learning_guidance_binding=_guidance_binding())
    writer = LearningGuidanceHistoryWriterV1(
        store=SQLiteMemoryStore(tmp_path / "memory.sqlite"),
        knowledge_graph=_guidance_graph(),
    )
    original_write = writer.write_assessment_once
    attempts = 0

    async def flaky_write(*, binding, record):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise LearningGuidanceHistoryWriterError(
                code="learning_guidance_history_persist_failed"
            )
        return await original_write(binding=binding, record=record)

    monkeypatch.setattr(writer, "write_assessment_once", flaky_write)
    service, classifier, generator = _service(monkeypatch, graph)
    monkeypatch.setattr(app_module, "stream_session_manager", _manager())
    request = _request(graph, service, history_writer=writer)

    first = await _payloads(
        await app_module.assessment_attempt_endpoint(
            THREAD_ID,
            _attempt(),
            request,
        )
    )
    second = await _payloads(
        await app_module.assessment_attempt_endpoint(
            THREAD_ID,
            _attempt(),
            request,
        )
    )

    assert [item["type"] for item in first] == [
        "stream_start",
        "stream_error",
        "stream_done",
    ]
    assert first[1]["data"]["recoverable"] is True
    assert [item["type"] for item in second] == [
        "stream_start",
        "activity_update",
        "assessment_final",
        "stream_done",
    ]
    assert second[1]["data"]["payload"]["status"] == "inserted"
    assert attempts == 2
    assert graph.update_count == 2
    classifier.assert_not_awaited()
    generator.assert_not_awaited()


@pytest.mark.anyio
async def test_wrong_attempt_returns_classification_and_complete_new_practice(
    monkeypatch: pytest.MonkeyPatch,
):
    graph = _CheckpointGraph()
    service, classifier, generator = _service(monkeypatch, graph)
    monkeypatch.setattr(app_module, "stream_session_manager", _manager())

    response = await app_module.assessment_attempt_endpoint(
        THREAD_ID,
        _attempt(answer="wrong-private-answer-5"),
        _request(graph, service),
    )
    payloads = await _payloads(response)

    final = payloads[-2]["data"]
    assert final["terminal_status"] == "incorrect"
    assert final["error_classification"]["error_type"] == "concept"
    assert final["adaptive_tasks"] == [
        _adaptive_batch().tasks[0].model_dump(mode="json")
    ]
    assert final["adaptive_tasks"][0]["question"] != QUESTION
    assert "wrong-private-answer-5" not in json.dumps(final, ensure_ascii=False)
    classifier.assert_awaited_once()
    generator.assert_awaited_once()
    journal_json = json.dumps(
        graph.values["assessment_attempt_journal"],
        ensure_ascii=False,
    )
    assert "wrong-private-answer-5" not in journal_json
    assert "accepted_answers" not in journal_json


@pytest.mark.anyio
async def test_request_id_payload_drift_returns_http_conflict(
    monkeypatch: pytest.MonkeyPatch,
):
    graph = _CheckpointGraph()
    service, _, _ = _service(monkeypatch, graph)
    monkeypatch.setattr(app_module, "stream_session_manager", _manager())
    request = _request(graph, service)
    first = await app_module.assessment_attempt_endpoint(
        THREAD_ID,
        _attempt(),
        request,
    )
    await _payloads(first)

    with pytest.raises(HTTPException) as exc_info:
        await app_module.assessment_attempt_endpoint(
            THREAD_ID,
            _attempt(answer="5"),
            request,
        )
    assert exc_info.value.status_code == 409


@pytest.mark.anyio
async def test_unknown_resource_is_a_safe_stream_error_and_is_not_cached(
    monkeypatch: pytest.MonkeyPatch,
):
    graph = _CheckpointGraph()
    service, classifier, generator = _service(monkeypatch, graph)
    monkeypatch.setattr(app_module, "stream_session_manager", _manager())

    response = await app_module.assessment_attempt_endpoint(
        THREAD_ID,
        _attempt(resource_id=OTHER_RESOURCE_ID),
        _request(graph, service),
    )
    payloads = await _payloads(response)

    assert [item["type"] for item in payloads] == [
        "stream_start",
        "stream_error",
        "stream_done",
    ]
    assert payloads[-2]["data"]["error_type"] == "assessment_resource_not_found"
    assert "assessment_attempt_journal" not in graph.values
    classifier.assert_not_awaited()
    generator.assert_not_awaited()


@pytest.mark.anyio
async def test_provider_failure_is_safe_and_replays_after_process_restart(
    monkeypatch: pytest.MonkeyPatch,
):
    graph = _CheckpointGraph()
    private_canary = "private-provider-answer-canary-814"
    classifier = AsyncMock(side_effect=RuntimeError(private_canary))
    generator = AsyncMock(return_value=_adaptive_batch())
    monkeypatch.setattr(app_module, "classify_assessment_error_v1", classifier)
    monkeypatch.setattr(app_module, "generate_adaptive_practice_v1", generator)
    first_service = app_module._build_assessment_attempt_service(
        graph,
        execution_lock=LocalAssessmentExecutionLock(),
    )
    monkeypatch.setattr(app_module, "stream_session_manager", _manager())

    first_response = await app_module.assessment_attempt_endpoint(
        THREAD_ID,
        _attempt(answer="wrong-answer"),
        _request(graph, first_service),
    )
    first_payloads = await _payloads(first_response)

    assert [item["type"] for item in first_payloads] == [
        "stream_start",
        "stream_error",
        "stream_done",
    ]
    first_error = first_payloads[1]["data"]
    assert first_error["error_type"] == "assessment_error_classification_failed"
    assert private_canary not in json.dumps(first_payloads, ensure_ascii=False)
    assert private_canary not in json.dumps(
        graph.values["assessment_attempt_journal"],
        ensure_ascii=False,
    )
    classifier.assert_awaited_once()
    generator.assert_not_awaited()

    replay_classifier = AsyncMock(side_effect=AssertionError("must not dispatch"))
    replay_generator = AsyncMock(side_effect=AssertionError("must not dispatch"))
    monkeypatch.setattr(
        app_module,
        "classify_assessment_error_v1",
        replay_classifier,
    )
    monkeypatch.setattr(
        app_module,
        "generate_adaptive_practice_v1",
        replay_generator,
    )
    replay_service = app_module._build_assessment_attempt_service(
        graph,
        execution_lock=LocalAssessmentExecutionLock(),
    )
    monkeypatch.setattr(app_module, "stream_session_manager", _manager())

    replay_response = await app_module.assessment_attempt_endpoint(
        THREAD_ID,
        _attempt(answer="wrong-answer"),
        _request(graph, replay_service),
    )
    replay_payloads = await _payloads(replay_response)

    assert replay_payloads[1]["data"] == first_error
    replay_classifier.assert_not_awaited()
    replay_generator.assert_not_awaited()


@pytest.mark.anyio
async def test_missing_checkpoint_returns_http_not_found(
    monkeypatch: pytest.MonkeyPatch,
):
    graph = _CheckpointGraph(with_checkpoint=False)
    service, _, _ = _service(monkeypatch, graph)
    monkeypatch.setattr(app_module, "stream_session_manager", _manager())

    with pytest.raises(HTTPException) as exc_info:
        await app_module.assessment_attempt_endpoint(
            THREAD_ID,
            _attempt(),
            _request(graph, service),
        )
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "assessment_checkpoint_not_found"


def test_openapi_exposes_only_the_strict_assessment_attempt_contract():
    app_module.app.openapi_schema = None
    schema = app_module.app.openapi()
    operation = schema["paths"]["/threads/{thread_id}/assessment-attempts"]["post"]
    request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    component_name = request_schema["$ref"].rsplit("/", 1)[-1]
    component = schema["components"]["schemas"][component_name]

    assert component["additionalProperties"] is False
    assert set(component["required"]) == {
        "schema_version",
        "request_id",
        "resource_id",
        "question_id",
        "answer",
        "time_spent_seconds",
    }
    assert component["properties"]["schema_version"]["const"] == (
        "assessment_attempt_v1"
    )
    request_id_pattern = component["properties"]["request_id"]["pattern"]
    assert re.fullmatch(request_id_pattern, REQUEST_ID)
    assert re.fullmatch(
        request_id_pattern,
        "019f5543-e3a2-7e31-8ed1-a62409098772",
    )
    assert re.fullmatch(request_id_pattern, "request-assessment-endpoint-1") is None
