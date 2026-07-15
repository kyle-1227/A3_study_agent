"""Fail-closed projection of durable assessment facts into guidance history."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ValidationError

from src.assessment.attempt_contracts import AssessmentLearningGuidanceBindingV1
from src.assessment.attempt_journal import AssessmentAttemptRecordV1
from src.learning_guidance.history_contract import (
    AssessmentHistorySourceV1,
    LEARNING_GUIDANCE_HISTORY_ID_PREFIX,
    LearningGuidanceHistoryBindingV1,
    LearningGuidanceHistoryWriteResultV1,
)
from src.learning_guidance.knowledge_graph import KnowledgeGraphV1
from src.memory.schema import EpisodicMemoryRecord
from src.memory.storage import MemoryStorageWriteError, SQLiteMemoryStore


class LearningGuidanceHistoryWriterError(RuntimeError):
    """Content-safe typed failure from assessment history projection."""

    def __init__(self, *, code: str) -> None:
        self.code = code
        super().__init__(f"{code}: learning-guidance history write failed")


def _python_boundary_payload(value: object) -> object:
    if isinstance(value, BaseModel):
        return {
            field_name: _python_boundary_payload(getattr(value, field_name))
            for field_name in type(value).model_fields
        }
    if type(value) is dict:
        return {key: _python_boundary_payload(item) for key, item in value.items()}
    if type(value) is list:
        return [_python_boundary_payload(item) for item in value]
    if type(value) is tuple:
        return tuple(_python_boundary_payload(item) for item in value)
    return value


class LearningGuidanceHistoryWriterV1:
    """Write only journal-committed assessment facts, exactly once."""

    def __init__(
        self,
        *,
        store: SQLiteMemoryStore,
        knowledge_graph: KnowledgeGraphV1,
    ) -> None:
        if not isinstance(store, SQLiteMemoryStore):
            raise TypeError("store must be SQLiteMemoryStore")
        if not isinstance(knowledge_graph, KnowledgeGraphV1):
            raise TypeError("knowledge_graph must be KnowledgeGraphV1")
        self._store = store
        self._knowledge_graph = knowledge_graph

    @property
    def store(self) -> SQLiteMemoryStore:
        return self._store

    async def write_assessment_once(
        self,
        *,
        binding: AssessmentLearningGuidanceBindingV1,
        record: AssessmentAttemptRecordV1,
    ) -> LearningGuidanceHistoryWriteResultV1:
        if not isinstance(binding, AssessmentLearningGuidanceBindingV1):
            raise TypeError("binding must be AssessmentLearningGuidanceBindingV1")
        if not isinstance(record, AssessmentAttemptRecordV1):
            raise TypeError("record must be AssessmentAttemptRecordV1")
        try:
            validated_binding = AssessmentLearningGuidanceBindingV1.model_validate(
                _python_boundary_payload(binding),
                strict=True,
            )
            validated_record = AssessmentAttemptRecordV1.model_validate(
                _python_boundary_payload(record),
                strict=True,
            )
        except ValidationError:
            raise LearningGuidanceHistoryWriterError(
                code="assessment_history_source_invalid"
            ) from None
        binding = validated_binding
        record = validated_record
        if record.status != "completed" or record.final is None:
            raise LearningGuidanceHistoryWriterError(
                code="assessment_history_source_not_completed"
            )
        if record.committed_at is None:
            raise LearningGuidanceHistoryWriterError(
                code="assessment_history_committed_at_missing"
            )
        final = record.final
        if final.request_id != record.request_id:
            raise LearningGuidanceHistoryWriterError(
                code="assessment_history_request_identity_mismatch"
            )
        if not self._topic_is_bound(
            subject=binding.subject,
            topic_id=binding.topic_id,
        ):
            raise LearningGuidanceHistoryWriterError(
                code="assessment_history_topic_invalid"
            )

        source = AssessmentHistorySourceV1(
            schema_version="assessment_history_source_v1",
            source_kind="assessment_attempt_v1",
            thread_id=final.thread_id,
            assessment_request_id=record.request_id,
            assessment_final_payload_hash=final.payload_hash,
            resource_id=final.resource_id,
            question_id=final.question_id,
            generation_request_id=binding.generation_request_id,
            assignment_contract_version=binding.assignment_contract_version,
            assignment_fingerprint=binding.assignment_fingerprint,
        )
        history_id = _assessment_history_id(source)
        observed_at = record.committed_at.isoformat()
        marker = LearningGuidanceHistoryBindingV1(
            schema_version="learning_guidance_history_event_v1",
            topic_id=binding.topic_id,
            event_type="assessment",
            observed_at=observed_at,
            outcome_score=1.0 if final.is_correct else 0.0,
        )
        fact = EpisodicMemoryRecord(
            memory_id=history_id,
            user_id=binding.user_id,
            memory_type="quiz_attempt",
            content=(
                f"Assessment outcome recorded for {binding.topic_id}: "
                f"{'correct' if final.is_correct else 'incorrect'}."
            ),
            importance=1.0,
            subject=binding.subject,
            metadata={
                "learning_guidance_v1": marker.model_dump(mode="json"),
                "assessment_history_source_v1": source.model_dump(mode="json"),
            },
            embedding=None,
            created_at=observed_at,
        )
        try:
            inserted = await self._store.insert_episodic_once_strict(fact)
        except MemoryStorageWriteError as exc:
            code = (
                "learning_guidance_history_event_conflict"
                if exc.code == "episodic_insert_conflict"
                else "learning_guidance_history_persist_failed"
            )
            raise LearningGuidanceHistoryWriterError(code=code) from exc
        try:
            return LearningGuidanceHistoryWriteResultV1(
                schema_version="learning_guidance_history_write_result_v1",
                status="inserted" if inserted else "replayed",
                history_id=history_id,
            )
        except ValidationError as exc:
            raise LearningGuidanceHistoryWriterError(
                code="learning_guidance_history_result_invalid"
            ) from exc

    def _topic_is_bound(self, *, subject: str, topic_id: str) -> bool:
        subject_node = self._knowledge_graph.subject(subject)
        return subject_node is not None and any(
            topic.topic_id == topic_id for topic in subject_node.topics
        )


def _assessment_history_id(source: AssessmentHistorySourceV1) -> str:
    identity = {
        "algorithm": "learning_guidance_assessment_history_v1",
        "source_kind": source.source_kind,
        "thread_id": source.thread_id,
        "assessment_request_id": source.assessment_request_id,
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{LEARNING_GUIDANCE_HISTORY_ID_PREFIX}{hashlib.sha256(encoded).hexdigest()}"


__all__ = [
    "LearningGuidanceHistoryWriterError",
    "LearningGuidanceHistoryWriterV1",
]
