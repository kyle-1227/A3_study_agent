"""Strict projection from content-free episodic metadata into guidance history."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import ValidationError

from src.learning_guidance.contracts import (
    LearnerHistoryEventV1,
    LearnerHistorySnapshotV1,
)
from src.learning_guidance.knowledge_graph import KnowledgeGraphV1
from src.learning_guidance.history_contract import (
    LEARNING_GUIDANCE_HISTORY_ID_PREFIX,
    LearningGuidanceHistoryBindingV1,
)
from src.memory.storage import SQLiteMemoryStore


HISTORY_ADAPTER_VERSION = "learning_guidance_history_adapter_v1"
HistoryAdapterErrorCode: TypeAlias = Literal[
    "history_binding_schema_invalid",
    "history_binding_topic_invalid",
    "history_snapshot_invalid",
]


class HistoryAdapterError(RuntimeError):
    """Content-safe typed failure for a declared V1 history marker."""

    def __init__(self, *, code: HistoryAdapterErrorCode) -> None:
        self.code = code
        super().__init__(f"{code}: learning-guidance history binding failed")


def _topic_is_bound(
    knowledge_graph: KnowledgeGraphV1,
    *,
    subject: str,
    topic_id: str,
) -> bool:
    subject_node = knowledge_graph.subject(subject)
    return subject_node is not None and any(
        topic.topic_id == topic_id for topic in subject_node.topics
    )


class HistorySnapshotAdapterV1:
    """Real strict adapter over a bounded SQLite metadata query."""

    version = HISTORY_ADAPTER_VERSION

    def __init__(
        self,
        *,
        store: SQLiteMemoryStore,
        knowledge_graph: KnowledgeGraphV1,
        history_limit: int,
    ) -> None:
        if not isinstance(store, SQLiteMemoryStore):
            raise TypeError("store must be SQLiteMemoryStore")
        if not isinstance(knowledge_graph, KnowledgeGraphV1):
            raise TypeError("knowledge_graph must be KnowledgeGraphV1")
        if (
            isinstance(history_limit, bool)
            or not isinstance(history_limit, int)
            or not 1 <= history_limit <= 500
        ):
            raise ValueError("history_limit must be between 1 and 500")
        self._store = store
        self._knowledge_graph = knowledge_graph
        self._history_limit = history_limit

    async def load(
        self,
        user_id: str,
        subject: str,
    ) -> LearnerHistorySnapshotV1 | None:
        records = await self._store.query_episodic_metadata_strict(
            user_id=user_id,
            subject=subject,
            memory_id_prefix=LEARNING_GUIDANCE_HISTORY_ID_PREFIX,
            limit=self._history_limit,
        )
        events: list[LearnerHistoryEventV1] = []
        for record in records:
            if "learning_guidance_v1" not in record.metadata:
                raise HistoryAdapterError(code="history_binding_schema_invalid")
            raw_binding = record.metadata["learning_guidance_v1"]
            try:
                binding = LearningGuidanceHistoryBindingV1.model_validate(raw_binding)
                observed_at = binding.parsed_observed_at()
            except (ValidationError, ValueError):
                raise HistoryAdapterError(
                    code="history_binding_schema_invalid"
                ) from None
            if not _topic_is_bound(
                self._knowledge_graph,
                subject=record.subject,
                topic_id=binding.topic_id,
            ):
                raise HistoryAdapterError(code="history_binding_topic_invalid")
            events.append(
                LearnerHistoryEventV1(
                    history_id=record.memory_id,
                    subject=record.subject,
                    topic_id=binding.topic_id,
                    event_type=binding.event_type,
                    observed_at=observed_at,
                    outcome_score=binding.outcome_score,
                )
            )
        if not events:
            return None
        try:
            return LearnerHistorySnapshotV1(
                schema_version="learner_history_snapshot_v1",
                user_id=user_id,
                subject=subject,
                events=tuple(events),
            )
        except ValidationError:
            raise HistoryAdapterError(code="history_snapshot_invalid") from None


__all__ = [
    "HISTORY_ADAPTER_VERSION",
    "HistoryAdapterError",
    "HistorySnapshotAdapterV1",
    "LearningGuidanceHistoryBindingV1",
]
