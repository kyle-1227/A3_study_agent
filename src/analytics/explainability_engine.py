"""
Explainability Engine — records and retrieves agent decision traces.

Every key graph node decision is recorded as an episodic system_event
with a structured decision_trace in metadata. This enables full
explainability: why did the agent make this decision, what evidence
was used, what reasoning steps were followed.

Output structure per trace:
    { decision, evidence, reasoning_steps, confidence }
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from src.analytics.types import DecisionTrace, DecisionTraceList
from src.memory.schema import EpisodicMemoryRecord
from src.memory.storage import MemoryStore, create_memory_store

logger = logging.getLogger(__name__)


async def record_decision_trace(
    user_id: str,
    node_name: str,
    decision: str,
    *,
    evidence: str = "",
    reasoning_steps: list[str] | None = None,
    confidence: float = 0.5,
    subject: str = "",
    store: MemoryStore | None = None,
) -> DecisionTrace:
    """Record an agent decision trace as an episodic memory.

    Args:
        user_id: Thread/user identifier.
        node_name: Graph node that made the decision (e.g. "supervisor", "evidence_judge").
        decision: The final decision made (e.g. "intent=academic", "keep evidence item").
        evidence: Concise summary of evidence used.
        reasoning_steps: Ordered list of reasoning steps leading to the decision.
        confidence: How confident the agent was (0–1).
        subject: Academic subject context.
        store: MemoryStore instance.

    Returns:
        The DecisionTrace that was recorded.
    """
    store = store or create_memory_store()

    trace_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    trace = DecisionTrace(
        trace_id=trace_id,
        node_name=node_name,
        timestamp=now,
        decision=decision,
        evidence=evidence[:500] if evidence else "",
        reasoning_steps=list(reasoning_steps or []),
        confidence=confidence,
        thread_id=user_id,
        subject=subject,
    )

    # Store as episodic memory (system_event type)
    record = EpisodicMemoryRecord(
        user_id=user_id,
        memory_type="system_event",
        content=f"[DecisionTrace] {node_name}: {decision[:200]}",
        importance=0.4,
        subject=subject,
        metadata={
            "decision_trace": trace.model_dump(),
            "trace_type": "agent_decision",
        },
    )

    try:
        await store.save_episodic(record)
        logger.debug(
            "Recorded decision trace id=%s node=%s decision=%s",
            trace_id,
            node_name,
            decision[:100],
        )
    except Exception:
        logger.exception("Failed to persist decision trace id=%s", trace_id)

    return trace


async def get_decision_traces(
    user_id: str,
    *,
    limit: int = 20,
    node_name: str | None = None,
    store: MemoryStore | None = None,
) -> DecisionTraceList:
    """Retrieve recent decision traces for a user.

    Queries episodic memories of type system_event that contain
    decision_trace metadata.

    Args:
        user_id: User identifier.
        limit: Max traces to return.
        node_name: Optional filter by graph node name.
        store: MemoryStore instance.

    Returns:
        DecisionTraceList with recent traces.
    """
    store = store or create_memory_store()

    try:
        records = await store.query_episodic(
            user_id,
            memory_type="system_event",
            limit=limit * 2,  # Query more to filter
        )
    except Exception:
        logger.exception("Failed to query decision traces")
        return DecisionTraceList(user_id=user_id, traces=[], total=0)

    traces: list[DecisionTrace] = []
    for rec in records:
        meta = rec.metadata or {}
        trace_data = meta.get("decision_trace")
        if not isinstance(trace_data, dict):
            continue
        try:
            trace = DecisionTrace(**trace_data)
            if node_name and trace.node_name != node_name:
                continue
            traces.append(trace)
        except Exception:
            continue

    traces.sort(key=lambda t: t.timestamp, reverse=True)
    traces = traces[:limit]

    logger.debug(
        "Retrieved %d decision traces for user=%s (node_filter=%s)",
        len(traces),
        user_id,
        node_name or "none",
    )
    return DecisionTraceList(user_id=user_id, traces=traces, total=len(traces))


async def record_decision_from_state(
    state: dict[str, Any],
    node_name: str,
    decision: str,
    *,
    evidence: str = "",
    reasoning_steps: list[str] | None = None,
    confidence: float = 0.5,
) -> DecisionTrace:
    """Convenience: record a decision trace directly from graph state.

    Extracts user_id and subject from state automatically.

    Args:
        state: LearningState dict from a graph node.
        node_name: Graph node name.
        decision: The decision made.
        evidence: Evidence summary.
        reasoning_steps: Reasoning chain.
        confidence: Confidence score.

    Returns:
        DecisionTrace that was recorded.
    """
    thread_id = state.get("thread_id")
    if (
        not isinstance(thread_id, str)
        or not thread_id.strip()
        or thread_id != thread_id.strip()
    ):
        raise ValueError("decision trace requires a normalized thread_id")
    subject = state.get("subject", "") or state.get("primary_subject", "")

    return await record_decision_trace(
        user_id=thread_id,
        node_name=node_name,
        decision=decision,
        evidence=evidence,
        reasoning_steps=reasoning_steps,
        confidence=confidence,
        subject=subject,
    )
