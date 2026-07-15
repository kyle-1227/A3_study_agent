"""
Growth Analyzer — tracks skill mastery over time from episodic memory.

Reads only authoritative, assessment-bound guidance history when computing
learning outcomes. Ordinary chat and generic behavior remain engagement data;
they are never converted into correctness or skill growth.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from src.analytics.types import GrowthAnalytics, GrowthDataPoint, GrowthSeries
from src.learning_guidance.history_contract import (
    LEARNING_GUIDANCE_HISTORY_ID_PREFIX,
    LearningGuidanceHistoryBindingV1,
)
from src.memory.retention import is_protected_episodic_memory_id
from src.memory.schema import EpisodicMemoryRecord
from src.memory.storage import MemoryStore, create_memory_store


async def analyze_growth(
    user_id: str,
    *,
    subject: str = "",
    days: int = 30,
    store: MemoryStore | None = None,
) -> GrowthAnalytics:
    """Analyze skill growth over time from episodic memory.

    Queries episodic memory over the specified time window, retains only strict
    assessment history records, groups by their bound topic, and computes rolling
    accuracy and skill level estimates.

    Args:
        user_id: The user to analyze.
        subject: Optional subject filter.
        days: Time window in days (default 30).
        store: MemoryStore instance.

    Returns:
        GrowthAnalytics with time-series data per topic.
    """
    store = store or create_memory_store()
    now = datetime.now(timezone.utc)
    start_time = (now - timedelta(days=days)).isoformat()

    records = await store.query_episodic(
        user_id,
        memory_id_prefix=LEARNING_GUIDANCE_HISTORY_ID_PREFIX,
        start_time=start_time,
        limit=500,
    )

    assessment_records: list[EpisodicMemoryRecord] = []
    for record in records:
        binding = _guidance_assessment_binding(record)
        if binding is None:
            continue
        if subject and record.subject != subject:
            continue
        assessment_records.append(record)
    records = assessment_records
    if not records:
        return GrowthAnalytics(
            user_id=user_id,
            subject=subject,
            days=days,
            generated_at=now.isoformat(),
        )

    # Group by topic (use subject as topic if topic not available)
    topic_groups: dict[str, list[EpisodicMemoryRecord]] = defaultdict(list)
    for rec in records:
        guidance = _guidance_assessment_binding(rec)
        if guidance is None:
            raise AssertionError("assessment record lost its strict history binding")
        topic_groups[guidance.topic_id].append(rec)

    # Build time series per topic
    series_list: list[GrowthSeries] = []
    total_correct = 0.0
    total_events = 0

    for topic, recs in topic_groups.items():
        # Sort by time
        recs.sort(key=lambda r: r.created_at or "")

        # Bin into daily windows
        daily_bins: dict[str, list] = defaultdict(list)
        for rec in recs:
            date_key = (rec.created_at or "")[:10]  # YYYY-MM-DD
            daily_bins[date_key].append(rec)

        data_points: list[GrowthDataPoint] = []
        running_level = 0.3  # Start from low estimate
        running_accuracy = 0.5
        event_count = 0

        for date_key in sorted(daily_bins.keys()):
            day_recs = daily_bins[date_key]
            event_count += len(day_recs)

            guidance_scores = [
                binding.outcome_score
                for binding in (
                    _guidance_assessment_binding(record) for record in day_recs
                )
                if binding is not None and binding.outcome_score is not None
            ]

            if not guidance_scores:
                raise AssertionError("assessment day has no authoritative outcome")
            observed_accuracy = sum(guidance_scores) / len(guidance_scores)
            running_accuracy = running_accuracy * 0.6 + observed_accuracy * 0.4

            # Adjust skill level based on importance-weighted events
            avg_importance = (
                sum(r.importance for r in day_recs) / len(day_recs) if day_recs else 0.5
            )
            delta = (observed_accuracy - 0.5) * 0.06 * avg_importance
            running_level = min(1.0, max(0.0, running_level + delta))

            data_points.append(
                GrowthDataPoint(
                    timestamp=f"{date_key}T00:00:00Z",
                    skill_level=round(running_level, 3),
                    accuracy=round(running_accuracy, 3),
                    topic=topic,
                    event_count=len(day_recs),
                    subject=subject,
                )
            )

        # Determine trend
        trend = _compute_trend(data_points)

        series_list.append(
            GrowthSeries(
                topic=topic,
                subject=subject,
                data_points=data_points,
                trend=trend,
                current_level=data_points[-1].skill_level if data_points else 0.0,
            )
        )

        for record in recs:
            guidance = _guidance_assessment_binding(record)
            if guidance is not None:
                if guidance.outcome_score is None:
                    raise ValueError(
                        "assessment guidance history requires outcome_score"
                    )
                total_correct += guidance.outcome_score
        total_events += len(recs)

    overall_accuracy = total_correct / max(total_events, 1)
    overall_trend = _overall_trend(series_list)

    return GrowthAnalytics(
        user_id=user_id,
        subject=subject,
        days=days,
        series=series_list,
        overall_accuracy=round(overall_accuracy, 3),
        overall_trend=overall_trend,
        total_events=total_events,
        generated_at=now.isoformat(),
    )


def _guidance_assessment_binding(
    record: EpisodicMemoryRecord,
) -> LearningGuidanceHistoryBindingV1 | None:
    if not is_protected_episodic_memory_id(record.memory_id):
        return None
    raw = record.metadata.get("learning_guidance_v1")
    binding = LearningGuidanceHistoryBindingV1.model_validate(raw)
    if binding.event_type != "assessment":
        raise ValueError("protected guidance history must be an assessment event")
    return binding


def _compute_trend(points: list[GrowthDataPoint]) -> str:
    """Determine trend from first half vs second half average level."""
    if len(points) < 2:
        return "stable"
    mid = len(points) // 2
    first_half = [p.skill_level for p in points[:mid]]
    second_half = [p.skill_level for p in points[mid:]]
    avg_first = sum(first_half) / len(first_half) if first_half else 0
    avg_second = sum(second_half) / len(second_half) if second_half else 0
    diff = avg_second - avg_first
    if diff > 0.05:
        return "improving"
    elif diff < -0.05:
        return "declining"
    return "stable"


def _overall_trend(series: list[GrowthSeries]) -> str:
    """Aggregate trend across all series."""
    trends = [s.trend for s in series]
    improving = trends.count("improving")
    declining = trends.count("declining")
    if improving > declining:
        return "improving"
    elif declining > improving:
        return "declining"
    return "stable"
