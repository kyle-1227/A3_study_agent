"""
Growth Analyzer — tracks skill mastery over time from episodic memory.

Reads episodic memory (quiz_attempt + error + learning_behavior events)
and semantic memory (growth trajectories) to build time-series growth curves.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from src.analytics.types import GrowthAnalytics, GrowthDataPoint, GrowthSeries
from src.memory.storage import MemoryStore, create_memory_store

logger = logging.getLogger(__name__)


async def analyze_growth(
    user_id: str,
    *,
    subject: str = "",
    days: int = 30,
    store: MemoryStore | None = None,
) -> GrowthAnalytics:
    """Analyze skill growth over time from episodic memory.

    Queries episodic memories of types quiz_attempt, error, and learning_behavior
    over the specified time window, groups by subject/topic, and computes
    rolling accuracy and skill level estimates.

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

    # Query episodic memories
    try:
        records = await store.query_episodic(
            user_id,
            start_time=start_time,
            limit=500,
        )
    except Exception as exc:
        logger.exception("Failed to query episodic memories for growth analysis")
        return GrowthAnalytics(
            user_id=user_id, subject=subject, days=days,
            generated_at=now.isoformat(),
        )

    if not records:
        return GrowthAnalytics(
            user_id=user_id, subject=subject, days=days,
            generated_at=now.isoformat(),
        )

    # Filter by subject if specified
    if subject:
        subject_lower = subject.lower()
        records = [
            r for r in records
            if subject_lower in (r.subject or "").lower()
            or subject_lower in (r.content or "").lower()
        ]

    # Group by topic (use subject as topic if topic not available)
    topic_groups: dict[str, list] = defaultdict(list)
    for rec in records:
        topic = rec.subject or "general"
        if subject and subject.lower() not in topic.lower():
            topic = f"{subject}/{topic}"
        topic_groups[topic].append(rec)

    # Build time series per topic
    series_list: list[GrowthSeries] = []
    total_correct = 0
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

            # Estimate accuracy from errors (errors = low accuracy)
            errors = [r for r in day_recs if r.memory_type == "error"]
            behaviors = [r for r in day_recs if r.memory_type == "learning_behavior"]

            if errors:
                running_accuracy = max(0.0, running_accuracy - 0.1 * len(errors))
            elif behaviors:
                running_accuracy = min(1.0, running_accuracy + 0.05 * len(behaviors))

            # Adjust skill level based on importance-weighted events
            avg_importance = (
                sum(r.importance for r in day_recs) / len(day_recs)
                if day_recs else 0.5
            )
            if errors:
                running_level = max(0.0, running_level - 0.05 * avg_importance)
            elif behaviors:
                running_level = min(1.0, running_level + 0.03 * avg_importance)

            data_points.append(GrowthDataPoint(
                timestamp=f"{date_key}T00:00:00Z",
                skill_level=round(running_level, 3),
                accuracy=round(running_accuracy, 3),
                topic=topic,
                event_count=len(day_recs),
                subject=subject,
            ))

        # Determine trend
        trend = _compute_trend(data_points)

        series_list.append(GrowthSeries(
            topic=topic,
            subject=subject,
            data_points=data_points,
            trend=trend,
            current_level=data_points[-1].skill_level if data_points else 0.0,
        ))

        total_correct += sum(
            1 for r in recs if r.memory_type != "error"
        )
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
