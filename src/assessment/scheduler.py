"""
Spaced Repetition Scheduler — Ebbinghaus forgetting curve based review scheduling.

Implements expanding-interval review: 1, 3, 7, 14, 30 days.
- Correct answer → advance to next interval
- Wrong answer → reset to day 1

Integrates with episodic memory to track review history and with the
recommendation engine to surface due reviews.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.assessment.types import ReviewSchedule
from src.config import get_setting
from src.memory.storage import MemoryStore, create_memory_store

logger = logging.getLogger(__name__)

# Default Ebbinghaus intervals (days)
DEFAULT_INTERVALS = [1, 3, 7, 14, 30]


class SpacedRepetitionScheduler:
    """Ebbinghaus forgetting curve based spaced repetition scheduler.

    Algorithm:
    - Intervals: [1, 3, 7, 14, 30] days (configurable)
    - Correct answer → advance to next interval index
    - Wrong answer → reset interval index to 0 (1 day)
    - Max intervals capped at len(intervals) - 1
    """

    def __init__(self, intervals: list[int] | None = None):
        self.intervals = intervals or self._load_intervals()

    @staticmethod
    def _load_intervals() -> list[int]:
        """Load intervals from settings or use defaults."""
        try:
            configured = get_setting("assessment.spaced_repetition.intervals", None)
            if configured and isinstance(configured, list):
                return [int(x) for x in configured]
        except Exception:
            pass
        return list(DEFAULT_INTERVALS)

    def compute_next_review(
        self,
        current_interval_index: int,
        was_correct: bool,
    ) -> tuple[int, int]:
        """Compute the next review interval.

        Args:
            current_interval_index: Current position in intervals array.
            was_correct: Whether the last attempt was correct.

        Returns:
            Tuple of (new_interval_index, days_until_next_review).
        """
        if not was_correct:
            # Reset to first interval
            return 0, self.intervals[0]

        # Advance to next interval (capped)
        new_index = min(current_interval_index + 1, len(self.intervals) - 1)
        return new_index, self.intervals[new_index]

    def compute_next_review_date(
        self,
        current_interval_index: int,
        was_correct: bool,
        from_date: datetime | None = None,
    ) -> str:
        """Compute the ISO timestamp for the next review.

        Args:
            current_interval_index: Current position.
            was_correct: Whether the last attempt was correct.
            from_date: Reference date (defaults to now).

        Returns:
            ISO 8601 timestamp string.
        """
        if from_date is None:
            from_date = datetime.now(timezone.utc)

        _, days = self.compute_next_review(current_interval_index, was_correct)
        next_date = from_date + timedelta(days=days)
        return next_date.isoformat()

    def is_due(self, next_review_at: str) -> bool:
        """Check if a review item is due now."""
        if not next_review_at:
            return True
        try:
            due_dt = datetime.fromisoformat(next_review_at.replace("Z", "+00:00"))
            return datetime.now(timezone.utc) >= due_dt
        except (ValueError, TypeError):
            return True  # Unparseable → assume due

    def create_schedule(
        self,
        knowledge_point: str,
        subject: str,
        topic: str = "",
    ) -> ReviewSchedule:
        """Create a new review schedule starting at interval 0."""
        next_date = self.compute_next_review_date(0, True)
        return ReviewSchedule(
            knowledge_point=knowledge_point,
            subject=subject,
            topic=topic,
            intervals=list(self.intervals),
            current_interval_index=0,
            next_review_at=next_date,
            review_count=0,
            last_performance=0.0,
            is_due=False,
        )

    def update_schedule(
        self,
        schedule: ReviewSchedule,
        was_correct: bool,
    ) -> ReviewSchedule:
        """Update a schedule after a review attempt.

        Advances or resets the interval based on correctness.
        """
        new_index, _ = self.compute_next_review(schedule.current_interval_index, was_correct)
        next_date = self.compute_next_review_date(
            schedule.current_interval_index, was_correct,
        )
        performance = 1.0 if was_correct else 0.0

        return ReviewSchedule(
            knowledge_point=schedule.knowledge_point,
            subject=schedule.subject,
            topic=schedule.topic,
            intervals=list(schedule.intervals),
            current_interval_index=new_index,
            next_review_at=next_date,
            review_count=schedule.review_count + 1,
            last_performance=performance,
            last_reviewed_at=datetime.now(timezone.utc).isoformat(),
            is_due=False,
        )


# ── Convenience functions ──────────────────────────────────────────────────


async def get_due_reviews(
    user_id: str,
    store: MemoryStore | None = None,
) -> list[ReviewSchedule]:
    """Get review schedules that are due today.

    Reads review schedules stored as episodic memory entries of type
    ``system_event`` with ``metadata.review_schedule``.

    Args:
        user_id: The user identifier.
        store: MemoryStore instance.

    Returns:
        List of due ReviewSchedule items.
    """
    store = store or create_memory_store()
    scheduler = SpacedRepetitionScheduler()

    try:
        # Query recent system events that contain review schedules
        records = await store.query_episodic(
            user_id,
            memory_type="system_event",
            limit=100,
        )
    except Exception as exc:
        logger.debug("Failed to query review schedules for user=%s: %s", user_id, exc)
        return []

    due_schedules: list[ReviewSchedule] = []
    for rec in records:
        meta = rec.metadata or {}
        schedule_data = meta.get("review_schedule")
        if not isinstance(schedule_data, dict):
            continue

        try:
            schedule = ReviewSchedule(**schedule_data)
            if scheduler.is_due(schedule.next_review_at):
                schedule.is_due = True
                due_schedules.append(schedule)
        except Exception:
            continue

    logger.debug(
        "Found %d due reviews out of %d schedules for user=%s",
        len(due_schedules), len(records), user_id,
    )
    return due_schedules


async def save_review_schedule(
    user_id: str,
    schedule: ReviewSchedule,
    store: MemoryStore | None = None,
) -> None:
    """Persist a review schedule as an episodic system_event.

    Args:
        user_id: The user identifier.
        schedule: The review schedule to save.
        store: MemoryStore instance.
    """
    store = store or create_memory_store()

    from src.memory.schema import EpisodicMemoryRecord

    record = EpisodicMemoryRecord(
        user_id=user_id,
        memory_type="system_event",
        content=f"Spaced repetition schedule for {schedule.knowledge_point} (interval={schedule.current_interval_index})",
        importance=0.4,
        subject=schedule.subject,
        metadata={"review_schedule": schedule.model_dump()},
    )
    await store.save_episodic(record)
