"""
Memory Dashboard — aggregator that bundles all analytics data into one response.

The dashboard endpoint calls growth analyzer, cognitive graph builder,
and explainability engine in parallel, then returns a unified DashboardData.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from src.analytics.cognitive_graph import build_cognitive_graph
from src.analytics.explainability_engine import get_decision_traces
from src.analytics.growth_analyzer import analyze_growth
from src.analytics.types import DashboardData, DecisionTraceList, GrowthAnalytics, CognitiveGraphData
from src.memory.storage import MemoryStore, create_memory_store
from src.profile.schema import UserProfile

logger = logging.getLogger(__name__)


async def get_dashboard_data(
    user_id: str,
    *,
    profile: UserProfile | None = None,
    subject: str = "",
    days: int = 30,
    store: MemoryStore | None = None,
) -> DashboardData:
    """Aggregate all analytics data for the dashboard.

    Fires growth analysis, cognitive graph, and decision traces in parallel.

    Args:
        user_id: The user to analyze.
        profile: UserProfile (optional — loaded if not provided).
        subject: Optional subject filter.
        days: Growth analysis time window.
        store: MemoryStore instance.

    Returns:
        DashboardData with all analytics.
    """
    store = store or create_memory_store()

    # Run all three queries in parallel
    growth_task = analyze_growth(user_id, subject=subject, days=days, store=store)
    cognitive_task = build_cognitive_graph(
        user_id, profile=profile, subject=subject, store=store,
    )
    decisions_task = get_decision_traces(user_id, limit=10, store=store)

    growth, cognitive, decisions = await asyncio.gather(
        growth_task, cognitive_task, decisions_task,
        return_exceptions=True,
    )

    # Handle partial failures gracefully
    if isinstance(growth, Exception):
        logger.warning("Growth analysis failed: %s", growth)
        growth = None
    if isinstance(cognitive, Exception):
        logger.warning("Cognitive graph failed: %s", cognitive)
        cognitive = None
    if isinstance(decisions, Exception):
        logger.warning("Decision traces failed: %s", decisions)
        decisions = None

    # Build stats summary
    stats: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": subject or "all",
        "days": days,
    }

    if growth:
        stats["total_events"] = growth.total_events
        stats["overall_accuracy"] = growth.overall_accuracy
        stats["overall_trend"] = growth.overall_trend
        stats["series_count"] = len(growth.series)
    if cognitive:
        stats["cognitive_nodes"] = cognitive.node_count
        stats["cognitive_edges"] = cognitive.edge_count
    if decisions:
        stats["decision_traces"] = decisions.total

    return DashboardData(
        user_id=user_id,
        growth=growth,
        cognitive_graph=cognitive,
        recent_decisions=decisions,
        stats_summary=stats,
    )
