"""
Learning Growth Analytics — skill mastery tracking, cognitive graph, explainable AI.

Components:
- Growth Analyzer: skill mastery curves from episodic memory
- Cognitive Graph: user cognitive model from KG + profile + memory
- Explainability Engine: agent decision trace recording + retrieval
- Memory Dashboard: aggregator for unified analytics API
"""

from src.analytics.types import (
    CognitiveEdge,
    CognitiveGraphData,
    CognitiveNode,
    DashboardData,
    DecisionTrace,
    DecisionTraceList,
    GrowthAnalytics,
    GrowthDataPoint,
    GrowthSeries,
)
from src.analytics.growth_analyzer import analyze_growth
from src.analytics.cognitive_graph import build_cognitive_graph
from src.analytics.explainability_engine import (
    get_decision_traces,
    record_decision_from_state,
    record_decision_trace,
)
from src.analytics.memory_dashboard import get_dashboard_data

__all__ = [
    # Types
    "CognitiveEdge",
    "CognitiveGraphData",
    "CognitiveNode",
    "DashboardData",
    "DecisionTrace",
    "DecisionTraceList",
    "GrowthAnalytics",
    "GrowthDataPoint",
    "GrowthSeries",
    # Functions
    "analyze_growth",
    "build_cognitive_graph",
    "get_decision_traces",
    "get_dashboard_data",
    "record_decision_from_state",
    "record_decision_trace",
]
