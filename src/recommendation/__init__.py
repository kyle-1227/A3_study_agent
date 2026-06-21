"""
Memory-driven Recommendation Engine — multi-factor scoring + ranked recommendations.

Combines weakness, forgetting curve, learning preference, and goal alignment
to produce explainable, ranked learning resource recommendations.
"""

from src.recommendation.types import Recommendation, RecommendationList, ScoreBreakdown
from src.recommendation.scorer import (
    weakness_score,
    forgetting_score,
    preference_score,
    goal_score,
    compute_combined_score,
)
from src.recommendation.engine import generate_recommendations

__all__ = [
    "Recommendation",
    "RecommendationList",
    "ScoreBreakdown",
    "weakness_score",
    "forgetting_score",
    "preference_score",
    "goal_score",
    "compute_combined_score",
    "generate_recommendations",
]
