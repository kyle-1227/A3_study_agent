"""
Dynamic Curriculum Engine — Knowledge Graph + Adaptive Learning Path Planner.

Builds on long-term memory (episodic + semantic) and user profile to compute
personalized learning paths with skip/reinforce/repeat decisions.
"""

from src.curriculum.types import KnowledgeNode, LearningPath, PathPlannerInput, PathStep
from src.curriculum.knowledge_graph import (
    KnowledgeGraph,
    load_knowledge_graph,
    get_topic,
    get_all_prerequisites,
    topological_sort,
)
from src.curriculum.path_planner import compute_learning_path

__all__ = [
    "KnowledgeNode",
    "LearningPath",
    "PathPlannerInput",
    "PathStep",
    "KnowledgeGraph",
    "load_knowledge_graph",
    "get_topic",
    "get_all_prerequisites",
    "topological_sort",
    "compute_learning_path",
]
