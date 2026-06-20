"""
Knowledge Graph — skill dependency graph loader and query engine.

Loads a YAML-defined graph of subjects, topics, and prerequisites.
Provides prerequisite traversal (BFS), topological sorting, and
dependency queries to drive the Dynamic Curriculum Engine path planner.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Optional

import yaml

from src.config import get_setting
from src.curriculum.types import KnowledgeNode

logger = logging.getLogger(__name__)

DEFAULT_KG_PATH = Path("data/knowledge_graph.yaml")


# ── Knowledge Graph Class ──────────────────────────────────────────────────


class KnowledgeGraph:
    """In-memory knowledge graph with prerequisite and topic queries.

    Loaded from a YAML file at init time. All queries are synchronous
    (O(N) at worst, N = number of topics).
    """

    def __init__(self, data: dict):
        """Build the graph from parsed YAML data.

        Args:
            data: Parsed YAML dict with ``subjects`` key.
        """
        self._topics: dict[str, KnowledgeNode] = {}
        self._subject_topics: dict[str, list[str]] = {}
        self._dependents: dict[str, set[str]] = {}  # topic_id → set of topics that depend on it

        for subject_key, subject_data in data.get("subjects", {}).items():
            topic_list = subject_data.get("topics", [])
            for topic_data in topic_list:
                topic_id = topic_data["id"]
                node = KnowledgeNode(
                    topic_id=topic_id,
                    name=topic_data.get("name", topic_id),
                    subject=subject_key,
                    difficulty=float(topic_data.get("difficulty", 0.5)),
                    estimated_hours=float(topic_data.get("estimated_hours", 10.0)),
                    prerequisites=list(topic_data.get("prerequisites", [])),
                    knowledge_points=list(topic_data.get("knowledge_points", [])),
                    resources=dict(topic_data.get("resources", {})),
                )
                self._topics[topic_id] = node
                self._subject_topics.setdefault(subject_key, []).append(topic_id)

        # Build reverse dependency map
        for topic_id, node in self._topics.items():
            for prereq_id in node.prerequisites:
                self._dependents.setdefault(prereq_id, set()).add(topic_id)

    # ── Query methods ──────────────────────────────────────────────────

    def get_topic(self, topic_id: str) -> KnowledgeNode | None:
        """Get a single topic by ID."""
        return self._topics.get(topic_id)

    def get_all_topics(self) -> list[KnowledgeNode]:
        """Get all topics in the graph."""
        return list(self._topics.values())

    def get_subject_topics(self, subject: str) -> list[KnowledgeNode]:
        """Get all topics in a subject."""
        topic_ids = self._subject_topics.get(subject, [])
        return [self._topics[tid] for tid in topic_ids if tid in self._topics]

    def get_subjects(self) -> list[str]:
        """List all subjects in the graph."""
        return list(self._subject_topics.keys())

    def get_prerequisites(self, topic_id: str) -> list[KnowledgeNode]:
        """Get direct prerequisites for a topic."""
        node = self._topics.get(topic_id)
        if not node:
            return []
        return [self._topics[pid] for pid in node.prerequisites if pid in self._topics]

    def get_all_prerequisites(self, topic_id: str) -> list[KnowledgeNode]:
        """Get all transitive prerequisites (BFS from topic upward).

        Returns prerequisites in dependency order (foundational first).
        """
        node = self._topics.get(topic_id)
        if not node:
            return []

        visited: set[str] = set()
        result: list[KnowledgeNode] = []

        def dfs(current_id: str):
            if current_id in visited:
                return
            visited.add(current_id)
            current = self._topics.get(current_id)
            if current is None:
                return
            for prereq_id in current.prerequisites:
                dfs(prereq_id)
            if current_id != topic_id:
                result.append(current)

        dfs(topic_id)
        return result  # Already in topological order from DFS post-order

    def get_dependents(self, topic_id: str) -> list[KnowledgeNode]:
        """Get all topics that depend on this topic (reverse lookup)."""
        dep_ids = self._dependents.get(topic_id, set())
        return [self._topics[did] for did in dep_ids if did in self._topics]

    def topological_sort(self, topic_ids: list[str] | None = None) -> list[KnowledgeNode]:
        """Topological sort of given topic IDs (or all topics) respecting prerequisites.

        Uses Kahn's algorithm. Returns topics in dependency order.
        """
        if topic_ids is None:
            topic_ids = list(self._topics.keys())

        target_set = set(topic_ids)

        # Build in-degree map (only counting prereqs within target set)
        in_degree: dict[str, int] = {}
        for tid in topic_ids:
            node = self._topics.get(tid)
            if node is None:
                continue
            in_degree[tid] = sum(1 for p in node.prerequisites if p in target_set)

        # Kahn's BFS
        queue: deque[str] = deque(tid for tid, deg in in_degree.items() if deg == 0)
        result: list[KnowledgeNode] = []

        while queue:
            tid = queue.popleft()
            node = self._topics.get(tid)
            if node:
                result.append(node)

            # Decrement in-degree for dependents
            for dep_id in self._dependents.get(tid, set()):
                if dep_id not in target_set:
                    continue
                in_degree[dep_id] = in_degree.get(dep_id, 0) - 1
                if in_degree[dep_id] == 0:
                    queue.append(dep_id)

        # If some nodes have cycles, add remaining in arbitrary order
        remaining = [tid for tid, deg in in_degree.items() if deg > 0]
        if remaining:
            logger.warning(
                "Knowledge graph has cycles involving: %s", remaining,
            )
            for tid in remaining:
                node = self._topics.get(tid)
                if node:
                    result.append(node)

        return result

    def get_available_resources(
        self, topic_id: str, resource_type: str | None = None,
    ) -> dict[str, list[str]]:
        """Get resources for a topic, optionally filtered by type."""
        node = self._topics.get(topic_id)
        if not node:
            return {}
        if resource_type:
            return {resource_type: node.resources.get(resource_type, [])}
        return dict(node.resources)

    def topic_count(self) -> int:
        return len(self._topics)


# ── Singleton ──────────────────────────────────────────────────────────────


_kg_instance: KnowledgeGraph | None = None


def load_knowledge_graph(path: str | Path | None = None) -> KnowledgeGraph:
    """Load the knowledge graph from YAML file (singleton).

    Args:
        path: Path to YAML file. Defaults to ``curriculum.knowledge_graph_path``
              from settings.yaml, then ``data/knowledge_graph.yaml``.

    Returns:
        KnowledgeGraph instance (cached after first load).
    """
    global _kg_instance
    if _kg_instance is not None and path is None:
        return _kg_instance

    if path is None:
        path = get_setting("curriculum.knowledge_graph_path", str(DEFAULT_KG_PATH))

    kg_path = Path(path)
    if not kg_path.is_absolute():
        kg_path = Path(__file__).parent.parent.parent / kg_path

    with open(kg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    _kg_instance = KnowledgeGraph(data)
    logger.info(
        "Loaded knowledge graph: %d topics across %d subjects",
        _kg_instance.topic_count(), len(_kg_instance.get_subjects()),
    )
    return _kg_instance


def reset_knowledge_graph() -> None:
    """Reset singleton (for testing)."""
    global _kg_instance
    _kg_instance = None


# ── Convenience functions (operate on singleton KG) ────────────────────────


def get_topic(topic_id: str) -> KnowledgeNode | None:
    """Get a single topic from the singleton knowledge graph."""
    kg = load_knowledge_graph()
    return kg.get_topic(topic_id)


def get_all_prerequisites(topic_id: str) -> list[KnowledgeNode]:
    """Get all transitive prerequisites for a topic."""
    kg = load_knowledge_graph()
    return kg.get_all_prerequisites(topic_id)


def topological_sort(topic_ids: list[str] | None = None) -> list[KnowledgeNode]:
    """Topological sort of topics respecting prerequisites."""
    kg = load_knowledge_graph()
    return kg.topological_sort(topic_ids)
