"""
Cognitive Graph Builder — constructs a user cognitive model graph.

Merges three data sources:
1. Knowledge Graph (static): topic nodes + prerequisite edges
2. User Profile (dynamic): skill nodes with level/confidence
3. Semantic Memory (dynamic): weakness nodes from weak_knowledge_points

Outputs nodes + edges suitable for React Flow visualization.
"""

from __future__ import annotations

import logging

from src.analytics.types import CognitiveEdge, CognitiveGraphData, CognitiveNode
from src.curriculum.knowledge_graph import KnowledgeGraph, load_knowledge_graph
from src.memory.storage import MemoryStore, create_memory_store
from src.profile.schema import LearningStyle, UserProfile

logger = logging.getLogger(__name__)

# Color mapping for node types
NODE_COLORS = {
    "skill": "#4f46e5",      # Indigo
    "weakness": "#ef4444",    # Red
    "preference": "#f59e0b",  # Amber
    "topic": "#10b981",       # Emerald
}


async def build_cognitive_graph(
    user_id: str,
    *,
    profile: UserProfile | None = None,
    subject: str = "",
    store: MemoryStore | None = None,
    kg: KnowledgeGraph | None = None,
) -> CognitiveGraphData:
    """Build the user's cognitive model graph.

    Args:
        user_id: User identifier.
        profile: UserProfile with skills and learning style.
        subject: Optional subject filter.
        store: MemoryStore for semantic memory weak points.
        kg: KnowledgeGraph for topic structure.

    Returns:
        CognitiveGraphData with nodes and edges.
    """
    kg = kg or load_knowledge_graph()
    store = store or create_memory_store()

    nodes: list[CognitiveNode] = []
    edges: list[CognitiveEdge] = []
    node_ids: set[str] = set()

    # ── 1. Topic nodes from Knowledge Graph ────────────────────────────
    if subject:
        topics = kg.get_subject_topics(subject)
    else:
        topics = kg.get_all_topics()

    for topic in topics[:20]:  # Limit to 20
        tid = f"topic:{topic.topic_id}"
        if tid in node_ids:
            continue
        node_ids.add(tid)
        nodes.append(CognitiveNode(
            id=tid,
            label=topic.name,
            type="topic",
            size=0.3 + topic.difficulty * 0.3,
            level=topic.difficulty,
            details=f"难度: {topic.difficulty:.0%}, 预计{topic.estimated_hours:.0f}h",
            color=NODE_COLORS["topic"],
        ))

    # ── 2. Skill nodes from Profile ────────────────────────────────────
    if profile and profile.skills:
        for skill_name, entry in profile.skills.items():
            if subject and subject.lower() not in skill_name.lower():
                # Check if skill maps to a topic
                matching = _find_matching_topics(kg, skill_name, subject)
                if not matching:
                    continue

            sid = f"skill:{skill_name}"
            if sid in node_ids:
                continue
            node_ids.add(sid)

            skill_level = entry.level
            skill_conf = entry.confidence
            size = skill_level * skill_conf * 0.6 + 0.2  # Min size 0.2

            nodes.append(CognitiveNode(
                id=sid,
                label=skill_name,
                type="skill",
                size=size,
                level=skill_level,
                confidence=skill_conf,
                details=f"水平: {skill_level:.0%}, 置信度: {skill_conf:.0%}, 观测{entry.evidence_count}次",
                color=NODE_COLORS["skill"],
            ))

            # Link skill to matching topic nodes
            for topic in kg.get_all_topics():
                tid = f"topic:{topic.topic_id}"
                if tid in node_ids and skill_name.lower() in topic.name.lower():
                    edges.append(CognitiveEdge(
                        source=sid,
                        target=tid,
                        weight=skill_level * skill_conf,
                        label="mastering",
                    ))

    # ── 3. Weakness nodes from Semantic Memory ────────────────────────
    try:
        summaries = await store.get_semantic_for_user(user_id, limit=10)
        weak_count: dict[str, int] = {}
        for s in summaries:
            for wp in s.weak_knowledge_points or []:
                weak_count[wp] = weak_count.get(wp, 0) + 1

        max_count = max(weak_count.values()) if weak_count else 1
        for wp, count in weak_count.items():
            wid = f"weakness:{wp}"
            if wid in node_ids:
                continue
            node_ids.add(wid)

            # More frequent = larger node
            size = 0.3 + (count / max_count) * 0.5

            nodes.append(CognitiveNode(
                id=wid,
                label=wp,
                type="weakness",
                size=size,
                details=f"出现{count}次，需要加强",
                color=NODE_COLORS["weakness"],
            ))

            # Link weakness to related topic nodes
            for topic in kg.get_all_topics():
                tid = f"topic:{topic.topic_id}"
                if tid in node_ids and wp.lower() in topic.name.lower():
                    edges.append(CognitiveEdge(
                        source=tid,
                        target=wid,
                        weight=count / max_count,
                        label="has_weakness",
                    ))
    except Exception as exc:
        logger.debug("Failed to load semantic memory for cognitive graph: %s", exc)

    # ── 4. Preference nodes from Learning Style ───────────────────────
    if profile:
        style = profile.learning_style
        style_dims = [
            ("prefer_examples", "喜欢案例", style.prefer_examples),
            ("prefer_visual", "喜欢可视化", style.prefer_visual),
            ("prefer_step_by_step", "喜欢分步讲解", style.prefer_step_by_step),
            ("prefer_concise", "喜欢简洁", style.prefer_concise),
            ("prefer_theory", "喜欢理论", style.prefer_theory),
            ("prefer_practice", "喜欢实践", style.prefer_practice),
            ("prefer_analogy", "喜欢类比", style.prefer_analogy),
        ]
        for dim_key, dim_label, dim_val in style_dims:
            if dim_val > 0.65:  # Only show strong preferences
                pid = f"preference:{dim_key}"
                if pid not in node_ids:
                    node_ids.add(pid)
                    nodes.append(CognitiveNode(
                        id=pid,
                        label=dim_label,
                        type="preference",
                        size=dim_val * 0.7,
                        details=f"偏好强度: {dim_val:.0%}",
                        color=NODE_COLORS["preference"],
                    ))

    # ── 5. KG prerequisite edges ──────────────────────────────────────
    for topic in topics:
        tid = f"topic:{topic.topic_id}"
        if tid not in node_ids:
            continue
        for prereq_id in topic.prerequisites:
            pid = f"topic:{prereq_id}"
            if pid in node_ids:
                edges.append(CognitiveEdge(
                    source=pid,
                    target=tid,
                    weight=0.8,
                    label="requires",
                ))

    summary_parts = [
        f"技能节点: {sum(1 for n in nodes if n.type == 'skill')}",
        f"薄弱节点: {sum(1 for n in nodes if n.type == 'weakness')}",
        f"偏好节点: {sum(1 for n in nodes if n.type == 'preference')}",
        f"主题节点: {sum(1 for n in nodes if n.type == 'topic')}",
        f"依赖边: {sum(1 for e in edges if e.label == 'requires')}",
    ]

    return CognitiveGraphData(
        user_id=user_id,
        nodes=nodes,
        edges=edges,
        summary="; ".join(summary_parts),
        node_count=len(nodes),
        edge_count=len(edges),
    )


def _find_matching_topics(kg: KnowledgeGraph, skill_name: str, subject: str) -> list:
    """Find KG topics matching a skill name within a subject."""
    skill_lower = skill_name.lower()
    matches = []
    for node in kg.get_all_topics():
        if subject and node.subject != subject:
            continue
        if skill_lower in node.name.lower():
            matches.append(node)
            continue
        for kp in node.knowledge_points:
            if skill_lower in kp.lower():
                matches.append(node)
                break
    return matches
