"""Production inventory gates for the checked-in learning-guidance graph."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath

from src.config.learning_guidance_config import load_learning_guidance_config
from src.learning_guidance.factory import (
    load_learning_guidance_runtime,
    resolve_knowledge_graph_path,
)
from src.learning_guidance.knowledge_graph import load_knowledge_graph


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "learning_guidance.yaml"
ARTIFACT_PATH = (
    PROJECT_ROOT / "config" / "learning_guidance" / "knowledge_graph_v1.yaml"
)
SOURCE_GROUP_MANIFEST_PATH = PROJECT_ROOT / "config" / "rag" / "source_groups.json"
SUBJECT_IDS = ("big_data", "computer", "machine_learning", "math", "python")
ARTIFACT_FINGERPRINT = (
    "c504e41ef2e481b30b940ac6cb04f661401f7907d1690efeafc1ed14680fa0b5"
)


def _source_group_subjects() -> dict[str, str]:
    payload = json.loads(SOURCE_GROUP_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "source_groups_v1"
    source_groups = payload["source_groups"]
    assert isinstance(source_groups, dict)

    subjects_by_group: dict[str, set[str]] = {}
    for source_path, source_group_id in source_groups.items():
        assert isinstance(source_path, str)
        assert isinstance(source_group_id, str)
        parts = PurePosixPath(source_path).parts
        if parts[0] == "data":
            parts = parts[1:]
        subject = parts[0]
        assert subject in SUBJECT_IDS
        subjects_by_group.setdefault(source_group_id, set()).add(subject)

    assert subjects_by_group
    assert all(len(subjects) == 1 for subjects in subjects_by_group.values())
    return {
        source_group_id: next(iter(subjects))
        for source_group_id, subjects in subjects_by_group.items()
    }


def test_production_artifact_is_strict_complete_and_source_backed() -> None:
    graph = load_knowledge_graph(ARTIFACT_PATH)

    assert graph.data_version == "2026.07.15-source-groups-v1"
    assert graph.artifact_fingerprint == ARTIFACT_FINGERPRINT
    assert tuple(subject.subject_id for subject in graph.subjects) == SUBJECT_IDS

    topic_ids = tuple(
        topic.topic_id for subject in graph.subjects for topic in subject.topics
    )
    resources = tuple(
        (subject.subject_id, resource)
        for subject in graph.subjects
        for topic in subject.topics
        for resource in topic.resources
    )
    resource_ids = tuple(resource.resource_id for _, resource in resources)
    assert len(topic_ids) == len(set(topic_ids))
    assert len(resource_ids) == len(set(resource_ids))
    assert all(resource.resource_type == "review_doc" for _, resource in resources)

    expected_subjects = _source_group_subjects()
    actual_subjects = {
        resource.resource_id: subject_id for subject_id, resource in resources
    }
    assert actual_subjects == expected_subjects

    for subject in graph.subjects:
        position_by_topic = {
            topic.topic_id: position for position, topic in enumerate(subject.topics)
        }
        for topic in subject.topics:
            assert all(
                position_by_topic[prerequisite_id] < position_by_topic[topic.topic_id]
                for prerequisite_id in topic.prerequisite_topic_ids
            )


def test_checked_in_config_loads_production_artifact_and_runtime(
    tmp_path: Path,
) -> None:
    config = load_learning_guidance_config(CONFIG_PATH)
    resolved = resolve_knowledge_graph_path(config=config, project_root=PROJECT_ROOT)
    assert resolved == ARTIFACT_PATH.resolve()

    runtime = load_learning_guidance_runtime(
        config_path=CONFIG_PATH,
        project_root=PROJECT_ROOT,
        profile_db_path=tmp_path / "profile.sqlite",
        memory_db_path=tmp_path / "memory.sqlite",
        clock=lambda: datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc),
    )

    assert runtime.knowledge_graph.artifact_fingerprint == ARTIFACT_FINGERPRINT
    assert (
        tuple(subject.subject_id for subject in runtime.knowledge_graph.subjects)
        == SUBJECT_IDS
    )
