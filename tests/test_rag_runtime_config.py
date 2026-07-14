"""Regression coverage for generated portable runtime config and chunk dry runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.init_rag_runtime_config import main
from src.config.rag_index_config import (
    load_rag_index_config,
    resolve_rag_index_config_paths,
)
from src.rag.parent_child.build_audit import (
    BuildAuditError,
    collect_chunk_stats,
    render_chunk_stats_markdown,
)
from src.rag.subject_catalog import SubjectCatalog, SubjectPolicyMapError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_SUBJECTS = (
    "big_data",
    "computer",
    "machine_learning",
    "math",
    "python",
)


def test_tracked_local_config_uses_bounded_long_run_embedding_retry() -> None:
    config = load_rag_index_config(PROJECT_ROOT / "config" / "rag" / "index.local.yaml")

    assert config.embedding.retry.max_attempts == 10
    assert config.embedding.retry.initial_backoff_seconds == 1.0
    assert config.embedding.retry.max_backoff_seconds == 60.0
    assert config.embedding.retry.multiplier == 2.0
    assert config.embedding.batch_size == 8
    assert config.embedding.max_in_flight_batches == 1
    assert config.embedding.provider_routing is not None
    assert config.embedding.provider_routing.allow_fallbacks is False


def _write_source_config(project_root: Path) -> Path:
    """Clone the tracked strict template while pointing it at a test corpus."""

    template = load_rag_index_config(
        PROJECT_ROOT / "config" / "rag" / "index.local.yaml"
    )
    payload = template.model_dump(mode="json")
    catalog = dict(payload["catalog"])
    storage = dict(payload["storage"])
    catalog["data_root"] = str(project_root / "data")
    storage["index_root"] = str(project_root / "template_indexes")
    storage["registry_path"] = str(
        project_root / "template_indexes" / "generation_registry.sqlite"
    )
    payload["catalog"] = catalog
    payload["storage"] = storage
    path = project_root / "config" / "rag" / "index.local.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _write_corpus(
    project_root: Path,
    *,
    extra_subject: bool = False,
    protected_atomic_block: bool = False,
) -> Path:
    data_root = project_root / "data"
    for subject in EXPECTED_SUBJECTS:
        source = data_root / subject / "course_notes.txt"
        source.parent.mkdir(parents=True, exist_ok=True)
        content = (
            "# Concepts\n"
            "A deterministic page-aware chunking test source with enough text "
            "to exercise the loader and splitter without provider access.\n"
        )
        if protected_atomic_block and subject == "math":
            content += "```python\n" + ("x" * 800) + "\n```\n"
        source.write_text(
            content,
            encoding="utf-8",
        )
    for excluded in ("evaluation", "_needs_ocr", "unclassified", ".hidden", ".cache"):
        source = data_root / excluded / "ignored.txt"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("excluded source", encoding="utf-8")
    if extra_subject:
        source = data_root / "unexpected" / "notes.txt"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("unexpected subject", encoding="utf-8")
    return data_root


def _write_source_groups(project_root: Path) -> Path:
    path = project_root / "config" / "rag" / "source_groups.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "source_groups_v1",
                "source_groups": {
                    f"{subject}/course_notes.txt": f"{subject}_fixture"
                    for subject in EXPECTED_SUBJECTS
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _runtime_args(project_root: Path) -> list[str]:
    return [
        "--project-root",
        str(project_root),
        "--source-config",
        "config/rag/index.local.yaml",
        "--data-root",
        "data",
        "--index-root",
        "indexes/parent_child",
        "--registry-path",
        "generation_registry.sqlite",
        "--output",
        "config/rag/index.runtime.yaml",
    ]


def _runtime_config(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    protected_atomic_block: bool = False,
) -> Path:
    _write_corpus(project_root, protected_atomic_block=protected_atomic_block)
    _write_source_config(project_root)
    outside = project_root.parent / "outside_cwd"
    outside.mkdir()
    monkeypatch.chdir(outside)
    assert main(_runtime_args(project_root)) == 0
    return project_root / "config" / "rag" / "index.runtime.yaml"


def test_runtime_config_is_relative_and_resolves_from_non_root_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    runtime_path = _runtime_config(
        project_root,
        monkeypatch,
        protected_atomic_block=True,
    )

    payload = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    assert payload["catalog"]["data_root"] == "data"
    assert payload["storage"]["index_root"] == "indexes/parent_child"
    assert payload["storage"]["registry_path"] == "generation_registry.sqlite"

    loaded = load_rag_index_config(runtime_path)
    resolved = resolve_rag_index_config_paths(loaded, project_root=project_root)
    assert resolved.catalog.data_root == (project_root / "data").resolve()
    assert (
        resolved.storage.index_root
        == (project_root / "indexes" / "parent_child").resolve()
    )
    assert (
        resolved.storage.resolved_registry_path()
        == (
            project_root / "indexes" / "parent_child" / "generation_registry.sqlite"
        ).resolve()
    )
    snapshot = SubjectCatalog(
        config=resolved.catalog,
        subject_policy_map=resolved.subject_policy_map,
    ).discover()
    assert snapshot.subject_ids() == EXPECTED_SUBJECTS


def test_runtime_config_rejects_an_unmapped_subject(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    _write_corpus(project_root, extra_subject=True)
    _write_source_config(project_root)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SubjectPolicyMapError, match="missing policy"):
        main(_runtime_args(project_root))


def test_chunk_stats_executes_real_loader_and_splitter_without_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    runtime_path = _runtime_config(
        project_root,
        monkeypatch,
        protected_atomic_block=True,
    )
    source_groups = _write_source_groups(project_root)

    stats = collect_chunk_stats(
        project_root=project_root,
        config_path=runtime_path,
        generation_id="pc_fixture_dry_run",
        source_groups_path=source_groups,
    )

    assert tuple(subject.subject for subject in stats.subjects) == EXPECTED_SUBJECTS
    assert stats.total_sources == len(EXPECTED_SUBJECTS)
    assert stats.total_parents >= len(EXPECTED_SUBJECTS)
    assert stats.total_children >= len(EXPECTED_SUBJECTS)
    assert stats.orphan_child_count == 0
    assert stats.empty_content_count == 0
    assert stats.protected_atomic_block_count >= 1
    assert stats.protected_atomic_block_violation_count == 0
    assert stats.source_group_completeness.missing_source_relpaths == ()
    assert "# RAG chunk statistics" in render_chunk_stats_markdown(stats)
    assert not (project_root / "indexes" / "parent_child").exists()

    payload = json.loads(source_groups.read_text(encoding="utf-8"))
    del payload["source_groups"]["math/course_notes.txt"]
    source_groups.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(BuildAuditError, match="source-group manifest is incomplete"):
        collect_chunk_stats(
            project_root=project_root,
            config_path=runtime_path,
            generation_id="pc_fixture_dry_run",
            source_groups_path=source_groups,
        )
