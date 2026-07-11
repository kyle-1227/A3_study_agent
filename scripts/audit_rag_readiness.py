"""Run a read-only RAG corpus/readiness audit from explicit configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

project_root_from_script = Path(__file__).resolve().parent.parent
if str(project_root_from_script) not in sys.path:
    sys.path.insert(0, str(project_root_from_script))

from src.config.rag_benchmark_config import load_rag_benchmark_config  # noqa: E402
from src.rag.parent_child._storage_io import (  # noqa: E402
    atomic_write_bytes,
    model_json_bytes,
)
from src.rag.readiness import (  # noqa: E402
    QueryInventoryRecord,
    ReadinessAuditArtifact,
    audit_rag_readiness,
    load_query_inventory,
    load_source_group_manifest,
)


def _contained_path(project_root: Path, value: Path, *, must_exist: bool) -> Path:
    root = project_root.resolve(strict=True)
    candidate = value if value.is_absolute() else root / value
    if candidate.is_symlink():
        raise ValueError("configured paths must not be symlinks")
    resolved = candidate.resolve(strict=must_exist)
    if not resolved.is_relative_to(root):
        raise ValueError("configured path must remain inside project_root")
    return resolved


def _relative_label(project_root: Path, path: Path) -> str:
    return path.resolve(strict=False).relative_to(project_root).as_posix()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run_audit(
    *,
    project_root: Path,
    benchmark_config_path: Path,
    data_root: Path,
    output_path: Path,
) -> ReadinessAuditArtifact:
    """Run and persist one explicit audit; missing datasets remain blockers."""

    root = project_root.resolve(strict=True)
    config_path = _contained_path(root, benchmark_config_path, must_exist=True)
    corpus_root = _contained_path(root, data_root, must_exist=True)
    output = _contained_path(root, output_path, must_exist=False)
    benchmark = load_rag_benchmark_config(config_path)
    source_groups_path = _contained_path(
        root, benchmark.source_group_manifest_path, must_exist=True
    )
    source_groups = load_source_group_manifest(source_groups_path)

    records: list[QueryInventoryRecord] = []
    missing_paths: list[str] = []
    for configured_path in (
        *benchmark.human_gold_paths,
        *benchmark.historical_annotated_paths,
        *benchmark.synthetic_smoke_paths,
    ):
        path = _contained_path(root, configured_path, must_exist=False)
        if not path.exists():
            missing_paths.append(_relative_label(root, path))
            continue
        records.extend(load_query_inventory(path))

    report = audit_rag_readiness(
        data_root=corpus_root,
        primary_subjects=benchmark.primary_subjects,
        supported_extensions=(".pdf", ".md", ".txt"),
        source_group_manifest=source_groups,
        query_records=tuple(records),
        low_text_page_chars=benchmark.low_text_page_chars,
        minimum_independent_sources=benchmark.min_independent_sources,
        minimum_subject_gold_queries=benchmark.min_subject_gold_queries,
        minimum_global_gold_queries=benchmark.min_global_gold_queries,
    )
    artifact = ReadinessAuditArtifact(
        schema_version="rag_readiness_artifact_v1",
        benchmark_config_path=_relative_label(root, config_path),
        data_root=_relative_label(root, corpus_root),
        missing_dataset_paths=tuple(sorted(missing_paths)),
        report=report,
    )
    relative_output = output.relative_to(root).as_posix()
    atomic_write_bytes(
        root, relative_output, model_json_bytes(artifact), overwrite=True
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = run_audit(
        project_root=args.project_root,
        benchmark_config_path=args.benchmark_config,
        data_root=args.data_root,
        output_path=args.output,
    )
    print(
        "RAG readiness audit written: "
        f"blocked={artifact.report.production_recommendation_blocked}, "
        f"subjects={len(artifact.report.subjects)}, "
        f"missing_datasets={len(artifact.missing_dataset_paths)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
