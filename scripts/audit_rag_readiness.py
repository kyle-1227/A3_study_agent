"""Run a read-only RAG readiness audit from strict config and GoldDataset input."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

project_root_from_script = Path(__file__).resolve().parent.parent
if str(project_root_from_script) not in sys.path:
    sys.path.insert(0, str(project_root_from_script))

from src.config.rag_benchmark_config import load_rag_benchmark_config  # noqa: E402
from src.config.rag_index_config import (  # noqa: E402
    RagIndexConfig,
    load_rag_index_config,
    resolve_rag_index_config_paths,
)
from src.rag.gold_dataset import (  # noqa: E402
    GoldDatasetAuthoringError,
    gold_dataset_to_query_inventory,
    load_gold_dataset,
    project_relative_path,
    resolve_project_path,
)
from src.rag.parent_child._storage_io import (  # noqa: E402
    model_json_bytes,
    sha256_file,
)
from src.rag.parent_child.project_paths import atomic_write_project_bytes  # noqa: E402
from src.rag.readiness import (  # noqa: E402
    ReadinessAuditArtifact,
    audit_rag_readiness,
    load_source_group_manifest,
)
from src.rag.subject_catalog import SubjectCatalog  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--index-config", type=Path, required=True)
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--gold-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser


def _load_project_index_config(
    *, project_root: Path, index_config_path: Path
) -> RagIndexConfig:
    root = resolve_project_path(
        project_root=project_root,
        value=project_root,
        must_exist=True,
    )
    config_path = resolve_project_path(
        project_root=root,
        value=index_config_path,
        must_exist=True,
    )
    config = load_rag_index_config(config_path)
    resolve_project_path(
        project_root=root,
        value=config.catalog.data_root,
        must_exist=True,
    )
    index_root = resolve_project_path(
        project_root=root,
        value=config.storage.index_root,
        must_exist=False,
    )
    registry_path = (
        config.storage.registry_path
        if config.storage.registry_path.is_absolute()
        else index_root / config.storage.registry_path
    )
    resolve_project_path(
        project_root=root,
        value=registry_path,
        must_exist=False,
    )
    return resolve_rag_index_config_paths(config, project_root=root)


def run_audit(
    *,
    project_root: Path,
    index_config_path: Path,
    benchmark_config_path: Path,
    gold_dataset_path: Path,
    output_path: Path,
    overwrite: bool,
) -> ReadinessAuditArtifact:
    """Run and persist one strict read-only audit from the final GoldDataset."""

    root = resolve_project_path(
        project_root=project_root,
        value=project_root,
        must_exist=True,
    )
    config_path = resolve_project_path(
        project_root=root,
        value=index_config_path,
        must_exist=True,
    )
    benchmark_path = resolve_project_path(
        project_root=root,
        value=benchmark_config_path,
        must_exist=True,
    )
    dataset_path = resolve_project_path(
        project_root=root,
        value=gold_dataset_path,
        must_exist=True,
    )
    output = resolve_project_path(
        project_root=root,
        value=output_path,
        must_exist=False,
    )
    index_config = _load_project_index_config(
        project_root=root,
        index_config_path=config_path,
    )
    benchmark = load_rag_benchmark_config(benchmark_path)
    for configured_path in (
        *benchmark.human_gold_paths,
        *benchmark.historical_annotated_paths,
        *benchmark.synthetic_smoke_paths,
        benchmark.source_group_manifest_path,
    ):
        resolve_project_path(
            project_root=root,
            value=configured_path,
            must_exist=False,
        )
    source_groups_path = resolve_project_path(
        project_root=root,
        value=benchmark.source_group_manifest_path,
        must_exist=True,
    )
    snapshot = SubjectCatalog(
        config=index_config.catalog,
        subject_policy_map=index_config.subject_policy_map,
    ).discover()
    if set(benchmark.primary_subjects) != set(snapshot.subject_ids()):
        raise GoldDatasetAuthoringError(
            "benchmark primary_subjects must exactly match SubjectCatalog subjects"
        )
    dataset = load_gold_dataset(dataset_path)
    report = audit_rag_readiness(
        catalog_snapshot=snapshot,
        source_group_manifest=load_source_group_manifest(source_groups_path),
        query_records=gold_dataset_to_query_inventory(dataset),
        low_text_page_chars=benchmark.low_text_page_chars,
        minimum_independent_sources=benchmark.min_independent_sources,
        minimum_subject_gold_queries=benchmark.min_subject_gold_queries,
        minimum_global_gold_queries=benchmark.min_global_gold_queries,
    )
    artifact = ReadinessAuditArtifact(
        schema_version="rag_readiness_artifact_v2",
        index_config_path=project_relative_path(project_root=root, path=config_path),
        benchmark_config_path=project_relative_path(
            project_root=root, path=benchmark_path
        ),
        gold_dataset_path=project_relative_path(project_root=root, path=dataset_path),
        gold_dataset_sha256=sha256_file(dataset_path),
        data_root=project_relative_path(
            project_root=root,
            path=index_config.catalog.data_root,
        ),
        report=report,
    )
    atomic_write_project_bytes(
        root,
        output,
        model_json_bytes(artifact),
        overwrite=overwrite,
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        artifact = run_audit(
            project_root=args.project_root,
            index_config_path=args.index_config,
            benchmark_config_path=args.benchmark_config,
            gold_dataset_path=args.gold_dataset,
            output_path=args.output,
            overwrite=args.overwrite,
        )
    except (GoldDatasetAuthoringError, OSError, ValueError) as exc:
        print(f"RAG readiness audit failed: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(
        "RAG readiness audit written: "
        f"blocked={artifact.report.production_recommendation_blocked}, "
        f"subjects={len(artifact.report.subjects)}"
    )
    if args.fail_on_blocked and artifact.report.production_recommendation_blocked:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
