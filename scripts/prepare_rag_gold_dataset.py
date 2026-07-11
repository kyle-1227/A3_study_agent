"""Create, inspect, validate, and export local policy-independent RAG gold data.

This script is intentionally local-only.  It never builds an index, opens a
generation registry, contacts a provider, or synthesizes gold evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

project_root_from_script = Path(__file__).resolve().parent.parent
if str(project_root_from_script) not in sys.path:
    sys.path.insert(0, str(project_root_from_script))

from src.config.rag_index_config import (  # noqa: E402
    RagIndexConfig,
    load_rag_index_config,
    resolve_rag_index_config_paths,
)
from src.rag.gold_dataset import (  # noqa: E402
    GoldDatasetAuthoringError,
    GoldDatasetDraft,
    load_gold_dataset,
    load_gold_dataset_draft_or_final,
    inspect_gold_source,
    resolve_project_path,
    validate_gold_dataset,
    write_gold_dataset_jsonl_exports,
    write_gold_model,
)
from src.rag.parent_child.evaluation import GoldDataset  # noqa: E402
from src.rag.readiness import load_source_group_manifest  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="create an empty local GoldDataset draft")
    init.add_argument("--project-root", type=Path, required=True)
    init.add_argument("--dataset-id", required=True)
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--overwrite", action="store_true")

    inspect_source = commands.add_parser(
        "inspect-source",
        help="write a cleaned, page-aware local source inspection artifact",
    )
    inspect_source.add_argument("--project-root", type=Path, required=True)
    inspect_source.add_argument("--index-config", type=Path, required=True)
    inspect_source.add_argument("--source-relpath", required=True)
    inspect_source.add_argument("--output", type=Path, required=True)
    inspect_source.add_argument("--overwrite", action="store_true")

    validate = commands.add_parser(
        "validate",
        help="prove draft/final Gold spans and seal a final GoldDataset",
    )
    validate.add_argument("--project-root", type=Path, required=True)
    validate.add_argument("--index-config", type=Path, required=True)
    validate.add_argument("--source-groups", type=Path, required=True)
    validate.add_argument("--input", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--overwrite", action="store_true")

    export = commands.add_parser(
        "export-readiness-jsonl",
        help="derive deterministic readiness inventories from a final GoldDataset",
    )
    export.add_argument("--project-root", type=Path, required=True)
    export.add_argument("--gold-dataset", type=Path, required=True)
    export.add_argument("--human-output", type=Path, required=True)
    export.add_argument("--historical-output", type=Path, required=True)
    export.add_argument("--synthetic-output", type=Path, required=True)
    export.add_argument("--overwrite", action="store_true")
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


def _seal_final_dataset(
    source: GoldDatasetDraft | GoldDataset,
) -> GoldDataset:
    if isinstance(source, GoldDataset):
        return source
    try:
        return GoldDataset(
            schema_version="gold_dataset_v1",
            dataset_id=source.dataset_id,
            queries=source.queries,
        )
    except Exception as exc:
        raise GoldDatasetAuthoringError(
            "GoldDataset draft cannot be sealed without at least one valid query"
        ) from exc


def run_init(
    *, project_root: Path, dataset_id: str, output_path: Path, overwrite: bool
) -> Path:
    draft = GoldDatasetDraft(
        schema_version="gold_dataset_draft_v1",
        dataset_id=dataset_id,
        queries=(),
    )
    return write_gold_model(
        project_root=project_root,
        output_path=output_path,
        model=draft,
        overwrite=overwrite,
    )


def run_inspect_source(
    *,
    project_root: Path,
    index_config_path: Path,
    source_relpath: str,
    output_path: Path,
    overwrite: bool,
) -> Path:
    config = _load_project_index_config(
        project_root=project_root,
        index_config_path=index_config_path,
    )
    inspection = inspect_gold_source(
        index_config=config,
        source_relpath=source_relpath,
    )
    return write_gold_model(
        project_root=project_root,
        output_path=output_path,
        model=inspection,
        overwrite=overwrite,
    )


def run_validate(
    *,
    project_root: Path,
    index_config_path: Path,
    source_groups_path: Path,
    input_path: Path,
    output_path: Path,
    overwrite: bool,
) -> Path:
    root = resolve_project_path(
        project_root=project_root,
        value=project_root,
        must_exist=True,
    )
    config = _load_project_index_config(
        project_root=root,
        index_config_path=index_config_path,
    )
    groups_path = resolve_project_path(
        project_root=root,
        value=source_groups_path,
        must_exist=True,
    )
    draft_path = resolve_project_path(
        project_root=root,
        value=input_path,
        must_exist=True,
    )
    source_groups = load_source_group_manifest(groups_path)
    dataset = _seal_final_dataset(load_gold_dataset_draft_or_final(draft_path))
    validated = validate_gold_dataset(
        dataset=dataset,
        index_config=config,
        source_groups=source_groups,
    )
    return write_gold_model(
        project_root=root,
        output_path=output_path,
        model=validated,
        overwrite=overwrite,
    )


def run_export_readiness_jsonl(
    *,
    project_root: Path,
    gold_dataset_path: Path,
    human_output: Path,
    historical_output: Path,
    synthetic_output: Path,
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    root = resolve_project_path(
        project_root=project_root,
        value=project_root,
        must_exist=True,
    )
    dataset_path = resolve_project_path(
        project_root=root,
        value=gold_dataset_path,
        must_exist=True,
    )
    return write_gold_dataset_jsonl_exports(
        project_root=root,
        dataset=load_gold_dataset(dataset_path),
        human_output=human_output,
        historical_output=historical_output,
        synthetic_output=synthetic_output,
        overwrite=overwrite,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            output = run_init(
                project_root=args.project_root,
                dataset_id=args.dataset_id,
                output_path=args.output,
                overwrite=args.overwrite,
            )
            print(f"GoldDataset draft written: {output}")
            return 0
        if args.command == "inspect-source":
            output = run_inspect_source(
                project_root=args.project_root,
                index_config_path=args.index_config,
                source_relpath=args.source_relpath,
                output_path=args.output,
                overwrite=args.overwrite,
            )
            print(f"Gold source inspection written: {output}")
            return 0
        if args.command == "validate":
            output = run_validate(
                project_root=args.project_root,
                index_config_path=args.index_config,
                source_groups_path=args.source_groups,
                input_path=args.input,
                output_path=args.output,
                overwrite=args.overwrite,
            )
            print(f"Validated GoldDataset written: {output}")
            return 0
        if args.command == "export-readiness-jsonl":
            outputs = run_export_readiness_jsonl(
                project_root=args.project_root,
                gold_dataset_path=args.gold_dataset,
                human_output=args.human_output,
                historical_output=args.historical_output,
                synthetic_output=args.synthetic_output,
                overwrite=args.overwrite,
            )
            print("Gold readiness inventories written: " + ", ".join(map(str, outputs)))
            return 0
    except (GoldDatasetAuthoringError, OSError, ValueError) as exc:
        print(f"GoldDataset preparation failed: {type(exc).__name__}", file=sys.stderr)
        return 2
    raise AssertionError("argparse accepted an unsupported command")


if __name__ == "__main__":
    raise SystemExit(main())
