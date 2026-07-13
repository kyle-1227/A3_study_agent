"""Dependency-injected CLI surface for checkpoint migration releases."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from src.checkpoint_migration.contracts import (
    CheckpointMigrationBatchResultV1,
    CheckpointMigrationSpecificationV1,
)
from src.checkpoint_migration.service import (
    CheckpointMigrationCheckpointer,
    CheckpointMigrationGraph,
    CheckpointMigrationProjector,
    CheckpointMigrationValidators,
    migrate_checkpoints,
)

MAX_SPECIFICATION_BYTES = 256 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or apply one approved checkpoint migration specification. "
            "Dry-run is the default; writes require --apply."
        )
    )
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the fully validated batch. Omit for a dry-run.",
    )
    return parser


def run_checkpoint_migration_cli(
    *,
    argv: Sequence[str],
    checkpointer: CheckpointMigrationCheckpointer,
    graph: CheckpointMigrationGraph,
    projector: CheckpointMigrationProjector,
    validators: CheckpointMigrationValidators,
) -> CheckpointMigrationBatchResultV1:
    """Run the CLI with production dependencies supplied by an approved adapter."""
    args = build_parser().parse_args(list(argv))
    specification = load_specification(args.specification)
    result = asyncio.run(
        migrate_checkpoints(
            specification=specification,
            checkpointer=checkpointer,
            graph=graph,
            projector=projector,
            validators=validators,
            mode="apply" if args.apply else "dry_run",
        )
    )
    print(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    )
    return result


def load_specification(path: Path) -> CheckpointMigrationSpecificationV1:
    """Load a bounded, non-symlink JSON specification with strict validation."""
    if path.is_symlink():
        raise ValueError("checkpoint migration specification must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("checkpoint migration specification must be a file")
    if resolved.stat().st_size > MAX_SPECIFICATION_BYTES:
        raise ValueError("checkpoint migration specification exceeds size limit")
    return CheckpointMigrationSpecificationV1.model_validate_json(
        resolved.read_text(encoding="utf-8"),
        strict=True,
    )


__all__ = [
    "MAX_SPECIFICATION_BYTES",
    "build_parser",
    "load_specification",
    "run_checkpoint_migration_cli",
]
