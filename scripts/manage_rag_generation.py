"""Explicit control-plane operations for READY parent-child generations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FROM_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FROM_SCRIPT))

from src.config.rag_index_config import (  # noqa: E402
    load_rag_index_config,
    resolve_rag_index_config_paths,
)
from src.rag.parent_child.registry import GenerationRegistry  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--index-config", type=Path, required=True)
    parser.add_argument(
        "--operation",
        choices=("activate", "rollback", "set-shadow", "clear-shadow", "cleanup"),
        required=True,
    )
    parser.add_argument("--generation-id")
    return parser


def _contained_config(project_root: Path, value: Path) -> Path:
    candidate = value if value.is_absolute() else project_root / value
    if candidate.is_symlink():
        raise ValueError("index config must not be a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(project_root) or not resolved.is_file():
        raise ValueError("index config must be a file inside project_root")
    return resolved


def _required_generation_id(value: str | None) -> str:
    if value is None:
        raise ValueError("generation-id is required for this operation")
    return value


def run_operation(
    *,
    project_root: Path,
    index_config_path: Path,
    operation: str,
    generation_id: str | None,
) -> str:
    root = project_root.resolve(strict=True)
    config = resolve_rag_index_config_paths(
        load_rag_index_config(_contained_config(root, index_config_path)),
        project_root=root,
    )
    needs_generation = operation in {"activate", "set-shadow", "cleanup"}
    if needs_generation != (generation_id is not None):
        raise ValueError(
            "generation-id is required exactly for activate, set-shadow, and cleanup"
        )
    with GenerationRegistry.open(
        config.storage.resolved_registry_path(),
        index_root=config.storage.index_root,
        expected_schema_version=config.storage.registry_schema_version,
        marker_schema_version=config.storage.owner_marker_schema_version,
        busy_timeout_seconds=config.storage.registry_busy_timeout_seconds,
    ) as registry:
        if operation == "activate":
            record = registry.activate(_required_generation_id(generation_id))
            return (
                f"activated generation={record.generation_id}, "
                f"revision={record.revision}"
            )
        if operation == "rollback":
            record = registry.rollback()
            return (
                f"rolled back generation={record.generation_id}, "
                f"revision={record.revision}"
            )
        if operation == "set-shadow":
            snapshot = registry.set_shadow(_required_generation_id(generation_id))
            return f"shadow generation={snapshot.shadow_generation_id}"
        if operation == "clear-shadow":
            snapshot = registry.set_shadow(None)
            return f"shadow generation={snapshot.shadow_generation_id}"
        if operation == "cleanup":
            target = _required_generation_id(generation_id)
            registry.cleanup_generation(target)
            return f"cleaned generation={target}"
    raise ValueError("unsupported generation operation")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(
        run_operation(
            project_root=args.project_root,
            index_config_path=args.index_config,
            operation=args.operation,
            generation_id=args.generation_id,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
