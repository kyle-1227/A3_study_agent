"""Migrate a verified Parent--Child artifact directory into primary revision 1+.

This command deliberately reads only the concrete artifact files needed by the
primary layout. It does not open the generation registry, inspect READY state,
or use a sealed manifest as a serving dependency.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import shutil
import sys

PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FROM_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FROM_SCRIPT))

from src.config.rag_index_config import (  # noqa: E402
    load_rag_index_config,
    resolve_rag_index_config_paths,
)
from src.rag.parent_child._storage_io import (  # noqa: E402
    resolve_under_root,
    validate_generation_id,
)
from src.rag.parent_child.manifests import SubjectManifest, read_strict_model  # noqa: E402
from src.rag.parent_child.primary_runtime import (  # noqa: E402
    PrimaryIndexWorkspace,
    primary_metadata_from_config,
    validate_primary_revision,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--index-config", type=Path, required=True)
    parser.add_argument("--source-artifact-identity", required=True)
    parser.add_argument("--build-id", required=True)
    return parser


def _contained_file(root: Path, value: Path) -> Path:
    path = value if value.is_absolute() else root / value
    if path.is_symlink():
        raise ValueError("index config must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(root):
        raise ValueError("index config must be a project-contained file")
    return resolved


def _copy_primary_artifact(source: Path, staging: Path, relative: str) -> None:
    source_path = resolve_under_root(source, relative, must_exist=True)
    target = resolve_under_root(staging, relative, must_exist=False)
    if source_path.is_symlink() or target.exists():
        raise ValueError("primary migration artifact path is unsafe")
    if source_path.is_dir():
        if any(item.is_symlink() for item in source_path.rglob("*")):
            raise ValueError("primary migration source contains a symlink")
        shutil.copytree(source_path, target)
    elif source_path.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
    else:
        raise ValueError("primary migration source artifact is invalid")


def migrate_primary(
    *,
    project_root: Path,
    index_config_path: Path,
    source_artifact_identity: str,
    build_id: str,
) -> int:
    root = project_root.resolve(strict=True)
    config = resolve_rag_index_config_paths(
        load_rag_index_config(_contained_file(root, index_config_path)),
        project_root=root,
    )
    source_id = validate_generation_id(source_artifact_identity)
    source = resolve_under_root(
        config.storage.index_root,
        source_id,
        must_exist=True,
    )
    if source.is_symlink() or not source.is_dir():
        raise ValueError("source Parent--Child artifact directory is invalid")
    subject_manifest = read_strict_model(
        source,
        "subject_manifest.json",
        SubjectManifest,
    )
    if subject_manifest.generation_id != source_id:
        raise ValueError("source subject manifest identity mismatch")
    subjects = tuple(
        entry.subject_id
        for entry in subject_manifest.entries
        if entry.exclusion_state == "active"
    )
    workspace = PrimaryIndexWorkspace.create(
        index_root=config.storage.index_root,
        build_id=build_id,
    )
    try:
        for relative in (
            "chroma_children",
            "parents.sqlite",
            "bm25",
            "policy_manifest.json",
            "subject_manifest.json",
        ):
            _copy_primary_artifact(source, workspace.staging_path, relative)
        metadata = primary_metadata_from_config(
            config,
            primary_revision=workspace.next_revision(),
            artifact_identity=source_id,
            available_subjects=subjects,
            built_at_utc=datetime.now(UTC),
        )
        result = workspace.publish(
            metadata=metadata,
            validate_staging=lambda artifact_root, primary_metadata: (
                validate_primary_revision(
                    config=config,
                    artifact_root=artifact_root,
                    metadata=primary_metadata,
                )
            ),
        )
    except BaseException:
        raise
    return result.primary_revision


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    revision = migrate_primary(
        project_root=args.project_root,
        index_config_path=args.index_config,
        source_artifact_identity=args.source_artifact_identity,
        build_id=args.build_id,
    )
    print(f"Parent--Child primary published: revision={revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
