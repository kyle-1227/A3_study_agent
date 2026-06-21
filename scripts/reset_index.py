"""Safely reset generated RAG index artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

try:
    from src.rag.indexer import _resolve_persist_dir
except ModuleNotFoundError:
    _resolve_persist_dir = None


def resolve_chroma_persist_dir(
    persist_directory: str | Path | None = None,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> Path:
    """Resolve the configured Chroma persist directory."""

    root = Path(project_root).resolve()
    if persist_directory is not None:
        path = Path(persist_directory)
    elif root == PROJECT_ROOT.resolve() and _resolve_persist_dir is not None:
        return Path(_resolve_persist_dir()).resolve()
    else:
        path = Path(os.getenv("CHROMA_PERSIST_DIR") or "chroma_store")

    if not path.is_absolute():
        path = root / path
    return path.resolve()


def reset_targets(
    *,
    project_root: str | Path = PROJECT_ROOT,
    persist_directory: str | Path | None = None,
) -> list[Path]:
    """Return the only paths this script is allowed to remove."""

    root = Path(project_root).resolve()
    targets = [
        resolve_chroma_persist_dir(persist_directory, project_root=root),
        root / "reports" / "build_manifest.json",
        root / "reports" / "parent_chunks.jsonl",
    ]
    unique_targets: list[Path] = []
    for target in targets:
        resolved = target.resolve()
        if resolved not in unique_targets:
            unique_targets.append(resolved)
    return unique_targets


def _is_root_path(path: Path) -> bool:
    anchor = path.anchor
    return bool(anchor) and path == Path(anchor).resolve()


def _assert_safe_delete_path(
    path: Path,
    *,
    project_root: Path,
    allowed_targets: set[Path],
) -> None:
    resolved = path.resolve()
    root = project_root.resolve()
    disallowed = {
        root,
        root / "data",
        root / ".env",
        root / "reports",
        Path.home().resolve(),
    }
    if str(resolved).strip() == "":
        raise ValueError("Refusing to delete an empty path")
    if _is_root_path(resolved):
        raise ValueError(f"Refusing to delete filesystem root: {resolved}")
    if resolved != root and root.is_relative_to(resolved):
        raise ValueError(f"Refusing to delete project ancestor: {resolved}")
    if resolved in disallowed:
        raise ValueError(f"Refusing to delete protected path: {resolved}")
    if resolved not in allowed_targets:
        raise ValueError(f"Refusing to delete unapproved path: {resolved}")


def validate_reset_targets(
    targets: list[Path],
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> None:
    """Validate every reset target before any deletion occurs."""

    root = Path(project_root).resolve()
    allowed_targets = {target.resolve() for target in targets}
    for target in targets:
        _assert_safe_delete_path(
            target, project_root=root, allowed_targets=allowed_targets
        )


def remove_targets(
    targets: list[Path],
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> list[str]:
    """Remove reset targets and return user-facing status lines."""

    validate_reset_targets(targets, project_root=project_root)
    messages: list[str] = []
    for target in targets:
        if not target.exists():
            messages.append(f"Skipped missing: {target}")
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        messages.append(f"Removed: {target}")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset generated RAG index artifacts.")
    parser.add_argument(
        "--yes", action="store_true", help="Actually delete generated index artifacts."
    )
    args = parser.parse_args()

    targets = reset_targets()
    validate_reset_targets(targets)

    print("=== reset_index ===")
    print(f"Chroma persist dir: {targets[0]}")
    print("Will remove:")
    for target in targets:
        print(f"- {target}")
    print()

    if not args.yes:
        print("Refusing to delete without --yes.")
        return

    for message in remove_targets(targets):
        print(message)


if __name__ == "__main__":
    main()
