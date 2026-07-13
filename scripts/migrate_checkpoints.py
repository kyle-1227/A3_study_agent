"""Checkpoint migration CLI entrypoint awaiting an approved production adapter.

The reusable CLI is intentionally dependency-injected. A production release must
wire approved checkpointer, graph updater, projector, and schema validators before
this script can apply data; this skeleton never reaches into checkpoint storage.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FROM_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FROM_SCRIPT))

from src.checkpoint_migration.cli import build_parser  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.error(
        "production checkpoint migration adapter is not wired; "
        "inject dependencies through run_checkpoint_migration_cli"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
