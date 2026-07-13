"""Generate one ignored, project-relative local RAG runtime configuration.

The source configuration remains the explicit, strict template for all provider
and chunk-policy contracts. This helper replaces only local filesystem paths so
a checkout can move without retaining another machine's absolute paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FROM_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FROM_SCRIPT))

from scripts.init_rag_index_config import (  # noqa: E402
    portable_runtime_config_from_source,
    write_portable_runtime_config,
)
from src.rag.parent_child.project_paths import (  # noqa: E402
    require_project_file,
    resolve_project_path,
    resolve_project_root,
)


class RuntimeConfigInitializationError(ValueError):
    """A runtime config cannot be generated from the explicit local template."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--registry-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run_init(args: argparse.Namespace) -> Path:
    """Generate the runtime config described by one parsed CLI namespace."""

    root = resolve_project_root(args.project_root)
    source = require_project_file(root, args.source_config)
    output = resolve_project_path(root, args.output, must_exist=False)
    if source == output:
        raise RuntimeConfigInitializationError(
            "runtime config output must not replace its source template"
        )
    config = portable_runtime_config_from_source(
        project_root=root,
        source_config_path=source,
        data_root=args.data_root,
        index_root=args.index_root,
        registry_path=args.registry_path,
    )
    return write_portable_runtime_config(
        project_root=root,
        output_path=args.output,
        config=config,
        overwrite=args.overwrite,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = run_init(args)
    print(f"Project-relative RAG runtime config generated: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
