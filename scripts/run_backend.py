"""Start the FastAPI application with the PostgreSQL-compatible event loop."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from src.config.server_runtime_config import (
    load_server_reload_config,
    resolve_uvicorn_reload_options,
)
from src.database.event_loop import postgres_compatible_event_loop


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    reload_group = parser.add_mutually_exclusive_group()
    reload_group.add_argument("--reload", dest="reload", action="store_true")
    reload_group.add_argument("--no-reload", dest="reload", action="store_false")
    parser.set_defaults(reload=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    reload_config = load_server_reload_config()
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        loop=postgres_compatible_event_loop,
        **resolve_uvicorn_reload_options(
            reload_config,
            workspace_root=Path(__file__).resolve().parent.parent,
            enabled_override=args.reload,
        ),
    )


if __name__ == "__main__":
    main()
