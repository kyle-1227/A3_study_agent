"""Start the FastAPI application with the PostgreSQL-compatible event loop."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn

from src.database.event_loop import postgres_compatible_event_loop


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        loop=postgres_compatible_event_loop,
    )


if __name__ == "__main__":
    main()
