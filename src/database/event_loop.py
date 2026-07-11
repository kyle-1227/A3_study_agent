"""Event-loop factory compatible with psycopg asynchronous connections."""

from __future__ import annotations

import asyncio
import os


def postgres_compatible_event_loop() -> asyncio.AbstractEventLoop:
    """Return a Selector loop on Windows and the platform default elsewhere."""
    if os.name == "nt":
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()
