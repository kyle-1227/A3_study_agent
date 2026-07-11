from __future__ import annotations

import asyncio

from scripts import run_backend
from src.database import event_loop as event_loop_module


def test_windows_backend_loop_is_psycopg_compatible(monkeypatch):
    monkeypatch.setattr(event_loop_module.os, "name", "nt")

    loop = event_loop_module.postgres_compatible_event_loop()
    try:
        assert isinstance(loop, asyncio.SelectorEventLoop)
    finally:
        loop.close()


def test_backend_launcher_passes_explicit_loop_factory(monkeypatch):
    captured = {}

    def fake_run(app, **kwargs):
        captured.update({"app": app, **kwargs})

    monkeypatch.setattr(run_backend.uvicorn, "run", fake_run)

    run_backend.main(["--host", "0.0.0.0", "--port", "9000", "--reload"])

    assert captured == {
        "app": "app:app",
        "host": "0.0.0.0",
        "port": 9000,
        "reload": True,
        "loop": event_loop_module.postgres_compatible_event_loop,
    }
