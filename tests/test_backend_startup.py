from __future__ import annotations

import asyncio
from pathlib import Path

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

    workspace_root = Path(run_backend.__file__).resolve().parent.parent
    assert captured["app"] == "app:app"
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9000
    assert captured["reload"] is True
    assert captured["loop"] is event_loop_module.postgres_compatible_event_loop
    assert captured["reload_dirs"] == [str(workspace_root)]
    assert str(workspace_root / "frontend") in captured["reload_excludes"]
    assert str(workspace_root / "tests") in captured["reload_excludes"]
