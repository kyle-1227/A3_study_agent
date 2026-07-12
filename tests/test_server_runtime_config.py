from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import scripts.run_backend as run_backend
import src.config.server_runtime_config as server_runtime_config


def _config(*, enabled: bool) -> server_runtime_config.ServerReloadConfig:
    return server_runtime_config.ServerReloadConfig.model_validate(
        {
            "enabled": enabled,
            "include_dirs": ["."],
            "exclude_dirs": ["frontend", "tests", "artifacts"],
        }
    )


def test_reload_options_are_explicit_and_workspace_bounded(tmp_path: Path):
    (tmp_path / "frontend").mkdir()
    options = server_runtime_config.resolve_uvicorn_reload_options(
        _config(enabled=True),
        workspace_root=tmp_path,
    )

    assert options["reload"] is True
    assert options["reload_dirs"] == [str(tmp_path.resolve())]
    assert options["reload_excludes"] == [
        str((tmp_path / "frontend").resolve()),
        str((tmp_path / "tests").resolve()),
        str((tmp_path / "artifacts").resolve()),
    ]


def test_disabled_reload_does_not_resolve_or_watch_directories(tmp_path: Path):
    options = server_runtime_config.resolve_uvicorn_reload_options(
        _config(enabled=False),
        workspace_root=tmp_path,
    )

    assert options == {"reload": False}


@pytest.mark.parametrize("path", ["../outside", "/absolute", "C:\\outside"])
def test_reload_paths_reject_workspace_escape(path: str):
    with pytest.raises(ValidationError):
        server_runtime_config.ServerReloadConfig.model_validate(
            {
                "enabled": True,
                "include_dirs": [path],
                "exclude_dirs": ["tests"],
            }
        )


def test_reload_configuration_is_required(monkeypatch):
    monkeypatch.setattr(
        server_runtime_config, "get_setting", lambda _key, _default: None
    )

    with pytest.raises(RuntimeError, match="server.reload configuration is required"):
        server_runtime_config.load_server_reload_config()


def test_backend_launcher_uses_config_and_only_explicitly_overrides_reload(monkeypatch):
    captured: dict[str, object] = {}
    override_values: list[bool | None] = []

    monkeypatch.setattr(
        run_backend, "load_server_reload_config", lambda: _config(enabled=False)
    )

    def resolve_options(
        _config_value: server_runtime_config.ServerReloadConfig,
        *,
        workspace_root: Path,
        enabled_override: bool | None,
    ) -> dict[str, bool]:
        override_values.append(enabled_override)
        assert workspace_root.name == "A3_study_agent"
        return {"reload": bool(enabled_override)}

    monkeypatch.setattr(run_backend, "resolve_uvicorn_reload_options", resolve_options)
    monkeypatch.setattr(
        run_backend.uvicorn,
        "run",
        lambda *args, **kwargs: captured.update({"args": args, "kwargs": kwargs}),
    )

    run_backend.main(["--host", "127.0.0.1", "--port", "8123"])
    run_backend.main(["--reload"])

    assert override_values == [None, True]
    assert captured["args"] == ("app:app",)
    assert captured["kwargs"]["reload"] is True
