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
    expected_workspace_root = Path(run_backend.__file__).resolve().parent.parent

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
        assert workspace_root == expected_workspace_root
        assert (workspace_root / "pyproject.toml").is_file()
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


def _runtime_state_config(
    directory: str = ".runtime_state",
) -> server_runtime_config.ServerRuntimeStateConfig:
    return server_runtime_config.ServerRuntimeStateConfig.model_validate(
        {"directory": directory}
    )


def test_runtime_state_paths_use_dedicated_workspace_directory(tmp_path: Path):
    paths = server_runtime_config.resolve_server_runtime_state_paths(
        _runtime_state_config(),
        workspace_root=tmp_path,
    )

    assert paths.directory == (tmp_path / ".runtime_state").resolve()
    assert paths.profile_db_path == paths.directory / "profile.db"
    assert paths.memory_db_path == paths.directory / "memory.db"


@pytest.mark.parametrize(
    "directory",
    [".", "./", "../outside", "/absolute", "C:\\outside", " state "],
)
def test_runtime_state_config_rejects_unsafe_directory(directory: str):
    with pytest.raises(ValidationError):
        _runtime_state_config(directory)


@pytest.mark.parametrize("directory", ["data", "data/runtime_state"])
def test_runtime_state_paths_reject_immutable_course_data(
    tmp_path: Path,
    directory: str,
):
    with pytest.raises(
        RuntimeError,
        match="runtime state directory must remain outside immutable course data",
    ):
        server_runtime_config.resolve_server_runtime_state_paths(
            _runtime_state_config(directory),
            workspace_root=tmp_path,
        )


def test_runtime_state_configuration_is_required(monkeypatch):
    monkeypatch.setattr(
        server_runtime_config,
        "get_setting",
        lambda _key, _default: None,
    )

    with pytest.raises(
        RuntimeError,
        match="server.runtime_state configuration is required",
    ):
        server_runtime_config.load_server_runtime_state_config()
