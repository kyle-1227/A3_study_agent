from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_rag_local_build as local_build


def _arguments(tmp_path: Path, *mode: str) -> list[str]:
    return [
        "--project-root",
        str(tmp_path),
        "--index-config",
        "config/rag/index.runtime.yaml",
        "--benchmark-config",
        "config/rag/benchmark.yaml",
        "--gold-dataset",
        "data/evaluation/gold_dataset_v2.json",
        "--build-id",
        "flat_test_build",
        "--generation-id",
        "pc_test_generation",
        "--code-revision",
        "a" * 40,
        "--run-id",
        "rag_test_run",
        *mode,
    ]


def _planned_report(inputs: local_build.BuildInputs) -> local_build.LocalBuildReport:
    return local_build.LocalBuildReport(
        schema_version="rag_local_build_report_v1",
        run_id=inputs.run_id,
        mode=inputs.mode,
        status="planned",
        requested_code_revision=inputs.code_revision,
        head_code_revision=None,
        revision_matches_head=None,
        runtime_config_path="config/rag/index.runtime.yaml",
        gold_dataset_path="data/evaluation/gold_dataset_v2.json",
        catalog=None,
        readiness=None,
        secrets=None,
        flat_baseline=None,
        generation=None,
        smoke_retrieval_path="reports/rag_build/rag_test_run/smoke_retrieval.json",
        grounded_smoke_path="reports/rag_build/rag_test_run/llm_grounded_smoke.json",
        stages=(),
        failure=None,
        experimental_only=True,
        activation_prohibited=True,
    )


def test_parser_requires_every_explicit_build_input() -> None:
    with pytest.raises(SystemExit):
        local_build._parser().parse_args([])

    with pytest.raises(SystemExit):
        local_build._parser().parse_args(
            [
                "--project-root",
                ".",
                "--index-config",
                "config/rag/index.runtime.yaml",
                "--gold-dataset",
                "data/evaluation/gold_dataset_v2.json",
                "--build-id",
                "flat_test",
                "--generation-id",
                "pc_test",
                "--code-revision",
                "a" * 40,
            ]
        )


def test_plan_mode_writes_no_report_or_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = local_build.main(_arguments(tmp_path))

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "planned"
    assert payload["head_code_revision"] is None
    assert payload["revision_matches_head"] is None
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "artifacts").exists()


def test_execute_stops_after_provider_probe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = local_build._build_inputs(
        local_build._parser().parse_args(_arguments(tmp_path, "--execute"))
    )
    context = SimpleNamespace(
        inputs=inputs,
        root=tmp_path,
        report_directory=tmp_path / "reports" / "rag_build" / inputs.run_id,
    )
    calls: list[str] = []

    monkeypatch.setattr(
        local_build,
        "_preflight_report",
        lambda _context: SimpleNamespace(
            dependencies=local_build.DependencyReport(
                schema_version="rag_build_dependency_report_v1",
                required_modules=("chromadb",),
                missing_modules=(),
            )
        ),
    )
    monkeypatch.setattr(local_build, "_write_model", lambda *_args: tmp_path / "report")
    monkeypatch.setattr(local_build, "_validate_new_targets", lambda _context: None)
    monkeypatch.setattr(
        local_build,
        "_placeholder_smoke_artifacts",
        lambda *_args, **_kwargs: (tmp_path / "smoke.json", tmp_path / "grounded.json"),
    )
    monkeypatch.setattr(
        local_build,
        "_catalog_summary",
        lambda _context: (
            local_build.CatalogSummary(
                schema_version="rag_build_catalog_summary_v1",
                subjects=(
                    local_build.CatalogSubjectSummary(
                        subject_id="subject", source_file_count=1
                    ),
                ),
            ),
            object(),
        ),
    )
    monkeypatch.setattr(
        local_build,
        "_readiness_summary",
        lambda _context: local_build.ReadinessSummary(
            schema_version="rag_build_readiness_summary_v1",
            audit_completed=True,
            evaluation_eligible=False,
            source_group_complete=True,
            global_blockers=("gold_not_ready",),
            audit_failure_type=None,
        ),
    )

    def fail_provider(_context: object) -> object:
        calls.append("provider")
        raise local_build.ProviderProbeFailed("provider")

    monkeypatch.setattr(local_build, "_run_provider_probe", fail_provider)
    monkeypatch.setattr(
        local_build,
        "_run_chunk_dry_run",
        lambda _context: calls.append("chunk"),
    )
    monkeypatch.setattr(
        local_build,
        "_failure_report",
        lambda **_kwargs: _planned_report(inputs),
    )
    monkeypatch.setattr(local_build, "_write_final_report", lambda *_args: None)

    report, exit_code = local_build._run_execute(context)

    assert exit_code == 1
    assert report.status == "planned"
    assert calls == ["provider"]


def test_llm_configuration_has_no_environment_or_provider_default(
    tmp_path: Path,
) -> None:
    inputs = local_build._build_inputs(
        local_build._parser().parse_args(_arguments(tmp_path, "--execute"))
    )
    context = SimpleNamespace(inputs=inputs)

    assert local_build._llm_probe_config(context) is None


def test_dependency_preflight_fails_without_selecting_an_alternative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_build,
        "find_spec",
        lambda module_name: None if module_name == "chromadb" else object(),
    )

    report = local_build._dependency_report()

    assert report.missing_modules == ("chromadb",)
    with pytest.raises(local_build.LocalBuildError, match="Dependency"):
        local_build._require_build_dependencies(report)
