from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_rag_local_build as local_build
from src.rag.parent_child.embedding_batches import (
    EmbeddingBatchExecutionError,
    iter_bounded_document_embedding_batches,
)
from src.rag.parent_child.flat_baseline import FlatBaselineError
from src.rag.parent_child.provider_clients import (
    ProviderProtocolError,
    ProviderReportedError,
    ProviderTransportError,
)


def test_embedding_rows_accept_exact_list_and_numpy_contracts() -> None:
    list_rows = [[1.0, 2.0], [3.0, 4.0]]
    array_rows = np.asarray(list_rows, dtype=np.float32)

    assert local_build._validated_embedding_rows(
        list_rows,
        expected_count=2,
        failure_code="FlatChromaSampleShapeInvalid",
    ) == tuple(list_rows)
    assert local_build._validated_embedding_rows(
        array_rows,
        expected_count=2,
        failure_code="GenerationChromaVectorSampleInvalid",
    ) == tuple(list_rows)


@pytest.mark.parametrize(
    "value,expected_count",
    (
        (np.asarray([1.0, 2.0], dtype=np.float32), 1),
        (np.asarray([[1.0, 2.0]], dtype=np.float32), 2),
        (([1.0, 2.0],), 1),
    ),
)
def test_embedding_rows_reject_undeclared_shape_contracts(
    value: object, expected_count: int
) -> None:
    with pytest.raises(
        local_build.LocalBuildError,
        match="FlatChromaSampleShapeInvalid",
    ):
        local_build._validated_embedding_rows(
            value,
            expected_count=expected_count,
            failure_code="FlatChromaSampleShapeInvalid",
        )


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
        "--no-embedding-cache",
        "--embedding-cache-busy-timeout-seconds",
        "2",
        *mode,
    ]


def _planned_report(inputs: local_build.BuildInputs) -> local_build.LocalBuildReport:
    return local_build.LocalBuildReport(
        schema_version="rag_local_build_report_v2",
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


def test_markdown_report_marks_unreached_build_artifacts_as_not_run(
    tmp_path: Path,
) -> None:
    """A failed provider stage must not be described as a completed local build."""

    report_directory = tmp_path / "reports" / "rag_build" / "rag_test_run"
    report_directory.mkdir(parents=True)
    context = SimpleNamespace(root=tmp_path, report_directory=report_directory)
    local_build._write_model(
        tmp_path,
        report_directory / "smoke_retrieval.json",
        local_build.SmokeRetrievalArtifact(
            schema_version="rag_smoke_retrieval_v1",
            status="not_run",
            reason="provider_probe_failed",
            generation_id=None,
            records=(),
        ),
    )
    local_build._write_model(
        tmp_path,
        report_directory / "llm_grounded_smoke.json",
        local_build.GroundedSmokeArtifact(
            schema_version="rag_llm_grounded_smoke_v1",
            status="not_run",
            reason="provider_probe_failed",
            records=(),
            private_output_written=False,
        ),
    )
    report = local_build.LocalBuildReport(
        schema_version="rag_local_build_report_v2",
        run_id="rag_test_run",
        mode="execute",
        status="failed",
        requested_code_revision="a" * 40,
        head_code_revision="a" * 40,
        revision_matches_head=True,
        runtime_config_path="config/rag/index.runtime.yaml",
        gold_dataset_path="data/evaluation/gold_dataset_v2.json",
        catalog=None,
        readiness=None,
        secrets=None,
        flat_baseline=None,
        generation=None,
        smoke_retrieval_path="reports/rag_build/rag_test_run/smoke_retrieval.json",
        grounded_smoke_path="reports/rag_build/rag_test_run/llm_grounded_smoke.json",
        stages=(
            local_build.StageRecord(
                stage="provider_probe",
                status="failed",
                duration_ms=1.0,
                failure_type="ProviderProbeFailed",
            ),
            local_build.StageRecord(
                stage="chunk_dry_run",
                status="not_run",
                duration_ms=0.0,
                failure_type=None,
            ),
        ),
        failure=local_build.FailureSummary(
            stage="provider_probe",
            error_type="ProviderProbeFailed",
            cause_chain=(
                local_build.FailureDiagnostic(
                    error_type="ProviderProbeFailed",
                    batch_start=None,
                    batch_size=None,
                    provider_code=None,
                    retryable=None,
                    attempts_exhausted=None,
                ),
            ),
        ),
        experimental_only=True,
        activation_prohibited=True,
    )

    markdown = local_build._render_build_markdown(context, report)

    assert "Provider probe: `not run` (stage=failed)" in markdown
    assert "Chunk dry run: `not run` (stage=not_run)" in markdown
    assert "## Flat baseline" in markdown
    assert "no local Chroma count" in markdown
    assert "## Parent-child generation" in markdown
    assert "no registry row, READY state" in markdown
    assert "Retrieval outcome: status=`not_run`" in markdown
    assert "Grounded LLM outcome: status=`not_run`" in markdown
    assert "Activation allowed: `false`" in markdown


def _raise_flat_embedding_failure(provider_error: BaseException) -> None:
    def fail(_texts: list[str]) -> list[list[float]]:
        raise provider_error

    try:
        list(
            iter_bounded_document_embedding_batches(
                texts=("one", "two"),
                batch_size=2,
                max_in_flight_batches=1,
                embed_documents=fail,
            )
        )
    except EmbeddingBatchExecutionError as exc:
        raise FlatBaselineError("flat baseline embedding provider failed") from exc
    raise AssertionError("the strict embedding provider failure was not propagated")


@pytest.mark.parametrize(
    "provider_error",
    (
        ProviderReportedError(
            code=503,
            retryable=True,
            attempts_exhausted=True,
        ),
        ProviderTransportError("transport secret must not be serialized"),
        ProviderProtocolError("protocol secret must not be serialized"),
    ),
    ids=("reported", "transport", "protocol"),
)
def test_failure_diagnostic_exposes_bounded_safe_cause_chain(
    provider_error: BaseException,
) -> None:
    secret = "sensitive-token-value"
    provider_error.response_body = {"Authorization": f"Bearer {secret}"}
    provider_error.request_url = f"https://provider.invalid/v1?api_key={secret}"

    with pytest.raises(FlatBaselineError) as captured:
        _raise_flat_embedding_failure(provider_error)

    chain = local_build._safe_failure_cause_chain(captured.value)
    payload = local_build.FailureSummary(
        stage="flat_baseline",
        error_type="FlatBaselineError",
        cause_chain=chain,
    ).model_dump_json()

    assert tuple(item.error_type for item in chain) == (
        "FlatBaselineError",
        "EmbeddingBatchExecutionError",
        type(provider_error).__name__,
    )
    assert len(chain) <= local_build._MAX_FAILURE_CAUSE_DEPTH
    assert chain[1].batch_start == 0
    assert chain[1].batch_size == 2
    if isinstance(provider_error, ProviderReportedError):
        assert chain[2].provider_code == 503
        assert chain[2].retryable is True
        assert chain[2].attempts_exhausted is True
    else:
        assert chain[2].provider_code is None
        assert chain[2].retryable is None
        assert chain[2].attempts_exhausted is None
    batch_failure = captured.value.__cause__
    assert isinstance(batch_failure, EmbeddingBatchExecutionError)
    assert batch_failure.__cause__ is None
    assert batch_failure.__context__ is None
    assert secret not in payload
    assert "provider.invalid" not in payload
    assert "Authorization" not in payload
    assert "transport secret" not in payload
    assert "protocol secret" not in payload


def test_failure_diagnostic_truncates_deep_exception_chains() -> None:
    failure: BaseException = RuntimeError("sensitive root text")
    for _ in range(local_build._MAX_FAILURE_CAUSE_DEPTH + 4):
        outer = RuntimeError("sensitive outer text")
        outer.__cause__ = failure
        failure = outer

    chain = local_build._safe_failure_cause_chain(failure)
    payload = local_build.FailureSummary(
        stage="flat_baseline",
        error_type="RuntimeError",
        cause_chain=chain,
    ).model_dump_json()

    assert len(chain) == local_build._MAX_FAILURE_CAUSE_DEPTH
    assert "sensitive root text" not in payload
    assert "sensitive outer text" not in payload
