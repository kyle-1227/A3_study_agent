from __future__ import annotations

import pytest

from scripts.build_index import _parser, _validate_subjects


def test_flat_baseline_cli_has_no_implicit_build_arguments() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args([])

    args = _parser().parse_args(
        [
            "--pipeline",
            "flat-baseline",
            "--data-root",
            "data",
            "--persist-dir",
            "artifacts/baseline-index",
            "--manifest-output",
            "artifacts/baseline-manifest.json",
            "--subject",
            "math",
            "--doc-type",
            "course_material",
            "--embedding-model",
            "configured-model",
            "--embedding-base-url",
            "https://provider.invalid/v1",
            "--embedding-api-key-env",
            "TEST_EMBEDDING_KEY",
            "--embedding-timeout-seconds",
            "10",
            "--embedding-document-input-type",
            "document",
            "--embedding-query-input-type",
            "query",
            "--index-batch-size",
            "8",
            "--index-max-retries",
            "2",
        ]
    )
    assert args.pipeline == "flat-baseline"
    assert args.subject == ["math"]


def test_flat_baseline_subjects_are_explicit_unique_and_normalized() -> None:
    assert _validate_subjects(["math", "python"]) == ("math", "python")
    with pytest.raises(ValueError, match="unique"):
        _validate_subjects(["math", "math"])
    with pytest.raises(ValueError, match="normalized"):
        _validate_subjects(["_needs_ocr"])
