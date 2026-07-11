from __future__ import annotations

import pytest

from scripts.doctor_rag_env import _parser, validate_subject_alignment


def test_parent_child_doctor_requires_every_explicit_input() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args([])

    args = _parser().parse_args(
        [
            "--project-root",
            ".",
            "--pipeline",
            "parent-child",
            "--index-config",
            "config/rag/index.yaml",
            "--benchmark-config",
            "config/rag/benchmark.yaml",
            "--rollout-config",
            "config/rag/rollout.yaml",
            "--output",
            "reports/rag-doctor.json",
        ]
    )
    assert args.pipeline == "parent-child"


def test_doctor_rejects_subject_drift_between_control_planes() -> None:
    validate_subject_alignment(
        index_subjects=("math", "python"),
        benchmark_subjects=("python", "math"),
        rollout_subjects=("math", "python"),
    )
    with pytest.raises(ValueError, match="benchmark"):
        validate_subject_alignment(
            index_subjects=("math", "python"),
            benchmark_subjects=("math",),
            rollout_subjects=("math", "python"),
        )
    with pytest.raises(ValueError, match="rollout"):
        validate_subject_alignment(
            index_subjects=("math", "python"),
            benchmark_subjects=("math", "python"),
            rollout_subjects=("math",),
        )
