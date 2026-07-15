"""Canonical I/O and content-free evidence rollout report tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from src.evaluation.evidence_rollout.io import (
    EvidenceRolloutArtifactError,
    canonical_model_bytes,
    load_canonical_json_model,
    publish_evidence_rollout_bundle,
)
from src.evaluation.evidence_rollout.report import (
    build_safe_report,
    render_safe_report_markdown,
)

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from evaluation.test_evidence_rollout_runner import (  # type: ignore[import-not-found]  # noqa: E402
    _run,
    _scenario,
)


_PRIVATE_QUERY = "PRIVATE_QUERY_CANARY"
_PRIVATE_URL = "https://private.example.invalid/evidence"
_PRIVATE_EVIDENCE = "PRIVATE_EVIDENCE_BODY_CANARY"
_PRIVATE_PROVIDER_BODY = "PRIVATE_PROVIDER_BODY_CANARY"
_PRIVATE_SECRET = "PRIVATE_SECRET_CANARY"
_FORBIDDEN_MARKERS = (
    _PRIVATE_QUERY,
    _PRIVATE_URL,
    _PRIVATE_EVIDENCE,
    _PRIVATE_PROVIDER_BODY,
    _PRIVATE_SECRET,
    "raw_provider_body",
    "authorization",
)


class _CanonicalFixtureV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["canonical_fixture_v1"]
    identifier: str


def _blocked_decision():
    query = " ".join(
        (
            _PRIVATE_QUERY,
            _PRIVATE_URL,
            _PRIVATE_EVIDENCE,
            _PRIVATE_PROVIDER_BODY,
            f"authorization={_PRIVATE_SECRET}",
        )
    )
    return asyncio.run(_run(_scenario(simple_query=query)))


@pytest.mark.parametrize(
    ("payload_factory", "expected_code"),
    [
        (
            lambda: json.dumps(
                {
                    "identifier": "fixture_1",
                    "schema_version": "canonical_fixture_v1",
                },
                indent=2,
                sort_keys=True,
            ).encode("utf-8"),
            "artifact_not_canonical",
        ),
        (
            lambda: json.dumps(
                {
                    "schema_version": "canonical_fixture_v1",
                    "identifier": "fixture_1",
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8"),
            "artifact_not_canonical",
        ),
        (
            lambda: json.dumps(
                {
                    "extra": "forbidden",
                    "identifier": "fixture_1",
                    "schema_version": "canonical_fixture_v1",
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8"),
            "artifact_contract_invalid",
        ),
    ],
    ids=("pretty", "noncanonical_key_order", "extra_field"),
)
def test_canonical_loader_rejects_noncanonical_or_drifted_json(
    tmp_path: Path,
    payload_factory,
    expected_code: str,
) -> None:
    path = tmp_path / "artifact.json"
    path.write_bytes(payload_factory())

    with pytest.raises(EvidenceRolloutArtifactError) as captured:
        load_canonical_json_model(path, _CanonicalFixtureV1)

    assert captured.value.code == expected_code


def test_canonical_loader_accepts_only_exact_model_bytes(tmp_path: Path) -> None:
    model = _CanonicalFixtureV1(
        schema_version="canonical_fixture_v1",
        identifier="fixture_1",
    )
    path = tmp_path / "artifact.json"
    path.write_bytes(canonical_model_bytes(model))

    assert load_canonical_json_model(path, _CanonicalFixtureV1) == model


def test_safe_report_and_markdown_do_not_expose_private_content() -> None:
    decision = _blocked_decision()
    report = build_safe_report(decision)
    rendered = (
        (canonical_model_bytes(report) + b"\n" + render_safe_report_markdown(report))
        .decode("utf-8")
        .casefold()
    )

    assert report.status == "blocked"
    assert report.activation_allowed is False
    for marker in _FORBIDDEN_MARKERS:
        assert marker.casefold() not in rendered


def test_atomic_bundle_publishes_three_safe_artifacts_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    decision = _blocked_decision()
    report = build_safe_report(decision)

    published = publish_evidence_rollout_bundle(
        project_root=tmp_path,
        output_directory=Path("bundle"),
        decision=decision,
        report=report,
    )

    assert published == tmp_path / "bundle"
    assert {item.name for item in published.iterdir()} == {
        "activation_decision.json",
        "safe_report.json",
        "safe_report.md",
    }
    assert (published / "activation_decision.json").read_bytes() == (
        canonical_model_bytes(decision)
    )
    assert (published / "safe_report.json").read_bytes() == canonical_model_bytes(
        report
    )
    assert (published / "safe_report.md").read_bytes() == (
        render_safe_report_markdown(report)
    )
    combined = b"\n".join(item.read_bytes() for item in published.iterdir()).decode(
        "utf-8"
    )
    for marker in _FORBIDDEN_MARKERS:
        assert marker.casefold() not in combined.casefold()

    with pytest.raises(FileExistsError):
        publish_evidence_rollout_bundle(
            project_root=tmp_path,
            output_directory=Path("bundle"),
            decision=decision,
            report=report,
        )
    assert not tuple(tmp_path.glob(".bundle.staging-*"))
