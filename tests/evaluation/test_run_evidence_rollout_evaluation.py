"""Hermetic CLI publication and content-free failure tests."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import cast

from scripts.run_evidence_rollout_evaluation import main
from src.config.evidence_benchmark_config import load_evidence_benchmark_config
from src.config.rag_rollout_config import load_rag_rollout_config
from src.evaluation.evidence_rollout.contracts import (
    EvidenceEvaluationRuntimeBindingV2,
    EvidenceRolloutDecisionV2,
    EvidenceVariantAttemptV2,
    HumanSemanticReviewBatchContentV2,
    HumanSemanticReviewV2,
    dataset_case_bindings,
    model_fingerprint,
    seal_human_semantic_review_batch,
    load_evidence_rollout_execution_config,
)
from src.evaluation.evidence_rollout.io import (
    canonical_model_bytes,
    load_canonical_json_model,
)
from src.evaluation.evidence_rollout.report import EvidenceRolloutSafeReportV2
from src.learning_guidance.knowledge_graph import load_knowledge_graph
from src.rag.parent_child.manifests import (
    ArtifactDescriptor,
    ArtifactType,
    Bm25ManifestIdentity,
    EmbeddingManifestIdentity,
    GenerationCounts,
    GenerationIntegrityCounts,
    GenerationManifest,
)

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from evaluation.test_evidence_rollout_runner import (  # type: ignore[import-not-found]  # noqa: E402
    EXECUTOR_FINGERPRINT,
    REVIEWER_IDENTITY_HASH,
    RUNTIME_FINGERPRINT,
    _MEASUREMENTS,
    _dataset,
    _observation,
    _signed_batch,
)


ROOT = Path(__file__).resolve().parents[2]
_PRIVATE_QUERY = "PRIVATE_CLI_QUERY_CANARY"
_PRIVATE_URL = "https://private.example.invalid/provider"
_PRIVATE_EVIDENCE = "PRIVATE_CLI_EVIDENCE_BODY"
_PRIVATE_PROVIDER_BODY = "PRIVATE_CLI_PROVIDER_BODY"
_PRIVATE_SECRET = "PRIVATE_CLI_SECRET"
_FORBIDDEN_MARKERS = (
    _PRIVATE_QUERY,
    _PRIVATE_URL,
    _PRIVATE_EVIDENCE,
    _PRIVATE_PROVIDER_BODY,
    _PRIVATE_SECRET,
    "raw_provider_body",
    "authorization",
)

_CONFIG_SOURCES = {
    "execution": ROOT / "config" / "evaluation" / "evidence_rollout.yaml",
    "benchmark": ROOT / "config" / "rag" / "evidence_benchmark.yaml",
    "rollout": ROOT / "config" / "rag" / "rollout.yaml",
    "protocol": (
        ROOT / "config" / "evaluation" / "evidence_semantic_review_protocol.md"
    ),
    "knowledge_graph": (
        ROOT / "config" / "learning_guidance" / "knowledge_graph_v1.yaml"
    ),
}


def _copy_config_inputs(project_root: Path) -> dict[str, Path]:
    destinations = {
        "execution": project_root / "inputs" / "evidence_rollout.yaml",
        "benchmark": project_root / "inputs" / "evidence_benchmark.yaml",
        "rollout": project_root / "inputs" / "rollout.yaml",
        "protocol": project_root / "inputs" / "review_protocol.md",
        "knowledge_graph": project_root / "inputs" / "knowledge_graph_v1.yaml",
    }
    for name, destination in destinations.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = _CONFIG_SOURCES[name].read_bytes()
        if name == "protocol":
            payload = payload.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        destination.write_bytes(payload)
    return destinations


def _generation_manifest(generation_id: str) -> GenerationManifest:
    artifact_types: tuple[ArtifactType, ...] = (
        "chroma_children",
        "parent_store",
        "bm25_corpus",
        "bm25_manifest",
        "policy_manifest",
        "subject_manifest",
        "build_report",
    )
    artifacts = tuple(
        ArtifactDescriptor(
            artifact_type=artifact_type,
            relative_path=f"artifacts/{position}-{artifact_type}.bin",
            sha256=f"{position + 1:x}" * 64,
            schema_version="artifact_v1",
            size_bytes=1,
        )
        for position, artifact_type in enumerate(artifact_types)
    )
    return GenerationManifest(
        schema_version="generation_manifest_v1",
        generation_id=generation_id,
        build_state="ready",
        code_revision="hermetic-test-revision",
        build_time_utc=datetime(2026, 7, 15, tzinfo=UTC),
        collection_name="hermetic_children",
        artifacts=artifacts,
        embedding=EmbeddingManifestIdentity(
            provider="hermetic-provider",
            model="hermetic-model",
            base_url_identity=_PRIVATE_URL,
            input_types=("document", "query"),
            fingerprint="8" * 64,
            dimension=3,
            distance_metric="cosine",
        ),
        bm25=Bm25ManifestIdentity(
            tokenizer_name="hermetic-tokenizer",
            tokenizer_version="v1",
            dictionary_sha256="9" * 64,
            tokenizer_fingerprint="a" * 64,
            artifact_format="jsonl",
        ),
        subject_manifest_sha256="b" * 64,
        policy_manifest_sha256="c" * 64,
        subject_fingerprint="d" * 64,
        policy_fingerprint="e" * 64,
        source_fingerprint="f" * 64,
        parent_id_set_sha256="0" * 64,
        child_id_set_sha256="1" * 64,
        counts=GenerationCounts(
            source_count=1,
            subject_count=2,
            parent_count=2,
            child_count=2,
            bm25_child_count=2,
        ),
        integrity=GenerationIntegrityCounts(
            duplicate_parent_count=0,
            duplicate_child_count=0,
            orphan_child_count=0,
            unreferenced_parent_count=0,
            generation_mismatch_count=0,
            policy_mismatch_count=0,
            subject_mismatch_count=0,
            bm25_mismatch_count=0,
            chroma_mismatch_count=0,
        ),
        validation_report_sha256="2" * 64,
        validation_passed=True,
    )


def _write_cli_fixture(project_root: Path) -> dict[str, Path]:
    config_paths = _copy_config_inputs(project_root)
    execution_config = load_evidence_rollout_execution_config(config_paths["execution"])
    benchmark_config = load_evidence_benchmark_config(config_paths["benchmark"])
    rollout_config = load_rag_rollout_config(config_paths["rollout"])
    knowledge_graph = load_knowledge_graph(config_paths["knowledge_graph"])
    query = " ".join(
        (
            _PRIVATE_QUERY,
            _PRIVATE_URL,
            _PRIVATE_EVIDENCE,
            _PRIVATE_PROVIDER_BODY,
            f"authorization={_PRIVATE_SECRET}",
        )
    )
    dataset = _dataset(simple_query=query)
    manifest = _generation_manifest("hermetic_generation_1")
    manifest_bytes = canonical_model_bytes(manifest)
    manifest_fingerprint = hashlib.sha256(manifest_bytes).hexdigest()
    binding = EvidenceEvaluationRuntimeBindingV2(
        schema_version="evidence_evaluation_runtime_binding_v2",
        run_id="hermetic_cli_run_1",
        execution_mode="hermetic",
        dataset_id=dataset.dataset_id,
        dataset_fingerprint=dataset.dataset_fingerprint,
        knowledge_graph_data_version=knowledge_graph.data_version,
        knowledge_graph_artifact_fingerprint=knowledge_graph.artifact_fingerprint,
        case_bindings=dataset_case_bindings(dataset),
        execution_config_fingerprint=model_fingerprint(execution_config),
        benchmark_config_fingerprint=model_fingerprint(benchmark_config),
        rollout_config_fingerprint=model_fingerprint(rollout_config),
        runtime_fingerprint=RUNTIME_FINGERPRINT,
        generation_id=manifest.generation_id,
        generation_manifest_fingerprint=manifest_fingerprint,
        executor_fingerprint=EXECUTOR_FINGERPRINT,
    )
    attempts = [
        EvidenceVariantAttemptV2(
            schema_version="evidence_variant_attempt_v2",
            case_id=case.case_id,
            variant=definition.variant,
            status="success",
            observation=_observation(
                case=case,
                definition=definition,
                binding=binding,
            ),
            failure_reason_code=None,
            failure_type=None,
        )
        for case in dataset.cases
        for definition in execution_config.variants
    ]
    attempt_batch = _signed_batch(attempts=attempts)
    reviews = []
    for attempt in attempts:
        observation = attempt.observation
        if observation is None:
            raise AssertionError("CLI fixture requires successful observations")
        measurement = _MEASUREMENTS[(attempt.case_id, attempt.variant)]
        reviews.append(
            HumanSemanticReviewV2(
                schema_version="human_semantic_review_v2",
                case_id=attempt.case_id,
                variant=attempt.variant,
                output_fingerprint=observation.output_fingerprint,
                reviewer_identity_hash=REVIEWER_IDENTITY_HASH,
                reviewed_at="2026-07-15T10:00:00+00:00",
                assessment_source="human",
                supported_claim_count=int(float(measurement["claim_support"]) * 100),
                claim_count=100,
                ungrounded_fact_count=int(float(measurement["ungrounded"]) * 100),
                fact_count=100,
            )
        )
    review_batch = seal_human_semantic_review_batch(
        HumanSemanticReviewBatchContentV2(
            schema_version="human_semantic_review_batch_v2",
            dataset_fingerprint=dataset.dataset_fingerprint,
            runtime_fingerprint=binding.runtime_fingerprint,
            generation_id=binding.generation_id,
            generation_manifest_fingerprint=(binding.generation_manifest_fingerprint),
            review_protocol_fingerprint=(
                execution_config.human_review_protocol_fingerprint
            ),
            reviews=reviews,
        )
    )
    paths = {
        **config_paths,
        "dataset": project_root / "inputs" / "dataset.json",
        "binding": project_root / "inputs" / "binding.json",
        "reviews": project_root / "inputs" / "reviews.json",
        "attempts": project_root / "inputs" / "attempts.json",
        "manifest": project_root / "inputs" / "generation_manifest.json",
    }
    for name, model in (
        ("dataset", dataset),
        ("binding", binding),
        ("reviews", review_batch),
        ("attempts", attempt_batch),
    ):
        paths[name].write_bytes(canonical_model_bytes(model))
    paths["manifest"].write_bytes(manifest_bytes)
    return paths


def _argv(project_root: Path, paths: dict[str, Path], output: Path) -> list[str]:
    return [
        "--project-root",
        str(project_root),
        "--execution-config",
        str(paths["execution"]),
        "--benchmark-config",
        str(paths["benchmark"]),
        "--rollout-config",
        str(paths["rollout"]),
        "--review-protocol",
        str(paths["protocol"]),
        "--dataset",
        str(paths["dataset"]),
        "--knowledge-graph",
        str(paths["knowledge_graph"]),
        "--runtime-binding",
        str(paths["binding"]),
        "--human-reviews",
        str(paths["reviews"]),
        "--attempt-batch",
        str(paths["attempts"]),
        "--generation-manifest",
        str(paths["manifest"]),
        "--output-dir",
        str(output),
    ]


def test_cli_hermetic_run_publishes_blocked_decision(
    tmp_path: Path,
    capsys,
) -> None:
    paths = _write_cli_fixture(tmp_path)
    output = tmp_path / "outputs" / "blocked-run"

    exit_code = main(_argv(tmp_path, paths, output))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert not captured.err
    summary = json.loads(captured.out)
    assert summary["status"] == "blocked"
    assert summary["activation_allowed"] is False
    assert {item.name for item in output.iterdir()} == {
        "activation_decision.json",
        "safe_report.json",
        "safe_report.md",
    }
    decision = load_canonical_json_model(
        output / "activation_decision.json",
        EvidenceRolloutDecisionV2,
    )
    report = load_canonical_json_model(
        output / "safe_report.json",
        EvidenceRolloutSafeReportV2,
    )
    assert decision.status == "blocked"
    assert decision.execution_mode == "hermetic"
    assert decision.activation_allowed is False
    assert decision.benchmark_eligible is True
    assert decision.rollout_activation_enabled is True
    assert decision.reason_codes == [
        "non_live_execution",
    ]
    assert report.reason_codes == decision.reason_codes
    combined = (
        captured.out
        + "\n"
        + "\n".join(item.read_text(encoding="utf-8") for item in output.iterdir())
    )
    for marker in _FORBIDDEN_MARKERS:
        assert marker.casefold() not in combined.casefold()


def test_cli_input_error_writes_only_content_free_failure(
    tmp_path: Path,
    capsys,
) -> None:
    paths = _write_cli_fixture(tmp_path)
    invalid_payload = json.loads(paths["dataset"].read_text(encoding="utf-8"))
    invalid_payload["raw_provider_body"] = " ".join(_FORBIDDEN_MARKERS)
    paths["dataset"].write_bytes(
        json.dumps(
            invalid_payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    output = tmp_path / "outputs" / "invalid-input"

    exit_code = main(_argv(tmp_path, paths, output))

    captured = capsys.readouterr()
    failure_path = output.with_name(output.name + ".failure.json")
    assert exit_code == 2
    assert not output.exists()
    assert failure_path.is_file()
    failure = cast(dict[str, object], json.loads(failure_path.read_bytes()))
    assert failure == {
        "failure_code": "artifact_contract_invalid",
        "failure_type": "EvidenceRolloutArtifactError",
        "schema_version": "evidence_rollout_cli_failure_v2",
        "status": "blocked",
    }
    emitted = captured.out + captured.err + failure_path.read_text(encoding="utf-8")
    for marker in _FORBIDDEN_MARKERS:
        assert marker.casefold() not in emitted.casefold()


def test_cli_invalid_knowledge_graph_fails_before_decision_without_content_leak(
    tmp_path: Path,
    capsys,
) -> None:
    paths = _write_cli_fixture(tmp_path)
    paths["knowledge_graph"].write_text(
        "schema_version: knowledge_graph_v1\nsubjects: [",
        encoding="utf-8",
    )
    output = tmp_path / "outputs" / "invalid-knowledge-graph"

    exit_code = main(_argv(tmp_path, paths, output))

    captured = capsys.readouterr()
    failure_path = output.with_name(output.name + ".failure.json")
    assert exit_code == 2
    assert not output.exists()
    assert failure_path.is_file()
    emitted = captured.out + captured.err + failure_path.read_text(encoding="utf-8")
    for marker in _FORBIDDEN_MARKERS:
        assert marker.casefold() not in emitted.casefold()
