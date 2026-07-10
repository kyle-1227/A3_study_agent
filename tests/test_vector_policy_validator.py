from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

from langchain_core.documents import Document
import pytest

from scripts import validate_retrieval_policy as validate_script
from src.rag.eval import vector_policy_validator as validator
from src.rag.eval.chunk_optimizer import BASELINE_CANDIDATE_NAME, ChunkPolicyCandidate
from src.rag.eval.vector_policy_validator import (
    RetrievalEvalCase,
    RetrievalPolicyValidationConfig,
    compute_retrieval_metrics,
    evidence_hit,
    validate_index_root,
    validate_retrieval_policies,
)
from src.rag.ids import normalize_for_hash, sha1_text


@pytest.fixture
def local_tmp_path():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        yield Path(tmpdir)


class FakeVectorStore:
    def __init__(self, results: list[tuple[Document, float]]) -> None:
        self.results = results
        self.filters: list[dict[str, Any] | None] = []

    def similarity_search_with_score(
        self, query: str, *, k: int, filter: dict[str, Any] | None = None
    ) -> list[tuple[Document, float]]:
        self.filters.append(filter)
        return self.results[:k]


class FailingVectorStore:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.filters: list[dict[str, Any] | None] = []

    def similarity_search_with_score(
        self, query: str, *, k: int, filter: dict[str, Any] | None = None
    ) -> list[tuple[Document, float]]:
        self.filters.append(filter)
        raise self.exc


def _candidate(name: str = BASELINE_CANDIDATE_NAME) -> ChunkPolicyCandidate:
    mode, size, overlap = name.split("_")
    return ChunkPolicyCandidate(
        name=name,
        splitter_mode=mode,
        chunk_size=int(size.removeprefix("size")),
        chunk_overlap=int(overlap.removeprefix("overlap")),
        too_short_chars=80,
    )


def _doc(
    text: str,
    *,
    chunk_id: str,
    subject: str = "alpha",
    source_relpath: str = "data/alpha/source.md",
    section_id: str | None = "sec_intro",
    section_path: str | None = "Intro",
    section_title: str | None = "Intro",
) -> Document:
    metadata: dict[str, Any] = {
        "doc_id": f"doc_{subject}",
        "chunk_id": chunk_id,
        "source_relpath": source_relpath,
        "source_file": Path(source_relpath).name,
        "source_file_sha1": f"file_{subject}",
        "source_file_size": 100,
        "subject": subject,
        "doc_type": "course_material",
        "chunk_index": 0,
        "chunk_policy_version": "structure_v1",
        "index_version": "a3_rag_v1",
        "content_sha1": sha1_text(normalize_for_hash(text)),
        "chunk_chars": len(text),
    }
    if section_id is not None:
        metadata["section_id"] = section_id
    if section_path is not None:
        metadata["section_path"] = section_path
    if section_title is not None:
        metadata["section_title"] = section_title
    return Document(page_content=text, metadata=metadata)


def _case(
    *,
    section_id: str | None = "sec_intro",
    section_path: str | None = "Intro",
    anchor_text: str = "",
    source_relpath: str = "data/alpha/source.md",
    baseline_chunk_id: str = "baseline_chunk",
) -> RetrievalEvalCase:
    anchor_hash = sha1_text(normalize_for_hash(anchor_text)) if anchor_text else None
    return RetrievalEvalCase(
        query_id="query_alpha",
        subject="alpha",
        query="intro retrieval query",
        query_type="section_title",
        difficulty="easy",
        gold_evidence_id="evidence_alpha",
        gold_source_relpath=source_relpath,
        gold_section_id=section_id,
        gold_section_path=section_path,
        gold_anchor_hash=anchor_hash,
        gold_anchor_type="anchor" if anchor_text else None,
        baseline_gold_chunk_id=baseline_chunk_id,
        gold_doc_id="doc_alpha",
        anchor_text=anchor_text,
    )


def _trace_events(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _trace_writer(local_tmp_path: Path, name: str = "trace.jsonl"):
    return validator.RetrievalTraceWriter(
        enabled=True,
        output_dir=local_tmp_path / "reports",
        trace_output=local_tmp_path / "reports" / name,
        run_id="testrun",
        project_root=local_tmp_path,
    )


def _successful_metrics(*, evidence_recall: float = 0.8) -> dict[str, Any]:
    return {
        "query_count": 50,
        "query_error_count": 0,
        "query_success_count": 50,
        "query_success_rate": 1.0,
        "failed_query_ids": [],
        "failed_queries": [],
        "load_error_count": 0,
        "index_error_count": 0,
        "chunk_count": 10,
        "source_count": 1,
        "chunk_count_ratio": 1.0,
        "index_build_status": "success",
        "embedding_success_rate": 1.0,
        "evidence_mrr": evidence_recall,
        "mrr": evidence_recall,
        "evidence_recall_at_5": evidence_recall,
        "recall_at_5": evidence_recall,
        "source_recall_at_5": 1.0,
        "section_recall_at_5": 1.0,
        "noise_at_5": 0.0,
        "baseline_chunk_recall_at_5": evidence_recall,
    }


def test_evidence_hit_ignores_policy_specific_chunk_id_when_section_matches():
    case = _case(baseline_chunk_id="baseline_chunk")
    retrieved = _doc("same evidence", chunk_id="candidate_chunk")

    assert evidence_hit(case, retrieved)


def test_same_source_without_section_or_anchor_is_source_only_not_evidence():
    case = _case(section_id="sec_gold", section_path="Gold")
    retrieved = _doc(
        "same source different section",
        chunk_id="candidate_chunk",
        section_id="sec_other",
        section_path="Other",
    )
    record = {
        "evidence_rank": 1 if evidence_hit(case, retrieved) else None,
        "section_rank": None,
        "source_rank": 1,
        "baseline_chunk_rank": None,
        "retrieved": [
            {
                "rank": 1,
                "is_noise": False,
                "is_gold_source": True,
                "is_gold_section": False,
                "is_gold_evidence": False,
            }
        ],
    }

    metrics = compute_retrieval_metrics(
        records=[record],
        top_k=(1, 5),
        chunk_count=1,
        source_count=1,
        baseline_chunk_count=1,
        index_build_status="success",
        embedding_success_rate=1.0,
    )

    assert metrics["evidence_recall_at_1"] == 0.0
    assert metrics["source_recall_at_5"] == 1.0


def test_anchor_hash_hit_counts_as_evidence_even_with_different_section():
    anchor = "stable anchor phrase for in-memory matching"
    case = _case(section_id="sec_gold", section_path="Gold", anchor_text=anchor)
    retrieved = _doc(
        f"candidate has {anchor} but a different section",
        chunk_id="candidate_chunk",
        section_id="sec_other",
        section_path="Other",
    )

    assert evidence_hit(case, retrieved, anchor_text=anchor)


def test_same_chunk_id_without_evidence_match_is_not_primary_hit():
    case = _case(section_id="sec_gold", section_path="Gold")
    retrieved = _doc(
        "same chunk id but wrong evidence",
        chunk_id="baseline_chunk",
        source_relpath="data/alpha/other.md",
        section_id="sec_other",
        section_path="Other",
    )

    assert not evidence_hit(case, retrieved)


def test_baseline_chunk_recall_is_diagnostic_not_evidence_recall():
    records = [
        {
            "evidence_rank": 1,
            "section_rank": 1,
            "source_rank": 1,
            "baseline_chunk_rank": None,
            "retrieved": [{"rank": 1, "is_noise": False}],
        }
    ]

    metrics = compute_retrieval_metrics(
        records=records,
        top_k=(1, 3, 5),
        chunk_count=1,
        source_count=1,
        baseline_chunk_count=1,
        index_build_status="success",
        embedding_success_rate=1.0,
    )

    assert metrics["evidence_recall_at_1"] == 1.0
    assert metrics["recall_at_1"] == metrics["evidence_recall_at_1"]
    assert metrics["evidence_mrr"] == 1.0
    assert metrics["baseline_chunk_recall_at_1"] == 0.0


def test_subject_load_failure_fail_fast_writes_trace_and_raises(
    local_tmp_path, monkeypatch
):
    data_dir = local_tmp_path / "data"
    (data_dir / "alpha").mkdir(parents=True)
    trace = _trace_writer(local_tmp_path)
    trace.clear()

    def fail_load_documents(*args, **kwargs):
        raise RuntimeError("load failed at C:\\Users\\kyle\\secret.pdf")

    monkeypatch.setattr(validator, "load_documents", fail_load_documents)

    with pytest.raises(RuntimeError, match="load failed"):
        validator._load_documents_for_policy(
            candidate=_candidate(),
            data_dir=data_dir,
            subjects=("alpha",),
            project_root=local_tmp_path,
            config=RetrievalPolicyValidationConfig(
                data_dir=data_dir,
                project_root=local_tmp_path,
                fail_fast=True,
            ),
            trace=trace,
            trace_base={"run_id": "testrun"},
        )
    trace.close()

    trace_text = trace.path.read_text(encoding="utf-8")
    assert "subject_load_failed" in trace_text
    assert "C:\\Users\\kyle" not in trace_text


def test_subject_load_failure_non_fail_fast_records_skipped(
    local_tmp_path, monkeypatch
):
    data_dir = local_tmp_path / "data"
    (data_dir / "alpha").mkdir(parents=True)
    trace = _trace_writer(local_tmp_path)
    trace.clear()

    def fail_load_documents(*args, **kwargs):
        raise RuntimeError("load failed")

    monkeypatch.setattr(validator, "load_documents", fail_load_documents)

    docs, skipped = validator._load_documents_for_policy(
        candidate=_candidate(),
        data_dir=data_dir,
        subjects=("alpha",),
        project_root=local_tmp_path,
        config=RetrievalPolicyValidationConfig(
            data_dir=data_dir,
            project_root=local_tmp_path,
            fail_fast=False,
        ),
        trace=trace,
        trace_base={"run_id": "testrun"},
    )
    trace.close()

    assert docs == []
    assert skipped[0]["subject"] == "alpha"
    assert skipped[0]["error_type"] == "RuntimeError"
    assert validator._load_error_count(skipped) == 1


def test_query_failure_fail_fast_writes_trace_and_raises(local_tmp_path):
    trace = _trace_writer(local_tmp_path)
    trace.clear()

    with pytest.raises(RuntimeError, match="query failed"):
        validator._evaluate_policy_queries(
            policy_name=BASELINE_CANDIDATE_NAME,
            vectorstore=FailingVectorStore(RuntimeError("query failed")),
            cases=[_case()],
            anchor_text_by_query_id={},
            duplicate_chunk_ids=set(),
            top_k=(1, 5),
            too_short_chars=80,
            trace=trace,
            trace_base={"run_id": "testrun"},
            fail_fast=True,
        )
    trace.close()

    events = _trace_events(trace.path)
    assert any(event["event"] == "vector_query_failed" for event in events)


def test_query_failure_non_fail_fast_records_failed_query(local_tmp_path):
    trace = _trace_writer(local_tmp_path)
    trace.clear()

    records, success_rate = validator._evaluate_policy_queries(
        policy_name=BASELINE_CANDIDATE_NAME,
        vectorstore=FailingVectorStore(RuntimeError("query failed")),
        cases=[_case()],
        anchor_text_by_query_id={},
        duplicate_chunk_ids=set(),
        top_k=(1, 5),
        too_short_chars=80,
        trace=trace,
        trace_base={"run_id": "testrun"},
        fail_fast=False,
    )
    trace.close()

    metrics = compute_retrieval_metrics(
        records=records,
        top_k=(1, 5),
        chunk_count=1,
        source_count=1,
        baseline_chunk_count=1,
        index_build_status="success",
        embedding_success_rate=success_rate,
    )
    assert metrics["query_error_count"] == 1
    assert metrics["query_success_count"] == 0
    assert metrics["query_success_rate"] == 0.0
    assert metrics["failed_query_ids"] == ["query_alpha"]
    assert validator._policy_entry_status(metrics) == "partial_failed"


def test_vector_filter_type_error_never_retries_without_filter(local_tmp_path):
    trace = _trace_writer(local_tmp_path)
    trace.clear()
    vectorstore = FailingVectorStore(
        TypeError("filter unsupported Authorization: Bearer SECRET")
    )

    records, _ = validator._evaluate_policy_queries(
        policy_name=BASELINE_CANDIDATE_NAME,
        vectorstore=vectorstore,
        cases=[_case()],
        anchor_text_by_query_id={},
        duplicate_chunk_ids=set(),
        top_k=(1, 5),
        too_short_chars=80,
        trace=trace,
        trace_base={"run_id": "testrun"},
        fail_fast=False,
    )
    trace.close()

    assert len(vectorstore.filters) == 1
    assert vectorstore.filters[0] == {"subject": {"$eq": "alpha"}}
    assert records[0]["error_type"] == "TypeError"
    trace_text = trace.path.read_text(encoding="utf-8")
    assert "vector_filter_unsupported" in trace_text
    assert "SECRET" not in trace_text


def test_failed_or_partial_candidates_are_not_recommended():
    baseline_metrics = _successful_metrics(evidence_recall=0.8)
    candidate_metrics = _successful_metrics(evidence_recall=1.0)
    candidate_metrics["query_error_count"] = 1
    candidate_metrics["query_success_rate"] = 0.98
    candidate_metrics["failed_query_ids"] = ["query_alpha"]
    candidate_metrics["failed_queries"] = [{"query_id": "query_alpha"}]
    policy_entries = [
        {
            "policy_name": BASELINE_CANDIDATE_NAME,
            "status": "success",
            "global_metrics": baseline_metrics,
            "metrics_delta_vs_baseline": {},
        },
        {
            "policy_name": "structure_size1000_overlap200",
            "status": "partial_failed",
            "global_metrics": candidate_metrics,
            "metrics_delta_vs_baseline": {
                "evidence_recall_at_5_delta": 0.2,
                "evidence_mrr_delta": 0.2,
                "source_recall_at_5_delta": 0.0,
                "noise_at_5_delta": 0.0,
            },
        },
    ]
    subject_report = {
        "alpha": {
            "policies": {
                BASELINE_CANDIDATE_NAME: {
                    "status": "success",
                    "metrics": baseline_metrics,
                    "metrics_delta_vs_baseline": {},
                },
                "structure_size1000_overlap200": {
                    "status": "partial_failed",
                    "metrics": candidate_metrics,
                    "metrics_delta_vs_baseline": {
                        "evidence_recall_at_5_delta": 0.2,
                        "evidence_mrr_delta": 0.2,
                        "source_recall_at_5_delta": 0.0,
                        "noise_at_5_delta": 0.0,
                    },
                },
            }
        }
    }

    recommendation = validator._build_recommendation(
        policy_entries=policy_entries,
        subject_report=subject_report,
        baseline_policy=BASELINE_CANDIDATE_NAME,
    )

    assert recommendation["global_best_policy"] == BASELINE_CANDIDATE_NAME
    assert recommendation["global_action"] == "keep_current_default"
    assert recommendation["subject_policy_map"]["alpha"]["recommended_policy"] is None


def test_index_root_rejects_production_chroma(local_tmp_path, monkeypatch):
    production = local_tmp_path / "chroma_store"
    monkeypatch.setattr(validator, "_resolve_persist_dir", lambda: str(production))

    with pytest.raises(ValueError, match="index_root"):
        validate_index_root(production, project_root=local_tmp_path)
    with pytest.raises(ValueError, match="index_root"):
        validate_index_root(production / "child", project_root=local_tmp_path)


def test_cli_trace_output_requires_trace_and_positive_values(local_tmp_path):
    with pytest.raises(SystemExit):
        validate_script.parse_args(
            ["--trace-output", str(local_tmp_path / "trace.jsonl")]
        )
    with pytest.raises(SystemExit):
        validate_script.parse_args(["--max-policies", "0"])
    with pytest.raises(SystemExit):
        validate_script.parse_args(["--top-k", "0"])


def test_trace_writer_disables_trace_when_file_append_fails(
    local_tmp_path, monkeypatch
):
    trace_output = local_tmp_path / "reports" / "trace.jsonl"
    writer = validator.RetrievalTraceWriter(
        enabled=True,
        output_dir=local_tmp_path / "reports",
        trace_output=trace_output,
        run_id="tracefail",
        project_root=local_tmp_path,
    )
    original_open = Path.open

    def flaky_open(self, *args, **kwargs):
        if self == trace_output:
            raise OSError(22, "Invalid argument")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", flaky_open)

    writer.write("retrieval_query_finished", {"run_id": "tracefail"})

    assert writer.write_failed is True
    assert writer.enabled is False
    assert writer.path_label is None


def test_cli_loads_dotenv_without_printing_secret(monkeypatch, capsys):
    calls: list[Path] = []

    def fake_load_dotenv(path):
        calls.append(path)
        return True

    def fake_validate(config):
        return {
            "dataset_report_path": "reports/dataset.json",
            "candidates_report_path": "reports/candidates.json",
            "subject_report_path": "reports/subject.json",
            "recommendation_report_path": "reports/recommendation.json",
            "trace_path": None,
            "recommendation_report": {
                "global_action": "keep_current_default",
                "global_best_policy": BASELINE_CANDIDATE_NAME,
            },
        }

    monkeypatch.setattr(validate_script, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(validate_script, "validate_retrieval_policies", fake_validate)

    validate_script.main(["--max-policies", "1"])

    output = capsys.readouterr().out
    assert calls == [validate_script.project_root / ".env"]
    assert "SECRET_VALUE" not in output
    assert "API_KEY" not in output


def test_manifest_fingerprint_mismatch_rebuilds_or_marks_invalid(
    local_tmp_path, monkeypatch
):
    candidate = _candidate()
    docs = [_doc("fresh content", chunk_id="chunk_fresh")]
    index_root = local_tmp_path / "reports" / "retrieval_vector_eval" / "indexes"
    policy_dir = index_root / candidate.name
    manifest_path = policy_dir / "index_manifest.json"
    policy_dir.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "build_status": "success",
                "policy_name": candidate.name,
                "chunk_count": len(docs),
                "source_count": 1,
                "policy_config_fingerprint": "old",
                "embedding_fingerprint": "old",
                "source_manifest_fingerprint": "old",
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    class FakeEmbedding:
        model = "fake-embedding-model"
        base_url = "https://provider.invalid"
        document_input_type = "passage"
        query_input_type = "query"
        timeout = 10
        api_key = "SECRET_VALUE"

    class FakeChroma:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_add_documents(vectorstore, documents, ids, batch_size, max_retries):
        calls.append("rebuilt")

    monkeypatch.setattr(
        validator, "_resolve_persist_dir", lambda: str(local_tmp_path / "chroma_store")
    )
    monkeypatch.setattr(validator, "_get_embedding", lambda: FakeEmbedding())
    monkeypatch.setattr(validator, "Chroma", FakeChroma)
    monkeypatch.setattr(validator, "_add_documents_resilient", fake_add_documents)

    _, manifest, _ = validator._build_or_load_policy_index(
        candidate=candidate,
        documents=docs,
        config=RetrievalPolicyValidationConfig(
            index_root=index_root,
            project_root=local_tmp_path,
        ),
        project_root=local_tmp_path,
    )

    assert calls == ["rebuilt"]
    assert manifest["build_status"] == "success"
    assert manifest["policy_config_fingerprint"] != "old"
    assert manifest["embedding_fingerprint"] != "old"
    assert manifest["source_manifest_fingerprint"] != "old"
    serialized = json.dumps(manifest, ensure_ascii=False)
    assert "SECRET_VALUE" not in serialized
    assert "provider.invalid" not in serialized

    manifest_path.write_text(
        json.dumps(
            {
                "build_status": "success",
                "policy_name": candidate.name,
                "chunk_count": len(docs),
                "source_count": 1,
                "policy_config_fingerprint": "old",
                "embedding_fingerprint": "old",
                "source_manifest_fingerprint": "old",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="invalid_index"):
        validator._build_or_load_policy_index(
            candidate=candidate,
            documents=docs,
            config=RetrievalPolicyValidationConfig(
                index_root=index_root,
                reuse_index=True,
                project_root=local_tmp_path,
            ),
            project_root=local_tmp_path,
        )
    invalid_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert invalid_manifest["build_status"] == "invalid_index"
    assert invalid_manifest["error_type"] == "invalid_index"


def test_section_evidence_has_anchor_and_can_hit_without_section_metadata():
    section_doc = _doc(
        "Stable section body anchor for policy-independent matching.",
        chunk_id="structure_chunk",
        section_id="sec_structure",
        section_path="Module > Topic",
        section_title="Topic",
    )
    dataset = validator.generate_retrieval_eval_dataset([section_doc])
    case = dataset.cases[0]
    retrieved = _doc(
        "Stable section body anchor for policy-independent matching.",
        chunk_id="candidate_chunk",
        section_id=None,
        section_path=None,
        section_title=None,
    )

    assert case.gold_anchor_hash
    assert case.query_type == "section_title"
    assert case.query_id in dataset.anchor_text_by_query_id
    assert evidence_hit(
        case,
        retrieved,
        anchor_text=dataset.anchor_text_by_query_id[case.query_id],
    )


def test_dataset_can_add_structure_supplemental_evidence():
    baseline_doc = _doc(
        "plain recursive text without section metadata",
        chunk_id="baseline_chunk",
        section_id=None,
        section_path=None,
        section_title=None,
    )
    structure_doc = _doc(
        "Stable structure-only section evidence.",
        chunk_id="structure_chunk",
        section_id="sec_structure",
        section_path="Module > Topic",
        section_title="Topic",
    )

    dataset = validator.generate_retrieval_eval_dataset(
        [baseline_doc], supplemental_documents=[structure_doc]
    )

    assert any(case.gold_section_id == "sec_structure" for case in dataset.cases)
    assert all(case.anchor_text for case in dataset.cases if case.gold_section_id)


def test_validator_uses_fake_vector_store_and_writes_safe_reports(
    local_tmp_path, monkeypatch
):
    anchor = "non serialized anchor evidence phrase"
    baseline_doc = _doc(
        f"baseline contains {anchor}",
        chunk_id="baseline_chunk",
        section_id=None,
        section_path=None,
        section_title=None,
    )
    candidate_doc = _doc(
        f"baseline contains {anchor}",
        chunk_id="candidate_chunk",
        section_id=None,
        section_path=None,
        section_title=None,
    )
    docs_by_policy = {
        BASELINE_CANDIDATE_NAME: [baseline_doc],
        "structure_size1000_overlap200": [candidate_doc],
    }

    policy_report = local_tmp_path / "reports" / "splitter_policy_candidates.json"
    policy_report.parent.mkdir(parents=True)
    policy_report.write_text(
        json.dumps(
            {
                "global_ranking": [
                    {
                        "policy_name": BASELINE_CANDIDATE_NAME,
                        "splitter_mode": "recursive",
                        "chunk_size": 1000,
                        "chunk_overlap": 200,
                        "status": "pass",
                        "score": 0.5,
                    },
                    {
                        "policy_name": "structure_size1000_overlap200",
                        "splitter_mode": "structure",
                        "chunk_size": 1000,
                        "chunk_overlap": 200,
                        "status": "pass",
                        "score": 0.9,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_load_documents_for_policy(
        *, candidate, data_dir, subjects, project_root, config, trace, trace_base
    ):
        return docs_by_policy[candidate.name], []

    def fake_build_or_load_policy_index(*, candidate, documents, config, project_root):
        policy_dir = (
            project_root
            / "reports"
            / "retrieval_vector_eval"
            / "indexes"
            / candidate.name
        )
        manifest_path = policy_dir / "index_manifest.json"
        policy_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "build_status": "success",
            "policy_name": candidate.name,
            "chunk_count": len(documents),
            "source_count": 1,
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return (
            FakeVectorStore([(doc, 0.01) for doc in documents]),
            manifest,
            manifest_path,
        )

    monkeypatch.setattr(
        validator, "_load_documents_for_policy", fake_load_documents_for_policy
    )
    monkeypatch.setattr(
        validator, "_build_or_load_policy_index", fake_build_or_load_policy_index
    )
    monkeypatch.setattr(
        validator, "_resolve_persist_dir", lambda: str(local_tmp_path / "chroma_store")
    )
    monkeypatch.setenv("RAG_SPLITTER_MODE", "invalid")
    trace_output = local_tmp_path / "reports" / "retrieval_trace.jsonl"
    trace_output.parent.mkdir(parents=True, exist_ok=True)
    trace_output.write_text("stale trace", encoding="utf-8")

    result = validate_retrieval_policies(
        RetrievalPolicyValidationConfig(
            data_dir=local_tmp_path / "data",
            output_dir=local_tmp_path / "reports",
            index_root=local_tmp_path / "reports" / "retrieval_vector_eval" / "indexes",
            policy_report=policy_report,
            max_policies=2,
            max_queries=1,
            top_k=(1, 5),
            trace_enabled=True,
            trace_output=trace_output,
            project_root=local_tmp_path,
        )
    )

    assert os.environ["RAG_SPLITTER_MODE"] == "invalid"
    candidate_entries = {
        item["policy_name"]: item for item in result["candidates_report"]["policies"]
    }
    candidate_metrics = candidate_entries["structure_size1000_overlap200"][
        "global_metrics"
    ]
    assert candidate_metrics["evidence_recall_at_1"] == 1.0
    assert candidate_metrics["baseline_chunk_recall_at_1"] == 0.0
    assert result["recommendation_report"]["do_not_auto_apply"] is True
    assert result["dataset_report_path"].endswith(
        "retrieval_vector_eval_dataset_max_queries1.json"
    )
    assert result["trace_path"] == "reports/retrieval_trace.jsonl"

    serialized_reports = json.dumps(
        {
            "dataset": result["dataset_report"],
            "candidates": result["candidates_report"],
            "subject": result["subject_report"],
            "recommendation": result["recommendation_report"],
        },
        ensure_ascii=False,
    )
    assert anchor not in serialized_reports
    assert "gold_anchor_text" not in serialized_reports
    assert "preview" not in serialized_reports
    assert "embedding_vector" not in serialized_reports
    assert str(local_tmp_path.resolve()) not in serialized_reports

    trace_text = trace_output.read_text(encoding="utf-8")
    assert "stale trace" not in trace_text
    assert "chunk_evaluated" not in trace_text
    assert anchor not in trace_text
    assert "intro retrieval query" not in trace_text
    assert str(local_tmp_path.resolve()) not in trace_text


def test_report_write_failure_always_raises_even_when_not_fail_fast(
    local_tmp_path, monkeypatch
):
    doc = _doc("baseline content for report failure", chunk_id="baseline_chunk")
    policy_report = local_tmp_path / "reports" / "splitter_policy_candidates.json"
    policy_report.parent.mkdir(parents=True)
    policy_report.write_text(
        json.dumps(
            {
                "global_ranking": [
                    {
                        "policy_name": BASELINE_CANDIDATE_NAME,
                        "splitter_mode": "recursive",
                        "chunk_size": 1000,
                        "chunk_overlap": 200,
                        "status": "pass",
                        "score": 0.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_load_documents_for_policy(
        *, candidate, data_dir, subjects, project_root, config, trace, trace_base
    ):
        return [doc], []

    def fail_write_json(payload, path):
        raise OSError("cannot write C:\\Users\\kyle\\report.json")

    trace_output = local_tmp_path / "reports" / "trace.jsonl"
    monkeypatch.setattr(
        validator, "_load_documents_for_policy", fake_load_documents_for_policy
    )
    monkeypatch.setattr(validator, "_write_json", fail_write_json)
    monkeypatch.setattr(
        validator, "_resolve_persist_dir", lambda: str(local_tmp_path / "chroma_store")
    )

    with pytest.raises(OSError, match="cannot write"):
        validate_retrieval_policies(
            RetrievalPolicyValidationConfig(
                data_dir=local_tmp_path / "data",
                output_dir=local_tmp_path / "reports",
                index_root=local_tmp_path
                / "reports"
                / "retrieval_vector_eval"
                / "indexes",
                policy_report=policy_report,
                max_policies=1,
                max_queries=1,
                top_k=(1, 5),
                trace_enabled=True,
                trace_output=trace_output,
                project_root=local_tmp_path,
                fail_fast=False,
            )
        )

    trace_text = trace_output.read_text(encoding="utf-8")
    assert "retrieval_policy_validation_failed" in trace_text
    assert "C:\\Users\\kyle" not in trace_text


def test_subject_missing_from_candidate_is_reported_as_fail(local_tmp_path):
    baseline_subject_metrics = {
        "alpha": {
            "query_count": 1,
            "chunk_count": 10,
            "source_count": 1,
            "chunk_count_ratio": 1.0,
            "index_build_status": "success",
            "embedding_success_rate": 1.0,
            "evidence_mrr": 1.0,
            "mrr": 1.0,
            "evidence_recall_at_5": 1.0,
            "recall_at_5": 1.0,
            "source_recall_at_5": 1.0,
            "section_recall_at_5": 1.0,
            "noise_at_5": 0.0,
            "baseline_chunk_recall_at_5": 1.0,
        }
    }
    candidate_subject_metrics: dict[str, dict[str, Any]] = {}

    report = validator._build_subject_report(
        per_policy_subject_metrics={
            BASELINE_CANDIDATE_NAME: baseline_subject_metrics,
            "structure_size1000_overlap200": candidate_subject_metrics,
        },
        baseline_policy=BASELINE_CANDIDATE_NAME,
        top_k=(5,),
    )

    candidate = report["alpha"]["policies"]["structure_size1000_overlap200"]
    assert candidate["status"] == "failed"
    assert candidate["reason"] == "subject missing from candidate report"
