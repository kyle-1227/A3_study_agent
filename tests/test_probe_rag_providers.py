"""Focused contracts for redacted real-provider probe orchestration."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest
import yaml
from pydantic import ValidationError

from scripts.probe_rag_providers import main
from src.config.rag_index_config import (
    EmbeddingConfig,
    RerankerConfig,
    load_rag_index_config,
)
from src.rag.parent_child.provider_probe import (
    LlmProbeConfig,
    StrictChatCompletionClient,
    run_provider_probe,
)
from src.rag.parent_child.provider_clients import ProviderEmbeddingDimensionError
from src.rag.parent_child.retrieval import RerankCandidate, RerankScore


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _write_probe_config(tmp_path: Path) -> Path:
    source = _REPOSITORY_ROOT / "config" / "rag" / "index.local.yaml"
    payload = load_rag_index_config(source).model_dump(mode="json")
    catalog = payload["catalog"]
    storage = payload["storage"]
    embedding = payload["embedding"]
    reranker = payload["reranker"]
    assert isinstance(catalog, dict)
    assert isinstance(storage, dict)
    assert isinstance(embedding, dict)
    assert isinstance(reranker, dict)
    catalog["data_root"] = "data"
    storage["index_root"] = "indexes/parent_child"
    storage["registry_path"] = "generation_registry.sqlite"
    embedding["expected_dimension"] = 2
    embedding["api_key_env"] = "PROBE_EMBEDDING_KEY"
    reranker["api_key_env"] = "PROBE_RERANKER_KEY"
    config_path = tmp_path / "config" / "index.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    return config_path


class _FakeEmbeddingClient:
    last_http_status = 200

    def __init__(self) -> None:
        self.closed = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        assert len(texts) == 2
        return [[1.0, 0.0], [0.0, 1.0]]

    def embed_query(self, text: str) -> list[float]:
        assert text
        return [0.5, 0.5]

    def close(self) -> None:
        self.closed = True


class _FakeRerankerClient:
    last_http_status = 200

    def __init__(self) -> None:
        self.closed = False

    def rerank(
        self,
        *,
        query: str,
        candidates: tuple[RerankCandidate, ...],
    ) -> tuple[RerankScore, ...]:
        assert query
        assert len(candidates) == 3
        return tuple(
            RerankScore(
                schema_version="rerank_score_v1",
                child_id=candidate.child_id,
                score=(0.9, 0.8, 0.1)[index],
            )
            for index, candidate in enumerate(candidates)
        )

    def close(self) -> None:
        self.closed = True


class _DimensionMismatchEmbeddingClient:
    last_http_status = 200

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        assert len(texts) == 2
        raise ProviderEmbeddingDimensionError(
            actual_dimension=3,
            expected_dimension=2,
        )

    def embed_query(self, text: str) -> list[float]:
        _ = text
        raise AssertionError("query must not run after a batch dimension mismatch")

    def close(self) -> None:
        return None


def _llm_config() -> LlmProbeConfig:
    return LlmProbeConfig(
        provider="configured-chat-provider",
        model="configured-chat-model",
        base_url="https://chat.invalid/v1",
        endpoint_path="/chat/completions",
        api_key_env="PROBE_LLM_KEY",
        timeout_seconds=5.0,
    )


def test_llm_probe_config_requires_an_environment_identifier() -> None:
    with pytest.raises(ValidationError):
        LlmProbeConfig(
            provider="configured-chat-provider",
            model="configured-chat-model",
            base_url="https://chat.invalid/v1",
            endpoint_path="/chat/completions",
            api_key_env="not-a-valid-env-name",
            timeout_seconds=5.0,
        )


def test_probe_writes_redacted_success_report_from_strict_mock_protocols(
    tmp_path: Path,
) -> None:
    config_path = _write_probe_config(tmp_path)
    embedding_client = _FakeEmbeddingClient()
    reranker_client = _FakeRerankerClient()
    received_payloads: list[dict[str, object]] = []
    auth_sentinel = f"probe-auth-{tmp_path.name}"

    def chat_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {auth_sentinel}"
        payload = json.loads(request.content)
        received_payloads.append(payload)
        return httpx.Response(
            200,
            json={
                "id": "chat-probe-1",
                "object": "chat.completion",
                "created": 1,
                "model": "configured-chat-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "先检索证据。",
                            "reasoning_content": None,
                            "refusal": None,
                        },
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "completion_tokens": 2,
                    "prompt_tokens": 3,
                    "total_tokens": 5,
                    "prompt_cache_hit_tokens": 0,
                    "prompt_cache_miss_tokens": 3,
                },
                "system_fingerprint": None,
                "service_tier": None,
            },
        )

    def embedding_factory(_config: EmbeddingConfig) -> _FakeEmbeddingClient:
        return embedding_client

    def reranker_factory(_config: RerankerConfig) -> _FakeRerankerClient:
        return reranker_client

    def llm_factory(config: LlmProbeConfig) -> StrictChatCompletionClient:
        return StrictChatCompletionClient(
            config=config,
            api_key=auth_sentinel,
            transport=httpx.MockTransport(chat_handler),
        )

    report = run_provider_probe(
        project_root=tmp_path,
        index_config_path=config_path.relative_to(tmp_path),
        run_id="probe_success",
        output_directory=Path("reports/rag_build/probe_success"),
        probe_llm_enabled=True,
        llm_config=_llm_config(),
        embedding_client_factory=embedding_factory,
        reranker_client_factory=reranker_factory,
        llm_client_factory=llm_factory,
    )

    report_path = (
        tmp_path / "reports" / "rag_build" / "probe_success" / "provider_probe.json"
    )
    serialized = report_path.read_text(encoding="utf-8")
    payload = json.loads(serialized)
    assert report.success is True
    assert report.failed_stage is None
    assert report.embedding.actual_dimension == 2
    assert report.embedding.batch_supported is True
    assert report.embedding.input_type_supported is True
    assert report.reranker.returned_indices_complete_unique is True
    assert report.reranker.relevant_documents_above_irrelevant is True
    assert report.llm.real_text_returned is True
    assert report.llm.output_sha256 is not None
    assert embedding_client.closed is True
    assert reranker_client.closed is True
    assert received_payloads == [
        {
            "model": "configured-chat-model",
            "messages": [
                {
                    "role": "user",
                    "content": "请用一句话说明：为什么 RAG 在回答前需要先检索证据？",
                }
            ],
        }
    ]
    assert payload["embedding"]["http_status"] == 200
    assert auth_sentinel not in serialized
    assert "Authorization" not in serialized
    assert "先检索证据" not in serialized
    assert "为什么 RAG" not in serialized


def test_missing_embedding_secret_writes_failure_report_without_network(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = _write_probe_config(tmp_path)
    monkeypatch.delenv("PROBE_EMBEDDING_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    report = run_provider_probe(
        project_root=tmp_path,
        index_config_path=config_path.relative_to(tmp_path),
        run_id="probe_missing_secret",
        output_directory=Path("reports/rag_build/probe_missing_secret"),
        probe_llm_enabled=False,
        llm_config=None,
    )

    report_path = (
        tmp_path
        / "reports"
        / "rag_build"
        / "probe_missing_secret"
        / "provider_probe.json"
    )
    serialized = report_path.read_text(encoding="utf-8")
    assert report.success is False
    assert report.failed_stage == "embedding"
    assert report.embedding.failure_type == "missing_secret"
    assert report.embedding.http_status is None
    assert report.reranker.status == "not_run"
    assert "PROBE_EMBEDDING_KEY" not in serialized
    assert "OPENROUTER_API_KEY" not in serialized


def test_dimension_mismatch_reports_the_real_dimension_without_repair(
    tmp_path: Path,
) -> None:
    config_path = _write_probe_config(tmp_path)

    report = run_provider_probe(
        project_root=tmp_path,
        index_config_path=config_path.relative_to(tmp_path),
        run_id="probe_dimension_mismatch",
        output_directory=Path("reports/rag_build/probe_dimension_mismatch"),
        probe_llm_enabled=False,
        llm_config=None,
        embedding_client_factory=lambda _config: _DimensionMismatchEmbeddingClient(),
    )

    assert report.success is False
    assert report.failed_stage == "embedding"
    assert report.embedding.failure_type == "provider_protocol"
    assert report.embedding.response_schema_valid is True
    assert report.embedding.actual_dimension == 3
    assert report.embedding.batch_supported is True
    assert report.embedding.input_type_supported is True


def test_cli_missing_secret_writes_report_and_returns_nonzero(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = _write_probe_config(tmp_path)
    monkeypatch.delenv("PROBE_EMBEDDING_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--index-config",
            str(config_path.relative_to(tmp_path)),
            "--run-id",
            "probe_cli_missing_secret",
            "--output-dir",
            "reports/rag_build/probe_cli_missing_secret",
        ]
    )

    report_path = (
        tmp_path
        / "reports"
        / "rag_build"
        / "probe_cli_missing_secret"
        / "provider_probe.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["success"] is False
    assert payload["failed_stage"] == "embedding"
    assert payload["embedding"]["failure_type"] == "missing_secret"


def test_requested_llm_without_explicit_config_is_a_typed_report_failure(
    tmp_path: Path,
) -> None:
    config_path = _write_probe_config(tmp_path)

    report = run_provider_probe(
        project_root=tmp_path,
        index_config_path=config_path.relative_to(tmp_path),
        run_id="probe_missing_llm_config",
        output_directory=Path("reports/rag_build/probe_missing_llm_config"),
        probe_llm_enabled=True,
        llm_config=None,
        embedding_client_factory=lambda _config: _FakeEmbeddingClient(),
        reranker_client_factory=lambda _config: _FakeRerankerClient(),
    )

    report_path = (
        tmp_path
        / "reports"
        / "rag_build"
        / "probe_missing_llm_config"
        / "provider_probe.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report.success is False
    assert report.failed_stage == "llm"
    assert report.llm.status == "failed"
    assert report.llm.failure_type == "configuration"
    assert payload["llm"]["provider"] is None


def test_openrouter_mapping_is_ephemeral_and_only_for_exact_configured_names(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = _write_probe_config(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    embedding = payload["embedding"]
    reranker = payload["reranker"]
    assert isinstance(embedding, dict)
    assert isinstance(reranker, dict)
    embedding["api_key_env"] = "RAG_EMBEDDING_API_KEY"
    reranker["api_key_env"] = "RAG_RERANKER_API_KEY"
    config_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    monkeypatch.delenv("RAG_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("RAG_RERANKER_API_KEY", raising=False)
    auth_sentinel = f"openrouter-probe-{tmp_path.name}"
    monkeypatch.setenv("OPENROUTER_API_KEY", auth_sentinel)

    def embedding_factory(_config: EmbeddingConfig) -> _FakeEmbeddingClient:
        assert os.environ["RAG_EMBEDDING_API_KEY"] == auth_sentinel
        return _FakeEmbeddingClient()

    def reranker_factory(_config: RerankerConfig) -> _FakeRerankerClient:
        assert os.environ["RAG_RERANKER_API_KEY"] == auth_sentinel
        return _FakeRerankerClient()

    report = run_provider_probe(
        project_root=tmp_path,
        index_config_path=config_path.relative_to(tmp_path),
        run_id="probe_ephemeral_mapping",
        output_directory=Path("reports/rag_build/probe_ephemeral_mapping"),
        probe_llm_enabled=False,
        llm_config=None,
        embedding_client_factory=embedding_factory,
        reranker_client_factory=reranker_factory,
    )

    report_path = (
        tmp_path
        / "reports"
        / "rag_build"
        / "probe_ephemeral_mapping"
        / "provider_probe.json"
    )
    assert report.success is True
    assert auth_sentinel not in report_path.read_text(encoding="utf-8")
