from __future__ import annotations

from pathlib import Path

from src.config.rag_index_config import (
    compute_chunk_policy_id,
    load_rag_index_config,
)
from src.rag.parent_child.bm25_artifact import compute_tokenizer_fingerprint
from src.rag.parent_child.builder import compute_embedding_fingerprint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "rag" / "index.production.yaml"
GENERATION_55_EMBEDDING_FINGERPRINT = (
    "d19fb0d655028fe3a3c04635bf4733a2c301014caa644059064090f007b1911e"
)
GENERATION_55_BM25_FINGERPRINT = (
    "67138eaa2bf15edf2c9577ef819875f605d9f7d71741639174382de134bd7417"
)


def test_inactive_production_candidate_config_matches_generation_55_identity() -> None:
    config = load_rag_index_config(CONFIG_PATH)

    assert not config.catalog.data_root.is_absolute()
    assert not config.storage.index_root.is_absolute()
    assert not config.storage.registry_path.is_absolute()
    assert config.retrieval.reranker_top_n == 20
    assert compute_embedding_fingerprint(config) == (
        GENERATION_55_EMBEDDING_FINGERPRINT
    )
    assert (
        compute_tokenizer_fingerprint(
            tokenizer_name=config.bm25.tokenizer,
            tokenizer_version=config.bm25.tokenizer_version,
            dictionary_sha256=config.bm25.dictionary_hash,
        )
        == GENERATION_55_BM25_FINGERPRINT
    )
    assert set(config.subject_policy_map.values()) == set(config.chunk_policies)
    assert all(
        compute_chunk_policy_id(policy) == policy_id
        for policy_id, policy in config.chunk_policies.items()
    )
    assert config.embedding.provider_routing is not None
    assert config.embedding.provider_routing.allow_fallbacks is False
    assert config.reranker.provider_routing is not None
    assert config.reranker.provider_routing.allow_fallbacks is False


def test_inactive_production_candidate_config_contains_only_secret_names() -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    config = load_rag_index_config(CONFIG_PATH)

    assert config.embedding.api_key_env == "RAG_EMBEDDING_API_KEY"
    assert config.reranker.api_key_env == "RAG_RERANKER_API_KEY"
    assert "api_key:" not in text
