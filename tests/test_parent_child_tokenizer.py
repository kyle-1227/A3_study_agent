from __future__ import annotations

from pathlib import Path

import jieba
import pytest

from src.config.rag_index_config import Bm25Config
from src.rag.parent_child._storage_io import sha256_file
from src.rag.parent_child.tokenizer import (
    ConfiguredJiebaTokenizer,
    TokenizerContractError,
    resolve_jieba_runtime_identity,
)


def _dictionary_path() -> Path:
    handle = jieba.dt.get_dict_file()
    try:
        name = handle.name
    finally:
        handle.close()
    return Path(name).resolve(strict=True)


def test_runtime_jieba_identity_matches_installed_dictionary() -> None:
    identity = resolve_jieba_runtime_identity()

    assert identity.tokenizer_version == jieba.__version__
    assert identity.dictionary_path == _dictionary_path()
    assert identity.dictionary_hash == sha256_file(identity.dictionary_path)


def test_configured_jieba_checks_runtime_and_dictionary_fingerprint() -> None:
    config = Bm25Config(
        tokenizer="jieba_builtin_precise_v1",
        tokenizer_version=jieba.__version__,
        dictionary_hash=sha256_file(_dictionary_path()),
        artifact_format="jsonl",
    )

    tokens = ConfiguredJiebaTokenizer(config=config)("机器学习模型")

    assert tokens
    assert all(token.strip() for token in tokens)


def test_configured_jieba_rejects_dictionary_mismatch() -> None:
    config = Bm25Config(
        tokenizer="jieba_builtin_precise_v1",
        tokenizer_version=jieba.__version__,
        dictionary_hash="0" * 64,
        artifact_format="jsonl",
    )
    with pytest.raises(TokenizerContractError, match="fingerprint"):
        ConfiguredJiebaTokenizer(config=config)
