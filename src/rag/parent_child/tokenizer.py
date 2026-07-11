"""Explicit, fingerprint-checked BM25 tokenizers for sealed generations."""

from __future__ import annotations

from pathlib import Path

import jieba

from src.config.rag_index_config import Bm25Config
from src.rag.parent_child._storage_io import sha256_file


class TokenizerContractError(RuntimeError):
    """Configured tokenizer implementation or dictionary identity is invalid."""


class ConfiguredJiebaTokenizer:
    """Jieba precise-mode tokenizer bound to package and dictionary fingerprints."""

    def __init__(self, *, config: Bm25Config) -> None:
        if config.tokenizer != "jieba_builtin_precise_v1":
            raise TokenizerContractError("unsupported BM25 tokenizer implementation")
        if config.tokenizer_version != jieba.__version__:
            raise TokenizerContractError(
                "configured Jieba version does not match runtime"
            )
        dictionary_handle = jieba.dt.get_dict_file()
        try:
            dictionary_name = getattr(dictionary_handle, "name", None)
        finally:
            dictionary_handle.close()
        if not isinstance(dictionary_name, str):
            raise TokenizerContractError("Jieba dictionary path is unavailable")
        dictionary_path = Path(dictionary_name).resolve(strict=True)
        if sha256_file(dictionary_path) != config.dictionary_hash:
            raise TokenizerContractError("Jieba dictionary fingerprint mismatch")
        self._tokenizer = jieba.Tokenizer(dictionary=str(dictionary_path))

    def __call__(self, text: str) -> tuple[str, ...]:
        if not isinstance(text, str) or not text:
            raise TokenizerContractError("tokenizer input must be non-empty text")
        try:
            return tuple(
                token.strip()
                for token in self._tokenizer.cut(text, cut_all=False, HMM=True)
                if token.strip()
            )
        except Exception as exc:
            raise TokenizerContractError("Jieba tokenization failed") from exc


__all__ = ["ConfiguredJiebaTokenizer", "TokenizerContractError"]
