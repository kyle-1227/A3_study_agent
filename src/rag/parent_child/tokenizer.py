"""Explicit, fingerprint-checked BM25 tokenizers for sealed generations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jieba

from src.config.rag_index_config import Bm25Config
from src.rag.parent_child._storage_io import sha256_file


class TokenizerContractError(RuntimeError):
    """Configured tokenizer implementation or dictionary identity is invalid."""


@dataclass(frozen=True)
class JiebaRuntimeIdentity:
    """The exact Jieba package and built-in dictionary used by this process."""

    tokenizer_version: str
    dictionary_path: Path
    dictionary_hash: str


def resolve_jieba_runtime_identity() -> JiebaRuntimeIdentity:
    """Read and fingerprint Jieba's configured built-in dictionary exactly once."""

    try:
        dictionary_handle = jieba.dt.get_dict_file()
    except Exception as exc:
        raise TokenizerContractError("Jieba dictionary path is unavailable") from exc
    try:
        dictionary_name = getattr(dictionary_handle, "name", None)
    finally:
        try:
            dictionary_handle.close()
        except Exception as exc:
            raise TokenizerContractError(
                "Jieba dictionary handle cannot be closed"
            ) from exc
    if not isinstance(dictionary_name, str):
        raise TokenizerContractError("Jieba dictionary path is unavailable")
    logical_path = Path(dictionary_name)
    if logical_path.is_symlink():
        raise TokenizerContractError("Jieba dictionary must not be a symlink")
    try:
        dictionary_path = logical_path.resolve(strict=True)
    except OSError as exc:
        raise TokenizerContractError("Jieba dictionary path is unavailable") from exc
    try:
        dictionary_hash = sha256_file(dictionary_path)
    except Exception as exc:
        raise TokenizerContractError(
            "Jieba dictionary fingerprint cannot be computed"
        ) from exc
    return JiebaRuntimeIdentity(
        tokenizer_version=jieba.__version__,
        dictionary_path=dictionary_path,
        dictionary_hash=dictionary_hash,
    )


class ConfiguredJiebaTokenizer:
    """Jieba precise-mode tokenizer bound to package and dictionary fingerprints."""

    def __init__(self, *, config: Bm25Config) -> None:
        if config.tokenizer != "jieba_builtin_precise_v1":
            raise TokenizerContractError("unsupported BM25 tokenizer implementation")
        identity = resolve_jieba_runtime_identity()
        if config.tokenizer_version != identity.tokenizer_version:
            raise TokenizerContractError(
                "configured Jieba version does not match runtime"
            )
        if identity.dictionary_hash != config.dictionary_hash:
            raise TokenizerContractError("Jieba dictionary fingerprint mismatch")
        self._tokenizer = jieba.Tokenizer(dictionary=str(identity.dictionary_path))

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


__all__ = [
    "ConfiguredJiebaTokenizer",
    "JiebaRuntimeIdentity",
    "TokenizerContractError",
    "resolve_jieba_runtime_identity",
]
