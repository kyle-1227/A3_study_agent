"""Deterministic test-only embedding providers."""

from __future__ import annotations

from src.memory.embeddings import EmbeddingProvider


class DeterministicFakeEmbeddingProvider(EmbeddingProvider):
    """Small fake embedder for tests that explicitly inject it."""

    def __init__(self, dim: int = 5, *, fail: Exception | None = None) -> None:
        self._dim = dim
        self._fail = fail

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._fail is not None:
            raise self._fail
        return [self._embed_one(text) for text in texts]

    def embedding_dim(self) -> int:
        return self._dim

    def _embed_one(self, text: str) -> list[float]:
        if self._dim <= 0:
            return []
        seed = sum(ord(char) for char in text)
        return [float((seed + index) % 11) / 10.0 for index in range(self._dim)]
