"""
Memory embedding providers — pluggable embedding backends for memory vector search.

Design:
- Abstract EmbeddingProvider interface for future provider swaps
- DeepSeekEmbeddingProvider: reuses the OpenAI-compatible embedding API pattern
  from src/rag/indexer.py (same HTTP client, same auth, same endpoint)
- DummyEmbeddingProvider: zero-vector fallback for keyword-only retrieval
- Singleton factory for lazy initialization
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

# Constants (mirror src/rag/indexer.py defaults)
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2:free")
DEFAULT_EMBEDDING_BASE_URL = os.getenv(
    "EMBEDDING_BASE_URL",
    os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
)
DEFAULT_EMBEDDING_TIMEOUT = 30.0
DEFAULT_DUMMY_DIM = 64


class EmbeddingProvider(ABC):
    """Pluggable embedding provider for memory vector search."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        ...

    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings produced by this provider."""
        ...


class DeepSeekEmbeddingProvider(EmbeddingProvider):
    """Uses the OpenAI-compatible /embeddings endpoint for memory vectorization.

    Reuses the same HTTP pattern as src/rag/indexer.py's OpenAICompatibleEmbeddings,
    but as a standalone async provider that doesn't require ChromaDB or LangChain.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_EMBEDDING_TIMEOUT,
    ):
        import httpx

        self._model = model or DEFAULT_EMBEDDING_MODEL
        self._base_url = (base_url or DEFAULT_EMBEDDING_BASE_URL).rstrip("/")
        self._timeout = timeout

        # Resolve API key
        if api_key:
            self._api_key = api_key
        else:
            api_key_env = os.getenv("EMBEDDING_API_KEY_ENV", "OPENROUTER_API_KEY")
            self._api_key = os.getenv(api_key_env)

        self._client = httpx.AsyncClient(timeout=self._timeout)
        self._dim: int | None = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via HTTP POST to /embeddings."""
        if not texts:
            return []
        if not self._api_key:
            logger.warning("No embedding API key configured; returning zero vectors")
            return [[0.0] * self.embedding_dim() for _ in texts]

        try:
            response = await self._client.post(
                f"{self._base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self._model, "input": texts},
            )
            response.raise_for_status()
            data = response.json()

            if data.get("error"):
                raise RuntimeError(f"Embedding provider returned error: {data['error']}")

            items = data.get("data")
            if not isinstance(items, list) or not items:
                raise ValueError("No embedding data received")

            embeddings: list[list[float]] = []
            for item in items:
                embedding = item.get("embedding") if isinstance(item, dict) else None
                if not isinstance(embedding, list):
                    raise ValueError("Embedding response item missing embedding vector")
                embeddings.append(embedding)

            if self._dim is None and embeddings:
                self._dim = len(embeddings[0])

            return embeddings

        except Exception as exc:
            logger.warning("Embedding API call failed: %s; returning zero vectors", exc)
            return [[0.0] * self.embedding_dim() for _ in texts]

    def embedding_dim(self) -> int:
        """Return cached dimension, or probe the API, or fall back to default."""
        if self._dim is not None:
            return self._dim
        # Probe with a tiny text if we have an API key
        if self._api_key:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We can't await in a sync method; return default
                    return DEFAULT_DUMMY_DIM
                embeddings = loop.run_until_complete(self.embed(["dimension probe"]))
                if embeddings and embeddings[0]:
                    self._dim = len(embeddings[0])
                    return self._dim
            except Exception:
                pass
        return DEFAULT_DUMMY_DIM


class DummyEmbeddingProvider(EmbeddingProvider):
    """Zero-vector fallback provider — enables keyword-only retrieval.

    When no embedding API is available, this provider returns zero vectors
    of a fixed dimension. Vector similarity scores will be zero, so retrieval
    falls back entirely on BM25 keyword scoring.
    """

    def __init__(self, dim: int = DEFAULT_DUMMY_DIM):
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self._dim for _ in texts]

    def embedding_dim(self) -> int:
        return self._dim


# ── Singleton ──────────────────────────────────────────────────────────────

_embedding_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    """Get or create the singleton embedding provider.

    Tries DeepSeekEmbeddingProvider first; falls back to DummyEmbeddingProvider
    if API key is missing or the provider fails to initialize.
    """
    global _embedding_provider
    if _embedding_provider is None:
        try:
            _embedding_provider = DeepSeekEmbeddingProvider()
            logger.info("Memory embedding provider: DeepSeek (model=%s)", _embedding_provider._model)
        except Exception as exc:
            logger.warning("Failed to create DeepSeek embedding provider: %s; using dummy", exc)
            _embedding_provider = DummyEmbeddingProvider()
    return _embedding_provider


def reset_embedding_provider() -> None:
    """Reset the singleton (useful for testing)."""
    global _embedding_provider
    _embedding_provider = None
