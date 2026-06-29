"""Memory embedding providers for vector search.

Production factory behavior is intentionally strict in Phase 0:
- provider configuration must be explicit,
- only real providers are accepted,
- missing API keys and provider/API failures raise typed errors,
- no synthetic or degraded fallback path is available here.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from src.config import get_setting
from src.memory.errors import (
    MemoryEmbeddingConfigError,
    MemoryEmbeddingRuntimeError,
    sanitize_memory_error,
)

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Pluggable embedding provider for memory vector search."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        ...

    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings produced by this provider."""
        ...


class DeepSeekEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible DeepSeek embedding provider for memory vectorization."""

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str,
        base_url: str,
        timeout: float,
        api_key: str | None = None,
    ) -> None:
        import httpx

        self._model = _non_empty_value(model, "memory.embedding.model")
        self._api_key_env = _non_empty_value(
            api_key_env, "memory.embedding.api_key_env"
        )
        self._base_url = _non_empty_value(base_url, "memory.embedding.base_url").rstrip(
            "/"
        )
        self._timeout = timeout
        self._api_key = (
            api_key if api_key is not None else os.getenv(self._api_key_env, "")
        )
        if not self._api_key:
            raise MemoryEmbeddingConfigError(f"{self._api_key_env} is not configured")
        self._client = httpx.AsyncClient(timeout=self._timeout)
        self._dim: int | None = None

    @classmethod
    def from_settings(cls) -> "DeepSeekEmbeddingProvider":
        """Build the provider from explicit memory embedding settings."""
        return cls(
            model=_required_str_setting("memory.embedding.model"),
            base_url=_required_str_setting("memory.embedding.base_url"),
            api_key_env=_required_str_setting("memory.embedding.api_key_env"),
            timeout=_required_positive_float_setting(
                "memory.embedding.timeout_seconds"
            ),
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via HTTP POST to `/embeddings`."""
        if not texts:
            return []
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
        except Exception as exc:
            raise MemoryEmbeddingRuntimeError(
                f"Embedding API call failed: {sanitize_memory_error(exc)}"
            ) from exc

        try:
            data = response.json()
        except Exception as exc:
            raise MemoryEmbeddingRuntimeError(
                f"Embedding response JSON decode failed: {sanitize_memory_error(exc)}"
            ) from exc

        if isinstance(data, dict) and data.get("error"):
            raise MemoryEmbeddingRuntimeError(
                f"Embedding provider returned error: {sanitize_memory_error(data['error'])}"
            )

        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list) or len(items) != len(texts):
            raise MemoryEmbeddingRuntimeError(
                "Embedding response data length did not match input"
            )

        embeddings: list[list[float]] = []
        for item in items:
            embedding = item.get("embedding") if isinstance(item, dict) else None
            embeddings.append(_validate_embedding_vector(embedding))

        if embeddings:
            self._dim = len(embeddings[0])
        return embeddings

    def embedding_dim(self) -> int:
        """Return the dimension after at least one successful embedding call."""
        if self._dim is None:
            raise MemoryEmbeddingRuntimeError(
                "Embedding dimension is unavailable before a successful embedding call"
            )
        return self._dim


_embedding_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    """Get or create the singleton production embedding provider."""
    global _embedding_provider
    if _embedding_provider is not None:
        return _embedding_provider

    provider_name = get_setting("memory.embedding_provider")
    if provider_name is None:
        raise MemoryEmbeddingConfigError("memory.embedding_provider is required")
    provider_name = str(provider_name).strip().lower()
    if provider_name != "deepseek":
        raise MemoryEmbeddingConfigError(
            f"Unsupported memory.embedding_provider={provider_name!r}; only 'deepseek' is allowed"
        )

    _embedding_provider = DeepSeekEmbeddingProvider.from_settings()
    logger.info("Memory embedding provider configured: %s", provider_name)
    return _embedding_provider


def reset_embedding_provider() -> None:
    """Reset the singleton between caller-managed lifecycles."""
    global _embedding_provider
    _embedding_provider = None


def _required_str_setting(key: str) -> str:
    value = get_setting(key)
    if value is None:
        raise MemoryEmbeddingConfigError(f"{key} is required")
    return _non_empty_value(value, key)


def _required_positive_float_setting(key: str) -> float:
    value = get_setting(key)
    if value is None:
        raise MemoryEmbeddingConfigError(f"{key} is required")
    if isinstance(value, bool):
        raise MemoryEmbeddingConfigError(f"{key} must be a positive number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise MemoryEmbeddingConfigError(f"{key} must be a positive number") from exc
    if parsed <= 0:
        raise MemoryEmbeddingConfigError(f"{key} must be > 0")
    return parsed


def _non_empty_value(value: Any, key: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise MemoryEmbeddingConfigError(f"{key} must be a non-empty string")
    return text


def _validate_embedding_vector(value: Any) -> list[float]:
    if not isinstance(value, list) or not value:
        raise MemoryEmbeddingRuntimeError(
            "Embedding response item missing embedding vector"
        )
    vector: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise MemoryEmbeddingRuntimeError(
                "Embedding vector contains non-numeric values"
            )
        vector.append(float(item))
    return vector
