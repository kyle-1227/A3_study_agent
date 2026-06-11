"""ChromaDB index builder with incremental upsert support.

Uses an OpenAI-compatible embedding API instead of local HuggingFace
models, eliminating heavy local dependencies.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import httpx
import math

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

COLLECTION_NAME = "a3_study_docs"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_EMBEDDING_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
DEFAULT_INDEX_BATCH_SIZE = 64
DEFAULT_INDEX_MAX_RETRIES = 3
DEFAULT_EMBEDDING_TIMEOUT = 60

logger = logging.getLogger(__name__)


def _l2_to_relevance(distance: float) -> float:
    """Convert Chroma L2 distance to a [0, 1] relevance score.

    Chroma's default distance metric is L2 (Euclidean).  For normalized
    embeddings the maximum possible L2 distance is sqrt(2).  We linearly
    map [0, sqrt(2)] → [1, 0] so that higher scores mean higher relevance
    and the values are always within [0, 1].
    """
    return 1.0 - distance / math.sqrt(2)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_persist_dir(persist_directory: Optional[str] = None) -> str:
    """Always resolve to an absolute path anchored at project root."""
    rel = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "chroma_store/")
    path = Path(rel)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return str(path)


def _embedding_api_key_env() -> str:
    return os.getenv("EMBEDDING_API_KEY_ENV", DEFAULT_EMBEDDING_API_KEY_ENV)


def _embedding_api_key() -> str | None:
    api_key_env = _embedding_api_key_env()
    api_key = os.getenv(api_key_env)
    if api_key:
        return api_key

    if api_key_env.startswith(("sk-", "or-")):
        logger.warning(
            "EMBEDDING_API_KEY_ENV appears to contain an API key value instead "
            "of an environment variable name. Prefer "
            "EMBEDDING_API_KEY_ENV=OPENROUTER_API_KEY."
        )
        return api_key_env

    return None


def _embedding_base_url() -> str:
    return os.getenv(
        "EMBEDDING_BASE_URL",
        os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
    )


def _embedding_request_timeout() -> float:
    try:
        return max(1.0, float(os.getenv("EMBEDDING_TIMEOUT", DEFAULT_EMBEDDING_TIMEOUT)))
    except ValueError:
        return DEFAULT_EMBEDDING_TIMEOUT


def _embedding_document_input_type() -> str:
    return os.getenv("EMBEDDING_DOCUMENT_INPUT_TYPE", "passage").strip()


def _embedding_query_input_type() -> str:
    return os.getenv("EMBEDDING_QUERY_INPUT_TYPE", "query").strip()


class OpenAICompatibleEmbeddings(Embeddings):
    """Small embeddings adapter for OpenAI-compatible `/embeddings` endpoints.

    LangChain's OpenAIEmbeddings sends tokenized input through the OpenAI SDK.
    Some OpenRouter embedding models, including NVIDIA Nemotron Embed, expect
    raw text strings plus an optional query/passage input_type. This adapter
    keeps Chroma compatibility while sending that provider-friendly payload.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str,
        batch_size: int = 64,
        document_input_type: str = "",
        query_input_type: str = "",
        timeout: float = DEFAULT_EMBEDDING_TIMEOUT,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.document_input_type = document_input_type
        self.query_input_type = query_input_type
        self.timeout = timeout

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            embeddings.extend(self._embed(batch, input_type=self.document_input_type))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], input_type=self.query_input_type)[0]

    def _embed(self, texts: list[str], *, input_type: str = "") -> list[list[float]]:
        if not texts:
            return []
        if not self.api_key:
            raise RuntimeError("Embedding API key is not configured")

        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }
        if input_type:
            payload["input_type"] = input_type

        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
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
        return embeddings


def _get_embedding(model_name: Optional[str] = None) -> Embeddings:
    """Create an OpenAI-compatible embedding client.

    Args:
        model_name: Override for the embedding model identifier.
            Falls back to ``EMBEDDING_MODEL`` env var, then
            ``DEFAULT_EMBEDDING_MODEL``.

    Returns:
        A configured ``OpenAIEmbeddings`` instance pointing at the configured
        embedding provider.
    """
    model_name = model_name or os.getenv(
        "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
    )
    return OpenAICompatibleEmbeddings(
        model=model_name,
        api_key=_embedding_api_key(),
        base_url=_embedding_base_url(),
        batch_size=64,
        document_input_type=_embedding_document_input_type(),
        query_input_type=_embedding_query_input_type(),
        timeout=_embedding_request_timeout(),
    )


def _index_batch_size() -> int:
    try:
        return max(1, int(os.getenv("INDEX_ADD_BATCH_SIZE", DEFAULT_INDEX_BATCH_SIZE)))
    except ValueError:
        return DEFAULT_INDEX_BATCH_SIZE


def _index_max_retries() -> int:
    try:
        return max(0, int(os.getenv("INDEX_MAX_RETRIES", DEFAULT_INDEX_MAX_RETRIES)))
    except ValueError:
        return DEFAULT_INDEX_MAX_RETRIES


def _content_id(doc: Document) -> str:
    """Deterministic ID from chunk content — true dedup across repeated runs."""
    digest = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
    return f"{doc.metadata.get('source_file', 'unknown')}_{digest}"


def _validate_embedding_provider(embedding: Embeddings) -> None:
    """Fail fast when the configured embedding provider/model is unavailable."""
    try:
        embedding.embed_documents(["embedding provider health check"])
    except Exception as exc:
        model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        api_key_env = _embedding_api_key_env()
        base_url = _embedding_base_url()
        raise RuntimeError(
            "Embedding provider health check failed before indexing. "
            f"model={model_name}, base_url={base_url}, api_key_env={api_key_env}. "
            "Please verify the embedding API key, EMBEDDING_MODEL, account quota, "
            f"and provider availability. Original error: {exc}"
        ) from exc


def _add_documents_resilient(
    vectorstore: Chroma,
    documents: list[Document],
    ids: list[str],
    *,
    batch_size: int,
    max_retries: int,
) -> None:
    """Add documents in small batches, retrying transient embedding failures."""
    total = len(documents)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        _add_batch_with_retry(
            vectorstore,
            documents[start:end],
            ids[start:end],
            max_retries=max_retries,
            batch_label=f"{start + 1}-{end}/{total}",
        )


def _add_batch_with_retry(
    vectorstore: Chroma,
    documents: list[Document],
    ids: list[str],
    *,
    max_retries: int,
    batch_label: str,
) -> None:
    """Retry a Chroma add batch; split once smaller if the provider keeps failing."""
    for attempt in range(max_retries + 1):
        try:
            vectorstore.add_documents(documents=documents, ids=ids)
            return
        except Exception as exc:
            if attempt >= max_retries:
                if len(documents) > 1:
                    mid = len(documents) // 2
                    logger.warning(
                        "Index batch %s failed after %s retries; splitting into %s and %s docs",
                        batch_label,
                        max_retries,
                        mid,
                        len(documents) - mid,
                    )
                    _add_batch_with_retry(
                        vectorstore,
                        documents[:mid],
                        ids[:mid],
                        max_retries=max_retries,
                        batch_label=f"{batch_label}:left",
                    )
                    _add_batch_with_retry(
                        vectorstore,
                        documents[mid:],
                        ids[mid:],
                        max_retries=max_retries,
                        batch_label=f"{batch_label}:right",
                    )
                    return

                source = documents[0].metadata.get("source_file", "unknown")
                raise RuntimeError(
                    f"Embedding/indexing failed for single chunk from {source}: {exc}"
                ) from exc

            delay = min(2 ** attempt, 8)
            logger.warning(
                "Index batch %s failed on attempt %s/%s (%s: %s); retrying in %ss",
                batch_label,
                attempt + 1,
                max_retries + 1,
                type(exc).__name__,
                exc,
                delay,
            )
            time.sleep(delay)


def build_index(
    documents: list[Document],
    persist_directory: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> Chroma:
    """Create (or update) a ChromaDB collection from *documents*.

    Uses md5 hash of chunk content as the dedup id so repeated runs are safe.

    Args:
        documents: List of LangChain Document objects to index.
        persist_directory: Override for the ChromaDB persistence path.
        embedding_model: Override for the embedding model identifier.

    Returns:
        The populated Chroma vectorstore instance.
    """
    persist_directory = _resolve_persist_dir(persist_directory)
    embedding = _get_embedding(embedding_model)
    _validate_embedding_provider(embedding)

    ids = [_content_id(doc) for doc in documents]

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=persist_directory,
        relevance_score_fn=_l2_to_relevance,
    )
    _add_documents_resilient(
        vectorstore,
        documents,
        ids,
        batch_size=_index_batch_size(),
        max_retries=_index_max_retries(),
    )
    return vectorstore


def load_index(
    persist_directory: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> Chroma:
    """Load an existing ChromaDB collection from disk.

    Args:
        persist_directory: Override for the ChromaDB persistence path.
        embedding_model: Override for the embedding model identifier.

    Returns:
        The loaded Chroma vectorstore instance.
    """
    persist_directory = _resolve_persist_dir(persist_directory)
    embedding = _get_embedding(embedding_model)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=persist_directory,
        relevance_score_fn=_l2_to_relevance,
    )
