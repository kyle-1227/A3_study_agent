"""OpenAI-compatible reranker API wrapper.

Calls the configured reranking endpoint to re-score candidate documents
against a query.  On any API failure the original documents are returned in
their existing order (graceful degradation).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from src.config import get_setting

logger = logging.getLogger(__name__)

_DEFAULT_RERANKER_BASE_URL = "https://api.siliconflow.cn/v1"
_DEFAULT_RERANKER_API_KEY_ENV = "SILICONFLOW_API_KEY"
_DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_TIMEOUT = 15  # seconds


def _reranker_api_key_env() -> str:
    return os.getenv("RERANKER_API_KEY_ENV", _DEFAULT_RERANKER_API_KEY_ENV)


def _reranker_api_key() -> str | None:
    api_key_env = _reranker_api_key_env()
    api_key = os.getenv(api_key_env)
    if api_key:
        return api_key

    if api_key_env.startswith(("sk-", "or-")):
        logger.warning(
            "RERANKER_API_KEY_ENV appears to contain an API key value instead "
            "of an environment variable name. Prefer "
            "RERANKER_API_KEY_ENV=SILICONFLOW_API_KEY."
        )
        return api_key_env

    return None


def _reranker_base_url() -> str:
    return os.getenv(
        "RERANKER_BASE_URL",
        get_setting("rag.reranker_base_url", _DEFAULT_RERANKER_BASE_URL),
    )


def _rerank_url() -> str:
    explicit_url = os.getenv("RERANKER_URL", get_setting("rag.reranker_url", ""))
    if explicit_url:
        return explicit_url
    return _reranker_base_url().rstrip("/") + "/rerank"


def rerank(
    query: str,
    documents: list[dict[str, Any]],
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """Rerank *documents* against *query* via the configured reranker.

    Parameters
    ----------
    query : str
        The user's search query.
    documents : list[dict]
        Candidate documents.  Each dict **must** have a ``"content"`` key.
    top_n : int, optional
        Number of results to return.  Defaults to ``rag.reranker_top_n``
        from settings (fallback: 5).

    Returns
    -------
    list[dict]
        The *top_n* documents sorted by reranker relevance, each with an
        added ``"rerank_score"`` key.  On failure, returns *documents*
        truncated to *top_n* in original order.
    """
    if not documents:
        return []

    if top_n is None:
        top_n = get_setting("rag.reranker_top_n", 5)

    api_key_env = _reranker_api_key_env()
    api_key = _reranker_api_key()
    model = os.getenv(
        "RERANKER_MODEL",
        get_setting("rag.reranker_model", _DEFAULT_RERANKER_MODEL),
    )

    doc_texts = [d["content"] for d in documents]

    try:
        resp = httpx.post(
            _rerank_url(),
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "query": query,
                "documents": doc_texts,
                "top_n": top_n,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.warning(
            "Reranker API call failed; returning original order "
            "(model=%s, api_key_env=%s, base_url=%s)",
            model,
            api_key_env,
            _reranker_base_url(),
            exc_info=True,
        )
        return documents[:top_n]

    results: list[dict[str, Any]] = data.get("results", [])
    ranked: list[dict[str, Any]] = []
    for item in results:
        idx = item["index"]
        if 0 <= idx < len(documents):
            doc = {**documents[idx], "rerank_score": item["relevance_score"]}
            ranked.append(doc)

    return ranked[:top_n]
