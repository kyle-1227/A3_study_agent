"""
Memory Retrieval — hybrid BM25 keyword + embedding vector similarity search.

Follows the pattern in src/rag/retriever.py:
- Jieba tokenization for Chinese text
- BM25Okapi scoring for keyword matching
- Cosine similarity for vector matching
- Weighted combination with importance boost

The BM25 corpus is rebuilt lazily on each call (O(N) where N is the user's
episodic + semantic memory count, typically < 200). This avoids the need for
a persistent BM25 index.
"""

from __future__ import annotations

import logging
import math
from collections import Counter

import jieba

from src.config import get_setting
from src.memory.schema import (
    EpisodicMemoryRecord,
    MemoryRetrievalResult,
    SemanticMemorySummary,
)
from src.memory.storage import MemoryStore, create_memory_store
from src.memory.embeddings import EmbeddingProvider, get_embedding_provider
from src.memory.errors import MemoryEmbeddingRuntimeError

logger = logging.getLogger(__name__)


# ── Cosine Similarity ─────────────────────────────────────────────────────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 if either vector is empty, zero-length, or of different length.
    """
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── BM25 Scoring ──────────────────────────────────────────────────────────


def _bm25_score_single(
    query_terms: list[str],
    doc_terms: list[str],
    corpus: list[list[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Compute BM25 score for a single document against a query.

    Uses the full corpus for IDF computation. This is a simplified BM25
    implementation — it computes IDF from the in-memory corpus at call time.

    Args:
        query_terms: Tokenized query.
        doc_terms: Tokenized document.
        corpus: Full tokenized corpus (for IDF + avg doc length).
        k1: Term frequency saturation parameter.
        b: Length normalization parameter.

    Returns:
        BM25 score (non-negative, typically 0–20).
    """
    if not query_terms or not doc_terms:
        return 0.0

    n_docs = len(corpus)
    if n_docs == 0:
        return 0.0

    doc_len = len(doc_terms)
    avg_doc_len = sum(len(d) for d in corpus) / n_docs if n_docs > 0 else 1.0
    if avg_doc_len == 0:
        avg_doc_len = 1.0

    # Term frequency in document
    doc_tf = Counter(doc_terms)

    # Document frequency across corpus
    df: Counter[str] = Counter()
    for c in corpus:
        unique_terms = set(c)
        for t in unique_terms:
            df[t] += 1

    score = 0.0
    for term in query_terms:
        tf = doc_tf.get(term, 0)
        if tf == 0:
            continue
        doc_freq = df.get(term, 0)
        if doc_freq == 0:
            continue

        # IDF (smoothed)
        idf = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

        # BM25 term score
        numerator = tf * (k1 + 1.0)
        denominator = tf + k1 * (1.0 - b + b * doc_len / avg_doc_len)
        if denominator == 0:
            continue
        score += idf * numerator / denominator

    return score


# ── Primary API ────────────────────────────────────────────────────────────


async def retrieve_top_k_memories(
    user_id: str,
    query: str,
    *,
    top_k: int = 5,
    keyword_weight: float | None = None,
    vector_weight: float | None = None,
    importance_boost: float | None = None,
    include_episodic: bool = True,
    include_semantic: bool = True,
    store: MemoryStore | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[MemoryRetrievalResult]:
    """Hybrid retrieval combining BM25 keyword + embedding vector search.

    1. Fetches all episodic + semantic memories for the user from the store.
    2. Gets a query embedding from the embedding provider.
    3. Builds a BM25 corpus from all candidate texts (jieba tokenized).
    4. Scores each candidate: weighted keyword + vector + importance boost.
    5. Returns top-K sorted by combined score.

    Args:
        user_id: The user to retrieve memories for.
        query: The current query text to match against.
        top_k: Max number of results to return.
        keyword_weight: Weight for BM25 score (0.0–1.0). Default from settings.
        vector_weight: Weight for cosine similarity score (0.0–1.0). Default from settings.
        importance_boost: Bonus added for high-importance memories. Default from settings.
        include_episodic: Whether to include episodic memories.
        include_semantic: Whether to include semantic summaries.
        store: Optional MemoryStore (uses singleton if not provided).
        embedding_provider: Optional explicit provider supplied by callers that
            already constructed one.

    Returns:
        List of MemoryRetrievalResult sorted by score descending, limited to top_k.
    """
    store = store or create_memory_store()

    # Read settings
    if keyword_weight is None:
        keyword_weight = float(get_setting("memory.retrieval.keyword_weight", 0.4))
    if vector_weight is None:
        vector_weight = float(get_setting("memory.retrieval.vector_weight", 0.6))
    if importance_boost is None:
        importance_boost = float(get_setting("memory.retrieval.importance_boost", 0.1))

    # 1. Fetch candidates
    episodic_records: list[EpisodicMemoryRecord] = []
    semantic_summaries: list[SemanticMemorySummary] = []

    if include_episodic:
        episodic_records = await store.get_all_episodic_for_user(user_id, limit=200)

    if include_semantic:
        semantic_summaries = await store.get_semantic_for_user(user_id, limit=20)

    if not episodic_records and not semantic_summaries:
        logger.debug("No memories found for user=%s", user_id)
        return []

    # 2. Get query embedding
    provider = embedding_provider or get_embedding_provider()
    embeddings = await provider.embed([query])
    if not embeddings or not embeddings[0]:
        raise MemoryEmbeddingRuntimeError(
            "Embedding provider returned no query embedding"
        )
    query_embedding = embeddings[0]

    # 3. Build candidate list and BM25 corpus
    # Each candidate is (doc_id, record, doc_type_label)
    candidates: list[tuple[str, EpisodicMemoryRecord | SemanticMemorySummary, str]] = []
    for rec in episodic_records:
        candidates.append((rec.memory_id, rec, "episodic"))
    for rec in semantic_summaries:
        candidates.append((rec.summary_id, rec, "semantic"))

    # Tokenize all candidate texts for BM25
    texts = [_get_record_content(c[1]) for c in candidates]
    tokenized_corpus = [jieba.lcut(t.lower()) for t in texts]

    # Tokenize query
    query_terms = jieba.lcut(query.lower())

    # 4. Score each candidate
    results: list[MemoryRetrievalResult] = []
    for idx, (doc_id, record, doc_type) in enumerate(candidates):
        # Keyword score (BM25)
        kw_score = _bm25_score_single(
            query_terms, tokenized_corpus[idx], tokenized_corpus
        )

        # Vector score (cosine similarity)
        vec_score = 0.0
        if query_embedding and record.embedding:
            vec_score = _cosine_similarity(query_embedding, record.embedding)

        # Normalize BM25 to [0, 1] range for combination with vector score
        # BM25 scores can be unbounded; we use tanh normalization
        kw_norm = math.tanh(kw_score / _expected_max_bm25(tokenized_corpus))

        # Combined score
        combined = keyword_weight * kw_norm + vector_weight * vec_score

        # Importance boost (small bonus for important memories)
        importance = getattr(record, "importance", 0.5)
        combined += importance_boost * importance

        # Determine match reason
        reasons: list[str] = []
        if kw_norm > 0.1:
            reasons.append("keyword_overlap")
        if vec_score > 0.3:
            reasons.append("vector_similarity")
        if importance > 0.7:
            reasons.append("high_importance")
        if not reasons:
            reasons.append("low_signal_history")

        results.append(
            MemoryRetrievalResult(
                memory=record,
                memory_type=doc_type,
                score=min(combined, 1.0),
                keyword_score=kw_norm,
                vector_score=vec_score,
                match_reason="+".join(reasons),
            )
        )

    # 5. Sort and return top-K
    results.sort(key=lambda r: r.score, reverse=True)

    top_results = results[:top_k]

    if get_setting("memory.log_memory_retrieval", True):
        logger.debug(
            "Memory retrieval for user=%s: %d candidates → %d results (top score=%.3f)",
            user_id,
            len(candidates),
            len(top_results),
            top_results[0].score if top_results else 0.0,
        )

    return top_results


# ── Helpers ────────────────────────────────────────────────────────────────


def _get_record_content(record: EpisodicMemoryRecord | SemanticMemorySummary) -> str:
    """Extract the primary text content from a memory record."""
    if isinstance(record, EpisodicMemoryRecord):
        return record.content
    # SemanticMemorySummary
    parts = [record.content]
    if record.weak_knowledge_points:
        parts.append("薄弱点: " + ", ".join(record.weak_knowledge_points))
    if record.learning_style_changes:
        parts.append("学习风格变化: " + record.learning_style_changes)
    if record.skill_growth_trajectory:
        parts.append("技能成长: " + record.skill_growth_trajectory)
    return " ".join(parts)


def _expected_max_bm25(tokenized_corpus: list[list[str]]) -> float:
    """Estimate a reasonable max BM25 score for normalization.

    Uses the average document length as a heuristic upper bound to prevent
    tanh normalization from always saturating at 1.0.
    """
    if not tokenized_corpus:
        return 1.0
    avg_len = sum(len(d) for d in tokenized_corpus) / len(tokenized_corpus)
    # A reasonable upper bound: ~avg_len * log(N) where N = corpus size
    return max(avg_len * math.log(len(tokenized_corpus) + 1), 1.0)
