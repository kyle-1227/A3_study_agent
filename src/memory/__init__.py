"""
Memory Layer — Long-term memory system for the A3 Study Agent.

Components:
- Episodic Memory: atomic learning events (quiz attempts, errors, behaviors)
- Semantic Memory: LLM-generated summaries aggregating N episodic events
- Retrieval: hybrid BM25 keyword + vector similarity search
- Consolidation: episodic → semantic compression + forgetting mechanism
- Embeddings: pluggable vector embedding providers

Usage::

    from src.memory import (
        create_memory_store, SQLiteMemoryStore,
        get_embedding_provider, DeepSeekEmbeddingProvider,
        write_episodic_memory, retrieve_top_k_memories,
        consolidate_episodic_to_semantic, maybe_consolidate, apply_forgetting,
    )
"""

from src.memory.schema import (
    EpisodicMemoryRecord,
    MemoryContextInjection,
    MemoryRetrievalResult,
    SemanticMemorySummary,
    SemanticSummaryStrictOutput,
)
from src.memory.storage import (
    MemoryStore,
    SQLiteMemoryStore,
    create_memory_store,
)
from src.memory.embeddings import (
    DeepSeekEmbeddingProvider,
    DummyEmbeddingProvider,
    EmbeddingProvider,
    get_embedding_provider,
    reset_embedding_provider,
)

__all__ = [
    # Schema
    "EpisodicMemoryRecord",
    "MemoryContextInjection",
    "MemoryRetrievalResult",
    "SemanticMemorySummary",
    "SemanticSummaryStrictOutput",
    # Storage
    "MemoryStore",
    "SQLiteMemoryStore",
    "create_memory_store",
    # Embeddings
    "DeepSeekEmbeddingProvider",
    "DummyEmbeddingProvider",
    "EmbeddingProvider",
    "get_embedding_provider",
    "reset_embedding_provider",
]
