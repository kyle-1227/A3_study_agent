"""Public import smoke tests for the retained long-term memory layer."""

from __future__ import annotations


def test_retained_memory_public_api_imports_without_legacy_prompt_contract():
    import src.memory as memory

    retained_exports = (
        memory.EpisodicMemoryRecord,
        memory.SemanticMemorySummary,
        memory.MemoryRetrievalResult,
        memory.create_memory_store,
    )

    assert all(retained_exports)
    assert issubclass(memory.SQLiteMemoryStore, memory.MemoryStore)
    assert callable(memory.get_embedding_provider)
    assert hasattr(memory.EmbeddingProvider, "embed")
    assert not hasattr(memory, "MemoryContext" + "Injection")
