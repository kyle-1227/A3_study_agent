"""Tool package exports.

Imports are resolved lazily so lightweight helpers can be used without loading
the full RAG/search stack.
"""

from __future__ import annotations

__all__ = ["rag_retrieve", "get_search_tool", "search"]


def __getattr__(name: str):
    if name == "rag_retrieve":
        from src.tools.rag_tool import rag_retrieve

        return rag_retrieve
    if name in {"get_search_tool", "search"}:
        from src.tools.search_tool import get_search_tool, search

        return {"get_search_tool": get_search_tool, "search": search}[name]
    raise AttributeError(f"module 'src.tools' has no attribute {name!r}")
