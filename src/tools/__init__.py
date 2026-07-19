"""Tool package exports for injected production boundaries."""

from __future__ import annotations

__all__ = ["make_primary_rag_tool", "search"]


def __getattr__(name: str):
    if name == "make_primary_rag_tool":
        from src.tools.rag_tool import make_primary_rag_tool

        return make_primary_rag_tool
    if name == "search":
        from src.tools.search_tool import search

        return search
    raise AttributeError(f"module 'src.tools' has no attribute {name!r}")
