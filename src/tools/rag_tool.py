"""Primary Parent--Child RAG tool factory.

There is intentionally no process-global flat Chroma retriever. A caller must
inject the already verified primary service that it owns.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.tools import BaseTool, tool


class PrimaryRagToolError(RuntimeError):
    """The caller did not provide a valid primary RAG boundary."""


PrimaryRagSearch = Callable[[str, str], dict]


def make_primary_rag_tool(search: PrimaryRagSearch) -> BaseTool:
    """Return a subject-bound tool backed by an injected primary service."""

    if not callable(search):
        raise TypeError("search must be a callable primary RAG boundary")

    @tool("rag_retrieve")
    def rag_retrieve(query: str, subject: str) -> dict:
        """Search one subject through the active Parent--Child primary."""

        if not isinstance(query, str) or not query or query != query.strip():
            raise PrimaryRagToolError("query must be nonblank and stripped")
        if not isinstance(subject, str) or not subject or subject != subject.strip():
            raise PrimaryRagToolError("subject must be nonblank and stripped")
        result = search(query, subject)
        if not isinstance(result, dict):
            raise PrimaryRagToolError("primary search must return a dictionary")
        return result

    return rag_retrieve


__all__ = ["PrimaryRagToolError", "PrimaryRagSearch", "make_primary_rag_tool"]
