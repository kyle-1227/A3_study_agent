"""Course-data package.

The retired flat Chroma indexer and retriever are intentionally not exported.
The served graph receives an injected Parent--Child primary runtime instead.
"""

from src.rag.loader import load_documents

__all__ = ["load_documents"]
