"""Strict parent-child RAG contracts and immutable generation primitives."""

from src.rag.parent_child.loader import load_cleaned_source
from src.rag.parent_child.models import (
    ChildDocument,
    ChildMetadata,
    CleanedSourceDocument,
    PageAwareLoaderConfig,
    PageSpan,
    ParentChildBundle,
    ParentChildPolicy,
    ParentRecord,
    SourceEntry,
    SourcePage,
)
from src.rag.parent_child.splitter import build_parent_child_bundle

__all__ = [
    "ChildDocument",
    "ChildMetadata",
    "CleanedSourceDocument",
    "PageAwareLoaderConfig",
    "PageSpan",
    "ParentChildBundle",
    "ParentChildPolicy",
    "ParentRecord",
    "SourceEntry",
    "SourcePage",
    "build_parent_child_bundle",
    "load_cleaned_source",
]
