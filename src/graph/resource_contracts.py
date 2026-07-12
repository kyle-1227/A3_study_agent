"""Compatibility exports for the low-level canonical resource contracts."""

from __future__ import annotations

from src.resource_contracts import (
    RESOURCE_ALIASES,
    RESOURCE_TYPE_ORDER,
    SUPPORTED_RESOURCE_TYPES,
    ResourceType,
    normalize_requested_resource_types,
    normalize_resource_type,
)


__all__ = [
    "RESOURCE_ALIASES",
    "RESOURCE_TYPE_ORDER",
    "SUPPORTED_RESOURCE_TYPES",
    "ResourceType",
    "normalize_requested_resource_types",
    "normalize_resource_type",
]
