"""Utilities for discovering available course subjects from local data."""

from __future__ import annotations

from functools import lru_cache
import re
from pathlib import Path

from src.config.rag_index_config import CatalogConfig, load_rag_index_config


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"
_PRODUCTION_INDEX_CONFIG_PATH = (
    _PROJECT_ROOT / "config" / "rag" / "index.production-candidate.inactive.yaml"
)


@lru_cache(maxsize=1)
def _production_catalog_config() -> CatalogConfig:
    """Load the strict production corpus discovery policy once per process."""

    return load_rag_index_config(_PRODUCTION_INDEX_CONFIG_PATH).catalog


def _is_excluded_directory(name: str, config: CatalogConfig) -> bool:
    if config.exclude_hidden and name.startswith("."):
        return True
    if config.exclude_cache_directories and name in config.cache_directory_names:
        return True
    if name in config.excluded_exact_names:
        return True
    if any(name.startswith(prefix) for prefix in config.excluded_prefixes):
        return True
    if config.exclude_unclassified and name == config.unclassified_directory_name:
        return True
    return config.exclude_needs_ocr and name == config.needs_ocr_directory_name


def normalize_subject(value: str) -> str:
    """Normalize a subject identifier returned by an LLM or read from disk."""
    normalized = value.strip().lower()
    normalized = re.sub(r"[\s-]+", "_", normalized)
    normalized = re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fff]", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def get_available_subjects_from_data(data_dir: Path | None = None) -> list[str]:
    """Return subjects admitted by the strict production catalog policy."""
    root = data_dir if data_dir is not None else _DEFAULT_DATA_DIR
    if not root.exists() or not root.is_dir():
        return []

    catalog = _production_catalog_config()
    subjects: list[str] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if _is_excluded_directory(child.name, catalog):
            continue
        if not any(child.iterdir()):
            continue
        subject = normalize_subject(child.name)
        if subject:
            subjects.append(subject)

    return sorted(set(subjects))
