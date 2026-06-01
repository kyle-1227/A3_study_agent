"""Utilities for discovering available course subjects from local data."""

from __future__ import annotations

import re
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"


def normalize_subject(value: str) -> str:
    """Normalize a subject identifier returned by an LLM or read from disk."""
    normalized = value.strip().lower()
    normalized = re.sub(r"[\s-]+", "_", normalized)
    normalized = re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fff]", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def get_available_subjects_from_data(data_dir: Path | None = None) -> list[str]:
    """Return stable subject identifiers from first-level non-empty data dirs."""
    root = data_dir or _DEFAULT_DATA_DIR
    if not root.exists() or not root.is_dir():
        return []

    subjects: list[str] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name == "__pycache__":
            continue
        if not any(child.iterdir()):
            continue
        subject = normalize_subject(child.name)
        if subject:
            subjects.append(subject)

    return sorted(set(subjects))
