"""Canonical fingerprints and exact-content identifiers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import TypeAlias

from src.rag.parent_child.models import PageAwareLoaderConfig, ParentChildPolicy


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Sequence["JsonValue"] | Mapping[str, "JsonValue"]


def canonical_json(value: JsonValue) -> str:
    """Return canonical UTF-8 JSON text used by every V1 identifier."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def sha1_content(content: str) -> str:
    """Hash exact UTF-8 content without whitespace normalization."""

    return hashlib.sha1(content.encode("utf-8")).hexdigest()


def make_policy_fingerprint(policy: ParentChildPolicy) -> str:
    """Return the manifest-derived, strictly validated output policy ID."""

    return policy.policy_id


def make_loader_policy_fingerprint(policy: PageAwareLoaderConfig) -> str:
    """Hash the complete page extraction, cleaning, and assembly policy."""

    payload = policy.model_dump(mode="json")
    if payload["pdf_ocr"] is None:
        # Preserve the established V1 identity when OCR is explicitly absent.
        payload.pop("pdf_ocr")
    else:
        pdf_ocr = dict(payload["pdf_ocr"])
        pdf_ocr.pop("binary_path")
        pdf_ocr.pop("tessdata_dir")
        payload["pdf_ocr"] = pdf_ocr
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def make_parent_id(
    *,
    doc_id: str,
    policy_id: str,
    parent_index: int,
    exact_parent_content_sha1: str,
) -> str:
    """Build a stable parent ID from an unambiguous canonical JSON tuple."""

    payload: JsonValue = [
        "parent_id_v1",
        doc_id,
        policy_id,
        parent_index,
        exact_parent_content_sha1,
    ]
    digest = hashlib.sha1(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"parent_{digest}"


def make_child_id(
    *,
    parent_id: str,
    child_index: int,
    exact_child_content_sha1: str,
) -> str:
    """Build a stable child ID from an unambiguous canonical JSON tuple."""

    payload: JsonValue = [
        "child_id_v1",
        parent_id,
        child_index,
        exact_child_content_sha1,
    ]
    digest = hashlib.sha1(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"child_{digest}"


def make_section_id(
    *,
    doc_id: str,
    section_start_char: int,
    section_title: str,
    section_path: tuple[str, ...],
) -> str:
    """Build an internal stable section ID for parent/child provenance."""

    payload: JsonValue = [
        "section_id_v1",
        doc_id,
        section_start_char,
        section_title,
        list(section_path),
    ]
    digest = hashlib.sha1(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"section_{digest}"
