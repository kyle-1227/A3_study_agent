"""Mindmap JSON tree validation and export helpers."""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

MAX_MINDMAP_DEPTH = 7
MAX_MINDMAP_NODES = 80
DEFAULT_ARTIFACT_DIR = Path(tempfile.gettempdir()) / "a3_study_agent_mindmaps"


def get_mindmap_artifact_dir() -> Path:
    """Return the directory used for generated mindmap files."""
    root = Path(os.getenv("MINDMAP_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR)))
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def sanitize_filename(value: str, default: str = "mindmap") -> str:
    """Create a conservative filename stem from a user/model supplied title."""
    cleaned = re.sub(r"[^\w\u4e00-\u9fff.-]+", "-", value.strip(), flags=re.UNICODE)
    cleaned = cleaned.strip(".-")[:60]
    return cleaned or default


def normalize_mindmap_tree(
    tree: dict[str, Any],
    *,
    max_depth: int = MAX_MINDMAP_DEPTH,
    max_nodes: int = MAX_MINDMAP_NODES,
) -> dict[str, Any]:
    """Validate and trim a mindmap JSON tree into the public shape.

    The output is always a dict containing ``title``, ``children`` and optional
    ``note``. Empty titles are replaced; excessive depth/node counts are
    truncated so downstream renderers cannot produce runaway artifacts.
    """
    counter = {"count": 0}

    def visit(raw: Any, depth: int) -> dict[str, Any] | None:
        if counter["count"] >= max_nodes:
            return None

        if not isinstance(raw, dict):
            raw = {"title": str(raw)}

        title = str(raw.get("title") or "").strip() or "未命名知识点"
        note = str(raw.get("note") or "").strip()
        counter["count"] += 1

        node: dict[str, Any] = {"title": title, "children": []}
        if note:
            node["note"] = note

        if depth >= max_depth:
            return node

        children = raw.get("children") or []
        if isinstance(children, list):
            for child in children:
                normalized = visit(child, depth + 1)
                if normalized is not None:
                    node["children"].append(normalized)
                if counter["count"] >= max_nodes:
                    break

        return node

    return visit(tree, 1) or {"title": "知识点思维导图", "children": []}


def create_xmind_artifact(
    tree: dict[str, Any], title: str | None = None
) -> dict[str, str]:
    """Generate an .xmind file and return public artifact metadata."""
    normalized = normalize_mindmap_tree(tree)
    artifact_id = uuid.uuid4().hex
    filename = f"{sanitize_filename(title or normalized['title'])}.xmind"
    artifact_dir = get_mindmap_artifact_dir() / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    file_path = artifact_dir / filename

    if not _try_export_with_xmind_library(normalized, file_path):
        _export_xmind_zip(normalized, file_path)

    return {
        "artifact_id": artifact_id,
        "filename": filename,
        "path": str(file_path),
        "xmind_url": f"/artifacts/mindmaps/{artifact_id}/{filename}",
    }


def _try_export_with_xmind_library(tree: dict[str, Any], path: Path) -> bool:
    """Use the optional xmind package when it is installed."""
    try:
        import xmind  # type: ignore
    except Exception:
        return False

    try:
        workbook = xmind.load(str(path))
        sheet = workbook.getPrimarySheet()
        sheet.setTitle(tree["title"])
        root = sheet.getRootTopic()
        root.setTitle(tree["title"])
        for child in tree.get("children", []):
            _add_xmind_topic(root, child)
        xmind.save(workbook, path=str(path))
        return True
    except Exception:
        return False


def _add_xmind_topic(parent: Any, node: dict[str, Any]) -> None:
    topic = parent.addSubTopic()
    topic.setTitle(node["title"])
    if node.get("note") and hasattr(topic, "setPlainNotes"):
        topic.setPlainNotes(node["note"])
    for child in node.get("children", []):
        _add_xmind_topic(topic, child)


def _export_xmind_zip(tree: dict[str, Any], path: Path) -> None:
    """Write a minimal XMind-compatible ZIP using only the standard library."""
    content_xml = _build_content_xml(tree)
    metadata = {
        "creator": {"name": "A3 Study Agent"},
        "format": "xmind",
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("content.xml", content_xml)
        zf.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False))
        zf.writestr("META-INF/manifest.xml", _build_manifest_xml())


def _build_content_xml(tree: dict[str, Any]) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<xmap-content xmlns="urn:xmind:xmap:xmlns:content:2.0" version="2.0">'
        f'<sheet id="{uuid.uuid4().hex}"><title>{escape(tree["title"])}</title>'
        f"{_topic_xml(tree)}"
        "</sheet></xmap-content>"
    )


def _topic_xml(node: dict[str, Any]) -> str:
    title = escape(node["title"])
    note = escape(node.get("note", ""))
    notes = f"<notes><plain>{note}</plain></notes>" if note else ""
    children = "".join(_topic_xml(child) for child in node.get("children", []))
    children_xml = (
        f'<children><topics type="attached">{children}</topics></children>'
        if children
        else ""
    )
    return f'<topic id="{uuid.uuid4().hex}"><title>{title}</title>{notes}{children_xml}</topic>'


def _build_manifest_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<manifest xmlns="urn:xmind:xmap:xmlns:manifest:1.0">'
        '<file-entry full-path="content.xml" media-type="text/xml"/>'
        '<file-entry full-path="metadata.json" media-type="application/json"/>'
        "</manifest>"
    )
