"""Trace-only schema drift analysis for structured LLM output.

The analyzer reports field-name drift, extra fields, missing required fields,
enum drift, and input metadata leakage. It never rewrites keys, constructs a
parsed result, or bypasses Pydantic validation.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.llm.schema_manifest import DriftGuardConfig, SchemaManifest


_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;\"'}]+"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;\"'}]+"),
    re.compile(r"(?i)(cookie\s*[:=]\s*)[^;\n\"'}]+"),
    re.compile(r"(?i)(x-api-key\s*[:=]\s*)[^\s,;\"'}]+"),
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]+"),
    re.compile(r"nvapi-[A-Za-z0-9_-]+"),
)


class SchemaDriftReport(BaseModel):
    """Trace-safe report for schema drift diagnostics."""

    model_config = ConfigDict(extra="forbid")

    parsed_ok: bool = False
    schema_name: str = ""
    node_name: str = ""
    top_level_keys: list[str] = Field(default_factory=list)
    expected_top_level_keys: list[str] = Field(default_factory=list)
    extra_fields_by_path: list[str] = Field(default_factory=list)
    missing_required_by_path: list[str] = Field(default_factory=list)
    alias_hits_by_path: dict[str, str] = Field(default_factory=dict)
    enum_drift_by_path: dict[str, dict[str, Any]] = Field(default_factory=dict)
    input_metadata_leak_by_path: list[str] = Field(default_factory=list)
    unknown_keys: list[str] = Field(default_factory=list)
    raw_preview: str = ""


def analyze_schema_drift_trace_only(
    raw_output: str | dict,
    manifest: SchemaManifest,
    drift_guard: DriftGuardConfig,
    node_name: str,
) -> SchemaDriftReport:
    """Analyze schema drift without mutating or normalizing the raw output."""

    data, parsed_ok = _coerce_raw(raw_output)
    report = SchemaDriftReport(
        parsed_ok=parsed_ok,
        schema_name=manifest.schema_name,
        node_name=node_name,
        expected_top_level_keys=list(manifest.top_level_fields),
        raw_preview=_safe_preview(raw_output),
    )
    if isinstance(data, dict):
        report.top_level_keys = sorted(str(key) for key in data.keys())
        expected_children, required_children, enum_by_path = _manifest_indexes(manifest)
        _walk_actual(
            data,
            actual_path="",
            pattern_path="",
            expected_children=expected_children,
            required_children=required_children,
            enum_by_path=enum_by_path,
            drift_guard=drift_guard,
            report=report,
        )
    report.extra_fields_by_path = sorted(set(report.extra_fields_by_path))
    report.missing_required_by_path = sorted(set(report.missing_required_by_path))
    report.input_metadata_leak_by_path = sorted(set(report.input_metadata_leak_by_path))
    report.unknown_keys = sorted(set(report.unknown_keys))
    report.alias_hits_by_path = dict(sorted(report.alias_hits_by_path.items()))
    report.enum_drift_by_path = dict(sorted(report.enum_drift_by_path.items()))
    return report


def drift_report_summary(
    report: SchemaDriftReport, *, max_items: int = 30
) -> dict[str, Any]:
    """Bounded summary for trace payloads and re-ask prompts."""

    return {
        "parsed_ok": report.parsed_ok,
        "schema_name": report.schema_name,
        "node_name": report.node_name,
        "top_level_keys": report.top_level_keys[:max_items],
        "expected_top_level_keys": report.expected_top_level_keys[:max_items],
        "extra_fields_by_path": report.extra_fields_by_path[:max_items],
        "missing_required_by_path": report.missing_required_by_path[:max_items],
        "alias_hits_by_path": dict(list(report.alias_hits_by_path.items())[:max_items]),
        "enum_drift_by_path": dict(list(report.enum_drift_by_path.items())[:max_items]),
        "input_metadata_leak_by_path": report.input_metadata_leak_by_path[:max_items],
        "unknown_keys": report.unknown_keys[:max_items],
        "raw_preview": report.raw_preview,
    }


def render_drift_report_text(
    report: SchemaDriftReport, *, max_chars: int = 1600
) -> str:
    """Render a compact drift report for a correction prompt."""

    payload = drift_report_summary(report, max_items=20)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if len(text) > max_chars:
        suffix = "...[drift report truncated]"
        return text[: max(0, max_chars - len(suffix))] + suffix
    return text


def _coerce_raw(raw_output: str | dict) -> tuple[Any, bool]:
    if isinstance(raw_output, dict):
        return raw_output, True
    text = str(raw_output or "").strip()
    if not text:
        return None, False
    try:
        return json.loads(text), True
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1]), True
            except Exception:
                return None, False
        return None, False


def _safe_preview(raw_output: Any, *, max_chars: int = 1000) -> str:
    if isinstance(raw_output, dict) and "choices" in raw_output:
        return "[provider response body redacted]"
    text = (
        raw_output
        if isinstance(raw_output, str)
        else json.dumps(raw_output, ensure_ascii=False, default=str)
    )
    stripped = str(text or "").strip()
    if '"choices"' in stripped[:300] and '"message"' in stripped[:800]:
        return "[provider response body redacted]"
    stripped = re.sub(r'(?i)("cookie"\s*:\s*")[^"]+(")', r"\1[REDACTED]\2", stripped)
    stripped = re.sub(
        r'(?i)("authorization"\s*:\s*"bearer\s+)[^"]+(")', r"\1[REDACTED]\2", stripped
    )
    stripped = re.sub(
        r'(?i)("api[_-]?key"\s*:\s*")[^"]+(")', r"\1[REDACTED]\2", stripped
    )
    for pattern in _SECRET_PATTERNS:
        if pattern.pattern.startswith("sk-or"):
            stripped = pattern.sub("sk-or-v1-[REDACTED]", stripped)
        elif pattern.pattern.startswith("sk-"):
            stripped = pattern.sub("sk-[REDACTED]", stripped)
        elif pattern.pattern.startswith("nvapi"):
            stripped = pattern.sub("nvapi-[REDACTED]", stripped)
        else:
            stripped = pattern.sub(r"\1[REDACTED]", stripped)
    return stripped[:max_chars] + (
        "...[truncated]" if len(stripped) > max_chars else ""
    )


def _manifest_indexes(
    manifest: SchemaManifest,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, list[str]]]:
    expected_children: dict[str, set[str]] = {}
    required_children: dict[str, set[str]] = {}
    enum_by_path: dict[str, list[str]] = {}

    for item in manifest.fields:
        if item.enum_values:
            enum_by_path[item.path] = list(item.enum_values)
        parent, leaf = _parent_and_leaf(item.path)
        if not leaf:
            continue
        expected_children.setdefault(parent, set()).add(leaf)
        # DeepSeek strict tool calling requires every object property to be
        # emitted explicitly. Pydantic remains the parsed truth, but for prompt
        # repair diagnostics we flag any omitted canonical field.
        required_children.setdefault(parent, set()).add(leaf)
    return expected_children, required_children, enum_by_path


def _parent_and_leaf(path: str) -> tuple[str, str]:
    if not path:
        return "", ""
    if path.endswith("[]"):
        return path[:-2], "[]"
    if "." not in path:
        return "", path
    parent, leaf = path.rsplit(".", 1)
    return parent, leaf


def _walk_actual(
    value: Any,
    *,
    actual_path: str,
    pattern_path: str,
    expected_children: dict[str, set[str]],
    required_children: dict[str, set[str]],
    enum_by_path: dict[str, list[str]],
    drift_guard: DriftGuardConfig,
    report: SchemaDriftReport,
) -> None:
    _check_enum(
        value,
        actual_path=actual_path,
        pattern_path=pattern_path,
        enum_by_path=enum_by_path,
        report=report,
    )
    if isinstance(value, list):
        item_pattern = pattern_path + "[]" if pattern_path else "[]"
        for index, item in enumerate(value):
            item_path = f"{actual_path}[{index}]" if actual_path else f"[{index}]"
            _walk_actual(
                item,
                actual_path=item_path,
                pattern_path=item_pattern,
                expected_children=expected_children,
                required_children=required_children,
                enum_by_path=enum_by_path,
                drift_guard=drift_guard,
                report=report,
            )
        return
    if not isinstance(value, dict):
        return

    expected = expected_children.get(pattern_path, set())
    required = required_children.get(pattern_path, set())
    present = {str(key) for key in value.keys()}
    for missing in sorted(required - present):
        report.missing_required_by_path.append(_join_actual(actual_path, missing))

    aliases_by_alias = _aliases_by_alias(drift_guard)
    forbidden = {field.lower(): field for field in drift_guard.forbidden_output_fields}
    for key, child in value.items():
        key_text = str(key)
        child_actual = _join_actual(actual_path, key_text)
        child_pattern = _join_pattern(pattern_path, key_text)
        if expected and key_text not in expected:
            report.extra_fields_by_path.append(child_actual)
            report.unknown_keys.append(key_text)
            canonical = aliases_by_alias.get(key_text.lower())
            if canonical:
                report.alias_hits_by_path[child_actual] = canonical
            if key_text.lower() in forbidden:
                report.input_metadata_leak_by_path.append(child_actual)
        _walk_actual(
            child,
            actual_path=child_actual,
            pattern_path=child_pattern,
            expected_children=expected_children,
            required_children=required_children,
            enum_by_path=enum_by_path,
            drift_guard=drift_guard,
            report=report,
        )


def _check_enum(
    value: Any,
    *,
    actual_path: str,
    pattern_path: str,
    enum_by_path: dict[str, list[str]],
    report: SchemaDriftReport,
) -> None:
    allowed = enum_by_path.get(pattern_path)
    if not allowed or isinstance(value, (dict, list)):
        return
    if str(value) not in {str(item) for item in allowed}:
        report.enum_drift_by_path[actual_path] = {
            "value": str(value),
            "allowed": allowed,
        }


def _aliases_by_alias(drift_guard: DriftGuardConfig) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for canonical, values in drift_guard.canonical_aliases.items():
        for alias in values:
            aliases[str(alias).lower()] = canonical
    return aliases


def _join_actual(parent: str, key: str) -> str:
    return f"{parent}.{key}" if parent else key


def _join_pattern(parent: str, key: str) -> str:
    return f"{parent}.{key}" if parent else key
