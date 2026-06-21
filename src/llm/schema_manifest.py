"""Schema manifest helpers for structured-output prompts and diagnostics.

These helpers derive a compact, stable manifest from the Pydantic schema that
is already used for parsing. The manifest is prompt/debug metadata only; it is
never used as a provider tool schema or as a parsed output schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.config import get_setting


_CONSTRAINT_KEYS = {
    "exclusiveMaximum",
    "exclusiveMinimum",
    "maxItems",
    "maxLength",
    "maximum",
    "minItems",
    "minLength",
    "minimum",
    "multipleOf",
    "pattern",
    "uniqueItems",
}


class StructuredOutputManifestConfig(BaseModel):
    """Runtime manifest injection settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_chars: int = Field(default=6000, ge=200)
    include_descriptions: bool = True
    include_constraints: bool = True
    include_enum_values: bool = True


class DriftGuardConfig(BaseModel):
    """Merged drift guard configuration for one structured-output node."""

    model_config = ConfigDict(extra="forbid")

    node_name: str = ""
    forbidden_output_fields: list[str] = Field(default_factory=list)
    canonical_aliases: dict[str, list[str]] = Field(default_factory=dict)
    drift_guard_source: str = ""
    drift_guard_config_validated: bool = True

    @model_validator(mode="after")
    def validate_lists(self):
        self.forbidden_output_fields = _dedupe_str_list(self.forbidden_output_fields)
        self.canonical_aliases = {
            str(key): _dedupe_str_list(values)
            for key, values in sorted(self.canonical_aliases.items())
        }
        return self


class _DefaultDriftGuardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    global_forbidden_output_fields: list[str] = Field(default_factory=list)
    global_forbidden_aliases: dict[str, list[str]] = Field(default_factory=dict)


class _NodeDriftGuardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forbidden_output_fields: list[str] = Field(default_factory=list)
    canonical_aliases: dict[str, list[str]] = Field(default_factory=dict)


class SchemaFieldManifest(BaseModel):
    """One canonical schema field path."""

    model_config = ConfigDict(extra="forbid")

    path: str
    field_type: str = ""
    required: bool = False
    enum_values: list[str] = Field(default_factory=list)
    description: str = ""
    constraints: dict[str, Any] = Field(default_factory=dict)
    default_kind: str = ""
    forbidden_aliases: list[str] = Field(default_factory=list)


class SchemaManifest(BaseModel):
    """Canonical prompt/debug manifest for a Pydantic output schema."""

    model_config = ConfigDict(extra="forbid")

    schema_name: str
    node_name: str = ""
    output_mode: str = ""
    top_level_fields: list[str] = Field(default_factory=list)
    fields: list[SchemaFieldManifest] = Field(default_factory=list)
    forbidden_output_fields: list[str] = Field(default_factory=list)
    canonical_aliases: dict[str, list[str]] = Field(default_factory=dict)
    drift_guard_source: str = ""
    drift_guard_config_validated: bool = True


@dataclass(frozen=True)
class _ResolvedNode:
    node: dict[str, Any]
    ref_name: str


def _dedupe_str_list(values: Any) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise TypeError("expected list[str]")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError("expected list[str]")
        item = value.strip()
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _merge_aliases(
    base: dict[str, list[str]],
    override: dict[str, list[str]],
) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for source in (base, override):
        if not isinstance(source, dict):
            raise TypeError("expected dict[str, list[str]]")
        for key, values in source.items():
            if not isinstance(key, str):
                raise TypeError("expected dict[str, list[str]]")
            merged[key] = _dedupe_str_list(
                [*(merged.get(key) or []), *_dedupe_str_list(values)]
            )
    return dict(sorted(merged.items()))


def _structured_output_section(config: object | None = None) -> dict[str, Any]:
    if config is not None:
        if not isinstance(config, dict):
            raise TypeError("structured_output config must be a mapping")
        return config
    value = get_setting("structured_output", {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("structured_output must be a mapping")
    return value


def get_structured_output_manifest_config(
    config: object | None = None,
) -> StructuredOutputManifestConfig:
    """Load and validate runtime manifest injection settings."""

    section = _structured_output_section(config)
    raw = section.get("manifest", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("structured_output.manifest must be a mapping")
    return StructuredOutputManifestConfig.model_validate(raw)


def load_drift_guard_config(
    node_name: str | None,
    config: object | None = None,
) -> DriftGuardConfig:
    """Load, validate, and merge default + node-specific drift guard config."""

    section = _structured_output_section(config)
    raw_guards = section.get("drift_guards", {})
    if raw_guards is None:
        raw_guards = {}
    if not isinstance(raw_guards, dict):
        raise TypeError("structured_output.drift_guards must be a mapping")

    default_raw = raw_guards.get("default", {}) or {}
    node_key = str(node_name or "")
    node_raw = raw_guards.get(node_key, {}) or {}
    default_cfg = _DefaultDriftGuardConfig.model_validate(default_raw)
    node_cfg = _NodeDriftGuardConfig.model_validate(node_raw)

    forbidden = _dedupe_str_list(
        [
            *default_cfg.global_forbidden_output_fields,
            *node_cfg.forbidden_output_fields,
        ]
    )
    aliases = _merge_aliases(
        default_cfg.global_forbidden_aliases, node_cfg.canonical_aliases
    )
    source = "default"
    if node_key and node_key in raw_guards:
        source = f"default+{node_key}"
    return DriftGuardConfig(
        node_name=node_key,
        forbidden_output_fields=forbidden,
        canonical_aliases=aliases,
        drift_guard_source=source,
        drift_guard_config_validated=True,
    )


def build_canonical_manifest(
    schema: type[BaseModel],
    node_name: str | None = None,
    output_mode: str | None = None,
    config: object | None = None,
) -> SchemaManifest:
    """Build a stable manifest from the Pydantic schema."""

    drift_guard = load_drift_guard_config(node_name, config=config)
    raw_schema = schema.model_json_schema()
    if not isinstance(raw_schema, dict):
        raise TypeError("Pydantic schema must be a JSON object")

    fields: list[SchemaFieldManifest] = []
    top_level_fields = sorted((raw_schema.get("properties") or {}).keys())
    _walk_schema(
        raw_schema,
        root_schema=raw_schema,
        path="",
        required_names=set(raw_schema.get("required") or []),
        fields=fields,
        drift_guard=drift_guard,
        seen_refs=set(),
    )

    unique: dict[str, SchemaFieldManifest] = {}
    for field_manifest in fields:
        if field_manifest.path:
            unique[field_manifest.path] = field_manifest

    return SchemaManifest(
        schema_name=schema.__name__,
        node_name=str(node_name or ""),
        output_mode=str(output_mode or ""),
        top_level_fields=top_level_fields,
        fields=[unique[path] for path in sorted(unique)],
        forbidden_output_fields=drift_guard.forbidden_output_fields,
        canonical_aliases=drift_guard.canonical_aliases,
        drift_guard_source=drift_guard.drift_guard_source,
        drift_guard_config_validated=drift_guard.drift_guard_config_validated,
    )


def render_manifest_text(
    manifest: SchemaManifest,
    max_chars: int | None = None,
    include_descriptions: bool = True,
    include_constraints: bool = True,
    include_enum_values: bool = True,
) -> str:
    """Render a compact, deterministic text manifest for prompt injection."""

    lines = [
        f"Canonical schema manifest: {manifest.schema_name}",
        f"Node: {manifest.node_name or '<unknown>'}",
        f"Output mode: {manifest.output_mode or '<unknown>'}",
        "Use canonical field names exactly. Do not output aliases, translations, abbreviations, or wrapper keys.",
        "Do not output fields that are not listed in this manifest.",
        "Do not omit required fields. If a field is optional/defaulted, still emit a schema-compatible empty value when strict tool calling requires it.",
    ]
    if manifest.top_level_fields:
        lines.append(f"Top-level fields: {', '.join(manifest.top_level_fields)}")
    if manifest.forbidden_output_fields:
        lines.append(
            f"Forbidden output fields: {', '.join(manifest.forbidden_output_fields)}"
        )
    if manifest.canonical_aliases:
        alias_lines = []
        for canonical, aliases in sorted(manifest.canonical_aliases.items()):
            if aliases:
                alias_lines.append(f"{canonical} <- {', '.join(aliases)}")
        if alias_lines:
            lines.append("Forbidden aliases: " + "; ".join(alias_lines))
    lines.append("Fields:")
    for item in manifest.fields:
        line = f"- {item.path}: type={item.field_type or 'unknown'}; required={str(item.required).lower()}"
        if include_enum_values and item.enum_values:
            line += f"; enum={json.dumps(item.enum_values, ensure_ascii=False)}"
        if include_constraints and item.constraints:
            line += f"; constraints={json.dumps(item.constraints, ensure_ascii=False, sort_keys=True)}"
        if item.forbidden_aliases:
            line += f"; aliases_forbidden={json.dumps(item.forbidden_aliases, ensure_ascii=False)}"
        if include_descriptions and item.description:
            line += f"; description={item.description}"
        lines.append(line)
    rendered = "\n".join(lines)
    if max_chars is not None and len(rendered) > max_chars:
        suffix = "\n[manifest truncated]"
        return rendered[: max(0, max_chars - len(suffix))].rstrip() + suffix
    return rendered


def _walk_schema(
    node: dict[str, Any],
    *,
    root_schema: dict[str, Any],
    path: str,
    required_names: set[str],
    fields: list[SchemaFieldManifest],
    drift_guard: DriftGuardConfig,
    seen_refs: set[str],
) -> None:
    resolved = _resolve_node(node, root_schema=root_schema)
    node = resolved.node
    if resolved.ref_name:
        if resolved.ref_name in seen_refs:
            return
        seen_refs = {*seen_refs, resolved.ref_name}

    node_type = _field_type(node, root_schema=root_schema)
    if path:
        leaf = _leaf_name(path)
        fields.append(
            SchemaFieldManifest(
                path=path,
                field_type=node_type,
                required=leaf in required_names,
                enum_values=_enum_values(node),
                description=str(node.get("description") or ""),
                constraints=_constraints(node),
                default_kind=_default_kind(node),
                forbidden_aliases=drift_guard.canonical_aliases.get(leaf, []),
            )
        )

    for union_key in ("anyOf", "oneOf", "allOf"):
        variants = node.get(union_key)
        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, dict) and not _is_null_schema(variant):
                    _walk_schema(
                        variant,
                        root_schema=root_schema,
                        path=path,
                        required_names=required_names,
                        fields=fields,
                        drift_guard=drift_guard,
                        seen_refs=seen_refs,
                    )

    properties = node.get("properties")
    if isinstance(properties, dict):
        child_required = set(node.get("required") or [])
        for name in sorted(properties):
            child_path = f"{path}.{name}" if path else str(name)
            child = properties[name]
            if isinstance(child, dict):
                _walk_schema(
                    child,
                    root_schema=root_schema,
                    path=child_path,
                    required_names=child_required,
                    fields=fields,
                    drift_guard=drift_guard,
                    seen_refs=seen_refs,
                )

    items = node.get("items")
    if isinstance(items, dict):
        item_path = f"{path}[]" if path else "[]"
        _walk_schema(
            items,
            root_schema=root_schema,
            path=item_path,
            required_names=set(),
            fields=fields,
            drift_guard=drift_guard,
            seen_refs=seen_refs,
        )


def _resolve_node(
    node: dict[str, Any], *, root_schema: dict[str, Any]
) -> _ResolvedNode:
    ref = node.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
        return _ResolvedNode(node=node, ref_name="")
    name = ref.rsplit("/", 1)[-1]
    defs = root_schema.get("$defs") or {}
    target = defs.get(name)
    if not isinstance(target, dict):
        return _ResolvedNode(node=node, ref_name=name)
    merged = {**target, **{key: value for key, value in node.items() if key != "$ref"}}
    return _ResolvedNode(node=merged, ref_name=name)


def _field_type(node: dict[str, Any], *, root_schema: dict[str, Any]) -> str:
    if "$ref" in node:
        return str(node["$ref"]).rsplit("/", 1)[-1]
    node_type = node.get("type")
    if isinstance(node_type, list):
        return "|".join(str(item) for item in node_type)
    if isinstance(node_type, str):
        if node_type == "array" and isinstance(node.get("items"), dict):
            item = _resolve_node(node["items"], root_schema=root_schema).node
            return f"array[{_field_type(item, root_schema=root_schema)}]"
        return node_type
    for key in ("anyOf", "oneOf", "allOf"):
        variants = node.get(key)
        if isinstance(variants, list):
            names = [
                _field_type(
                    _resolve_node(variant, root_schema=root_schema).node,
                    root_schema=root_schema,
                )
                for variant in variants
                if isinstance(variant, dict) and not _is_null_schema(variant)
            ]
            return f"{key}[{', '.join(names)}]"
    if "enum" in node or "const" in node:
        return "enum"
    return ""


def _is_null_schema(node: dict[str, Any]) -> bool:
    return node.get("type") == "null"


def _enum_values(node: dict[str, Any]) -> list[str]:
    if isinstance(node.get("enum"), list):
        return [str(item) for item in node["enum"]]
    if "const" in node:
        return [str(node["const"])]
    values: list[str] = []
    for key in ("anyOf", "oneOf"):
        variants = node.get(key)
        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, dict):
                    values.extend(_enum_values(variant))
    return _dedupe_str_list(values)


def _constraints(node: dict[str, Any]) -> dict[str, Any]:
    return {key: node[key] for key in sorted(_CONSTRAINT_KEYS) if key in node}


def _default_kind(node: dict[str, Any]) -> str:
    if "default" not in node:
        return ""
    value = node.get("default")
    if value is None:
        return "none"
    if isinstance(value, (str, int, float, bool, list, dict)):
        return type(value).__name__
    if isinstance(value, Enum):
        return "enum"
    return "value"


def _leaf_name(path: str) -> str:
    cleaned = path.replace("[]", "")
    if "." not in cleaned:
        return cleaned
    return cleaned.rsplit(".", 1)[-1]


def manifest_summary(
    manifest: SchemaManifest, *, max_fields: int = 40
) -> dict[str, Any]:
    """Small trace-safe manifest summary."""

    return {
        "schema_name": manifest.schema_name,
        "node_name": manifest.node_name,
        "output_mode": manifest.output_mode,
        "top_level_fields": manifest.top_level_fields,
        "field_paths": [field.path for field in manifest.fields[:max_fields]],
        "field_count": len(manifest.fields),
        "forbidden_output_fields": manifest.forbidden_output_fields,
        "canonical_aliases": manifest.canonical_aliases,
        "drift_guard_source": manifest.drift_guard_source,
        "drift_guard_config_validated": manifest.drift_guard_config_validated,
    }


def config_error_message(exc: Exception) -> str:
    """Human-readable config error with Pydantic details when available."""

    if isinstance(exc, ValidationError):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"
