"""Strict domain contracts for page-aware parent-child chunking."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_SHA1_PATTERN = r"^[0-9a-f]{40}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_DOC_ID_PATTERN = r"^doc_[0-9a-f]{40}$"
_PARENT_ID_PATTERN = r"^parent_[0-9a-f]{40}$"
_CHILD_ID_PATTERN = r"^child_[0-9a-f]{40}$"
_SECTION_ID_PATTERN = r"^section_[0-9a-f]{40}$"
_IDENTIFIER_RE = re.compile(r"^[^/\\\x00]+$")


def _sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _validate_nonempty_identifier(value: str, *, field_name: str) -> str:
    if value != value.strip() or not value:
        raise ValueError(f"{field_name} must be non-empty and already stripped")
    if _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must not contain path separators or NUL")
    return value


def _validate_source_file(value: str) -> str:
    _validate_nonempty_identifier(value, field_name="source_file")
    if value in {".", ".."}:
        raise ValueError("source_file must be a file name")
    return value


def _validate_source_relpath(value: str) -> str:
    if value != value.strip() or not value:
        raise ValueError("source_relpath must be non-empty and already stripped")
    if "\\" in value or "\x00" in value:
        raise ValueError("source_relpath must use POSIX separators and contain no NUL")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError("source_relpath must not contain empty, dot, or parent parts")
    path = PurePosixPath(value)
    if path.is_absolute() or raw_parts[0].endswith(":"):
        raise ValueError("source_relpath must be relative to the configured data root")
    return path.as_posix()


def _validate_section_path(value: tuple[str, ...]) -> tuple[str, ...]:
    for item in value:
        if not item or item != item.strip():
            raise ValueError(
                "section_path items must be non-empty and already stripped"
            )
    return value


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class SourceEntry(_StrictFrozenModel):
    """Explicit source and containment boundary passed to the loader."""

    schema_version: Literal["source_entry_v1"]
    source_path: Path
    data_root: Path
    subject: str
    doc_type: str

    @field_validator("subject", "doc_type")
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "identifier")
        return _validate_nonempty_identifier(value, field_name=field_name)


class TesseractLanguageAsset(_StrictFrozenModel):
    """One immutable language model identity in configured OCR order."""

    language: str
    traineddata_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        if re.fullmatch(r"[a-z0-9_]+", value) is None:
            raise ValueError("OCR language must be a lower-case identifier")
        return value


class TesseractOcrRuntimeConfig(_StrictFrozenModel):
    """Resolved runtime identity for exact-source Tesseract extraction."""

    schema_version: Literal["tesseract_ocr_runtime_v1"]
    engine_protocol: Literal["tesseract_cli_tsv_v1"]
    extraction_method: Literal["tesseract_cli_tsv_v1"]
    source_relpaths: tuple[str, ...] = Field(min_length=1)
    binary_path: Path
    binary_sha256: str = Field(pattern=_SHA256_PATTERN)
    runtime_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_version: str
    renderer_protocol: Literal["pymupdf_pixmap_png_v1"]
    pymupdf_version: str
    mupdf_version: str
    tessdata_dir: Path
    language_assets: tuple[TesseractLanguageAsset, ...] = Field(min_length=1)
    dpi: int = Field(ge=72, le=600)
    render_colorspace: Literal["rgb"]
    render_alpha: Literal[False]
    render_annotations: Literal[True]
    oem: Literal[1]
    psm: Literal[3]
    thread_limit: Literal[1]
    timeout_seconds: float = Field(gt=0.0, le=300.0)
    output_format: Literal["tsv_lines_v1"]
    empty_page_policy: Literal["allow_empty_physical_page_v1"]

    @field_validator(
        "extraction_method",
        "expected_version",
        "pymupdf_version",
        "mupdf_version",
    )
    @classmethod
    def validate_labels(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "OCR label")
        return _validate_nonempty_identifier(value, field_name=field_name)

    @field_validator("source_relpaths")
    @classmethod
    def validate_source_relpaths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(_validate_source_relpath(item) for item in value)
        if any(PurePosixPath(item).suffix.casefold() != ".pdf" for item in validated):
            raise ValueError("OCR source_relpaths must identify PDF sources")
        if len(set(validated)) != len(validated):
            raise ValueError("OCR source_relpaths must not contain duplicates")
        if validated != tuple(sorted(validated)):
            raise ValueError("OCR source_relpaths must be sorted")
        return validated

    @field_validator("language_assets")
    @classmethod
    def validate_language_assets(
        cls, value: tuple[TesseractLanguageAsset, ...]
    ) -> tuple[TesseractLanguageAsset, ...]:
        languages = tuple(asset.language for asset in value)
        if len(set(languages)) != len(languages):
            raise ValueError("OCR languages must not contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_runtime_inventory(self) -> "TesseractOcrRuntimeConfig":
        if not self.binary_path.is_absolute() or not self.tessdata_dir.is_absolute():
            raise ValueError("OCR runtime paths must be resolved absolute paths")
        return self


class PageAwareLoaderConfig(_StrictFrozenModel):
    """Complete, explicit policy for extraction, cleaning, and page assembly."""

    schema_version: Literal["page_aware_loader_policy_v1"]
    extraction_algorithm_version: str
    page_assembly_algorithm_version: str
    cleaning_algorithm_version: Literal["page_clean_v2"]
    cleaning_policy_id: str
    nul_character_policy: Literal["replace_with_space_v1", "reject"]
    page_separator: str
    normalize_newlines: bool
    strip_trailing_whitespace: bool
    strip_outer_blank_lines: bool
    header_top_lines: int = Field(gt=0)
    footer_bottom_lines: int = Field(gt=0)
    repeated_line_min_pages: int = Field(gt=0)
    repeated_line_min_ratio: float = Field(gt=0.0, le=1.0)
    collapse_blank_lines: bool
    paragraph_deduplication: bool
    supported_extensions: tuple[str, ...] = Field(min_length=1)
    pdf_extraction_method: str
    text_extraction_method: str
    pdf_ocr: TesseractOcrRuntimeConfig | None

    @field_validator(
        "extraction_algorithm_version",
        "page_assembly_algorithm_version",
        "cleaning_algorithm_version",
        "cleaning_policy_id",
        "pdf_extraction_method",
        "text_extraction_method",
    )
    @classmethod
    def validate_policy_labels(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "policy_label")
        return _validate_nonempty_identifier(value, field_name=field_name)

    @field_validator("page_separator")
    @classmethod
    def validate_page_separator(cls, value: str) -> str:
        if not value:
            raise ValueError("page_separator must be non-empty")
        return value

    @field_validator("supported_extensions")
    @classmethod
    def validate_extensions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for extension in value:
            if (
                not extension.startswith(".")
                or extension != extension.lower()
                or extension != extension.strip()
                or "/" in extension
                or "\\" in extension
            ):
                raise ValueError(
                    "supported_extensions must contain normalized lowercase suffixes"
                )
        if len(set(value)) != len(value):
            raise ValueError("supported_extensions must not contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_repeated_line_policy(self) -> PageAwareLoaderConfig:
        if self.repeated_line_min_pages < 2:
            raise ValueError("repeated_line_min_pages must be at least 2")
        if self.extraction_algorithm_version == "page_extract_v1":
            if self.pdf_ocr is not None:
                raise ValueError("page_extract_v1 requires pdf_ocr to be null")
        elif self.extraction_algorithm_version == "page_extract_tesseract_v2":
            if self.pdf_ocr is None:
                raise ValueError("page_extract_tesseract_v2 requires an OCR runtime")
        else:
            raise ValueError("unsupported page extraction algorithm version")
        return self


class SourcePage(_StrictFrozenModel):
    """One extracted page, retaining both raw and cleaned page-local text."""

    schema_version: Literal["source_page_v1"]
    page_number: int = Field(ge=1)
    extraction_method: str
    raw_text: str
    cleaned_text: str
    raw_chars: int = Field(ge=0)
    cleaned_chars: int = Field(ge=0)
    raw_content_sha1: str = Field(pattern=_SHA1_PATTERN)
    cleaned_content_sha1: str = Field(pattern=_SHA1_PATTERN)

    @field_validator("extraction_method")
    @classmethod
    def validate_extraction_method(cls, value: str) -> str:
        return _validate_nonempty_identifier(value, field_name="extraction_method")

    @model_validator(mode="after")
    def validate_content_contract(self) -> SourcePage:
        if self.raw_chars != len(self.raw_text):
            raise ValueError("raw_chars must equal len(raw_text)")
        if self.cleaned_chars != len(self.cleaned_text):
            raise ValueError("cleaned_chars must equal len(cleaned_text)")
        if self.raw_content_sha1 != _sha1_text(self.raw_text):
            raise ValueError("raw_content_sha1 must hash the exact raw_text")
        if self.cleaned_content_sha1 != _sha1_text(self.cleaned_text):
            raise ValueError("cleaned_content_sha1 must hash the exact cleaned_text")
        return self


class PageSpan(_StrictFrozenModel):
    """Page ownership in the cleaned-document offset space.

    ``[start_char, content_end_char)`` is page text. The remaining portion up
    to ``end_char`` is the following page separator and belongs to this page.
    This makes page spans a gap-free partition of the assembled document.
    """

    schema_version: Literal["page_span_v1"]
    page_number: int = Field(ge=1)
    start_char: int = Field(ge=0)
    content_end_char: int = Field(ge=0)
    end_char: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> PageSpan:
        if not self.start_char <= self.content_end_char <= self.end_char:
            raise ValueError(
                "PageSpan requires start_char <= content_end_char <= end_char"
            )
        return self


class CleanedSourceDocument(_StrictFrozenModel):
    """Page-aware source in the authoritative cleaned-document offset space."""

    schema_version: Literal["cleaned_source_document_v1"]
    doc_id: str = Field(pattern=_DOC_ID_PATTERN)
    subject: str
    source_file: str
    source_relpath: str
    source_file_sha1: str = Field(pattern=_SHA1_PATTERN)
    doc_type: str
    extraction_method: str
    cleaning_policy_id: str
    loader_policy_id: str = Field(pattern=_SHA256_PATTERN)
    pagination_kind: Literal["physical", "logical"]
    page_separator: str
    source_pages: tuple[SourcePage, ...] = Field(min_length=1)
    page_spans: tuple[PageSpan, ...] = Field(min_length=1)
    content: str = Field(min_length=1)
    content_sha1: str = Field(pattern=_SHA1_PATTERN)

    @field_validator("subject", "doc_type", "extraction_method", "cleaning_policy_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "identifier")
        return _validate_nonempty_identifier(value, field_name=field_name)

    @field_validator("source_file")
    @classmethod
    def validate_source_file(cls, value: str) -> str:
        return _validate_source_file(value)

    @field_validator("source_relpath")
    @classmethod
    def validate_source_relpath(cls, value: str) -> str:
        return _validate_source_relpath(value)

    @field_validator("page_separator")
    @classmethod
    def validate_page_separator(cls, value: str) -> str:
        if not value:
            raise ValueError("page_separator must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_document_contract(self) -> CleanedSourceDocument:
        if PurePosixPath(self.source_relpath).name != self.source_file:
            raise ValueError("source_file must match the source_relpath basename")
        if self.content_sha1 != _sha1_text(self.content):
            raise ValueError("content_sha1 must hash the exact cleaned content")
        if len(self.source_pages) != len(self.page_spans):
            raise ValueError("source_pages and page_spans must have equal length")
        expected_pages = tuple(range(1, len(self.source_pages) + 1))
        if tuple(page.page_number for page in self.source_pages) != expected_pages:
            raise ValueError("source page numbers must be contiguous and 1-based")
        if tuple(span.page_number for span in self.page_spans) != expected_pages:
            raise ValueError("page span numbers must be contiguous and 1-based")

        cursor = 0
        last_index = len(self.page_spans) - 1
        for index, (page, span) in enumerate(
            zip(self.source_pages, self.page_spans, strict=True)
        ):
            if page.extraction_method != self.extraction_method:
                raise ValueError("all pages must use the document extraction_method")
            if span.start_char != cursor:
                raise ValueError("page spans must form a gap-free ordered partition")
            if span.content_end_char - span.start_char != page.cleaned_chars:
                raise ValueError("page text span length must equal cleaned page length")
            if (
                self.content[span.start_char : span.content_end_char]
                != page.cleaned_text
            ):
                raise ValueError(
                    "page text must equal its exact cleaned-document slice"
                )
            separator = self.content[span.content_end_char : span.end_char]
            expected_separator = self.page_separator if index < last_index else ""
            if separator != expected_separator:
                raise ValueError("page span separator does not match page_separator")
            cursor = span.end_char
        if cursor != len(self.content):
            raise ValueError("page spans must cover the complete cleaned document")
        return self


class ParentChildPolicy(_StrictFrozenModel):
    """Complete output-affecting policy for deterministic parent-child splits."""

    schema_version: Literal["parent_child_policy_v1"]
    canonicalization_version: Literal["canonical_json_v1"]
    id_algorithm_version: Literal["parent_child_id_v1"]
    metadata_contract_version: Literal["parent_child_metadata_v1"]
    policy_id: str = Field(pattern=_SHA256_PATTERN)
    structure_detector_version: str
    structure_pattern_set_version: str
    structure_merge_version: str
    short_unit_chars: int = Field(gt=0)
    parent_split_algorithm: Literal["span_recursive_v1"]
    child_split_algorithm: Literal["span_recursive_v1"]
    loader_policy_id: str = Field(pattern=_SHA256_PATTERN)
    cleaning_policy_id: str
    parent_size: int = Field(gt=0)
    parent_overlap: int = Field(ge=0)
    parent_hard_max: int = Field(gt=0)
    child_size: int = Field(gt=0)
    child_overlap: int = Field(ge=0)
    child_hard_max: int = Field(gt=0)
    parent_separators: tuple[str, ...] = Field(min_length=1)
    child_separators: tuple[str, ...] = Field(min_length=1)
    major_section_max_level: int = Field(ge=1, le=6)
    atomic_fenced_code_blocks: bool
    atomic_markdown_tables: bool
    atomic_list_blocks: bool
    atomic_display_math: bool

    @field_validator(
        "structure_detector_version",
        "structure_pattern_set_version",
        "structure_merge_version",
        "cleaning_policy_id",
    )
    @classmethod
    def validate_policy_labels(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "policy_label")
        return _validate_nonempty_identifier(value, field_name=field_name)

    @field_validator("parent_separators", "child_separators")
    @classmethod
    def validate_separators(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not separator for separator in value):
            raise ValueError("separators must not contain empty strings")
        if len(set(value)) != len(value):
            raise ValueError("separators must not contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_size_contract(self) -> ParentChildPolicy:
        if self.parent_overlap >= self.parent_size:
            raise ValueError("parent_overlap must be smaller than parent_size")
        if self.child_overlap >= self.child_size:
            raise ValueError("child_overlap must be smaller than child_size")
        if self.parent_size > self.parent_hard_max:
            raise ValueError("parent_size must not exceed parent_hard_max")
        if self.child_size > self.child_hard_max:
            raise ValueError("child_size must not exceed child_hard_max")
        if self.child_hard_max > self.parent_hard_max:
            raise ValueError("child_hard_max must not exceed parent_hard_max")
        return self


class ParentRecord(_StrictFrozenModel):
    """Authoritative parent text and provenance record."""

    schema_version: Literal["parent_record_v1"]
    parent_id: str = Field(pattern=_PARENT_ID_PATTERN)
    doc_id: str = Field(pattern=_DOC_ID_PATTERN)
    subject: str
    generation_id: str
    policy_id: str = Field(pattern=_SHA256_PATTERN)
    parent_index: int = Field(ge=0)
    source_file: str
    source_relpath: str
    source_file_sha1: str = Field(pattern=_SHA1_PATTERN)
    doc_type: str
    extraction_method: str
    cleaning_policy_id: str
    section_id: str = Field(pattern=_SECTION_ID_PATTERN)
    section_title: str
    section_path: tuple[str, ...]
    pagination_kind: Literal["physical", "logical"]
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    parent_chars: int = Field(gt=0)
    content_sha1: str = Field(pattern=_SHA1_PATTERN)
    content: str = Field(min_length=1)

    @field_validator(
        "subject",
        "generation_id",
        "doc_type",
        "extraction_method",
        "cleaning_policy_id",
    )
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "identifier")
        return _validate_nonempty_identifier(value, field_name=field_name)

    @field_validator("source_file")
    @classmethod
    def validate_source_file(cls, value: str) -> str:
        return _validate_source_file(value)

    @field_validator("source_relpath")
    @classmethod
    def validate_source_relpath(cls, value: str) -> str:
        return _validate_source_relpath(value)

    @field_validator("section_path")
    @classmethod
    def validate_section_path(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_section_path(value)

    @model_validator(mode="after")
    def validate_parent_contract(self) -> ParentRecord:
        if PurePosixPath(self.source_relpath).name != self.source_file:
            raise ValueError("source_file must match the source_relpath basename")
        if self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start")
        if self.end_char <= self.start_char:
            raise ValueError("parent span must be non-empty")
        if self.parent_chars != self.end_char - self.start_char:
            raise ValueError("parent_chars must equal end_char - start_char")
        if self.parent_chars != len(self.content):
            raise ValueError("parent_chars must equal len(content)")
        if self.content_sha1 != _sha1_text(self.content):
            raise ValueError("content_sha1 must hash the exact parent content")
        return self


class ChildMetadata(_StrictFrozenModel):
    """Strict child identity, provenance, and parent-relative offsets."""

    schema_version: Literal["child_metadata_v1"]
    child_id: str = Field(pattern=_CHILD_ID_PATTERN)
    parent_id: str = Field(pattern=_PARENT_ID_PATTERN)
    doc_id: str = Field(pattern=_DOC_ID_PATTERN)
    subject: str
    generation_id: str
    policy_id: str = Field(pattern=_SHA256_PATTERN)
    child_index: int = Field(ge=0)
    child_start_in_parent: int = Field(ge=0)
    child_end_in_parent: int = Field(gt=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    child_chars: int = Field(gt=0)
    content_sha1: str = Field(pattern=_SHA1_PATTERN)
    source_file: str
    source_relpath: str
    source_file_sha1: str = Field(pattern=_SHA1_PATTERN)
    doc_type: str
    section_id: str = Field(pattern=_SECTION_ID_PATTERN)
    section_title: str
    section_path: tuple[str, ...]
    pagination_kind: Literal["physical", "logical"]
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)

    @field_validator("subject", "generation_id", "doc_type")
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "identifier")
        return _validate_nonempty_identifier(value, field_name=field_name)

    @field_validator("source_file")
    @classmethod
    def validate_source_file(cls, value: str) -> str:
        return _validate_source_file(value)

    @field_validator("source_relpath")
    @classmethod
    def validate_source_relpath(cls, value: str) -> str:
        return _validate_source_relpath(value)

    @field_validator("section_path")
    @classmethod
    def validate_section_path(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_section_path(value)

    @model_validator(mode="after")
    def validate_child_metadata_contract(self) -> ChildMetadata:
        if PurePosixPath(self.source_relpath).name != self.source_file:
            raise ValueError("source_file must match the source_relpath basename")
        if self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start")
        if self.child_end_in_parent <= self.child_start_in_parent:
            raise ValueError("child parent-relative span must be non-empty")
        if self.end_char <= self.start_char:
            raise ValueError("child absolute span must be non-empty")
        if self.child_chars != self.child_end_in_parent - self.child_start_in_parent:
            raise ValueError("child_chars must equal the parent-relative span length")
        if self.child_chars != self.end_char - self.start_char:
            raise ValueError("child_chars must equal the absolute span length")
        return self

    def to_chroma_metadata(self) -> dict[str, str | int | float | bool]:
        """Serialize the domain model to validated scalar-only Chroma metadata."""

        values = self.model_dump(mode="python")
        section_path = values.pop("section_path")
        values["section_path"] = json.dumps(
            list(section_path),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return {
            key: value
            for key, value in values.items()
            if isinstance(value, (str, int, float, bool))
            and (not isinstance(value, str) or bool(value))
        }

    @classmethod
    def from_chroma_metadata(
        cls,
        metadata: dict[str, object],
    ) -> ChildMetadata:
        """Decode the exact scalar Chroma representation without alias repair."""

        payload = dict(metadata)
        encoded_path = payload.get("section_path")
        if not isinstance(encoded_path, str):
            raise ValueError("Chroma section_path must be canonical JSON text")
        try:
            decoded_path = json.loads(encoded_path)
        except json.JSONDecodeError as exc:
            raise ValueError("Chroma section_path is invalid JSON") from exc
        if not isinstance(decoded_path, list) or any(
            not isinstance(item, str) for item in decoded_path
        ):
            raise ValueError("Chroma section_path must decode to a string list")
        canonical_path = json.dumps(
            decoded_path,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if canonical_path != encoded_path:
            raise ValueError("Chroma section_path is not canonical JSON")
        payload["section_path"] = tuple(decoded_path)

        # Empty section titles are intentionally omitted at the scalar boundary.
        # Their one valid decoded representation is the explicit domain empty title.
        if "section_title" not in payload:
            payload["section_title"] = ""
        return cls.model_validate(payload)


class ChildDocument(_StrictFrozenModel):
    """Child text paired with strict scalar-serializable metadata."""

    schema_version: Literal["child_document_v1"]
    content: str = Field(min_length=1)
    metadata: ChildMetadata

    @model_validator(mode="after")
    def validate_child_content(self) -> ChildDocument:
        if len(self.content) != self.metadata.child_chars:
            raise ValueError("child content length must equal metadata.child_chars")
        if _sha1_text(self.content) != self.metadata.content_sha1:
            raise ValueError("child content hash must equal metadata.content_sha1")
        return self


class ParentChildBundle(_StrictFrozenModel):
    """Complete, validated output for one cleaned source document."""

    schema_version: Literal["parent_child_bundle_v1"]
    generation_id: str
    policy_id: str = Field(pattern=_SHA256_PATTERN)
    policy: ParentChildPolicy
    source: CleanedSourceDocument
    parents: tuple[ParentRecord, ...] = Field(min_length=1)
    children: tuple[ChildDocument, ...] = Field(min_length=1)

    @field_validator("generation_id")
    @classmethod
    def validate_generation_id(cls, value: str) -> str:
        return _validate_nonempty_identifier(value, field_name="generation_id")

    @model_validator(mode="after")
    def validate_policy_identity(self) -> ParentChildBundle:
        if self.policy_id != self.policy.policy_id:
            raise ValueError("bundle policy_id must equal policy.policy_id")
        return self
