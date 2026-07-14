"""Strict, fingerprinted Tesseract extraction for explicitly selected PDFs."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess

from src.rag.parent_child.exceptions import (
    OcrProtocolError,
    OcrRuntimeIdentityError,
)
from src.rag.parent_child.models import TesseractOcrRuntimeConfig


_TSV_FIELDS = (
    "level",
    "page_num",
    "block_num",
    "par_num",
    "line_num",
    "word_num",
    "left",
    "top",
    "width",
    "height",
    "conf",
    "text",
)
_REPARSE_POINT_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
_SAFE_ENVIRONMENT_NAMES = ("SystemRoot", "TEMP", "TMP", "WINDIR")
_NO_SPACE_BEFORE = frozenset(",.!?;:%)]}>，。！？；：、）】》」』％")
_NO_SPACE_AFTER = frozenset("([{<（【《「『")


def _is_link_or_reparse(path: Path) -> bool:
    try:
        status = path.lstat()
    except OSError as exc:
        raise OcrRuntimeIdentityError(
            "Configured OCR runtime artifact could not be inspected"
        ) from exc
    attributes = getattr(status, "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & _REPARSE_POINT_ATTRIBUTE)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise OcrRuntimeIdentityError(
            "Configured OCR runtime artifact could not be hashed"
        ) from exc
    return digest.hexdigest()


def compute_tesseract_runtime_manifest_sha256(
    binary_path: Path, tessdata_dir: Path
) -> str:
    """Hash the executable, adjacent DLLs, and exact TSV output config."""

    runtime_files = tuple(
        sorted(
            (binary_path, *binary_path.parent.glob("*.dll")),
            key=lambda item: item.name,
        )
    )
    configs_dir = tessdata_dir / "configs"
    if not configs_dir.is_dir() or _is_link_or_reparse(configs_dir):
        raise OcrRuntimeIdentityError(
            "Configured OCR TSV config directory is missing or linked"
        )
    tsv_config = configs_dir / "tsv"
    if not tsv_config.is_file() or _is_link_or_reparse(tsv_config):
        raise OcrRuntimeIdentityError(
            "Configured OCR TSV output config is missing or linked"
        )
    rows: list[list[str]] = []
    for path in runtime_files:
        if not path.is_file() or _is_link_or_reparse(path):
            raise OcrRuntimeIdentityError(
                "Configured OCR runtime file is missing or linked"
            )
        rows.append([f"runtime/{path.name}", _sha256_file(path)])
    rows.append(["tessdata/configs/tsv", _sha256_file(tsv_config)])
    encoded = json.dumps(
        rows,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _subprocess_environment(*, thread_limit: int | None) -> dict[str, str]:
    environment = {
        name: os.environ[name] for name in _SAFE_ENVIRONMENT_NAMES if name in os.environ
    }
    if thread_limit is not None:
        environment["OMP_THREAD_LIMIT"] = str(thread_limit)
    return environment


def _identity_command(command: tuple[str, ...], *, timeout_seconds: float) -> bytes:
    try:
        completed = subprocess.run(
            command,
            check=False,
            shell=False,
            capture_output=True,
            timeout=timeout_seconds,
            env=_subprocess_environment(thread_limit=None),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OcrRuntimeIdentityError(
            "Configured OCR runtime identity probe failed"
        ) from exc
    if completed.returncode != 0:
        raise OcrRuntimeIdentityError(
            "Configured OCR runtime identity probe returned non-zero"
        )
    return completed.stdout if completed.stdout else completed.stderr


def validate_tesseract_runtime(config: TesseractOcrRuntimeConfig) -> None:
    """Recompute every configured binary and language identity."""

    try:
        import fitz
    except ImportError as exc:
        raise OcrRuntimeIdentityError(
            "Configured OCR renderer runtime is unavailable"
        ) from exc
    if fitz.VersionBind != config.pymupdf_version:
        raise OcrRuntimeIdentityError("Configured PyMuPDF version mismatch")
    if fitz.VersionFitz != config.mupdf_version:
        raise OcrRuntimeIdentityError("Configured MuPDF version mismatch")

    binary = config.binary_path
    tessdata = config.tessdata_dir
    if not binary.is_file() or _is_link_or_reparse(binary):
        raise OcrRuntimeIdentityError(
            "Configured OCR binary must be a non-symlink regular file"
        )
    if not tessdata.is_dir() or _is_link_or_reparse(tessdata):
        raise OcrRuntimeIdentityError(
            "Configured OCR tessdata must be a non-symlink directory"
        )
    if _sha256_file(binary) != config.binary_sha256:
        raise OcrRuntimeIdentityError("Configured OCR binary SHA-256 mismatch")
    if (
        compute_tesseract_runtime_manifest_sha256(binary, tessdata)
        != config.runtime_manifest_sha256
    ):
        raise OcrRuntimeIdentityError("Configured OCR runtime manifest mismatch")

    for asset in config.language_assets:
        traineddata = tessdata / f"{asset.language}.traineddata"
        if not traineddata.is_file() or _is_link_or_reparse(traineddata):
            raise OcrRuntimeIdentityError(
                "Configured OCR traineddata must be a non-symlink regular file"
            )
        if _sha256_file(traineddata) != asset.traineddata_sha256:
            raise OcrRuntimeIdentityError("Configured OCR traineddata SHA-256 mismatch")

    version_output = _identity_command(
        (str(binary), "--version"),
        timeout_seconds=config.timeout_seconds,
    )
    try:
        first_line = version_output.decode("utf-8", errors="strict").splitlines()[0]
    except (UnicodeDecodeError, IndexError) as exc:
        raise OcrRuntimeIdentityError(
            "Configured OCR version response was invalid UTF-8"
        ) from exc
    if first_line.strip() != f"tesseract v{config.expected_version}":
        raise OcrRuntimeIdentityError("Configured OCR version mismatch")

    languages_output = _identity_command(
        (
            str(binary),
            "--tessdata-dir",
            str(tessdata),
            "--list-langs",
        ),
        timeout_seconds=config.timeout_seconds,
    )
    try:
        language_lines = tuple(
            line.strip()
            for line in languages_output.decode("utf-8", errors="strict").splitlines()
            if line.strip()
        )
    except UnicodeDecodeError as exc:
        raise OcrRuntimeIdentityError(
            "Configured OCR language response was invalid UTF-8"
        ) from exc
    if not language_lines:
        raise OcrRuntimeIdentityError(
            "Configured OCR language response omitted its header"
        )
    header = re.fullmatch(
        r'List of available languages in ".+" \((\d+)\):',
        language_lines[0],
    )
    if header is None:
        raise OcrRuntimeIdentityError(
            "Configured OCR language response header was invalid"
        )
    reported_languages = language_lines[1:]
    if len(set(reported_languages)) != len(reported_languages):
        raise OcrRuntimeIdentityError(
            "Configured OCR language response contained duplicates"
        )
    if int(header.group(1)) != len(reported_languages):
        raise OcrRuntimeIdentityError(
            "Configured OCR language response count was inconsistent"
        )
    available = set(reported_languages)
    if not {asset.language for asset in config.language_assets}.issubset(available):
        raise OcrRuntimeIdentityError(
            "Configured OCR runtime did not report every required language"
        )


def _parse_nonnegative_int(row: dict[str, str | None], field_name: str) -> int:
    value = row.get(field_name)
    if value is None:
        raise OcrProtocolError("Tesseract TSV row omitted a required field")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise OcrProtocolError("Tesseract TSV integer field was invalid") from exc
    if parsed < 0:
        raise OcrProtocolError("Tesseract TSV integer field was negative")
    return parsed


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (0x3400 <= codepoint <= 0x9FFF) or (0xF900 <= codepoint <= 0xFAFF)


def _join_ocr_words(words: list[str]) -> str:
    if not words:
        return ""
    output = words[0]
    for word in words[1:]:
        previous = output[-1]
        current = word[0]
        if not (
            (_is_cjk(previous) and _is_cjk(current))
            or current in _NO_SPACE_BEFORE
            or previous in _NO_SPACE_AFTER
        ):
            output += " "
        output += word
    return output


def parse_tesseract_tsv(payload: bytes) -> str:
    """Validate one TSV response and assemble deterministic line text."""

    try:
        decoded = payload.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise OcrProtocolError("Tesseract TSV was not valid UTF-8") from exc
    reader = csv.DictReader(io.StringIO(decoded), delimiter="\t")
    if tuple(reader.fieldnames or ()) != _TSV_FIELDS:
        raise OcrProtocolError("Tesseract TSV header did not match the protocol")

    assembled: list[str] = []
    saw_page_row = False
    active_key: tuple[int, int, int, int] | None = None
    active_word_number = 0
    active_words: list[str] = []
    for row in reader:
        if None in row or any(row.get(field) is None for field in _TSV_FIELDS):
            raise OcrProtocolError("Tesseract TSV row shape was invalid")
        level = _parse_nonnegative_int(row, "level")
        if level not in {1, 2, 3, 4, 5}:
            raise OcrProtocolError("Tesseract TSV level was outside the protocol")
        page_number = _parse_nonnegative_int(row, "page_num")
        block_number = _parse_nonnegative_int(row, "block_num")
        paragraph_number = _parse_nonnegative_int(row, "par_num")
        line_number = _parse_nonnegative_int(row, "line_num")
        word_number = _parse_nonnegative_int(row, "word_num")
        _parse_nonnegative_int(row, "left")
        _parse_nonnegative_int(row, "top")
        _parse_nonnegative_int(row, "width")
        _parse_nonnegative_int(row, "height")
        confidence_text = row["conf"]
        if confidence_text is None:
            raise OcrProtocolError("Tesseract TSV confidence field was missing")
        try:
            confidence = float(confidence_text)
        except ValueError as exc:
            raise OcrProtocolError(
                "Tesseract TSV confidence field was invalid"
            ) from exc
        if not math.isfinite(confidence) or not -1.0 <= confidence <= 100.0:
            raise OcrProtocolError(
                "Tesseract TSV confidence was outside the protocol range"
            )
        hierarchy = (
            page_number,
            block_number,
            paragraph_number,
            line_number,
            word_number,
        )
        required_positive = level
        expected_zero_tail = hierarchy[required_positive:]
        if hierarchy[0] != 1 or any(
            value < 1 for value in hierarchy[:required_positive]
        ):
            raise OcrProtocolError("Tesseract TSV hierarchy coordinates were invalid")
        if any(value != 0 for value in expected_zero_tail):
            raise OcrProtocolError(
                "Tesseract TSV hierarchy tail coordinates were invalid"
            )
        if level == 1:
            if saw_page_row:
                raise OcrProtocolError(
                    "Tesseract TSV contained duplicate single-page rows"
                )
            saw_page_row = True
        if level != 5:
            continue
        word = (row["text"] or "").strip()
        if not word:
            continue
        if confidence < 0.0:
            raise OcrProtocolError(
                "Tesseract TSV recognized word confidence was negative"
            )
        key = (page_number, block_number, paragraph_number, line_number)
        if key != active_key:
            if active_key is not None:
                assembled.append(_join_ocr_words(active_words))
                if key <= active_key:
                    raise OcrProtocolError(
                        "Tesseract TSV line ordering was not deterministic"
                    )
            active_key = key
            active_word_number = 0
            active_words = []
        if word_number <= active_word_number:
            raise OcrProtocolError("Tesseract TSV word ordering was not deterministic")
        active_word_number = word_number
        active_words.append(word)
    if active_key is not None:
        assembled.append(_join_ocr_words(active_words))
    if not saw_page_row:
        raise OcrProtocolError("Tesseract TSV omitted its single-page row")
    return "\n".join(assembled)


class TesseractCliOcr:
    """One validated no-fallback Tesseract subprocess boundary."""

    def __init__(self, config: TesseractOcrRuntimeConfig) -> None:
        validate_tesseract_runtime(config)
        self._config = config

    def recognize_png(self, png_bytes: bytes) -> str:
        if not png_bytes:
            raise OcrProtocolError("Rendered OCR page was empty")
        config = self._config
        command = (
            str(config.binary_path),
            "stdin",
            "stdout",
            "--tessdata-dir",
            str(config.tessdata_dir),
            "-l",
            "+".join(asset.language for asset in config.language_assets),
            "--oem",
            str(config.oem),
            "--psm",
            str(config.psm),
            "--dpi",
            str(config.dpi),
            "tsv",
        )
        environment = _subprocess_environment(thread_limit=config.thread_limit)
        try:
            completed = subprocess.run(
                command,
                input=png_bytes,
                check=False,
                shell=False,
                capture_output=True,
                timeout=config.timeout_seconds,
                cwd=config.binary_path.parent,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise OcrProtocolError("Configured Tesseract OCR request failed") from exc
        if completed.returncode != 0:
            raise OcrProtocolError(
                "Configured Tesseract OCR returned a non-zero exit code"
            )
        return parse_tesseract_tsv(completed.stdout)


def extract_pdf_pages_with_tesseract(
    path: Path, config: TesseractOcrRuntimeConfig
) -> tuple[str, ...]:
    """Render and OCR every physical page without writing intermediate images."""

    import fitz

    engine = TesseractCliOcr(config)
    pages: list[str] = []
    try:
        with fitz.open(path) as document:
            matrix = fitz.Matrix(config.dpi / 72.0, config.dpi / 72.0)
            for page in document:
                pixmap = page.get_pixmap(
                    matrix=matrix,
                    colorspace=fitz.csRGB,
                    alpha=config.render_alpha,
                    annots=config.render_annotations,
                )
                pages.append(engine.recognize_png(pixmap.tobytes("png")))
    except (OcrProtocolError, OcrRuntimeIdentityError):
        raise
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise OcrProtocolError(
            "PyMuPDF could not render the configured OCR source"
        ) from exc
    return tuple(pages)


__all__ = [
    "TesseractCliOcr",
    "compute_tesseract_runtime_manifest_sha256",
    "extract_pdf_pages_with_tesseract",
    "parse_tesseract_tsv",
    "validate_tesseract_runtime",
]
