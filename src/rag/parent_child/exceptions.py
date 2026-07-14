"""Typed failures for the parent-child document pipeline."""

from __future__ import annotations


class ParentChildError(Exception):
    """Base class for explicit parent-child pipeline failures."""


class SourcePathError(ParentChildError):
    """Raised when a source path violates the configured data-root boundary."""


class UnsupportedSourceTypeError(ParentChildError):
    """Raised when a source extension is not explicitly configured."""


class EmptySourceError(ParentChildError):
    """Raised when extraction and cleaning produce no indexable content."""


class SourceExtractionError(ParentChildError):
    """Raised when an explicitly supported source cannot be decoded or read."""


class OcrRuntimeIdentityError(SourceExtractionError):
    """Raised when the configured OCR runtime does not match its fingerprints."""


class OcrProtocolError(SourceExtractionError):
    """Raised when OCR execution or its strict response contract fails."""


class ParentChildInvariantError(ParentChildError):
    """Raised when a domain invariant cannot be proven."""


class AtomicSpanTooLargeError(ParentChildError):
    """Raised when preserving an atomic block would exceed a hard limit."""

    def __init__(
        self,
        *,
        block_kind: str,
        block_chars: int,
        hard_max_chars: int,
        start_char: int,
        end_char: int,
    ) -> None:
        self.block_kind = block_kind
        self.block_chars = block_chars
        self.hard_max_chars = hard_max_chars
        self.start_char = start_char
        self.end_char = end_char
        super().__init__(
            "Atomic block exceeds configured hard maximum: "
            f"kind={block_kind}, chars={block_chars}, "
            f"hard_max_chars={hard_max_chars}, span=[{start_char},{end_char})"
        )
