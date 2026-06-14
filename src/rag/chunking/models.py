"""Data models for standalone document structure detection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentSection:
    """A detected document section span."""

    title: str
    level: int
    start_line: int
    end_line: int
    start_char: int
    end_char: int
    heading_style: str
    section_path: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "level": self.level,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "heading_style": self.heading_style,
            "section_path": list(self.section_path),
        }
