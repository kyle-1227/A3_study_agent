"""Incremental extraction of one JSON string field from tool arguments."""

from __future__ import annotations

import json
from dataclasses import dataclass


class ToolArgumentStreamError(ValueError):
    """Raised when streamed tool arguments violate the incremental contract."""


@dataclass(frozen=True)
class PartialStringField:
    value: str
    complete: bool


class IncrementalJSONStringFieldDecoder:
    """Expose only appended decoded text for a configured top-level JSON field."""

    def __init__(self, field_name: str) -> None:
        if not field_name.strip():
            raise ToolArgumentStreamError("field_name is required")
        self.field_name = field_name
        self._arguments = ""
        self._emitted = ""

    @property
    def arguments(self) -> str:
        return self._arguments

    @property
    def value(self) -> str:
        return self._emitted

    def feed(self, fragment: str) -> str:
        if not isinstance(fragment, str):
            raise ToolArgumentStreamError("tool argument fragment must be a string")
        self._arguments += fragment
        current = extract_partial_top_level_string(
            self._arguments,
            field_name=self.field_name,
        ).value
        if not current.startswith(self._emitted):
            raise ToolArgumentStreamError("streamed field content diverged")
        delta = current[len(self._emitted) :]
        self._emitted = current
        return delta


def extract_partial_top_level_string(
    arguments: str,
    *,
    field_name: str,
) -> PartialStringField:
    """Decode the available prefix of a top-level JSON string without repair."""

    text = str(arguments)
    index = _skip_whitespace(text, 0)
    if index >= len(text) or text[index] != "{":
        return PartialStringField(value="", complete=False)
    index += 1
    decoder = json.JSONDecoder()

    while True:
        index = _skip_whitespace(text, index)
        if index >= len(text) or text[index] == "}":
            return PartialStringField(value="", complete=False)
        parsed_key = _parse_complete_json_string(text, index)
        if parsed_key is None:
            return PartialStringField(value="", complete=False)
        key, index = parsed_key
        index = _skip_whitespace(text, index)
        if index >= len(text) or text[index] != ":":
            return PartialStringField(value="", complete=False)
        index = _skip_whitespace(text, index + 1)

        if key == field_name:
            return _parse_partial_json_string(text, index)

        try:
            _ignored, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            return PartialStringField(value="", complete=False)
        index = _skip_whitespace(text, end)
        if index >= len(text):
            return PartialStringField(value="", complete=False)
        if text[index] == ",":
            index += 1
            continue
        return PartialStringField(value="", complete=False)


def _parse_complete_json_string(text: str, index: int) -> tuple[str, int] | None:
    if index >= len(text) or text[index] != '"':
        return None
    end = index + 1
    escaped = False
    while end < len(text):
        char = text[end]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            raw = text[index : end + 1]
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ToolArgumentStreamError("invalid JSON string key") from exc
            if not isinstance(value, str):
                raise ToolArgumentStreamError("JSON object key must be a string")
            return value, end + 1
        end += 1
    return None


def _parse_partial_json_string(text: str, index: int) -> PartialStringField:
    if index >= len(text) or text[index] != '"':
        return PartialStringField(value="", complete=False)
    output: list[str] = []
    cursor = index + 1
    while cursor < len(text):
        char = text[cursor]
        if char == '"':
            return PartialStringField(value="".join(output), complete=True)
        if char != "\\":
            if ord(char) < 0x20:
                raise ToolArgumentStreamError("unescaped control character")
            output.append(char)
            cursor += 1
            continue

        cursor += 1
        if cursor >= len(text):
            break
        escape = text[cursor]
        simple = {
            '"': '"',
            "\\": "\\",
            "/": "/",
            "b": "\b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
        }
        if escape in simple:
            output.append(simple[escape])
            cursor += 1
            continue
        if escape != "u":
            raise ToolArgumentStreamError("invalid JSON string escape")
        if cursor + 4 >= len(text):
            break
        raw_hex = text[cursor + 1 : cursor + 5]
        if any(char not in "0123456789abcdefABCDEF" for char in raw_hex):
            raise ToolArgumentStreamError("invalid unicode escape")
        codepoint = int(raw_hex, 16)
        cursor += 5
        if 0xD800 <= codepoint <= 0xDBFF:
            if cursor + 5 >= len(text) or text[cursor : cursor + 2] != "\\u":
                break
            low_hex = text[cursor + 2 : cursor + 6]
            if any(char not in "0123456789abcdefABCDEF" for char in low_hex):
                raise ToolArgumentStreamError("invalid low surrogate escape")
            low = int(low_hex, 16)
            if not 0xDC00 <= low <= 0xDFFF:
                raise ToolArgumentStreamError("invalid unicode surrogate pair")
            codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00)
            cursor += 6
        elif 0xDC00 <= codepoint <= 0xDFFF:
            raise ToolArgumentStreamError("unexpected low surrogate escape")
        output.append(chr(codepoint))
    return PartialStringField(value="".join(output), complete=False)


def _skip_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index


__all__ = [
    "IncrementalJSONStringFieldDecoder",
    "PartialStringField",
    "ToolArgumentStreamError",
    "extract_partial_top_level_string",
]
