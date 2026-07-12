"""Incremental strict-tool argument decoding tests."""

from __future__ import annotations

import pytest

from src.llm.tool_argument_stream import (
    IncrementalJSONStringFieldDecoder,
    ToolArgumentStreamError,
    extract_partial_top_level_string,
)


def test_decoder_emits_only_new_answer_text_across_fragments() -> None:
    decoder = IncrementalJSONStringFieldDecoder("answer")

    assert decoder.feed('{"grounding_status":"general_knowledge",') == ""
    assert decoder.feed('"answer":"你好，') == "你好，"
    assert decoder.feed("world\\n") == "world\n"
    assert decoder.feed('第二行","suggestions":[]') == "第二行"
    assert decoder.value == "你好，world\n第二行"


def test_decoder_waits_for_complete_escape_sequences() -> None:
    decoder = IncrementalJSONStringFieldDecoder("answer")

    assert decoder.feed('{"answer":"A\\u4f') == "A"
    assert decoder.feed("60\\u597d") == "你好"
    assert decoder.feed("\\ud83d") == ""
    assert decoder.feed('\\ude00"}') == "😀"


def test_decoder_does_not_expose_other_fields_or_raw_json() -> None:
    result = extract_partial_top_level_string(
        '{"uncertainty_note":"secret-ish diagnostic","answer":"safe',
        field_name="answer",
    )
    assert result.value == "safe"
    assert result.complete is False


def test_decoder_rejects_invalid_escape_without_repair() -> None:
    decoder = IncrementalJSONStringFieldDecoder("answer")
    with pytest.raises(ToolArgumentStreamError, match="invalid JSON string escape"):
        decoder.feed('{"answer":"bad\\q"}')
