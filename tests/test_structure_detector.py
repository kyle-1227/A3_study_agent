from __future__ import annotations

from src.rag.chunking import DocumentSection, detect_document_sections


def test_detect_document_sections_handles_generic_markdown_and_numbering():
    text = """# Overview
Intro paragraph.

## Setup
Steps here.

1.1 Data Preparation
Details here.

2 Practice
More details.
"""

    sections = detect_document_sections(text)

    assert [section.title for section in sections] == [
        "Overview",
        "Setup",
        "Data Preparation",
        "Practice",
    ]
    assert sections[0].level == 1
    assert sections[1].level == 2
    assert sections[2].heading_style == "numbered"
    assert all(isinstance(section, DocumentSection) for section in sections)


def test_detect_document_sections_handles_cjk_numbered_structure():
    text = """一、基础概念
正文。

二、实践任务
正文。

第 3 章 总结
正文。
"""

    sections = detect_document_sections(text)

    assert [section.title for section in sections] == ["基础概念", "实践任务", "总结"]
    assert [section.heading_style for section in sections] == [
        "cjk_numbered",
        "cjk_numbered",
        "cjk_chapter",
    ]


def test_detect_document_sections_ignores_plain_sentences():
    text = """This is an ordinary sentence.
Another ordinary sentence.
There are no structural headings here.
"""

    assert detect_document_sections(text) == []


def test_detect_document_sections_reports_spans_and_paths():
    text = """# Parent
body
## Child
body
# Next
body
"""

    sections = detect_document_sections(text)

    assert sections[0].section_path == ("Parent",)
    assert sections[1].section_path == ("Parent", "Child")
    assert sections[2].section_path == ("Next",)
    assert sections[0].start_char == 0
    assert sections[0].end_char <= sections[1].start_char
