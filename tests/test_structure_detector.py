from __future__ import annotations

import pytest

from src.rag.chunking import DocumentSection, detect_document_sections, get_section_text


def test_detect_document_sections_empty_text_returns_empty():
    assert detect_document_sections("") == []


def test_detect_document_sections_non_empty_without_headings_returns_fallback():
    text = "This is an ordinary sentence.\nAnother ordinary sentence."

    sections = detect_document_sections(text)

    assert sections == [
        DocumentSection(
            title="",
            level=0,
            start_line=0,
            end_line=1,
            start_char=0,
            end_char=len(text),
            heading_style="fallback_full_document",
            section_path=(),
        )
    ]


def test_detect_document_sections_preserves_preamble_before_first_heading():
    text = "Course overview before headings.\n\n# Setup\nSteps here.\n"

    sections = detect_document_sections(text)

    assert [section.title for section in sections] == ["Preamble", "Setup"]
    assert sections[0].heading_style == "preamble"
    assert sections[0].section_path == ()
    assert (
        get_section_text(text, sections[0]).strip()
        == "Course overview before headings."
    )


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


@pytest.mark.parametrize(
    ("line", "title", "style", "level"),
    [
        ("第1章 大数据概述", "大数据概述", "cjk_chapter", 1),
        ("1.1 Spark 简介", "Spark 简介", "numbered", 2),
        ("1.1.1 RDD 基础", "RDD 基础", "numbered", 3),
        ("一、基础概念", "基础概念", "cjk_numbered", 1),
        ("（一）变量与数据类型", "变量与数据类型", "cjk_parenthesized", 2),
        ("实验一 Hadoop 环境搭建", "Hadoop 环境搭建", "cjk_lab", 2),
        ("实验 1 Hadoop 环境搭建", "Hadoop 环境搭建", "cjk_lab", 2),
        ("任务一：安装虚拟机", "安装虚拟机", "cjk_task", 2),
        ("任务 1：安装虚拟机", "安装虚拟机", "cjk_task", 2),
        ("模块一：Python 基础", "Python 基础", "cjk_module", 2),
        ("模块 1：Python 基础", "Python 基础", "cjk_module", 2),
        ("项目一：学生成绩统计", "学生成绩统计", "cjk_project", 2),
        ("项目 1：学生成绩统计", "学生成绩统计", "cjk_project", 2),
    ],
)
def test_detect_document_sections_handles_cjk_course_headings(
    line, title, style, level
):
    sections = detect_document_sections(f"{line}\n正文内容")

    assert len(sections) == 1
    assert sections[0].title == title
    assert sections[0].heading_style == style
    assert sections[0].level == level


@pytest.mark.parametrize(
    ("line", "title", "style", "level"),
    [
        ("Module 1: Python Basics", "Python Basics", "module", 1),
        ("Lab 1: Hadoop Setup", "Hadoop Setup", "lab", 2),
        ("Task 1: Install Environment", "Install Environment", "task", 2),
        ("Project 1: Student Score Analysis", "Student Score Analysis", "project", 2),
    ],
)
def test_detect_document_sections_handles_english_course_headings(
    line, title, style, level
):
    sections = detect_document_sections(f"{line}\nBody")

    assert len(sections) == 1
    assert sections[0].title == title
    assert sections[0].heading_style == style
    assert sections[0].level == level


def test_detect_document_sections_reports_spans_and_paths():
    text = """第1章 大数据概述
body
1.1 大数据定义
body
1.1.1 RDD 基础
body
第2章 总结
body
"""

    sections = detect_document_sections(text)

    assert [section.section_path for section in sections] == [
        ("大数据概述",),
        ("大数据概述", "大数据定义"),
        ("大数据概述", "大数据定义", "RDD 基础"),
        ("总结",),
    ]
    assert sections[0].start_char == 0
    assert sections[0].end_char <= sections[1].start_char
    assert get_section_text(text, sections[1]).startswith("1.1 大数据定义")


def test_get_section_text_clamps_out_of_bounds_span():
    text = "short text"
    section = DocumentSection(
        title="manual",
        level=1,
        start_line=0,
        end_line=0,
        start_char=-20,
        end_char=500,
        heading_style="manual",
    )

    assert get_section_text(text, section) == text


def test_detect_document_sections_keeps_spans_in_bounds():
    text = "Intro\n\n# Heading\nBody"

    sections = detect_document_sections(text)

    for section in sections:
        assert section.start_char >= 0
        assert section.end_char <= len(text)
        assert section.start_char < section.end_char
        assert section.end_line >= section.start_line


def test_detect_document_sections_does_not_misclassify_long_plain_sentence():
    text = (
        "This is a very long ordinary sentence that explains a concept in prose "
        "rather than naming a document section, and it should not be treated as a heading."
    )

    sections = detect_document_sections(text)

    assert len(sections) == 1
    assert sections[0].heading_style == "fallback_full_document"
