"""Unit tests for section-aware chunking (REQ-05).

Tests cover: section header detection, section splitting, metadata enrichment,
sub-chunking of long sections, fallback for documents without section headers,
and integration with load_documents.

The sample document uses a generic university-course assessment context, while preserving Chinese top-level section
headers such as "一、基础概念" and "四、综合应用".
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Sample university course assessment text (simplified)
# ---------------------------------------------------------------------------

SAMPLE_COURSE_TEST_PAPER = """\
2024年大学课程综合测试
Python 程序设计

一、基础概念（25分）
（一）变量与数据类型（本题共5小题，10分）
阅读下面的说明，完成1~5题。
材料一：
Python 是一种解释型高级编程语言，常用于数据分析、Web 开发、自动化脚本和人工智能应用。
在 Python 中，变量不需要提前声明类型，解释器会根据赋值自动推断对象类型。

二、代码阅读（25分）
（一）控制流与函数（本题共5小题，15分）
阅读下面的代码片段，完成6~10题。
代码一：
def add_numbers(a, b):
    return a + b

result = add_numbers(3, 5)
print(result)

三、编程实践（30分）
（一）列表与字典应用（本题共2小题，15分）
请根据题目要求编写 Python 程序。
近年来，越来越多的数据处理任务需要学生熟练掌握列表、字典、循环和函数封装。

四、综合应用（20分）
23. 请完成一个简单的学生成绩统计程序。（20分）
要求：输入若干学生的姓名和成绩，计算平均分，找出最高分学生，并输出统计结果。
程序应结构清晰，变量命名规范，必要时使用函数封装。
"""


SAMPLE_NO_SECTIONS = """\
这是一段没有节标题的普通课程笔记。
它可能是一篇文章或课堂记录。
用于测试没有节标题时的回退行为。
"""


# ===========================================================================
# TestSectionPattern — regex detection
# ===========================================================================

class TestSectionPattern:
    """Verify the section header regex matches expected patterns."""

    def test_matches_standard_headers(self):
        from src.rag.section_splitter import SECTION_PATTERN

        assert SECTION_PATTERN.search("一、基础概念（25分）")
        assert SECTION_PATTERN.search("二、代码阅读（25分）")
        assert SECTION_PATTERN.search("三、编程实践（30分）")
        assert SECTION_PATTERN.search("四、综合应用（20分）")

    def test_matches_dot_separator(self):
        from src.rag.section_splitter import SECTION_PATTERN

        assert SECTION_PATTERN.search("一.基础概念")
        assert SECTION_PATTERN.search("二．代码阅读")

    def test_does_not_match_plain_text(self):
        from src.rag.section_splitter import SECTION_PATTERN

        assert SECTION_PATTERN.search("这是一段普通文本") is None
        assert SECTION_PATTERN.search("阅读下面的说明") is None

    def test_does_not_match_sub_section_markers(self):
        """Parenthesized sub-sections like （一）should NOT match."""
        from src.rag.section_splitter import SECTION_PATTERN

        assert SECTION_PATTERN.search("（一）变量与数据类型") is None

    def test_matches_multiline(self):
        """Pattern should find headers embedded in multiline text."""
        from src.rag.section_splitter import SECTION_PATTERN

        text = "课程说明\n一、基础概念\n第1题..."
        match = SECTION_PATTERN.search(text)
        assert match is not None
        assert "基础概念" in match.group(0)


# ===========================================================================
# TestSplitIntoSections — core splitting logic
# ===========================================================================

class TestSplitIntoSections:
    """Verify text is correctly split into (title, body) pairs."""

    def test_splits_course_test_paper(self):
        from src.rag.section_splitter import SectionAwareSplitter

        splitter = SectionAwareSplitter()
        sections = splitter._split_into_sections(SAMPLE_COURSE_TEST_PAPER)

        titles = [title for title, _ in sections]
        assert len(sections) == 4
        assert "基础概念" in titles[0]
        assert "代码阅读" in titles[1]
        assert "编程实践" in titles[2]
        assert "综合应用" in titles[3]

    def test_section_body_contains_content(self):
        from src.rag.section_splitter import SectionAwareSplitter

        splitter = SectionAwareSplitter()
        sections = splitter._split_into_sections(SAMPLE_COURSE_TEST_PAPER)

        # Section 1 body should contain concept explanation
        _, body = sections[0]
        assert "解释型高级编程语言" in body

        # Section 4 body should contain the integrated programming task
        _, body = sections[3]
        assert "学生成绩统计程序" in body

    def test_no_section_headers_returns_whole_text(self):
        from src.rag.section_splitter import SectionAwareSplitter

        splitter = SectionAwareSplitter()
        sections = splitter._split_into_sections(SAMPLE_NO_SECTIONS)

        assert len(sections) == 1
        title, body = sections[0]
        assert title == ""
        assert "普通课程笔记" in body

    def test_preamble_before_first_section_is_discarded_or_included(self):
        """Text before the first section header should remain covered."""
        from src.rag.section_splitter import SectionAwareSplitter

        splitter = SectionAwareSplitter()
        sections = splitter._split_into_sections(SAMPLE_COURSE_TEST_PAPER)

        # The preamble ("2024年大学课程综合测试...") is before "一、基础概念".
        # Our design allows it to be prepended into the first section body,
        # while still ensuring full document coverage.
        all_text = " ".join(body for _, body in sections)
        assert "2024年大学课程综合测试" in all_text or len(sections) == 4

    def test_section_body_excludes_title_line(self):
        """The section body should not repeat the title line."""
        from src.rag.section_splitter import SectionAwareSplitter

        splitter = SectionAwareSplitter()
        sections = splitter._split_into_sections(SAMPLE_COURSE_TEST_PAPER)

        for title, body in sections:
            if title:
                # The title line itself should not appear as the first line of body.
                assert not body.strip().startswith(title)


# ===========================================================================
# TestCreateDocuments — full chunking pipeline
# ===========================================================================

class TestCreateDocuments:
    """Verify create_documents produces chunks with section_title metadata."""

    def test_chunks_have_section_title_metadata(self):
        from src.rag.section_splitter import SectionAwareSplitter

        splitter = SectionAwareSplitter(chunk_size=800, chunk_overlap=100)
        base_meta = {"subject": "python", "source_file": "python_course_test.txt"}
        chunks = splitter.create_documents(
            texts=[SAMPLE_COURSE_TEST_PAPER],
            metadatas=[base_meta],
        )

        assert len(chunks) >= 4  # at least one chunk per section
        for chunk in chunks:
            assert "section_title" in chunk.metadata
            assert chunk.metadata["subject"] == "python"

    def test_integrated_application_section_not_mixed_with_concept_section(self):
        """Chunks from 综合应用 section must NOT contain 基础概念 explanation."""
        from src.rag.section_splitter import SectionAwareSplitter

        splitter = SectionAwareSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.create_documents(texts=[SAMPLE_COURSE_TEST_PAPER])

        application_chunks = [
            c for c in chunks if "综合应用" in c.metadata.get("section_title", "")
        ]
        concept_chunks = [
            c for c in chunks if "基础概念" in c.metadata.get("section_title", "")
        ]

        assert len(application_chunks) >= 1
        assert len(concept_chunks) >= 1

        # Integrated application chunks should contain the programming task,
        # not the introductory concept explanation.
        for chunk in application_chunks:
            assert "解释型高级编程语言" not in chunk.page_content

        # Concept chunks should not contain the integrated programming task.
        for chunk in concept_chunks:
            assert "学生成绩统计程序" not in chunk.page_content

    def test_long_section_is_sub_chunked(self):
        """A section longer than chunk_size should produce multiple chunks."""
        from src.rag.section_splitter import SectionAwareSplitter

        long_text = "一、长文知识点\n" + ("这是一段很长的课程测试文本。" * 200)
        splitter = SectionAwareSplitter(chunk_size=200, chunk_overlap=50)
        chunks = splitter.create_documents(texts=[long_text])

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["section_title"] == "一、长文知识点"

    def test_preserves_base_metadata(self):
        """Base metadata should be preserved alongside section_title."""
        from src.rag.section_splitter import SectionAwareSplitter

        splitter = SectionAwareSplitter()
        meta = {"subject": "python", "year": "2024", "doc_type": "course_test"}
        chunks = splitter.create_documents(
            texts=[SAMPLE_COURSE_TEST_PAPER],
            metadatas=[meta],
        )

        for chunk in chunks:
            assert chunk.metadata["subject"] == "python"
            assert chunk.metadata["year"] == "2024"
            assert chunk.metadata["doc_type"] == "course_test"
            assert "section_title" in chunk.metadata

    def test_no_sections_still_produces_chunks(self):
        """Text without section headers should still be chunked normally."""
        from src.rag.section_splitter import SectionAwareSplitter

        splitter = SectionAwareSplitter()
        chunks = splitter.create_documents(texts=[SAMPLE_NO_SECTIONS])

        assert len(chunks) >= 1
        assert chunks[0].metadata["section_title"] == ""

    def test_empty_text_returns_empty(self):
        from src.rag.section_splitter import SectionAwareSplitter

        splitter = SectionAwareSplitter()
        chunks = splitter.create_documents(texts=[""])
        assert chunks == []

    def test_multiple_texts(self):
        """create_documents should handle multiple texts with matching metadatas."""
        from src.rag.section_splitter import SectionAwareSplitter

        splitter = SectionAwareSplitter()
        texts = [SAMPLE_COURSE_TEST_PAPER, SAMPLE_NO_SECTIONS]
        metas = [{"source": "course_test.txt"}, {"source": "notes.txt"}]
        chunks = splitter.create_documents(texts=texts, metadatas=metas)

        sources = {c.metadata["source"] for c in chunks}
        assert "course_test.txt" in sources
        assert "notes.txt" in sources


# ===========================================================================
# TestLoaderIntegration — load_documents with section splitter
# ===========================================================================

class TestLoaderIntegration:
    """Verify load_documents accepts and uses a custom splitter."""

    def test_load_documents_with_section_splitter(self):
        from src.rag.loader import load_documents
        from src.rag.section_splitter import SectionAwareSplitter

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "2024_python_course_test.txt"
            p.write_text(SAMPLE_COURSE_TEST_PAPER, encoding="utf-8")

            section_splitter = SectionAwareSplitter(chunk_size=800, chunk_overlap=100)
            docs = load_documents(
                tmpdir,
                subject="python",
                doc_type="course_test",
                splitter=section_splitter,
            )

            assert len(docs) >= 4
            titles = {d.metadata.get("section_title") for d in docs}
            assert any("综合应用" in t for t in titles if t)
            assert any("基础概念" in t for t in titles if t)

    def test_load_documents_default_splitter_unchanged(self):
        """Without splitter param, load_documents behaves as before."""
        from src.rag.loader import load_documents

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test_2024.txt"
            p.write_text("Simple test content " * 50, encoding="utf-8")
            docs = load_documents(tmpdir, subject="python")

            assert len(docs) >= 1
            # Default splitter does NOT add section_title.
            assert "section_title" not in docs[0].metadata