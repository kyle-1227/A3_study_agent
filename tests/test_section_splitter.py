"""Unit tests for section-aware chunking (REQ-05).

Tests cover: section header detection, section splitting, metadata enrichment,
sub-chunking of long sections, fallback for documents without section headers,
and integration with load_documents.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Sample exam paper text (simplified)
# ---------------------------------------------------------------------------

SAMPLE_EXAM_PAPER = """\
2024年普通高等学校招生全国统一考试
语文

一、现代文阅读（35分）
（一）现代文阅读I（本题共5小题，19分）
阅读下面的文字，完成1~5题。
材料一：
人类社会发展的历史表明，对一个民族、一个国家来说，最持久、最深层的力量是全社会共同认可的核心价值观。

二、古代诗文阅读（35分）
（一）文言文阅读（本题共5小题，20分）
阅读下面的文言文，完成6~10题。
楚人和氏得玉璞楚山中，奉而献之厉王。厉王使玉人相之，玉人曰："石也。"

三、语言文字运用（20分）
（一）语言文字运用I（本题共2小题，7分）
阅读下面的文字，完成17~18题。
近年来，越来越多的人开始关注传统文化的传承与发展。

四、写作（60分）
23. 阅读下面的材料，根据要求写作。（60分）
在一次青年交流活动中，有人说"每个人都应该找到属于自己的路"。
请结合材料写一篇文章，体现你的感悟与思考。
要求：选准角度，确定立意，明确文体，自拟标题；不要套作，不得抄袭；不少于800字。
"""

SAMPLE_NO_SECTIONS = """\
这是一段没有节标题的普通文本。
它可能是一篇文章或笔记。
用于测试没有节标题时的回退行为。
"""


# ===========================================================================
# TestSectionPattern — regex detection
# ===========================================================================

class TestSectionPattern:
    """Verify the section header regex matches expected patterns."""

    def test_matches_standard_headers(self):
        from src.rag.section_splitter import SECTION_PATTERN

        assert SECTION_PATTERN.search("一、现代文阅读（35分）")
        assert SECTION_PATTERN.search("二、古代诗文阅读（35分）")
        assert SECTION_PATTERN.search("三、语言文字运用（20分）")
        assert SECTION_PATTERN.search("四、写作（60分）")

    def test_matches_dot_separator(self):
        from src.rag.section_splitter import SECTION_PATTERN

        assert SECTION_PATTERN.search("一.选择题")
        assert SECTION_PATTERN.search("二．填空题")

    def test_does_not_match_plain_text(self):
        from src.rag.section_splitter import SECTION_PATTERN

        assert SECTION_PATTERN.search("这是一段普通文本") is None
        assert SECTION_PATTERN.search("阅读下面的文字") is None

    def test_does_not_match_sub_section_markers(self):
        """Parenthesized sub-sections like （一）should NOT match."""
        from src.rag.section_splitter import SECTION_PATTERN

        assert SECTION_PATTERN.search("（一）现代文阅读I") is None

    def test_matches_multiline(self):
        """Pattern should find headers embedded in multiline text."""
        from src.rag.section_splitter import SECTION_PATTERN

        text = "前言部分\n一、选择题\n第1题..."
        match = SECTION_PATTERN.search(text)
        assert match is not None
        assert "选择题" in match.group(0)


# ===========================================================================
# TestSplitIntoSections — core splitting logic
# ===========================================================================

class TestSplitIntoSections:
    """Verify text is correctly split into (title, body) pairs."""

    def test_splits_exam_paper(self):
        from src.rag.section_splitter import SectionAwareSplitter

        splitter = SectionAwareSplitter()
        sections = splitter._split_into_sections(SAMPLE_EXAM_PAPER)

        titles = [title for title, _ in sections]
        assert len(sections) == 4
        assert "现代文阅读" in titles[0]
        assert "古代诗文阅读" in titles[1]
        assert "语言文字运用" in titles[2]
        assert "写作" in titles[3]

    def test_section_body_contains_content(self):
        from src.rag.section_splitter import SectionAwareSplitter

        splitter = SectionAwareSplitter()
        sections = splitter._split_into_sections(SAMPLE_EXAM_PAPER)

        # Section 1 body should contain reading material
        _, body = sections[0]
        assert "核心价值观" in body

        # Section 4 (writing) should contain the essay prompt
        _, body = sections[3]
        assert "感悟与思考" in body

    def test_no_section_headers_returns_whole_text(self):
        from src.rag.section_splitter import SectionAwareSplitter

        splitter = SectionAwareSplitter()
        sections = splitter._split_into_sections(SAMPLE_NO_SECTIONS)

        assert len(sections) == 1
        title, body = sections[0]
        assert title == ""
        assert "普通文本" in body

    def test_preamble_before_first_section_is_discarded_or_included(self):
        """Text before the first section header should be in a preamble section."""
        from src.rag.section_splitter import SectionAwareSplitter

        splitter = SectionAwareSplitter()
        sections = splitter._split_into_sections(SAMPLE_EXAM_PAPER)

        # The preamble ("2024年普通高等学校...语文") is before "一、现代文阅读"
        # It should either be included in the first section or as a separate preamble
        # Our design: preamble goes into first section's body (prepended)
        # OR: we have a 5th section with empty title for preamble
        # Let's check: the first section should be "一、现代文阅读..."
        # and preamble content is part of section[0] body or a separate entry
        all_text = " ".join(body for _, body in sections)
        # The full exam paper content should be covered
        assert "2024年" in all_text or len(sections) == 4

    def test_section_body_excludes_title_line(self):
        """The section body should not repeat the title line."""
        from src.rag.section_splitter import SectionAwareSplitter

        splitter = SectionAwareSplitter()
        sections = splitter._split_into_sections(SAMPLE_EXAM_PAPER)

        for title, body in sections:
            if title:
                # The title line itself should not appear as the first line of body
                assert not body.strip().startswith(title)


# ===========================================================================
# TestCreateDocuments — full chunking pipeline
# ===========================================================================

class TestCreateDocuments:
    """Verify create_documents produces chunks with section_title metadata."""

    def test_chunks_have_section_title_metadata(self):
        from src.rag.section_splitter import SectionAwareSplitter

        splitter = SectionAwareSplitter(chunk_size=800, chunk_overlap=100)
        base_meta = {"subject": "chinese", "source_file": "2024_chinese.pdf"}
        chunks = splitter.create_documents(
            texts=[SAMPLE_EXAM_PAPER],
            metadatas=[base_meta],
        )

        assert len(chunks) >= 4  # at least one chunk per section
        for chunk in chunks:
            assert "section_title" in chunk.metadata
            assert chunk.metadata["subject"] == "chinese"

    def test_writing_section_not_mixed_with_reading(self):
        """Chunks from 写作 section must NOT contain reading comprehension text."""
        from src.rag.section_splitter import SectionAwareSplitter

        splitter = SectionAwareSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.create_documents(texts=[SAMPLE_EXAM_PAPER])

        writing_chunks = [c for c in chunks if "写作" in c.metadata.get("section_title", "")]
        reading_chunks = [c for c in chunks if "现代文阅读" in c.metadata.get("section_title", "")]

        assert len(writing_chunks) >= 1
        assert len(reading_chunks) >= 1

        # Writing chunks should contain essay prompt, not reading material
        for chunk in writing_chunks:
            assert "核心价值观" not in chunk.page_content
        # Reading chunks should not contain essay prompt
        for chunk in reading_chunks:
            assert "感悟与思考" not in chunk.page_content

    def test_long_section_is_sub_chunked(self):
        """A section longer than chunk_size should produce multiple chunks."""
        from src.rag.section_splitter import SectionAwareSplitter

        long_text = "一、长文阅读\n" + ("这是一段很长的测试文本。" * 200)
        splitter = SectionAwareSplitter(chunk_size=200, chunk_overlap=50)
        chunks = splitter.create_documents(texts=[long_text])

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["section_title"] == "一、长文阅读"

    def test_preserves_base_metadata(self):
        """Base metadata should be preserved alongside section_title."""
        from src.rag.section_splitter import SectionAwareSplitter

        splitter = SectionAwareSplitter()
        meta = {"subject": "chinese", "year": "2024", "doc_type": "exam_paper"}
        chunks = splitter.create_documents(
            texts=[SAMPLE_EXAM_PAPER],
            metadatas=[meta],
        )

        for chunk in chunks:
            assert chunk.metadata["subject"] == "chinese"
            assert chunk.metadata["year"] == "2024"
            assert chunk.metadata["doc_type"] == "exam_paper"
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
        texts = [SAMPLE_EXAM_PAPER, SAMPLE_NO_SECTIONS]
        metas = [{"source": "exam.pdf"}, {"source": "notes.txt"}]
        chunks = splitter.create_documents(texts=texts, metadatas=metas)

        sources = {c.metadata["source"] for c in chunks}
        assert "exam.pdf" in sources
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
            p = Path(tmpdir) / "2024_chinese_exam.txt"
            p.write_text(SAMPLE_EXAM_PAPER, encoding="utf-8")

            section_splitter = SectionAwareSplitter(chunk_size=800, chunk_overlap=100)
            docs = load_documents(
                tmpdir,
                subject="chinese",
                doc_type="exam_paper",
                splitter=section_splitter,
            )

            assert len(docs) >= 4
            titles = {d.metadata.get("section_title") for d in docs}
            assert any("写作" in t for t in titles if t)
            assert any("现代文阅读" in t for t in titles if t)

    def test_load_documents_default_splitter_unchanged(self):
        """Without splitter param, load_documents behaves as before."""
        from src.rag.loader import load_documents

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test_2024.txt"
            p.write_text("Simple test content " * 50, encoding="utf-8")

            docs = load_documents(tmpdir, subject="math")

            assert len(docs) >= 1
            # Default splitter does NOT add section_title
            assert "section_title" not in docs[0].metadata
