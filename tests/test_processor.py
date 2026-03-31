"""Tests for the document processor module."""

import os

import fitz
import pytest

from swiftbits.processor import Chunk, _chunk_text, process_document


def _create_pdf(path, pages_text: list[str]):
    """Create a test PDF with given page texts."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        rect = fitz.Rect(72, 72, 540, 720)
        page.insert_textbox(rect, text, fontsize=11)
    doc.save(str(path))
    doc.close()


def _create_password_pdf(path, text="Secret content"):
    """Create a password-protected PDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    doc.save(str(path), encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="pass", owner_pw="pass")
    doc.close()


class TestChunkText:
    def test_single_chunk_when_text_fits(self):
        text = "Hello world, this is a short text."
        result = _chunk_text(text, chunk_size=512)
        assert len(result) == 1
        assert result[0] == text

    def test_splits_on_paragraph_boundary(self):
        para1 = "A" * 200
        para2 = "B" * 200
        text = f"{para1}\n\n{para2}"
        result = _chunk_text(text, chunk_size=250, chunk_overlap=20)
        assert len(result) >= 2
        # First chunk should end at or near paragraph boundary
        assert para1 in result[0] or result[0].startswith("A")

    def test_splits_on_sentence_boundary(self):
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        result = _chunk_text(text, chunk_size=50, chunk_overlap=10)
        assert len(result) >= 2

    def test_overlap_present(self):
        text = "word " * 200  # 1000 chars
        result = _chunk_text(text, chunk_size=100, chunk_overlap=30)
        assert len(result) > 1
        # Check overlap: end of chunk N should appear at start of chunk N+1
        for i in range(len(result) - 1):
            tail = result[i][-30:]
            assert tail in result[i + 1], "Overlap not found between consecutive chunks"

    def test_small_chunk_merged(self):
        # Text that would produce a tiny trailing chunk
        text = "A" * 500 + " " + "B" * 30
        result = _chunk_text(text, chunk_size=510, chunk_overlap=10)
        # The small trailing bit should be merged
        for chunk in result:
            assert len(chunk) >= 50 or len(result) == 1

    def test_hard_cutoff_no_separator(self):
        # No separators at all — must hard-cut
        text = "A" * 1000
        result = _chunk_text(text, chunk_size=200, chunk_overlap=20)
        assert len(result) >= 2

    def test_overlap_must_be_less_than_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            _chunk_text("some text", chunk_size=100, chunk_overlap=100)

    def test_empty_text_returns_empty(self):
        assert _chunk_text("", chunk_size=512) == []


class TestProcessDocument:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            process_document("/nonexistent/file.pdf")

    def test_unsupported_type(self, tmp_path):
        f = tmp_path / "test.docx"
        f.write_text("fake")
        with pytest.raises(ValueError, match="Unsupported file type"):
            process_document(str(f))

    def test_single_page_pdf(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        _create_pdf(pdf, ["This is a short document."])
        chunks = process_document(str(pdf))
        assert len(chunks) == 1
        assert chunks[0].metadata["source"] == "test.pdf"
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].metadata["total_chunks"] == 1
        assert chunks[0].metadata["page_numbers"] == [1]
        assert chunks[0].metadata["char_count"] > 0

    def test_multipage_pdf_tracks_pages(self, tmp_path):
        pdf = tmp_path / "multi.pdf"
        # Create pages with lots of text to produce multiple chunks
        page1_text = "Page one content is here. " * 100
        page2_text = "Page two content is here. " * 100
        _create_pdf(pdf, [page1_text, page2_text])
        chunks = process_document(str(pdf), chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1
        # At least one chunk should reference page 1
        page1_chunks = [c for c in chunks if 1 in c.metadata["page_numbers"]]
        assert len(page1_chunks) > 0
        # At least one chunk should reference page 2
        page2_chunks = [c for c in chunks if 2 in c.metadata["page_numbers"]]
        assert len(page2_chunks) > 0

    def test_empty_pdf(self, tmp_path):
        pdf = tmp_path / "empty.pdf"
        doc = fitz.open()
        doc.new_page()  # blank page with no text
        doc.save(str(pdf))
        doc.close()
        with pytest.raises(ValueError, match="No text content"):
            process_document(str(pdf))

    def test_password_protected_pdf(self, tmp_path):
        pdf = tmp_path / "locked.pdf"
        _create_password_pdf(pdf)
        with pytest.raises(ValueError, match="password-protected"):
            process_document(str(pdf))

    def test_metadata_fields_present(self, tmp_path):
        pdf = tmp_path / "meta.pdf"
        _create_pdf(pdf, ["Some text content here for testing metadata fields."])
        chunks = process_document(str(pdf))
        meta = chunks[0].metadata
        assert "source" in meta
        assert "page_numbers" in meta
        assert "chunk_index" in meta
        assert "total_chunks" in meta
        assert "char_count" in meta

    def test_source_is_filename_only(self, tmp_path):
        subdir = tmp_path / "nested" / "dir"
        subdir.mkdir(parents=True)
        pdf = subdir / "deep.pdf"
        _create_pdf(pdf, ["Content"])
        chunks = process_document(str(pdf))
        assert chunks[0].metadata["source"] == "deep.pdf"


class TestProcessTxt:
    def test_single_chunk_txt(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("This is a short text file.")
        chunks = process_document(str(f))
        assert len(chunks) == 1
        assert "short text file" in chunks[0].text
        assert chunks[0].metadata["source"] == "notes.txt"
        assert chunks[0].metadata["page_numbers"] == [1]
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].metadata["total_chunks"] == 1

    def test_multiline_txt(self, tmp_path):
        f = tmp_path / "long.txt"
        f.write_text("Some content here. " * 100)
        chunks = process_document(str(f), chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_empty_txt(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        with pytest.raises(ValueError, match="No text content"):
            process_document(str(f))

    def test_whitespace_only_txt(self, tmp_path):
        f = tmp_path / "blank.txt"
        f.write_text("   \n\n\n   ")
        with pytest.raises(ValueError, match="No text content"):
            process_document(str(f))

    def test_txt_metadata_fields(self, tmp_path):
        f = tmp_path / "meta.txt"
        f.write_text("Testing metadata fields in a text file.")
        chunks = process_document(str(f))
        meta = chunks[0].metadata
        assert "source" in meta
        assert "page_numbers" in meta
        assert "chunk_index" in meta
        assert "total_chunks" in meta
        assert "char_count" in meta


class TestProcessMd:
    def test_single_chunk_md(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Hello\n\nThis is a markdown file.")
        chunks = process_document(str(f))
        assert len(chunks) == 1
        assert "# Hello" in chunks[0].text
        assert chunks[0].metadata["source"] == "readme.md"
        assert chunks[0].metadata["page_numbers"] == [1]

    def test_md_with_formatting(self, tmp_path):
        f = tmp_path / "formatted.md"
        f.write_text(
            "# Title\n\n"
            "## Section\n\n"
            "- item 1\n"
            "- item 2\n\n"
            "```python\nprint('hello')\n```\n\n"
            "Some **bold** and *italic* text."
        )
        chunks = process_document(str(f))
        assert len(chunks) >= 1
        # Markdown formatting is preserved as-is
        full_text = "".join(c.text for c in chunks)
        assert "# Title" in full_text
        assert "```python" in full_text
        assert "**bold**" in full_text

    def test_empty_md(self, tmp_path):
        f = tmp_path / "empty.md"
        f.write_text("")
        with pytest.raises(ValueError, match="No text content"):
            process_document(str(f))

    def test_md_metadata(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("# Doc\n\nContent here.")
        chunks = process_document(str(f))
        meta = chunks[0].metadata
        assert meta["source"] == "doc.md"
        assert meta["page_numbers"] == [1]
        assert meta["chunk_index"] == 0
        assert meta["char_count"] > 0
