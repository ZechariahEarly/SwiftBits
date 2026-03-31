"""SwiftBits document processor — PDF parsing and recursive text chunking."""

import os
import re
from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class Chunk:
    """A text chunk with metadata."""

    text: str
    metadata: dict


def _extract_text_from_pdf(file_path: str) -> list[tuple[int, str]]:
    """
    Extract text from a PDF file.

    Returns list of (page_number, page_text) tuples. Page numbers are 1-indexed.
    """
    doc = fitz.open(file_path)

    if doc.is_encrypted:
        doc.close()
        raise ValueError("PDF is password-protected")

    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        # Collapse multiple newlines, strip whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        if text:
            pages.append((i + 1, text))

    doc.close()
    return pages


def _chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[str]:
    """Split text into overlapping chunks, preferring natural boundaries."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    if len(text) <= chunk_size:
        return [text] if text else []

    separators = ["\n\n", ". ", "? ", "! ", "\n", " "]
    chunks = []
    start = 0

    while start < len(text):
        # If remaining text fits in one chunk, take it all
        if len(text) - start <= chunk_size:
            chunk = text[start:]
            if chunk.strip():
                # Merge small trailing chunks with previous
                if chunks and len(chunk) < 50:
                    chunks[-1] += chunk
                else:
                    chunks.append(chunk)
            break

        # Find best split point within the chunk_size window
        end = start + chunk_size
        split_pos = None

        for sep in separators:
            # Search backwards from end for this separator
            pos = text.rfind(sep, start, end)
            if pos > start:
                split_pos = pos + len(sep)
                break

        # Hard cutoff if no separator found
        if split_pos is None:
            split_pos = end

        chunk = text[start:split_pos]
        if chunk.strip():
            chunks.append(chunk)

        # Next chunk starts overlap characters before the split
        start = split_pos - chunk_overlap
        if start < 0:
            start = 0
        # Ensure we make forward progress
        if start <= (split_pos - chunk_size) or start < split_pos - chunk_overlap:
            start = max(split_pos - chunk_overlap, 0)
        # Prevent infinite loop
        if start >= split_pos:
            start = split_pos

    # Merge any chunks smaller than 50 chars with their predecessor
    merged = []
    for chunk in chunks:
        if merged and len(chunk) < 50:
            merged[-1] += chunk
        else:
            merged.append(chunk)

    return merged


def _build_page_offset_map(
    pages: list[tuple[int, str]],
) -> list[tuple[int, int]]:
    """
    Build a mapping of character offsets to page numbers.

    Returns list of (start_offset, page_number) tuples, sorted by offset.
    """
    offset_map = []
    current_offset = 0
    for page_num, text in pages:
        offset_map.append((current_offset, page_num))
        current_offset += len(text) + 1  # +1 for the newline joining pages
    return offset_map


def _get_page_numbers(
    chunk_start: int,
    chunk_end: int,
    offset_map: list[tuple[int, int]],
) -> list[int]:
    """Determine which pages a chunk spans based on character offsets."""
    pages = []
    for i, (offset, page_num) in enumerate(offset_map):
        # Determine the end of this page's text
        if i + 1 < len(offset_map):
            page_end = offset_map[i + 1][0]
        else:
            page_end = float("inf")

        # Check if chunk overlaps with this page
        if chunk_start < page_end and chunk_end > offset:
            pages.append(page_num)

    return pages


def _extract_text_from_plaintext(file_path: str) -> list[tuple[int, str]]:
    """
    Extract text from a plain text file (.txt or .md).

    Returns list with a single (1, text) tuple since plain text has no pages.
    """
    with open(file_path, encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if not text:
        return []
    return [(1, text)]


PARSERS = {
    ".pdf": _extract_text_from_pdf,
    ".txt": _extract_text_from_plaintext,
    ".md": _extract_text_from_plaintext,
}


def process_document(
    file_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Parse a document and return a list of text chunks with metadata.

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If the file type is unsupported or no text extracted.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in PARSERS:
        supported = ", ".join(PARSERS.keys())
        raise ValueError(f"Unsupported file type: {ext} (supported: {supported})")

    parser = PARSERS[ext]
    pages = parser(file_path)

    if not pages:
        raise ValueError(f"No text content extracted from {os.path.basename(file_path)}")

    # Concatenate all page text
    full_text = "\n".join(text for _, text in pages)
    offset_map = _build_page_offset_map(pages)

    # Chunk the text
    chunk_texts = _chunk_text(full_text, chunk_size, chunk_overlap)

    # Map chunks to page numbers
    source = os.path.basename(file_path)
    chunks = []
    current_offset = 0

    for i, chunk_text in enumerate(chunk_texts):
        # Find where this chunk appears in the full text
        chunk_start = full_text.find(chunk_text, current_offset)
        if chunk_start == -1:
            chunk_start = current_offset
        chunk_end = chunk_start + len(chunk_text)
        current_offset = chunk_end

        page_numbers = _get_page_numbers(chunk_start, chunk_end, offset_map)

        chunks.append(
            Chunk(
                text=chunk_text,
                metadata={
                    "source": source,
                    "page_numbers": page_numbers,
                    "chunk_index": i,
                    "total_chunks": len(chunk_texts),
                    "char_count": len(chunk_text),
                },
            )
        )

    return chunks
