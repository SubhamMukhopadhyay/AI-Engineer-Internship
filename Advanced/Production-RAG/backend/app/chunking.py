"""
chunking.py
-----------
Overlapping, sentence-boundary-aware chunking, parameterised via
Settings (CHUNK_SIZE / CHUNK_OVERLAP).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings
from app.ingestion import PageText
from app.preprocessing import clean_text


@dataclass
class Chunk:
    doc_id: str
    doc_name: str
    chunk_id: str
    page_number: int
    text: str


def chunk_pages(
    doc_id: str,
    doc_name: str,
    pages: list[PageText],
    is_markdown: bool = False,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 4

    chunks: list[Chunk] = []
    counter = 0
    for page in pages:
        text = clean_text(page.text, is_markdown=is_markdown)
        if not text:
            continue
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            if end < n:
                boundary = text.rfind(". ", start, end)
                if boundary != -1 and boundary > start + chunk_size * 0.5:
                    end = boundary + 1
            piece = text[start:end].strip()
            if piece:
                counter += 1
                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        doc_name=doc_name,
                        chunk_id=f"{doc_id}-{counter}",
                        page_number=page.page_number,
                        text=piece,
                    )
                )
            if end >= n:
                break
            start = max(end - chunk_overlap, start + 1)
    return chunks
