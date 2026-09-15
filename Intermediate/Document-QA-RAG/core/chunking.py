"""
chunking.py
-----------
Splits extracted page text into overlapping chunks suitable for
embedding and retrieval.

Defaults (overridable via env vars, see .env.example):
    CHUNK_SIZE=900        characters per chunk
    CHUNK_OVERLAP=150     characters shared between consecutive chunks

Why overlap: without it, a fact that spans a chunk boundary can become
unretrievable because neither chunk fully contains it. A ~15-20%
overlap keeps chunks self-contained without exploding the index size.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from core.extraction import PageText


@dataclass
class Chunk:
    doc_id: str
    doc_name: str
    chunk_id: str
    page_number: int
    text: str


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(
    doc_id: str,
    doc_name: str,
    pages: list[PageText],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "150"))
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 4

    chunks: list[Chunk] = []
    counter = 0
    for page in pages:
        text = _clean(page.text)
        if not text:
            continue
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            # try to break on a sentence/paragraph boundary near `end`
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
