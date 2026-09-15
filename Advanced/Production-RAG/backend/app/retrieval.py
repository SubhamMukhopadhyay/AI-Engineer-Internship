"""
retrieval.py
------------
Wraps the vector index search with document-scoping and returns plain
dicts suitable for the typed LangGraph state.
"""

from __future__ import annotations

from app.indexing import get_index
from app.logging_config import get_logger

logger = get_logger(__name__)


def retrieve(question: str, top_k: int, doc_ids: list[str] | None = None) -> list[dict]:
    index = get_index()
    results = index.search(question, top_k=top_k, doc_ids=doc_ids)
    out = []
    for chunk, score in results:
        out.append(
            {
                "doc_id": chunk.doc_id,
                "doc_name": chunk.doc_name,
                "chunk_id": chunk.chunk_id,
                "page_number": chunk.page_number,
                "text": chunk.text,
                "similarity": score,
            }
        )
    logger.debug("Retrieved %d chunks for question=%r", len(out), question)
    return out
