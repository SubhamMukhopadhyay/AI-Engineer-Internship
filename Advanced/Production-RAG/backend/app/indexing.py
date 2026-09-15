"""
indexing.py
-----------
FAISS index management: builds/maintains an in-memory index of chunk
embeddings with metadata, and supports save/load to disk so the index
survives a backend restart (INDEX_DIR).
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict

import faiss
import numpy as np

from app.chunking import Chunk
from app.config import get_settings
from app.embeddings import embed_texts
from app.logging_config import get_logger

logger = get_logger(__name__)


class VectorIndex:
    """Thread-safe in-memory FAISS index with JSON-backed metadata persistence."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []
        self.dim: int | None = None
        self._load_from_disk()

    # -- persistence -----------------------------------------------
    def _paths(self) -> tuple[str, str]:
        settings = get_settings()
        os.makedirs(settings.index_dir, exist_ok=True)
        return (
            os.path.join(settings.index_dir, "index.faiss"),
            os.path.join(settings.index_dir, "meta.json"),
        )

    def _load_from_disk(self) -> None:
        index_path, meta_path = self._paths()
        if os.path.exists(index_path) and os.path.exists(meta_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(meta_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.chunks = [Chunk(**c) for c in raw]
                self.dim = self.index.d
                logger.info("Loaded persisted index with %d chunks", len(self.chunks))
            except Exception as exc:
                logger.warning("Failed to load persisted index (%s); starting fresh.", exc)
                self.index = None
                self.chunks = []

    def _save_to_disk(self) -> None:
        if self.index is None:
            return
        index_path, meta_path = self._paths()
        try:
            faiss.write_index(self.index, index_path)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump([asdict(c) for c in self.chunks], f)
        except Exception as exc:
            logger.warning("Failed to persist index: %s", exc)

    # -- core ops -----------------------------------------------
    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = embed_texts([c.text for c in chunks])
        with self._lock:
            if self.index is None:
                self.dim = vectors.shape[1]
                self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(vectors)
            self.chunks.extend(chunks)
            self._save_to_disk()
        logger.info("Indexed %d chunks (total now %d)", len(chunks), len(self.chunks))

    def search(
        self, query: str, top_k: int = 8, doc_ids: list[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        if self.is_empty():
            return []
        qvec = embed_texts([query])
        fetch_k = min(top_k * 5 if doc_ids else top_k, self.index.ntotal)
        scores, indices = self.index.search(qvec, fetch_k)

        results: list[tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            if doc_ids and chunk.doc_id not in doc_ids:
                continue
            results.append((chunk, float(score)))
            if len(results) >= top_k:
                break
        return results

    def document_ids(self) -> list[str]:
        seen = []
        for c in self.chunks:
            if c.doc_id not in seen:
                seen.append(c.doc_id)
        return seen

    def documents_summary(self) -> dict[str, dict]:
        summary: dict[str, dict] = {}
        for c in self.chunks:
            entry = summary.setdefault(c.doc_id, {"name": c.doc_name, "chunks": 0})
            entry["chunks"] += 1
        return summary

    def clear(self) -> None:
        with self._lock:
            self.index = None
            self.chunks = []
            self.dim = None
            index_path, meta_path = self._paths()
            for p in (index_path, meta_path):
                if os.path.exists(p):
                    os.remove(p)


_index_singleton: VectorIndex | None = None
_singleton_lock = threading.Lock()


def get_index() -> VectorIndex:
    global _index_singleton
    if _index_singleton is None:
        with _singleton_lock:
            if _index_singleton is None:
                _index_singleton = VectorIndex()
    return _index_singleton


def reset_index_singleton() -> None:
    """Test-only helper: force get_index() to rebuild from current settings."""
    global _index_singleton
    with _singleton_lock:
        _index_singleton = None
