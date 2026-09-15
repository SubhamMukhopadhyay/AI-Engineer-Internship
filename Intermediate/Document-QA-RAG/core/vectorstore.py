"""
vectorstore.py
--------------
Embedding generation and a FAISS-backed similarity index over document
chunks.

Embedding backend:
    Primary:  sentence-transformers (semantic embeddings, downloaded from
              HuggingFace on first run — set EMBEDDING_MODEL). No LLM API
              key is required for this; it's a separate, free, local model.
    Fallback: a local hashing/TF-IDF-style embedder (core/local_embedder.py)
              that requires no network access at all. This is used
              automatically if the sentence-transformers model cannot be
              downloaded (e.g. an offline/sandboxed environment), so the
              app keeps working end-to-end. It is weaker semantically than
              a real transformer model — see the project README.

Only *answer generation* calls the configured LLM API; embeddings never do.
"""

from __future__ import annotations

import os
import threading

import faiss
import numpy as np

from core.chunking import Chunk
from core.local_embedder import LocalHashingEmbedder

_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
_model_lock = threading.Lock()
_model = None
_backend_name = None


def get_embedding_model():
    """Returns an object exposing .encode(texts, normalize_embeddings=True, show_progress_bar=False) -> np.ndarray"""
    global _model, _backend_name
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    from sentence_transformers import SentenceTransformer

                    _model = SentenceTransformer(_MODEL_NAME)
                    _backend_name = f"sentence-transformers:{_MODEL_NAME}"
                except Exception:
                    # Offline / model download failed — fall back to a
                    # dependency-light local embedder so the app still works.
                    _model = LocalHashingEmbedder()
                    _backend_name = "local-hashing-fallback"
    return _model


def embedding_backend_name() -> str:
    get_embedding_model()
    return _backend_name or "unknown"


class VectorStore:
    """A small in-memory FAISS index of chunk embeddings with metadata."""

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []
        self.dim: int | None = None

    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        model = get_embedding_model()
        texts = [c.text for c in chunks]
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        vectors = np.asarray(vectors, dtype="float32")

        if self.index is None:
            self.dim = vectors.shape[1]
            self.index = faiss.IndexFlatIP(self.dim)  # cosine sim via normalized vectors

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(
        self, query: str, top_k: int = 5, doc_ids: list[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        if self.is_empty():
            return []
        model = get_embedding_model()
        qvec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        qvec = np.asarray(qvec, dtype="float32")

        # Over-fetch when scoping to specific documents so filtering doesn't
        # starve us of results.
        fetch_k = top_k * 5 if doc_ids else top_k
        fetch_k = min(fetch_k, self.index.ntotal)
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
