"""
embeddings.py
-------------
Embedding backend selection: prefers sentence-transformers (real
semantic embeddings, downloaded from HuggingFace on first run), and
falls back automatically to a zero-network hashing embedder if the
model cannot be downloaded (offline/sandboxed environments). This
keeps the whole pipeline runnable in restricted network environments
while still giving best-quality embeddings when internet is available.
"""

from __future__ import annotations

import hashlib
import re
import threading

import numpy as np

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_model = None
_backend_name: str | None = None

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class LocalHashingEmbedder:
    """Dependency-light, zero-network fallback embedder (lexical, not semantic)."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        tokens = self._tokens(text)
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        for a, b in zip(tokens, tokens[1:]):
            h = int(hashlib.md5(f"{a}_{b}".encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 0.5
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False) -> np.ndarray:
        return np.vstack([self._vector(t) for t in texts])


def get_embedding_model():
    global _model, _backend_name
    if _model is None:
        with _lock:
            if _model is None:
                settings = get_settings()
                try:
                    from sentence_transformers import SentenceTransformer

                    _model = SentenceTransformer(settings.embedding_model)
                    _backend_name = f"sentence-transformers:{settings.embedding_model}"
                    logger.info("Loaded embedding backend: %s", _backend_name)
                except Exception as exc:
                    logger.warning(
                        "Could not load sentence-transformers model (%s). "
                        "Falling back to local hashing embedder.",
                        exc,
                    )
                    _model = LocalHashingEmbedder()
                    _backend_name = "local-hashing-fallback"
    return _model


def embedding_backend_name() -> str:
    get_embedding_model()
    return _backend_name or "unknown"


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype="float32")
