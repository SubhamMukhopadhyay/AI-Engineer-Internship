"""
local_embedder.py
------------------
A zero-network-dependency embedding fallback used automatically when a
real sentence-transformers model cannot be downloaded (e.g. offline or
sandboxed environments).

It builds fixed-dimension vectors from hashed word n-gram TF counts,
L2-normalized so that FAISS inner-product search behaves like cosine
similarity. This is a lexical/bag-of-words style representation — it
captures shared vocabulary between the query and chunks, but not deep
semantic paraphrase similarity the way a trained transformer model does.

It exists purely for resilience; production deployments should set a
real EMBEDDING_MODEL and rely on sentence-transformers instead.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class LocalHashingEmbedder:
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
        # simple bigram signal adds a little word-order sensitivity
        for a, b in zip(tokens, tokens[1:]):
            h = int(hashlib.md5(f"{a}_{b}".encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 0.5
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def encode(
        self, texts: list[str], normalize_embeddings: bool = True, show_progress_bar: bool = False
    ) -> np.ndarray:
        return np.vstack([self._vector(t) for t in texts])
