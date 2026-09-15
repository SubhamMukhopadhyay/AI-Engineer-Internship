"""
reranking.py
------------
Reranks retrieved chunks against the query using a cross-encoder model
when available (RERANKER_MODEL), improving precision over raw vector
similarity alone. Falls back to a lightweight lexical overlap score
(no network/model download needed) so the pipeline still functions in
restricted environments.
"""

from __future__ import annotations

import re
import threading

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_cross_encoder = None
_tried_load = False
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _get_cross_encoder():
    global _cross_encoder, _tried_load
    if _tried_load:
        return _cross_encoder
    with _lock:
        if _tried_load:
            return _cross_encoder
        _tried_load = True
        settings = get_settings()
        if not settings.use_reranker:
            return None
        try:
            from sentence_transformers import CrossEncoder

            _cross_encoder = CrossEncoder(settings.reranker_model)
            logger.info("Loaded cross-encoder reranker: %s", settings.reranker_model)
        except Exception as exc:
            logger.warning(
                "Could not load cross-encoder reranker (%s). "
                "Falling back to lexical overlap reranking.",
                exc,
            )
            _cross_encoder = None
    return _cross_encoder


def _lexical_overlap_score(query: str, text: str) -> float:
    q_tokens = set(_TOKEN_RE.findall(query.lower()))
    t_tokens = set(_TOKEN_RE.findall(text.lower()))
    if not q_tokens or not t_tokens:
        return 0.0
    overlap = len(q_tokens & t_tokens)
    return overlap / (len(q_tokens) ** 0.5)


def rerank(question: str, retrieved: list[dict], top_k: int) -> list[dict]:
    if not retrieved:
        return []

    encoder = _get_cross_encoder()
    if encoder is not None:
        pairs = [(question, r["text"]) for r in retrieved]
        try:
            scores = encoder.predict(pairs)
        except Exception as exc:
            logger.warning("Cross-encoder prediction failed (%s); using lexical fallback.", exc)
            scores = [_lexical_overlap_score(question, r["text"]) for r in retrieved]
    else:
        scores = [_lexical_overlap_score(question, r["text"]) for r in retrieved]

    for r, s in zip(retrieved, scores):
        r["rerank_score"] = float(s)

    ranked = sorted(retrieved, key=lambda r: r["rerank_score"], reverse=True)
    return ranked[:top_k]
