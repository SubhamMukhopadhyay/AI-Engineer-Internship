"""
generation.py
-------------
Calls the LLM to produce a grounded answer from retrieved/reranked
context, and detects the "not found in documents" marker.
"""

from __future__ import annotations

from app.llm_client import LLMClient
from app.prompts import (
    GENERATION_SYSTEM_PROMPT,
    NOT_FOUND_MARKER,
    build_generation_prompt,
)


def generate_answer(client: LLMClient, question: str, context_chunks: list[dict]) -> tuple[str, bool]:
    """Returns (answer_text, is_grounded)."""
    user_prompt = build_generation_prompt(question, context_chunks)
    raw = client.chat(GENERATION_SYSTEM_PROMPT, user_prompt, temperature=0.1, max_tokens=900)

    if raw.strip().startswith(NOT_FOUND_MARKER):
        explanation = raw.strip()[len(NOT_FOUND_MARKER):].strip(" :\n-")
        text = "⚠️ The indexed document(s) do not contain enough information to answer this question."
        if explanation:
            text += f"\n\n{explanation}"
        return text, False

    return raw, True
