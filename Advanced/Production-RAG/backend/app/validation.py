"""
validation.py
--------------
Two kinds of validation used by the workflow:

1. Query validation — cheap, local, no LLM call: rejects empty/too-short/
   too-long questions before any retrieval or generation happens.
2. Answer validation — an LLM-based fact-check pass that reviews whether
   the generated answer is actually supported by the retrieved context.
   This is a second line of defence against hallucination on top of the
   generation-prompt constraints.
"""

from __future__ import annotations

from app.llm_client import LLMClient, LLMRequestError
from app.logging_config import get_logger
from app.prompts import VALIDATION_SYSTEM_PROMPT, build_validation_prompt

logger = get_logger(__name__)

MIN_QUESTION_LEN = 3
MAX_QUESTION_LEN = 2000


def validate_query(question: str) -> tuple[bool, str]:
    q = (question or "").strip()
    if not q:
        return False, "Question must not be empty."
    if len(q) < MIN_QUESTION_LEN:
        return False, "Question is too short to be meaningful."
    if len(q) > MAX_QUESTION_LEN:
        return False, f"Question exceeds the {MAX_QUESTION_LEN} character limit."
    return True, ""


def validate_answer(
    client: LLMClient, question: str, context_chunks: list[dict], answer: str, grounded: bool
) -> tuple[bool, str]:
    """
    Returns (is_valid, notes). If the answer was already flagged as
    "not grounded" by the generation step, we trust that determination
    and skip a second LLM call.
    """
    if not grounded:
        return True, "Answer correctly reported insufficient context; no fact-check needed."

    if not context_chunks:
        return False, "No context was available to validate the answer against."

    try:
        prompt = build_validation_prompt(question, context_chunks, answer)
        raw = client.chat(VALIDATION_SYSTEM_PROMPT, prompt, temperature=0.0, max_tokens=150)
    except LLMRequestError as exc:
        logger.warning("Answer validation call failed: %s", exc)
        return True, f"Validation step could not run ({exc}); answer shown unvalidated."

    lowered = raw.lower()
    is_valid = "valid: yes" in lowered or lowered.strip().startswith("yes")
    notes = raw.strip()
    return is_valid, notes
