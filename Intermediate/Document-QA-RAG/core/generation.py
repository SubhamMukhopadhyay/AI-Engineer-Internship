"""
generation.py
-------------
Builds the grounded answer-generation prompt and calls the LLM.

The system prompt explicitly forbids the model from answering out of
general knowledge and forces it to say when the documents don't
contain the answer — this is the core hallucination-reduction
mechanism of this project.
"""

from __future__ import annotations

from core.llm_client import LLMClient
from core.vectorstore import VectorStore

NOT_FOUND_MARKER = "DOCUMENT_DOES_NOT_CONTAIN_ANSWER"

SYSTEM_PROMPT = f"""You are a document question-answering assistant.

STRICT RULES:
1. Answer ONLY using the CONTEXT provided below. Do not use outside/general knowledge.
2. Do not invent, assume, or infer facts that are not explicitly present in the CONTEXT.
3. If the CONTEXT does not contain enough information to answer the question,
   respond with exactly this marker on its own first line: {NOT_FOUND_MARKER}
   followed by a one-sentence explanation of what is missing.
4. When you do answer, cite which source chunk(s) support each claim using
   the bracketed labels shown in the CONTEXT, e.g. [Source 1].
5. Be concise and factual."""


def build_user_prompt(question: str, retrieved: list[tuple]) -> str:
    context_blocks = []
    for i, (chunk, score) in enumerate(retrieved, start=1):
        page_info = f", page {chunk.page_number}" if chunk.page_number else ""
        context_blocks.append(
            f"[Source {i}] (document: {chunk.doc_name}{page_info}, "
            f"similarity: {score:.2f})\n{chunk.text}"
        )
    context = "\n\n".join(context_blocks) if context_blocks else "(no context retrieved)"
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"


def generate_answer(
    client: LLMClient, question: str, retrieved: list[tuple]
) -> tuple[str, bool]:
    """
    Returns (answer_text, is_grounded). is_grounded is False when the
    model reports the documents don't contain the answer.
    """
    user_prompt = build_user_prompt(question, retrieved)
    raw = client.chat(SYSTEM_PROMPT, user_prompt, temperature=0.1, max_tokens=900)
    if raw.strip().startswith(NOT_FOUND_MARKER):
        explanation = raw.strip()[len(NOT_FOUND_MARKER):].strip(" :\n-")
        text = (
            "⚠️ The uploaded document(s) do not contain enough information "
            "to answer this question."
        )
        if explanation:
            text += f"\n\n{explanation}"
        return text, False
    return raw, True
