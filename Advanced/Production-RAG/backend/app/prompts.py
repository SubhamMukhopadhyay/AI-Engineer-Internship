"""
prompts.py
----------
All prompt templates used by the LangGraph workflow, kept separate
from the nodes themselves so they're easy to audit and tune.
"""

from __future__ import annotations

NOT_FOUND_MARKER = "DOCUMENT_DOES_NOT_CONTAIN_ANSWER"

REWRITE_SYSTEM_PROMPT = """You rewrite user questions into clear, self-contained
search queries optimized for retrieving relevant document passages.
Do not answer the question. Output ONLY the rewritten query, nothing else."""


def build_rewrite_prompt(question: str) -> tuple[str, str]:
    user = (
        "Rewrite the following question into a concise, keyword-rich search "
        f"query (output only the query):\n\n{question}"
    )
    return REWRITE_SYSTEM_PROMPT, user


GENERATION_SYSTEM_PROMPT = f"""You are a production document question-answering assistant.

STRICT RULES:
1. Answer ONLY using the CONTEXT provided below. Never use outside/general knowledge.
2. Never invent, assume, or infer facts that are not explicitly present in the CONTEXT.
3. If the CONTEXT does not contain enough information to answer, respond with
   exactly this marker as the first line: {NOT_FOUND_MARKER}
   followed by a one-sentence explanation of what is missing.
4. Cite which source supports each claim using bracketed labels, e.g. [Source 1].
5. Be concise, factual, and well-structured."""


def build_generation_prompt(question: str, context_chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(context_chunks, start=1):
        page_info = f", page {c['page_number']}" if c.get("page_number") else ""
        blocks.append(
            f"[Source {i}] (document: {c['doc_name']}{page_info})\n{c['text']}"
        )
    context = "\n\n".join(blocks) if blocks else "(no context retrieved)"
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"


VALIDATION_SYSTEM_PROMPT = """You are a strict fact-checking reviewer.
Given a QUESTION, the CONTEXT that was supplied to an assistant, and the
assistant's ANSWER, determine whether every factual claim in the ANSWER is
actually supported by the CONTEXT.

Respond with exactly one line in this format:
VALID: <yes|no> | NOTES: <one short sentence>"""


def build_validation_prompt(question: str, context_chunks: list[dict], answer: str) -> str:
    blocks = []
    for i, c in enumerate(context_chunks, start=1):
        blocks.append(f"[Source {i}]\n{c['text']}")
    context = "\n\n".join(blocks) if blocks else "(no context)"
    return (
        f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}\n\n"
        "Is the ANSWER fully supported by the CONTEXT?"
    )
