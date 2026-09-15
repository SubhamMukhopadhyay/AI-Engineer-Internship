"""
prompts.py
----------
Centralised prompt construction for every StudyMate AI feature.

Design principle: each feature gets (a) a fixed system prompt that
defines the assistant's role/constraints, and (b) a user prompt that
injects the student's content. Keeping construction in one place
makes the "how/why" of each prompt auditable and testable, and keeps
llm_client.py free of any feature-specific logic.
"""

from __future__ import annotations


def build_summary_prompt(content: str) -> tuple[str, str]:
    system = (
        "You are StudyMate, an expert study assistant. You write clear, "
        "accurate summaries of student notes. You NEVER invent facts that "
        "are not present in the provided text. You preserve key terms, "
        "definitions, and numbers exactly."
    )
    user = (
        "Summarize the following study notes for quick revision.\n"
        "Rules:\n"
        "- Use short bullet points grouped under headings if the content "
        "has multiple topics.\n"
        "- Keep only information present in the notes.\n"
        "- Aim for roughly 25-40% of the original length.\n\n"
        f"NOTES:\n---\n{content}\n---"
    )
    return system, user


def build_quiz_prompt(content: str) -> tuple[str, str]:
    system = (
        "You are StudyMate, an expert quiz generator. You create quiz "
        "questions strictly based on the provided study material. You "
        "never introduce facts absent from the material."
    )
    user = (
        "Generate a quiz from the following study notes.\n"
        "Rules:\n"
        "- Produce 5 multiple-choice questions (A-D) plus 2 short-answer "
        "questions.\n"
        "- After all questions, provide an 'Answer Key' section.\n"
        "- Base every question only on the notes below.\n\n"
        f"NOTES:\n---\n{content}\n---"
    )
    return system, user


def build_improve_answer_prompt(content: str) -> tuple[str, str]:
    system = (
        "You are StudyMate, an expert exam-answer coach. You improve a "
        "student's draft answer for clarity, structure, completeness and "
        "correctness, while preserving their original intent and voice "
        "where reasonable."
    )
    user = (
        "The text below is a student's draft answer to some exam or "
        "homework question. Improve it.\n"
        "Rules:\n"
        "- First show the 'Improved Answer'.\n"
        "- Then a short 'What Changed & Why' bullet list.\n"
        "- Point out any factual errors you noticed, if any.\n\n"
        f"DRAFT ANSWER:\n---\n{content}\n---"
    )
    return system, user


def build_explain_concept_prompt(content: str) -> tuple[str, str]:
    system = (
        "You are StudyMate, a patient teacher. You explain concepts at a "
        "level a motivated student can understand, using simple language, "
        "an analogy, and a concrete example."
    )
    user = (
        "Explain the following concept or text so a student can truly "
        "understand it.\n"
        "Rules:\n"
        "- Start with a one-sentence plain-language definition.\n"
        "- Then a short analogy.\n"
        "- Then a concrete worked example.\n"
        "- Keep it focused; avoid unrelated tangents.\n\n"
        f"CONCEPT / TEXT:\n---\n{content}\n---"
    )
    return system, user


def build_study_plan_prompt(content: str) -> tuple[str, str]:
    system = (
        "You are StudyMate, a study-planning assistant. You turn a list "
        "of topics or notes into a realistic, structured study plan."
    )
    user = (
        "Based on the following notes/topics, create a structured study "
        "plan.\n"
        "Rules:\n"
        "- Break the material into logical study sessions.\n"
        "- For each session, list the sub-topics to cover and one "
        "self-check question.\n"
        "- Keep it realistic and specific to the content given.\n\n"
        f"NOTES/TOPICS:\n---\n{content}\n---"
    )
    return system, user


FEATURES = {
    "Summarize Notes": build_summary_prompt,
    "Generate Quiz": build_quiz_prompt,
    "Improve My Answer": build_improve_answer_prompt,
    "Explain Concept": build_explain_concept_prompt,
    "Build Study Plan": build_study_plan_prompt,
}
