"""
models.py
---------
All typed Pydantic models shared across the backend: API request/response
schemas, internal data records, and the typed LangGraph workflow state.

Centralising these makes the FastAPI layer, the workflow, and the
retrieval/generation modules all speak the same contract, and gives
FastAPI automatic request validation for free.
"""

from __future__ import annotations

from typing import Optional, TypedDict

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------
# Documents / chunks
# ---------------------------------------------------------------------

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int
    file_type: str


class SourceChunk(BaseModel):
    doc_id: str
    doc_name: str
    chunk_id: str
    page_number: int = 0
    text: str
    similarity: float
    rerank_score: Optional[float] = None


# ---------------------------------------------------------------------
# API requests / responses
# ---------------------------------------------------------------------

class IngestResponse(BaseModel):
    documents: list[DocumentInfo]
    errors: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str
    doc_ids: Optional[list[str]] = None
    top_k: int = Field(default=8, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question must not be empty")
        return v.strip()


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    valid: bool
    validation_notes: str = ""
    rewritten_query: str
    sources: list[SourceChunk]
    steps: list[str]  # ordered list of workflow node names that executed


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool
    index_size: int
    embedding_backend: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


# ---------------------------------------------------------------------
# LangGraph typed state
# ---------------------------------------------------------------------

class RAGState(TypedDict, total=False):
    # input
    question: str
    doc_ids: Optional[list[str]]
    top_k: int

    # workflow-produced fields
    query_valid: bool
    query_error: str
    rewritten_query: str
    retrieved: list[dict]          # list of {chunk fields..., similarity}
    retrieval_sufficient: bool
    reranked: list[dict]           # list of {chunk fields..., similarity, rerank_score}
    answer: str
    grounded: bool
    answer_valid: bool
    validation_notes: str
    steps: list[str]
