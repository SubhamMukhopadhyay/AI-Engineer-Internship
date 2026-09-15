"""
main.py
-------
FastAPI application exposing the Production RAG system:

    POST /api/ingest    - upload and index PDF/TXT/Markdown documents
    POST /api/query     - ask a grounded question (runs the LangGraph workflow)
    GET  /api/documents  - list indexed documents
    DELETE /api/documents - clear the index
    GET  /api/health    - health/status check
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.chunking import chunk_pages
from app.config import get_settings
from app.embeddings import embedding_backend_name
from app.indexing import get_index
from app.ingestion import IngestionError, detect_file_type, extract_pages
from app.llm_client import LLMClient
from app.logging_config import configure_logging, get_logger
from app.models import (
    DocumentInfo,
    DocumentListResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from app.workflow import run_workflow

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)

app = FastAPI(
    title="Production RAG API",
    description="A production-style, LangGraph-orchestrated RAG backend.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    index = get_index()
    return HealthResponse(
        status="ok",
        llm_configured=LLMClient().is_configured(),
        index_size=len(index.chunks),
        embedding_backend=embedding_backend_name(),
    )


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...)) -> IngestResponse:
    index = get_index()
    documents: list[DocumentInfo] = []
    errors: list[str] = []

    for f in files:
        try:
            raw = await f.read()
            pages = extract_pages(f.filename, raw)
            doc_id = str(uuid.uuid4())[:8]
            file_type = detect_file_type(f.filename)
            chunks = chunk_pages(
                doc_id, f.filename, pages, is_markdown=(file_type == "markdown")
            )
            if not chunks:
                raise IngestionError(f"'{f.filename}' produced no usable chunks.")
            index.add_chunks(chunks)
            documents.append(
                DocumentInfo(
                    doc_id=doc_id,
                    filename=f.filename,
                    num_chunks=len(chunks),
                    file_type=file_type,
                )
            )
        except IngestionError as e:
            errors.append(str(e))
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("Unexpected ingestion error for %s", f.filename)
            errors.append(f"Unexpected error processing '{f.filename}': {e}")

    if not documents and errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    return IngestResponse(documents=documents, errors=errors)


@app.get("/api/documents", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    index = get_index()
    docs = [
        DocumentInfo(doc_id=did, filename=meta["name"], num_chunks=meta["chunks"], file_type="")
        for did, meta in index.documents_summary().items()
    ]
    return DocumentListResponse(documents=docs)


@app.delete("/api/documents")
def clear_documents() -> dict:
    index = get_index()
    index.clear()
    return {"status": "cleared"}


@app.post("/api/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    index = get_index()
    if index.is_empty():
        raise HTTPException(
            status_code=400,
            detail="No documents are indexed yet. Ingest documents before querying.",
        )

    try:
        final_state = run_workflow(payload.question, payload.doc_ids, payload.top_k)
    except Exception as e:
        logger.exception("Workflow execution failed")
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {e}")

    if not final_state.get("query_valid", True):
        raise HTTPException(status_code=422, detail=final_state.get("query_error", "Invalid query."))

    sources_data = final_state.get("reranked") or final_state.get("retrieved") or []
    sources = [
        SourceChunk(
            doc_id=s["doc_id"],
            doc_name=s["doc_name"],
            chunk_id=s["chunk_id"],
            page_number=s.get("page_number", 0),
            text=s["text"],
            similarity=s["similarity"],
            rerank_score=s.get("rerank_score"),
        )
        for s in sources_data
    ]

    return QueryResponse(
        answer=final_state.get("answer", ""),
        grounded=final_state.get("grounded", False),
        valid=final_state.get("answer_valid", True),
        validation_notes=final_state.get("validation_notes", ""),
        rewritten_query=final_state.get("rewritten_query", payload.question),
        sources=sources,
        steps=final_state.get("steps", []),
    )
