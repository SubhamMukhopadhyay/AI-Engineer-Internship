"""
test_pipeline.py
-----------------
Tests for ingestion, chunking, indexing/retrieval, workflow routing, and
the FastAPI endpoints. These do not require an LLM API key — generation-
dependent behaviour is tested via the graceful "not configured" path,
which is itself a required behaviour of the system.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from app.ingestion import IngestionError, extract_pages
from app.chunking import chunk_pages
from app.indexing import VectorIndex
from app.validation import validate_query


SAMPLE_TEXT = (
    b"The mitochondria is the powerhouse of the cell and generates ATP "
    b"through cellular respiration. Chloroplasts perform photosynthesis "
    b"in plant cells using sunlight, water, and carbon dioxide."
)


# ---------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------

def test_extract_txt_ok():
    pages = extract_pages("bio.txt", SAMPLE_TEXT)
    assert len(pages) == 1
    assert "mitochondria" in pages[0].text


def test_extract_empty_file_raises():
    with pytest.raises(IngestionError):
        extract_pages("empty.txt", b"")


def test_extract_bad_extension_raises():
    with pytest.raises(IngestionError):
        extract_pages("virus.exe", b"binary-content")


def test_extract_corrupt_pdf_raises():
    with pytest.raises(IngestionError):
        extract_pages("bad.pdf", b"this is not a pdf")


# ---------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------

def test_chunking_produces_overlap():
    pages = extract_pages("bio.txt", SAMPLE_TEXT)
    chunks = chunk_pages("doc1", "bio.txt", pages, chunk_size=80, chunk_overlap=20)
    assert len(chunks) >= 2
    assert all(c.doc_id == "doc1" for c in chunks)


def test_chunking_empty_pages_returns_empty():
    chunks = chunk_pages("doc1", "empty.txt", [])
    assert chunks == []


# ---------------------------------------------------------------------
# Indexing / retrieval
# ---------------------------------------------------------------------

def test_index_and_search(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    from app.config import get_settings

    get_settings.cache_clear()
    index = VectorIndex()
    pages = extract_pages("bio.txt", SAMPLE_TEXT)
    chunks = chunk_pages("doc1", "bio.txt", pages, chunk_size=80, chunk_overlap=20)
    index.add_chunks(chunks)

    assert not index.is_empty()
    results = index.search("What produces ATP in the cell?", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert "mitochondria" in top_chunk.text.lower() or "atp" in top_chunk.text.lower()
    get_settings.cache_clear()


def test_document_scoped_search(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index2"))
    from app.config import get_settings

    get_settings.cache_clear()
    index = VectorIndex()
    pages_a = extract_pages("a.txt", SAMPLE_TEXT)
    pages_b = extract_pages("b.txt", b"Paris is the capital of France and is known for the Eiffel Tower.")
    chunks_a = chunk_pages("docA", "a.txt", pages_a)
    chunks_b = chunk_pages("docB", "b.txt", pages_b)
    index.add_chunks(chunks_a)
    index.add_chunks(chunks_b)

    scoped = index.search("capital city", top_k=5, doc_ids=["docA"])
    assert all(c.doc_id == "docA" for c, _ in scoped)
    get_settings.cache_clear()


# ---------------------------------------------------------------------
# Query validation
# ---------------------------------------------------------------------

def test_validate_query_rejects_empty():
    ok, err = validate_query("   ")
    assert not ok
    assert err


def test_validate_query_accepts_normal():
    ok, err = validate_query("What is the capital of France?")
    assert ok
    assert err == ""


# ---------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "api_index"))
    from app.config import get_settings

    get_settings.cache_clear()
    from app.indexing import reset_index_singleton

    reset_index_singleton()
    from app.main import app

    with TestClient(app) as c:
        yield c
    reset_index_singleton()
    get_settings.cache_clear()


def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "llm_configured" in body


def test_ingest_and_query_flow(client):
    r = client.post(
        "/api/ingest", files=[("files", ("cell.txt", SAMPLE_TEXT, "text/plain"))]
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["documents"]) == 1
    assert body["errors"] == []

    r = client.get("/api/documents")
    assert r.status_code == 200
    assert len(r.json()["documents"]) == 1

    r = client.post("/api/query", json={"question": "What does the mitochondria do?"})
    assert r.status_code == 200
    body = r.json()
    # Workflow must execute all seven nodes in order regardless of LLM config.
    assert body["steps"] == [
        "validate_query",
        "rewrite_query",
        "retrieve_documents",
        "evaluate_retrieval",
        "rerank_improve",
        "generate_answer",
        "validate_answer",
    ]
    assert len(body["sources"]) > 0


def test_query_without_documents_returns_400(client):
    r = client.post("/api/query", json={"question": "Anything?"})
    assert r.status_code == 400


def test_query_empty_question_returns_422(client):
    r = client.post("/api/query", json={"question": "   "})
    assert r.status_code == 422


def test_ingest_bad_extension_returns_400(client):
    r = client.post(
        "/api/ingest", files=[("files", ("malware.exe", b"binary", "application/octet-stream"))]
    )
    assert r.status_code == 400


def test_clear_documents(client):
    client.post("/api/ingest", files=[("files", ("cell.txt", SAMPLE_TEXT, "text/plain"))])
    r = client.delete("/api/documents")
    assert r.status_code == 200
    r = client.get("/api/documents")
    assert r.json()["documents"] == []
