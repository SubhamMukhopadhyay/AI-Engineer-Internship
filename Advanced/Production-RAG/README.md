# 🏭 Production RAG (Advanced Project)

A production-style, modular RAG system: **FastAPI** backend orchestrated by
a real **LangGraph** workflow, **FAISS** vector search, a **Streamlit**
frontend, typed **Pydantic** models throughout, and **Docker** deployment.

## Architecture

```
Advanced/Production-RAG/
├── backend/
│   ├── app/
│   │   ├── config.py        # pydantic-settings configuration
│   │   ├── models.py        # Pydantic API schemas + typed LangGraph state
│   │   ├── ingestion.py     # multi-format validation + extraction (PDF/TXT/MD)
│   │   ├── preprocessing.py # text cleaning
│   │   ├── chunking.py      # overlapping, boundary-aware chunking
│   │   ├── embeddings.py    # embedding backend (+ offline fallback)
│   │   ├── indexing.py      # FAISS index, persistence, document scoping
│   │   ├── retrieval.py     # retrieval wrapper
│   │   ├── reranking.py     # cross-encoder reranking (+ lexical fallback)
│   │   ├── generation.py    # grounded answer generation
│   │   ├── prompts.py       # all prompt templates
│   │   ├── validation.py    # query validation + LLM-based answer fact-check
│   │   ├── llm_client.py    # provider-agnostic OpenAI-compatible client
│   │   ├── workflow.py      # the LangGraph graph — actually executed
│   │   ├── logging_config.py
│   │   └── main.py          # FastAPI routes
│   ├── tests/test_pipeline.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── streamlit_app.py     # thin client over the FastAPI backend
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
└── .env.example
```

## The LangGraph workflow (actually executed, not just imported)

```
START
  ↓
validate_query        — rejects empty/too-short/too-long questions, no LLM call
  ↓ (invalid → END)
rewrite_query          — LLM rewrites the question into a search-optimized query
  ↓
retrieve_documents     — FAISS similarity search (optionally document-scoped)
  ↓
evaluate_retrieval     — checks whether retrieved similarity is sufficient
  ↓
rerank_improve         — cross-encoder (or lexical fallback) reranks candidates
  ↓
generate_answer        — LLM answers ONLY from reranked context, or reports
  ↓                       "not found in documents"
validate_answer        — a second LLM pass fact-checks the answer against
  ↓                       the context it was given
END
```

State is a typed `RAGState` (`TypedDict`, see `models.py`) threaded through
every node. Each `/api/query` response includes the exact ordered list of
node names that executed (`steps`), so the workflow is observable from the
UI — you can see in the Streamlit app which stages ran for every answer.

## Why each stage exists

- **Query rewriting** improves recall for vague or conversational questions
  before they ever hit the vector index.
- **Retrieval evaluation** is a cheap, explainable gate (similarity
  threshold) that lets the graph reason about whether it has enough to
  work with, before spending an LLM call on generation.
- **Reranking** re-scores the top candidates with a model that looks at the
  (query, chunk) pair jointly — this generally beats bi-encoder similarity
  alone at picking the *most relevant* chunk, not just a similar one.
- **Answer validation** is a second, independent LLM pass whose only job is
  to check the answer against the context — a deliberate second line of
  defence against hallucination beyond the generation prompt's constraints.

## Document scoping

Every chunk carries its `doc_id`. `/api/query` accepts an optional
`doc_ids` list; retrieval then only considers chunks from those documents,
and the returned `sources` always show which document(s) backed the
answer — so multiple unrelated documents never get silently mixed together
unless you deliberately search across all of them.

## Embeddings & reranking: online vs offline

By default, both the embedder (`sentence-transformers/all-MiniLM-L6-v2`)
and reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) are downloaded from
HuggingFace on first run — no LLM API key is needed for either. If the
download fails (fully offline/sandboxed environment), the backend logs a
warning and automatically falls back to a dependency-light local
hashing embedder and a lexical-overlap reranker, so the system still runs
end-to-end. Production deployments should ensure outbound access to
HuggingFace (or pre-bake the models into the Docker image) to get full
semantic quality.

## Installation (without Docker)

```bash
# Backend
cd Advanced/Production-RAG/backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your real LLM key
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd Advanced/Production-RAG/frontend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
BACKEND_URL=http://localhost:8000 streamlit run streamlit_app.py
```

## Running with Docker

```bash
cd Advanced/Production-RAG
cp .env.example .env   # fill in your real LLM key
docker compose up --build
```

- Backend: http://localhost:8000 (docs at `/docs`)
- Frontend: http://localhost:8501

**Note on this repository's build/test environment:** Docker itself could
not be executed inside the sandbox used to build this project (no Docker
daemon available). The `Dockerfile`s and `docker-compose.yml` were
validated for syntax, instruction structure, and consistency with
`requirements.txt`/application imports, but an actual `docker build` /
`docker compose up` run was **not** performed there — this should be
verified in a normal Docker-capable environment before production use.

## Environment variables

| Variable          | Description                                        |
|-------------------|-----------------------------------------------------|
| `LLM_API_KEY`     | Secret key for the generation/validation LLM        |
| `LLM_BASE_URL`    | OpenAI-compatible base URL                           |
| `LLM_MODEL`       | Model name                                           |
| `EMBEDDING_MODEL` | sentence-transformers model name                     |
| `RERANKER_MODEL`  | cross-encoder model name                             |
| `USE_RERANKER`    | Enable/disable the reranking stage                   |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking parameters                     |
| `RETRIEVAL_TOP_K` / `RERANK_TOP_K` | How many chunks to retrieve/keep     |
| `INDEX_DIR` / `UPLOAD_DIR` | Persistence locations                       |
| `MAX_FILE_SIZE_MB`| Upload size limit                                    |
| `BACKEND_URL`     | (frontend only) where the FastAPI backend lives      |

## API endpoints

| Method | Path             | Purpose                              |
|--------|------------------|----------------------------------------|
| GET    | `/api/health`    | Status, LLM config, index size, embedding backend |
| POST   | `/api/ingest`    | Upload + index PDF/TXT/Markdown files  |
| GET    | `/api/documents` | List indexed documents                 |
| DELETE | `/api/documents` | Clear the entire index                 |
| POST   | `/api/query`     | Ask a question; runs the LangGraph workflow |

## Testing

```bash
cd backend
pytest tests/ -v
```

16 tests cover: file validation/extraction (txt/pdf, empty, corrupt,
bad extension), chunking, FAISS indexing + similarity search, document
scoping, query validation, and full FastAPI request/response flows
(ingest → query, missing documents, invalid query, clearing the index).
All tests pass without requiring an LLM API key — generation-dependent
behaviour is exercised via the system's required "LLM not configured"
graceful-degradation path, and the workflow's node execution order is
asserted directly.

## Logging / observability

Structured logging (`logging_config.py`) logs each workflow node's
execution at DEBUG and key lifecycle events (indexing, embedding backend
selection, reranker fallback, retrieval counts) at INFO/WARNING. The
`/api/query` response's `steps` field gives per-request visibility into
exactly which graph nodes ran.

## Known limitations

- Scanned/image-only PDFs are not OCR'd.
- The offline embedding/reranking fallbacks are lexical, not semantic —
  expect meaningfully better retrieval/ranking quality with real network
  access to HuggingFace.
- The FAISS index persists to a local volume (`INDEX_DIR`); there is no
  distributed/sharded index — this is a single-node design appropriate for
  a portfolio-scale production-*style* system, not a multi-tenant SaaS.
- Docker builds were validated statically, not executed, in the sandbox
  used to develop this project (see note above) — validate a real
  `docker compose up` before relying on it.
- Answer validation is itself an LLM call and therefore has the same
  imperfect-reliability characteristics as any LLM-based check; it is a
  second line of defence, not a guarantee.
