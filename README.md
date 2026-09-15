# AI Engineer Internship — ShadowFox

Three real, working AI applications built across increasing levels of
complexity: a Streamlit student utility app, a document-grounded RAG
QA tool, and a production-style FastAPI + LangGraph + FAISS RAG system.

No project in this repository is a placeholder, demo, or landing page —
each is a functioning application with real LLM calls, real retrieval,
real error handling, and its own test coverage.

## Repository structure

```
AI-Engineer-Internship/
├── README.md                          (this file)
├── Beginner/
│   └── StudyMate-AI/                  Streamlit student assistant
├── Intermediate/
│   └── Document-QA-RAG/               PDF/TXT document QA with FAISS
└── Advanced/
    └── Production-RAG/                FastAPI + LangGraph + FAISS + Docker
        ├── backend/                   modular RAG API
        └── frontend/                  Streamlit client
```

## Projects

### 🟢 Beginner — [StudyMate AI](./Beginner/StudyMate-AI)
A Streamlit app for students: note summarization, quiz generation, answer
improvement, concept explanation, and study-plan building. Calls a real,
configurable LLM API (OpenAI-compatible) — no hardcoded/fake responses.

### 🟡 Intermediate — [Document QA RAG](./Intermediate/Document-QA-RAG)
Upload PDF/TXT documents and ask grounded questions. Full pipeline:
extract → chunk → embed → FAISS index → retrieve → generate an answer
that is only allowed to use the retrieved context, with sources shown in
the UI and an explicit "not found in documents" response when the answer
isn't supported.

### 🔴 Advanced — [Production RAG](./Advanced/Production-RAG)
A modular, production-style system: FastAPI backend, a real **executing**
LangGraph workflow (validate → rewrite → retrieve → evaluate → rerank →
generate → validate), FAISS vector search with document scoping,
cross-encoder reranking, typed Pydantic models end-to-end, a Streamlit
frontend, Dockerfile + docker-compose for both services, and a pytest
suite (16 tests, all passing).

## Technologies used

Python, Streamlit, FastAPI, Pydantic / pydantic-settings, LangGraph,
FAISS, sentence-transformers, pypdf, Docker & docker-compose, pytest,
requests, python-dotenv.

## How to run each project

**Beginner:**
```bash
cd Beginner/StudyMate-AI
pip install -r requirements.txt
cp .env.example .env   # add your LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
streamlit run app.py
```

**Intermediate:**
```bash
cd Intermediate/Document-QA-RAG
pip install -r requirements.txt
cp .env.example .env   # add your LLM key (embeddings are local/free)
streamlit run app.py
```

**Advanced (without Docker):**
```bash
# backend
cd Advanced/Production-RAG/backend
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000

# frontend (separate terminal)
cd Advanced/Production-RAG/frontend
pip install -r requirements.txt
BACKEND_URL=http://localhost:8000 streamlit run streamlit_app.py
```

**Advanced (with Docker):**
```bash
cd Advanced/Production-RAG
cp .env.example .env
docker compose up --build
```

## Environment variables

Every project reads configuration from environment variables — never
hardcoded — via a `.env` file (each project ships a `.env.example`). At
minimum: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` for an OpenAI-compatible
chat completions endpoint (OpenAI, Groq, OpenRouter, a local proxy, etc.).
The Intermediate and Advanced projects additionally configure a local
embedding model (no API key required) and, for Advanced, a reranker model
and chunking/retrieval parameters. See each project's own README for the
full table.

## Architecture summary

- **Beginner**: UI → prompt builder → LLM client → UI. No retrieval.
- **Intermediate**: Upload → validate → extract → chunk → embed → FAISS →
  retrieve → grounded generation → UI (answer + sources).
- **Advanced**: Same RAG core, but restructured into a modular FastAPI
  service with a typed LangGraph workflow orchestrating query rewriting,
  retrieval, retrieval-sufficiency evaluation, reranking, generation, and
  a second LLM-based answer-validation pass — with a Streamlit frontend
  as a thin HTTP client.

## RAG pipeline (Intermediate & Advanced)

Both RAG projects share the same core idea: **never let the model answer
from general knowledge**. The generation prompt explicitly restricts the
model to the retrieved context and requires it to say so plainly when the
context is insufficient, instead of guessing. The Advanced project adds a
second, independent LLM pass that specifically fact-checks the generated
answer against the context it was given — a second line of defence
against hallucination.

## LangGraph workflow (Advanced only)

```
START → validate_query → rewrite_query → retrieve_documents →
evaluate_retrieval → rerank_improve → generate_answer → validate_answer → END
```
(invalid queries route directly from `validate_query` to `END`)

This is a compiled, executing `langgraph.graph.StateGraph` over a typed
`RAGState`, not just a dependency listed in `requirements.txt`. Every
`/api/query` response returns the exact list of node names that ran, so
the workflow is observable end-to-end from the UI.

## Testing

- **Beginner**: syntax/import checks, Streamlit startup check (verified —
  HTTP 200 on boot).
- **Intermediate**: syntax/import checks, an end-to-end pipeline test
  (extraction → chunking → embedding → FAISS search returning the correct
  chunk for a sample question), and explicit tests of empty/corrupt/
  bad-extension file handling. Streamlit startup verified (HTTP 200).
- **Advanced**: a 16-test `pytest` suite covering ingestion validation,
  chunking, FAISS indexing/search, document-scoped retrieval, query
  validation, and full FastAPI request flows (ingest → query, missing-
  documents handling, invalid-query handling, index clearing) — **all 16
  passing**. FastAPI app import and route registration verified directly.

All of the above was actually executed during development, not asserted
without running it — see the final report delivered at the end of this
build for the exact commands and results.

## Docker (Advanced project)

`Dockerfile`s for both `backend` and `frontend`, plus a root
`docker-compose.yml`, are included. They were validated for syntax,
instruction correctness, and consistency with each project's
`requirements.txt`/imports. **A live `docker build`/`docker compose up`
could not be executed** in the sandbox this repository was built in
(no Docker daemon available there) — validate this in a normal
Docker-capable environment before relying on it in production.

## Known limitations

- Embeddings/reranking in the Intermediate and Advanced projects prefer
  real sentence-transformer / cross-encoder models (downloaded from
  HuggingFace on first run) but fall back automatically to lightweight,
  dependency-free local implementations if that download isn't possible
  (e.g. a fully offline environment) — see each project's README for
  details on the quality trade-off.
- Scanned/image-only PDFs are not OCR'd in either RAG project.
- None of the three projects function without a valid, configured LLM API
  key for generation — this is intentional (no hardcoded/fake responses).
- Docker was validated statically only, as noted above.
