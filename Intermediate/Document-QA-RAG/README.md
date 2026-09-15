# 🔎 Document QA RAG (Intermediate Project)

A real retrieval-augmented question-answering app over uploaded PDF/TXT
documents, built with Streamlit + FAISS.

## Pipeline

```
Upload → Validate → Extract text → Clean/preprocess → Chunk →
Embed → Store in FAISS → Retrieve top-k → Generate grounded answer →
Display answer + retrieved sources
```

## Features

- PDF and TXT upload (multi-file)
- File validation: extension, size, empty-file, corrupt/encrypted PDF handling
- Text extraction with per-page tracking (for PDFs)
- Chunking with configurable size/overlap and sentence-boundary snapping
- Local embeddings + a FAISS `IndexFlatIP` (cosine) similarity index
- Document-scoped retrieval (limit a query to specific uploaded files)
- Grounded generation: the LLM is instructed to answer **only** from
  retrieved context
- Explicit "document does not contain enough information" response when
  the answer isn't supported
- Retrieved source chunks (with document name, page number, similarity
  score) shown in an expandable panel
- Empty-query validation, retrieval-failure handling, LLM-failure handling

## Architecture

```
app.py                  Streamlit UI + orchestration
core/extraction.py      File validation + PDF/TXT text extraction
core/chunking.py        Text cleaning + overlapping chunking
core/vectorstore.py     Embeddings + FAISS index + similarity search
core/local_embedder.py  Zero-network fallback embedder
core/generation.py      Grounded-answer prompt + hallucination-guard parsing
core/llm_client.py      Provider-agnostic OpenAI-compatible chat client
```

## Chunk size & overlap

Default `CHUNK_SIZE=900` characters, `CHUNK_OVERLAP=150` characters
(configurable via `.env`). Chunking tries to break on a sentence boundary
(`". "`) near the target size instead of a hard character cut, so chunks
stay readable. Overlap exists so a fact sitting near a chunk boundary is
still fully contained in at least one chunk — without it, retrieval can
miss facts that get split across two chunks.

## Embedding model

By default this project uses **sentence-transformers** (`all-MiniLM-L6-v2`,
configurable via `EMBEDDING_MODEL`) — a free, local model downloaded from
HuggingFace on first run. **No LLM API key is used for embeddings.**

If the sentence-transformers model cannot be downloaded (e.g. a fully
offline/sandboxed environment), the app automatically falls back to
`core/local_embedder.py`, a dependency-light hashed n-gram embedder that
requires zero network access. This keeps the whole pipeline runnable
end-to-end even without internet access, at the cost of weaker semantic
matching (it's closer to lexical/bag-of-words similarity than true
semantic embedding). For real use, keep internet access available so the
real transformer model loads.

## Vector search

FAISS `IndexFlatIP` over L2-normalized vectors = cosine similarity.
Retrieval optionally over-fetches and filters by `doc_id` to support
document-scoped queries without needing a separate index per document.

## Retrieval process

1. Embed the user's question with the same embedding backend used for chunks.
2. Search the FAISS index for the top-k most similar chunks (optionally
   restricted to selected documents).
3. Pass the retrieved chunks — labeled `[Source 1]`, `[Source 2]`, ... with
   document name/page/similarity — into the generation prompt.

## Grounding & hallucination reduction

The generation system prompt (`core/generation.py`) explicitly instructs
the model to:

- Answer **only** using the supplied context, never general knowledge
- Never invent facts absent from the context
- Emit a specific marker (`DOCUMENT_DOES_NOT_CONTAIN_ANSWER`) as the first
  line of its reply when the context is insufficient, which the app
  detects and turns into a clear "⚠️ not found in documents" UI state
  instead of letting the model guess
- Cite which `[Source N]` supports each claim

## Installation

```bash
cd Intermediate/Document-QA-RAG
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env with your real LLM key
```

## Environment variables

| Variable         | Description                                    |
|-------------------|------------------------------------------------|
| `LLM_API_KEY`     | Secret key for answer-generation LLM           |
| `LLM_BASE_URL`    | OpenAI-compatible base URL                     |
| `LLM_MODEL`       | Model name for generation                      |
| `EMBEDDING_MODEL` | sentence-transformers model name               |
| `CHUNK_SIZE`      | Characters per chunk (default 900)             |
| `CHUNK_OVERLAP`   | Overlap characters between chunks (default 150)|

## Run

```bash
streamlit run app.py
```

## Limitations

- Scanned/image-only PDFs are not OCR'd — extraction will report no text.
- The offline fallback embedder is lexical, not semantic; expect noticeably
  better retrieval quality when the real sentence-transformers model can
  be downloaded.
- The FAISS index is in-memory per session only (no persistence to disk
  between app restarts) — this is intentional for the intermediate scope;
  the Advanced project adds a production-oriented architecture.
- Answer quality depends on the configured LLM's instruction-following.
