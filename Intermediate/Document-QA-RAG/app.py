"""
Document QA RAG — Intermediate Project
=========================================
Real retrieval-augmented QA over uploaded PDF/TXT documents.

Pipeline:
Upload -> Validate -> Extract -> Clean -> Chunk -> Embed -> FAISS index
       -> Retrieve -> Generate grounded answer -> Show answer + sources

Run:
    streamlit run app.py
"""

from __future__ import annotations

import time
import uuid

import streamlit as st
from dotenv import load_dotenv

from core.extraction import DocumentError, extract_pages
from core.chunking import chunk_pages
from core.vectorstore import VectorStore
from core.llm_client import LLMClient, LLMConfigError, LLMRequestError
from core.generation import generate_answer

load_dotenv()

st.set_page_config(page_title="Document QA RAG", page_icon="🔎", layout="wide")

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    :root {
        --bg: #0f1115; --card: #171a21; --border: #2a2e38;
        --accent: #34d399; --text: #e7e9ee; --muted: #9aa1ad;
    }
    .stApp { background: var(--bg); }
    .hdr {
        padding: 1.1rem 1.4rem; border-radius: 14px;
        background: linear-gradient(135deg, #142019 0%, #12141b 100%);
        border: 1px solid var(--border); margin-bottom: 1.2rem;
    }
    .hdr h1 { margin: 0; font-size: 1.6rem; color: var(--text); }
    .hdr p { margin: 0.25rem 0 0 0; color: var(--muted); font-size: 0.95rem; }
    .card {
        background: var(--card); border: 1px solid var(--border);
        border-radius: 14px; padding: 1.1rem 1.3rem; margin-bottom: 1rem;
    }
    .answer {
        background: var(--card); border: 1px solid var(--border);
        border-left: 3px solid var(--accent); border-radius: 10px;
        padding: 1.1rem 1.3rem; white-space: pre-wrap; line-height: 1.55;
        color: var(--text);
    }
    .src {
        background: #12141b; border: 1px solid var(--border);
        border-radius: 8px; padding: 0.7rem 0.9rem; margin-bottom: 0.5rem;
        font-size: 0.85rem; color: var(--muted); white-space: pre-wrap;
    }
    .doc-chip {
        display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
        background: rgba(52,211,153,0.15); color: var(--accent);
        font-size: 0.75rem; font-weight: 600; margin: 0.15rem 0.3rem 0.15rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hdr">
        <h1>🔎 Document QA RAG</h1>
        <p>Upload PDF/TXT documents and ask questions grounded strictly in their content.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "store" not in st.session_state:
    st.session_state.store = VectorStore()
if "doc_meta" not in st.session_state:
    st.session_state.doc_meta = {}  # doc_id -> {"name":..., "chunks": n}
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: question, answer, sources, grounded

client = LLMClient()

# ------------------------------------------------------------------
# Sidebar: upload + config status
# ------------------------------------------------------------------
with st.sidebar:
    st.subheader("⚙️ Configuration")
    if client.is_configured():
        st.success(f"Model: `{client.model}`")
    else:
        st.error("LLM environment variables are not fully set (see .env.example).")

    st.divider()
    st.subheader("📄 Upload documents")
    uploaded_files = st.file_uploader(
        "PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True
    )
    ingest_clicked = st.button("📥 Ingest uploaded files", use_container_width=True)

    st.divider()
    st.subheader("📚 Indexed documents")
    if st.session_state.doc_meta:
        for did, meta in st.session_state.doc_meta.items():
            st.caption(f"• {meta['name']} — {meta['chunks']} chunks")
    else:
        st.caption("No documents indexed yet.")

    if st.session_state.doc_meta:
        if st.button("🗑️ Clear all documents", use_container_width=True):
            st.session_state.store = VectorStore()
            st.session_state.doc_meta = {}
            st.session_state.history = []
            st.rerun()

# ------------------------------------------------------------------
# Ingestion
# ------------------------------------------------------------------
if ingest_clicked:
    if not uploaded_files:
        st.sidebar.warning("Please select at least one PDF or TXT file first.")
    else:
        progress = st.sidebar.progress(0.0, text="Starting ingestion...")
        total = len(uploaded_files)
        errors = []
        for i, f in enumerate(uploaded_files, start=1):
            try:
                raw = f.getvalue()
                pages = extract_pages(f.name, raw)
                doc_id = str(uuid.uuid4())[:8]
                chunks = chunk_pages(doc_id, f.name, pages)
                if not chunks:
                    raise DocumentError(f"'{f.name}' produced no usable chunks.")
                st.session_state.store.add_chunks(chunks)
                st.session_state.doc_meta[doc_id] = {
                    "name": f.name,
                    "chunks": len(chunks),
                }
            except DocumentError as e:
                errors.append(str(e))
            except Exception as e:
                errors.append(f"Unexpected error processing '{f.name}': {e}")
            progress.progress(i / total, text=f"Processed {i}/{total} files")
        progress.empty()
        if errors:
            for e in errors:
                st.sidebar.error(e)
        else:
            st.sidebar.success(f"Ingested {total} file(s) successfully.")

# ------------------------------------------------------------------
# Q&A
# ------------------------------------------------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Ask a question")

    doc_options = {meta["name"]: did for did, meta in st.session_state.doc_meta.items()}
    scope_names = st.multiselect(
        "Limit to specific document(s) (optional — leave empty to search all)",
        list(doc_options.keys()),
    )
    scoped_doc_ids = [doc_options[n] for n in scope_names] if scope_names else None

    question = st.text_input(
        "Question", placeholder="What does the document say about ...?"
    )
    top_k = st.slider("Number of chunks to retrieve", 2, 10, 4)
    ask_clicked = st.button("🔍 Ask", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if ask_clicked:
    q = question.strip()
    if not q:
        st.warning("Please enter a question.")
    elif st.session_state.store.is_empty():
        st.warning("Please upload and ingest at least one document first.")
    elif not client.is_configured():
        st.error("LLM is not configured. Set LLM_API_KEY, LLM_BASE_URL, LLM_MODEL.")
    else:
        with st.spinner("Retrieving relevant chunks..."):
            try:
                retrieved = st.session_state.store.search(
                    q, top_k=top_k, doc_ids=scoped_doc_ids
                )
            except Exception as e:
                retrieved = []
                st.error(f"Retrieval failed: {e}")

        if not retrieved:
            st.error(
                "No relevant chunks were retrieved. Try rephrasing the "
                "question or check that the right documents are selected."
            )
        else:
            with st.spinner("Generating grounded answer..."):
                try:
                    start = time.time()
                    answer, grounded = generate_answer(client, q, retrieved)
                    elapsed = time.time() - start
                    st.session_state.history.insert(
                        0,
                        {
                            "question": q,
                            "answer": answer,
                            "sources": retrieved,
                            "grounded": grounded,
                            "elapsed": elapsed,
                        },
                    )
                except LLMConfigError as e:
                    st.error(f"Configuration error: {e}")
                except LLMRequestError as e:
                    st.error(f"AI request failed: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Answer")
    if st.session_state.history:
        latest = st.session_state.history[0]
        badge = "✅ Grounded in documents" if latest["grounded"] else "⚠️ Not found in documents"
        st.caption(f"{badge} · {latest['elapsed']:.1f}s")
        st.markdown(f'<div class="answer">{latest["answer"]}</div>', unsafe_allow_html=True)

        with st.expander(f"📎 Retrieved sources ({len(latest['sources'])})", expanded=False):
            for i, (chunk, score) in enumerate(latest["sources"], start=1):
                page_info = f" · page {chunk.page_number}" if chunk.page_number else ""
                st.markdown(
                    f'<div class="src"><b>[Source {i}]</b> {chunk.doc_name}{page_info} '
                    f"· similarity {score:.2f}<br><br>{chunk.text}</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("Ask a question after ingesting documents to see a grounded answer here.")
    st.markdown("</div>", unsafe_allow_html=True)

if len(st.session_state.history) > 1:
    st.markdown("#### 🕘 Previous questions")
    for item in st.session_state.history[1:]:
        with st.expander(item["question"]):
            st.write(item["answer"])

st.caption(
    "Answers are generated only from retrieved document chunks. If the "
    "documents don't contain the answer, the app says so explicitly "
    "instead of falling back to general knowledge."
)
