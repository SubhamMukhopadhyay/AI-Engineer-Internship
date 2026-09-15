"""
Production RAG — Streamlit Frontend
=====================================
Talks to the FastAPI backend (see ../backend) over HTTP. Set
BACKEND_URL to point at the API (defaults to http://localhost:8000,
or http://backend:8000 inside docker-compose).

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Production RAG", page_icon="🏭", layout="wide")

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    :root {
        --bg: #0f1115; --card: #171a21; --border: #2a2e38;
        --accent: #f0b429; --text: #e7e9ee; --muted: #9aa1ad;
    }
    .stApp { background: var(--bg); }
    .hdr {
        padding: 1.1rem 1.4rem; border-radius: 14px;
        background: linear-gradient(135deg, #241d10 0%, #12141b 100%);
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
    .step-chip {
        display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
        background: rgba(240,180,41,0.15); color: var(--accent);
        font-size: 0.72rem; font-weight: 600; margin: 0.15rem 0.3rem 0.15rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hdr">
        <h1>🏭 Production RAG</h1>
        <p>FastAPI + LangGraph + FAISS — a modular, observable retrieval-augmented QA system.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str):
    return requests.get(f"{BACKEND_URL}{path}", timeout=15)


def api_post(path: str, **kwargs):
    return requests.post(f"{BACKEND_URL}{path}", timeout=90, **kwargs)


def api_delete(path: str):
    return requests.delete(f"{BACKEND_URL}{path}", timeout=15)


# ------------------------------------------------------------------
# Sidebar: backend status + ingestion
# ------------------------------------------------------------------
with st.sidebar:
    st.subheader("⚙️ Backend status")
    try:
        health = api_get("/api/health").json()
        st.success(f"API reachable · model: `{health.get('llm_configured')}`")
        st.caption(f"Embedding backend: {health.get('embedding_backend')}")
        st.caption(f"Indexed chunks: {health.get('index_size')}")
        if not health.get("llm_configured"):
            st.warning("LLM not configured on the backend (.env). Generation will be skipped.")
        backend_reachable = True
    except Exception as e:
        st.error(f"Cannot reach backend at {BACKEND_URL}\n\n{e}")
        backend_reachable = False

    st.divider()
    st.subheader("📄 Upload documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or Markdown", type=["pdf", "txt", "md", "markdown"], accept_multiple_files=True
    )
    ingest_clicked = st.button("📥 Ingest", use_container_width=True, disabled=not backend_reachable)

    st.divider()
    st.subheader("📚 Indexed documents")
    doc_list = []
    if backend_reachable:
        try:
            doc_list = api_get("/api/documents").json().get("documents", [])
        except Exception:
            doc_list = []
    if doc_list:
        for d in doc_list:
            st.caption(f"• {d['filename']} — {d['num_chunks']} chunks")
        if st.button("🗑️ Clear all documents", use_container_width=True):
            api_delete("/api/documents")
            st.rerun()
    else:
        st.caption("No documents indexed yet.")

if ingest_clicked:
    if not uploaded_files:
        st.sidebar.warning("Select at least one file first.")
    else:
        with st.spinner("Uploading and indexing..."):
            try:
                files_payload = [
                    ("files", (f.name, f.getvalue(), "application/octet-stream"))
                    for f in uploaded_files
                ]
                resp = api_post("/api/ingest", files=files_payload)
                if resp.status_code == 200:
                    body = resp.json()
                    st.sidebar.success(f"Indexed {len(body['documents'])} file(s).")
                    for err in body.get("errors", []):
                        st.sidebar.error(err)
                else:
                    st.sidebar.error(f"Ingestion failed ({resp.status_code}): {resp.text[:300]}")
            except requests.exceptions.RequestException as e:
                st.sidebar.error(f"Could not reach backend: {e}")
        st.rerun()

# ------------------------------------------------------------------
# Query
# ------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Ask a question")
    doc_names = {d["filename"]: d["doc_id"] for d in doc_list}
    scope_names = st.multiselect(
        "Limit to specific document(s) (optional)", list(doc_names.keys())
    )
    scoped_ids = [doc_names[n] for n in scope_names] if scope_names else None
    question = st.text_input("Question", placeholder="Ask something about your documents...")
    top_k = st.slider("Chunks to retrieve", 2, 15, 8)
    ask_clicked = st.button("🔍 Ask", type="primary", use_container_width=True, disabled=not backend_reachable)
    st.markdown("</div>", unsafe_allow_html=True)

if ask_clicked:
    q = (question or "").strip()
    if not q:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Running LangGraph workflow: validate → rewrite → retrieve → evaluate → rerank → generate → validate ..."):
            try:
                resp = api_post(
                    "/api/query",
                    json={"question": q, "doc_ids": scoped_ids, "top_k": top_k},
                )
                if resp.status_code == 200:
                    result = resp.json()
                    result["question"] = q
                    st.session_state.history.insert(0, result)
                elif resp.status_code == 400:
                    st.error(resp.json().get("detail", "No documents indexed yet."))
                elif resp.status_code == 422:
                    st.error(f"Invalid query: {resp.json().get('detail')}")
                else:
                    st.error(f"Query failed ({resp.status_code}): {resp.text[:300]}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach backend: {e}")

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Answer")
    if st.session_state.history:
        latest = st.session_state.history[0]
        badge = "✅ Grounded" if latest["grounded"] else "⚠️ Not found in documents"
        valid_badge = "✅ Validated" if latest["valid"] else "⚠️ Validation flagged an issue"
        st.caption(f"{badge} · {valid_badge}")
        if latest.get("rewritten_query") and latest["rewritten_query"] != latest.get("question", ""):
            st.caption(f"Rewritten query: _{latest['rewritten_query']}_")

        st.markdown(f'<div class="answer">{latest["answer"]}</div>', unsafe_allow_html=True)

        if latest.get("validation_notes"):
            st.caption(f"Validator notes: {latest['validation_notes']}")

        st.markdown("**Workflow steps executed:**")
        st.markdown(
            "".join(f'<span class="step-chip">{s}</span>' for s in latest.get("steps", [])),
            unsafe_allow_html=True,
        )

        with st.expander(f"📎 Retrieved sources ({len(latest['sources'])})"):
            for i, s in enumerate(latest["sources"], start=1):
                page_info = f" · page {s['page_number']}" if s.get("page_number") else ""
                rerank_info = f" · rerank {s['rerank_score']:.2f}" if s.get("rerank_score") is not None else ""
                st.markdown(
                    f'<div class="src"><b>[Source {i}]</b> {s["doc_name"]}{page_info} '
                    f"· similarity {s['similarity']:.2f}{rerank_info}<br><br>{s['text']}</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("Ask a question after ingesting documents to see a grounded, validated answer here.")
    st.markdown("</div>", unsafe_allow_html=True)

if len(st.session_state.history) > 1:
    st.markdown("#### 🕘 Previous questions")
    for item in st.session_state.history[1:]:
        with st.expander(item.get("question", "previous question")):
            st.write(item["answer"])

st.caption(
    "This UI is a thin client over the FastAPI backend — all retrieval, "
    "reranking, generation, and validation happen server-side via the "
    "LangGraph workflow."
)
