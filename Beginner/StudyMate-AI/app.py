"""
StudyMate AI — Beginner Project
=================================
A real AI-powered student utility application built with Streamlit.

Run:
    streamlit run app.py

Configuration is read from environment variables (see .env.example).
No API key is ever hardcoded in this file.
"""

from __future__ import annotations

import os
import time

import streamlit as st
from dotenv import load_dotenv

from llm_client import LLMClient, LLMConfigError, LLMRequestError
from prompts import FEATURES

load_dotenv()

st.set_page_config(
    page_title="StudyMate AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Styling — premium, minimal, no unnecessary animation
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    :root {
        --sm-bg: #0f1115;
        --sm-card: #171a21;
        --sm-border: #2a2e38;
        --sm-accent: #6c8cff;
        --sm-text: #e7e9ee;
        --sm-muted: #9aa1ad;
    }

    .stApp {
        background: var(--sm-bg);
    }

    .sm-header {
        padding: 1.1rem 1.4rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #1b1f2a 0%, #12141b 100%);
        border: 1px solid var(--sm-border);
        margin-bottom: 1.2rem;
    }
    .sm-header h1 {
        margin: 0;
        font-size: 1.6rem;
        color: var(--sm-text);
    }
    .sm-header p {
        margin: 0.25rem 0 0 0;
        color: var(--sm-muted);
        font-size: 0.95rem;
    }

    .sm-card {
        background: var(--sm-card);
        border: 1px solid var(--sm-border);
        border-radius: 14px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 1rem;
    }

    .sm-result {
        background: var(--sm-card);
        border: 1px solid var(--sm-border);
        border-left: 3px solid var(--sm-accent);
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        white-space: pre-wrap;
        line-height: 1.55;
        color: var(--sm-text);
    }

    .sm-badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        background: rgba(108,140,255,0.15);
        color: var(--sm-accent);
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }

    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="sm-header">
        <h1>📚 StudyMate AI</h1>
        <p>Paste your notes, pick a study tool, get a real AI-generated result.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = ""
if "content" not in st.session_state:
    st.session_state.content = ""
if "last_feature" not in st.session_state:
    st.session_state.last_feature = ""

client = LLMClient()

# ---------------------------------------------------------------------
# Sidebar — configuration status & navigation
# ---------------------------------------------------------------------
with st.sidebar:
    st.subheader("⚙️ Configuration")
    if client.is_configured():
        st.success(f"Model configured: `{client.model}`")
    else:
        st.error("LLM environment variables are not fully set.")
        st.caption(
            "Set LLM_API_KEY, LLM_BASE_URL and LLM_MODEL in a `.env` file "
            "(see `.env.example`)."
        )
    st.divider()
    st.subheader("🧭 Features")
    for name in FEATURES:
        st.caption(f"• {name}")

# ---------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="sm-card">', unsafe_allow_html=True)
    st.markdown("#### 1. Your content")
    content = st.text_area(
        "Paste your notes, question, or draft answer",
        value=st.session_state.content,
        height=280,
        placeholder="Paste study notes, a concept, or your draft answer here...",
        label_visibility="collapsed",
    )
    st.session_state.content = content

    st.markdown("#### 2. Choose a study tool")
    feature = st.selectbox(
        "Feature",
        list(FEATURES.keys()),
        label_visibility="collapsed",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        run_clicked = st.button("✨ Generate", use_container_width=True, type="primary")
    with col_b:
        clear_clicked = st.button("🗑️ Clear / Reset", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if clear_clicked:
    st.session_state.content = ""
    st.session_state.result = ""
    st.session_state.last_feature = ""
    st.rerun()

if run_clicked:
    stripped = content.strip()
    if not stripped:
        st.warning("Please enter some content before generating a result.")
    elif len(stripped) < 10:
        st.warning(
            "That content looks too short to work with. Please paste more "
            "detail (at least a sentence or two)."
        )
    elif len(stripped) > 20000:
        st.warning(
            "That content is very long (>20,000 characters). Please shorten "
            "it so the request stays within model limits."
        )
    elif not client.is_configured():
        st.error(
            "LLM is not configured. Set LLM_API_KEY, LLM_BASE_URL and "
            "LLM_MODEL in your `.env` file, then restart the app."
        )
    else:
        build_prompt = FEATURES[feature]
        system_prompt, user_prompt = build_prompt(stripped)
        with st.spinner(f"Generating: {feature} ..."):
            try:
                start = time.time()
                output = client.chat(system_prompt, user_prompt)
                elapsed = time.time() - start
                st.session_state.result = output
                st.session_state.last_feature = feature
                st.toast(f"Done in {elapsed:.1f}s", icon="✅")
            except LLMConfigError as e:
                st.error(f"Configuration error: {e}")
            except LLMRequestError as e:
                st.error(f"AI request failed: {e}")
            except Exception as e:  # pragma: no cover - last-resort safety net
                st.error(f"Unexpected error: {e}")

with right:
    st.markdown('<div class="sm-card">', unsafe_allow_html=True)
    st.markdown("#### 3. Result")
    if st.session_state.result:
        st.markdown(
            f'<span class="sm-badge">{st.session_state.last_feature}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="sm-result">{st.session_state.result}</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            "⬇️ Copy / Download result as .txt",
            data=st.session_state.result,
            file_name="studymate_result.txt",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.info(
            "Your AI-generated result will appear here after you click "
            "**Generate**.\n\nNothing is shown until a real response comes "
            "back from the configured LLM API."
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.caption(
    "StudyMate AI calls a real LLM API for every result — there are no "
    "hardcoded or canned responses. If the API is not configured or fails, "
    "you will see an explicit error message instead of fake output."
)
