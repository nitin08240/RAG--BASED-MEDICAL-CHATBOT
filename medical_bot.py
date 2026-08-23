import os
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_MODELS = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "llama-3.1-70b-versatile"]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Consult — Medical Reference Assistant",
    page_icon="\U0001FA7A",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design tokens — clinical chart / lab-note identity
#   Background: pale mint paper, not the generic AI-cream
#   Accent:     deep teal (trust, clinical) + muted amber (used sparingly,
#               for the "you" side of the conversation only)
#   Type:       Source Serif 4 for headers (journal feel), IBM Plex Sans for
#               body, IBM Plex Mono for citations / stats (lab readout feel)
# ---------------------------------------------------------------------------
CSS = """
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
    --paper: #EEF4F1;
    --card: #FFFFFF;
    --ink: #1C2B29;
    --ink-soft: #4B5F5B;
    --teal: #1F6F64;
    --teal-dark: #14504A;
    --amber: #C98A3E;
    --border: #D6E3DE;
    --danger: #B84C4C;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--paper) !important;
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--ink);
}

[data-testid="stHeader"] { background: transparent; }

/* ---- Sidebar: styled like a patient chart clipboard ---- */
[data-testid="stSidebar"] {
    background-color: var(--teal-dark) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: #EAF3F0 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: 'Source Serif 4', serif;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }

.stat-chip {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
    font-family: 'IBM Plex Mono', monospace;
}
.stat-chip .label {
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    opacity: 0.75;
}
.stat-chip .value {
    font-size: 1.25rem;
    font-weight: 500;
    color: #7FDCC9 !important;
}

/* ---- Header ---- */
.consult-header {
    font-family: 'Source Serif 4', serif;
    font-weight: 700;
    font-size: 2.1rem;
    color: var(--ink);
    margin-bottom: 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.consult-sub {
    color: var(--ink-soft);
    font-size: 0.95rem;
    margin-top: 2px;
    margin-bottom: 6px;
}

/* pulse divider — the signature element */
.pulse-line {
    width: 100%;
    height: 22px;
    margin: 6px 0 22px 0;
    background-image: repeating-linear-gradient(
        to right,
        var(--border) 0px, var(--border) 2px, transparent 2px, transparent 8px
    );
    background-position: center;
    background-size: 100% 1px;
    background-repeat: no-repeat;
    position: relative;
}
.pulse-line::before {
    content: "";
    position: absolute;
    left: 0; top: 50%;
    width: 100%; height: 2px;
    background: var(--teal);
    clip-path: polygon(
        0% 50%, 38% 50%, 42% 10%, 46% 90%, 50% 50%, 100% 50%, 100% 52%, 0% 52%
    );
    opacity: 0.55;
}

/* ---- Chat bubbles ---- */
[data-testid="stChatMessage"] {
    background: transparent;
    padding: 4px 0;
}
[data-testid="stChatMessageContent"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
}

/* Quick-question chips */
.stButton>button {
    background: var(--card);
    border: 1px solid var(--border);
    color: var(--teal-dark);
    border-radius: 20px;
    font-size: 0.85rem;
    padding: 4px 14px;
}
.stButton>button:hover {
    border-color: var(--teal);
    color: var(--teal);
}

/* Source citation chip */
.source-chip {
    border-left: 3px solid var(--teal);
    background: #F5FAF8;
    border-radius: 4px;
    padding: 8px 12px;
    margin-bottom: 8px;
    font-size: 0.85rem;
}
.source-chip .src-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--teal-dark);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 3px;
}

.disclaimer-banner {
    background: #FBF0E4;
    border: 1px solid #E8CDA0;
    color: #6B4A1E;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.82rem;
    margin-bottom: 18px;
}
</style>
"""
# Markdown treats 4+ space indented lines as code blocks, which breaks
# <style> injection. Strip leading whitespace from every line so the CSS
# renders as actual styling instead of literal text.
_css_flat = "\n".join(line.lstrip() for line in CSS.split("\n"))
st.markdown(_css_flat, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


@st.cache_resource(show_spinner=False)
def get_base_prompt():
    return hub.pull("langchain-ai/retrieval-qa-chat")


def build_chain(model_name: str, temperature: float, k: int):
    api_key = os.environ.get("GROQ_API_KEY")
    llm = ChatGroq(model=model_name, temperature=temperature, max_tokens=768, api_key=api_key)
    vectorstore = get_vectorstore()
    prompt = get_base_prompt()
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return create_retrieval_chain(retriever, combine_docs_chain)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.now()
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "response_times" not in st.session_state:
    st.session_state.response_times = []

# ---------------------------------------------------------------------------
# Sidebar — the "chart"
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### \U0001F4CB Session Chart")

    st.markdown(
        f'<div class="stat-chip"><div class="label">Messages</div>'
        f'<div class="value">{len(st.session_state.messages)}</div></div>',
        unsafe_allow_html=True,
    )
    avg_time = (
        sum(st.session_state.response_times) / len(st.session_state.response_times)
        if st.session_state.response_times else 0
    )
    st.markdown(
        f'<div class="stat-chip"><div class="label">Avg. response</div>'
        f'<div class="value">{avg_time:.1f}s</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="stat-chip"><div class="label">Session started</div>'
        f'<div class="value" style="font-size:0.95rem;">'
        f'{st.session_state.session_start.strftime("%H:%M:%S")}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### \u2699\uFE0F Retrieval Settings")
    model_choice = st.selectbox("Model", GROQ_MODELS, index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.05,
                             help="Lower = more literal/consistent. Higher = more exploratory.")
    k_sources = st.slider("Sources retrieved (k)", 1, 8, 3,
                           help="How many document chunks are pulled in to ground each answer.")
    show_sources = st.checkbox("Show source citations", value=True)

    st.markdown("---")
    if st.button("\U0001F5D1\uFE0F Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.response_times = []
        st.rerun()

    if st.session_state.messages:
        transcript = "\n\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages
        )
        st.download_button(
            "\u2B07\uFE0F Download transcript",
            transcript,
            file_name="consult_transcript.txt",
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="consult-header">\U0001FA7A Consult</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="consult-sub">Answers grounded in your indexed reference documents — every claim is traceable to a source.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="pulse-line"></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer-banner">\u26A0\uFE0F This tool retrieves from a fixed document set '
    'and does not replace professional medical advice. Always verify against current clinical guidance.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Quick-start chips (only shown before first message)
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(3)
    sample_questions = [
        "What are the symptoms described for this condition?",
        "What treatment options are mentioned?",
        "Are there any noted risk factors?",
    ]
    for col, q in zip(cols, sample_questions):
        with col:
            if st.button(q, use_container_width=True):
                st.session_state.pending_prompt = q
                st.rerun()

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------
for message in st.session_state.messages:
    avatar = "\U0001F9D1" if message["role"] == "user" else "\U0001FA7A"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources") and show_sources:
            with st.expander(f"\U0001F4C4 {len(message['sources'])} source(s) cited"):
                for src in message["sources"]:
                    st.markdown(
                        f'<div class="source-chip"><div class="src-meta">{src["meta"]}</div>{src["preview"]}</div>',
                        unsafe_allow_html=True,
                    )

# ---------------------------------------------------------------------------
# Input — either typed or a clicked quick-start chip
# ---------------------------------------------------------------------------
typed_prompt = st.chat_input("Ask about your indexed documents...")
prompt = st.session_state.pending_prompt or typed_prompt
st.session_state.pending_prompt = None

if prompt:
    st.chat_message("user", avatar="\U0001F9D1").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="\U0001FA7A"):
        with st.spinner("Reviewing indexed literature..."):
            try:
                start = time.time()
                chain = build_chain(model_choice, temperature, k_sources)
                response = chain.invoke({"input": prompt})
                elapsed = time.time() - start
                st.session_state.response_times.append(elapsed)

                answer = response["answer"]
                st.markdown(answer)

                sources = []
                for doc in response.get("context", []):
                    page = doc.metadata.get("page", "?")
                    src_name = doc.metadata.get("source", "unknown").split("/")[-1].split("\\")[-1]
                    sources.append({
                        "meta": f"{src_name} · page {page}",
                        "preview": doc.page_content[:220].strip() + "...",
                    })

                if sources and show_sources:
                    with st.expander(f"\U0001F4C4 {len(sources)} source(s) cited"):
                        for src in sources:
                            st.markdown(
                                f'<div class="source-chip"><div class="src-meta">{src["meta"]}</div>{src["preview"]}</div>',
                                unsafe_allow_html=True,
                            )

                st.caption(f"Responded in {elapsed:.1f}s using {model_choice}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as e:
                st.error(f"Something went wrong retrieving an answer: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {e}",
                    "sources": [],
                })