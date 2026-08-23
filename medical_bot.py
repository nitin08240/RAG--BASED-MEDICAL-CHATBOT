import html
import os
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DB_FAISS_PATH = Path("vectorstore/db_faiss")
GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
]

st.set_page_config(
    page_title="MediGuide RAG Assistant",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root {
    --bg: #f4f7f6;
    --panel: #ffffff;
    --panel-soft: #eef6f4;
    --ink: #172321;
    --muted: #5d706c;
    --teal: #176b5d;
    --teal-dark: #0f4c43;
    --amber: #b7772f;
    --red: #a84848;
    --border: #d9e6e2;
    --shadow: 0 14px 34px rgba(17, 48, 43, 0.08);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg);
    color: var(--ink);
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 1.4rem;
    max-width: 1180px;
}

[data-testid="stSidebar"] {
    background: #123f39;
    border-right: 1px solid rgba(255, 255, 255, 0.12);
}

[data-testid="stSidebar"] * {
    color: #edf8f5 !important;
}

[data-testid="stSidebar"] .stSelectbox div,
[data-testid="stSidebar"] .stSlider div,
[data-testid="stSidebar"] .stCheckbox div {
    color: #edf8f5 !important;
}

.hero {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 18px;
    align-items: center;
    padding: 24px 28px;
    background:
        linear-gradient(135deg, rgba(23, 107, 93, 0.10), rgba(183, 119, 47, 0.06)),
        var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    box-shadow: var(--shadow);
    margin-bottom: 20px;
}

.hero h1 {
    margin: 0;
    font-size: 2.15rem;
    line-height: 1.12;
    letter-spacing: 0;
}

.hero p {
    color: var(--muted);
    margin: 8px 0 0;
    max-width: 760px;
    font-size: 1rem;
}

.brand-kicker {
    color: var(--teal);
    font-size: 0.78rem;
    font-weight: 750;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.status-stack {
    display: grid;
    grid-template-columns: repeat(2, minmax(120px, 1fr));
    gap: 10px;
    min-width: 280px;
}

.metric {
    border: 1px solid var(--border);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.78);
    padding: 12px 14px;
}

.metric-label {
    color: var(--muted);
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.metric-value {
    margin-top: 4px;
    color: var(--ink);
    font-size: 1.18rem;
    font-weight: 700;
}

.notice {
    border: 1px solid #e3c790;
    border-left: 4px solid var(--amber);
    border-radius: 8px;
    background: #fff8ec;
    color: #684817;
    padding: 12px 14px;
    margin-bottom: 18px;
    font-size: 0.92rem;
}

.setup-error {
    border: 1px solid #e4bbbb;
    border-left: 4px solid var(--red);
    border-radius: 8px;
    background: #fff5f5;
    color: #6d2626;
    padding: 12px 14px;
    margin-bottom: 18px;
    font-size: 0.92rem;
}

.quick-title {
    margin: 0 0 10px;
    color: var(--muted);
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.source-chip {
    border: 1px solid var(--border);
    border-left: 4px solid var(--teal);
    border-radius: 8px;
    background: #f8fbfa;
    padding: 10px 12px;
    margin-bottom: 10px;
}

.source-meta {
    color: var(--teal-dark);
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 5px;
}

.source-preview {
    color: var(--muted);
    font-size: 0.9rem;
    line-height: 1.45;
}

.sidebar-title {
    font-size: 1.35rem;
    font-weight: 800;
    margin-bottom: 3px;
}

.sidebar-subtitle {
    color: rgba(237, 248, 245, 0.72) !important;
    font-size: 0.86rem;
    margin-bottom: 18px;
}

.sidebar-panel {
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.07);
    padding: 12px;
    margin-bottom: 12px;
}

.sidebar-panel .label {
    color: rgba(237, 248, 245, 0.70) !important;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}

.sidebar-panel .value {
    color: #ffffff !important;
    font-size: 1.25rem;
    font-weight: 800;
    margin-top: 2px;
}

.stButton > button {
    border-radius: 8px;
    border: 1px solid var(--border);
    background: #ffffff;
    color: var(--teal-dark);
    font-weight: 650;
    min-height: 2.6rem;
}

.stButton > button:hover {
    border-color: var(--teal);
    color: var(--teal);
}

[data-testid="stChatMessage"] {
    background: transparent;
    padding: 0.35rem 0;
}

[data-testid="stChatMessageContent"] {
    border: 1px solid var(--border);
    border-radius: 8px;
    background: #ffffff;
    padding: 0.9rem 1rem;
}

@media (max-width: 860px) {
    .hero {
        grid-template-columns: 1fr;
        padding: 20px;
    }

    .status-stack {
        grid-template-columns: 1fr;
        min-width: 0;
    }

    .hero h1 {
        font-size: 1.65rem;
    }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def init_state():
    defaults = {
        "messages": [],
        "session_start": datetime.now(),
        "pending_prompt": None,
        "response_times": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        str(DB_FAISS_PATH),
        embedding_model,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource(show_spinner=False)
def get_base_prompt():
    return hub.pull("langchain-ai/retrieval-qa-chat")


def build_chain(model_name: str, temperature: float, k: int):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")

    if not DB_FAISS_PATH.exists():
        raise RuntimeError(
            "FAISS vectorstore is missing. Run: python create_memory_for_llm.py"
        )

    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=768,
        api_key=api_key,
    )
    vectorstore = get_vectorstore()
    prompt = get_base_prompt()
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return create_retrieval_chain(retriever, combine_docs_chain)


def source_cards(docs):
    cards = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        source_name = Path(doc.metadata.get("source", "unknown")).name
        preview = doc.page_content[:260].strip().replace("\n", " ")
        cards.append(
            {
                "meta": f"{source_name} | page {page}",
                "preview": preview + ("..." if len(doc.page_content) > 260 else ""),
            }
        )
    return cards


def render_source_card(source):
    meta = html.escape(source["meta"])
    preview = html.escape(source["preview"])
    st.markdown(
        f"""
        <div class="source-chip">
            <div class="source-meta">{meta}</div>
            <div class="source-preview">{preview}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


init_state()

message_count = len(st.session_state.messages)
avg_time = (
    sum(st.session_state.response_times) / len(st.session_state.response_times)
    if st.session_state.response_times
    else 0
)
vector_status = "Ready" if DB_FAISS_PATH.exists() else "Missing"
api_status = "Connected" if os.environ.get("GROQ_API_KEY") else "Missing"

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-title">MediGuide</div>
        <div class="sidebar-subtitle">Document-grounded medical assistant</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="sidebar-panel">
            <div class="label">Messages</div>
            <div class="value">{message_count}</div>
        </div>
        <div class="sidebar-panel">
            <div class="label">Avg response</div>
            <div class="value">{avg_time:.1f}s</div>
        </div>
        <div class="sidebar-panel">
            <div class="label">Session started</div>
            <div class="value">{st.session_state.session_start.strftime("%H:%M")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Retrieval")
    model_choice = st.selectbox("Model", GROQ_MODELS, index=0)
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help="Lower values keep responses more literal and consistent.",
    )
    k_sources = st.slider(
        "Sources",
        min_value=1,
        max_value=8,
        value=4,
        help="Number of document chunks retrieved for each answer.",
    )
    show_sources = st.toggle("Show citations", value=True)

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.response_times = []
        st.rerun()

    if st.session_state.messages:
        transcript = "\n\n".join(
            f"{message['role'].upper()}: {message['content']}"
            for message in st.session_state.messages
        )
        st.download_button(
            "Download transcript",
            transcript,
            file_name="medical_chat_transcript.txt",
            use_container_width=True,
        )

st.markdown(
    f"""
    <section class="hero">
        <div>
            <div class="brand-kicker">RAG Medical Chatbot</div>
            <h1>MediGuide Clinical Reference Assistant</h1>
            <p>Ask questions against your indexed medical PDFs and review source-backed responses from the local FAISS knowledge base.</p>
        </div>
        <div class="status-stack">
            <div class="metric">
                <div class="metric-label">Vector DB</div>
                <div class="metric-value">{vector_status}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Groq API</div>
                <div class="metric-value">{api_status}</div>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="notice">
        This assistant retrieves from a fixed document set and is not a substitute
        for professional medical advice, diagnosis, or treatment.
    </div>
    """,
    unsafe_allow_html=True,
)

if not os.environ.get("GROQ_API_KEY"):
    st.markdown(
        """
        <div class="setup-error">
            GROQ_API_KEY is not configured. Add GROQ_API_KEY=your_key_here to a .env file before asking questions.
        </div>
        """,
        unsafe_allow_html=True,
    )

if not DB_FAISS_PATH.exists():
    st.markdown(
        """
        <div class="setup-error">
            The FAISS vector database was not found. Run python create_memory_for_llm.py after placing PDFs in the data folder.
        </div>
        """,
        unsafe_allow_html=True,
    )

if not st.session_state.messages:
    st.markdown('<p class="quick-title">Quick prompts</p>', unsafe_allow_html=True)
    quick_prompts = [
        "Summarize the key symptoms for diabetes.",
        "What treatments are mentioned for hypertension?",
        "List common risk factors for asthma.",
    ]
    cols = st.columns(3)
    for col, question in zip(cols, quick_prompts):
        with col:
            if st.button(question, use_container_width=True):
                st.session_state.pending_prompt = question
                st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources") and show_sources:
            with st.expander(f"{len(message['sources'])} source(s) used"):
                for source in message["sources"]:
                    render_source_card(source)

typed_prompt = st.chat_input("Ask about your indexed medical documents...")
prompt = st.session_state.pending_prompt or typed_prompt
st.session_state.pending_prompt = None

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching references and preparing answer..."):
            try:
                start = time.time()
                chain = build_chain(model_choice, temperature, k_sources)
                response = chain.invoke({"input": prompt})
                elapsed = time.time() - start
                st.session_state.response_times.append(elapsed)

                answer = response["answer"]
                sources = source_cards(response.get("context", []))

                st.markdown(answer)
                if sources and show_sources:
                    with st.expander(f"{len(sources)} source(s) used"):
                        for source in sources:
                            render_source_card(source)
                st.caption(f"Responded in {elapsed:.1f}s using {model_choice}")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )
            except Exception as exc:
                error_text = f"Error: {exc}"
                st.error(error_text)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_text,
                        "sources": [],
                    }
                )
