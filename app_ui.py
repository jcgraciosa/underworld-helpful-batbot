import json
import os
import random
import requests
import streamlit as st
from PIL import Image

API_URL = os.environ.get("BOT_API_URL", "http://localhost:8001")

st.set_page_config(
    page_title="HelpfulBatBot — Underworld3 Assistant",
    page_icon="🦇",
    layout="centered",
)

# --- Sidebar ---
with st.sidebar:
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "uw3_logo.png")
    st.image(Image.open(logo_path), use_container_width=True)
    st.title("HelpfulBatBot 🦇")
    st.markdown(
        "A RAG-based assistant for [Underworld3](https://github.com/underworldcode/underworld3), "
        "an open-source finite element geodynamics framework."
    )
    st.divider()

    # Health check
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        data = r.json()
        if data.get("status") == "loading":
            st.warning(f"Backend is loading the index ({data.get('doc_count', 0)} chunks so far)…")
        else:
            st.success(f"Backend online — {data.get('doc_count', '?')} chunks indexed")
    except Exception:
        st.error("Backend offline or starting up. Please wait and refresh.")

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"API: {API_URL}")

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = "🦇" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("Sources"):
                for c in msg["citations"]:
                    st.markdown(f"- `{c}`")

_RETRIEVAL_MESSAGES = [
    "Rifting through the knowledge base...",
    "Subducting your query...",
    "Convecting through relevant chunks...",
    "Cross-encoding the vibes...",
    "Interrogating the source code...",
    "Triangulating finite element wisdom...",
    "Pressurising the PETSc solver...",
    "Plume-ing through the docs...",
    "Hypothetically speaking...",
    "Sieving through indexed chunks...",
    "Asking the mantle for answers...",
    "Reranking tectonic possibilities...",
]

# --- Input ---
if prompt := st.chat_input("Ask a question about Underworld3..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🦇"):
        citations = []
        status_placeholder = st.empty()
        status_placeholder.markdown(f"*{random.choice(_RETRIEVAL_MESSAGES)}*")

        def stream_response():
            first_text = True
            try:
                with requests.post(
                    f"{API_URL}/ask/stream",
                    json={"question": prompt, "max_context_items": 10},
                    stream=True,
                    timeout=600,
                ) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line:
                            continue
                        if line.startswith(b"data: "):
                            data_str = line[6:].decode("utf-8")
                            if data_str == "[DONE]":
                                break
                            try:
                                event = json.loads(data_str)
                                if event["type"] == "text":
                                    if first_text:
                                        status_placeholder.empty()
                                    first_text = False
                                    yield event["text"]
                                elif event["type"] == "citations":
                                    citations.extend(event.get("citations", []))
                                elif event["type"] == "error":
                                    status_placeholder.empty()
                                    yield event.get("text", "An error occurred.")
                            except json.JSONDecodeError:
                                pass
            except requests.exceptions.Timeout:
                status_placeholder.empty()
                yield "The request timed out. Please try again in a moment."
            except Exception as e:
                status_placeholder.empty()
                yield f"Error contacting the backend: {e}"

        answer = st.write_stream(stream_response())

        if citations:
            with st.expander("Sources"):
                for c in citations:
                    st.markdown(f"- `{c}`")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
    })
