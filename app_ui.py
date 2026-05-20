import os
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

# --- Input ---
if prompt := st.chat_input("Ask a question about Underworld3..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🦇"):
        with st.spinner("Retrieving and generating answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": prompt, "max_context_items": 10},
                    timeout=600,
                )
                response.raise_for_status()
                data = response.json()
                answer = data["answer"]
                citations = data.get("citations", [])
            except requests.exceptions.Timeout:
                answer = "The request timed out. The server may be starting up — please try again in a moment."
                citations = []
            except Exception as e:
                answer = f"Error contacting the backend: {e}"
                citations = []

        st.markdown(answer)
        if citations:
            with st.expander("Sources"):
                for c in citations:
                    st.markdown(f"- `{c}`")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
    })
