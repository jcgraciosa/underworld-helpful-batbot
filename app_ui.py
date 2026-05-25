import json
import os
import random
import time
import requests
import streamlit as st
from PIL import Image

API_URL = os.environ.get("BOT_API_URL", "http://localhost:8001")

st.set_page_config(
    page_title="HelpfulBatBot — Underworld3 Assistant",
    page_icon="🦇",
    layout="centered",
)

_FAQ_QUESTIONS = [
    "How do I install and set up Underworld3?",
    "How do I create a mesh in Underworld3?",
    "How do I set up and solve a Stokes flow problem?",
    "How do I add boundary conditions?",
    "What are SwarmVariables and how do I use them?",
    "How do I visualise results with pyvista or VTK?",
]

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

_DEEP_MESSAGES = [
    "Asking the manager...",
    "Scanning source files...",
    "Reading line by line...",
    "Cross-referencing the implementation...",
    "Checking the exact values...",
    "Consulting the codebase directly...",
    "Searching, then reading...",
    "Going deeper underground...",
]

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

    st.divider()
    st.markdown("**Suggested questions**")
    for faq_q in _FAQ_QUESTIONS:
        if st.button(faq_q, use_container_width=True, key=f"faq_{faq_q[:30]}"):
            st.session_state["faq_question"] = faq_q
            st.rerun()

    st.caption(f"API: {API_URL}")


def _render_followup_buttons(followups: list, key_prefix: str):
    if not followups:
        return
    st.markdown("**Follow-up questions:**")
    for i, q in enumerate(followups):
        label = q["text"] + (" *(Ask the manager 🦇)*" if q.get("deep") else "")
        if st.button(label, key=f"{key_prefix}_{i}", use_container_width=True):
            if q.get("deep"):
                st.session_state["deep_question"] = q["text"]
            else:
                st.session_state["faq_question"] = q["text"]
            st.rerun()


# --- Extract session state triggers before rendering history ---
_faq_trigger = None
_deep_trigger = None
if "faq_question" in st.session_state:
    _faq_trigger = st.session_state["faq_question"]
    del st.session_state["faq_question"]
if "deep_question" in st.session_state:
    _deep_trigger = st.session_state["deep_question"]
    del st.session_state["deep_question"]

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for i, msg in enumerate(st.session_state.messages):
    avatar = "🦇" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("Sources"):
                for c in msg["citations"]:
                    st.markdown(f"- `{c}`")
        if msg.get("followups"):
            _render_followup_buttons(msg["followups"], key_prefix=f"hist_{i}")

# --- Input & response ---
chat_input = st.chat_input("Ask a question about Underworld3...")
prompt = chat_input or _faq_trigger or _deep_trigger
use_agent_plus = (_deep_trigger is not None) and (chat_input is None) and (_faq_trigger is None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🦇"):
        citations = []
        followups = []
        endpoint = "/ask/agent-plus/stream" if use_agent_plus else "/ask/stream"
        msg_cycle = (_DEEP_MESSAGES if use_agent_plus else _RETRIEVAL_MESSAGES).copy()
        random.shuffle(msg_cycle)
        msg_idx = [0]
        status_placeholder = st.empty()
        status_placeholder.markdown(f"*{msg_cycle[0]}*")

        def stream_response():
            first_text = True
            try:
                with requests.post(
                    f"{API_URL}{endpoint}",
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
                                elif event["type"] == "status":
                                    msg_idx[0] = (msg_idx[0] + 1) % len(msg_cycle)
                                    status_placeholder.markdown(f"*{msg_cycle[msg_idx[0]]}*")
                                elif event["type"] == "citations":
                                    citations.extend(event.get("citations", []))
                                elif event["type"] == "followups":
                                    followups.extend(event.get("questions", []))
                                elif event["type"] == "error":
                                    if first_text:
                                        status_placeholder.empty()
                                    first_text = False
                                    yield event.get("text", "An error occurred.")
                            except json.JSONDecodeError:
                                pass
            except requests.exceptions.Timeout:
                status_placeholder.empty()
                yield "The request timed out. Please try again in a moment."
            except Exception as e:
                status_placeholder.empty()
                yield f"Error contacting the backend: {e}"

        t_start = time.time()
        answer = st.write_stream(stream_response())
        st.caption(f"⏱ {time.time() - t_start:.1f}s")

        if citations:
            with st.expander("Sources"):
                for c in citations:
                    st.markdown(f"- `{c}`")

        if followups:
            _render_followup_buttons(followups, key_prefix=f"new_{len(st.session_state.messages)}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "followups": followups,
    })
