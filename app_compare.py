import json
import os
import random
import time
import threading
import requests
import streamlit as st
from PIL import Image

API_URL = os.environ.get("BOT_API_URL", "http://localhost:8001")

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

st.set_page_config(
    page_title="HelpfulBatBot — Compare",
    page_icon="🦇",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { min-width: 220px; max-width: 220px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "uw3_logo.png")
    st.image(Image.open(logo_path), use_container_width=True)
    st.title("HelpfulBatBot 🦇")
    st.markdown("**Comparison mode** — Standard RAG vs Agent RAG")
    st.divider()

    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        data = r.json()
        if data.get("status") == "loading":
            st.warning(f"Backend loading ({data.get('doc_count', 0)} chunks so far)…")
        else:
            st.success(f"Backend online — {data.get('doc_count', '?')} chunks indexed")
    except Exception:
        st.error("Backend offline or starting up.")

    st.divider()
    st.markdown(
        "**Standard RAG** — single retrieval pass\n"
        "- Fast (~5–10 s)\n"
        "- Fixed k=10 chunks retrieved once\n\n"
        "**🦇 Agent RAG** — multi-step search\n"
        "- Slower (~20–30 s)\n"
        "- Claude decides how many times to search\n"
        "- Better on complex or multi-hop questions"
    )
    st.divider()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"API: {API_URL}")


def _collect_agent(question, result):
    """Background thread: collect agent SSE response into result dict."""
    start = time.time()
    try:
        with requests.post(
            f"{API_URL}/ask/agent/stream",
            json={"question": question, "max_context_items": 10},
            stream=True,
            timeout=600,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith(b"data: "):
                    raw = line[6:].decode("utf-8")
                    if raw == "[DONE]":
                        break
                    try:
                        ev = json.loads(raw)
                        if ev["type"] == "text":
                            result["tokens"].append(ev["text"])
                        elif ev["type"] == "citations":
                            result["citations"].extend(ev.get("citations", []))
                        elif ev["type"] == "error":
                            result["error"] = ev.get("text", "An error occurred.")
                    except json.JSONDecodeError:
                        pass
    except requests.exceptions.Timeout:
        result["error"] = "The request timed out."
    except Exception as e:
        result["error"] = f"Error contacting the backend: {e}"
    result["elapsed"] = time.time() - start
    result["done"] = True


def _render_stored(msg):
    """Render a stored side-by-side comparison from chat history."""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🦇 Standard RAG**")
        st.markdown(msg["rag_answer"])
        st.caption(f"⏱ {msg['rag_elapsed']:.1f}s")
        if msg.get("rag_citations"):
            with st.expander("Sources"):
                for c in msg["rag_citations"]:
                    st.markdown(f"- `{c}`")
    with col2:
        st.markdown("**🦇 Agent RAG**")
        st.markdown(msg["agent_answer"])
        st.caption(f"⏱ {msg['agent_elapsed']:.1f}s")
        if msg.get("agent_citations"):
            with st.expander("Sources"):
                for c in msg["agent_citations"]:
                    st.markdown(f"- `{c}`")


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🦇"):
            _render_stored(msg)

if prompt := st.chat_input("Ask a question about Underworld3..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🦇"):
        col1, col2 = st.columns(2)

        rag_result = {"citations": [], "elapsed": 0.0}
        agent_result = {
            "tokens": [], "citations": [], "elapsed": 0.0,
            "done": False, "error": None,
        }

        # Start agent thread before RAG so both clocks start at the same moment
        agent_thread = threading.Thread(
            target=_collect_agent, args=(prompt, agent_result), daemon=True
        )
        agent_thread.start()
        agent_start = time.time()

        # Left column: stream standard RAG token by token
        with col1:
            st.markdown("**🦇 Standard RAG**")
            msg_cycle = _RETRIEVAL_MESSAGES.copy()
            random.shuffle(msg_cycle)
            msg_idx = [0]
            status_ph = st.empty()
            status_ph.markdown(f"*{msg_cycle[0]}*")

            def rag_stream():
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
                                raw = line[6:].decode("utf-8")
                                if raw == "[DONE]":
                                    break
                                try:
                                    ev = json.loads(raw)
                                    if ev["type"] == "text":
                                        if first_text:
                                            status_ph.empty()
                                        first_text = False
                                        yield ev["text"]
                                    elif ev["type"] == "status":
                                        msg_idx[0] = (msg_idx[0] + 1) % len(msg_cycle)
                                        status_ph.markdown(f"*{msg_cycle[msg_idx[0]]}*")
                                    elif ev["type"] == "citations":
                                        rag_result["citations"].extend(ev.get("citations", []))
                                    elif ev["type"] == "error":
                                        if first_text:
                                            status_ph.empty()
                                        first_text = False
                                        yield ev.get("text", "An error occurred.")
                                except json.JSONDecodeError:
                                    pass
                except requests.exceptions.Timeout:
                    status_ph.empty()
                    yield "The request timed out."
                except Exception as e:
                    status_ph.empty()
                    yield f"Error contacting the backend: {e}"

            rag_start = time.time()
            rag_answer = st.write_stream(rag_stream()) or ""
            rag_result["elapsed"] = time.time() - rag_start
            st.caption(f"⏱ {rag_result['elapsed']:.1f}s")
            if rag_result["citations"]:
                with st.expander("Sources"):
                    for c in rag_result["citations"]:
                        st.markdown(f"- `{c}`")

        # Right column: wait for agent (it has been running in parallel), then display
        with col2:
            st.markdown("**🦇 Agent RAG**")
            if not agent_result["done"]:
                with st.spinner("Agent still searching…"):
                    agent_thread.join(timeout=300)

            agent_elapsed = agent_result.get("elapsed") or (time.time() - agent_start)

            if agent_result.get("error"):
                st.error(agent_result["error"])
                agent_answer = agent_result["error"]
            else:
                agent_answer = "".join(agent_result["tokens"])
                st.markdown(agent_answer)

            st.caption(f"⏱ {agent_elapsed:.1f}s")
            if agent_result["citations"]:
                with st.expander("Sources"):
                    for c in agent_result["citations"]:
                        st.markdown(f"- `{c}`")

        st.session_state.messages.append({
            "role": "assistant",
            "rag_answer": rag_answer,
            "agent_answer": agent_answer,
            "rag_elapsed": rag_result["elapsed"],
            "agent_elapsed": agent_elapsed,
            "rag_citations": rag_result["citations"],
            "agent_citations": agent_result["citations"],
        })
