"""Streamlit interface for the local PDF RAG assistant."""

import hashlib
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from chatbot.config import load_settings
from chatbot.pipeline import ask_question, index_pdfs
from chatbot.vector_store import collection_count

CHAT_FILE = ROOT / "chat_history.json"
settings = load_settings()


def load_chats() -> dict:
    if not CHAT_FILE.exists():
        return {}
    try:
        data = json.loads(CHAT_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        st.warning(f"Could not load chat history: {error}")
        return {}
    return data if isinstance(data, dict) else {}


def save_chats(chats: dict) -> None:
    temporary = CHAT_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(chats, indent=2), encoding="utf-8")
    temporary.replace(CHAT_FILE)


def new_chat(chats: dict) -> str:
    chat_id = str(uuid.uuid4())
    chats[chat_id] = {
        "title": "New chat",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "messages": [],
    }
    return chat_id


st.set_page_config(page_title="PDF Knowledge Assistant", page_icon="📚", layout="wide")
if "chats" not in st.session_state:
    st.session_state.chats = load_chats()
if not st.session_state.chats:
    st.session_state.active_chat = new_chat(st.session_state.chats)
elif "active_chat" not in st.session_state or st.session_state.active_chat not in st.session_state.chats:
    st.session_state.active_chat = next(iter(st.session_state.chats))

with st.sidebar:
    st.title("📚 PDF Assistant")
    if st.button("New chat", use_container_width=True):
        st.session_state.active_chat = new_chat(st.session_state.chats)
        save_chats(st.session_state.chats)
        st.rerun()
    uploaded = st.file_uploader("Add PDF", type=["pdf"])
    if uploaded is not None:
        settings.pdf_dir.mkdir(parents=True, exist_ok=True)
        file_bytes = uploaded.getvalue()
        digest = hashlib.sha256(file_bytes).hexdigest()
        upload_key = f"{uploaded.name}:{digest}"
        if st.session_state.get("saved_upload_key") != upload_key:
            target = settings.pdf_dir / Path(uploaded.name).name
            target.write_bytes(file_bytes)
            st.session_state.saved_upload_key = upload_key
            st.success(f"Saved PDF to {target}")
    if st.button("Index PDFs", use_container_width=True):
        with st.spinner("Building a replacement index..."):
            try:
                count = index_pdfs(settings)
            except (OSError, ValueError, RuntimeError) as error:
                st.error(f"Indexing failed; the previous index was preserved: {error}")
            else:
                st.success(f"Indexed {count} passages.")
    st.caption(f"Indexed passages: {collection_count(settings)}")
    st.divider()
    for chat_id, chat in st.session_state.chats.items():
        if st.button(chat["title"], key=chat_id, use_container_width=True):
            st.session_state.active_chat = chat_id
            st.rerun()

active = st.session_state.chats[st.session_state.active_chat]
st.title("PDF Knowledge Assistant")
st.caption("Answers use only your local PDFs and Ollama models.")
for message in active["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for index, source in enumerate(message["sources"], start=1):
                    st.markdown(f"**[{index}] {source['source']} — page {source['page']}**")
                    st.caption(source["text"])

question = st.chat_input("Ask a question about your PDFs...")
if question:
    active["messages"].append({"role": "user", "content": question})
    if active["title"] == "New chat":
        active["title"] = question[:45]
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            result = ask_question(question, settings)
        except (OSError, RuntimeError, KeyError) as error:
            st.error(f"Could not answer the question: {error}")
            result = {"answer": "The local PDF assistant is unavailable.", "sources": []}
        st.markdown(result["answer"])
        active["messages"].append(
            {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
        )
    save_chats(st.session_state.chats)
