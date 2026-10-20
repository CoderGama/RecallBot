"""Run with: python -m streamlit run app.py"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from recallbot.engine import RecallBot
from recallbot.extractor import get_nlp, label_for
from recallbot.retrieval import SemanticIndex
from recallbot.storage import MemoryStore

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("RECALLBOT_DATA_DIR", str(ROOT / "data"))).expanduser().resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="RecallBot | Conversational Memory", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.block-container { max-width: 1100px; padding-top: 1.8rem; }
[data-testid="stChatMessage"] { border: 1px solid rgba(128,128,128,.16); border-radius: 15px; }
.hero-title { font-size: 2.4rem; font-weight: 750; letter-spacing: -.05rem; }
.sub { opacity:.78; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Opening SQLite memory database...")
def load_db(db_path: str):
    return MemoryStore(db_path)


@st.cache_resource(show_spinner="Loading semantic embedding model (initial download requires internet)...")
def load_semantic_index(chroma_path: str):
    return SemanticIndex(chroma_path)


try:
    store = load_db(str(DATA_DIR / "memories.sqlite3"))
    index = load_semantic_index(str(DATA_DIR / "chroma"))
except Exception as exc:
    st.error(f"Could not start RecallBot: {type(exc).__name__}: {exc}")
    st.info("Install the dependencies from requirements.txt and make sure the first model download can access Hugging Face.")
    st.stop()

with st.sidebar:
    st.header("🧠 RecallBot")
    st.caption("Local-first · NLP college project")
    profile = st.text_input("Memory profile", value="default", max_chars=40,
                            help="Different profile names keep each person's data separate.").strip() or "default"
    use_nli = st.toggle("Enable optional NLI model", value=False,
                        help="Downloads an additional Hugging Face model on first contradiction. OFF = fast rule-based clarification.")
    st.divider()
    st.subheader("Memory controls")
    active = store.active(profile)
    st.metric("Active facts", len(active))
    st.metric("Historical facts", len(store.all(profile)) - len(active))
    st.caption("SQLite stores facts and chat history; ChromaDB stores semantic vectors.")
    export_rows = [{"attribute": label_for(m["attribute"]), "value": m["value"],
                    "statement": m["statement"], "created_at": m["created_at"]}
                   for m in active]
    st.download_button("Download memories (CSV)",
                       data=pd.DataFrame(export_rows, columns=["attribute", "value", "statement", "created_at"]).to_csv(index=False),
                       file_name="recallbot_memories.csv", mime="text/csv", use_container_width=True)
    if st.button("Clear chat history", use_container_width=True):
        store.clear_chat(profile)
        st.rerun()
    with st.expander("Danger zone"):
        st.caption("Permanently delete all memories and conversation history for this profile.")
        confirm = st.checkbox("I understand the data will be deleted", key=f"confirm_{profile}")
        if st.button("Delete all profile data", disabled=not confirm, use_container_width=True):
            store.clear_all(profile)
            index.sync([], profile)
            st.rerun()
    st.divider()
    st.subheader("Suggested prompts")
    st.code("My name is Alex\nI live in Pune\nWhere do I live?\nI live in Mumbai\nWhat do you remember about me?", language="text")
    st.caption("Optional spaCy NER model: `python -m spacy download en_core_web_sm`")

bot = RecallBot(store, index, profile=profile, use_nli=use_nli)
if st.session_state.get("synced_profile") != profile:
    with st.spinner("Synchronizing saved facts with semantic index..."):
        index.sync(store.active(profile), profile)
    st.session_state["synced_profile"] = profile

st.markdown('<div class="hero-title">🧠 RecallBot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Contradiction-Aware Conversational Memory Chatbot · Built with Streamlit, spaCy, Sentence Transformers, ChromaDB & SQLite</div>', unsafe_allow_html=True)

chat_tab, memory_tab, explain_tab = st.tabs(["💬 Chat", "📚 Memory Explorer", "🔎 NLP Inspector"])

with chat_tab:
    st.info("Tell RecallBot facts about yourself. It remembers them locally, retrieves them later, and asks before replacing conflicting facts.", icon="💡")
    history = store.history(profile)
    if not history:
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown("Hi! I'm RecallBot. Try **My name is Alex**, **I live in Pune**, or **What do you remember about me?**")
    for entry in history:
        with st.chat_message(entry["role"], avatar="🧠" if entry["role"] == "assistant" else None):
            st.markdown(entry["content"])

    pending = store.get_pending(profile)
    if pending:
        st.warning("A contradiction needs your decision before I store more facts.")
        left, right = st.columns(2)
        with left:
            if st.button("✅ Update with new fact", use_container_width=True, type="primary"):
                bot.resolve("update")
                st.rerun()
        with right:
            if st.button("↩️ Keep earlier fact", use_container_width=True):
                bot.resolve("keep")
                st.rerun()

    prompt = st.chat_input("Tell me something about yourself, or ask what I remember...")
    if prompt:
        bot.reply(prompt)
        st.rerun()

with memory_tab:
    memories = store.active(profile)
    if not memories:
        st.caption("No saved memories. Add a fact in Chat to see it here.")
    else:
        for m in memories:
            with st.container(border=True):
                top, remove = st.columns([5, 1])
                with top:
                    st.markdown(f"**{label_for(m['attribute'])}:** {m['value']}")
                    st.caption(f"Evidence: {m['statement']} · Saved: {m['created_at']}")
                with remove:
                    if st.button("Delete", key=f"del_{m['id']}"):
                        store.remove(profile, m["id"])
                        index.delete(m["id"])
                        st.rerun()
    with st.expander("Superseded / historical facts"):
        inactive = [m for m in store.all(profile) if not m["active"]]
        if inactive:
            st.dataframe(pd.DataFrame(inactive)[["attribute", "value", "statement", "superseded_at"]], hide_index=True)
        else:
            st.caption("No previous fact versions yet.")

with explain_tab:
    st.markdown("**Explore the NLP pipeline** — tokenization, recognized entities, fact extraction and semantic retrieval.")
    sample = st.text_area("Enter text to inspect", "My name is Alex and I live in Pune.", height=85)
    if st.button("Analyze text"):
        from recallbot.extractor import extract_facts
        doc = get_nlp()(sample)
        st.write("**spaCy tokens:**", [t.text for t in doc])
        st.write("**spaCy named entities:**", [(e.text, e.label_) for e in doc.ents])
        st.write("**Structured facts:**", [f.__dict__ for f in extract_facts(sample)])
        st.write("**Related memories (Sentence Transformers + ChromaDB):**", index.query(profile, sample, limit=5))
        if use_nli:
            st.caption("NLI is invoked only when two facts for the same attribute conflict.")

st.caption("Privacy: memories stay in the local data/ directory. First-run models are downloaded from Hugging Face; the app does not send chat messages to an LLM API.")
