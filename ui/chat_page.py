from __future__ import annotations

import streamlit as st

from chatbot.runtime import RecallBotRuntime
from utils import loads


def render_chat(runtime: RecallBotRuntime) -> None:
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = runtime.sqlite.create_conversation("Main chat")

    left, right = st.columns([2.4, 1])
    with right:
        st.markdown('<div class="card ok">', unsafe_allow_html=True)
        st.caption("Session")
        st.write("Open **NLP insights** to load or reset labeled demo data without deleting your own memories.")
        if st.button("New conversation", use_container_width=True):
            st.session_state.conversation_id = runtime.sqlite.create_conversation("Main chat")
            st.rerun()
        status = runtime.status
        st.write(f"Embeddings: `{status['embedding_backend']}`")
        st.write(f"NLI: `{status['nli_backend']}`")
        st.write(f"Vectors: `{status['vector_backend']}`")
        if status["nli_warning"]:
            st.info(status["nli_warning"])
        if status["embedding_warning"]:
            st.info(status["embedding_warning"])
        st.markdown("</div>", unsafe_allow_html=True)

    with left:
        messages = runtime.sqlite.list_messages(st.session_state.conversation_id)
        for msg in messages:
            avatar = "◉" if msg["role"] == "assistant" else "○"
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                meta = loads(msg.get("metadata_json"), {})
                if msg["role"] == "assistant" and meta:
                    retrieved_ids = meta.get("retrieved_ids") or []
                    detection = meta.get("detection") or {}
                    if retrieved_ids:
                        st.markdown(
                            f'<span class="badge blue">memory retrieved</span>',
                            unsafe_allow_html=True,
                        )
                    relation = detection.get("relation")
                    if relation in {"contradiction", "ambiguous", "change_over_time"}:
                        st.markdown('<div class="card warn">', unsafe_allow_html=True)
                        st.markdown(
                            f'<span class="badge purple">{relation}</span> '
                            f'<span class="badge">confidence {float(detection.get("confidence") or 0):.2f}</span>',
                            unsafe_allow_html=True,
                        )
                        st.write(detection.get("explanation") or "")
                        old = (detection.get("similar_memory") or {}).get("fact_text")
                        if old:
                            st.caption(f"Earlier statement: {old}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    with st.expander("Technical details"):
                        st.json(
                            {
                                "relation": relation,
                                "similarity": detection.get("similarity"),
                                "nli": detection.get("nli"),
                                "policy": detection.get("policy"),
                                "coref": meta.get("coref"),
                                "entities": meta.get("entities"),
                                "nli_backend": meta.get("nli_backend"),
                                "embedding_backend": meta.get("embedding_backend"),
                            }
                        )

        prompt = st.chat_input("Share a fact, ask what I remember, or correct an earlier statement")
        if prompt:
            with st.spinner("Retrieving memories and checking for contradictions..."):
                result = runtime.conversations.process(st.session_state.conversation_id, prompt)
            if not result.get("ok"):
                st.warning(result.get("assistant_text"))
            st.rerun()
