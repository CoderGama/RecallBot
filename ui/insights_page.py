from __future__ import annotations

import pandas as pd
import streamlit as st

from chatbot.demo import load_demo, reset_demo
from chatbot.runtime import RecallBotRuntime


def render_insights(runtime: RecallBotRuntime) -> None:
    st.subheader("NLP insights")
    counts = runtime.sqlite.counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Conversations", counts["conversations"])
    c2.metric("Stored memories", counts["memories"])
    c3.metric("Potential contradictions", counts["contradictions"])
    c4.metric("Resolved contradictions", counts["resolved_contradictions"])

    timeline = runtime.sqlite.memory_timeline()
    events = runtime.sqlite.contradiction_timeline()
    if timeline:
        mem_df = pd.DataFrame(timeline).set_index("day")
        st.markdown("**Memories over time**")
        st.bar_chart(mem_df)
    if events:
        ev_df = pd.DataFrame(events).set_index("day")
        st.markdown("**Contradiction events over time**")
        st.line_chart(ev_df)
    if not timeline and not events:
        st.caption("Charts appear after the app stores real memories or demo data.")

    st.markdown("### Memory retrieval history")
    retrievals = runtime.sqlite.list_retrievals(20)
    if retrievals:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "query": r["query_text"],
                        "hits": ", ".join(r.get("retrieved_memory_ids") or []),
                        "scores": r.get("scores"),
                        "at": r["created_at"],
                    }
                    for r in retrievals
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No retrievals logged yet.")

    st.markdown("### Recent extracted entities")
    ents = runtime.sqlite.recent_entities()
    if ents:
        st.dataframe(pd.DataFrame(ents)[["text", "label", "created_at"]], use_container_width=True, hide_index=True)
    else:
        st.caption("Entities will appear after messages are processed.")

    st.divider()
    st.subheader("Demo mode")
    st.write(
        "Loads labeled sample turns (college project, preferences, corrections, negation, migration). "
        "Demo rows are tagged and can be reset without deleting your own chats."
    )
    d1, d2 = st.columns(2)
    with d1:
        if st.button("Load sample / demo data"):
            with st.spinner("Running demo conversation through the NLP pipeline..."):
                conv_id = load_demo(runtime)
            st.session_state.conversation_id = conv_id
            st.success("Demo conversation loaded and labeled as demo data.")
            st.rerun()
    with d2:
        confirm = st.checkbox("Confirm reset of demo dataset only")
        if st.button("Reset demo dataset", disabled=not confirm):
            report = reset_demo(runtime)
            st.success(f"Removed demo memories: {report.get('deleted_demo_memories')}")
            st.json(report)
            st.rerun()
