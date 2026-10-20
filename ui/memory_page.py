from __future__ import annotations

import pandas as pd
import streamlit as st

from chatbot.runtime import RecallBotRuntime


def render_memory(runtime: RecallBotRuntime) -> None:
    st.subheader("Memory dashboard")
    search = st.text_input("Search memories", placeholder="language, exam, project...")
    status = st.selectbox(
        "Status filter",
        ["all except deleted", "active", "superseded", "review", "deleted"],
    )
    if status == "all except deleted":
        rows = runtime.sqlite.list_memories(search=search or None)
    else:
        rows = runtime.sqlite.list_memories(status=status, include_deleted=True, search=search or None)

    if not rows:
        st.info("No memories stored yet. Chat first, or load demo data.")
        return

    frame = pd.DataFrame(
        [
            {
                "id": r["id"],
                "fact": r["fact_text"],
                "subject": r.get("subject"),
                "attribute": r.get("attribute"),
                "value": r.get("value"),
                "status": r.get("status"),
                "created_at": r.get("created_at"),
                "demo": bool(r.get("is_demo")),
                "confidence": r.get("confidence"),
            }
            for r in rows
        ]
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)

    options = {f"{r['status']} · {r['fact_text'][:80]}": r["id"] for r in rows}
    selected_label = st.selectbox("Inspect a memory", list(options))
    memory = runtime.sqlite.get_memory(options[selected_label])
    if memory:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(memory["fact_text"])
        st.caption(
            f"{memory['subject']} / {memory['attribute']} = {memory['value']} · "
            f"{memory['status']} · {memory['created_at']}"
        )
        st.json(
            {
                "entities": memory.get("entities"),
                "metadata": memory.get("metadata"),
                "superseded_by": memory.get("superseded_by"),
                "superseded_reason": memory.get("superseded_reason"),
            }
        )
        st.markdown("</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            new_text = st.text_input("Correct wording", value=memory["fact_text"])
            if st.button("Save correction"):
                runtime.memory.correct_memory(memory["id"], new_text)
                st.success("Memory corrected. Previous wording remains superseded.")
                st.rerun()
        with c2:
            if st.button("Deactivate (soft delete)"):
                runtime.memory.deactivate(memory["id"])
                st.success("Memory deactivated and removed from the vector index.")
                st.rerun()
        with c3:
            confirm = st.checkbox("Confirm permanent delete")
            if st.button("Delete from SQLite and vectors", disabled=not confirm):
                runtime.memory.delete(memory["id"])
                st.success("Deleted from both stores.")
                st.rerun()

    st.divider()
    if st.button("Rebuild / validate vector index"):
        with st.spinner("Rebuilding embeddings from active SQLite memories..."):
            report = runtime.memory.rebuild_index()
        st.json(report)
        if report.get("consistent"):
            st.success("SQLite active memories and the vector index are consistent.")
        else:
            st.warning("Index rebuilt. Review missing/extra IDs above.")
