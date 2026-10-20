from __future__ import annotations

import streamlit as st

from chatbot.runtime import RecallBotRuntime


def render_contradictions(runtime: RecallBotRuntime) -> None:
    st.subheader("Contradiction review")
    open_events = runtime.sqlite.list_contradictions(status="open")
    resolved = runtime.sqlite.list_contradictions(status="resolved")
    st.caption(f"{len(open_events)} open · {len(resolved)} resolved")

    if not open_events:
        st.info("No open contradiction or clarification items. They appear when chat finds a possible conflict.")
    for event in open_events:
        st.markdown('<div class="card warn">', unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            st.markdown("**Earlier statement**")
            st.write(event.get("old_statement") or "—")
        with right:
            st.markdown("**New statement**")
            st.write(event.get("new_statement") or "—")
        st.markdown(
            f'<span class="badge purple">{event.get("predicted_relation")}</span>'
            f'<span class="badge">confidence {float(event.get("confidence") or 0):.2f}</span>'
            f'<span class="badge blue">similarity {float(event.get("similarity") or 0):.2f}</span>',
            unsafe_allow_html=True,
        )
        st.write(event.get("explanation") or "")
        with st.expander("NLI / retrieval details"):
            st.json(event.get("technical") or {})
        a, b, c = st.columns(3)
        with a:
            if st.button("Yes, my information changed.", key=f"ch_{event['id']}"):
                result = runtime.conversations.resolve_event(event["id"], "changed")
                st.success(result.get("message") or result.get("error"))
                st.rerun()
        with b:
            if st.button("No, these refer to different things.", key=f"df_{event['id']}"):
                result = runtime.conversations.resolve_event(event["id"], "different")
                st.success(result.get("message") or result.get("error"))
                st.rerun()
        with c:
            if st.button("Correct the stored memory.", key=f"co_{event['id']}"):
                result = runtime.conversations.resolve_event(event["id"], "correct_old")
                st.success(result.get("message") or result.get("error"))
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if resolved:
        st.markdown("### Resolved")
        for event in resolved[:20]:
            st.markdown(
                f"- `{event.get('resolution')}` · {event.get('predicted_relation')} · "
                f"{event.get('new_statement')}"
            )
