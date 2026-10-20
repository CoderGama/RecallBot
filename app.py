"""RecallBot Streamlit entry point."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from chatbot.runtime import RecallBotRuntime
from ui.chat_page import render_chat
from ui.contradiction_page import render_contradictions
from ui.insights_page import render_insights
from ui.memory_page import render_memory
from ui.theme import hero, inject


@st.cache_resource
def get_runtime() -> RecallBotRuntime:
    return RecallBotRuntime()


def main() -> None:
    inject()
    runtime = get_runtime()
    hero()
    page = st.sidebar.radio(
        "Navigate",
        ["Chat", "Memory dashboard", "Contradiction review", "NLP insights"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "All data stays on this machine. Core chat does not call a paid LLM API."
    )
    if page == "Chat":
        render_chat(runtime)
    elif page == "Memory dashboard":
        render_memory(runtime)
    elif page == "Contradiction review":
        render_contradictions(runtime)
    else:
        render_insights(runtime)


if __name__ == "__main__":
    main()
