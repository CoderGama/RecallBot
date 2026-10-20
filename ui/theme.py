"""Shared dark-navy Streamlit styling."""

from __future__ import annotations

import streamlit as st

from config import APP_NAME, APP_SUBTITLE

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: "IBM Plex Sans", "Source Sans 3", sans-serif;
}
.stApp {
    background: radial-gradient(1200px 600px at 10% -10%, #1b2a4a 0%, #0b1220 45%, #070b14 100%);
    color: #e8eefc;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10182b 0%, #0a1020 100%);
    border-right: 1px solid #243154;
}
h1, h2, h3 { color: #f4f7ff !important; letter-spacing: 0.2px; }
.hero-kicker {
    color: #9bb7ff;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-size: 0.78rem;
    font-weight: 600;
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0.15rem 0 0.35rem 0;
}
.hero-sub { color: #b7c4e6; margin-bottom: 1.2rem; }
.card {
    background: rgba(16, 27, 48, 0.86);
    border: 1px solid #2c3f66;
    border-radius: 16px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.card.warn {
    border-color: #7c5cff;
    background: linear-gradient(180deg, rgba(80, 55, 150, 0.35), rgba(16, 27, 48, 0.9));
}
.card.ok {
    border-color: #3d7dff;
}
.badge {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.35rem;
    background: #243154;
    color: #d5e2ff;
}
.badge.purple { background: #4b3b8a; }
.badge.blue { background: #1f4b9c; }
.badge.amber { background: #8a5a12; }
.small { color: #9db0d6; font-size: 0.88rem; }
.stChatMessage { background: transparent; }
div[data-testid="stChatMessage"] {
    background: rgba(18, 28, 51, 0.7);
    border: 1px solid #2a3d63;
    border-radius: 14px;
    padding: 0.4rem 0.6rem;
}
</style>
"""


def inject() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="◉", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


def hero() -> None:
    st.markdown(
        f"""
        <div class="hero-kicker">Local NLP memory system</div>
        <div class="hero-title">{APP_NAME}</div>
        <div class="hero-sub">{APP_SUBTITLE}</div>
        """,
        unsafe_allow_html=True,
    )
