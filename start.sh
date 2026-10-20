#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
if ! python -c 'import streamlit, chromadb, sentence_transformers, spacy' >/dev/null 2>&1; then
    python -m pip install -r requirements.txt
fi
exec python -m streamlit run app.py
