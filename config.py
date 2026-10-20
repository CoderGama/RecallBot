"""RecallBot configuration. Paths are derived from this file's location."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SQLITE_PATH = DATA_DIR / "recallbot.sqlite3"
CHROMA_PATH = DATA_DIR / "chroma"
NUMPY_INDEX_PATH = DATA_DIR / "vector_index.npz"
NUMPY_META_PATH = DATA_DIR / "vector_index_meta.json"

APP_NAME = "RecallBot"
APP_SUBTITLE = "An AI chatbot that remembers, connects, and clarifies."

EMBEDDING_MODEL_NAME = os.environ.get("RECALLBOT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
NLI_MODEL_NAME = os.environ.get(
    "RECALLBOT_NLI_MODEL",
    "typeform/distilbert-base-uncased-mnli",
)
SPACY_MODEL_NAME = os.environ.get("RECALLBOT_SPACY_MODEL", "en_core_web_sm")

# Retrieval and contradiction thresholds (configurable for experiments)
SIMILARITY_THRESHOLD = float(os.environ.get("RECALLBOT_SIM_THRESHOLD", "0.42"))
STRONG_SIMILARITY_THRESHOLD = float(os.environ.get("RECALLBOT_STRONG_SIM", "0.62"))
DUPLICATE_SIMILARITY_THRESHOLD = float(os.environ.get("RECALLBOT_DUP_SIM", "0.92"))
NLI_CONFIDENCE_THRESHOLD = float(os.environ.get("RECALLBOT_NLI_THRESHOLD", "0.55"))
TOP_K_MEMORIES = int(os.environ.get("RECALLBOT_TOP_K", "5"))
EMBEDDING_DIM = 384  # MiniLM default; fallback embedder matches this

MEMORY_STATUS_ACTIVE = "active"
MEMORY_STATUS_SUPERSEDED = "superseded"
MEMORY_STATUS_REVIEW = "review"
MEMORY_STATUS_DELETED = "deleted"

DEVICE = os.environ.get("RECALLBOT_DEVICE", "cpu")
TEST_MODE = os.environ.get("RECALLBOT_TEST_MODE", "").lower() in {"1", "true", "yes"}

DEMO_TAG = "[DEMO]"

DATA_DIR.mkdir(parents=True, exist_ok=True)
