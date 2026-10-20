from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["RECALLBOT_TEST_MODE"] = "1"

from chatbot.runtime import RecallBotRuntime  # noqa: E402


class ClusterEmbedder:
    """Tiny deterministic embedder so paraphrase tests work without MiniLM."""

    backend = "test_clusters"
    warning = None
    dim = 32

    KEYS = [
        "degree",
        "computer",
        "science",
        "studying",
        "study",
        "subject",
        "java",
        "python",
        "favorite",
        "language",
        "exam",
        "monday",
        "tuesday",
        "project",
        "alpha",
        "beta",
        "chatbot",
        "completed",
        "migrated",
        "college",
        "learning",
        "location",
        "pune",
        "mumbai",
    ]
    GROUPS = [
        ["degree", "computer", "science", "studying", "study", "subject", "cs"],
        ["java", "python", "language", "favorite", "programming"],
        ["exam", "monday", "tuesday"],
        ["project", "college", "chatbot", "app"],
        ["pune", "mumbai", "live", "location"],
    ]

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        rows = [self._vec(t) for t in texts]
        mat = np.vstack(rows)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        return mat / norms

    def encode_one(self, text: str):
        return self.encode(text)[0].tolist()

    def _vec(self, text: str) -> np.ndarray:
        lower = text.lower()
        vec = np.zeros(self.dim, dtype=np.float32)
        for i, key in enumerate(self.KEYS):
            if key in lower:
                vec[i] = 1.0
        for i, group in enumerate(self.GROUPS):
            if any(token in lower for token in group):
                vec[len(self.KEYS) + i] = 1.2
        if vec.sum() == 0:
            vec[0] = 0.1
        return vec


@pytest.fixture
def runtime(tmp_path):
    rt = RecallBotRuntime(
        sqlite_path=tmp_path / "test.sqlite3",
        chroma_path=tmp_path / "chroma",
        numpy_path=tmp_path / "idx.npz",
        meta_path=tmp_path / "idx.json",
        force_fallback=True,
    )
    rt.embedder = ClusterEmbedder()
    rt.memory.embedder = rt.embedder
    rt.retriever.embedder = rt.embedder
    return rt
