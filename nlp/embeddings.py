"""Sentence embeddings with MiniLM and a local hashed-token fallback."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

import numpy as np

from config import EMBEDDING_DIM, EMBEDDING_MODEL_NAME, TEST_MODE

_TOKEN = re.compile(r"[a-z0-9\+#\.]+")


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix / norms


class FallbackEmbedder:
    """Deterministic bag-of-hashed-tokens embedder used when MiniLM is unavailable.

    This is not a substitute for transformer semantics, but overlapping wording
    still produces higher cosine similarity than unrelated text.
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim
        self.backend = "fallback_hash"

    def encode(self, texts: str | Iterable[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        rows = [self._vector(text) for text in texts]
        return _normalize(np.vstack(rows))

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN.findall(text.lower())
        grams = tokens[:]
        compact = "".join(tokens)
        for i in range(len(compact) - 2):
            grams.append(compact[i : i + 3])
        for gram in grams:
            digest = hashlib.md5(gram.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        if not grams:
            vec[0] = 1.0
        return vec


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.backend = "sentence_transformers"
        self.model_name = model_name

    def encode(self, texts: str | Iterable[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        vectors = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)


class Embedder:
    """Production embedder with lazy MiniLM load and CPU-only inference."""

    def __init__(self, force_fallback: bool = False) -> None:
        self._impl = None
        self.force_fallback = force_fallback or TEST_MODE
        self.warning: str | None = None

    @property
    def backend(self) -> str:
        return self._get().backend

    def _get(self):
        if self._impl is not None:
            return self._impl
        if not self.force_fallback:
            try:
                self._impl = SentenceTransformerEmbedder()
                return self._impl
            except Exception as exc:
                self.warning = (
                    "Sentence Transformer model could not be loaded "
                    f"({exc}). Using hashed-token fallback embeddings."
                )
        self._impl = FallbackEmbedder()
        return self._impl

    def encode(self, texts: str | Iterable[str]) -> np.ndarray:
        return self._get().encode(texts)

    def encode_one(self, text: str) -> list[float]:
        return self.encode(text)[0].tolist()
