"""Persistent semantic memory: ChromaDB with a NumPy cosine-index fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from config import CHROMA_PATH, NUMPY_INDEX_PATH, NUMPY_META_PATH, TEST_MODE


def _cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.array([])
    q = query / (np.linalg.norm(query) + 1e-12)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return (matrix / norms) @ q


class VectorStore:
    """Stores memory embeddings keyed by the same IDs used in SQLite."""

    def __init__(
        self,
        chroma_path: str | Path | None = None,
        numpy_path: str | Path | None = None,
        meta_path: str | Path | None = None,
    ) -> None:
        self.chroma_path = Path(chroma_path) if chroma_path else CHROMA_PATH
        self.numpy_path = Path(numpy_path) if numpy_path else NUMPY_INDEX_PATH
        self.meta_path = Path(meta_path) if meta_path else NUMPY_META_PATH
        self.backend = "numpy"
        self._collection = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._matrix = np.zeros((0, 1), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        self._try_chroma()
        if self.backend != "chroma":
            self._load_numpy()

    def _try_chroma(self) -> None:
        if TEST_MODE:
            self.backend = "numpy"
            self._collection = None
            return
        try:
            import chromadb

            self.chroma_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.chroma_path))
            self._collection = client.get_or_create_collection(
                name="recallbot_memories",
                metadata={"hnsw:space": "cosine"},
            )
            self.backend = "chroma"
        except Exception:
            self.backend = "numpy"
            self._collection = None

    def _load_numpy(self) -> None:
        if self.numpy_path.exists() and self.meta_path.exists():
            data = np.load(self.numpy_path, allow_pickle=False)
            self._matrix = data["vectors"].astype(np.float32)
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self._ids = list(meta.get("ids", []))
            self._texts = list(meta.get("texts", []))
        else:
            self._ids = []
            self._texts = []
            self._matrix = np.zeros((0, 1), dtype=np.float32)

    def _save_numpy(self) -> None:
        self.numpy_path.parent.mkdir(parents=True, exist_ok=True)
        if self._matrix.size == 0:
            empty = np.zeros((0, 1), dtype=np.float32)
            np.savez(self.numpy_path, vectors=empty)
        else:
            np.savez(self.numpy_path, vectors=self._matrix.astype(np.float32))
        self.meta_path.write_text(
            json.dumps({"ids": self._ids, "texts": self._texts}),
            encoding="utf-8",
        )

    def upsert(self, memory_id: str, text: str, embedding: list[float] | np.ndarray) -> None:
        vector = np.asarray(embedding, dtype=np.float32).tolist()
        if self.backend == "chroma" and self._collection is not None:
            try:
                self._collection.upsert(
                    ids=[memory_id],
                    documents=[text],
                    embeddings=[vector],
                    metadatas=[{"memory_id": memory_id}],
                )
                return
            except Exception:
                self.backend = "numpy"
                self._collection = None
                self._load_numpy()
        self.delete(memory_id)
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if self._matrix.size == 0 or self._matrix.shape[1] != vec.shape[1]:
            self._matrix = vec
            self._ids = [memory_id]
            self._texts = [text]
        else:
            self._matrix = np.vstack([self._matrix, vec])
            self._ids.append(memory_id)
            self._texts.append(text)
        self._save_numpy()

    def delete(self, memory_id: str) -> None:
        if self.backend == "chroma" and self._collection is not None:
            try:
                self._collection.delete(ids=[memory_id])
                return
            except Exception:
                pass
        if memory_id not in self._ids:
            return
        idx = self._ids.index(memory_id)
        self._ids.pop(idx)
        self._texts.pop(idx)
        self._matrix = np.delete(self._matrix, idx, axis=0)
        self._save_numpy()

    def query(
        self,
        embedding: list[float] | np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        vector = np.asarray(embedding, dtype=np.float32)
        if self.backend == "chroma" and self._collection is not None:
            try:
                count = int(self._collection.count())
                if count <= 0:
                    return []
                result = self._collection.query(
                    query_embeddings=[vector.tolist()],
                    n_results=min(max(top_k, 1), count),
                    include=["documents", "distances", "metadatas"],
                )
                hits = []
                ids = (result.get("ids") or [[]])[0]
                docs = (result.get("documents") or [[]])[0]
                dists = (result.get("distances") or [[]])[0]
                for mem_id, doc, dist in zip(ids, docs, dists):
                    similarity = 1.0 - float(dist)
                    hits.append(
                        {
                            "id": mem_id,
                            "text": doc,
                            "similarity": similarity,
                        }
                    )
                return hits
            except Exception:
                self.backend = "numpy"
                self._load_numpy()
        if not self._ids or self._matrix.size == 0:
            return []
        if self._matrix.shape[1] != vector.shape[0]:
            return []
        scores = _cosine_scores(vector, self._matrix)
        order = np.argsort(-scores)[:top_k]
        hits = []
        for i in order:
            hits.append(
                {
                    "id": self._ids[int(i)],
                    "text": self._texts[int(i)],
                    "similarity": float(scores[int(i)]),
                }
            )
        return hits

    def contains(self, memory_id: str) -> bool:
        if self.backend == "chroma" and self._collection is not None:
            try:
                got = self._collection.get(ids=[memory_id])
                return bool(got and got.get("ids"))
            except Exception:
                return False
        return memory_id in self._ids

    def count(self) -> int:
        if self.backend == "chroma" and self._collection is not None:
            try:
                return int(self._collection.count())
            except Exception:
                return 0
        return len(self._ids)

    def rebuild_from_records(
        self,
        records: list[dict[str, Any]],
        encode_fn,
    ) -> dict[str, Any]:
        """Rebuild the vector index from active SQLite memories."""
        errors: list[str] = []
        rebuilt = 0
        if self.backend == "chroma" and self._collection is not None:
            try:
                existing = self._collection.get()
                if existing and existing.get("ids"):
                    self._collection.delete(ids=existing["ids"])
            except Exception as exc:
                errors.append(f"Chroma clear failed: {exc}")
                self.backend = "numpy"
        self._ids = []
        self._texts = []
        self._matrix = np.zeros((0, 1), dtype=np.float32)
        for rec in records:
            try:
                embedding = encode_fn(rec["fact_text"])
                self.upsert(rec["id"], rec["fact_text"], embedding)
                rebuilt += 1
            except Exception as exc:
                errors.append(f"{rec.get('id')}: {exc}")
        if self.backend != "chroma":
            self._save_numpy()
        return {
            "backend": self.backend,
            "rebuilt": rebuilt,
            "errors": errors,
            "index_size": self.count(),
        }

    def ids_in_index(self) -> set[str]:
        if self.backend == "chroma" and self._collection is not None:
            try:
                got = self._collection.get()
                return set(got.get("ids") or [])
            except Exception:
                return set()
        return set(self._ids)
