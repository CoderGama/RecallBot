"""Retrieve related memories using embeddings plus light entity filters."""

from __future__ import annotations

from typing import Any

from config import SIMILARITY_THRESHOLD, TOP_K_MEMORIES
from database.vector_store import VectorStore
from nlp.embeddings import Embedder


class SemanticRetriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def search(
        self,
        query: str,
        sqlite_lookup,
        top_k: int = TOP_K_MEMORIES,
        threshold: float = SIMILARITY_THRESHOLD,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        embedding = self.embedder.encode_one(query)
        hits = self.vector_store.query(embedding, top_k=max(top_k, 8))
        results: list[dict[str, Any]] = []
        for hit in hits:
            if hit["similarity"] < threshold:
                continue
            record = sqlite_lookup(hit["id"])
            if not record:
                continue
            if record.get("status") == "deleted":
                continue
            if active_only and record.get("status") != "active":
                continue
            merged = dict(record)
            merged["similarity"] = hit["similarity"]
            results.append(merged)
        results.sort(key=lambda item: item["similarity"], reverse=True)
        return results[:top_k]
