"""Sentence Transformer embeddings and persistent ChromaDB cosine search."""
from __future__ import annotations

from pathlib import Path


class SemanticIndex:
    def __init__(self, path: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        import chromadb
        from sentence_transformers import SentenceTransformer

        Path(path).mkdir(parents=True, exist_ok=True)
        self.encoder = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            name="recallbot_memory", metadata={"hnsw:space": "cosine"}
        )

    def _vector(self, text: str) -> list[float]:
        vector = self.encoder.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def upsert(self, memory: dict) -> None:
        self.collection.upsert(
            ids=[memory["id"]], embeddings=[self._vector(memory["statement"])],
            documents=[memory["statement"]],
            metadatas=[{"profile": memory["profile"], "attribute": memory["attribute"],
                        "value": memory["value"]}],
        )

    def delete(self, memory_id: str) -> None:
        self.collection.delete(ids=[memory_id])

    def query(self, profile: str, text: str, limit: int = 5) -> list[dict]:
        if not self.collection.count():
            return []
        result = self.collection.query(
            query_embeddings=[self._vector(text)], n_results=min(limit, self.collection.count()),
            where={"profile": profile}, include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        return [{"id": id_, "statement": result["documents"][0][i],
                 "distance": result["distances"][0][i],
                 "score": max(0.0, 1.0 - float(result["distances"][0][i])),
                 "attribute": result["metadatas"][0][i]["attribute"]}
                for i, id_ in enumerate(ids)]

    def sync(self, active_memories: list[dict], profile: str) -> None:
        """Repair missing/outdated entries and remove stale entries for a profile."""
        indexed = self.collection.get(where={"profile": profile}, include=[])
        existing_ids = set(indexed["ids"])
        active_ids = {m["id"] for m in active_memories}
        stale = list(existing_ids - active_ids)
        if stale:
            self.collection.delete(ids=stale)
        for item in active_memories:
            if item["id"] not in existing_ids:
                self.upsert(item)


class DummyIndex:
    """Tiny deterministic index for offline unit tests; not used in the app."""
    def __init__(self):
        self.entries = {}

    def upsert(self, memory):
        self.entries[memory["id"]] = dict(memory)

    def delete(self, memory_id):
        self.entries.pop(memory_id, None)

    def query(self, profile, text, limit=5):
        terms = set(text.casefold().split())
        result = []
        for memory in self.entries.values():
            if memory["profile"] != profile:
                continue
            words = set(memory["statement"].casefold().split())
            similarity = len(words & terms) / max(1, len(words | terms))
            result.append({"id": memory["id"], "statement": memory["statement"],
                           "attribute": memory["attribute"], "score": similarity})
        return sorted(result, key=lambda r: -r["score"])[:limit]

    def sync(self, active_memories, profile):
        for key in list(self.entries):
            if self.entries[key]["profile"] == profile:
                self.entries.pop(key)
        for item in active_memories:
            self.upsert(item)
