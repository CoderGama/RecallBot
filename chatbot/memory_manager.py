"""Structured + semantic memory with supersede/review/delete policies."""

from __future__ import annotations

from typing import Any

from config import (
    DUPLICATE_SIMILARITY_THRESHOLD,
    MEMORY_STATUS_ACTIVE,
    MEMORY_STATUS_DELETED,
    MEMORY_STATUS_REVIEW,
    MEMORY_STATUS_SUPERSEDED,
)
from database.sqlite_manager import SQLiteManager
from database.vector_store import VectorStore
from nlp.embeddings import Embedder


class MemoryManager:
    def __init__(
        self,
        sqlite: SQLiteManager,
        vectors: VectorStore,
        embedder: Embedder,
    ) -> None:
        self.sqlite = sqlite
        self.vectors = vectors
        self.embedder = embedder

    def add_memory(self, record: dict[str, Any], skip_duplicate: bool = True) -> dict[str, Any]:
        if skip_duplicate:
            duplicate = self.find_duplicate_record(record)
            if duplicate:
                return {"id": duplicate["id"], "status": "duplicate", "memory": duplicate}
        memory_id = self.sqlite.insert_memory(record)
        try:
            embedding = self.embedder.encode_one(record["fact_text"])
            self.vectors.upsert(memory_id, record["fact_text"], embedding)
        except Exception as exc:
            return {
                "id": memory_id,
                "status": record.get("status", MEMORY_STATUS_ACTIVE),
                "sync_error": str(exc),
            }
        return {"id": memory_id, "status": record.get("status", MEMORY_STATUS_ACTIVE)}

    def find_duplicate_record(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Compare atomic fact identities, never entire-message embeddings alone."""
        attr = str(record.get("attribute") or "")
        subject = str(record.get("subject") or "")
        value = str(record.get("value") or "").strip().casefold()
        polarity = str(record.get("polarity") or "affirmative").lower()
        project = str((record.get("metadata") or {}).get("project_hint") or "").lower()
        if not attr or not value:
            return self.find_duplicate(record.get("fact_text", ""))
        for old in self.sqlite.active_memories():
            if (str(old.get("subject") or "").casefold() == subject.casefold()
                    and str(old.get("attribute") or "").casefold() == attr.casefold()
                    and str(old.get("value") or "").strip().casefold() == value
                    and str(old.get("polarity") or "affirmative").lower() == polarity
                    and str((old.get("metadata") or {}).get("project_hint") or "").lower() == project
                    and int(old.get("is_demo") or 0) == int(record.get("is_demo") or 0)):
                return old
        return None

    def find_duplicate(self, text: str) -> dict[str, Any] | None:
        if not text:
            return None
        embedding = self.embedder.encode_one(text)
        hits = self.vectors.query(embedding, top_k=3)
        for hit in hits:
            if hit["similarity"] >= DUPLICATE_SIMILARITY_THRESHOLD:
                mem = self.sqlite.get_memory(hit["id"])
                if mem and mem.get("status") == MEMORY_STATUS_ACTIVE:
                    return mem
        return None

    def supersede(
        self,
        old_id: str,
        new_record: dict[str, Any],
        reason: str,
    ) -> str:
        new_id = self.add_memory(new_record, skip_duplicate=False)["id"]
        self.sqlite.update_memory(
            old_id,
            status=MEMORY_STATUS_SUPERSEDED,
            superseded_by=new_id,
            superseded_reason=reason,
        )
        try:
            self.vectors.delete(old_id)
        except Exception:
            pass
        return new_id

    def mark_review(self, record: dict[str, Any]) -> str:
        record = dict(record)
        record["status"] = MEMORY_STATUS_REVIEW
        return self.add_memory(record, skip_duplicate=False)["id"]

    def activate(self, memory_id: str) -> None:
        mem = self.sqlite.get_memory(memory_id)
        if not mem:
            return
        self.sqlite.update_memory(memory_id, status=MEMORY_STATUS_ACTIVE)
        embedding = self.embedder.encode_one(mem["fact_text"])
        self.vectors.upsert(memory_id, mem["fact_text"], embedding)

    def deactivate(self, memory_id: str) -> None:
        self.sqlite.update_memory(memory_id, status=MEMORY_STATUS_DELETED)
        try:
            self.vectors.delete(memory_id)
        except Exception:
            pass

    def correct_memory(self, memory_id: str, new_text: str, extra: dict[str, Any] | None = None) -> str:
        old = self.sqlite.get_memory(memory_id)
        if not old:
            raise ValueError("Memory not found.")
        payload = {
            "fact_text": new_text,
            "subject": (extra or {}).get("subject", old.get("subject")),
            "attribute": (extra or {}).get("attribute", old.get("attribute")),
            "value": (extra or {}).get("value", old.get("value")),
            "source_message_id": old.get("source_message_id"),
            "conversation_id": old.get("conversation_id"),
            "confidence": 0.9,
            "status": MEMORY_STATUS_ACTIVE,
            "is_demo": old.get("is_demo", 0),
            "polarity": (extra or {}).get("polarity", old.get("polarity")),
            "entities": old.get("entities") or [],
            "metadata": {"corrected_from": memory_id},
        }
        return self.supersede(memory_id, payload, reason="user_correction")

    def delete(self, memory_id: str) -> None:
        self.sqlite.hard_delete_memory(memory_id)
        try:
            self.vectors.delete(memory_id)
        except Exception:
            pass

    def rebuild_index(self) -> dict[str, Any]:
        records = self.sqlite.active_memories()
        report = self.vectors.rebuild_from_records(records, self.embedder.encode_one)
        sqlite_ids = {r["id"] for r in records}
        index_ids = self.vectors.ids_in_index()
        report["sqlite_active"] = len(sqlite_ids)
        report["missing_in_index"] = sorted(sqlite_ids - index_ids)
        report["extra_in_index"] = sorted(index_ids - sqlite_ids)
        report["consistent"] = not report["missing_in_index"] and not report["extra_in_index"]
        return report
