"""SQLite persistence for conversations, memories, and review events."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from config import SQLITE_PATH
from utils import dumps, loads, new_id, utc_now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL,
    is_demo INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    fact_text TEXT NOT NULL,
    subject TEXT,
    attribute TEXT,
    value TEXT,
    source_message_id TEXT,
    conversation_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    confidence REAL DEFAULT 0.7,
    status TEXT NOT NULL,
    is_demo INTEGER DEFAULT 0,
    polarity TEXT,
    entities_json TEXT,
    metadata_json TEXT,
    superseded_by TEXT,
    superseded_reason TEXT
);

CREATE TABLE IF NOT EXISTS contradiction_events (
    id TEXT PRIMARY KEY,
    new_statement TEXT NOT NULL,
    old_memory_id TEXT,
    old_statement TEXT,
    predicted_relation TEXT,
    confidence REAL,
    similarity REAL,
    explanation TEXT,
    status TEXT NOT NULL,
    resolution TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    is_demo INTEGER DEFAULT 0,
    conversation_id TEXT,
    nli_available INTEGER DEFAULT 0,
    technical_json TEXT
);

CREATE TABLE IF NOT EXISTS retrieval_logs (
    id TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    retrieved_memory_ids TEXT,
    scores_json TEXT,
    created_at TEXT NOT NULL,
    conversation_id TEXT
);

CREATE TABLE IF NOT EXISTS extracted_entities (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    label TEXT NOT NULL,
    source_message_id TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_events_status ON contradiction_events(status);
"""


class SQLiteManager:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else SQLITE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def session(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.session() as conn:
            conn.executescript(SCHEMA)

    def create_conversation(self, title: str = "Chat", is_demo: bool = False) -> str:
        conv_id = new_id("conv")
        with self.session() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, is_demo) VALUES (?, ?, ?, ?)",
                (conv_id, title, utc_now_iso(), int(is_demo)),
            )
        return conv_id

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        msg_id = new_id("msg")
        with self.session() as conn:
            conn.execute(
                """INSERT INTO messages
                   (id, conversation_id, role, content, created_at, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (msg_id, conversation_id, role, content, utc_now_iso(), dumps(metadata or {})),
            )
        return msg_id

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self.session() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC, id ASC",
                (conversation_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_conversations(self) -> list[dict[str, Any]]:
        with self.session() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def insert_memory(self, record: dict[str, Any]) -> str:
        memory_id = record.get("id") or new_id("mem")
        now = utc_now_iso()
        payload = {
            "id": memory_id,
            "fact_text": record["fact_text"],
            "subject": record.get("subject"),
            "attribute": record.get("attribute"),
            "value": record.get("value"),
            "source_message_id": record.get("source_message_id"),
            "conversation_id": record.get("conversation_id"),
            "created_at": record.get("created_at") or now,
            "updated_at": now,
            "confidence": record.get("confidence", 0.7),
            "status": record.get("status", "active"),
            "is_demo": int(record.get("is_demo", 0)),
            "polarity": record.get("polarity"),
            "entities_json": dumps(record.get("entities") or []),
            "metadata_json": dumps(record.get("metadata") or {}),
            "superseded_by": record.get("superseded_by"),
            "superseded_reason": record.get("superseded_reason"),
        }
        with self.session() as conn:
            conn.execute(
                """INSERT INTO memories (
                    id, fact_text, subject, attribute, value, source_message_id,
                    conversation_id, created_at, updated_at, confidence, status,
                    is_demo, polarity, entities_json, metadata_json,
                    superseded_by, superseded_reason
                ) VALUES (
                    :id, :fact_text, :subject, :attribute, :value, :source_message_id,
                    :conversation_id, :created_at, :updated_at, :confidence, :status,
                    :is_demo, :polarity, :entities_json, :metadata_json,
                    :superseded_by, :superseded_reason
                )""",
                payload,
            )
        return memory_id

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        with self.session() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            return None
        return self._hydrate_memory(dict(row))

    def list_memories(
        self,
        status: str | None = None,
        include_deleted: bool = False,
        search: str | None = None,
        is_demo: bool | None = None,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        elif not include_deleted:
            clauses.append("status != 'deleted'")
        if search:
            clauses.append("(fact_text LIKE ? OR subject LIKE ? OR attribute LIKE ? OR value LIKE ?)")
            like = f"%{search}%"
            params.extend([like, like, like, like])
        if is_demo is not None:
            clauses.append("is_demo = ?")
            params.append(int(is_demo))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.session() as conn:
            rows = conn.execute(
                f"SELECT * FROM memories {where} ORDER BY created_at DESC",
                params,
            ).fetchall()
        return [self._hydrate_memory(dict(r)) for r in rows]

    def active_memories(self) -> list[dict[str, Any]]:
        return self.list_memories(status="active")

    def update_memory(self, memory_id: str, **fields: Any) -> None:
        if not fields:
            return
        allowed = {
            "fact_text",
            "subject",
            "attribute",
            "value",
            "confidence",
            "status",
            "polarity",
            "superseded_by",
            "superseded_reason",
            "metadata",
            "entities",
        }
        assignments = ["updated_at = ?"]
        params: list[Any] = [utc_now_iso()]
        for key, value in fields.items():
            if key not in allowed:
                continue
            column = key
            if key == "metadata":
                column = "metadata_json"
                value = dumps(value)
            elif key == "entities":
                column = "entities_json"
                value = dumps(value)
            assignments.append(f"{column} = ?")
            params.append(value)
        params.append(memory_id)
        with self.session() as conn:
            conn.execute(
                f"UPDATE memories SET {', '.join(assignments)} WHERE id = ?",
                params,
            )

    def hard_delete_memory(self, memory_id: str) -> None:
        with self.session() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

    def insert_contradiction(self, record: dict[str, Any]) -> str:
        event_id = record.get("id") or new_id("evt")
        with self.session() as conn:
            conn.execute(
                """INSERT INTO contradiction_events (
                    id, new_statement, old_memory_id, old_statement,
                    predicted_relation, confidence, similarity, explanation,
                    status, resolution, created_at, resolved_at, is_demo,
                    conversation_id, nli_available, technical_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event_id,
                    record["new_statement"],
                    record.get("old_memory_id"),
                    record.get("old_statement"),
                    record.get("predicted_relation"),
                    record.get("confidence"),
                    record.get("similarity"),
                    record.get("explanation"),
                    record.get("status", "open"),
                    record.get("resolution"),
                    record.get("created_at") or utc_now_iso(),
                    record.get("resolved_at"),
                    int(record.get("is_demo", 0)),
                    record.get("conversation_id"),
                    int(record.get("nli_available", 0)),
                    dumps(record.get("technical") or {}),
                ),
            )
        return event_id

    def list_contradictions(self, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM contradiction_events"
        params: list[Any] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC"
        with self.session() as conn:
            rows = conn.execute(sql, params).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["technical"] = loads(item.pop("technical_json", None), {})
            results.append(item)
        return results

    def get_contradiction(self, event_id: str) -> dict[str, Any] | None:
        with self.session() as conn:
            row = conn.execute(
                "SELECT * FROM contradiction_events WHERE id = ?", (event_id,)
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["technical"] = loads(item.pop("technical_json", None), {})
        return item

    def resolve_contradiction(self, event_id: str, resolution: str) -> None:
        with self.session() as conn:
            conn.execute(
                """UPDATE contradiction_events
                   SET status = 'resolved', resolution = ?, resolved_at = ?
                   WHERE id = ?""",
                (resolution, utc_now_iso(), event_id),
            )

    def log_retrieval(
        self,
        query_text: str,
        memory_ids: list[str],
        scores: list[float],
        conversation_id: str | None = None,
    ) -> str:
        log_id = new_id("ret")
        with self.session() as conn:
            conn.execute(
                """INSERT INTO retrieval_logs
                   (id, query_text, retrieved_memory_ids, scores_json, created_at, conversation_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    log_id,
                    query_text,
                    dumps(memory_ids),
                    dumps(scores),
                    utc_now_iso(),
                    conversation_id,
                ),
            )
        return log_id

    def list_retrievals(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.session() as conn:
            rows = conn.execute(
                "SELECT * FROM retrieval_logs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["retrieved_memory_ids"] = loads(item.get("retrieved_memory_ids"), [])
            item["scores"] = loads(item.get("scores_json"), [])
            results.append(item)
        return results

    def insert_entities(self, entities: list[dict[str, str]], source_message_id: str | None) -> None:
        if not entities:
            return
        now = utc_now_iso()
        with self.session() as conn:
            for ent in entities:
                conn.execute(
                    """INSERT INTO extracted_entities (id, text, label, source_message_id, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (new_id("ent"), ent.get("text", ""), ent.get("label", ""), source_message_id, now),
                )

    def recent_entities(self, limit: int = 40) -> list[dict[str, Any]]:
        with self.session() as conn:
            rows = conn.execute(
                "SELECT * FROM extracted_entities ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def counts(self) -> dict[str, int]:
        with self.session() as conn:
            conversations = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            memories = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE status != 'deleted'"
            ).fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE status = 'active'"
            ).fetchone()[0]
            contradictions = conn.execute("SELECT COUNT(*) FROM contradiction_events").fetchone()[0]
            resolved = conn.execute(
                "SELECT COUNT(*) FROM contradiction_events WHERE status = 'resolved'"
            ).fetchone()[0]
            open_events = conn.execute(
                "SELECT COUNT(*) FROM contradiction_events WHERE status = 'open'"
            ).fetchone()[0]
        return {
            "conversations": conversations,
            "messages": messages,
            "memories": memories,
            "active_memories": active,
            "contradictions": contradictions,
            "resolved_contradictions": resolved,
            "open_contradictions": open_events,
        }

    def memory_timeline(self) -> list[dict[str, Any]]:
        with self.session() as conn:
            rows = conn.execute(
                """SELECT substr(created_at, 1, 10) AS day,
                          COUNT(*) AS memories,
                          SUM(CASE WHEN status = 'superseded' THEN 1 ELSE 0 END) AS superseded
                   FROM memories
                   WHERE status != 'deleted'
                   GROUP BY day
                   ORDER BY day"""
            ).fetchall()
        return [dict(r) for r in rows]

    def contradiction_timeline(self) -> list[dict[str, Any]]:
        with self.session() as conn:
            rows = conn.execute(
                """SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS events
                   FROM contradiction_events
                   GROUP BY day
                   ORDER BY day"""
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_demo_data(self) -> list[str]:
        """Remove demo memories/events/conversations. Returns deleted memory IDs."""
        deleted_ids: list[str] = []
        with self.session() as conn:
            rows = conn.execute("SELECT id FROM memories WHERE is_demo = 1").fetchall()
            deleted_ids = [r["id"] for r in rows]
            demo_convs = [
                r["id"]
                for r in conn.execute("SELECT id FROM conversations WHERE is_demo = 1").fetchall()
            ]
            conn.execute("DELETE FROM contradiction_events WHERE is_demo = 1")
            conn.execute("DELETE FROM memories WHERE is_demo = 1")
            if demo_convs:
                placeholders = ",".join("?" * len(demo_convs))
                conn.execute(
                    f"DELETE FROM messages WHERE conversation_id IN ({placeholders})",
                    demo_convs,
                )
                conn.execute(
                    f"DELETE FROM conversations WHERE id IN ({placeholders})",
                    demo_convs,
                )
        return deleted_ids

    @staticmethod
    def _hydrate_memory(row: dict[str, Any]) -> dict[str, Any]:
        row["entities"] = loads(row.pop("entities_json", None), [])
        row["metadata"] = loads(row.pop("metadata_json", None), {})
        return row
