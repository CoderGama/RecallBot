"""SQLite is the source of truth; ChromaDB is a rebuildable search index."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .models import Fact


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryStore:
    def __init__(self, database_path: str = ":memory:"):
        if database_path != ":memory:":
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(database_path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA foreign_keys=ON")
        self.con.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            profile TEXT NOT NULL,
            subject TEXT NOT NULL,
            attribute TEXT NOT NULL,
            value TEXT NOT NULL,
            statement TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            superseded_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_memory_profile_attr ON memories(profile, attribute, active);
        CREATE TABLE IF NOT EXISTS pending (
            id TEXT PRIMARY KEY,
            profile TEXT NOT NULL UNIQUE,
            existing_id TEXT NOT NULL,
            candidate_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chat (
            id TEXT PRIMARY KEY,
            profile TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chat_profile ON chat(profile, created_at);
        """)
        self.con.commit()

    def add_memory(self, profile: str, fact: Fact) -> dict:
        item = dict(id=uuid.uuid4().hex, profile=profile, subject=fact.subject,
                    attribute=fact.attribute, value=fact.value, statement=fact.statement,
                    active=1, created_at=utc_now(), superseded_at=None)
        with self.con:
            self.con.execute("""INSERT INTO memories
               (id,profile,subject,attribute,value,statement,active,created_at,superseded_at)
               VALUES (:id,:profile,:subject,:attribute,:value,:statement,:active,:created_at,:superseded_at)""", item)
        return item

    def active(self, profile: str) -> list[dict]:
        cur = self.con.execute("SELECT * FROM memories WHERE profile=? AND active=1 ORDER BY created_at,id", (profile,))
        return [dict(row) for row in cur.fetchall()]

    def all(self, profile: str) -> list[dict]:
        cur = self.con.execute("SELECT * FROM memories WHERE profile=? ORDER BY created_at,id", (profile,))
        return [dict(row) for row in cur.fetchall()]

    def by_attribute(self, profile: str, attribute: str) -> list[dict]:
        cur = self.con.execute("SELECT * FROM memories WHERE profile=? AND attribute=? AND active=1 ORDER BY created_at DESC,id DESC", (profile, attribute))
        return [dict(row) for row in cur.fetchall()]

    def get(self, profile: str, memory_id: str) -> dict | None:
        row = self.con.execute("SELECT * FROM memories WHERE profile=? AND id=?", (profile, memory_id)).fetchone()
        return dict(row) if row else None

    def supersede(self, profile: str, memory_id: str) -> None:
        with self.con:
            self.con.execute("UPDATE memories SET active=0,superseded_at=? WHERE profile=? AND id=?", (utc_now(), profile, memory_id))

    def remove(self, profile: str, memory_id: str) -> None:
        with self.con:
            self.con.execute("DELETE FROM memories WHERE profile=? AND id=?", (profile, memory_id))
            # Clear pending pointing at deleted memory.
            self.con.execute("DELETE FROM pending WHERE profile=? AND existing_id=?", (profile, memory_id))

    def save_pending(self, profile: str, old_id: str, fact: Fact) -> None:
        with self.con:
            self.con.execute("""INSERT INTO pending(id,profile,existing_id,candidate_json,created_at)
               VALUES (?,?,?,?,?) ON CONFLICT(profile) DO UPDATE SET
               existing_id=excluded.existing_id,candidate_json=excluded.candidate_json,created_at=excluded.created_at""",
               (uuid.uuid4().hex, profile, old_id, json.dumps(fact.__dict__), utc_now()))

    def get_pending(self, profile: str) -> dict | None:
        row = self.con.execute("SELECT * FROM pending WHERE profile=?", (profile,)).fetchone()
        if not row:
            return None
        pending = dict(row)
        pending["fact"] = Fact(**json.loads(pending.pop("candidate_json")))
        return pending

    def clear_pending(self, profile: str) -> None:
        with self.con:
            self.con.execute("DELETE FROM pending WHERE profile=?", (profile,))

    def log_chat(self, profile: str, role: str, content: str) -> None:
        with self.con:
            self.con.execute("INSERT INTO chat VALUES (?,?,?,?,?)", (uuid.uuid4().hex, profile, role, content, utc_now()))

    def history(self, profile: str) -> list[dict]:
        rows = self.con.execute("SELECT role,content FROM chat WHERE profile=? ORDER BY rowid", (profile,)).fetchall()
        return [dict(row) for row in rows]

    def clear_chat(self, profile: str) -> None:
        with self.con:
            self.con.execute("DELETE FROM chat WHERE profile=?", (profile,))

    def clear_all(self, profile: str) -> None:
        with self.con:
            self.con.execute("DELETE FROM pending WHERE profile=?", (profile,))
            self.con.execute("DELETE FROM memories WHERE profile=?", (profile,))
            self.con.execute("DELETE FROM chat WHERE profile=?", (profile,))
