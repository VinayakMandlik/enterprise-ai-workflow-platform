"""
Long-term agent memory — persists facts across sessions using SQLite.

Short-term memory (this conversation, right now) lives in the
LangGraph state and dies with the process. This is what survives
a restart: the agent can recall things from yesterday's session.
"""
import sqlite3
import json
import time
from pathlib import Path

from app.config import get_settings


class LongTermMemory:
    def __init__(self):
        settings = get_settings()
        db_path = Path(settings.memory_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(user_id, key)
            )
            """
        )
        self._conn.commit()

    def remember(self, user_id: str, key: str, value: dict):
        self._conn.execute(
            """
            INSERT INTO memories (user_id, key, value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (user_id, key, json.dumps(value), time.time()),
        )
        self._conn.commit()

    def recall(self, user_id: str, key: str) -> dict | None:
        row = self._conn.execute(
            "SELECT value FROM memories WHERE user_id = ? AND key = ?",
            (user_id, key),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def recall_all(self, user_id: str) -> dict:
        rows = self._conn.execute(
            "SELECT key, value FROM memories WHERE user_id = ?", (user_id,)
        ).fetchall()
        return {k: json.loads(v) for k, v in rows}

    def forget(self, user_id: str, key: str):
        """Deletes a specific memory entry."""
        self._conn.execute(
            "DELETE FROM memories WHERE user_id = ? AND key = ?",
            (user_id, key),
        )
        self._conn.commit()


_singleton: LongTermMemory | None = None


def get_long_term_memory() -> LongTermMemory:
    global _singleton
    if _singleton is None:
        _singleton = LongTermMemory()
    return _singleton