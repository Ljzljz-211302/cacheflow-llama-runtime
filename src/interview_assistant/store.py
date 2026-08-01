from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ConversationStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._connections: set[sqlite3.Connection] = set()
        self._connections_lock = threading.Lock()
        connection = self._connection()
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                citations_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS messages_session_id_id ON messages(session_id, id);
            """
        )

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(
                self.path,
                timeout=10,
                isolation_level=None,
                check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=10000")
            self._local.connection = connection
            with self._connections_lock:
                self._connections.add(connection)
        return connection

    def close(self) -> None:
        with self._connections_lock:
            connections = list(self._connections)
            self._connections.clear()
        for connection in connections:
            connection.close()
        self._local = threading.local()

    def create_session(self, title: str = "新的面试练习") -> dict[str, str]:
        clean_title = " ".join(title.split())[:80] or "新的面试练习"
        session_id = uuid.uuid4().hex
        timestamp = _now()
        self._connection().execute(
            "INSERT INTO sessions(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, clean_title, timestamp, timestamp),
        )
        return {"id": session_id, "title": clean_title, "created_at": timestamp, "updated_at": timestamp}

    def list_sessions(self) -> list[dict[str, Any]]:
        rows = self._connection().execute(
            """SELECT sessions.*, COUNT(messages.id) AS message_count
               FROM sessions LEFT JOIN messages ON messages.session_id = sessions.id
               GROUP BY sessions.id ORDER BY sessions.updated_at DESC"""
        ).fetchall()
        return [dict(row) for row in rows]

    def require_session(self, session_id: str) -> None:
        row = self._connection().execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(session_id)

    def add_message(self, session_id: str, role: str, content: str, citations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        self.require_session(session_id)
        timestamp = _now()
        connection = self._connection()
        cursor = connection.execute(
            "INSERT INTO messages(session_id, role, content, citations_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, json.dumps(citations or [], ensure_ascii=False), timestamp),
        )
        connection.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (timestamp, session_id))
        return {"id": cursor.lastrowid, "role": role, "content": content, "citations": citations or [], "created_at": timestamp}

    def messages(self, session_id: str, limit: int = 100) -> list[dict[str, Any]]:
        self.require_session(session_id)
        rows = self._connection().execute(
            """SELECT * FROM (SELECT id, role, content, citations_json, created_at
               FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?)
               ORDER BY id""",
            (session_id, limit),
        ).fetchall()
        messages = []
        for row in rows:
            message = dict(row)
            message["citations"] = json.loads(message.pop("citations_json"))
            messages.append(message)
        return messages
