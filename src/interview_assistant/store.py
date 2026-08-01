from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

from .domain import CitationPayload, SessionSummary, StoredMessage


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ConversationStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
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

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def close(self) -> None:
        # Connections are operation-scoped; retained for a uniform server API.
        return None

    def create_session(self, title: str = "新的面试练习") -> SessionSummary:
        clean_title = " ".join(title.split())[:80] or "新的面试练习"
        session_id = uuid.uuid4().hex
        timestamp = _now()
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO sessions(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, clean_title, timestamp, timestamp),
            )
        return {"id": session_id, "title": clean_title, "created_at": timestamp, "updated_at": timestamp}

    def list_sessions(self) -> list[SessionSummary]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT sessions.*, COUNT(messages.id) AS message_count
                   FROM sessions LEFT JOIN messages ON messages.session_id = sessions.id
                   GROUP BY sessions.id ORDER BY sessions.updated_at DESC"""
            ).fetchall()
        return [cast(SessionSummary, dict(row)) for row in rows]

    def require_session(self, session_id: str) -> None:
        with self._connection() as connection:
            row = connection.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(session_id)

    def add_message(
        self,
        session_id: str,
        role: Literal["user", "assistant"],
        content: str,
        citations: list[CitationPayload] | None = None,
    ) -> StoredMessage:
        timestamp = _now()
        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    "INSERT INTO messages(session_id, role, content, citations_json, created_at) VALUES (?, ?, ?, ?, ?)",
                    (session_id, role, content, json.dumps(citations or [], ensure_ascii=False), timestamp),
                )
                updated = connection.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?",
                    (timestamp, session_id),
                )
                if cursor.rowcount != 1 or updated.rowcount != 1:
                    raise KeyError(session_id)
        except sqlite3.IntegrityError as exc:
            raise KeyError(session_id) from exc
        message_id = cursor.lastrowid
        if message_id is None:
            raise RuntimeError("SQLite did not return a message id")
        return {"id": message_id, "role": role, "content": content, "citations": citations or [], "created_at": timestamp}

    def messages(self, session_id: str, limit: int = 100) -> list[StoredMessage]:
        with self._connection() as connection:
            exists = connection.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if exists is None:
                raise KeyError(session_id)
            rows = connection.execute(
                """SELECT * FROM (SELECT id, role, content, citations_json, created_at
                   FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?)
                   ORDER BY id""",
                (session_id, limit),
            ).fetchall()
        messages: list[StoredMessage] = []
        for row in rows:
            message = dict(row)
            message["citations"] = json.loads(message.pop("citations_json"))
            messages.append(cast(StoredMessage, message))
        return messages
