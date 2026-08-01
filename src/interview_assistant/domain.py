from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


Role = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    role: Role
    content: str


class CitationPayload(TypedDict):
    source: str
    title: str
    excerpt: str
    score: float


class StoredMessage(TypedDict):
    id: int
    role: Literal["user", "assistant"]
    content: str
    citations: list[CitationPayload]
    created_at: str


class SessionSummary(TypedDict):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: NotRequired[int]


class StreamEvent(TypedDict):
    type: Literal["citations", "delta", "done", "error"]
    content: NotRequired[str]
    citations: NotRequired[list[CitationPayload]]
    message: NotRequired[StoredMessage]
    error: NotRequired[str]
