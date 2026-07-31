"""Archived prototype domain model; it is not imported by production runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RequestState(str, Enum):
    NEW = "new"
    TOKENIZING = "tokenizing"
    QUEUED = "queued"
    DISPATCHED = "dispatched"
    RESTORING = "restoring"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


TERMINAL_STATES = {
    RequestState.COMPLETED,
    RequestState.CANCELLED,
    RequestState.TIMED_OUT,
    RequestState.FAILED,
}


_ALLOWED_TRANSITIONS = {
    RequestState.NEW: {RequestState.TOKENIZING, RequestState.CANCELLED, RequestState.FAILED},
    RequestState.TOKENIZING: {RequestState.QUEUED, RequestState.CANCELLED, RequestState.FAILED},
    RequestState.QUEUED: {RequestState.DISPATCHED, RequestState.CANCELLED, RequestState.TIMED_OUT},
    RequestState.DISPATCHED: {
        RequestState.RESTORING,
        RequestState.RUNNING,
        RequestState.CANCELLED,
        RequestState.TIMED_OUT,
        RequestState.FAILED,
    },
    RequestState.RESTORING: {
        RequestState.RUNNING,
        RequestState.CANCELLED,
        RequestState.TIMED_OUT,
        RequestState.FAILED,
    },
    RequestState.RUNNING: {
        RequestState.COMPLETED,
        RequestState.CANCELLED,
        RequestState.TIMED_OUT,
        RequestState.FAILED,
    },
}


@dataclass
class RequestRecord:
    request_id: str
    conversation_id: str
    model: str
    tokens: tuple[int, ...]
    payload: dict[str, Any]
    created_at: float
    deadline_at: float
    state: RequestState = RequestState.NEW
    state_changed_at: float = 0.0
    assigned_slot: int | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    transitions: list[tuple[RequestState, float]] = field(default_factory=list)

    def transition(self, next_state: RequestState, now: float) -> None:
        if self.state in TERMINAL_STATES:
            raise ValueError(f"terminal request cannot transition: {self.state} -> {next_state}")
        allowed = _ALLOWED_TRANSITIONS.get(self.state, set())
        if next_state not in allowed:
            raise ValueError(f"invalid request transition: {self.state} -> {next_state}")
        self.state = next_state
        self.state_changed_at = now
        self.transitions.append((next_state, now))


@dataclass
class SlotRecord:
    slot_id: int
    model: str
    conversation_id: str | None = None
    tokens: tuple[int, ...] = ()
    running_request_id: str | None = None
    last_used: float = -1.0
    backend_slot_id: int | None = None

    @property
    def is_free(self) -> bool:
        return self.running_request_id is None

    @property
    def executor_slot_id(self) -> int:
        return self.slot_id if self.backend_slot_id is None else self.backend_slot_id


@dataclass(frozen=True)
class SchedulingDecision:
    request_id: str
    slot_id: int
    reused_tokens: int
    evicted_tokens: int
    score: float
    urgent: bool


class QueueFullError(RuntimeError):
    pass
