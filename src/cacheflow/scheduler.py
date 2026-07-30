from __future__ import annotations

from .domain import (
    QueueFullError,
    RequestRecord,
    RequestState,
    SchedulingDecision,
    SlotRecord,
)
from .policy import AgingCachePolicy
from .prefix_index import PrefixIndex


class SchedulerCore:
    def __init__(
        self,
        slots: list[SlotRecord],
        *,
        max_queue: int = 128,
        policy: AgingCachePolicy | None = None,
    ) -> None:
        self.slots = {slot.slot_id: slot for slot in slots}
        self.max_queue = max_queue
        self.policy = policy or AgingCachePolicy()
        self.requests: dict[str, RequestRecord] = {}
        self.pending: list[str] = []
        self.indexes: dict[str, PrefixIndex] = {}
        for slot in slots:
            index = self.indexes.setdefault(slot.model, PrefixIndex())
            if slot.tokens:
                index.upsert(slot.slot_id, slot.tokens)

    def submit(self, request: RequestRecord, now: float) -> None:
        if len(self.pending) >= self.max_queue:
            raise QueueFullError(f"request queue limit reached: {self.max_queue}")
        if request.request_id in self.requests:
            raise ValueError(f"duplicate request id: {request.request_id}")
        if request.state != RequestState.TOKENIZING:
            raise ValueError("request must be tokenized before scheduler submission")
        request.transition(RequestState.QUEUED, now)
        self.requests[request.request_id] = request
        self.pending.append(request.request_id)
        self.indexes.setdefault(request.model, PrefixIndex())

    def expire(self, now: float) -> list[str]:
        expired: list[str] = []
        for request_id in list(self.pending):
            request = self.requests[request_id]
            if request.deadline_at <= now:
                request.transition(RequestState.TIMED_OUT, now)
                self.pending.remove(request_id)
                expired.append(request_id)
        return expired

    def cancel(self, request_id: str, now: float) -> bool:
        request = self.requests.get(request_id)
        if request is None or request.state != RequestState.QUEUED:
            return False
        request.transition(RequestState.CANCELLED, now)
        self.pending.remove(request_id)
        return True

    def plan(self, now: float) -> list[SchedulingDecision]:
        self.expire(now)
        decisions: list[SchedulingDecision] = []
        while self.pending:
            queued = [self.requests[request_id] for request_id in self.pending]
            decision = self.policy.choose(
                queued, list(self.slots.values()), self.indexes, now
            )
            if decision is None:
                break
            request = self.requests[decision.request_id]
            slot = self.slots[decision.slot_id]
            request.transition(RequestState.DISPATCHED, now)
            request.assigned_slot = slot.slot_id
            slot.running_request_id = request.request_id
            self.pending.remove(request.request_id)
            decisions.append(decision)
        return decisions

    def mark_restoring(self, request_id: str, now: float) -> None:
        self.requests[request_id].transition(RequestState.RESTORING, now)

    def mark_running(self, request_id: str, now: float) -> None:
        request = self.requests[request_id]
        request.transition(RequestState.RUNNING, now)

    def complete(
        self,
        request_id: str,
        now: float,
        *,
        cached_tokens: tuple[int, ...],
        result: dict,
    ) -> None:
        request = self.requests[request_id]
        request.transition(RequestState.COMPLETED, now)
        request.result = result
        slot = self.slots[request.assigned_slot]
        slot.conversation_id = request.conversation_id
        slot.tokens = cached_tokens
        slot.running_request_id = None
        slot.last_used = now
        self.indexes[slot.model].upsert(slot.slot_id, cached_tokens)

    def fail(self, request_id: str, now: float, error: str) -> None:
        request = self.requests[request_id]
        request.transition(RequestState.FAILED, now)
        request.error = error
        if request.assigned_slot is not None:
            self.slots[request.assigned_slot].running_request_id = None

    def snapshot(self) -> dict:
        return {
            "queue_depth": len(self.pending),
            "pending": list(self.pending),
            "slots": [
                {
                    "slot_id": slot.slot_id,
                    "model": slot.model,
                    "conversation_id": slot.conversation_id,
                    "cached_tokens": len(slot.tokens),
                    "running_request_id": slot.running_request_id,
                }
                for slot in self.slots.values()
            ],
        }
