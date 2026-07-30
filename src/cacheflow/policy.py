from __future__ import annotations

from .domain import RequestRecord, SchedulingDecision, SlotRecord
from .prefix_index import PrefixIndex


class AgingCachePolicy:
    def __init__(
        self,
        *,
        eviction_penalty: float = 0.5,
        wait_age_weight: float = 0.02,
        max_wait_ms: float = 500.0,
    ) -> None:
        if eviction_penalty < 0 or wait_age_weight < 0 or max_wait_ms <= 0:
            raise ValueError("invalid scheduling policy parameters")
        self.eviction_penalty = eviction_penalty
        self.wait_age_weight = wait_age_weight
        self.max_wait_ms = max_wait_ms

    def choose(
        self,
        requests: list[RequestRecord],
        slots: list[SlotRecord],
        indexes: dict[str, PrefixIndex],
        now: float,
    ) -> SchedulingDecision | None:
        free_slots = [slot for slot in slots if slot.is_free]
        if not requests or not free_slots:
            return None
        urgent = [
            request
            for request in requests
            if (now - request.created_at) * 1000 >= self.max_wait_ms
            or request.deadline_at <= now
        ]
        candidate_requests = (
            [min(urgent, key=lambda request: (request.deadline_at, request.created_at))]
            if urgent
            else requests
        )

        best: SchedulingDecision | None = None
        for request in candidate_requests:
            matches = indexes[request.model].match_lengths(request.tokens)
            wait_ms = max(0.0, (now - request.created_at) * 1000)
            for slot in free_slots:
                if slot.model != request.model:
                    continue
                reused = matches.get(slot.slot_id, 0)
                evicted = max(len(slot.tokens) - reused, 0)
                score = (
                    reused
                    - self.eviction_penalty * evicted
                    + self.wait_age_weight * wait_ms
                )
                decision = SchedulingDecision(
                    request.request_id,
                    slot.slot_id,
                    reused,
                    evicted,
                    score,
                    bool(urgent),
                )
                if best is None or self._key(decision, request, slot) > self._key(
                    best,
                    next(item for item in requests if item.request_id == best.request_id),
                    next(item for item in slots if item.slot_id == best.slot_id),
                ):
                    best = decision
        return best

    @staticmethod
    def _key(
        decision: SchedulingDecision, request: RequestRecord, slot: SlotRecord
    ) -> tuple[float, float, float, int]:
        return (decision.score, -request.created_at, -slot.last_used, -slot.slot_id)
