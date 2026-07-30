from __future__ import annotations

import random
from dataclasses import dataclass

from .metrics import percentile


@dataclass(frozen=True)
class TraceRequest:
    conversation_id: int
    prompt_tokens: int


@dataclass
class SimulatedSlot:
    conversation_id: int | None = None
    cached_tokens: int = 0
    last_used: int = -1


def generate_conversation_trace(
    *,
    requests: int,
    revisit_probability: float,
    active_conversations: int,
    system_tokens: int,
    turn_tokens_min: int,
    turn_tokens_max: int,
    seed: int,
) -> list[TraceRequest]:
    rng = random.Random(seed)
    histories: dict[int, int] = {}
    recent: list[int] = []
    next_id = 0
    trace: list[TraceRequest] = []
    for _ in range(requests):
        revisit = recent and rng.random() < revisit_probability
        if revisit:
            conversation_id = rng.choice(recent)
        else:
            conversation_id = next_id
            next_id += 1
            histories[conversation_id] = system_tokens
            recent.append(conversation_id)
            if len(recent) > active_conversations:
                recent.pop(0)
        histories[conversation_id] += rng.randint(turn_tokens_min, turn_tokens_max)
        trace.append(TraceRequest(conversation_id, histories[conversation_id]))
    return trace


def _reuse_tokens(slot: SimulatedSlot, request: TraceRequest, system_tokens: int) -> int:
    if slot.cached_tokens == 0:
        return 0
    if slot.conversation_id == request.conversation_id:
        return min(slot.cached_tokens, request.prompt_tokens)
    return min(system_tokens, slot.cached_tokens, request.prompt_tokens)


def select_slot(
    slots: list[SimulatedSlot],
    request: TraceRequest,
    *,
    policy: str,
    eviction_penalty: float,
    system_tokens: int,
) -> tuple[int, int, int]:
    # Fill never-used capacity before considering an eviction.  Real serving
    # schedulers should not destroy a warm prefix while an empty slot exists.
    empty = next(
        (index for index, slot in enumerate(slots) if slot.cached_tokens == 0),
        None,
    )
    if empty is not None:
        return empty, 0, 0

    if policy == "lru":
        selected = min(range(len(slots)), key=lambda index: slots[index].last_used)
        reuse = _reuse_tokens(slots[selected], request, system_tokens)
        return selected, reuse, max(slots[selected].cached_tokens - reuse, 0)

    candidates: list[tuple[float, int, int, int]] = []
    for index, slot in enumerate(slots):
        reuse = _reuse_tokens(slot, request, system_tokens)
        evicted = max(slot.cached_tokens - reuse, 0)
        score = float(reuse) if policy == "lcp" else reuse - eviction_penalty * evicted
        candidates.append((score, -slot.last_used, index, reuse))
    _, _, selected, reuse = max(candidates)
    return selected, reuse, max(slots[selected].cached_tokens - reuse, 0)


def simulate_trace(
    trace: list[TraceRequest],
    *,
    slots_count: int,
    policy: str,
    eviction_penalty: float = 0.0,
    system_tokens: int = 20,
) -> dict[str, float]:
    if policy not in {"lru", "lcp", "cost_aware"}:
        raise ValueError(f"unknown policy: {policy}")
    slots = [SimulatedSlot() for _ in range(slots_count)]
    processed: list[int] = []
    reused_total = 0
    evicted_total = 0
    requested_total = 0
    for tick, request in enumerate(trace):
        index, reused, evicted = select_slot(
            slots,
            request,
            policy=policy,
            eviction_penalty=eviction_penalty,
            system_tokens=system_tokens,
        )
        prompt_processed = max(request.prompt_tokens - reused, 1)
        processed.append(prompt_processed)
        reused_total += reused
        evicted_total += evicted
        requested_total += request.prompt_tokens
        slots[index] = SimulatedSlot(request.conversation_id, request.prompt_tokens, tick)
    return {
        "requests": float(len(trace)),
        "prefill_tokens_total": float(sum(processed)),
        "prefill_tokens_p95": percentile(processed, 0.95),
        "cache_hit_ratio": reused_total / requested_total if requested_total else 0.0,
        "evicted_tokens_total": float(evicted_total),
    }
