from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import TracebackType
from typing import Generic, TypeVar, cast


InputT = TypeVar("InputT")
ResultT = TypeVar("ResultT")


def complete_latin_orders(
    modes: Sequence[InputT], trials: int
) -> tuple[tuple[InputT, ...], ...]:
    """Return complete Williams-balanced Latin blocks for an experiment.

    Requiring whole blocks balances process position. The Williams ordering
    also makes each directed predecessor pair occur once per block, controlling
    first-order carryover from fresh-process loading and thermal state.
    """

    if not modes:
        raise ValueError("modes must not be empty")
    if len(set(modes)) != len(modes):
        raise ValueError("modes must be unique")
    if len(modes) % 2 != 0:
        raise ValueError("Williams-balanced modes must have even cardinality")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if trials % len(modes) != 0:
        raise ValueError(
            f"trials must contain complete Latin blocks of {len(modes)}"
        )

    mode_tuple = tuple(modes)
    count = len(mode_tuple)
    # Williams' even-order first row: 0, 1, n-1, 2, n-2, ...; modular
    # rotations of this row form a position- and first-order-balanced square.
    first_row = [0]
    for offset in range(1, count // 2 + 1):
        first_row.append(offset)
        if offset < count // 2:
            first_row.append(count - offset)
    return tuple(
        tuple(mode_tuple[(index + offset) % count] for offset in first_row)
        for index in range(trials)
    )


def joint_williams_orders(
    backends: Sequence[str], modes: Sequence[str], trials: int
) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Balance the actual fresh-process stream over backend × policy."""

    if not backends:
        raise ValueError("backends must not be empty")
    if not modes:
        raise ValueError("modes must not be empty")
    treatments = tuple((backend, mode) for backend in backends for mode in modes)
    return complete_latin_orders(treatments, trials)


@dataclass(frozen=True)
class StaggeredWave(Generic[ResultT]):
    results: tuple[ResultT, ...]
    observed_send_order: tuple[int, ...]


class _AdmissionState:
    def __init__(self, stagger_s: float) -> None:
        self.condition = threading.Condition()
        self.next_index = 0
        self.first_admission = time.perf_counter()
        self.stagger_s = stagger_s
        self.observed_send_order: list[int] = []


class _OrderedSendGuard(AbstractContextManager[None]):
    """Hold the ordering turn around the caller's actual socket send."""

    def __init__(self, state: _AdmissionState, index: int) -> None:
        self._state = state
        self._index = index
        self._entered = False
        self._finished = False

    def __enter__(self) -> None:
        if self._entered or self._finished:
            raise RuntimeError("send guard may only be used once")
        condition = self._state.condition
        condition.acquire()
        condition.wait_for(lambda: self._index == self._state.next_index)
        deadline = self._state.first_admission + self._index * self._state.stagger_s
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        self._entered = True
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self._entered or self._finished:
            raise RuntimeError("send guard exit without a matching entry")
        self._state.observed_send_order.append(self._index)
        self._state.next_index += 1
        self._finished = True
        self._state.condition.notify_all()
        self._state.condition.release()
        return None

    def cancel_if_unused(self) -> None:
        """Advance the queue if a worker fails before reaching socket send."""

        if self._entered or self._finished:
            return
        with self._state.condition:
            self._state.condition.wait_for(
                lambda: self._index == self._state.next_index
            )
            self._state.next_index += 1
            self._finished = True
            self._state.condition.notify_all()


def run_staggered_wave(
    items: Sequence[InputT],
    worker: Callable[[InputT, AbstractContextManager[None]], ResultT],
    *,
    max_workers: int,
    admission_stagger_s: float = 0.010,
) -> StaggeredWave[ResultT]:
    """Run overlapping requests while ordering their actual socket sends.

    The worker must pass its one-shot send guard to the HTTP client and hold it
    only around connect/request-body transmission. Response wait and streaming
    happen after the guard is released, so requests still overlap.
    """

    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if admission_stagger_s < 0:
        raise ValueError("admission_stagger_s must be nonnegative")
    if not items:
        return StaggeredWave((), ())

    state = _AdmissionState(admission_stagger_s)

    def invoke(index: int, item: InputT) -> tuple[int, ResultT]:
        guard = _OrderedSendGuard(state, index)
        try:
            return index, worker(item, guard)
        finally:
            guard.cancel_if_unused()

    missing = object()
    ordered: list[ResultT | object] = [missing] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(invoke, index, item) for index, item in enumerate(items)]
        for future in futures:
            index, value = future.result()
            ordered[index] = value

    if any(value is missing for value in ordered):
        raise RuntimeError("staggered wave completed with a missing result")
    return StaggeredWave(
        tuple(cast(ResultT, value) for value in ordered),
        tuple(state.observed_send_order),
    )
