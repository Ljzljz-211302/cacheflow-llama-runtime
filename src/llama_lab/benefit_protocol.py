from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar, cast


InputT = TypeVar("InputT")
ResultT = TypeVar("ResultT")


def run_staggered_wave(
    items: Sequence[InputT],
    worker: Callable[[InputT], ResultT],
    *,
    max_workers: int,
    admission_stagger_s: float = 0.010,
) -> list[ResultT]:
    """Run an overlapping wave with a deterministic client admission order.

    ThreadPoolExecutor does not promise that submitted callables reach their
    first I/O operation in submission order. That distinction matters for a
    serving trace: racing HTTP clients can change slot assignment and turn a
    paired policy comparison into two different workloads. This coordinator
    admits calls in item order with a small minimum separation while retaining
    overlap after admission.
    """

    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if admission_stagger_s < 0:
        raise ValueError("admission_stagger_s must be nonnegative")
    if not items:
        return []

    condition = threading.Condition()
    next_index = 0
    first_admission = time.perf_counter()

    def invoke(index: int, item: InputT) -> tuple[int, ResultT]:
        nonlocal next_index
        with condition:
            condition.wait_for(lambda: index == next_index)
            deadline = first_admission + index * admission_stagger_s
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            next_index += 1
            condition.notify_all()
        return index, worker(item)

    missing = object()
    ordered: list[ResultT | object] = [missing] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(invoke, index, item) for index, item in enumerate(items)]
        for future in futures:
            index, value = future.result()
            ordered[index] = value

    # Every future either populated its slot or raised above. A private sentinel
    # keeps None available as a legitimate generic worker result.
    if any(value is missing for value in ordered):
        raise RuntimeError("staggered wave completed with a missing result")
    return [cast(ResultT, value) for value in ordered]
