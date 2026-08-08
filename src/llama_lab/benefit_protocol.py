from __future__ import annotations

import threading
import time
import re
import statistics
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import TracebackType
from typing import Generic, TypeVar, cast


InputT = TypeVar("InputT")
ResultT = TypeVar("ResultT")


_PROMETHEUS_SAMPLE_RE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)'
    r'(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)$'
)
_PROMETHEUS_LABEL_RE = re.compile(r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<value>[^"]*)"')


def prometheus_histogram_quantile_upper_bound(
    text: str,
    name: str,
    labels: dict[str, str],
    quantile: float,
) -> float:
    """Return the conservative bucket upper bound for a histogram quantile."""

    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    metric_name = f"llamacpp:{name}_bucket"
    buckets: list[tuple[float, float]] = []
    for raw_line in text.splitlines():
        match = _PROMETHEUS_SAMPLE_RE.match(raw_line.strip())
        if not match or match.group("name") != metric_name:
            continue
        sample_labels = {
            item.group("key"): item.group("value")
            for item in _PROMETHEUS_LABEL_RE.finditer(match.group("labels") or "")
        }
        if any(sample_labels.get(key) != value for key, value in labels.items()):
            continue
        if "le" not in sample_labels:
            continue
        upper = (
            float("inf")
            if sample_labels["le"] == "+Inf"
            else float(sample_labels["le"])
        )
        buckets.append((upper, float(match.group("value"))))
    if not buckets:
        raise RuntimeError(f"missing histogram {name} with {labels}")
    buckets.sort(key=lambda item: item[0])
    if buckets[-1][0] != float("inf") or buckets[-1][1] <= 0:
        raise RuntimeError(f"histogram {name} is missing a nonempty +Inf bucket")
    if any(right[1] < left[1] for left, right in zip(buckets, buckets[1:])):
        raise RuntimeError(f"histogram {name} buckets are not cumulative")
    target = quantile * buckets[-1][1]
    for upper, count in buckets:
        if count >= target:
            return upper
    raise RuntimeError(f"histogram {name} has no quantile bucket")


@dataclass(frozen=True)
class ShortLivedAcceptance:
    passed: bool
    performance_status: str
    paired_regression: float
    choose_p99_us: float
    choose_samples: int
    cacheflow_decisions: int
    non_probe_cacheflow_decisions: int
    violation: str = ""


def evaluate_short_lived_acceptance(
    learned_rows: Sequence[dict[str, object]],
    *,
    maximum_regression: float,
    maximum_choose_p99_us: float,
) -> ShortLivedAcceptance:
    """Separate measured intervention effects from a fail-closed null path.

    An experiment with no CacheFlow action cannot identify CacheFlow's causal
    end-to-end effect. It may still verify the production fallback when the
    online chooser itself stays inside its preregistered latency budget.
    """

    if not learned_rows:
        raise ValueError("learned_rows must not be empty")
    paired_regression = statistics.median(
        float(row["upstream_regression_ratio"]) for row in learned_rows
    )
    choose_p99_us = max(float(row["benefit_choose_p99_us"]) for row in learned_rows)
    choose_samples = round(
        sum(float(row["benefit_choose_samples"]) for row in learned_rows)
    )
    cacheflow_decisions = round(
        sum(float(row["cacheflow_decisions"]) for row in learned_rows)
    )
    exploration_decisions = round(
        sum(float(row["exploration_decisions"]) for row in learned_rows)
    )
    non_probe = max(0, cacheflow_decisions - exploration_decisions)
    if choose_samples <= 0:
        return ShortLivedAcceptance(
            False,
            "chooser_evidence_missing",
            paired_regression,
            choose_p99_us,
            choose_samples,
            cacheflow_decisions,
            non_probe,
            "choose latency histogram has no learned-policy samples",
        )
    if choose_p99_us > maximum_choose_p99_us:
        return ShortLivedAcceptance(
            False,
            "chooser_over_budget",
            paired_regression,
            choose_p99_us,
            choose_samples,
            cacheflow_decisions,
            non_probe,
            f"choose P99 {choose_p99_us:.1f} us exceeds {maximum_choose_p99_us:.1f} us",
        )
    if paired_regression <= maximum_regression:
        return ShortLivedAcceptance(
            True,
            "non_regressed",
            paired_regression,
            choose_p99_us,
            choose_samples,
            cacheflow_decisions,
            non_probe,
        )
    if cacheflow_decisions == 0:
        return ShortLivedAcceptance(
            True,
            "inconclusive_no_intervention",
            paired_regression,
            choose_p99_us,
            choose_samples,
            cacheflow_decisions,
            non_probe,
        )
    return ShortLivedAcceptance(
        False,
        "regressed_with_intervention",
        paired_regression,
        choose_p99_us,
        choose_samples,
        cacheflow_decisions,
        non_probe,
        f"paired median regression {paired_regression:.1%} exceeds {maximum_regression:.1%}",
    )


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
