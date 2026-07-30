from __future__ import annotations

import threading
from collections import defaultdict


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = defaultdict(float)
        self._observations: dict[str, list[float]] = defaultdict(list)

    def increment(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._observations[name].append(value)

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            result = dict(self._counters)
            result.update(self._gauges)
            for name, values in self._observations.items():
                result[f"{name}_count"] = float(len(values))
                result[f"{name}_sum"] = sum(values)
                result[f"{name}_max"] = max(values, default=0.0)
            return result

    def prometheus(self) -> str:
        samples = self.snapshot()
        return "".join(
            f"# TYPE cacheflow_{name} gauge\ncacheflow_{name} {value}\n"
            for name, value in sorted(samples.items())
        )
