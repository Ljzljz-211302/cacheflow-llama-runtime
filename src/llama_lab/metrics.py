from __future__ import annotations

import math
from collections.abc import Iterable


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("at least one value is required")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_latency(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("at least one row is required")
    ttft = [row["ttft_ms"] for row in rows]
    tpot = [row["tpot_ms"] for row in rows]
    total = [row["total_ms"] for row in rows]
    output_tps = [row["output_tps"] for row in rows]
    return {
        "requests": float(len(rows)),
        "ttft_p50_ms": percentile(ttft, 0.50),
        "ttft_p95_ms": percentile(ttft, 0.95),
        "tpot_p50_ms": percentile(tpot, 0.50),
        "tpot_p95_ms": percentile(tpot, 0.95),
        "total_p50_ms": percentile(total, 0.50),
        "total_p95_ms": percentile(total, 0.95),
        "mean_output_tps": sum(output_tps) / len(output_tps),
    }
