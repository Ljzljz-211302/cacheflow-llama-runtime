from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class CudaProfileTrial:
    mode: str
    trial: int
    cacheflow_decisions: int
    prefill_chunks: int
    prefill_tokens: int
    kernel_launches: int
    copy_bytes: int
    cuda_event_ms: float
    gpu_busy_ratio: float
    maximum_idle_gap_ms: float
    ttft_p95_ms: float
    execute_duration_us: float


@dataclass(frozen=True)
class CudaCausalResult:
    passed: bool
    violations: tuple[str, ...]
    paired_trials: int
    cacheflow_decision_delta: float
    prefill_chunk_delta: float
    prefill_token_delta: float
    kernel_launch_delta: float
    copy_bytes_delta: float
    cuda_event_delta_ms: float
    gpu_busy_delta: float
    idle_gap_delta_ms: float
    ttft_delta_ms: float
    execute_duration_delta_us: float


def analyze_cuda_causality(
    rows: list[CudaProfileTrial],
    minimum_trials: int = 3,
    minimum_ttft_effect_ms: float = 5.0,
    minimum_execute_effect_us: float = 1000.0,
) -> CudaCausalResult:
    by_key = {(row.trial, row.mode): row for row in rows}
    trials = sorted({row.trial for row in rows})
    if any((trial, "upstream") not in by_key or (trial, "always") not in by_key for trial in trials):
        raise ValueError("CUDA profiling requires paired upstream/always processes per trial")

    def deltas(attribute: str) -> list[float]:
        return [
            float(getattr(by_key[(trial, "always")], attribute))
            - float(getattr(by_key[(trial, "upstream")], attribute))
            for trial in trials
        ]

    values = {
        "cacheflow_decision_delta": statistics.median(deltas("cacheflow_decisions")),
        "prefill_chunk_delta": statistics.median(deltas("prefill_chunks")),
        "prefill_token_delta": statistics.median(deltas("prefill_tokens")),
        "kernel_launch_delta": statistics.median(deltas("kernel_launches")),
        "copy_bytes_delta": statistics.median(deltas("copy_bytes")),
        "cuda_event_delta_ms": statistics.median(deltas("cuda_event_ms")),
        "gpu_busy_delta": statistics.median(deltas("gpu_busy_ratio")),
        "idle_gap_delta_ms": statistics.median(deltas("maximum_idle_gap_ms")),
        "ttft_delta_ms": statistics.median(deltas("ttft_p95_ms")),
        "execute_duration_delta_us": statistics.median(deltas("execute_duration_us")),
    }
    violations: list[str] = []
    if len(trials) < minimum_trials:
        violations.append(f"only {len(trials)} paired trials; require {minimum_trials}")
    if values["cacheflow_decision_delta"] <= 0:
        violations.append("policy intervention did not change CacheFlow decisions")
    if (
        values["prefill_chunk_delta"] == 0
        and values["prefill_token_delta"] == 0
    ):
        violations.append("policy intervention did not change the scheduler mediator")
    cuda_evidence = (
        values["kernel_launch_delta"] != 0
        or values["copy_bytes_delta"] != 0
        or abs(values["cuda_event_delta_ms"]) >= 0.01
        or abs(values["gpu_busy_delta"]) >= 0.02
        or abs(values["idle_gap_delta_ms"]) >= 50.0
    )
    if not cuda_evidence:
        violations.append("scheduler change produced no measurable CUDA mediator")
    if not all(
        math.isfinite(values[name])
        for name in ("ttft_delta_ms", "execute_duration_delta_us")
    ):
        violations.append("missing finite latency outcome")
    elif (
        abs(values["ttft_delta_ms"]) < minimum_ttft_effect_ms
        or abs(values["execute_duration_delta_us"]) < minimum_execute_effect_us
    ):
        violations.append("policy intervention produced no material Engine/TTFT outcome")
    return CudaCausalResult(
        passed=not violations,
        violations=tuple(violations),
        paired_trials=len(trials),
        **values,
    )
