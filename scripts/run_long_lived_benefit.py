#!/usr/bin/env python3
"""Validate online learning, confidence enablement, and shift fallback in one server."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llama_lab.benefit_experiment import (  # noqa: E402
    BenefitSnapshot,
    LongLivedAcceptance,
    PhaseEvidence,
    evaluate_long_lived,
)
from llama_lab.server_bench import wait_until_ready  # noqa: E402
from llama_lab.streaming import stream_chat  # noqa: E402


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = fraction * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def fetch_snapshot(base_url: str, backend: str) -> BenefitSnapshot:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as response:
        return BenefitSnapshot.from_prometheus(response.read().decode(), backend)


def stable_requests(wave: int) -> list[tuple[str, int]]:
    shared = "shared cacheflow production runtime context " * 64
    return [
        (shared + "long prefill scheduling evidence " * 96 + f" wave {wave} A", 6),
        (shared + "short decode latency answer A " + f"wave {wave}", 36),
        (shared + "long KV pressure and chunking evidence " * 88 + f" wave {wave} B", 6),
        (shared + "short decode latency answer B " + f"wave {wave}", 36),
        (shared + "long online ridge confidence evidence " * 92 + f" wave {wave} C", 6),
        (shared + "short decode latency answer C " + f"wave {wave}", 36),
    ]


def cold_requests(wave: int) -> list[tuple[str, int]]:
    # Keep the initial phase deliberately short: enough competing prefills to
    # establish the upstream model, but not a full high-intensity wave that can
    # both explore and converge before the first observable boundary.
    common = "cold start upstream baseline " * 8
    return [
        (common + (f"independent branch {index} " * 48) + f"wave {wave}", 2)
        for index in range(4)
    ]


def shift_requests(wave: int) -> list[tuple[str, int]]:
    # Sequential lone-prefill requests deliberately leave the multi-prefill
    # region where chunked CacheFlow has a structural latency opportunity.
    return [
        (
            "distribution shifted to independent throughput-only prompt " * 180
            + f"shift wave {wave}",
            8,
        )
    ]


def execute_wave(
    base_url: str,
    phase: str,
    wave: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    if phase == "cold_start":
        requests = cold_requests(wave)
    elif phase == "distribution_shift":
        requests = shift_requests(wave)
    else:
        requests = stable_requests(wave)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                stream_chat,
                base_url,
                prompt,
                "local-model",
                predict,
                300,
                20260801 + wave * len(requests) + index,
            ): index
            for index, (prompt, predict) in enumerate(requests)
        }
        for future in as_completed(futures):
            result = future.result()
            if not result["text"]:
                raise AssertionError(f"empty output in {phase} wave {wave}")
            rows.append(result)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    )
    parser.add_argument("--cold-waves", type=int, default=1)
    parser.add_argument("--stable-waves", type=int, default=48)
    parser.add_argument("--shift-waves", type=int, default=4)
    parser.add_argument("--max-ttft-ms", type=float, default=500.0)
    parser.add_argument("--minimum-observations", type=int, default=12)
    parser.add_argument("--confidence-beta", type=float, default=1.0)
    parser.add_argument("--observe-only", action="store_true")
    parser.add_argument("--port", type=int, default=19680)
    args = parser.parse_args()
    if min(args.cold_waves, args.stable_waves, args.shift_waves) < 1:
        raise ValueError("every phase must contain at least one wave")
    model = args.model.resolve()
    if not model.exists():
        raise FileNotFoundError(f"model not found: {model}")

    cuda = args.backend == "cuda"
    server = ROOT / (
        "build/patched-cuda-ninja3/bin/llama-server.exe"
        if cuda
        else "build/patched-cpu-noui/bin/Release/llama-server.exe"
    )
    base_url = f"http://127.0.0.1:{args.port}"
    log_path = ROOT / "results/raw" / f"long-lived-benefit-{args.backend}.log"
    command = [
        str(server),
        "-m", str(model),
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "-c", "6144",
        "-np", "4",
        "-b", "512",
        "-ub", "512",
        "-t", "8",
        "-ngl", "99" if cuda else "0",
        "--no-kv-unified",
        "--metrics",
        "--no-warmup",
        "--scheduler-policy", "cacheflow",
        "--benefit-policy", "learned",
        "--benefit-min-observations", str(args.minimum_observations),
        "--benefit-exploration-interval", "1",
        "--benefit-confidence-beta", str(args.confidence_beta),
        "--benefit-safety-margin-ms", "0.05",
        "--benefit-drift-ratio", "2.5",
        "--benefit-drift-consecutive", "2",
        "--benefit-cooldown-decisions", "6",
        "--prefill-chunk-min", "16",
        "--prefill-chunk-max", "128",
        "--kv-block-runtime",
        "--kv-block-size", "16",
    ]
    environment = os.environ.copy()
    if cuda:
        environment["PATH"] = (
            str(ROOT / "runtime/cuda-dev/Library/bin")
            + os.pathsep
            + environment.get("PATH", "")
        )

    wave_rows: list[dict[str, Any]] = []
    phase_evidence: list[PhaseEvidence] = []
    phase_plan = (
        ("cold_start", args.cold_waves, 4),
        ("stable_reuse", args.stable_waves, 4),
        ("distribution_shift", args.shift_waves, 1),
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            wait_until_ready(base_url, process=process, log_path=log_path)
            previous = fetch_snapshot(base_url, args.backend)
            global_wave = 0
            for phase, waves, concurrency in phase_plan:
                phase_start = previous
                phase_ttft: list[float] = []
                positive_waves = 0
                positive_streak = 0
                maximum_positive_streak = 0
                for phase_wave in range(waves):
                    global_wave += 1
                    requests = execute_wave(base_url, phase, global_wave, concurrency)
                    phase_ttft.extend(float(row["ttft_ms"]) for row in requests)
                    current = fetch_snapshot(base_url, args.backend)
                    delta = current.delta(previous)
                    if delta.positive_decisions > 0:
                        positive_waves += 1
                        positive_streak += 1
                        maximum_positive_streak = max(maximum_positive_streak, positive_streak)
                    else:
                        positive_streak = 0
                    wave_rows.append(
                        {
                            "backend": args.backend,
                            "model": model.name,
                            "phase": phase,
                            "phase_wave": phase_wave + 1,
                            "global_wave": global_wave,
                            "requests": len(requests),
                            "ttft_p95_ms": percentile(
                                [float(row["ttft_ms"]) for row in requests], 0.95
                            ),
                            "upstream_decisions": delta.upstream_decisions,
                            "cacheflow_decisions": delta.cacheflow_decisions,
                            "exploration_decisions": delta.exploration_decisions,
                            "positive_decisions": delta.positive_decisions,
                            "drift_events": delta.drift_events,
                            "safety_fallbacks": delta.safety_fallbacks,
                            "cooldown_remaining": current.cooldown_remaining,
                            "predicted_benefit_ms": current.predicted_benefit_ms,
                            "uncertainty_ms": current.uncertainty_ms,
                        }
                    )
                    print(
                        f"{phase} {phase_wave + 1}/{waves}: "
                        f"positive={current.positive_decisions} "
                        f"explore={current.exploration_decisions} "
                        f"drift={current.drift_events}"
                    )
                    previous = current
                phase_delta = previous.delta(phase_start)
                phase_evidence.append(
                    PhaseEvidence(
                        phase=phase,
                        upstream_decisions=phase_delta.upstream_decisions,
                        cacheflow_decisions=phase_delta.cacheflow_decisions,
                        exploration_decisions=phase_delta.exploration_decisions,
                        positive_decisions=phase_delta.positive_decisions,
                        drift_events=phase_delta.drift_events,
                        safety_fallbacks=phase_delta.safety_fallbacks,
                        ttft_p95_ms=percentile(phase_ttft, 0.95),
                        predicted_benefit_ms=previous.predicted_benefit_ms,
                        uncertainty_ms=previous.uncertainty_ms,
                        positive_waves=positive_waves,
                        max_consecutive_positive_waves=maximum_positive_streak,
                    )
                )
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    acceptance = evaluate_long_lived(
        phase_evidence,
        LongLivedAcceptance(maximum_ttft_ms=args.max_ttft_ms),
    )
    wave_path = ROOT / "results" / f"long_lived_benefit_{args.backend}_waves.csv"
    summary_path = ROOT / "results" / f"long_lived_benefit_{args.backend}_summary.json"
    with wave_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(wave_rows[0]))
        writer.writeheader()
        writer.writerows(wave_rows)
    summary = {
        "backend": args.backend,
        "model": model.name,
        "minimum_observations": args.minimum_observations,
        "confidence_beta": args.confidence_beta,
        "server_pid_reused_across_phases": True,
        "waves": len(wave_rows),
        "phases": [asdict(phase) for phase in phase_evidence],
        "acceptance": asdict(acceptance),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not acceptance.passed and not args.observe_only:
        raise RuntimeError("; ".join(acceptance.violations))


if __name__ == "__main__":
    main()
