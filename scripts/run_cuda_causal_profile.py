#!/usr/bin/env python3
"""Controlled policy -> scheduler -> CUDA -> TTFT causal profiling chain."""

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

from llama_lab.benefit_experiment import labeled_metric  # noqa: E402
from llama_lab.cuda_causality import CudaProfileTrial, analyze_cuda_causality  # noqa: E402
from llama_lab.prometheus import parse_prometheus_text  # noqa: E402
from llama_lab.server_bench import wait_until_ready  # noqa: E402
from llama_lab.streaming import stream_chat  # noqa: E402


MODES = ("upstream", "always")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = fraction * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def start_gpu_sampler(path: Path) -> tuple[subprocess.Popen[bytes], Any]:
    output = path.open("wb")
    try:
        process = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu",
                "--format=csv,noheader,nounits",
                "--loop-ms=100",
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        output.close()
        raise
    return process, output


def stop_gpu_sampler(process: subprocess.Popen[bytes], output: Any) -> list[float]:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    output.close()
    values: list[float] = []
    path = Path(output.name)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            values.append(float(line.rsplit(",", 1)[1].strip()))
        except (IndexError, ValueError):
            continue
    if not values:
        raise RuntimeError(f"nvidia-smi produced no GPU utilization samples: {path}")
    return values


def maximum_idle_gap_ms(samples: list[float]) -> float:
    longest = current = 0
    for value in samples:
        if value <= 5.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest * 100.0


def workload(base_url: str, trial: int) -> list[dict[str, Any]]:
    shared = "CUDA causal profiling shared prefix " * 72
    requests = [
        (shared + "large prefill branch A " * 92, 8),
        (shared + "latency-sensitive decode A", 48),
        (shared + "large prefill branch B " * 86, 8),
        (shared + "latency-sensitive decode B", 48),
        (shared + "large prefill branch C " * 88, 8),
        (shared + "latency-sensitive decode C", 48),
    ]
    rows: list[dict[str, Any]] = []
    for wave in range(2):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(
                    stream_chat,
                    base_url,
                    prompt + f" trial {trial} wave {wave}",
                    "local-model",
                    predict,
                    300,
                    20260820 + trial * 100 + wave * len(requests) + index,
                ): f"trial-{trial}-wave-{wave}-request-{index}"
                for index, (prompt, predict) in enumerate(requests)
            }
            for future in as_completed(futures):
                result = future.result()
                if not result["text"]:
                    raise AssertionError("CUDA profiling request returned no output")
                result["request_id"] = futures[future]
                rows.append(result)
    return rows


def run_trial(
    mode: str, trial: int, output_dir: Path
) -> tuple[CudaProfileTrial, dict[str, Any]]:
    server = ROOT / "build/patched-cuda-ninja3/bin/llama-server.exe"
    model = ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    port = 19800 + MODES.index(mode) * 20 + trial
    base_url = f"http://127.0.0.1:{port}"
    raw = output_dir / "raw"
    log_path = raw / f"cuda-causal-{mode}-{trial}.log"
    trace_path = raw / f"cuda-causal-{mode}-{trial}.json"
    gpu_path = raw / f"cuda-causal-gpu-{mode}-{trial}.csv"
    command = [
        str(server), "-m", str(model),
        "--host", "127.0.0.1", "--port", str(port),
        "-c", "4096", "-np", "4", "-b", "512", "-ub", "512", "-t", "8",
        "-ngl", "99", "--no-kv-unified", "--metrics", "--no-warmup",
        "--scheduler-policy", "cacheflow", "--benefit-policy", mode,
        "--prefill-chunk-min", "16", "--prefill-chunk-max", "128",
        "--kv-block-runtime", "--kv-block-size", "16",
        "--engine-trace", str(trace_path),
    ]
    environment = os.environ.copy()
    environment["PATH"] = (
        str(ROOT / "runtime/cuda-dev/Library/bin")
        + os.pathsep
        + environment.get("PATH", "")
    )
    raw.mkdir(parents=True, exist_ok=True)
    sampler: subprocess.Popen[bytes] | None = None
    sampler_output = None
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
            sampler, sampler_output = start_gpu_sampler(gpu_path)
            request_rows = workload(base_url, trial)
            with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as response:
                prometheus = response.read().decode()
            time.sleep(0.2)
        finally:
            try:
                if sampler is not None and sampler_output is not None:
                    gpu_samples = stop_gpu_sampler(sampler, sampler_output)
                else:
                    gpu_samples = []
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

    samples = parse_prometheus_text(prometheus)
    events = json.loads(trace_path.read_text(encoding="utf-8"))["traceEvents"]
    execute_us = sum(
        float(event["dur"])
        for event in events
        if event.get("name") == "execute" and event.get("ph") == "X"
    )
    cuda_event_ms = 1000.0 * sum(
        samples.get(name, 0.0)
        for name in (
            "llamacpp:cuda_kv_copy_seconds",
            "llamacpp:cuda_kv_swap_out_seconds",
            "llamacpp:cuda_kv_swap_in_seconds",
        )
    )
    profile = CudaProfileTrial(
        mode=mode,
        trial=trial,
        cacheflow_decisions=int(labeled_metric(
            prometheus,
            "llamacpp:benefit_decisions_total",
            {"backend": "cuda", "action": "cacheflow"},
        )),
        prefill_chunks=int(samples["llamacpp:prefill_chunks_scheduled_total"]),
        prefill_tokens=int(samples["llamacpp:prefill_tokens_scheduled_total"]),
        kernel_launches=int(samples["llamacpp:cuda_kv_kernel_launches_total"]),
        copy_bytes=int(samples["llamacpp:cuda_kv_copy_bytes_total"]),
        cuda_event_ms=cuda_event_ms,
        gpu_busy_ratio=sum(value > 5.0 for value in gpu_samples) / len(gpu_samples),
        maximum_idle_gap_ms=maximum_idle_gap_ms(gpu_samples),
        ttft_p95_ms=percentile([float(row["ttft_ms"]) for row in request_rows], 0.95),
        execute_duration_us=execute_us,
    )
    evidence_metrics = {
        name: samples.get(name, 0.0)
        for name in (
            "llamacpp:prefill_chunks_scheduled_total",
            "llamacpp:prefill_tokens_scheduled_total",
            "llamacpp:cuda_kv_kernel_launches_total",
            "llamacpp:cuda_kv_copy_bytes_total",
            "llamacpp:cuda_kv_copy_seconds",
            "llamacpp:cuda_kv_swap_out_seconds",
            "llamacpp:cuda_kv_swap_in_seconds",
        )
    }
    evidence = {
        "mode": mode,
        "trial": trial,
        "trial_id": f"service-trial-{trial}-{mode}",
        "server_pid": process.pid,
        "requests": [
            {
                "request_id": row["request_id"],
                "ttft_ms": row["ttft_ms"],
                "total_ms": row.get("total_ms"),
            }
            for row in request_rows
        ],
        "gpu_utilization_samples_percent": gpu_samples,
        "engine_trace_events": events,
        "prometheus_snapshot": evidence_metrics,
    }
    return profile, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if args.trials < 1:
        raise ValueError("trials must be positive")
    rows: list[CudaProfileTrial] = []
    evidence_rows: list[dict[str, Any]] = []
    for trial in range(1, args.trials + 1):
        shift = (trial - 1) % len(MODES)
        order = MODES[shift:] + MODES[:shift]
        for mode in order:
            row, evidence = run_trial(mode, trial, args.output_dir)
            rows.append(row)
            evidence_rows.append(evidence)
            print(f"trial={trial} mode={mode} ttft_p95={row.ttft_p95_ms:.2f} ms")

    result = analyze_cuda_causality(rows, minimum_trials=args.trials)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = args.output_dir / "cuda_causal_profile_trials.csv"
    with trials_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    summary = {
        "method": "paired Latin intervention + CUDA Events + 100 ms nvidia-smi activity sampling + in-process Engine trace",
        "scope": "custom CacheFlow KV kernels/events; not a full Nsight Compute kernel census",
        "chain": [
            "benefit policy intervention",
            "prefill token/chunk allocation",
            "CUDA KV kernels/copy events and GPU idle/busy samples",
            "Engine execute duration and request TTFT P95",
        ],
        "result": asdict(result),
    }
    summary_path = args.output_dir / "cuda_causal_profile_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    evidence_path = args.output_dir / "cuda_causal_profile_evidence.json"
    evidence_path.write_text(
        json.dumps({"sampling_interval_ms": 100, "trials": evidence_rows}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not result.passed:
        raise RuntimeError("; ".join(result.violations))


if __name__ == "__main__":
    main()
