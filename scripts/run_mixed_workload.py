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
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llama_lab.server_bench import wait_until_ready
from llama_lab.streaming import stream_chat


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = quantile * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def metrics(base_url: str) -> str:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as response:
        return response.read().decode()


def run_trial(
    backend: str, policy: str, trial: int, prefix: str, engine_trace: bool
) -> tuple[list[dict[str, Any]], str]:
    cuda = backend == "cuda"
    server = ROOT / (
        "build/patched-cuda-ninja3/bin/llama-server.exe"
        if cuda else "build/patched-cpu-noui/bin/Release/llama-server.exe"
    )
    model = ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    port = 8130 + (10 if cuda else 0) + (3 if policy == "cacheflow" else 0) + trial
    base_url = f"http://127.0.0.1:{port}"
    log_path = ROOT / f"results/raw/{prefix}-{backend}-{policy}-trial-{trial}.log"
    command = [
        str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
        "-c", "4096", "-np", "4", "-t", "8", "-ngl", "99" if cuda else "0",
        "--no-kv-unified", "--metrics", "--no-warmup", "--scheduler-policy", policy,
    ]
    if policy == "cacheflow":
        command.extend(["--kv-block-runtime", "--kv-block-size", "16"])
    if engine_trace:
        command.extend(["--engine-trace", str(
            ROOT / f"results/raw/engine-trace-{backend}-{policy}-trial-{trial}.json"
        )])
    environment = os.environ.copy()
    if cuda:
        cuda_bin = ROOT / "runtime/cuda-dev/Library/bin"
        environment["PATH"] = str(cuda_bin) + os.pathsep + environment.get("PATH", "")

    shared = "shared system instruction for mixed serving " * 60
    requests = [
        ("long_prefill", shared + (" detailed evidence and reasoning" * 100), 8),
        ("short_decode", shared + " concise answer A", 48),
        ("long_prefill", shared + (" architecture analysis" * 120), 8),
        ("short_decode", shared + " concise answer B", 48),
        ("long_prefill", shared + (" database compiler math" * 110), 8),
        ("short_decode", shared + " concise answer C", 48),
        ("long_prefill", shared + (" CUDA scheduling kernel" * 110), 8),
        ("short_decode", shared + " concise answer D", 48),
    ]

    rows: list[dict[str, Any]] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            wait_until_ready(base_url, process=process, log_path=log_path)
            # Seed one reusable prefix before the mixed burst.
            stream_chat(base_url, shared + " donor", max_tokens=1)
            burst_started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=4) as pool:
                pending = {
                    pool.submit(stream_chat, base_url, prompt, "local-model", predict, 180):
                    (index, kind)
                    for index, (kind, prompt, predict) in enumerate(requests)
                }
                for future in as_completed(pending):
                    index, kind = pending[future]
                    result = future.result()
                    rows.append({
                        "backend": backend,
                        "policy": policy,
                        "trial": trial,
                        "request": index,
                        "kind": kind,
                        "ttft_ms": result["ttft_ms"],
                        "tpot_ms": result["tpot_ms"],
                        "total_ms": result["total_ms"],
                        "prompt_tokens": result["prompt_tokens"],
                        "completion_tokens": result["completion_tokens"],
                        "output_tps": result["output_tps"],
                    })
            burst_ms = (time.perf_counter() - burst_started) * 1000
            prometheus = metrics(base_url)
            for row in rows:
                row["burst_ms"] = burst_ms
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    return rows, prometheus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--backend", choices=["cpu", "cuda", "both"], default="both")
    parser.add_argument("--policy", choices=["upstream", "cacheflow", "both"], default="both")
    parser.add_argument("--output-prefix", default="mixed_workload")
    parser.add_argument("--engine-trace", action="store_true")
    args = parser.parse_args()
    backends = ["cpu", "cuda"] if args.backend == "both" else [args.backend]
    policies = ["upstream", "cacheflow"] if args.policy == "both" else [args.policy]
    all_rows: list[dict[str, Any]] = []
    for backend in backends:
        for policy in policies:
            for trial in range(1, args.trials + 1):
                rows, prometheus = run_trial(
                    backend, policy, trial, args.output_prefix, args.engine_trace
                )
                all_rows.extend(rows)
                (ROOT / f"results/raw/{args.output_prefix}-{backend}-{policy}-trial-{trial}.prom").write_text(
                    prometheus, encoding="utf-8"
                )

    trials_path = ROOT / f"results/{args.output_prefix}_trials.csv"
    with trials_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(all_rows, key=lambda row: (
            row["backend"], row["policy"], row["trial"], row["request"])))

    summary: list[dict[str, Any]] = []
    for backend in backends:
        for policy in policies:
            group = [row for row in all_rows if row["backend"] == backend and row["policy"] == policy]
            summary.append({
                "backend": backend,
                "policy": policy,
                "trials": args.trials,
                "requests": len(group),
                "ttft_median_ms": statistics.median(row["ttft_ms"] for row in group),
                "ttft_p95_ms": percentile([row["ttft_ms"] for row in group], 0.95),
                "tpot_p95_ms": percentile([row["tpot_ms"] for row in group], 0.95),
                "latency_p95_ms": percentile([row["total_ms"] for row in group], 0.95),
                "burst_median_ms": statistics.median(row["burst_ms"] for row in group),
                "aggregate_output_tps": (sum(row["completion_tokens"] for row in group) / args.trials) /
                    (statistics.median(row["burst_ms"] for row in group) / 1000),
            })
    summary_path = ROOT / f"results/{args.output_prefix}_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
