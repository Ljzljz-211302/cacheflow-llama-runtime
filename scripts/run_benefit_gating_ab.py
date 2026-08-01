from __future__ import annotations

import argparse
import csv
import hashlib
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


MODES = ("upstream", "always", "rule", "learned")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = fraction * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def metric(text: str, name: str, labels: dict[str, str]) -> float:
    prefix = f"llamacpp:{name}{{"
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        label_text, value = line.rsplit(" ", 1)
        if all(f'{key}="{expected}"' in label_text for key, expected in labels.items()):
            return float(value)
    raise RuntimeError(f"missing metric {name} with {labels}")


def run_trial(backend: str, mode: str, trial: int) -> dict[str, Any]:
    cuda = backend == "cuda"
    server = ROOT / (
        "build/patched-cuda-ninja3/bin/llama-server.exe"
        if cuda else "build/patched-cpu-noui/bin/Release/llama-server.exe"
    )
    model = ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    port = 18600 + (100 if cuda else 0) + MODES.index(mode) * 20 + trial
    base_url = f"http://127.0.0.1:{port}"
    log_path = ROOT / "results/raw" / f"benefit-{backend}-{mode}-{trial}.log"
    command = [
        str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
        "-c", "4096", "-np", "4", "-b", "512", "-ub", "512", "-t", "8", "-ngl", "99" if cuda else "0",
        "--no-kv-unified", "--metrics", "--no-warmup", "--scheduler-policy", "cacheflow",
        "--benefit-policy", mode, "--benefit-min-observations", "3",
        "--benefit-exploration-interval", "1", "--prefill-chunk-min", "16",
        "--prefill-chunk-max", "128", "--kv-block-runtime", "--kv-block-size", "16",
    ]
    environment = os.environ.copy()
    if cuda:
        environment["PATH"] = str(ROOT / "runtime/cuda-dev/Library/bin") + os.pathsep + environment.get("PATH", "")

    shared = "shared production serving context " * 48
    requests = [
        ("long_prefill", shared + " detailed architecture evidence " * 80, 8),
        ("short_decode", shared + " concise deterministic answer A", 48),
        ("long_prefill", shared + " CUDA KV scheduling analysis " * 90, 8),
        ("short_decode", shared + " concise deterministic answer B", 48),
        ("long_prefill", shared + " compiler database mathematics " * 85, 8),
        ("short_decode", shared + " concise deterministic answer C", 48),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            wait_until_ready(base_url, process=process, log_path=log_path)
            stream_chat(base_url, shared + " donor", max_tokens=1)
            started = time.perf_counter()
            # Two waves keep the online state alive long enough to leave cold
            # start; a one-shot burst only measures the fallback, not learning.
            for wave in range(2):
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = {
                        pool.submit(
                            stream_chat, base_url, prompt + f" wave {wave}",
                            "local-model", predict, 180, 1000 + wave * len(requests) + index,
                        ): (wave * len(requests) + index, kind)
                        for index, (kind, prompt, predict) in enumerate(requests)
                    }
                    for future in as_completed(futures):
                        index, kind = futures[future]
                        result = future.result()
                        rows.append({
                            "request": index,
                            "kind": kind,
                            "ttft_ms": result["ttft_ms"],
                            "total_ms": result["total_ms"],
                            "completion_tokens": result["completion_tokens"],
                            "output_text": result["text"],
                            "output_hash": hashlib.sha256(result["text"].encode()).hexdigest(),
                        })
            burst_ms = (time.perf_counter() - started) * 1000
            with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as response:
                prometheus = response.read().decode()
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    backend_label = "cuda" if cuda else "cpu"
    short_ttft = [float(row["ttft_ms"]) for row in rows if row["kind"] == "short_decode"]
    objective = percentile(short_ttft, 0.95) + 0.25 * burst_ms
    return {
        "backend": backend,
        "mode": mode,
        "trial": trial,
        "objective_ms": objective,
        "short_ttft_p95_ms": percentile(short_ttft, 0.95),
        "latency_p95_ms": percentile([float(row["total_ms"]) for row in rows], 0.95),
        "burst_ms": burst_ms,
        "output_hashes": ";".join(str(row["output_hash"]) for row in sorted(rows, key=lambda item: item["request"])),
        "completion_counts": ";".join(str(row["completion_tokens"]) for row in sorted(rows, key=lambda item: item["request"])),
        "outputs_json": json.dumps([str(row["output_text"]) for row in sorted(rows, key=lambda item: item["request"])], ensure_ascii=False),
        "exact_hash_match_ratio": 0.0,
        "upstream_decisions": metric(prometheus, "benefit_decisions_total", {"backend": backend_label, "action": "upstream"}),
        "cacheflow_decisions": metric(prometheus, "benefit_decisions_total", {"backend": backend_label, "action": "cacheflow"}),
        "exploration_decisions": metric(prometheus, "benefit_exploration_total", {"backend": backend_label}),
        "safety_fallbacks": metric(prometheus, "benefit_safety_fallback_total", {"backend": backend_label}),
        "drift_events": metric(prometheus, "benefit_drift_total", {"backend": backend_label}),
        "positive_lower_bound_decisions": metric(
            prometheus, "benefit_reason_total",
            {"backend": backend_label, "reason": "positive_lower_bound"},
        ),
        "insufficient_evidence_decisions": metric(
            prometheus, "benefit_reason_total",
            {"backend": backend_label, "reason": "insufficient_evidence"},
        ),
        "oracle_objective_ms": "",
        "oracle_regret_ratio": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fresh-process conservative benefit-gating A/B/oracle evaluation")
    parser.add_argument("--backend", choices=("cpu", "cuda", "both"), default="both")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--max-regression", type=float, default=0.03)
    parser.add_argument("--max-oracle-regret", type=float, default=0.20)
    args = parser.parse_args()
    if args.trials < 1:
        raise ValueError("trials must be positive")
    backends = ["cpu", "cuda"] if args.backend == "both" else [args.backend]
    rows: list[dict[str, Any]] = []
    for backend in backends:
        for trial in range(1, args.trials + 1):
            # Latin rotation removes systematic process-order/thermal bias.
            shift = (trial - 1) % len(MODES)
            order = MODES[shift:] + MODES[:shift]
            rows.extend(run_trial(backend, mode, trial) for mode in order)

    # Batch composition can move near-tied greedy logits across a floating-point
    # boundary and can change where EOS appears. Exact text hashes and token
    # counts are retained for audit; HTTP/SSE validity is enforced in run_trial,
    # while semantic quality is covered by run_quality.py in the full suite.
    for backend in backends:
        for trial in range(1, args.trials + 1):
            paired_rows = [row for row in rows if row["backend"] == backend and row["trial"] == trial]
            split_hashes = [str(row["output_hashes"]).split(";") for row in paired_rows]
            matched = sum(len({values[index] for values in split_hashes}) == 1
                          for index in range(len(split_hashes[0])))
            ratio = matched / len(split_hashes[0])
            for row in paired_rows:
                row["exact_hash_match_ratio"] = ratio

    summaries: list[dict[str, Any]] = []
    violations: list[str] = []
    for backend in backends:
        for trial in range(1, args.trials + 1):
            paired = {str(row["mode"]): row for row in rows
                      if row["backend"] == backend and row["trial"] == trial}
            oracle = min(float(paired[mode]["objective_ms"]) for mode in ("upstream", "always", "rule"))
            learned = float(paired["learned"]["objective_ms"])
            paired["learned"]["oracle_objective_ms"] = oracle
            paired["learned"]["oracle_regret_ratio"] = (learned - oracle) / max(oracle, 1e-9)
        for mode in MODES:
            group = [row for row in rows if row["backend"] == backend and row["mode"] == mode]
            summaries.append({
                "backend": backend,
                "mode": mode,
                "trials": len(group),
                "objective_median_ms": statistics.median(float(row["objective_ms"]) for row in group),
                "ttft_p95_ms": percentile([float(row["short_ttft_p95_ms"]) for row in group], 0.95),
                "burst_median_ms": statistics.median(float(row["burst_ms"]) for row in group),
                "cacheflow_decisions": sum(float(row["cacheflow_decisions"]) for row in group),
                "exploration_decisions": sum(float(row["exploration_decisions"]) for row in group),
                "positive_lower_bound_decisions": sum(float(row["positive_lower_bound_decisions"]) for row in group),
                "exact_hash_match_ratio": statistics.median(float(row["exact_hash_match_ratio"]) for row in group),
            })
        learned_group = [row for row in rows if row["backend"] == backend and row["mode"] == "learned"]
        wrong_enable_trials = 0
        harmful_trials = 0
        for trial in range(1, args.trials + 1):
            paired = {str(row["mode"]): row for row in rows
                      if row["backend"] == backend and row["trial"] == trial}
            harmful = float(paired["always"]["objective_ms"]) > float(paired["upstream"]["objective_ms"]) * 1.03
            if harmful:
                harmful_trials += 1
                learned_non_probe = float(paired["learned"]["cacheflow_decisions"]) - float(paired["learned"]["exploration_decisions"])
                wrong_enable_trials += learned_non_probe > 0
        summaries.append({
            "backend": backend,
            "mode": "oracle",
            "trials": len(learned_group),
            "objective_median_ms": statistics.median(float(row["oracle_objective_ms"]) for row in learned_group),
            "ttft_p95_ms": 0.0,
            "burst_median_ms": 0.0,
            "cacheflow_decisions": 0.0,
            "exploration_decisions": 0.0,
            "positive_lower_bound_decisions": 0.0,
            "exact_hash_match_ratio": 1.0,
        })

        by_mode = {str(row["mode"]): row for row in summaries if row["backend"] == backend}
        learned_objective = float(by_mode["learned"]["objective_median_ms"])
        upstream_objective = float(by_mode["upstream"]["objective_median_ms"])
        regret = statistics.median(float(row["oracle_regret_ratio"]) for row in learned_group)
        if learned_objective > upstream_objective * (1 + args.max_regression):
            violations.append(f"{backend} learned policy regressed upstream by more than {args.max_regression:.1%}")
        if regret > args.max_oracle_regret:
            violations.append(f"{backend} learned median oracle regret {regret:.1%} exceeds {args.max_oracle_regret:.1%}")
        if harmful_trials and wrong_enable_trials / harmful_trials > 0.20:
            violations.append(f"{backend} learned wrong-enable rate exceeds 20% on harmful traces")

    trials_path = ROOT / "results/benefit_gating_trials.csv"
    summary_path = ROOT / "results/benefit_gating_summary.csv"
    for path, data in ((trials_path, rows), (summary_path, summaries)):
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    if violations:
        raise RuntimeError("; ".join(violations))


if __name__ == "__main__":
    main()
