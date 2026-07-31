#!/usr/bin/env python3
"""Three-fresh-process P95 benchmark for whole-sequence copy vs tail COW."""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build/patched-cuda-ninja3/bin/bench-kv-cow-cuda.exe"
RAW = ROOT / "results/raw/cuda-kv-cow.csv"
SUMMARY = ROOT / "results/cuda-kv-cow-summary.json"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def main() -> None:
    env = os.environ.copy()
    env["PATH"] = str(ROOT / "runtime/cuda-dev/Library/bin") + os.pathsep + env.get("PATH", "")
    all_rows: list[dict[str, str]] = []
    for process_trial in range(3):
        completed = subprocess.run(
            [str(EXE)], cwd=ROOT, env=env, check=True, text=True, capture_output=True
        )
        rows = list(csv.DictReader(completed.stdout.splitlines()))
        if len(rows) != 200:
            raise AssertionError(f"fresh process {process_trial} returned {len(rows)} rows")
        for row in rows:
            row["process_trial"] = str(process_trial)
            all_rows.append(row)
    RAW.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["process_trial", "method", "trial", "end_to_end_ms", "bytes", "copy_launches", "extra_device_bytes"]
    with RAW.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        grouped[row["method"]].append(row)
    cases = []
    for method, rows in sorted(grouped.items()):
        values = [float(row["end_to_end_ms"]) for row in rows]
        cases.append({
            "method": method,
            "samples": len(values),
            "median_end_to_end_ms": statistics.median(values),
            "p95_end_to_end_ms": percentile(values, 0.95),
            "bytes_per_operation": int(rows[0]["bytes"]),
            "copy_launches_per_operation": int(rows[0]["copy_launches"]),
            "extra_device_bytes": int(rows[0]["extra_device_bytes"]),
        })
    lookup = {case["method"]: case for case in cases}
    baseline = lookup["whole_sequence_copy"]["p95_end_to_end_ms"]
    variant = lookup["tail_block_cow"]["p95_end_to_end_ms"]
    if not isinstance(baseline, float) or not isinstance(variant, float) or variant >= baseline:
        raise AssertionError(f"tail COW did not improve P95: {baseline} -> {variant}")
    summary = {
        "device": "NVIDIA GeForce RTX 4050 Laptop GPU",
        "model_kv_geometry": "Qwen2.5-0.5B F16 KV, 24 layers x 2 heads x 64 dims",
        "fresh_process_trials": 3,
        "cases": cases,
        "p95_improvement_percent": (baseline - variant) / baseline * 100.0,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
