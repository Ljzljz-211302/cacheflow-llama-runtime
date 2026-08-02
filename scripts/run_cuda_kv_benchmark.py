#!/usr/bin/env python3
"""Run the reproducible CUDA KV transport benchmark and persist raw + summary data."""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build" / "patched-cuda-ninja3" / "bin" / "bench-kv-block-cuda.exe"
RAW = ROOT / "results" / "raw" / "cuda-kv-transport.csv"
SUMMARY = ROOT / "results" / "cuda-kv-transport-summary.json"


def main() -> None:
    if not EXE.exists():
        raise SystemExit(f"benchmark executable is missing: {EXE}")
    env = os.environ.copy()
    cuda_bin = ROOT / "runtime" / "cuda-dev" / "Library" / "bin"
    env["PATH"] = str(cuda_bin) + os.pathsep + env.get("PATH", "")
    completed = subprocess.run(
        [str(EXE)], cwd=ROOT, env=env, check=True, text=True, capture_output=True
    )
    rows = list(csv.DictReader(completed.stdout.splitlines()))
    if len(rows) != 160:
        raise AssertionError(f"expected 160 benchmark samples, received {len(rows)}")
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_text(completed.stdout, encoding="utf-8", newline="")

    samples: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        samples[(row["method"], int(row["blocks"]))].append(
            {"gpu_ms": float(row["gpu_ms"]), "end_to_end_ms": float(row["end_to_end_ms"])}
        )
    summary: dict[str, object] = {
        "device": "NVIDIA GeForce RTX 4050 Laptop GPU",
        "layout": {"layers": 32, "kv_heads": 8, "head_dim": 128, "block_tokens": 16, "dtype": "fp16"},
        "trials_per_case": 20,
        "design": "paired trials with alternating scalar/vectorized execution order",
        "cases": [],
        "paired_comparisons": [],
    }
    cases = summary["cases"]
    assert isinstance(cases, list)
    for (method, blocks), values in sorted(samples.items(), key=lambda item: (item[0][1], item[0][0])):
        cases.append(
            {
                "method": method,
                "blocks": blocks,
                "median_gpu_ms": statistics.median(v["gpu_ms"] for v in values),
                "median_end_to_end_ms": statistics.median(v["end_to_end_ms"] for v in values),
            }
        )
    comparisons = summary["paired_comparisons"]
    assert isinstance(comparisons, list)
    material_improvement = False
    worst_gpu_improvement = float("inf")
    best_gpu_improvement = float("-inf")
    for blocks in (1, 4, 16, 32):
        scalar = samples[("scalar_gather_scatter", blocks)]
        vectorized = samples[("vectorized_gather_scatter", blocks)]
        gpu_improvements = [
            (left["gpu_ms"] - right["gpu_ms"]) / left["gpu_ms"] * 100.0
            for left, right in zip(scalar, vectorized)
        ]
        end_to_end_improvements = [
            (left["end_to_end_ms"] - right["end_to_end_ms"]) / left["end_to_end_ms"] * 100.0
            for left, right in zip(scalar, vectorized)
        ]
        median_gpu_improvement = statistics.median(gpu_improvements)
        worst_gpu_improvement = min(worst_gpu_improvement, median_gpu_improvement)
        best_gpu_improvement = max(best_gpu_improvement, median_gpu_improvement)
        comparisons.append({
            "blocks": blocks,
            "paired_median_gpu_improvement_percent": median_gpu_improvement,
            "paired_median_end_to_end_improvement_percent": statistics.median(end_to_end_improvements),
        })
        if median_gpu_improvement < -3.0:
            raise AssertionError(f"vectorized remap regressed by more than 3% for {blocks} blocks")
        material_improvement = material_improvement or median_gpu_improvement >= 10.0
    if not material_improvement:
        raise AssertionError("vectorized remap produced no material paired GPU improvement")
    summary["acceptance"] = {
        "maximum_allowed_paired_regression_percent": 3.0,
        "required_material_improvement_percent": 10.0,
        "worst_observed_paired_gpu_improvement_percent": worst_gpu_improvement,
        "best_observed_paired_gpu_improvement_percent": best_gpu_improvement,
        "passed": True,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"raw": str(RAW), "summary": str(SUMMARY), "samples": len(rows)}))


if __name__ == "__main__":
    main()
