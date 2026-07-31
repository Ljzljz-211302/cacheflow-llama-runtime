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
    if len(rows) != 40:
        raise AssertionError(f"expected 40 benchmark samples, received {len(rows)}")
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
        "trials_per_case": 5,
        "cases": [],
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
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"raw": str(RAW), "summary": str(SUMMARY), "samples": len(rows)}))


if __name__ == "__main__":
    main()
