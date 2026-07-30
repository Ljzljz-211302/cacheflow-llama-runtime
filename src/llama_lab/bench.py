from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gpu_memory import GpuMemorySampler


def parse_llama_bench_json(text: str) -> list[dict[str, Any]]:
    payload = text.strip()
    if not payload:
        raise ValueError("llama-bench returned empty stdout")
    data = json.loads(payload)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError("llama-bench JSON must be an object or list of objects")
    return data


def capture_environment(bench_exe: Path) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
    }
    commands = {
        "gpu": [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        "llama_devices": [str(bench_exe), "--list-devices"],
    }
    for key, command in commands.items():
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            environment[key] = (completed.stdout + completed.stderr).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            environment[key] = f"unavailable: {exc}"
    return environment


def run_benchmark(config_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.parent.parent
    bench_exe = (root / config["bench_exe"]).resolve()
    if not bench_exe.exists():
        raise FileNotFoundError(f"llama-bench not found: {bench_exe}")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "environment.json").write_text(
        json.dumps(capture_environment(bench_exe), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    models = {model["key"]: model for model in config["models"]}
    for case in config["cases"]:
        for model_key in case["models"]:
            model = models[model_key]
            model_path = (root / model["path"]).resolve()
            if not model_path.exists():
                raise FileNotFoundError(f"model not found: {model_path}")
            command = [
                str(bench_exe),
                "-m",
                str(model_path),
                "-p",
                str(case["prompt_tokens"]),
                "-n",
                str(case["generation_tokens"]),
                "-t",
                str(case["threads"]),
                "-ngl",
                str(case["gpu_layers"]),
                "-r",
                str(case["repetitions"]),
                "-fa",
                case.get("flash_attention", "auto"),
                "-o",
                "json",
            ]
            with GpuMemorySampler() as memory:
                completed = subprocess.run(
                    command,
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env={**os.environ, "NO_COLOR": "1"},
                )
            raw_path = raw_dir / f"{case['name']}-{model_key}.json"
            raw_path.write_text(completed.stdout, encoding="utf-8")
            for record in parse_llama_bench_json(completed.stdout):
                if int(record.get("n_prompt", 0)) > 0:
                    test_name = f"pp{record['n_prompt']}"
                elif int(record.get("n_gen", 0)) > 0:
                    test_name = f"tg{record['n_gen']}"
                else:
                    test_name = "unknown"
                rows.append(
                    {
                        "captured_at_utc": timestamp,
                        "case": case["name"],
                        "model_key": model_key,
                        "quantization": model["quantization"],
                        "test": test_name,
                        "execution": (
                            "CPU-only" if int(case["gpu_layers"]) == 0 else "CUDA"
                        ),
                        "gpu_layers_requested": case["gpu_layers"],
                        "threads_requested": case["threads"],
                        "gpu_memory_baseline_mib": memory.baseline_mib,
                        "gpu_memory_peak_mib": memory.peak_mib,
                        "gpu_memory_increment_mib": memory.increment_mib,
                        **record,
                    }
                )

    write_csv(output_dir / "baseline.csv", rows)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty benchmark")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
