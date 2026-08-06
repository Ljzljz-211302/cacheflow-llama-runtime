#!/usr/bin/env python3
"""Run a preregistered experiment and emit manifest.json plus trials.jsonl."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llama_lab.research_protocol import (  # noqa: E402
    load_research_protocol,
    package_cpu_correctness_run,
    package_cuda_remap_trials,
)


def git_revision(directory: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=directory,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def git_dirty(directory: Path) -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=directory,
        check=True,
        text=True,
        capture_output=True,
    )
    return bool(completed.stdout.strip())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_snapshot(command: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command, check=False, text=True, capture_output=True, timeout=15
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def capture_environment() -> dict[str, object]:
    nvidia_fields = (
        "uuid,pci.bus_id,name,driver_version,pstate,temperature.gpu,power.draw,"
        "power.limit,clocks.current.sm,clocks.current.memory"
    )
    nvcc = ROOT / "runtime" / "cuda-dev" / "Library" / "bin" / "nvcc.exe"
    return {
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "nvidia_smi": command_snapshot(
            ["nvidia-smi", f"--query-gpu={nvidia_fields}", "--format=csv,noheader,nounits"]
        ),
        "nvcc": command_snapshot([str(nvcc), "--version"]),
        "cmake": command_snapshot(["cmake", "--version"]),
        "controlled_environment": {
            name: os.environ.get(name)
            for name in ("CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run/package a preregistered CacheFlow research experiment"
    )
    parser.add_argument(
        "--experiment",
        choices=("h1-vector-remap", "cpu-correctness"),
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or dedicated directory for manifest.json and trials.jsonl",
    )
    parser.add_argument(
        "--skip-execution",
        action="store_true",
        help="Package an already-generated H1 source CSV; not valid for CPU fallback",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "results" / "raw" / "cuda-kv-transport.csv",
    )
    args = parser.parse_args()

    protocol_path = ROOT / "config" / "research_protocol.json"
    claims_path = ROOT / "config" / "research_claims.json"
    environment_path = ROOT / "results" / "environment.json"
    protocol = load_research_protocol(protocol_path, claims_path)
    captured_at = dt.datetime.now(dt.timezone.utc).isoformat()
    outer_revision = git_revision(ROOT)
    outer_dirty = git_dirty(ROOT)
    current_environment = capture_environment()

    if args.experiment == "h1-vector-remap":
        workload = protocol["workloads"]["h1_vector_remap"]
        command = [str(item) for item in workload["runner"]]
        if not args.skip_execution:
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                (str(ROOT / "src"), str(ROOT / "prototypes"))
            )
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        artifacts = package_cuda_remap_trials(
            protocol_path,
            claims_path,
            args.source,
            environment_path,
            args.output_dir,
            command,
            outer_revision,
            captured_at,
            vendor_revision=git_revision(ROOT / "vendor" / "llama.cpp"),
            environment_override=current_environment,
            repository_dirty=outer_dirty,
            vendor_dirty=git_dirty(ROOT / "vendor" / "llama.cpp"),
            binary_sha256=file_sha256(
                ROOT / "build" / "patched-cuda-ninja3" / "bin" / "bench-kv-block-cuda.exe"
            ),
        )
    else:
        if args.skip_execution:
            parser.error("--skip-execution is not valid for cpu-correctness")
        command = [str(item) for item in protocol["cpu_correctness_fallback"]["command"]]
        start = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT, check=False)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        artifacts = package_cpu_correctness_run(
            protocol_path,
            claims_path,
            environment_path,
            args.output_dir,
            command,
            outer_revision,
            captured_at,
            exit_code=completed.returncode,
            elapsed_ms=elapsed_ms,
            environment_override=current_environment,
            repository_dirty=outer_dirty,
        )
        if completed.returncode != 0:
            print(json.dumps({"manifest": str(artifacts.manifest), "status": "failed"}))
            return completed.returncode

    print(
        json.dumps(
            {"manifest": str(artifacts.manifest), "trials": str(artifacts.trials)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
