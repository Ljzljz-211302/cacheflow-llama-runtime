#!/usr/bin/env python3
"""Prove learned-policy state survives restart and corruption fails closed."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llama_lab.server_bench import wait_until_ready  # noqa: E402


PORT = 19720
BASE_URL = f"http://127.0.0.1:{PORT}"
CHECKPOINT = ROOT / "results/raw/benefit-policy-production-state.json"
LOG = ROOT / "results/raw/benefit-checkpoint-smoke.log"


def prometheus() -> str:
    with urllib.request.urlopen(f"{BASE_URL}/metrics", timeout=30) as response:
        return response.read().decode()


def metric(text: str, sample: str) -> float:
    prefix = f"llamacpp:{sample} "
    line = next((row for row in text.splitlines() if row.startswith(prefix)), None)
    if line is None:
        raise AssertionError(f"missing Prometheus sample: {sample}")
    return float(line.rsplit(" ", 1)[1])


def completion(index: int) -> None:
    shared = "production restart checkpoint shared prefix " * 80
    request_tag = f"{index % 100:02d}"
    payload = {
        "prompt": shared + (f"independent request {request_tag} " * 90),
        "n_predict": 4,
        "temperature": 0,
        "cache_prompt": False,
    }
    request = urllib.request.Request(
        f"{BASE_URL}/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    if not result.get("content"):
        raise AssertionError("checkpoint smoke request returned empty content")


def start(log: object) -> subprocess.Popen[bytes]:
    executable = ROOT / "build/patched-cpu-noui/bin/Release/llama-server.exe"
    model = ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    command = [
        str(executable), "-m", str(model),
        "--host", "127.0.0.1", "--port", str(PORT),
        "-c", "4096", "-np", "4", "-b", "512", "-ub", "512",
        "-ngl", "0", "-t", "8", "--no-warmup", "--metrics",
        "--scheduler-policy", "cacheflow", "--benefit-policy", "learned",
        "--benefit-min-observations", "2", "--benefit-exploration-interval", "1",
        "--benefit-confidence-beta", "0.1", "--benefit-safety-margin-ms", "0.05",
        "--benefit-checkpoint", str(CHECKPOINT),
        "--benefit-checkpoint-key", "qwen2.5-0.5b-q4-cpu-production-smoke-v1",
        "--benefit-checkpoint-interval", "1",
        "--prefill-chunk-min", "16", "--prefill-chunk-max", "128",
    ]
    environment = os.environ.copy()
    environment["PATH"] = str(executable.parent) + os.pathsep + environment.get("PATH", "")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    wait_until_ready(BASE_URL, process=process, log_path=LOG)
    return process


def stop(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.unlink(missing_ok=True)
    Path(str(CHECKPOINT) + ".tmp").unlink(missing_ok=True)
    evidence: dict[str, float | bool | str] = {}

    with LOG.open("wb") as log:
        first = start(log)
        try:
            for wave in range(4):
                with ThreadPoolExecutor(max_workers=4) as pool:
                    list(pool.map(completion, range(wave * 4, wave * 4 + 4)))
                samples = prometheus()
                completed = metric(samples, 'benefit_checkpoint_save_total{result="completed"}')
                observations = (
                    metric(samples, 'benefit_observations_total{backend="cpu",action="upstream"}')
                    + metric(samples, 'benefit_observations_total{backend="cpu",action="cacheflow"}')
                )
                if completed >= 1 and observations >= 2:
                    break
            else:
                raise AssertionError("live server produced no durable benefit checkpoint")
            evidence["observations_before_restart"] = observations
            evidence["saves_before_restart"] = completed
        finally:
            stop(first)

        if not CHECKPOINT.exists() or CHECKPOINT.stat().st_size == 0:
            raise AssertionError("checkpoint was not committed before process termination")

        second = start(log)
        try:
            restored_samples = prometheus()
            restored = metric(restored_samples, 'benefit_checkpoint_restore_total{result="restored"}')
            restored_observations = (
                metric(restored_samples, 'benefit_observations_total{backend="cpu",action="upstream"}')
                + metric(restored_samples, 'benefit_observations_total{backend="cpu",action="cacheflow"}')
            )
            if restored != 1 or restored_observations < observations:
                raise AssertionError(
                    f"checkpoint did not restore: restored={restored}, observations={restored_observations}"
                )
            completion(1000)
            evidence["restored"] = True
            evidence["observations_after_restart"] = restored_observations
        finally:
            stop(second)

        CHECKPOINT.write_text('{"schema_version":1,"state":"truncated', encoding="utf-8")
        third = start(log)
        try:
            corrupt_samples = prometheus()
            failures = metric(corrupt_samples, 'benefit_checkpoint_restore_total{result="failed"}')
            cold_observations = (
                metric(corrupt_samples, 'benefit_observations_total{backend="cpu",action="upstream"}')
                + metric(corrupt_samples, 'benefit_observations_total{backend="cpu",action="cacheflow"}')
            )
            if failures != 1 or cold_observations != 0:
                raise AssertionError(
                    f"corrupt checkpoint did not fail closed: failures={failures}, observations={cold_observations}"
                )
            completion(2000)
            evidence["corruption_failed_closed"] = True
        finally:
            stop(third)

    CHECKPOINT.unlink(missing_ok=True)
    Path(str(CHECKPOINT) + ".tmp").unlink(missing_ok=True)
    evidence["log"] = str(LOG.relative_to(ROOT))
    output = ROOT / "results/benefit-checkpoint-smoke.json"
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
