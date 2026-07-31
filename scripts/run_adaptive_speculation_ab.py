from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.999999))
    return ordered[index]


def run_case(backend: str, mode: str, trial: int) -> dict[str, object]:
    cuda = backend == "cuda"
    server = ROOT / (
        "build/patched-cuda-ninja3/bin/llama-server.exe" if cuda
        else "build/patched-cpu-noui/bin/Release/llama-server.exe"
    )
    port = 18400 + (100 if cuda else 0) + {"none": 0, "fixed": 10, "adaptive": 20}[mode] + trial
    base = f"http://127.0.0.1:{port}"
    log_path = ROOT / "results/raw" / f"adaptive-spec-{backend}-{mode}-{trial}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(server), "-m", str(ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        "--host", "127.0.0.1", "--port", str(port), "-c", "2048", "-np", "1",
        "-t", "12", "-ngl", "99" if cuda else "0", "--metrics",
        "--no-cache-idle-slots", "--no-warmup",
    ]
    if mode != "none":
        command += [
            "--spec-type", "ngram-simple",
            "--spec-ngram-simple-size-n", "3",
            "--spec-ngram-simple-size-m", "8",
            "--spec-ngram-simple-min-hits", "1",
        ]
    if mode == "adaptive":
        command += [
            "--speculative-adaptive", "--speculative-disable-acceptance", "0.25",
            "--speculative-kv-pressure", "0.80", "--speculative-cooldown", "6",
        ]
    environment = os.environ.copy()
    if cuda:
        environment["PATH"] = str(ROOT / "runtime/cuda-dev/Library/bin") + os.pathsep + environment.get("PATH", "")
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            for _ in range(180):
                if process.poll() is not None:
                    raise RuntimeError(f"server exited early; inspect {log_path}")
                try:
                    urllib.request.urlopen(f"{base}/health", timeout=2).close()
                    break
                except Exception:
                    time.sleep(0.5)
            prompt = (
                "Repeat the exact pattern without explanation:\n"
                + "red blue green yellow " * 180
                + "\nContinue the pattern: red blue"
            )
            request = urllib.request.Request(
                f"{base}/completion",
                data=json.dumps({
                    "prompt": prompt, "n_predict": 96, "temperature": 0, "seed": 1234,
                    "cache_prompt": False,
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.load(response)
            wall_ms = (time.perf_counter() - started) * 1000
            with urllib.request.urlopen(f"{base}/metrics", timeout=30) as response:
                metrics = response.read().decode()
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    def sample(name: str) -> float:
        prefix = f"llamacpp:{name} "
        line = next(item for item in metrics.splitlines() if item.startswith(prefix))
        return float(line.split()[-1])
    timings = result.get("timings", {})
    content = str(result.get("content", ""))
    return {
        "backend": backend,
        "mode": mode,
        "trial": trial,
        "wall_ms": wall_ms,
        "decode_ms": timings.get("predicted_ms", 0),  # type: ignore[union-attr]
        "decode_tokens": timings.get("predicted_n", 0),  # type: ignore[union-attr]
        "draft_tokens": sample("draft_tokens_total"),
        "accepted_tokens": sample("draft_tokens_accepted_total"),
        "acceptance_ratio": sample("draft_acceptance_ratio"),
        "adaptive_draft_length": sample("adaptive_draft_length"),
        "output_sha256": hashlib.sha256(content.encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "cuda", "both"), default="both")
    args = parser.parse_args()
    backends = ["cpu", "cuda"] if args.backend == "both" else [args.backend]
    modes = ["none", "fixed", "adaptive"]
    rows = [run_case(backend, mode, trial) for backend in backends for trial in range(1, 4) for mode in modes]
    for backend in backends:
        for trial in range(1, 4):
            hashes = {row["output_sha256"] for row in rows if row["backend"] == backend and row["trial"] == trial}
            if len(hashes) != 1:
                raise AssertionError(f"speculation changed greedy output for {backend} trial {trial}: {hashes}")
    raw_path = ROOT / "results/adaptive_speculation_trials.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries: list[dict[str, object]] = []
    numeric = [field for field in rows[0] if field not in {"backend", "mode", "trial", "output_sha256"}]
    for backend in backends:
        for mode in modes:
            group = [row for row in rows if row["backend"] == backend and row["mode"] == mode]
            summary: dict[str, object] = {"backend": backend, "mode": mode, "trials": len(group)}
            for field in numeric:
                values = [float(row[field]) for row in group]
                summary[f"{field}_median"] = statistics.median(values)
                summary[f"{field}_p95"] = percentile(values, 0.95)
            summaries.append(summary)
    summary_path = ROOT / "results/adaptive_speculation_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
