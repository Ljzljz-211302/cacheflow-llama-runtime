from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction + 0.999999)))
    return ordered[index]


def completion(base: str, prompt: str, n_predict: int) -> tuple[float, dict[str, object]]:
    request = urllib.request.Request(
        f"{base}/completion",
        data=json.dumps({
            "prompt": prompt, "n_predict": n_predict, "temperature": 0, "cache_prompt": False,
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=240) as response:
        result = json.load(response)
    return (time.perf_counter() - started) * 1000, result


def run_case(backend: str, mode: str, trial: int) -> dict[str, object]:
    cuda = backend == "cuda"
    server = ROOT / (
        "build/patched-cuda-ninja3/bin/llama-server.exe" if cuda
        else "build/patched-cpu-noui/bin/Release/llama-server.exe"
    )
    port = 18200 + (100 if cuda else 0) + {"greedy": 0, "fixed64": 10, "fixed256": 20, "adaptive": 30}[mode] + trial
    base = f"http://127.0.0.1:{port}"
    log_path = ROOT / "results/raw" / f"adaptive-prefill-{backend}-{mode}-{trial}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(server), "-m", str(ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        "--host", "127.0.0.1", "--port", str(port), "-c", "4096", "-np", "4",
        "-t", "12", "-ngl", "99" if cuda else "0", "--metrics", "--cont-batching",
        "--no-cache-idle-slots", "--no-warmup",
    ]
    if mode == "fixed64":
        command += ["--prefill-chunk-size", "64"]
    elif mode == "fixed256":
        command += ["--prefill-chunk-size", "256"]
    elif mode == "adaptive":
        command += [
            "--prefill-chunk-adaptive", "--prefill-chunk-min", "16",
            "--prefill-chunk-max", "512", "--prefill-target-iteration-ms", "25" if cuda else "35",
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
            decode_prompt = "Explain continuous batching and KV ownership. " * 12
            prefill_prompts = [f"interfering prefill {index} " * 200 for index in range(3)]
            with ThreadPoolExecutor(max_workers=4) as pool:
                decode_future = pool.submit(completion, base, decode_prompt, 128)
                time.sleep(0.05)
                prefill_futures = [pool.submit(completion, base, prompt, 1) for prompt in prefill_prompts]
                decode_wall, decode = decode_future.result()
                prefill = [future.result() for future in prefill_futures]
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
    decode_timings = decode.get("timings", {})
    return {
        "backend": backend,
        "mode": mode,
        "trial": trial,
        "decode_wall_ms": decode_wall,
        "decode_model_ms": decode_timings.get("predicted_ms", 0),  # type: ignore[union-attr]
        "prefill_wall_p95_ms": percentile([item[0] for item in prefill], 0.95),
        "scheduler_iterations": sample("scheduler_iterations_total"),
        "prefill_chunks": sample("prefill_chunks_scheduled_total"),
        "effective_chunk": sample("adaptive_prefill_chunk"),
        "iteration_ewma_ms": sample("adaptive_prefill_iteration_milliseconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "cuda", "both"), default="both")
    args = parser.parse_args()
    backends = ["cpu", "cuda"] if args.backend == "both" else [args.backend]
    modes = ["greedy", "fixed64", "fixed256", "adaptive"]
    rows = [run_case(backend, mode, trial) for backend in backends for mode in modes for trial in range(1, 4)]
    raw_path = ROOT / "results/adaptive_prefill_trials.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries: list[dict[str, object]] = []
    fields = [field for field in rows[0] if field not in {"backend", "mode", "trial"}]
    for backend in backends:
        for mode in modes:
            group = [row for row in rows if row["backend"] == backend and row["mode"] == mode]
            summary: dict[str, object] = {"backend": backend, "mode": mode, "trials": len(group)}
            for field in fields:
                values = [float(row[field]) for row in group]
                summary[f"{field}_median"] = statistics.median(values)
                summary[f"{field}_p95"] = percentile(values, 0.95)
            summaries.append(summary)
    summary_path = ROOT / "results/adaptive_prefill_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
