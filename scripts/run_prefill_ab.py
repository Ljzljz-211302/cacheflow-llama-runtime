from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llama_lab.prometheus import parse_prometheus_text, require_engine_metrics
from llama_lab.server_bench import wait_until_ready


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def timed_completion(base_url: str, prompt: str, n_predict: int) -> tuple[float, dict]:
    started = time.perf_counter()
    result = post_json(
        f"{base_url}/completion",
        {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0,
            "cache_prompt": False,
        },
    )
    return (time.perf_counter() - started) * 1000, result


def run_case(config: dict, root: Path, chunk_size: int, trial: int, output: Path) -> dict:
    port = int(config["port"]) + chunk_size + trial
    base_url = f"http://127.0.0.1:{port}"
    log_path = output / "raw" / f"prefill-ab-{chunk_size}-trial-{trial}.log"
    command = [
        str((root / config["server_exe"]).resolve()),
        "-m", str((root / config["model"]).resolve()),
        "--host", "127.0.0.1", "--port", str(port),
        "-c", str(config["context"]), "-np", str(config["parallel"]),
        "-t", str(config["threads"]), "-ngl", "0", "--metrics",
        "--cont-batching", "--prefill-chunk-size", str(chunk_size), "--no-warmup",
    ]
    decode_prompt = "Explain why KV cache matters in transformer inference. " * 8
    # Keep the prompt below the per-slot context limit while still making
    # prefill long enough to interfere with an active decode stream.
    interfering_prompt = "long prefill token " * 450

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_until_ready(base_url, process=process, log_path=log_path)
            with ThreadPoolExecutor(max_workers=2) as pool:
                decode_future = pool.submit(
                    timed_completion,
                    base_url,
                    decode_prompt,
                    int(config["decode_tokens"]),
                )
                time.sleep(float(config["interference_delay_ms"]) / 1000)
                prefill_wall_ms, prefill_result = timed_completion(base_url, interfering_prompt, 1)
                decode_wall_ms, decode_result = decode_future.result()
            with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as response:
                metrics = parse_prometheus_text(response.read().decode())
            require_engine_metrics(metrics)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    decode_timings = decode_result.get("timings", {})
    prefill_timings = prefill_result.get("timings", {})
    return {
        "prefill_chunk_size": chunk_size,
        "trial": trial,
        "decode_wall_ms": decode_wall_ms,
        "decode_model_ms": decode_timings.get("predicted_ms", 0),
        "interfering_prefill_wall_ms": prefill_wall_ms,
        "interfering_prefill_model_ms": prefill_timings.get("prompt_ms", 0),
        "scheduler_iterations": metrics["llamacpp:scheduler_iterations_total"],
        "decode_tokens_scheduled": metrics["llamacpp:decode_tokens_scheduled_total"],
        "prefill_tokens_scheduled": metrics["llamacpp:prefill_tokens_scheduled_total"],
        "prefill_chunks_scheduled": metrics["llamacpp:prefill_chunks_scheduled_total"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/prefill_ab.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    config_path = args.config.resolve()
    root = config_path.parent.parent
    output = args.output.resolve()
    (output / "raw").mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = [
        run_case(config, root, int(chunk), trial, output)
        for chunk in config["chunk_sizes"]
        for trial in range(1, int(config["trials"]) + 1)
    ]
    with (output / "prefill_ab_trials.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for chunk in config["chunk_sizes"]:
        group = [row for row in rows if row["prefill_chunk_size"] == int(chunk)]
        summaries.append({
            "prefill_chunk_size": chunk,
            "trials": len(group),
            **{
                f"{field}_median": statistics.median(float(row[field]) for row in group)
                for field in rows[0]
                if field not in {"prefill_chunk_size", "trial"}
            },
        })
    with (output / "prefill_ab.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
