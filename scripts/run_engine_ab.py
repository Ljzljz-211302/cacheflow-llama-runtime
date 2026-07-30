from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llama_lab.prometheus import parse_prometheus_text, require_engine_metrics
from llama_lab.server_bench import wait_until_ready


def _post_json(url: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _completion(base_url: str, prompt: str, id_slot: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "n_predict": 1,
        "temperature": 0,
        "cache_prompt": True,
    }
    if id_slot is not None:
        payload["id_slot"] = id_slot
    return _post_json(f"{base_url}/completion", payload)


def _metrics(base_url: str) -> tuple[str, dict[str, float]]:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as response:
        text = response.read().decode("utf-8")
    samples = parse_prometheus_text(text)
    require_engine_metrics(samples)
    return text, samples


def run_case(
    config: dict[str, Any], root: Path, penalty: float, trial: int, output: Path
) -> dict[str, Any]:
    server_exe = (root / config["server_exe"]).resolve()
    model = (root / config["model"]).resolve()
    port = int(config["port"]) + round(penalty * 10)
    base_url = f"http://127.0.0.1:{port}"
    log_path = output / "raw" / f"engine-ab-{penalty:.2f}-trial-{trial}.log"
    command = [
        str(server_exe), "-m", str(model),
        "--host", "127.0.0.1", "--port", str(port),
        "-c", str(config["context"]), "-np", str(config["parallel"]),
        "-t", str(config["threads"]), "-ngl", "0", "--metrics", "--slots",
        "--slot-prompt-similarity", str(config["slot_prompt_similarity"]),
        "--slot-cache-eviction-penalty", str(penalty),
        "--no-warmup",
    ]

    common = " common" * 60
    long_slot_prompt = common + " alpha" * 20 + " long" * 400
    short_slot_prompt = common + " beta" * 10
    target_prompt = common + " alpha" * 20 + " target" * 5

    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_until_ready(base_url, process=process, log_path=log_path)
            _completion(base_url, long_slot_prompt, id_slot=0)
            _completion(base_url, short_slot_prompt, id_slot=1)
            started = time.perf_counter()
            response = _completion(base_url, target_prompt)
            target_wall_ms = (time.perf_counter() - started) * 1000
            followup_started = time.perf_counter()
            followup = _completion(base_url, long_slot_prompt + " followup" * 5)
            followup_wall_ms = (time.perf_counter() - followup_started) * 1000
            metrics_text, metrics = _metrics(base_url)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    (output / "raw" / f"engine-metrics-{penalty:.2f}-trial-{trial}.prom").write_text(
        metrics_text, encoding="utf-8"
    )
    timings = response.get("timings", {})
    followup_timings = followup.get("timings", {})
    target_processed = int(timings.get("prompt_n", 0))
    followup_processed = int(followup_timings.get("prompt_n", 0))
    return {
        "eviction_penalty": penalty,
        "trial": trial,
        "target_wall_ms": target_wall_ms,
        "target_prompt_processed_tokens": target_processed,
        "target_prompt_ms": timings.get("prompt_ms", 0),
        "followup_wall_ms": followup_wall_ms,
        "followup_prompt_processed_tokens": followup_processed,
        "followup_prompt_ms": followup_timings.get("prompt_ms", 0),
        "sequence_wall_ms": target_wall_ms + followup_wall_ms,
        "sequence_prompt_processed_tokens": target_processed + followup_processed,
        "cache_selections": metrics["llamacpp:slot_cache_selections_total"],
        "lru_selections": metrics["llamacpp:slot_lru_selections_total"],
        "selection_reused_tokens": metrics["llamacpp:slot_reused_tokens_total"],
        "selection_evicted_tokens": metrics["llamacpp:slot_evicted_tokens_total"],
        "prompt_tokens_cached_total": metrics["llamacpp:prompt_tokens_cached_total"],
        "kv_cache_tokens": metrics["llamacpp:kv_cache_tokens"],
        "kv_cache_capacity_tokens": metrics["llamacpp:kv_cache_capacity_tokens"],
        "memory_model_mib": metrics["llamacpp:memory_model_bytes"] / (1024 * 1024),
        "memory_context_mib": metrics["llamacpp:memory_context_bytes"] / (1024 * 1024),
        "memory_compute_mib": metrics["llamacpp:memory_compute_bytes"] / (1024 * 1024),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/engine_ab.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    config_path = args.config.resolve()
    root = config_path.parent.parent
    output = args.output.resolve()
    (output / "raw").mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    trial_rows = [
        run_case(config, root, float(penalty), trial, output)
        for penalty in config["penalties"]
        for trial in range(1, int(config["trials"]) + 1)
    ]
    with (output / "engine_ab_trials.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trial_rows[0]))
        writer.writeheader()
        writer.writerows(trial_rows)

    summary_rows: list[dict[str, Any]] = []
    metric_fields = [field for field in trial_rows[0] if field not in {"eviction_penalty", "trial"}]
    for penalty in config["penalties"]:
        group = [row for row in trial_rows if row["eviction_penalty"] == float(penalty)]
        summary_rows.append(
            {
                "eviction_penalty": penalty,
                "trials": len(group),
                **{
                    f"{field}_median": statistics.median(float(row[field]) for row in group)
                    for field in metric_fields
                },
            }
        )
    with (output / "engine_ab.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {len(trial_rows)} trials and {len(summary_rows)} scheduler summaries")


if __name__ == "__main__":
    main()
