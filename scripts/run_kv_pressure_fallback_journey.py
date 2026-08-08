from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from production_journey import ROOT, cuda_environment, get_text, metric, request_json, terminate_process, wait_ready


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server",
        type=Path,
        default=ROOT / "build/patched-cuda-ninja3/bin/llama-server.exe",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    )
    parser.add_argument("--port", type=int, default=8131)
    args = parser.parse_args()

    raw = ROOT / "results/raw"
    raw.mkdir(parents=True, exist_ok=True)
    log_path = raw / "kv-pressure-fallback-journey.log"
    base_url = f"http://127.0.0.1:{args.port}"
    environment = cuda_environment()
    command = [
        str(args.server.resolve()),
        "-m", str(args.model.resolve()),
        "--host", "127.0.0.1", "--port", str(args.port),
        "-c", "512", "-np", "2", "-t", "8", "-ngl", "99",
        "--metrics", "--slots", "--kv-unified", "--kv-block-runtime",
        "--kv-block-size", "16", "--no-cache-idle-slots", "--no-warmup",
    ]
    # Each resident prompt fits alone, while the third admission exceeds the
    # shared 512-token KV capacity and requires an idle-victim fallback.
    prompts = [
        "resident alpha cache " * 110,
        "resident beta cache " * 110,
        "pressure request gamma " * 110,
    ]
    requests = [
        {"prompt": prompts[0], "id_slot": 0},
        {"prompt": prompts[1], "id_slot": 1},
        {"prompt": prompts[2]},
    ]
    for request in requests:
        request.update({"n_predict": 1, "temperature": 0, "cache_prompt": True})

    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            wait_ready(base_url, process, log_path)
            responses = [request_json(f"{base_url}/completion", request)[1] for request in requests]
            metrics = get_text(f"{base_url}/metrics")
            if process.poll() is not None:
                raise AssertionError(f"server crashed under KV pressure; inspect {log_path}")
        finally:
            terminate_process(process)

    if any("error" in response for response in responses):
        raise AssertionError(f"a user request failed under pressure: {responses}")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    purge_lines = [line for line in log_text.splitlines() if "proactively purging KV slot" in line]
    if not purge_lines:
        raise AssertionError(f"capacity-pressure fallback did not execute; inspect {log_path}")

    pressure = metric(metrics, "llamacpp:kv_admission_pressure_total")
    evictions = metric(metrics, "llamacpp:kv_proactive_evictions_total")
    reclaimed = metric(metrics, "llamacpp:kv_proactive_reclaimed_tokens_total")
    failures = metric(metrics, "llamacpp:kv_admission_failures_total")
    if pressure < 1 or evictions < 1 or reclaimed < 1 or failures != 0:
        raise AssertionError(
            "invalid pressure fallback metrics: "
            f"pressure={pressure}, evictions={evictions}, reclaimed={reclaimed}, failures={failures}"
        )
    print(json.dumps({
        "requests_succeeded": len(responses),
        "admission_pressure": pressure,
        "proactive_evictions": evictions,
        "reclaimed_tokens": reclaimed,
        "admission_failures": failures,
        "evidence_log": purge_lines[-1],
        "log": str(log_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
