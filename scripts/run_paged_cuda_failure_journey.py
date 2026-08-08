from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from production_journey import ROOT, cuda_environment, get_text, metric, request_json, terminate_process, wait_ready


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server", type=Path,
        default=ROOT / "build/patched-cuda-ninja3/bin/llama-server.exe",
    )
    parser.add_argument(
        "--model", type=Path,
        default=ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    )
    parser.add_argument("--port", type=int, default=8132)
    args = parser.parse_args()

    raw = ROOT / "results/raw"
    raw.mkdir(parents=True, exist_ok=True)
    log_path = raw / "paged-cuda-failure-journey.log"
    base = f"http://127.0.0.1:{args.port}"
    environment = cuda_environment({
        "CACHEFLOW_TEST_POST_SUCCESS_FAILURES": "1",
        "CACHEFLOW_TEST_POST_SUCCESS_FAILURE_AFTER": "1",
        "CACHEFLOW_TEST_DECODE_FAILURE_CODE": "-3",
    })
    # The first successful decode warms the resident prefix. The next successful
    # (Paged) graph execution is reported as a late CUDA failure at the runtime
    # boundary, modeling asynchronous launch/execution failure detection.
    command = [
        str(args.server.resolve()), "-m", str(args.model.resolve()),
        "--host", "127.0.0.1", "--port", str(args.port),
        "-c", "1024", "-np", "1", "-t", "8", "-ngl", "99",
        "--metrics", "--slots", "--kv-unified", "--kv-block-runtime",
        "--kv-block-size", "16", "--kv-paged-decode",
        "--kv-action-policy", "analytical", "--kv-action-override", "paged",
        "--no-cache-idle-slots", "--cache-ram", "0", "--no-warmup",
    ]
    payload = {
        "prompt": "Briefly say hello.", "id_slot": 0,
        "n_predict": 1, "temperature": 0, "seed": 20260808,
        "cache_prompt": True,
    }

    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            wait_ready(base, process, log_path)
            warm_status, warm = request_json(f"{base}/completion", payload)
            failed_status, failed = request_json(f"{base}/completion", payload)
            recovered_status, recovered = request_json(f"{base}/completion", payload)
            metrics = get_text(f"{base}/metrics")
            if process.poll() is not None:
                raise AssertionError(f"server crashed after late CUDA failure; inspect {log_path}")
        finally:
            terminate_process(process)

    if warm_status != 200 or "error" in warm:
        raise AssertionError(f"warm request failed: {warm}")
    if failed_status < 400 or "error" not in failed:
        raise AssertionError(f"late CUDA failure was not surfaced: status={failed_status}, body={failed}")
    if recovered_status != 200 or "error" in recovered:
        raise AssertionError(f"request after CUDA failure did not recover: {recovered}")
    if recovered.get("content") != warm.get("content"):
        raise AssertionError(f"post-failure output diverged: warm={warm}, recovered={recovered}")
    # cache-ram is disabled, so a full prompt re-evaluation proves the failed
    # slot did not leave reusable partial sequence state behind.
    warm_prompt_n = warm.get("timings", {}).get("prompt_n")
    recovered_prompt_n = recovered.get("timings", {}).get("prompt_n")
    if recovered_prompt_n != warm_prompt_n:
        raise AssertionError(
            f"failed sequence state leaked into recovery: warm prompt_n={warm_prompt_n}, "
            f"recovered prompt_n={recovered_prompt_n}"
        )
    paged_calls = metric(metrics, "llamacpp:paged_decode_calls_total")
    paged_observations = metric(metrics, 'llamacpp:kv_action_observations_total{action="paged"}')
    paged_failures = metric(metrics, 'llamacpp:kv_action_failures_total{action="paged"}')
    if paged_calls < 1 or paged_observations < 1 or paged_failures != 1:
        raise AssertionError(
            "failure did not cross the Paged production boundary: "
            f"calls={paged_calls}, observations={paged_observations}, failures={paged_failures}"
        )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if "injected a late CUDA failure after successful llama_decode" not in log_text or "Compute error" not in log_text:
        raise AssertionError(f"compute failure was not logged; inspect {log_path}")
    print(json.dumps({
        "warm_status": warm_status,
        "failed_status": failed_status,
        "recovered_status": recovered_status,
        "output": recovered.get("content"),
        "full_prompt_recomputed": recovered_prompt_n,
        "paged_calls": paged_calls,
        "paged_observations": paged_observations,
        "paged_failures": paged_failures,
        "log": str(log_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
