from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def post(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main() -> None:
    server = ROOT / "build/patched-cuda-ninja3/bin/llama-server.exe"
    model = ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    log_path = ROOT / "results/raw/cuda-real-swap-server-smoke.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    port = 8111
    base = f"http://127.0.0.1:{port}"
    command = [
        str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
        "-c", "2048", "-np", "2", "-t", "8", "-ngl", "99", "--no-kv-unified",
        "--cache-ram", "128", "--cache-idle-slots", "--slot-prompt-similarity", "0.1",
        "--kv-block-runtime", "--kv-block-size", "16", "--metrics", "--no-warmup", "-lv", "4",
    ]
    environment = os.environ.copy()
    environment["PATH"] = str(ROOT / "runtime/cuda-dev/Library/bin") + os.pathsep + environment.get("PATH", "")
    prompt_a = "CUDA swap persistent prefix " * 80 + "request A"
    prompt_b = "independent request B " * 40
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError(f"server exited early; inspect {log_path}")
                try:
                    urllib.request.urlopen(f"{base}/health", timeout=2).close()
                    break
                except Exception:
                    time.sleep(0.5)
            first = post(f"{base}/completion", {
                "prompt": prompt_a, "id_slot": 0, "n_predict": 1, "temperature": 0, "cache_prompt": True,
            })
            post(f"{base}/completion", {
                "prompt": prompt_b, "id_slot": 1, "n_predict": 1, "temperature": 0, "cache_prompt": True,
            })
            restored = post(f"{base}/completion", {
                "prompt": prompt_a, "n_predict": 1, "temperature": 0, "cache_prompt": True,
            })
            with urllib.request.urlopen(f"{base}/metrics", timeout=30) as response:
                prometheus = response.read().decode()
            time.sleep(0.5)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    required = [
        "CacheFlow CUDA swapped out sequence 0",
        "CacheFlow CUDA restored sequence 0",
        "restored real CUDA KV stream for slot 0",
    ]
    missing = [item for item in required if item not in log_text]
    if missing:
        raise AssertionError(f"missing real server swap evidence {missing}; inspect {log_path}")
    metric_evidence = [
        "# TYPE llamacpp:requests_swapped_total counter",
        "llamacpp:requests_swapped_total 2",
        "llamacpp:requests_restored_total 1",
        "# TYPE llamacpp:adaptive_prefill_chunk gauge",
        "# TYPE llamacpp:kv_shared_blocks gauge",
        "# TYPE llamacpp:cuda_kv_copy_bytes_total counter",
        "# TYPE llamacpp:cuda_kv_swap_out_seconds counter",
        "# TYPE llamacpp:cuda_kv_swap_in_seconds counter",
        "# TYPE llamacpp:cuda_kv_events_waited_total counter",
        "# TYPE llamacpp:cuda_kv_pinned_pool_bytes gauge",
        "# TYPE llamacpp:cuda_kv_backend_errors_total counter",
    ]
    missing_metrics = [item for item in metric_evidence if item not in prometheus]
    if missing_metrics:
        raise AssertionError(f"missing Prometheus evidence: {missing_metrics}")
    metric_values: dict[str, float] = {}
    for line in prometheus.splitlines():
        if line.startswith("llamacpp:"):
            name, value = line.split(" ", 1)
            metric_values[name] = float(value)
    if metric_values.get("llamacpp:cuda_kv_copy_bytes_total", 0) <= 0:
        raise AssertionError("real CUDA KV byte counter did not advance")
    if metric_values.get("llamacpp:cuda_kv_events_waited_total", 0) < 2:
        raise AssertionError("real CUDA KV event counter did not advance")
    if metric_values.get("llamacpp:cuda_kv_swap_out_seconds", 0) <= 0 or \
            metric_values.get("llamacpp:cuda_kv_swap_in_seconds", 0) <= 0:
        raise AssertionError("real CUDA KV event timings did not advance")
    if metric_values.get("llamacpp:cuda_kv_backend_errors_total", -1) != 0:
        raise AssertionError("real CUDA KV backend reported an error")
    first_prompt = first.get("timings", {}).get("prompt_n", 0)  # type: ignore[union-attr]
    restored_prompt = restored.get("timings", {}).get("prompt_n", 0)  # type: ignore[union-attr]
    if not isinstance(restored_prompt, int) or restored_prompt >= first_prompt:
        raise AssertionError(f"restored request did not reduce prefill: {first_prompt} -> {restored_prompt}")
    print(json.dumps({
        "first_prompt_tokens": first_prompt,
        "restored_prompt_tokens": restored_prompt,
        "cuda_copy_bytes": metric_values["llamacpp:cuda_kv_copy_bytes_total"],
        "cuda_events_waited": metric_values["llamacpp:cuda_kv_events_waited_total"],
        "log": str(log_path),
    }))


if __name__ == "__main__":
    main()
