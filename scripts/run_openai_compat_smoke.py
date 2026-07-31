#!/usr/bin/env python3
"""Verify OpenAI chat-completions JSON and SSE behavior on the patched server."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    port = 8112
    base = f"http://127.0.0.1:{port}"
    server = ROOT / "build/patched-cuda-ninja3/bin/llama-server.exe"
    model = ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    log_path = ROOT / "results/raw/openai-compat-smoke.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = str(ROOT / "runtime/cuda-dev/Library/bin") + os.pathsep + env.get("PATH", "")
    command = [
        str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
        "-c", "1024", "-np", "2", "-ngl", "99", "--no-warmup", "--metrics",
        "--kv-block-runtime", "--prefill-chunk-adaptive",
    ]
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError(f"server exited early; inspect {log_path}")
                try:
                    urllib.request.urlopen(f"{base}/health", timeout=2).close()
                    break
                except Exception:
                    time.sleep(0.5)
            payload = {
                "model": "qwen2.5-0.5b-instruct",
                "messages": [{"role": "user", "content": "Reply with exactly: cacheflow-ready"}],
                "temperature": 0,
                "max_tokens": 16,
            }
            request = urllib.request.Request(f"{base}/v1/chat/completions",
                    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=120) as response:
                nonstream = json.load(response)
            if nonstream.get("object") != "chat.completion" or not nonstream.get("choices"):
                raise AssertionError(f"invalid non-streaming schema: {nonstream}")
            choice = nonstream["choices"][0]
            if choice.get("message", {}).get("role") != "assistant" or "finish_reason" not in choice:
                raise AssertionError(f"invalid chat choice: {choice}")

            payload["stream"] = True
            request = urllib.request.Request(f"{base}/v1/chat/completions",
                    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            events: list[dict[str, object]] = []
            done = False
            with urllib.request.urlopen(request, timeout=120) as response:
                if "text/event-stream" not in response.headers.get("Content-Type", ""):
                    raise AssertionError("streaming response is not SSE")
                for raw in response:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        done = True
                        break
                    events.append(json.loads(data))
            if not done or not events:
                raise AssertionError("SSE stream omitted chunks or [DONE]")
            if any(event.get("object") != "chat.completion.chunk" for event in events):
                raise AssertionError("SSE chunk schema mismatch")
            if not any(event.get("choices") for event in events):
                raise AssertionError("SSE stream has no choices")
            with urllib.request.urlopen(f"{base}/metrics", timeout=30) as response:
                prometheus = response.read().decode()
            for metric in (
                "time_to_first_token_seconds", "time_per_output_token_seconds",
                "request_latency_seconds", "request_queue_seconds",
            ):
                if f"# TYPE llamacpp:{metric} histogram" not in prometheus:
                    raise AssertionError(f"missing native latency histogram: {metric}")
                count_line = next((line for line in prometheus.splitlines()
                        if line.startswith(f"llamacpp:{metric}_count ")), "")
                if not count_line or float(count_line.split()[-1]) < 2:
                    raise AssertionError(f"latency histogram did not observe both requests: {metric}")
            required_metrics = {
                "counter": (
                    "scheduler_iterations_total", "kv_copy_on_write_total",
                    "kv_evicted_blocks_total", "kv_swap_bytes_total", "kv_restore_seconds",
                    "cuda_kv_kernel_launches_total", "cuda_kv_backend_errors_total",
                    "speculation_disabled_total",
                ),
                "gauge": (
                    "batch_tokens", "batch_sequences", "prefill_starvation_ms",
                    "kv_prefix_hit_ratio", "kv_blocks_used", "kv_blocks_free",
                    "kv_shared_blocks", "adaptive_draft_length",
                    "speculation_net_saved_ms",
                ),
            }
            for metric_type, names in required_metrics.items():
                for metric in names:
                    declaration = f"# TYPE llamacpp:{metric} {metric_type}"
                    if declaration not in prometheus:
                        raise AssertionError(f"missing native {metric_type}: {metric}")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    print(json.dumps({
        "nonstream_object": nonstream["object"],
        "stream_chunks": len(events),
        "stream_done": done,
        "native_latency_histograms": 4,
        "log": str(log_path),
    }))


if __name__ == "__main__":
    main()
