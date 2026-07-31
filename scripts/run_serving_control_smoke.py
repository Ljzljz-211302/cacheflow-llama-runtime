#!/usr/bin/env python3
"""Exercise disconnect cancellation, queue backpressure, and generation deadlines."""

from __future__ import annotations

import http.client
import json
import os
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def completion(base: str, prompt: str, n_predict: int, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "prompt": prompt, "n_predict": n_predict, "temperature": 0, "cache_prompt": False,
    }
    payload.update(extra)
    request = urllib.request.Request(f"{base}/completion", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def metric(base: str, name: str) -> float:
    with urllib.request.urlopen(f"{base}/metrics", timeout=30) as response:
        text = response.read().decode()
    prefix = f"llamacpp:{name} "
    return float(next(line for line in text.splitlines() if line.startswith(prefix)).split()[-1])


def disconnect_stream(port: int) -> None:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    payload = json.dumps({
        "prompt": "Produce an endless numbered technical list.\n", "n_predict": 1024,
        "temperature": 0, "stream": True, "cache_prompt": False,
    })
    connection.request("POST", "/completion", body=payload, headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    if response.status != 200:
        raise AssertionError(f"stream returned HTTP {response.status}")
    first = response.readline()
    if not first:
        raise AssertionError("stream produced no first event")
    connection.close()  # server_response_reader destructor must enqueue CANCEL


def main() -> None:
    port = 8115
    base = f"http://127.0.0.1:{port}"
    executable = ROOT / "build/patched-cpu-noui/bin/Release/llama-server.exe"
    log_path = ROOT / "results/raw/serving-control-smoke.log"
    command = [
        str(executable), "-m", str(ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        "--host", "127.0.0.1", "--port", str(port), "-c", "4096", "-np", "1",
        "-ngl", "0", "-t", "12", "--no-warmup", "--metrics", "--scheduler-policy", "cacheflow",
    ]
    environment = os.environ.copy()
    environment["PATH"] = str(executable.parent) + os.pathsep + environment.get("PATH", "")
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            for _ in range(180):
                if process.poll() is not None:
                    raise RuntimeError(f"server exited; inspect {log_path}")
                try:
                    urllib.request.urlopen(f"{base}/health", timeout=2).close()
                    break
                except Exception:
                    time.sleep(0.5)

            disconnect_stream(port)
            for _ in range(100):
                if metric(base, "requests_processing") == 0:
                    break
                time.sleep(0.05)
            else:
                raise AssertionError("disconnected request was not cancelled")
            probe = completion(base, "Reply with OK", 4)
            if not probe.get("content"):
                raise AssertionError("server did not recover after cancellation")

            prompts = [("queued long prefill " + str(index) + " ") * 220 for index in range(4)]
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(completion, base, prompt, 24) for prompt in prompts]
                deferred_peak = 0.0
                while any(not future.done() for future in futures):
                    deferred_peak = max(deferred_peak, metric(base, "requests_deferred"))
                    time.sleep(0.02)
                queued_results = [future.result() for future in futures]
            if deferred_peak < 1 or any(not result.get("content") for result in queued_results):
                raise AssertionError(f"backpressure evidence failed: deferred_peak={deferred_peak}")

            deadline = completion(base,
                    "Write one very short numbered line per item and continue for hundreds of items.\n",
                    256, t_max_predict_ms=10)
            predicted = int(deadline.get("timings", {}).get("predicted_n", 256))
            if predicted >= 256:
                raise AssertionError("generation deadline did not stop the request")
            final_probe = completion(base, "Reply healthy", 4)
            if not final_probe.get("content"):
                raise AssertionError("server did not recover after deadline")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if "cancel task" not in log_text:
        raise AssertionError("server log has no cancellation evidence")
    print(json.dumps({
        "disconnect_cancelled": True,
        "post_cancel_recovered": True,
        "deferred_peak": deferred_peak,
        "deadline_predicted_tokens": predicted,
        "post_deadline_recovered": True,
        "log": str(log_path),
    }))


if __name__ == "__main__":
    main()
