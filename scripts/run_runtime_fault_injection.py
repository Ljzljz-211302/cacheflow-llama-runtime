from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "build/patched-cuda-ninja3/bin/llama-server.exe"
MODEL = ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"


def post(url: str, prompt: str) -> tuple[int, dict[str, object]]:
    request = urllib.request.Request(
        url,
        data=json.dumps({"prompt": prompt, "n_predict": 2, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def run_case(code: int, port: int) -> dict[str, object]:
    label = "oom_retry" if code == 1 else "compute_rollback"
    log_path = ROOT / "results/raw" / f"runtime-fault-{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PATH"] = str(ROOT / "runtime/cuda-dev/Library/bin") + os.pathsep + environment.get("PATH", "")
    environment["CACHEFLOW_TEST_DECODE_FAILURES"] = "1"
    environment["CACHEFLOW_TEST_DECODE_FAILURE_CODE"] = str(code)
    base = f"http://127.0.0.1:{port}"
    command = [
        str(SERVER), "-m", str(MODEL), "--host", "127.0.0.1", "--port", str(port),
        "-c", "1024", "-np", "2", "-t", "8", "-ngl", "99", "--kv-unified",
        "--no-cache-idle-slots", "--no-warmup", "-lv", "4",
    ]
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError(f"fault server exited early; inspect {log_path}")
                try:
                    urllib.request.urlopen(f"{base}/health", timeout=2).close()
                    break
                except Exception:
                    time.sleep(0.5)
            first_status, first = post(f"{base}/completion", "fault injection first request " * 20)
            second_status, second = post(f"{base}/completion", "healthy request after injected failure")
            if process.poll() is not None:
                raise AssertionError(f"server crashed after fault; inspect {log_path}")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if code == 1:
        if first_status != 200 or "retrying with smaller batch size" not in text:
            raise AssertionError(f"OOM did not retry transactionally; inspect {log_path}")
    else:
        if first_status < 400 or "Compute error" not in text:
            raise AssertionError(f"compute failure did not roll back request; inspect {log_path}")
    if second_status != 200 or "error" in second:
        raise AssertionError(f"post-fault request failed: {second}")
    return {"case": label, "first_status": first_status, "second_status": second_status, "log": str(log_path)}


def main() -> None:
    print(json.dumps([run_case(1, 8113), run_case(-2, 8114)]))


if __name__ == "__main__":
    main()
