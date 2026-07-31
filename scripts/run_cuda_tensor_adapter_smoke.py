from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server",
        type=Path,
        default=ROOT / "build/patched-cuda-ninja3/bin/llama-server.exe",
    )
    parser.add_argument("--port", type=int, default=8110)
    args = parser.parse_args()
    model = ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    log_path = ROOT / "results/raw/cuda-real-kv-adapter-smoke.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{args.port}"
    command = [
        str(args.server.resolve()), "-m", str(model),
        "--host", "127.0.0.1", "--port", str(args.port),
        "-c", "2048", "-np", "2", "-t", "8", "-ngl", "99",
        "--no-kv-unified", "--no-cache-idle-slots", "--no-warmup", "-lv", "4",
    ]
    environment = os.environ.copy()
    cuda_bin = ROOT / "runtime/cuda-dev/Library/bin"
    environment["PATH"] = str(cuda_bin) + os.pathsep + environment.get("PATH", "")
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
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError(f"CUDA server exited early; inspect {log_path}")
                try:
                    with urllib.request.urlopen(f"{url}/health", timeout=2):
                        break
                except Exception:
                    time.sleep(0.5)
            else:
                raise TimeoutError(f"CUDA server did not become ready; inspect {log_path}")

            payload = {
                "prompt": "Explain paged KV cache ownership in one sentence. " * 12,
                "n_predict": 8,
                "n": 2,
                "temperature": 0,
                "cache_prompt": True,
            }
            request = urllib.request.Request(
                f"{url}/completion",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.load(response)
            time.sleep(0.5)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    text = log_path.read_text(encoding="utf-8", errors="replace")
    evidence = [line for line in text.splitlines()
                if "CacheFlow CUDA adapter copied" in line]
    if not evidence:
        raise AssertionError(f"real llama KV tensor adapter did not run; inspect {log_path}")
    if not result:
        raise AssertionError("multi-completion response is empty")
    print(json.dumps({"evidence_log": evidence[-1], "log": str(log_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
