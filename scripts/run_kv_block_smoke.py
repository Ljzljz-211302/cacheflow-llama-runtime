from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def wait_ready(url: str, process: subprocess.Popen[bytes], log_path: Path) -> None:
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError(f"server exited early; inspect {log_path}")
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2):
                return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"server did not become ready; inspect {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server",
        type=Path,
        default=ROOT / "build/patched-cpu-noui/bin/Release/llama-server.exe",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    )
    parser.add_argument("--port", type=int, default=8107)
    parser.add_argument("--mode", choices=("share", "preempt"), default="share")
    parser.add_argument("--gpu-layers", type=int, default=0)
    args = parser.parse_args()

    raw = ROOT / "results/raw"
    raw.mkdir(parents=True, exist_ok=True)
    log_path = raw / "kv-block-runtime-smoke.log"
    base_url = f"http://127.0.0.1:{args.port}"
    command = [
        str(args.server.resolve()),
        "-m", str(args.model.resolve()),
        "--host", "127.0.0.1", "--port", str(args.port),
        "-c", "2048", "-np", "2", "-t", "8", "-ngl", str(args.gpu_layers),
        "--metrics", "--slots", "--kv-unified",
        "--kv-block-runtime", "--kv-block-size", "16",
        "--no-warmup",
    ]
    if args.mode == "share":
        command.append("--no-cache-idle-slots")
    prefix = "shared system prefix " * 90
    requests = [
        {"prompt": prefix + " donor ending", "id_slot": 0},
        {"prompt": "unrelated cache " * 50, "id_slot": 1},
        {"prompt": prefix + " destination ending", "id_slot": 1},
    ]
    if args.mode == "preempt":
        requests[-1].pop("id_slot")
    for payload in requests:
        payload.update({"n_predict": 1, "temperature": 0, "cache_prompt": True})

    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            wait_ready(base_url, process, log_path)
            results = [post_json(f"{base_url}/completion", payload) for payload in requests]
            time.sleep(0.5)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    share_lines = [line for line in log_text.splitlines() if "prefix KV blocks" in line]
    restore_lines = [line for line in log_text.splitlines() if "found better prompt" in line]
    if args.mode == "share" and not share_lines:
        raise AssertionError(f"cross-slot block sharing was not observed; inspect {log_path}")
    if args.mode == "preempt" and not restore_lines:
        raise AssertionError(f"preempted prompt was not restored; inspect {log_path}")
    destination = results[-1]
    if "error" in destination:
        raise AssertionError(f"destination request failed: {destination['error']}")
    summary = {
        "donor_prompt_n": results[0].get("timings", {}).get("prompt_n"),
        "unrelated_prompt_n": results[1].get("timings", {}).get("prompt_n"),
        "destination_prompt_n": destination.get("timings", {}).get("prompt_n"),
        "destination_cache_n": destination.get("timings", {}).get("cache_n"),
        "mode": args.mode,
        "evidence_log": (share_lines if args.mode == "share" else restore_lines)[-1],
        "log": str(log_path),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
