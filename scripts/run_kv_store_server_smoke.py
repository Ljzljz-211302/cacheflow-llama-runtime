from __future__ import annotations

import json
import subprocess
import tempfile
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


def run_mode(mode: str, port: int, swap_path: str, fault: str = "none") -> dict[str, object]:
    server = ROOT / "build/patched-cpu-noui/bin/Release/llama-server.exe"
    model = ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    suffix = "" if fault == "none" else f"-{fault}"
    log_path = ROOT / f"results/raw/{mode}-kv-store{suffix}-server-smoke.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{port}"
    command = [
        str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
        "-c", "2048", "-np", "2", "-t", "8", "--no-kv-unified",
        "--cache-ram", "128", "--cache-idle-slots", "--slot-prompt-similarity", "0.1",
        "--kv-block-runtime", "--kv-block-size", "16", "--kv-swap-path", swap_path,
        "--kv-swap-budget-mib", "128", "--metrics", "--no-warmup", "-lv", "4",
    ]
    if fault != "none":
        command.extend(["--kv-swap-fault", fault])
    prompt_a = "transactional swap persistent prefix " * 80 + "request A"
    prompt_b = "unrelated eviction request " * 45 + "request B"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
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
            else:
                raise TimeoutError(f"server did not become healthy; inspect {log_path}")

            first = post(f"{base}/completion", {
                "prompt": prompt_a, "id_slot": 0, "n_predict": 1,
                "temperature": 0, "cache_prompt": True,
            })
            post(f"{base}/completion", {
                "prompt": prompt_b, "id_slot": 1, "n_predict": 1,
                "temperature": 0, "cache_prompt": True,
            })
            restored = post(f"{base}/completion", {
                "prompt": prompt_a, "n_predict": 1,
                "temperature": 0, "cache_prompt": True,
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
        f"{mode} KV swap enabled",
    ]
    if fault == "none":
        required.extend([
            "stored KV state for slot 0",
            "restored KV state from transactional store for slot 0",
        ])
    elif fault == "next-save":
        required.extend([
            "injected host save failure",
            "keeping resident KV",
        ])
    elif fault == "next-restore":
        required.extend([
            "injected host restore failure",
            "request will recompute",
        ])
    missing = [item for item in required if item not in log_text]
    if missing:
        raise AssertionError(f"missing production store evidence {missing}; inspect {log_path}")
    expected_restores = 1 if fault == "none" else 0
    if f"llamacpp:requests_restored_total {expected_restores}" not in prometheus:
        raise AssertionError(f"unexpected restore counter for fault={fault}")
    expected_failures = 0 if fault == "none" else 1
    if f"llamacpp:kv_store_failures_total {expected_failures}" not in prometheus:
        raise AssertionError(f"unexpected store failure counter for fault={fault}")
    first_prompt = first.get("timings", {}).get("prompt_n", 0)  # type: ignore[union-attr]
    restored_prompt = restored.get("timings", {}).get("prompt_n", 0)  # type: ignore[union-attr]
    if not isinstance(restored_prompt, int):
        raise AssertionError(f"invalid prompt timing for fault={fault}")
    if fault == "next-restore":
        if restored_prompt != first_prompt:
            raise AssertionError(
                f"restore failure did not fall back to full recompute: {first_prompt} -> {restored_prompt}"
            )
    elif restored_prompt >= first_prompt:
        raise AssertionError(f"{mode} reuse did not reduce prefill: {first_prompt} -> {restored_prompt}")
    if first.get("content") != restored.get("content"):
        raise AssertionError(f"{mode} restore changed deterministic model output")
    return {
        "mode": mode,
        "fault": fault,
        "first_prompt_tokens": first_prompt,
        "restored_prompt_tokens": restored_prompt,
        "log": str(log_path),
    }


def main() -> None:
    results = [
        run_mode("host", 8113, "memory"),
        run_mode("host", 8115, "memory", "next-save"),
        run_mode("host", 8116, "memory", "next-restore"),
    ]
    with tempfile.TemporaryDirectory(prefix="cacheflow-kv-swap-") as directory:
        results.append(run_mode("file", 8114, directory))
    print(json.dumps(results))


if __name__ == "__main__":
    main()
