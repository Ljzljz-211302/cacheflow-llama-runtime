#!/usr/bin/env python3
"""Real-model acceptance matrix required by docs/architecture.md section 13.4."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "Q4_K_M": ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "Q8_0": ROOT / "models/qwen2.5-0.5b-instruct-q8_0.gguf",
    "F16": ROOT / "models/qwen2.5-0.5b-instruct-fp16.gguf",
}


def request(base: str, prompt: str, n_predict: int = 4) -> dict[str, object]:
    body = json.dumps({
        "prompt": prompt, "n_predict": n_predict, "temperature": 0,
        "seed": 20260730, "cache_prompt": False,
    }).encode()
    started = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(
            f"{base}/completion", data=body,
            headers={"Content-Type": "application/json"}), timeout=300) as response:
        result = json.load(response)
    result["wall_ms"] = (time.perf_counter() - started) * 1000
    return result


def run_server(case: str, model: Path, backend: str, parallel: int, context: int,
        prompts: list[str]) -> list[dict[str, object]]:
    cuda = backend == "cuda"
    executable = ROOT / ("build/patched-cuda-ninja3/bin/llama-server.exe" if cuda
            else "build/patched-cpu-noui/bin/Release/llama-server.exe")
    port = 19000 + sum(ord(char) for char in case) % 2000
    base = f"http://127.0.0.1:{port}"
    log_path = ROOT / "results/raw" / f"matrix-{case}.log"
    command = [
        str(executable), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
        "-c", str(context * parallel), "-np", str(parallel), "-t", "12",
        "-ngl", "99" if cuda else "0", "--no-warmup", "--scheduler-policy", "cacheflow",
        "--no-cache-idle-slots", "--cont-batching",
    ]
    environment = os.environ.copy()
    environment["PATH"] = str(executable.parent) + os.pathsep + (
        str(ROOT / "runtime/cuda-dev/Library/bin") + os.pathsep if cuda else "") + environment.get("PATH", "")
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            for _ in range(240):
                if process.poll() is not None:
                    raise RuntimeError(f"server exited in {case}; inspect {log_path}")
                try:
                    urllib.request.urlopen(f"{base}/health", timeout=2).close()
                    break
                except Exception:
                    time.sleep(0.5)
            else:
                raise TimeoutError(f"server health timeout in {case}")
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                results = list(pool.map(lambda prompt: request(base, prompt), prompts))
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    if len(results) != len(prompts) or any(not str(result.get("content", "")) for result in results):
        raise AssertionError(f"missing output in {case}")
    return results


def main() -> None:
    rows: list[dict[str, object]] = []

    # Quantization x backend: every required weight format executes on both CPU and CUDA.
    for quantization, model in MODELS.items():
        for backend in ("cpu", "cuda"):
            case = f"model-{quantization.lower()}-{backend}"
            results = run_server(case, model, backend, 1, 512,
                    ["CacheFlow model/backend acceptance: explain paged KV briefly."])
            timing = results[0].get("timings", {})
            rows.append({
                "category": "model_backend", "case": case, "quantization": quantization,
                "backend": backend, "parallel": 1, "context": 512,
                "requests": 1, "prompt_tokens": timing.get("prompt_n", 0),
                "generated_tokens": timing.get("predicted_n", 0),
                "max_wall_ms": results[0]["wall_ms"],
            })

    # Continuous batching at every required concurrency level.
    for parallel in (1, 2, 4, 8):
        case = f"concurrency-{parallel}"
        prompts = [f"request {index}: define continuous batching in one sentence " * 4
                for index in range(parallel)]
        results = run_server(case, MODELS["Q4_K_M"], "cuda", parallel, 512, prompts)
        rows.append({
            "category": "concurrency", "case": case, "quantization": "Q4_K_M",
            "backend": "cuda", "parallel": parallel, "context": 512,
            "requests": len(results),
            "prompt_tokens": sum(int(result.get("timings", {}).get("prompt_n", 0)) for result in results),
            "generated_tokens": sum(int(result.get("timings", {}).get("predicted_n", 0)) for result in results),
            "max_wall_ms": max(float(result["wall_ms"]) for result in results),
        })

    # 128/512/2K/long context coverage. Repeated " x" is one token for Qwen BPE;
    # prompt_n from the real server is retained as evidence rather than assumed.
    for target in (128, 512, 2048, 4096):
        case = f"context-{target}"
        results = run_server(case, MODELS["Q4_K_M"], "cuda", 1, target + 128,
                ["KV context:" + " x" * target])
        timing = results[0].get("timings", {})
        prompt_tokens = int(timing.get("prompt_n", 0))
        if prompt_tokens < int(target * 0.9):
            raise AssertionError(f"{case} encoded only {prompt_tokens} prompt tokens")
        rows.append({
            "category": "context", "case": case, "quantization": "Q4_K_M",
            "backend": "cuda", "parallel": 1, "context": target + 128,
            "requests": 1, "prompt_tokens": prompt_tokens,
            "generated_tokens": timing.get("predicted_n", 0),
            "max_wall_ms": results[0]["wall_ms"],
        })

    output = ROOT / "results/model-acceptance-matrix.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    expected = 6 + 4 + 4
    if len(rows) != expected:
        raise AssertionError(f"matrix has {len(rows)} rows, expected {expected}")
    print(json.dumps({"cases": len(rows), "output": str(output), "status": "passed"}))


if __name__ == "__main__":
    main()
