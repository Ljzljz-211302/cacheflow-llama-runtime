#!/usr/bin/env python3
"""Exact-output guardrail for the pinned upstream and CacheFlow upstream mode."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
PROMPTS = [
    "Implement binary search in C++ and state its complexity.",
    "Explain why transformer KV cache grows linearly with context.",
    "数据库事务的四个 ACID 属性分别是什么？",
    "Given x^2 - 5x + 6 = 0, list all roots.",
    "Name two causes of GPU kernel launch overhead.",
]


def post(port: int, prompt: str) -> dict[str, object]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps({
            "prompt": prompt, "n_predict": 32, "temperature": 0,
            "seed": 20260730, "cache_prompt": False,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def run(name: str, executable: Path, port: int, extra: list[str]) -> list[dict[str, object]]:
    log_path = ROOT / f"results/raw/upstream-compat-{name}.log"
    command = [
        str(executable), "-m", str(MODEL), "--host", "127.0.0.1", "--port", str(port),
        "-c", "1024", "-np", "2", "-ngl", "0", "-t", "8", "--no-warmup", *extra,
    ]
    env = os.environ.copy()
    env["PATH"] = str(executable.parent) + os.pathsep + env.get("PATH", "")
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError(f"{name} server exited early; inspect {log_path}")
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).close()
                    break
                except Exception:
                    time.sleep(0.5)
            return [post(port, prompt) for prompt in PROMPTS]
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main() -> None:
    upstream_executable = ROOT / "runtime/upstream-build/bin/Release/llama-server.exe"
    if not upstream_executable.exists():
        raise SystemExit("same-toolchain pinned upstream build is missing; build runtime/upstream-src first")
    upstream = run("pinned-msvc", upstream_executable, 8113, [])
    patched = run("patched", ROOT / "build/patched-cpu-noui/bin/Release/llama-server.exe",
            8114, ["--scheduler-policy", "upstream"])
    evidence = []
    for index, (left, right) in enumerate(zip(upstream, patched)):
        left_content = str(left.get("content", ""))
        right_content = str(right.get("content", ""))
        if left_content != right_content:
            raise AssertionError(f"output mismatch for prompt {index}: {left_content!r} != {right_content!r}")
        digest = hashlib.sha256(left_content.encode()).hexdigest()
        evidence.append({"prompt": index, "sha256": digest, "characters": len(left_content)})
    output = {
        "pinned_revision": "acd79d603cb2e1c84c0886137b80f1ad649b6857",
        "toolchain": "MSVC 19.37.32822 x64 for both baseline and variant",
        "mode": "--scheduler-policy upstream",
        "model": MODEL.name,
        "exact_matches": len(evidence),
        "cases": evidence,
    }
    path = ROOT / "results/upstream-compatibility.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
