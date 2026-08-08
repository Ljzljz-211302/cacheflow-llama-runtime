from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def cuda_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    cuda_bin = ROOT / "runtime/cuda-dev/Library/bin"
    environment["PATH"] = str(cuda_bin) + os.pathsep + environment.get("PATH", "")
    if extra:
        environment.update(extra)
    return environment


def request_json(
    url: str, payload: dict[str, Any], timeout: int = 180,
) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def get_text(url: str, timeout: int = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def metric(text: str, name: str) -> float:
    name = name.rstrip()
    match = re.search(rf"^{re.escape(name)}\s+([0-9.eE+-]+)$", text, re.MULTILINE)
    if not match:
        raise AssertionError(f"missing metric {name}")
    return float(match.group(1))


def wait_ready(
    base_url: str, process: subprocess.Popen[bytes], log_path: Path, attempts: int = 180,
) -> None:
    for _ in range(attempts):
        if process.poll() is not None:
            raise RuntimeError(f"server exited early; inspect {log_path}")
        try:
            get_text(f"{base_url}/health", timeout=2)
            return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"server did not become ready; inspect {log_path}")


def terminate_process(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
