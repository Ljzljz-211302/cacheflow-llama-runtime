from __future__ import annotations

import csv
import json
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .gpu_memory import GpuMemorySampler, query_gpu_used_mib
from .metrics import summarize_latency
from .streaming import stream_chat


def wait_until_ready(
    base_url: str,
    timeout: float = 90.0,
    process: subprocess.Popen[str] | None = None,
    log_path: Path | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            location = f"; inspect {log_path}" if log_path else ""
            raise RuntimeError(
                f"llama-server exited with code {process.returncode}{location}"
            )
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(0.5)
    raise TimeoutError(f"llama-server did not become healthy: {last_error}")


def _run_group(
    base_url: str,
    prompts: list[str],
    concurrency: int,
    repetitions: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    workload = [
        prompts[index % len(prompts)]
        for index in range(concurrency * repetitions)
    ]
    rows: list[dict[str, Any]] = []
    group_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(stream_chat, base_url, prompt, "local-model", max_tokens)
            for prompt in workload
        ]
        for future in as_completed(futures):
            rows.append(future.result())
    group_seconds = time.perf_counter() - group_started
    total_tokens = sum(int(row["completion_tokens"]) for row in rows)
    for row in rows:
        row["concurrency"] = concurrency
        row["group_output_tps"] = total_tokens / max(group_seconds, 1e-9)
    return rows


def run_server_benchmark(config_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.parent.parent
    server_exe = (root / config["server_exe"]).resolve()
    model_path = (root / config["model"]).resolve()
    if not server_exe.exists():
        raise FileNotFoundError(f"llama-server not found: {server_exe}")
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    port = int(config["port"])
    base_url = f"http://127.0.0.1:{port}"
    command = [
        str(server_exe),
        "-m",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        str(config["context"]),
        "-ngl",
        str(config["gpu_layers"]),
        "-np",
        str(max(config["concurrency"])),
        "--metrics",
    ]
    log_path = raw_dir / "server.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        gpu_before_server = query_gpu_used_mib()
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_until_ready(base_url, process=process, log_path=log_path)
            gpu_server_idle = query_gpu_used_mib()
            stream_chat(
                base_url,
                config["prompts"][0],
                max_tokens=min(8, int(config["max_tokens"])),
            )
            rows: list[dict[str, Any]] = []
            for concurrency in config["concurrency"]:
                with GpuMemorySampler() as memory:
                    group_rows = _run_group(
                        base_url,
                        config["prompts"],
                        int(concurrency),
                        int(config["repetitions"]),
                        int(config["max_tokens"]),
                    )
                for row in group_rows:
                    row["gpu_before_server_mib"] = gpu_before_server
                    row["gpu_server_idle_mib"] = gpu_server_idle
                    row["gpu_memory_peak_mib"] = memory.peak_mib
                    row["gpu_memory_increment_mib"] = (
                        max(0.0, memory.peak_mib - gpu_before_server)
                        if memory.peak_mib is not None and gpu_before_server is not None
                        else None
                    )
                rows.extend(group_rows)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    raw_path = raw_dir / "server_requests.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_rows: list[dict[str, Any]] = []
    for concurrency in config["concurrency"]:
        group = [row for row in rows if row["concurrency"] == concurrency]
        summary = summarize_latency(group)
        summary_rows.append(
            {
                "concurrency": concurrency,
                **summary,
                "aggregate_output_tps": group[0]["group_output_tps"],
                "gpu_server_idle_mib": group[0]["gpu_server_idle_mib"],
                "gpu_memory_peak_mib": group[0]["gpu_memory_peak_mib"],
                "gpu_memory_increment_mib": group[0]["gpu_memory_increment_mib"],
            }
        )
    with (output_dir / "server_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows
