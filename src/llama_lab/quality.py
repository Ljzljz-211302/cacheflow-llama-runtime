from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .server_bench import wait_until_ready
from .streaming import stream_chat


def score_response(
    text: str,
    required_patterns: list[str],
    forbidden_patterns: list[str] | None = None,
) -> bool:
    has_required = all(
        re.search(pattern, text, flags=re.IGNORECASE) is not None
        for pattern in required_patterns
    )
    has_forbidden = any(
        re.search(pattern, text, flags=re.IGNORECASE) is not None
        for pattern in (forbidden_patterns or [])
    )
    return has_required and not has_forbidden


def run_quality_evaluation(config_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.parent.parent
    server_exe = (root / config["server_exe"]).resolve()
    if not server_exe.exists():
        raise FileNotFoundError(f"llama-server not found: {server_exe}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for model_index, model in enumerate(config["models"]):
        model_path = (root / model["path"]).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")
        port = int(config["port"]) + model_index
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
            "1",
        ]
        log_path = raw_dir / f"quality-{model['key']}.log"
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                wait_until_ready(base_url, process=process, log_path=log_path)
                for task in config["tasks"]:
                    response = stream_chat(
                        base_url,
                        task["prompt"],
                        max_tokens=int(config["max_tokens"]),
                    )
                    rows.append(
                        {
                            "model_key": model["key"],
                            "quantization": model["quantization"],
                            "task": task["id"],
                            "passed": score_response(
                                response["text"],
                                task["required_patterns"],
                                task.get("forbidden_patterns"),
                            ),
                            "text": response["text"].replace("\n", " "),
                        }
                    )
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

    with (output_dir / "quality_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary: list[dict[str, Any]] = []
    for model in config["models"]:
        model_rows = [row for row in rows if row["model_key"] == model["key"]]
        passed = sum(bool(row["passed"]) for row in model_rows)
        summary.append(
            {
                "model_key": model["key"],
                "quantization": model["quantization"],
                "passed": passed,
                "total": len(model_rows),
                "accuracy": passed / len(model_rows),
            }
        )
    with (output_dir / "quality_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary
