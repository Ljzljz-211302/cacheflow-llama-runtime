#!/usr/bin/env python3
"""Link no-profiler service effects to a matching Nsight Systems CUDA trace."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llama_lab.cuda_profile_evidence import parse_nsys_sqlite  # noqa: E402
from llama_lab.research_protocol import file_sha256  # noqa: E402
from run_cuda_profile_experiment import discover_nsys, git_value, is_dirty  # noqa: E402


def run(command: list[str], stdout: Path, stderr: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    cuda_bin = ROOT / "runtime/cuda-dev/Library/bin"
    environment["PATH"] = str(cuda_bin) + os.pathsep + environment.get("PATH", "")
    completed = subprocess.run(
        command, cwd=ROOT, env=environment, text=True, capture_output=True, check=False
    )
    stdout.write_text(completed.stdout, encoding="utf-8", newline="")
    stderr.write_text(completed.stderr, encoding="utf-8", newline="")
    return completed


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if args.trials < 3:
        raise SystemExit("confirmatory service causal experiment requires at least 3 pairs")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("output directory must be absent or empty")
    nsys = discover_nsys()
    if nsys is None:
        raise SystemExit("Nsight Systems CLI not found; set NSYS_PATH")
    vendor = ROOT / "vendor/llama.cpp"
    dirty = is_dirty(ROOT)
    vendor_dirty = is_dirty(vendor)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = args.output_dir / "no-profiler"
    profiled = args.output_dir / "nsys-profiled"
    baseline.mkdir()
    profiled.mkdir()

    base_command = [
        sys.executable,
        str(ROOT / "scripts/run_cuda_causal_profile.py"),
        "--trials",
        str(args.trials),
        "--output-dir",
        str(baseline),
    ]
    base_run = run(
        base_command,
        args.output_dir / "no-profiler.stdout.txt",
        args.output_dir / "no-profiler.stderr.txt",
    )
    if base_run.returncode != 0:
        raise RuntimeError(f"no-profiler service run failed: {base_run.returncode}")

    trace_prefix = args.output_dir / "service-causal"
    profiled_target = [
        sys.executable,
        str(ROOT / "scripts/run_cuda_causal_profile.py"),
        "--trials",
        str(args.trials),
        "--output-dir",
        str(profiled),
    ]
    nsys_command = [
        str(nsys),
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--force-overwrite=true",
        f"--output={trace_prefix}",
        *profiled_target,
    ]
    profiled_run = run(
        nsys_command,
        args.output_dir / "nsys.stdout.txt",
        args.output_dir / "nsys.stderr.txt",
    )
    report_path = trace_prefix.with_suffix(".nsys-rep")
    if profiled_run.returncode != 0 or not report_path.exists():
        raise RuntimeError(f"NSYS service run failed: {profiled_run.returncode}")
    sqlite_path = trace_prefix.with_suffix(".sqlite")
    export_command = [
        str(nsys), "export", "--type=sqlite", "--force-overwrite=true",
        f"--output={sqlite_path}", str(report_path),
    ]
    exported = run(
        export_command,
        args.output_dir / "nsys-export.stdout.txt",
        args.output_dir / "nsys-export.stderr.txt",
    )
    if exported.returncode != 0 or not sqlite_path.exists():
        raise RuntimeError(f"NSYS service export failed: {exported.returncode}")

    baseline_summary = load_json(baseline / "cuda_causal_profile_summary.json")
    profiled_evidence = load_json(profiled / "cuda_causal_profile_evidence.json")
    with (profiled / "cuda_causal_profile_trials.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        profiled_rows = {
            (int(row["trial"]), row["mode"]): row for row in csv.DictReader(handle)
        }
    links: list[dict[str, Any]] = []
    total_profiled_kv_launches = 0
    for evidence in profiled_evidence["trials"]:
        trial = int(evidence["trial"])
        mode = str(evidence["mode"])
        pid = int(evidence["server_pid"])
        timeline = parse_nsys_sqlite(
            sqlite_path,
            kernel_patterns=("llama_kv_remap",),
            process_ids={pid},
        )
        expected = int(profiled_rows[(trial, mode)]["kernel_launches"])
        if int(timeline["kernel_launches"]) != expected:
            raise RuntimeError(
                f"trial={trial} mode={mode}: NSYS launches "
                f"{timeline['kernel_launches']} != runtime counter {expected}"
            )
        total_profiled_kv_launches += int(timeline["kernel_launches"])
        links.append(
            {
                "trial_id": evidence["trial_id"],
                "trial": trial,
                "mode": mode,
                "server_pid": pid,
                "request_ids": [request["request_id"] for request in evidence["requests"]],
                "request_ttft_ms": [request["ttft_ms"] for request in evidence["requests"]],
                "scheduler_action": {
                    "cacheflow_decisions": int(
                        profiled_rows[(trial, mode)]["cacheflow_decisions"]
                    ),
                    "prefill_chunks": int(profiled_rows[(trial, mode)]["prefill_chunks"]),
                    "prefill_tokens": int(profiled_rows[(trial, mode)]["prefill_tokens"]),
                },
                "kv_action": {
                    "kernel_launches": expected,
                    "copy_bytes": int(profiled_rows[(trial, mode)]["copy_bytes"]),
                    "cuda_event_ms": float(profiled_rows[(trial, mode)]["cuda_event_ms"]),
                },
                "nsys": timeline,
                "request_outcome": {
                    "ttft_p95_ms": float(profiled_rows[(trial, mode)]["ttft_p95_ms"]),
                    "execute_duration_us": float(
                        profiled_rows[(trial, mode)]["execute_duration_us"]
                    ),
                },
            }
        )
    if total_profiled_kv_launches <= 0:
        raise RuntimeError("service trace contains no custom KV kernel launches")
    if len(links) != args.trials * 2:
        raise RuntimeError("service trace linkage is incomplete")
    link_path = args.output_dir / "causal-links.json"
    link_path.write_text(
        json.dumps(
            {
                "linkage_key": "trial_id + server_pid + deterministic request_ids",
                "no_profiler_primary_result": baseline_summary["result"],
                "profiled_links": links,
                "measurement_boundary": (
                    "No-profiler paired result is primary; matching seeded NSYS executions "
                    "supply scheduler/action/CUDA/request mechanism links."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": "h2-service-nsight-causal",
        "code_revision": git_value(["rev-parse", "HEAD"]),
        "vendor_revision": git_value(["rev-parse", "HEAD"], directory=vendor),
        "repository_dirty_before_run": dirty,
        "vendor_dirty_before_run": vendor_dirty,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "trials": args.trials,
        "no_profiler_command": base_command,
        "nsys_command": nsys_command,
        "export_command": export_command,
        "artifacts": {
            "causal_links_sha256": file_sha256(link_path),
            "no_profiler_summary_sha256": file_sha256(
                baseline / "cuda_causal_profile_summary.json"
            ),
            "nsys_report_sha256": file_sha256(report_path),
            "nsys_sqlite_sha256": file_sha256(sqlite_path),
        },
        "linked_trial_modes": len(links),
        "linked_request_count": sum(len(link["request_ids"]) for link in links),
        "custom_kv_launches_in_nsys": total_profiled_kv_launches,
        "protocol_compliant": not dirty and not vendor_dirty and len(links) == args.trials * 2,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
