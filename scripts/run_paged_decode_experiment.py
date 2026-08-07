#!/usr/bin/env python3
"""Run the preregistered restricted paged-decode experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llama_lab.cuda_profile_evidence import (  # noqa: E402
    build_ncu_command,
    build_nsys_command,
    parse_ncu_csv,
    parse_nsys_sqlite,
)
from llama_lab.paged_decode_evidence import (  # noqa: E402
    analyze_paged_decode_trials,
    load_paged_decode_protocol,
    validate_paged_decode_artifact,
)
from llama_lab.research_protocol import file_sha256  # noqa: E402


def run(command: list[str], *, environment: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=ROOT, env=environment, text=True, capture_output=True, check=False
    )


def git_output(directory: Path, *arguments: str) -> str:
    completed = run(["git", "-C", str(directory), *arguments])
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    return completed.stdout.strip()


def discover(name: str) -> Path | None:
    configured = os.environ.get(f"{name.upper()}_PATH")
    if configured and Path(configured).is_file():
        return Path(configured)
    found = shutil.which(name)
    if found:
        return Path(found)
    if name == "nsys":
        candidates = sorted(
            (ROOT / "runtime").glob("nsight-systems-*/**/target-windows-x64/nsys.exe"),
            reverse=True,
        )
    else:
        candidates = sorted(
            Path(os.environ.get("ProgramFiles", "C:/Program Files")).glob(
                "NVIDIA Corporation/Nsight Compute */**/ncu.exe"
            ),
            reverse=True,
        )
    return candidates[0] if candidates else None


def cuda_environment() -> dict[str, str]:
    environment = os.environ.copy()
    cuda_bin = ROOT / "runtime/cuda-dev/Library/bin"
    environment["PATH"] = str(cuda_bin) + os.pathsep + environment.get("PATH", "")
    return environment


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="")


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def target(binary: Path, regime: dict[str, Any], method: str, repetitions: int, profile: bool) -> list[str]:
    command = [
        str(binary), "--shape", str(regime["shape"]), "--context", str(regime["context"]),
        "--batch", str(regime["batch"]), "--layout", str(regime["layout"]),
        "--method", method, "--repetitions", str(repetitions),
    ]
    if profile:
        command.append("--profile")
    return command


def parse_trial(row: dict[str, str]) -> dict[str, Any]:
    integer_fields = ("context", "batch", "trial", "order_in_pair", "random_seed", "logical_kv_bytes")
    float_fields = ("host_enqueue_ms", "gpu_ms", "end_to_end_ms", "max_abs_error")
    parsed: dict[str, Any] = dict(row)
    for name in integer_fields:
        parsed[name] = int(parsed[name])
    for name in float_fields:
        parsed[name] = float(parsed[name])
    return parsed


def capture_nsys(
    nsys: Path, binary: Path, regime: dict[str, Any], method: str,
    repetitions: int, profile_dir: Path, patterns: list[str]
) -> dict[str, Any]:
    prefix = profile_dir / "nsys"
    command = build_nsys_command(
        nsys, prefix, target(binary, regime, method, repetitions, True)
    )
    completed = run(command, environment=cuda_environment())
    write_text(profile_dir / "nsys.stdout.txt", completed.stdout)
    write_text(profile_dir / "nsys.stderr.txt", completed.stderr)
    report = prefix.with_suffix(".nsys-rep")
    if completed.returncode or not report.is_file():
        raise RuntimeError(f"NSYS capture failed for {regime['id']} {method}")
    sqlite = prefix.with_suffix(".sqlite")
    export_command = [
        str(nsys), "export", "--type=sqlite", "--force-overwrite=true",
        f"--output={sqlite}", str(report),
    ]
    exported = run(export_command, environment=cuda_environment())
    write_text(profile_dir / "nsys-export.stdout.txt", exported.stdout)
    write_text(profile_dir / "nsys-export.stderr.txt", exported.stderr)
    if exported.returncode or not sqlite.is_file():
        raise RuntimeError(f"NSYS export failed for {regime['id']} {method}")
    parsed = parse_nsys_sqlite(sqlite, kernel_patterns=tuple(patterns))
    if parsed["kernel_launches"] != repetitions:
        raise RuntimeError(
            f"NSYS {regime['id']} {method} found {parsed['kernel_launches']} launches, expected {repetitions}"
        )
    return {
        "capture_command": command, "capture_exit_code": completed.returncode,
        "export_command": export_command, "export_exit_code": exported.returncode,
        "report": str(report.relative_to(profile_dir.parents[1])),
        "sqlite": str(sqlite.relative_to(profile_dir.parents[1])),
        "report_sha256": file_sha256(report), "sqlite_sha256": file_sha256(sqlite),
        "parsed": parsed,
    }


def capture_ncu(
    ncu: Path | None, binary: Path, regime: dict[str, Any], method: str,
    profile_dir: Path, patterns: list[str]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if ncu is None:
        path = profile_dir / "ncu-unavailable.txt"
        write_text(path, "Nsight Compute CLI was not found.\n")
        return None, {"available": False, "failure_log": str(path), "exit_code": None}
    prefix = profile_dir / "ncu"
    command = build_ncu_command(
        ncu, prefix, target(binary, regime, method, 1, True),
        kernel_name_regex=".*paged_decode_attention.*",
    )
    completed = run(command, environment=cuda_environment())
    write_text(profile_dir / "ncu.stdout.txt", completed.stdout)
    write_text(profile_dir / "ncu.stderr.txt", completed.stderr)
    csv_path = prefix.with_suffix(".csv")
    failure_text = completed.stdout + completed.stderr
    status = {
        "available": True, "command": command, "exit_code": completed.returncode,
        "permission_denied": "ERR_NVGPUCTRPERM" in failure_text,
        "driver_incompatible": "driver is not compatible" in failure_text.lower(),
        "csv_sha256": file_sha256(csv_path) if csv_path.is_file() else None,
    }
    if completed.returncode or not csv_path.is_file():
        return None, status
    try:
        return parse_ncu_csv(csv_path, kernel_patterns=tuple(patterns)), status
    except ValueError as error:
        status["parse_error"] = str(error)
        return None, status


def markdown(report: dict[str, Any], manifest_stub: dict[str, Any]) -> str:
    lines = [
        "# H3 restricted paged-decode report", "", "## Scope", "",
        "This is a decode-only CUDA prototype, not a production llama.cpp dispatch integration. "
        "The primary effects are 20-pair no-profiler CUDA-event measurements. NSYS replays only "
        "bind kernel identity and launch count; profiler durations are not primary timings.", "",
        "## No-profiler paired results", "",
        "| Regime | GPU improvement median [95% CI] | End-to-end improvement median [95% CI] | Class |",
        "|---|---:|---:|---|",
    ]
    for item in report["analysis"]["regimes"]:
        gpu, e2e = item["gpu_effect"], item["end_to_end_effect"]
        lines.append(
            f"| {item['regime_id']} | {gpu['median_improvement_percent']:+.2f}% "
            f"[{gpu['ci_lower_percent']:+.2f}%, {gpu['ci_upper_percent']:+.2f}%] | "
            f"{e2e['median_improvement_percent']:+.2f}% "
            f"[{e2e['ci_lower_percent']:+.2f}%, {e2e['ci_upper_percent']:+.2f}%] | "
            f"{item['effect_class']} |"
        )
    decision = report["analysis"]["next_kernel_decision"]
    lines += [
        "", "## Next kernel decision", "",
        f"The preregistered rule selected **{decision['selected']}**: {decision['reason']}. "
        "This is a mechanism hypothesis, not a demonstrated speedup; the selected variant must "
        "still be implemented and profiled.", "", "## Evidence boundaries", "",
        f"- NSYS coverage is complete for {len(manifest_stub['profiles'])} method-specific captures.",
        "- NCU hardware counters are incomplete, so memory-bound classification, occupancy "
        "explanation, and DRAM byte attribution are prohibited.",
        "- The D64 shape matches the local Qwen2.5-0.5B geometry. D128 matches the Qwen2.5-7B "
        "kernel geometry only and is not end-to-end 7B serving evidence.",
        "- Negative, neutral, and uncertain regimes are retained without outcome-based deletion.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "config/paged_decode_protocol.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("output directory must be absent or empty")
    vendor = ROOT / "vendor/llama.cpp"
    if git_output(ROOT, "status", "--porcelain") or git_output(vendor, "status", "--porcelain"):
        raise SystemExit("formal experiment requires clean outer and vendor worktrees")
    protocol = load_paged_decode_protocol(args.protocol)
    binary = ROOT / protocol["benchmark_binary"]
    if not binary.is_file():
        raise SystemExit(f"benchmark binary missing: {binary}")
    nsys = discover("nsys")
    if nsys is None:
        raise SystemExit("Nsight Systems CLI not found")
    ncu = discover("ncu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    profiles_root = args.output_dir / "profiles"
    raw_dir.mkdir(); profiles_root.mkdir()
    rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    repetitions = int(protocol["no_profiler"]["paired_trials_per_regime"])
    for regime in protocol["regimes"]:
        command = target(binary, regime, "paired", repetitions, False)
        completed = run(command, environment=cuda_environment())
        csv_path = raw_dir / f"{regime['id']}.csv"
        stderr_path = raw_dir / f"{regime['id']}.stderr.txt"
        write_text(csv_path, completed.stdout); write_text(stderr_path, completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"benchmark failed for {regime['id']}")
        parsed = [parse_trial(row) for row in csv.DictReader(completed.stdout.splitlines())]
        if len(parsed) != repetitions * 2:
            raise RuntimeError(f"wrong row count for {regime['id']}")
        rows.extend(parsed)
        commands.append({
            "regime_id": regime["id"], "command": command,
            "stdout_sha256": file_sha256(csv_path), "stderr_sha256": file_sha256(stderr_path),
        })
    trials_path = args.output_dir / "trials.jsonl"
    write_text(trials_path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    analysis = analyze_paged_decode_trials(rows, protocol)
    profiles: dict[str, Any] = {}
    ncu_statuses: dict[str, Any] = {}
    ncu_complete = True
    regime_by_id = {row["id"]: row for row in protocol["regimes"]}
    profile_repetitions = int(protocol["profiler"]["repetitions_per_method"])
    for regime_id in protocol["profiler"]["regime_ids"]:
        regime = regime_by_id[regime_id]
        for method in protocol["no_profiler"]["methods"]:
            profile_id = f"{regime_id}-{method}"
            profile_dir = profiles_root / profile_id
            profile_dir.mkdir()
            patterns = protocol["profiler"]["kernel_patterns"][method]
            profiles[profile_id] = capture_nsys(
                nsys, binary, regime, method, profile_repetitions, profile_dir, patterns
            )
            ncu_parsed, ncu_status = capture_ncu(
                ncu, binary, regime, method, profile_dir, patterns
            )
            ncu_statuses[profile_id] = {"parsed": ncu_parsed, "status": ncu_status}
            ncu_complete = ncu_complete and ncu_parsed is not None
    report = {
        "schema_version": 1, "analysis": analysis,
        "ncu_hardware_counters_complete": ncu_complete,
        "prohibited_claims": [] if ncu_complete else [
            "memory-bound classification", "occupancy explanation", "DRAM byte attribution"
        ],
        "claim_boundary": "prototype only; not production dispatch or end-to-end serving evidence",
    }
    report_path = args.output_dir / "report.json"
    write_json(report_path, report)
    manifest_stub = {"profiles": profiles}
    report_md = args.output_dir / "report.md"
    write_text(report_md, markdown(report, manifest_stub))
    artifact_hashes = {
        "trials.jsonl": file_sha256(trials_path),
        "report.json": file_sha256(report_path), "report.md": file_sha256(report_md),
    }
    for path in sorted(raw_dir.iterdir()):
        artifact_hashes[str(path.relative_to(args.output_dir)).replace("\\", "/")] = file_sha256(path)
    manifest = {
        "schema_version": 1, "experiment_id": "h3-paged-decode",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": file_sha256(args.protocol),
        "code_revision": git_output(ROOT, "rev-parse", "HEAD"),
        "vendor_revision": git_output(vendor, "rev-parse", "HEAD"),
        "repository_dirty_before_run": False, "vendor_dirty_before_run": False,
        "binary_sha256": file_sha256(binary),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "invocation": [sys.executable, *sys.argv], "raw_trial_count": len(rows),
        "commands": commands, "profiles": profiles, "ncu": ncu_statuses,
        "ncu_complete": ncu_complete, "artifact_hashes": artifact_hashes,
        "protocol_compliant": True,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    validate_paged_decode_artifact(args.output_dir, args.protocol)
    print(json.dumps({
        "artifact": str(args.output_dir), "raw_trials": len(rows),
        "next_kernel": analysis["next_kernel_decision"]["selected"],
        "ncu_complete": ncu_complete,
    }))


if __name__ == "__main__":
    main()
