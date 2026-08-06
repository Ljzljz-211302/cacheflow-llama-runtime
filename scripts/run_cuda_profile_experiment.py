#!/usr/bin/env python3
"""Run the preregistered no-profiler + Nsight KV mechanism experiment."""

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
    characterize_regimes,
    parse_ncu_csv,
    parse_nsys_sqlite,
    require_complete_ncu_capture,
    require_complete_nsys_capture,
)
from llama_lab.research_protocol import file_sha256  # noqa: E402


def command_output(command: list[str]) -> str:
    completed = subprocess.run(
        command, cwd=ROOT, check=True, text=True, capture_output=True
    )
    return completed.stdout.strip()


def git_value(arguments: list[str], *, directory: Path = ROOT) -> str:
    return command_output(["git", "-C", str(directory), *arguments])


def is_dirty(directory: Path) -> bool:
    return bool(git_value(["status", "--porcelain"], directory=directory))


def discover_nsys() -> Path | None:
    configured = os.environ.get("NSYS_PATH")
    if configured and Path(configured).is_file():
        return Path(configured)
    found = shutil.which("nsys")
    if found:
        return Path(found)
    candidates = sorted(
        (ROOT / "runtime").glob("nsight-systems-*/**/target-windows-x64/nsys.exe"),
        reverse=True,
    )
    return candidates[0] if candidates else None


def discover_ncu() -> Path | None:
    configured = os.environ.get("NCU_PATH")
    if configured and Path(configured).is_file():
        return Path(configured)
    found = shutil.which("ncu")
    if found:
        return Path(found)
    roots = [Path(os.environ.get("ProgramFiles", "C:/Program Files"))]
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(root.glob("NVIDIA Corporation/Nsight Compute */**/ncu.exe"))
    return sorted(candidates, reverse=True)[0] if candidates else None


def cuda_environment() -> dict[str, str]:
    environment = os.environ.copy()
    cuda_bin = ROOT / "runtime/cuda-dev/Library/bin"
    environment["PATH"] = str(cuda_bin) + os.pathsep + environment.get("PATH", "")
    return environment


def run_logged(
    command: list[str], stdout_path: Path, stderr_path: Path
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=cuda_environment(),
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8", newline="")
    stderr_path.write_text(completed.stderr, encoding="utf-8", newline="")
    return completed


def no_profiler_records(
    binary: Path, regime: dict[str, Any], repetitions: int, raw_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    command = [
        str(binary),
        "--profile",
        "--blocks",
        str(regime["blocks"]),
        "--layout",
        str(regime["layout"]),
        "--method",
        "paired",
        "--repetitions",
        str(repetitions),
    ]
    output_path = raw_dir / f"{regime['id']}-no-profiler.csv"
    error_path = raw_dir / f"{regime['id']}-no-profiler.stderr.txt"
    completed = run_logged(command, output_path, error_path)
    if completed.returncode != 0:
        raise RuntimeError(
            f"no-profiler workload {regime['id']} failed with {completed.returncode}"
        )
    rows = list(csv.DictReader(completed.stdout.splitlines()))
    if len(rows) != repetitions * 2:
        raise RuntimeError(
            f"{regime['id']} produced {len(rows)} rows, expected {repetitions * 2}"
        )
    records: list[dict[str, Any]] = []
    pair_orders: dict[int, set[int]] = {}
    for row in rows:
        trial = int(row["trial"])
        pair_orders.setdefault(trial, set()).add(int(row["order_in_pair"]))
        records.append(
            {
                "regime_id": regime["id"],
                "layout": regime["layout"],
                "phase": "confirmatory",
                "pair_id": f"{regime['id']}-trial-{trial}",
                "trial": trial,
                "method": row["method"],
                "blocks": int(row["blocks"]),
                "bytes": int(row["bytes"]),
                "order_in_pair": int(row["order_in_pair"]),
                "random_seed": int(row["random_seed"]),
                "timing_ms": {
                    "host_enqueue_ms": float(row["host_enqueue_ms"]),
                    "synchronized_kernel_ms": float(row["gpu_ms"]),
                    "end_to_end_ms": float(row["end_to_end_ms"]),
                },
            }
        )
    if sorted(pair_orders) != list(range(repetitions)):
        raise RuntimeError(f"{regime['id']} trial IDs are incomplete")
    if any(order != {0, 1} for order in pair_orders.values()):
        raise RuntimeError(f"{regime['id']} treatment order is not paired")
    return records, {
        "command": command,
        "exit_code": completed.returncode,
        "stdout_sha256": file_sha256(output_path),
        "stderr_sha256": file_sha256(error_path),
    }


def nsys_capture(
    nsys: Path,
    binary: Path,
    regime: dict[str, Any],
    repetitions: int,
    profile_dir: Path,
    patterns: dict[str, list[str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    prefix = profile_dir / "nsys"
    target = [
        str(binary), "--profile", "--blocks", str(regime["blocks"]),
        "--layout", str(regime["layout"]), "--method", "paired",
        "--repetitions", str(repetitions),
    ]
    command = build_nsys_command(nsys, prefix, target)
    completed = run_logged(
        command, profile_dir / "nsys.stdout.txt", profile_dir / "nsys.stderr.txt"
    )
    report = prefix.with_suffix(".nsys-rep")
    if completed.returncode != 0 or not report.exists():
        raise RuntimeError(
            f"Nsight Systems capture {regime['id']} failed with {completed.returncode}"
        )
    database = prefix.with_suffix(".sqlite")
    export_command = [
        str(nsys), "export", "--type=sqlite", "--force-overwrite=true",
        f"--output={database}", str(report),
    ]
    exported = run_logged(
        export_command,
        profile_dir / "nsys-export.stdout.txt",
        profile_dir / "nsys-export.stderr.txt",
    )
    if exported.returncode != 0 or not database.exists():
        raise RuntimeError(
            f"Nsight Systems export {regime['id']} failed with {exported.returncode}"
        )
    parsed = {
        method: parse_nsys_sqlite(database, kernel_patterns=tuple(method_patterns))
        for method, method_patterns in patterns.items()
    }
    require_complete_nsys_capture(
        parsed, expected_launches_per_method=repetitions * 2
    )
    return parsed, {
        "capture_command": command,
        "capture_exit_code": completed.returncode,
        "export_command": export_command,
        "export_exit_code": exported.returncode,
        "report_sha256": file_sha256(report),
        "sqlite_sha256": file_sha256(database),
    }


def ncu_capture(
    ncu: Path,
    binary: Path,
    regime: dict[str, Any],
    repetitions: int,
    profile_dir: Path,
    patterns: dict[str, list[str]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    prefix = profile_dir / "ncu"
    target = [
        str(binary), "--profile", "--blocks", str(regime["blocks"]),
        "--layout", str(regime["layout"]), "--method", "paired",
        "--repetitions", str(repetitions),
    ]
    command = build_ncu_command(ncu, prefix, target)
    completed = run_logged(
        command, profile_dir / "ncu-target.stdout.txt", profile_dir / "ncu.stderr.txt"
    )
    csv_path = prefix.with_suffix(".csv")
    error_text = ""
    if csv_path.exists():
        error_text += csv_path.read_text(encoding="utf-8", errors="replace")
    error_text += completed.stderr
    status: dict[str, Any] = {
        "command": command,
        "exit_code": completed.returncode,
        "csv_sha256": file_sha256(csv_path) if csv_path.exists() else None,
        "permission_denied": "ERR_NVGPUCTRPERM" in error_text,
        "driver_incompatible": "driver is not compatible" in error_text.lower(),
    }
    if completed.returncode != 0 or not csv_path.exists():
        return None, status
    try:
        parsed = {
            method: parse_ncu_csv(csv_path, kernel_patterns=tuple(method_patterns))
            for method, method_patterns in patterns.items()
        }
        require_complete_ncu_capture(parsed)
    except ValueError as error:
        status["parse_error"] = str(error)
        return None, status
    status["parsed"] = True
    return parsed, status


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_report_markdown(
    report: dict[str, Any],
    protocol: dict[str, Any],
    commands: list[dict[str, Any]],
) -> str:
    layouts = {regime["id"]: regime["layout"] for regime in protocol["regimes"]}
    lines = [
        "# H2 KV movement profiling report",
        "",
        "## Result boundary",
        "",
        "No-profiler paired CUDA-event and end-to-end timings are the primary effects. "
        "Nsight Systems traces are separate mechanism observations and are never used as "
        "latency samples. Nsight Compute hardware-counter claims are allowed only when all "
        "configured metrics were captured.",
        "",
        "## Regimes and uncertainty",
        "",
        "| Regime | Layout | Blocks | GPU improvement, median [95% CI] | End-to-end improvement, median [95% CI] | Effect | NSYS scalar/vector kernel ms | Launches scalar/vector |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for item in report["regimes"]:
        gpu = item["gpu_effect"]
        end_to_end = item["end_to_end_effect"]
        scalar = item["nsys"]["scalar"]
        vector = item["nsys"]["vectorized"]
        lines.append(
            f"| {item['regime_id']} | {layouts[item['regime_id']]} | {item['blocks']} | "
            f"{gpu['median_improvement_percent']:+.2f}% "
            f"[{gpu['ci_lower_percent']:+.2f}%, {gpu['ci_upper_percent']:+.2f}%] | "
            f"{end_to_end['median_improvement_percent']:+.2f}% "
            f"[{end_to_end['ci_lower_percent']:+.2f}%, "
            f"{end_to_end['ci_upper_percent']:+.2f}%] | {item['effect_class']} | "
            f"{scalar['kernel_duration_ms']:.6f}/{vector['kernel_duration_ms']:.6f} | "
            f"{scalar['kernel_launches']}/{vector['kernel_launches']} |"
        )
    lines.extend(
        [
            "",
            "The 95% intervals are deterministic paired percentile-bootstrap intervals of "
            "the median, using the fixed sample count and seed in the protocol. No outcome-based "
            "outlier deletion or optional stopping is used.",
            "",
            "## Causal interpretation",
            "",
            "- Equal scalar/vector launch counts in each NSYS capture rule out a reduction in "
            "launch count as the cause of vectorization gains. Kernel duration changes remain "
            "visible in the same profiler range.",
            "- The misaligned regime is retained as a layout counterexample: the vector kernel "
            "executes its scalar fallback lanes with a grid sized for vector work. Its paired "
            "end-to-end effect is reported even when it loses.",
            "- Effective payload GB/s in `report.json` is logical bytes divided by CUDA-event "
            "time. It is not hardware DRAM throughput and cannot establish a roofline ceiling.",
            "",
            "## Nsight Compute boundary",
            "",
        ]
    )
    if report["ncu_hardware_counters_complete"]:
        lines.append(
            "All configured NCU metrics were captured; per-regime raw CSV and reports are under "
            "`profiles/`."
        )
    else:
        permission = any(item["ncu"].get("permission_denied") for item in commands)
        incompatible = any(item["ncu"].get("driver_incompatible") for item in commands)
        reasons = []
        if permission:
            reasons.append("ERR_NVGPUCTRPERM")
        if incompatible:
            reasons.append("installed profiler/driver compatibility check")
        lines.append(
            "NCU hardware counters are incomplete (observed: "
            + ", ".join(reasons or ["capture failure"])
            + "). Therefore this report prohibits memory-bound, roofline, achieved-occupancy, "
            "and hardware-DRAM-byte claims. The failed commands and logs are retained rather "
            "than replaced with estimates."
        )
    lines.extend(
        [
            "",
            "## Raw evidence map",
            "",
            "For every regime, `raw/<regime>-no-profiler.csv` contains 20 paired trials. "
            "`profiles/<regime>/nsys.nsys-rep` is the native report and `nsys.sqlite` is the "
            "parser input. `manifest.json` records commands, tool versions, revisions, dirty "
            "flags, binary/protocol hashes, and every trace hash.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "config/cuda_profile_protocol.json"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-ncu-unavailable",
        action="store_true",
        help="Retain NCU permission/compatibility failures and prohibit hardware-counter claims",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory must be absent or empty: {output_dir}")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or len(protocol.get("regimes", [])) < 3:
        raise SystemExit("invalid CUDA profiling protocol")
    binary = ROOT / protocol["benchmark_binary"]
    if not binary.exists():
        raise SystemExit(f"benchmark binary is missing: {binary}")
    nsys = discover_nsys()
    ncu = discover_ncu()
    if nsys is None:
        raise SystemExit("Nsight Systems CLI not found; set NSYS_PATH")
    if ncu is None and not args.allow_ncu_unavailable:
        raise SystemExit("Nsight Compute CLI not found; set NCU_PATH")

    dirty = is_dirty(ROOT)
    vendor = ROOT / "vendor/llama.cpp"
    vendor_dirty = is_dirty(vendor)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    profile_root = output_dir / "profiles"
    raw_dir.mkdir()
    profile_root.mkdir()

    repetitions = int(protocol["no_profiler"]["paired_trials_per_regime"])
    profile_repetitions = int(protocol["profiler"]["repetitions_per_regime"])
    patterns = protocol["profiler"]["kernel_patterns"]
    records: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    profiler: dict[str, dict[str, Any]] = {}
    ncu_complete = True
    for regime in protocol["regimes"]:
        regime_records, baseline_command = no_profiler_records(
            binary, regime, repetitions, raw_dir
        )
        records.extend(regime_records)
        profile_dir = profile_root / regime["id"]
        profile_dir.mkdir()
        nsys_parsed, nsys_status = nsys_capture(
            nsys, binary, regime, profile_repetitions, profile_dir, patterns
        )
        ncu_parsed = None
        ncu_status: dict[str, Any] = {"available": ncu is not None}
        if ncu is not None:
            ncu_parsed, ncu_status = ncu_capture(
                ncu, binary, regime, profile_repetitions, profile_dir, patterns
            )
        if ncu_parsed is None:
            ncu_complete = False
            if not args.allow_ncu_unavailable:
                raise RuntimeError(
                    f"NCU capture failed for {regime['id']}; rerun with "
                    "--allow-ncu-unavailable only to publish a limited negative result"
                )
        profiler[regime["id"]] = {"nsys": nsys_parsed, "ncu": ncu_parsed}
        commands.append(
            {
                "regime_id": regime["id"],
                "no_profiler": baseline_command,
                "nsys": nsys_status,
                "ncu": ncu_status,
            }
        )

    trials_path = output_dir / "trials.jsonl"
    trials_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
        newline="",
    )
    no_profiler = protocol["no_profiler"]
    report = characterize_regimes(
        records,
        protocol["regimes"],
        profiler,
        confidence_level=float(no_profiler["confidence_level"]),
        bootstrap_resamples=int(no_profiler["bootstrap_resamples"]),
        seed=int(protocol["random_seed_base"]),
        material_improvement_percent=float(
            no_profiler["material_improvement_percent"]
        ),
        maximum_regression_percent=float(
            no_profiler["maximum_regression_percent"]
        ),
    )
    report["ncu_hardware_counters_complete"] = ncu_complete
    report["limited_claims_only"] = not ncu_complete
    report["prohibited_claims"] = (
        []
        if ncu_complete
        else [
            "memory-bound or roofline classification",
            "achieved occupancy explanation",
            "hardware DRAM byte attribution",
        ]
    )
    if not report["contains_neutral_or_loss"]:
        raise RuntimeError("protocol requires at least one neutral or losing regime")
    report_path = output_dir / "report.json"
    write_json(report_path, report)
    report_markdown_path = output_dir / "report.md"
    report_markdown_path.write_text(
        render_report_markdown(report, protocol, commands), encoding="utf-8", newline=""
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": "h2-kv-profile",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": file_sha256(args.protocol),
        "code_revision": git_value(["rev-parse", "HEAD"]),
        "vendor_revision": git_value(["rev-parse", "HEAD"], directory=vendor),
        "repository_dirty_before_run": dirty,
        "vendor_dirty_before_run": vendor_dirty,
        "binary_sha256": file_sha256(binary),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "invocation": [sys.executable, *sys.argv],
        "tools": {
            "nsys": {
                "path": str(nsys),
                "version": command_output([str(nsys), "--version"]),
            },
            "ncu": {
                "path": str(ncu) if ncu else None,
                "version": command_output([str(ncu), "--version"]) if ncu else None,
            },
        },
        "raw_trial_count": len(records),
        "regime_count": len(protocol["regimes"]),
        "commands": commands,
        "artifact_hashes": {
            "trials": file_sha256(trials_path),
            "report": file_sha256(report_path),
            "report_markdown": file_sha256(report_markdown_path),
        },
        "nsys_complete": True,
        "ncu_complete": ncu_complete,
        "limited_claims_protocol_compliant": (
            not dirty
            and not vendor_dirty
            and report["contains_neutral_or_loss"]
            and report["all_claims_link_raw_to_end_to_end"]
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "manifest": str(output_dir / "manifest.json"),
                "regimes": len(protocol["regimes"]),
                "raw_trials": len(records),
                "nsys_complete": True,
                "ncu_complete": ncu_complete,
                "limited_claims": not ncu_complete,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
