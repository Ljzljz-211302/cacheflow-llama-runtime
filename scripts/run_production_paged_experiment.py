from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_production_paged_journey import metric, run_mode


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llama_lab.cuda_profile_evidence import parse_nsys_sqlite  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_median_interval(values: list[float], seed: int, resamples: int = 10000) -> list[float]:
    generator = random.Random(seed)
    medians = [
        statistics.median(generator.choices(values, k=len(values)))
        for _ in range(resamples)
    ]
    return [percentile(medians, 0.025), percentile(medians, 0.975)]


def git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    return completed.stdout.strip()


def device_identity() -> dict[str, str]:
    completed = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("production experiment requires exactly one visible GPU")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 3:
        raise RuntimeError("unexpected nvidia-smi result")
    return {"name": fields[0], "compute_capability": fields[1], "driver_version": fields[2]}


def discover_nsys() -> Path:
    candidates = sorted(
        (ROOT / "runtime").glob("nsight-systems-*/**/target-windows-x64/nsys.exe"),
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("Nsight Systems executable is unavailable")
    return candidates[0]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "minimum": min(values),
        "maximum": max(values),
    }


def action_reason_total(metrics: str, action: str) -> float:
    prefix = f'llamacpp:kv_action_decisions_by_reason_total{{action="{action}",reason='
    return sum(
        float(line.split()[-1])
        for line in metrics.splitlines()
        if line.startswith(prefix)
    )


def validate_artifact(protocol_path: Path, output: Path) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    rows = json.loads((output / "trials.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    expected_files = {
        "trials.json", "summary.json", "report.md", "mechanism.json",
        "profile/production-paged.nsys-rep", "profile/production-paged.sqlite",
        "profile/production-paged-server.log",
    } | {
        str(Path(row["log"]).as_posix()) for row in rows
    }
    if set(manifest["files"]) != expected_files:
        raise AssertionError("artifact manifest file set differs from the trial evidence tree")
    for relative, expected_hash in manifest["files"].items():
        if sha256(output / relative) != expected_hash:
            raise AssertionError(f"artifact hash mismatch: {relative}")
    if summary["protocol_sha256"] != sha256(protocol_path):
        raise AssertionError("artifact is not bound to the preregistered protocol")
    pairs = int(protocol["paired_trials"])
    keys = {(int(row["pair"]), str(row["action"])) for row in rows}
    if len(rows) != 2 * pairs or keys != {
        (pair, action) for pair in range(1, pairs + 1) for action in ("direct", "paged")
    }:
        raise AssertionError("artifact does not contain the exact preregistered pair/action matrix")
    by_pair = {pair: {} for pair in range(1, pairs + 1)}
    for row in rows:
        by_pair[int(row["pair"])][str(row["action"])] = row
        expected_order = ("direct", "paged") if int(row["pair"]) % 2 else ("paged", "direct")
        order_in_pair = int(row.get("order_in_pair", 0))
        if order_in_pair not in (1, 2) or row["action"] != expected_order[order_in_pair - 1]:
            raise AssertionError("artifact trial order differs from preregistered AB/BA order")
        expected_context = int(protocol["production_envelope"]["measured_context_tokens"])
        if int(row["cache_tokens"]) + int(row["prompt_tokens"]) != expected_context:
            raise AssertionError("artifact trial did not cross the registered page boundary")
        if row["action"] == "paged" and (row["paged_calls"] < 1 or row["paged_fallbacks"] != 0):
            raise AssertionError("artifact contains a Paged trial outside the production graph")
        if row["action_observations"] < 1:
            raise AssertionError("artifact contains an action without complete cost observation")
        if row["action_reason_decisions"] != row["action_decisions"]:
            raise AssertionError("artifact action decisions are not bound to their reasons")
    if any(pair["direct"]["content"] != pair["paged"]["content"] for pair in by_pair.values()):
        raise AssertionError("artifact contains a differential output mismatch")
    direct = [by_pair[pair]["direct"]["client_elapsed_ms"] for pair in sorted(by_pair)]
    paged = [by_pair[pair]["paged"]["client_elapsed_ms"] for pair in sorted(by_pair)]
    effects = [after - before for before, after in zip(direct, paged)]
    if summary["direct_client_elapsed_ms"] != summarize(direct):
        raise AssertionError("Direct summary is not derived from raw trials")
    if summary["paged_client_elapsed_ms"] != summarize(paged):
        raise AssertionError("Paged summary is not derived from raw trials")
    if summary["paired_paged_minus_direct_ms"] != summarize(effects):
        raise AssertionError("paired effect summary is not derived from raw trials")
    bootstrap_interval = bootstrap_median_interval(
        effects, int(protocol["random_seed"]), resamples=10000,
    )
    p95_regression = (
        summary["paged_client_elapsed_ms"]["p95"]
        / summary["direct_client_elapsed_ms"]["p95"] - 1.0
    ) * 100.0
    promotion_limit = float(
        protocol["acceptance"]["promotion_latency_p95_maximum_regression_percent"]
    )
    expected_conclusions = {
        "schema_version": 1,
        "protocol_version": protocol["protocol_version"],
        "paired_trials": pairs,
        "paired_median_bootstrap_95_ms": bootstrap_interval,
        "p95_regression_percent": p95_regression,
        "promotion_limit_percent": promotion_limit,
        "promotion_passed": p95_regression <= promotion_limit,
        "correctness_passed": all(
            pair["direct"]["content"] == pair["paged"]["content"]
            for pair in by_pair.values()
        ),
        "production_graph_entries": int(sum(
            row["paged_calls"] for row in rows if row["action"] == "paged"
        )),
        "paged_fallbacks": int(sum(
            row["paged_fallbacks"] for row in rows if row["action"] == "paged"
        )),
    }
    for field, expected in expected_conclusions.items():
        if summary.get(field) != expected:
            raise AssertionError(f"{field} is not derived from raw trials and protocol")
    if not summary.get("worktree_clean_before_run"):
        raise AssertionError("formal artifact was not collected from a clean worktree")
    expected_device = protocol["device"]
    observed_device = summary.get("device", {})
    if any(observed_device.get(field) != expected_device[field]
           for field in ("name", "compute_capability")):
        raise AssertionError("artifact device differs from the preregistered device")
    mechanism = json.loads((output / "mechanism.json").read_text(encoding="utf-8"))
    reparsed_mechanism = parse_nsys_sqlite(
        output / "profile/production-paged.sqlite",
        kernel_patterns=("cacheflow_paged_decode_fattn_k1",),
    )
    for field, expected in reparsed_mechanism.items():
        if mechanism.get(field) != expected:
            raise AssertionError(f"mechanism {field} is not derived from the NSYS SQLite")
    if mechanism["kernel_launches"] != 24 or not any(
        "cacheflow_paged_decode_fattn_k1" in name for name in mechanism["kernel_names"]
    ):
        raise AssertionError("profile does not contain one production Paged kernel per model layer")
    if summary.get("nsys_production_paged_kernel_launches") != mechanism["kernel_launches"]:
        raise AssertionError("summary kernel launch count differs from mechanism evidence")
    if summary.get("nsys_production_paged_kernel_duration_ms_non_primary") != mechanism["kernel_duration_ms"]:
        raise AssertionError("summary kernel duration differs from mechanism evidence")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path,
        default=ROOT / "config/production_paged_protocol_v1.1.json",
    )
    parser.add_argument(
        "--server", type=Path,
        default=ROOT / "build/patched-cuda-ninja3/bin/llama-server.exe",
    )
    parser.add_argument(
        "--model", type=Path,
        default=ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/research/h7-production-paged-v1.1.0",
    )
    parser.add_argument("--port-base", type=int, default=8140)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        validate_artifact(args.protocol, args.output)
        print(json.dumps({"validated": str(args.output)}, ensure_ascii=False))
        return

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    pairs = int(protocol["paired_trials"])
    seed = int(protocol["random_seed"])
    device = device_identity()
    if device["name"] != protocol["device"]["name"] or (
        device["compute_capability"] != protocol["device"]["compute_capability"]
    ):
        raise RuntimeError(f"device differs from preregistration: {device}")

    clean_before_run = not bool(git_output("status", "--porcelain"))
    args.output.mkdir(parents=True, exist_ok=True)
    raw_output = args.output / "raw"
    raw_output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for pair in range(1, pairs + 1):
        order = ["direct", "paged"] if pair % 2 else ["paged", "direct"]
        for order_in_pair, action in enumerate(order, 1):
            label = f"experiment-pair-{pair:02d}-{order_in_pair}"
            result, metrics, log_path = run_mode(
                args.server, args.model, args.port_base + (pair - 1) * 2 + order_in_pair - 1,
                action, label, prompt=str(protocol["request"]["prompt"]),
            )
            artifact_log = raw_output / f"{action}-pair-{pair:02d}-{order_in_pair}.log"
            shutil.copy2(log_path, artifact_log)
            row = {
                "pair": pair,
                "order_in_pair": order_in_pair,
                "action": action,
                "content": result["content"],
                "client_elapsed_ms": float(result["_client_elapsed_ms"]),
                "prompt_ms": float(result["timings"]["prompt_ms"]),
                "prompt_tokens": int(result["timings"]["prompt_n"]),
                "cache_tokens": int(result["timings"]["cache_n"]),
                "action_decisions": metric(
                    metrics, f'llamacpp:kv_action_decisions_total{{action="{action}"}}'
                ),
                "action_reason_decisions": action_reason_total(metrics, action),
                "action_observations": metric(
                    metrics, f'llamacpp:kv_action_observations_total{{action="{action}"}}'
                ),
                "action_cost_seconds": metric(
                    metrics, f'llamacpp:kv_action_observation_seconds_total{{action="{action}"}}'
                ),
                "paged_calls": metric(metrics, "llamacpp:paged_decode_calls_total "),
                "paged_fallbacks": metric(metrics, "llamacpp:paged_decode_fallbacks_total "),
                "log": str(artifact_log.relative_to(args.output).as_posix()),
                "log_sha256": sha256(artifact_log),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    by_pair: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(int(row["pair"]), {})[str(row["action"])] = row
    if any(set(pair_rows) != {"direct", "paged"} for pair_rows in by_pair.values()):
        raise AssertionError("incomplete action pair")
    if any(pair_rows["direct"]["content"] != pair_rows["paged"]["content"] for pair_rows in by_pair.values()):
        raise AssertionError("Paged output differs from its paired Direct output")
    if any(row["paged_calls"] < 1 or row["paged_fallbacks"] != 0 for row in rows if row["action"] == "paged"):
        raise AssertionError("a Paged trial did not enter the production graph exactly within its envelope")
    if any(row["action_observations"] < 1 for row in rows):
        raise AssertionError("an action trial has no complete cost observation")
    if any(row["action_reason_decisions"] != row["action_decisions"] for row in rows):
        raise AssertionError("an action trial has unattributed decision reasons")

    direct_ms = [by_pair[pair]["direct"]["client_elapsed_ms"] for pair in sorted(by_pair)]
    paged_ms = [by_pair[pair]["paged"]["client_elapsed_ms"] for pair in sorted(by_pair)]
    effects = [paged - direct for direct, paged in zip(direct_ms, paged_ms)]
    direct_summary = summarize(direct_ms)
    paged_summary = summarize(paged_ms)
    effect_summary = summarize(effects)
    p95_regression = (paged_summary["p95"] / direct_summary["p95"] - 1.0) * 100.0
    promotion_limit = float(protocol["acceptance"]["promotion_latency_p95_maximum_regression_percent"])

    nsys = discover_nsys()
    profile_dir = args.output / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    raw_prefix = ROOT / "results/raw/h7-production-paged-nsys"
    nsys_launcher = [
        str(nsys), "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
        "--duration=6", "--kill=true", "--force-overwrite=true", f"--output={raw_prefix}",
    ]
    profile_result, _, profile_log = run_mode(
        args.server, args.model, args.port_base + pairs * 2, "paged", "formal-nsys",
        launcher=nsys_launcher, launcher_manages_lifetime=True,
        prompt=str(protocol["request"]["prompt"]),
    )
    report_source = raw_prefix.with_suffix(".nsys-rep")
    if not report_source.exists():
        raise RuntimeError("Nsight Systems did not produce a report")
    report_path = profile_dir / "production-paged.nsys-rep"
    sqlite_path = profile_dir / "production-paged.sqlite"
    server_log_path = profile_dir / "production-paged-server.log"
    shutil.copy2(report_source, report_path)
    shutil.copy2(profile_log, server_log_path)
    exported = subprocess.run(
        [
            str(nsys), "export", "--type=sqlite", "--force-overwrite=true",
            f"--output={sqlite_path}", str(report_path),
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if exported.returncode or not sqlite_path.exists():
        raise RuntimeError(f"Nsight Systems export failed: {exported.stderr}")
    mechanism = parse_nsys_sqlite(
        sqlite_path, kernel_patterns=("cacheflow_paged_decode_fattn_k1",),
    )
    if mechanism["kernel_launches"] != 24:
        raise AssertionError(f"expected 24 production Paged layer launches, got {mechanism}")
    mechanism.update({
        "timing_role": "mechanism-only; no-profiler paired client latency remains primary",
        "request_content": profile_result["content"],
        "nsys_command": nsys_launcher,
    })
    (args.output / "mechanism.json").write_text(
        json.dumps(mechanism, indent=2) + "\n", encoding="utf-8",
    )

    summary = {
        "schema_version": 1,
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256(args.protocol),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_output("rev-parse", "HEAD"),
        "vendor_revision": git_output("-C", "vendor/llama.cpp", "rev-parse", "HEAD"),
        "worktree_clean_before_run": clean_before_run,
        "server_sha256": sha256(args.server),
        "model_sha256": sha256(args.model),
        "device": device,
        "paired_trials": pairs,
        "direct_client_elapsed_ms": direct_summary,
        "paged_client_elapsed_ms": paged_summary,
        "paired_paged_minus_direct_ms": effect_summary,
        "paired_median_bootstrap_95_ms": bootstrap_median_interval(effects, seed),
        "p95_regression_percent": p95_regression,
        "promotion_limit_percent": promotion_limit,
        "promotion_passed": p95_regression <= promotion_limit,
        "correctness_passed": True,
        "production_graph_entries": int(sum(row["paged_calls"] for row in rows if row["action"] == "paged")),
        "paged_fallbacks": int(sum(row["paged_fallbacks"] for row in rows if row["action"] == "paged")),
        "nsys_production_paged_kernel_launches": mechanism["kernel_launches"],
        "nsys_production_paged_kernel_duration_ms_non_primary": mechanism["kernel_duration_ms"],
    }
    (args.output / "trials.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    outcome = "passed" if summary["promotion_passed"] else "did not pass"
    report = f"""# Issue #7 production Paged service experiment

- Protocol: `{protocol['protocol_version']}` (`{summary['protocol_sha256']}`)
- Device: {device['name']} (compute capability {device['compute_capability']})
- Paired trials: {pairs}, alternating AB/BA, no outcome-based deletion
- Direct client latency: median {direct_summary['median']:.3f} ms, P95 {direct_summary['p95']:.3f} ms
- Paged client latency: median {paged_summary['median']:.3f} ms, P95 {paged_summary['p95']:.3f} ms
- Paired Paged - Direct: median {effect_summary['median']:+.3f} ms, P95 {effect_summary['p95']:+.3f} ms
- Paired median bootstrap 95% interval: [{summary['paired_median_bootstrap_95_ms'][0]:+.3f}, {summary['paired_median_bootstrap_95_ms'][1]:+.3f}] ms
- P95 regression: {p95_regression:+.2f}% (preregistered promotion limit: +{promotion_limit:.2f}%)
- Correctness: exact output match in every pair; {summary['production_graph_entries']} production Paged graph entries; {summary['paged_fallbacks']} fallbacks
- Mechanism replay: {mechanism['kernel_launches']} `cacheflow_paged_decode_fattn_k1<64>` launches (one per model layer); profiler timing is non-primary

The opt-in production promotion gate **{outcome}**. This result applies only to the preregistered Qwen2.5-0.5B, batch-1, short-context envelope. It is not a long-context, memory-bound, or universal speedup claim.
"""
    (args.output / "report.md").write_text(report, encoding="utf-8")
    evidence_files = [
        "trials.json", "summary.json", "report.md", "mechanism.json",
        "profile/production-paged.nsys-rep", "profile/production-paged.sqlite",
        "profile/production-paged-server.log",
    ] + [row["log"] for row in rows]
    manifest = {
        "schema_version": 1,
        "files": {relative: sha256(args.output / relative) for relative in sorted(evidence_files)},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    validate_artifact(args.protocol, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
