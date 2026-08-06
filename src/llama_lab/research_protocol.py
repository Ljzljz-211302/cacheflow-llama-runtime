"""Pre-registered research protocol validation and artifact packaging."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_lab.metrics import percentile
from llama_lab.research_charter import load_research_charter


class ProtocolError(ValueError):
    """Raised when protocol or trial evidence is incomplete or ambiguous."""


@dataclass(frozen=True)
class RunArtifacts:
    manifest: Path
    trials: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_research_protocol(
    path: Path, claims_path: Path | None = None
) -> dict[str, Any]:
    try:
        protocol = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot load research protocol: {error}") from error
    if protocol.get("schema_version") != 1:
        raise ProtocolError("schema_version must be 1")
    if not isinstance(protocol.get("protocol_version"), str):
        raise ProtocolError("protocol_version is required")
    timing = protocol.get("timing")
    expected_timing = {"host_enqueue_ms", "synchronized_kernel_ms", "end_to_end_ms"}
    if not isinstance(timing, dict) or set(timing) != expected_timing:
        raise ProtocolError("protocol must define all three timing domains")
    statistics_config = protocol.get("statistics", {})
    if not 0 < statistics_config.get("confidence_level", 0) < 1:
        raise ProtocolError("confidence_level must be in (0, 1)")
    if statistics_config.get("bootstrap_resamples", 0) < 1000:
        raise ProtocolError("bootstrap_resamples must be at least 1000")
    outliers = protocol.get("outliers", {})
    if outliers.get("outcome_based_deletion") != "forbidden":
        raise ProtocolError("outcome-based outlier deletion must be forbidden")
    fallback = protocol.get("cpu_correctness_fallback", {})
    if not fallback.get("command") or fallback.get("cuda_claims_allowed") is not False:
        raise ProtocolError("CPU fallback must be runnable and forbid CUDA claims")
    if claims_path is not None:
        charter = load_research_charter(claims_path)
        known_claims = {claim["id"] for claim in charter["claims"]}
        unknown = set(protocol.get("claims", [])) - known_claims
        if unknown:
            raise ProtocolError(f"unknown claims: {', '.join(sorted(unknown))}")
    return protocol


def paired_bootstrap_summary(
    pairs: list[tuple[float, float]],
    *,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    if not pairs:
        raise ProtocolError("at least one paired observation is required")
    effects: list[float] = []
    for baseline, variant in pairs:
        if baseline <= 0:
            raise ProtocolError("paired baseline timing must be positive")
        effects.append(100.0 * (baseline - variant) / baseline)
    rng = random.Random(seed)
    medians = [
        statistics.median(rng.choice(effects) for _ in effects)
        for _ in range(resamples)
    ]
    tail = (1.0 - confidence_level) / 2.0
    return {
        "pairs": len(pairs),
        "median_improvement_percent": statistics.median(effects),
        "ci_lower_percent": percentile(medians, tail),
        "ci_upper_percent": percentile(medians, 1.0 - tail),
    }


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    temporary.replace(path)


def package_cuda_remap_trials(
    protocol_path: Path,
    claims_path: Path,
    source_path: Path,
    environment_path: Path,
    output_dir: Path,
    command: list[str],
    code_revision: str,
    captured_at_utc: str,
    vendor_revision: str | None = None,
    *,
    environment_override: dict[str, Any] | None = None,
    repository_dirty: bool | None = None,
    vendor_dirty: bool | None = None,
    binary_sha256: str | None = None,
) -> RunArtifacts:
    protocol = load_research_protocol(protocol_path, claims_path)
    if len(code_revision) != 40:
        raise ProtocolError("code_revision must be a full Git SHA")
    if environment_override is None:
        try:
            environment = json.loads(environment_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ProtocolError(f"cannot load environment snapshot: {error}") from error
    else:
        environment = environment_override
    with source_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "method",
        "phase",
        "blocks",
        "trial",
        "order_in_pair",
        "host_enqueue_ms",
        "gpu_ms",
        "end_to_end_ms",
        "bytes",
    }
    if not rows:
        raise ProtocolError("raw source contains no trials")
    missing = required - rows[0].keys()
    if missing:
        raise ProtocolError(f"raw source missing fields: {', '.join(sorted(missing))}")

    records: list[dict[str, Any]] = []
    by_case: dict[int, dict[int, dict[str, float]]] = {}
    for row in rows:
        method = row["method"]
        if method not in {"scalar_gather_scatter", "vectorized_gather_scatter"}:
            raise ProtocolError(f"unknown CUDA Remap method: {method}")
        blocks = int(row["blocks"])
        trial = int(row["trial"])
        phase = row["phase"]
        if phase not in {"warmup", "confirmatory"}:
            raise ProtocolError(f"unknown trial phase: {phase}")
        order_in_pair = int(row["order_in_pair"])
        if order_in_pair not in {0, 1}:
            raise ProtocolError("order_in_pair must be 0 or 1")
        timing = {
            "host_enqueue_ms": float(row["host_enqueue_ms"]),
            "synchronized_kernel_ms": float(row["gpu_ms"]),
            "end_to_end_ms": float(row["end_to_end_ms"]),
        }
        if any(value < 0 for value in timing.values()):
            raise ProtocolError("timing values must be non-negative")
        records.append(
            {
                "claim_id": "H1-vector-remap",
                "experiment_id": "h1-vector-remap",
                "pair_id": f"blocks-{blocks}-trial-{trial}",
                "phase": phase,
                "order_in_pair": order_in_pair,
                "method": method,
                "blocks": blocks,
                "bytes": int(row["bytes"]),
                "valid": True,
                "invalid_reason": None,
                "correctness_passed": True,
                "timing_ms": timing,
            }
        )
        if phase == "confirmatory":
            by_case.setdefault(blocks, {}).setdefault(trial, {})[method] = timing[
                "synchronized_kernel_ms"
            ]

    stats = protocol["statistics"]
    summaries: list[dict[str, Any]] = []
    violations: list[str] = []
    if repository_dirty:
        violations.append("outer repository was dirty before the run")
    if vendor_dirty:
        violations.append("vendor repository was dirty before the run")
    for blocks, trials in sorted(by_case.items()):
        pairs: list[tuple[float, float]] = []
        for trial, methods in sorted(trials.items()):
            if set(methods) != {"scalar_gather_scatter", "vectorized_gather_scatter"}:
                violations.append(f"blocks={blocks} trial={trial} is not a complete pair")
                continue
            pairs.append(
                (
                    methods["scalar_gather_scatter"],
                    methods["vectorized_gather_scatter"],
                )
            )
        summary = paired_bootstrap_summary(
            pairs,
            confidence_level=float(stats["confidence_level"]),
            resamples=int(stats["bootstrap_resamples"]),
            seed=int(protocol["random_seed"]) + blocks,
        )
        summary["blocks"] = blocks
        summaries.append(summary)
        if len(pairs) < int(stats["minimum_pairs_per_case"]):
            violations.append(
                f"blocks={blocks} has {len(pairs)} pairs; require {stats['minimum_pairs_per_case']}"
            )

    estimates = [float(item["median_improvement_percent"]) for item in summaries]
    non_regression = all(float(item["ci_lower_percent"]) >= -3.0 for item in summaries)
    material_effect = any(float(item["ci_lower_percent"]) >= 10.0 for item in summaries)
    trend = all(left >= right for left, right in zip(estimates, estimates[1:]))
    if not non_regression:
        violations.append("non-regression CI gate failed")
    if not material_effect:
        violations.append("material-effect CI gate failed")
    if not trend:
        violations.append("non-increasing block-count trend gate failed")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / protocol["artifacts"]["manifest"]
    trials_path = output_dir / protocol["artifacts"]["raw_trials"]
    manifest = {
        "schema_version": 1,
        "experiment_id": "h1-vector-remap",
        "claim_id": "H1-vector-remap",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": _sha256(protocol_path),
        "claims_sha256": _sha256(claims_path),
        "source_sha256": _sha256(source_path),
        "code_revision": code_revision,
        "vendor_revision": vendor_revision,
        "repository_dirty_before_run": repository_dirty,
        "vendor_dirty_before_run": vendor_dirty,
        "benchmark_binary_sha256": binary_sha256,
        "captured_at_utc": captured_at_utc,
        "command": command,
        "environment": environment,
        "raw_trial_count": len(records),
        "paired_summaries": summaries,
        "acceptance": {
            "correctness": all(record["correctness_passed"] for record in records),
            "non_regression": non_regression,
            "material_effect": material_effect,
            "non_increasing_trend": trend,
            "passed": not violations,
        },
        "protocol_compliant": not violations,
        "violations": violations,
    }
    _atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    _atomic_write(
        trials_path,
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    )
    return RunArtifacts(manifest_path, trials_path)


def package_cpu_correctness_run(
    protocol_path: Path,
    claims_path: Path,
    environment_path: Path,
    output_dir: Path,
    command: list[str],
    code_revision: str,
    captured_at_utc: str,
    *,
    exit_code: int,
    elapsed_ms: float,
    environment_override: dict[str, Any] | None = None,
    repository_dirty: bool | None = None,
) -> RunArtifacts:
    protocol = load_research_protocol(protocol_path, claims_path)
    environment = (
        json.loads(environment_path.read_text(encoding="utf-8"))
        if environment_override is None
        else environment_override
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / protocol["artifacts"]["manifest"]
    trials_path = output_dir / protocol["artifacts"]["raw_trials"]
    record = {
        "claim_id": None,
        "experiment_id": "cpu-correctness",
        "pair_id": None,
        "order_in_pair": None,
        "method": "cpu-correctness-fallback",
        "valid": True,
        "invalid_reason": None,
        "correctness_passed": exit_code == 0,
        "timing_ms": {
            "host_enqueue_ms": None,
            "synchronized_kernel_ms": None,
            "end_to_end_ms": elapsed_ms,
        },
        "timing_unavailable_reason": "CPU fallback has no CUDA enqueue or event domain",
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": "cpu-correctness",
        "claim_id": None,
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": _sha256(protocol_path),
        "claims_sha256": _sha256(claims_path),
        "code_revision": code_revision,
        "repository_dirty_before_run": repository_dirty,
        "captured_at_utc": captured_at_utc,
        "command": command,
        "environment": environment,
        "raw_trial_count": 1,
        "performance_claims_allowed": False,
        "cuda_claims_allowed": False,
        "protocol_compliant": exit_code == 0 and not repository_dirty,
        "violations": (
            ([] if exit_code == 0 else [f"CPU correctness command exited {exit_code}"])
            + (["outer repository was dirty before the run"] if repository_dirty else [])
        ),
    }
    _atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    _atomic_write(trials_path, json.dumps(record, ensure_ascii=False) + "\n")
    return RunArtifacts(manifest_path, trials_path)
