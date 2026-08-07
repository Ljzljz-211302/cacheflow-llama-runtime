"""Evidence analysis and fail-closed validation for restricted paged decode."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from llama_lab.cuda_profile_evidence import parse_nsys_sqlite
from llama_lab.research_protocol import file_sha256, paired_bootstrap_summary


def load_paged_decode_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1:
        raise ValueError("paged decode protocol schema_version must be 1")
    if protocol.get("contract", {}).get("page_size_tokens") != 16:
        raise ValueError("paged decode protocol must lock page size 16")
    shapes = protocol.get("shapes", {})
    expected = {
        "qwen2.5-0.5b": (14, 2, 64),
        "qwen2.5-7b-shape": (28, 4, 128),
    }
    for name, (query_heads, kv_heads, head_dim) in expected.items():
        shape = shapes.get(name, {})
        if (shape.get("query_heads"), shape.get("kv_heads"), shape.get("head_dim")) != (
            query_heads, kv_heads, head_dim
        ):
            raise ValueError(f"paged decode protocol has invalid {name} geometry")
        if shape.get("gqa_ratio") != 7:
            raise ValueError(f"paged decode protocol has invalid {name} GQA ratio")
    regimes = protocol.get("regimes", [])
    if len(regimes) != 9 or len({row.get("id") for row in regimes}) != len(regimes):
        raise ValueError("paged decode protocol must contain nine unique regimes")
    no_profiler = protocol.get("no_profiler", {})
    if no_profiler.get("methods") != ["contiguous", "paged"]:
        raise ValueError("paged decode protocol must compare contiguous and paged")
    if int(no_profiler.get("paired_trials_per_regime", 0)) < 20:
        raise ValueError("paged decode protocol requires at least 20 paired trials")
    return protocol


def _expected_seed(regime: dict[str, Any]) -> int:
    return (
        20260807
        + int(regime["context"]) * 17
        + int(regime["batch"]) * 101
        + (1009 if regime["layout"] == "fragmented" else 0)
        + (10007 if regime["shape"] == "qwen2.5-7b-shape" else 0)
    )


def _validate_and_group(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, dict[int, dict[str, dict[str, Any]]]]:
    regimes = {str(row["id"]): row for row in protocol["regimes"]}
    expected_trials = int(protocol["no_profiler"]["paired_trials_per_regime"])
    grouped: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    by_signature = {
        (row["shape"], int(row["context"]), int(row["batch"]), row["layout"]): row
        for row in protocol["regimes"]
    }
    for row in rows:
        signature = (
            str(row.get("shape")), int(row.get("context", -1)),
            int(row.get("batch", -1)), str(row.get("layout")),
        )
        regime = by_signature.get(signature)
        if regime is None:
            raise ValueError("paged decode trial shape/context/batch/layout differs from protocol")
        if row.get("phase") != "confirmatory":
            raise ValueError("paged decode trial phase must be confirmatory")
        method = str(row.get("method"))
        trial = int(row.get("trial", -1))
        if method not in {"contiguous", "paged"} or not 0 <= trial < expected_trials:
            raise ValueError("paged decode trial method or index is invalid")
        if method in grouped[str(regime["id"])][trial]:
            raise ValueError("paged decode trial contains a duplicate method")
        if int(row.get("random_seed", -1)) != _expected_seed(regime):
            raise ValueError("paged decode random seed differs from protocol")
        if float(row.get("gpu_ms", 0)) <= 0 or float(row.get("end_to_end_ms", 0)) <= 0:
            raise ValueError("paged decode timing must be positive")
        if float(row.get("max_abs_error", 1)) > 1e-5:
            raise ValueError("paged decode CUDA differential error exceeds benchmark gate")
        grouped[str(regime["id"])][trial][method] = row
    if set(grouped) != set(regimes):
        raise ValueError("paged decode regime coverage is incomplete")
    for regime_id, trials in grouped.items():
        if set(trials) != set(range(expected_trials)):
            raise ValueError(f"{regime_id} pair count differs from protocol")
        for trial, pair in trials.items():
            if set(pair) != {"contiguous", "paged"}:
                raise ValueError(f"{regime_id} trial {trial} pair is incomplete")
            if {int(item["order_in_pair"]) for item in pair.values()} != {0, 1}:
                raise ValueError(f"{regime_id} trial {trial} order is not complementary")
            fixed = ("shape", "context", "batch", "layout", "random_seed", "logical_kv_bytes")
            if any(pair["contiguous"][name] != pair["paged"][name] for name in fixed):
                raise ValueError(f"{regime_id} trial {trial} pair inputs differ")
    return grouped


def _effect_class(summary: dict[str, Any], material: float, regression: float) -> str:
    lower = float(summary["ci_lower_percent"])
    upper = float(summary["ci_upper_percent"])
    if lower >= material:
        return "material-win"
    if upper < -regression:
        return "material-loss"
    if lower >= -regression and upper < material:
        return "neutral"
    return "uncertain"


def analyze_paged_decode_trials(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    grouped = _validate_and_group(rows, protocol)
    config = protocol["no_profiler"]
    results: list[dict[str, Any]] = []
    for index, regime in enumerate(protocol["regimes"]):
        regime_id = str(regime["id"])
        trials = grouped[regime_id]
        gpu_pairs = [
            (float(trials[i]["contiguous"]["gpu_ms"]), float(trials[i]["paged"]["gpu_ms"]))
            for i in sorted(trials)
        ]
        e2e_pairs = [
            (float(trials[i]["contiguous"]["end_to_end_ms"]),
             float(trials[i]["paged"]["end_to_end_ms"]))
            for i in sorted(trials)
        ]
        gpu = paired_bootstrap_summary(
            gpu_pairs, confidence_level=float(config["confidence_level"]),
            resamples=int(config["bootstrap_resamples"]),
            seed=int(protocol["random_seed_base"]) + index * 2,
        )
        e2e = paired_bootstrap_summary(
            e2e_pairs, confidence_level=float(config["confidence_level"]),
            resamples=int(config["bootstrap_resamples"]),
            seed=int(protocol["random_seed_base"]) + index * 2 + 1,
        )
        results.append({
            "regime_id": regime_id,
            "shape": regime["shape"], "context": regime["context"],
            "batch": regime["batch"], "layout": regime["layout"],
            "raw_pair_ids": [f"{regime_id}-trial-{i}" for i in sorted(trials)],
            "gpu_effect": gpu, "end_to_end_effect": e2e,
            "median_no_profiler_gpu_ms": {
                "contiguous": statistics.median(pair[0] for pair in gpu_pairs),
                "paged": statistics.median(pair[1] for pair in gpu_pairs),
            },
            "effect_class": _effect_class(
                gpu, float(config["material_improvement_percent"]),
                float(config["maximum_regression_percent"]),
            ),
        })
    result_by_id = {row["regime_id"]: row for row in results}
    split_evidence = []
    for prefix in ("q05", "q7"):
        b1 = float(result_by_id[f"{prefix}-long-b1-fragmented"]["gpu_effect"]["median_improvement_percent"])
        b4 = float(result_by_id[f"{prefix}-long-b4-fragmented"]["gpu_effect"]["median_improvement_percent"])
        split_evidence.append((-b1) - (-b4))
    long_regressions = [
        -float(row["gpu_effect"]["median_improvement_percent"])
        for row in results if "long-" in row["regime_id"]
    ]
    if max(split_evidence) >= 5.0:
        selected = "K3-split-KV"
        reason = "batch-1 long-context regression exceeds batch-4 by at least 5 percentage points"
    elif max(long_regressions) >= 3.0:
        selected = "K2-GQA-reuse"
        reason = "a long-context ratio-7 regime regresses by at least 3 percent"
    else:
        selected = "retain-K1"
        reason = "neither preregistered K2 nor K3 trigger fired"
    return {
        "schema_version": 1,
        "regimes": results,
        "contains_neutral_or_loss": any(
            row["effect_class"] in {"neutral", "material-loss"} for row in results
        ),
        "next_kernel_decision": {
            "selected": selected, "reason": reason,
            "split_k_batch_gap_percentage_points": split_evidence,
            "maximum_long_context_regression_percent": max(long_regressions),
            "status": "hypothesis pending implementation and profiling",
        },
        "measurement_warning": "No-profiler CUDA events determine effects; profiler replay only diagnoses launches.",
    }


def validate_paged_decode_artifact(
    artifact_dir: Path, protocol_path: Path
) -> dict[str, Any]:
    protocol = load_paged_decode_protocol(protocol_path)
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
    trials_path = artifact_dir / "trials.jsonl"
    rows = [json.loads(line) for line in trials_path.read_text(encoding="utf-8").splitlines() if line]
    if manifest.get("protocol_sha256") != file_sha256(protocol_path):
        raise ValueError("paged decode protocol hash differs")
    expected_tree = {
        path.relative_to(artifact_dir).as_posix()
        for path in artifact_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(manifest.get("artifact_hashes", {})) != expected_tree:
        raise ValueError("paged decode artifact file tree differs from manifest")
    for relative, expected_hash in manifest.get("artifact_hashes", {}).items():
        path = artifact_dir / relative
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise ValueError(f"paged decode artifact hash differs: {relative}")
    recomputed = analyze_paged_decode_trials(rows, protocol)
    if report.get("analysis") != recomputed:
        raise ValueError("paged decode report analysis differs from raw trials")
    expected_rows = len(protocol["regimes"]) * int(
        protocol["no_profiler"]["paired_trials_per_regime"]
    ) * 2
    if len(rows) != expected_rows or manifest.get("raw_trial_count") != expected_rows:
        raise ValueError("paged decode raw trial count differs")
    profiles = manifest.get("profiles", {})
    expected_profile_ids = {
        f"{regime_id}-{method}"
        for regime_id in protocol["profiler"]["regime_ids"]
        for method in protocol["no_profiler"]["methods"]
    }
    if set(profiles) != expected_profile_ids:
        raise ValueError("paged decode profiler coverage differs")
    for profile_id, evidence in profiles.items():
        sqlite_path = artifact_dir / evidence["sqlite"]
        report_path = artifact_dir / evidence["report"]
        if file_sha256(sqlite_path) != evidence["sqlite_sha256"]:
            raise ValueError(f"paged decode NSYS SQLite hash differs: {profile_id}")
        if file_sha256(report_path) != evidence["report_sha256"]:
            raise ValueError(f"paged decode NSYS report hash differs: {profile_id}")
        method = profile_id.rsplit("-", 1)[1]
        parsed = parse_nsys_sqlite(
            sqlite_path,
            kernel_patterns=tuple(protocol["profiler"]["kernel_patterns"][method]),
        )
        if parsed != evidence["parsed"]:
            raise ValueError(f"paged decode NSYS parsed evidence differs: {profile_id}")
        if parsed["kernel_launches"] != int(protocol["profiler"]["repetitions_per_method"]):
            raise ValueError(f"paged decode NSYS launch count differs: {profile_id}")
    if not manifest.get("ncu_complete", False):
        prohibited = set(report.get("prohibited_claims", []))
        required = {"memory-bound classification", "occupancy explanation", "DRAM byte attribution"}
        if not required <= prohibited:
            raise ValueError("paged decode incomplete NCU does not prohibit counter claims")
        ncu = manifest.get("ncu", {})
        if set(ncu) != expected_profile_ids:
            raise ValueError("paged decode NCU failure coverage differs")
        for profile_id, evidence in ncu.items():
            status = evidence.get("status", {})
            if evidence.get("parsed") is not None or status.get("exit_code") == 0:
                raise ValueError(f"paged decode incomplete NCU status is inconsistent: {profile_id}")
            if status.get("available") and not (
                status.get("permission_denied") or status.get("driver_incompatible")
            ):
                raise ValueError(f"paged decode NCU failure reason is not auditable: {profile_id}")
        reasons = set(report.get("ncu_failure_reasons", []))
        if not reasons or not reasons <= {
            "ERR_NVGPUCTRPERM", "driver incompatibility", "CLI unavailable"
        }:
            raise ValueError("paged decode NCU failure reasons differ")
    return {"manifest": manifest, "report": report, "rows": rows}
