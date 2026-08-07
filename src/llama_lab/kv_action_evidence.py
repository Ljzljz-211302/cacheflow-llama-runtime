"""Trace-safe evaluation for the unified KV execution action policy."""

from __future__ import annotations

import json
import hashlib
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ACTIONS = ("direct", "remap", "paged", "device_swap", "host_swap", "recompute")
FEATURES = (
    "context_tokens", "batch", "page_runs", "kv_pressure", "kv_bytes", "reuse_distance"
)


def load_kv_action_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1:
        raise ValueError("KV action protocol schema_version must be 1")
    if protocol.get("models") != ["H0", "A1", "T1", "L1"]:
        raise ValueError("KV action protocol must compare H0/A1/T1/L1")
    if protocol.get("split", {}).get("unit") != "trace_id":
        raise ValueError("KV action protocol split unit must be trace_id")
    if not protocol.get("split", {}).get("row_randomization_prohibited"):
        raise ValueError("KV action protocol must prohibit row randomization")
    return protocol


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_kv_action_rows(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    required = {
        "snapshot_id", "trace_id", "session_id", "prefix_family", "split",
        "timestamp_order", "backend", "action", "baseline_action",
        "analytical_cost_ms", "observed_cost_ms", *FEATURES,
    }
    if not rows:
        raise ValueError("KV action evaluation rows are empty")
    split_by_trace: dict[str, str] = {}
    split_by_group: dict[tuple[str, str], str] = {}
    timestamp_by_trace: dict[str, set[int]] = defaultdict(set)
    snapshots: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"KV action row missing fields: {sorted(missing)}")
        if row["split"] not in {"train", "evaluation"}:
            raise ValueError("KV action row split must be train or evaluation")
        if row["action"] not in ACTIONS or row["baseline_action"] not in ACTIONS:
            raise ValueError("KV action row contains an unknown action")
        for field in (*FEATURES, "analytical_cost_ms", "observed_cost_ms"):
            if not _finite(row[field]) or float(row[field]) < 0:
                raise ValueError(f"KV action row has invalid {field}")
        trace = str(row["trace_id"])
        prior = split_by_trace.setdefault(trace, row["split"])
        if prior != row["split"]:
            raise ValueError("trace_id leaks across train/evaluation")
        group = (str(row["session_id"]), str(row["prefix_family"]))
        prior_group = split_by_group.setdefault(group, row["split"])
        if prior_group != row["split"]:
            raise ValueError("session/prefix family leaks across train/evaluation")
        order = int(row["timestamp_order"])
        timestamp_by_trace[trace].add(order)
        snapshots[str(row["snapshot_id"])].append(row)
    train_traces = {trace for trace, split in split_by_trace.items() if split == "train"}
    eval_traces = {trace for trace, split in split_by_trace.items() if split == "evaluation"}
    split = protocol["split"]
    if len(train_traces) < int(split["minimum_train_traces"]):
        raise ValueError("KV action training trace count is below protocol")
    if len(eval_traces) < int(split["minimum_evaluation_traces"]):
        raise ValueError("KV action evaluation trace count is below protocol")
    train_max = max(int(row["timestamp_order"]) for row in rows if row["split"] == "train")
    eval_min = min(int(row["timestamp_order"]) for row in rows if row["split"] == "evaluation")
    if train_max >= eval_min:
        raise ValueError("KV action evaluation does not occur after training")
    for snapshot_id, group_rows in snapshots.items():
        actions = [row["action"] for row in group_rows]
        if len(actions) != len(set(actions)) or len(actions) < 2:
            raise ValueError(f"snapshot {snapshot_id} lacks unique paired actions")
        baseline = {row["baseline_action"] for row in group_rows}
        if len(baseline) != 1 or next(iter(baseline)) not in actions:
            raise ValueError(f"snapshot {snapshot_id} has an invalid H0 baseline")
        fixed = ("trace_id", "session_id", "prefix_family", "split", "timestamp_order", *FEATURES)
        reference = group_rows[0]
        if any(any(row[field] != reference[field] for field in fixed) for row in group_rows[1:]):
            raise ValueError(f"snapshot {snapshot_id} changes paired features")


def _bucket(value: float, edges: Iterable[float]) -> int:
    for index, edge in enumerate(edges):
        if value <= edge:
            return index
    return sum(1 for _ in edges)


def _table_key(row: dict[str, Any], protocol: dict[str, Any]) -> tuple[Any, ...]:
    edges = protocol["table_buckets"]
    return (
        row["backend"], row["action"],
        _bucket(float(row["context_tokens"]), edges["context_tokens"]),
        _bucket(float(row["batch"]), edges["batch"]),
        _bucket(float(row["page_runs"]), edges["page_runs"]),
        _bucket(float(row["kv_pressure"]), edges["kv_pressure"]),
    )


def _scaled_features(row: dict[str, Any]) -> list[float]:
    return [
        1.0,
        min(4.0, float(row["context_tokens"]) / 4096.0),
        min(4.0, float(row["batch"]) / 4.0),
        min(4.0, float(row["page_runs"]) / 64.0),
        float(row["kv_pressure"]),
        min(4.0, float(row["kv_bytes"]) / (1024.0**3)),
        min(4.0, math.log1p(float(row["reuse_distance"])) / 20.0),
    ]


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    augmented = [row[:] + [rhs[index]] for index, row in enumerate(matrix)]
    size = len(rhs)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        if abs(divisor) < 1e-12:
            continue
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                left - factor * right
                for left, right in zip(augmented[row], augmented[column])
            ]
    return [augmented[index][-1] for index in range(size)]


def _fit_ridge(rows: list[dict[str, Any]], ridge_lambda: float) -> tuple[list[float], float]:
    width = len(_scaled_features(rows[0]))
    normal = [[0.0] * width for _ in range(width)]
    rhs = [0.0] * width
    for row in rows:
        features = _scaled_features(row)
        cost = float(row["observed_cost_ms"])
        for left in range(width):
            rhs[left] += features[left] * cost
            for right in range(width):
                normal[left][right] += features[left] * features[right]
    for index in range(width):
        normal[index][index] += ridge_lambda
    weights = _solve(normal, rhs)
    residuals = [
        float(row["observed_cost_ms"]) - sum(
            weight * feature for weight, feature in zip(weights, _scaled_features(row))
        )
        for row in rows
    ]
    rmse = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    return weights, rmse


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(quantile * (len(ordered) - 1)))]


def evaluate_kv_action_models(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    validate_kv_action_rows(rows, protocol)
    train = [row for row in rows if row["split"] == "train"]
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    table_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in train:
        table_values[_table_key(row, protocol)].append(float(row["observed_cost_ms"]))
    table = {key: statistics.median(values) for key, values in table_values.items()}
    learned_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train:
        learned_rows[row["action"]].append(row)
    minimum = int(protocol["learned"]["minimum_action_observations"])
    learned = {
        action: _fit_ridge(action_rows, float(protocol["learned"]["ridge_lambda"]))
        for action, action_rows in learned_rows.items() if len(action_rows) >= minimum
    }
    snapshots: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluation:
        snapshots[str(row["snapshot_id"])].append(row)
    results: dict[str, list[dict[str, Any]]] = {name: [] for name in protocol["models"]}
    margin = float(protocol["learned"]["switch_margin_ms"])
    beta = float(protocol["learned"]["confidence_beta"])
    harm_ratio = float(protocol["harm"]["relative_to_h0_percent"]) / 100.0
    for snapshot_id, snapshot in snapshots.items():
        by_action = {row["action"]: row for row in snapshot}
        baseline = snapshot[0]["baseline_action"]
        oracle = min(snapshot, key=lambda row: float(row["observed_cost_ms"]))
        choices: dict[str, str] = {
            "H0": baseline,
            "A1": min(snapshot, key=lambda row: float(row["analytical_cost_ms"]))["action"],
        }
        table_candidates = [row for row in snapshot if _table_key(row, protocol) in table]
        choices["T1"] = min(
            table_candidates, key=lambda row: table[_table_key(row, protocol)]
        )["action"] if table_candidates else baseline
        baseline_model = learned.get(baseline)
        learned_choice = baseline
        if baseline_model:
            features = _scaled_features(snapshot[0])
            baseline_prediction = max(0.0, sum(
                weight * feature for weight, feature in zip(baseline_model[0], features)
            ))
            baseline_lower = max(0.0, baseline_prediction - beta * baseline_model[1])
            candidates: list[tuple[float, str]] = []
            for action in by_action:
                model = learned.get(action)
                if not model:
                    continue
                prediction = max(0.0, sum(
                    weight * feature for weight, feature in zip(model[0], features)
                ))
                candidates.append((prediction + beta * model[1], action))
            if candidates:
                upper, candidate = min(candidates)
                if candidate != baseline and upper + margin < baseline_lower:
                    learned_choice = candidate
        choices["L1"] = learned_choice
        oracle_cost = float(oracle["observed_cost_ms"])
        baseline_cost = float(by_action[baseline]["observed_cost_ms"])
        for model, action in choices.items():
            cost = float(by_action[action]["observed_cost_ms"])
            results[model].append({
                "snapshot_id": snapshot_id,
                "chosen": action,
                "oracle": oracle["action"],
                "cost_ms": cost,
                "regret_ms": cost - oracle_cost,
                "harmful": cost > baseline_cost * (1.0 + harm_ratio),
            })
    summary: dict[str, Any] = {}
    for model, decisions in results.items():
        regrets = [float(row["regret_ms"]) for row in decisions]
        summary[model] = {
            "decisions": len(decisions),
            "mean_regret_ms": statistics.fmean(regrets),
            "median_regret_ms": statistics.median(regrets),
            "p95_regret_ms": _percentile(regrets, 0.95),
            "p99_regret_ms": _percentile(regrets, 0.99),
            "max_regret_ms": max(regrets),
            "cumulative_regret_ms": sum(regrets),
            "harmful_decisions": sum(bool(row["harmful"]) for row in decisions),
            "harmful_rate": statistics.fmean(bool(row["harmful"]) for row in decisions),
            "choices": {action: sum(row["chosen"] == action for row in decisions) for action in ACTIONS},
        }
    return {
        "train_traces": len({row["trace_id"] for row in train}),
        "evaluation_traces": len({row["trace_id"] for row in evaluation}),
        "group_overlap": False,
        "models": summary,
        "decisions": results,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_kv_action_artifact(artifact: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = load_kv_action_protocol(protocol_path)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((artifact / "report.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_version") != "h4-kv-action-v1.0.0":
        raise ValueError("KV action artifact version differs")
    if manifest.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("KV action protocol hash differs")
    project_root = protocol_path.resolve().parents[1]
    model_relative = manifest.get("model_path")
    if not isinstance(model_relative, str) or Path(model_relative).is_absolute():
        raise ValueError("KV action model provenance is missing")
    model_path = project_root / model_relative
    if not model_path.is_file() or manifest.get("model_sha256") != _sha256(model_path):
        raise ValueError("KV action model hash differs")
    expected_runs = {
        "direct": {"port": 19840, "kv_action_policy": "fixed", "cache_idle_slots": False},
        "device_swap": {"port": 19841, "kv_action_policy": "fixed", "cache_idle_slots": True},
        "host_swap": {
            "port": 19842, "kv_action_policy": "fixed", "cache_idle_slots": True,
            "kv_swap_path": "memory", "kv_swap_budget_mib": 256,
        },
        "recompute": {"port": 19843, "kv_action_policy": "fixed", "cache_idle_slots": False},
    }
    if manifest.get("runs") != expected_runs:
        raise ValueError("KV action service run configuration differs")
    outer_commit = manifest.get("outer_commit")
    vendor_commit = manifest.get("vendor_commit")
    if not isinstance(outer_commit, str) or not isinstance(vendor_commit, str):
        raise ValueError("KV action source provenance is missing")
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{outer_commit}^{{commit}}"], cwd=project_root,
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["git", "cat-file", "-e", f"{vendor_commit}^{{commit}}"],
            cwd=project_root / "vendor/llama.cpp", check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        patch_path = manifest.get("patch_path")
        upstream_revision = manifest.get("upstream_revision")
        if not isinstance(patch_path, str) or not isinstance(upstream_revision, str):
            raise ValueError("KV action replay-patch provenance is missing")
        committed_patch = subprocess.check_output(
            ["git", "show", f"{outer_commit}:{patch_path}"], cwd=project_root,
        )
        vendor_patch = subprocess.check_output(
            ["git", "diff", "--binary", f"{upstream_revision}..{vendor_commit}"],
            cwd=project_root / "vendor/llama.cpp",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("KV action source commit is unavailable") from error
    if manifest.get("patch_sha256") != hashlib.sha256(committed_patch).hexdigest():
        raise ValueError("KV action committed replay-patch hash differs")
    if committed_patch != vendor_patch:
        raise ValueError("KV action replay patch does not bind the vendor commit")
    expected_files = set(manifest.get("files", {}))
    actual_files = {
        path.relative_to(artifact).as_posix()
        for path in artifact.rglob("*") if path.is_file() and path.name != "manifest.json"
    }
    if expected_files != actual_files:
        raise ValueError("KV action artifact file tree differs from manifest")
    for relative, expected in manifest["files"].items():
        if _sha256(artifact / relative) != expected:
            raise ValueError(f"KV action artifact hash differs: {relative}")
    rows = [
        json.loads(line) for line in (artifact / "paired-actions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines() if line
    ]
    for row in rows:
        if row.get("selected_delta") != 1 or row.get("observation_delta") != 1:
            raise ValueError("KV action runtime decision/observation linkage differs")
        if row.get("paged_decisions") != 0:
            raise ValueError("Paged action crossed the formal production evidence gate")
        if row.get("invalid_features") != 0:
            raise ValueError("KV action formal run contains invalid feature fallbacks")
    recomputed = evaluate_kv_action_models(rows, protocol)
    if report.get("analysis") != recomputed:
        raise ValueError("KV action report analysis differs from paired trials")
    overhead_rows = [
        json.loads(line) for line in (artifact / "overhead.jsonl").read_text(
            encoding="utf-8"
        ).splitlines() if line
    ]
    if len(overhead_rows) != 5:
        raise ValueError("KV action overhead regime coverage differs")
    gates = protocol["overhead_gates"]
    ratios = sorted(float(row["decision_to_action_ratio"]) for row in rows)
    expected_overhead = {
        "p99_choose_microseconds": max(float(row["p99_ns"]) for row in overhead_rows) / 1000.0,
        "measured_max_choose_microseconds": max(float(row["max_ns"]) for row in overhead_rows) / 1000.0,
        "hot_loop_allocations": sum(int(row["allocations"]) for row in overhead_rows),
        "scheduler_cpu_ratio_p99": ratios[int(0.99 * (len(ratios) - 1))],
    }
    expected_overhead["passed"] = (
        expected_overhead["p99_choose_microseconds"] <= gates["p99_choose_microseconds_max"]
        and expected_overhead["hot_loop_allocations"] == gates["hot_loop_allocations"]
        and expected_overhead["scheduler_cpu_ratio_p99"] <= gates["scheduler_cpu_ratio_p99_max"]
    )
    if report.get("overhead") != expected_overhead or not expected_overhead["passed"]:
        raise ValueError("KV action overhead gates differ or did not pass")
    if recomputed["models"]["L1"] != recomputed["models"]["H0"]:
        raise ValueError("formal selection requires L1 to fail closed exactly to H0")
    return report
