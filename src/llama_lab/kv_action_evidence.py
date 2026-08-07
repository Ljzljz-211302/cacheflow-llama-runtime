"""Trace-safe evaluation for the unified KV execution action policy."""

from __future__ import annotations

import json
import hashlib
import math
import random
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ACTIONS = ("direct", "remap", "paged", "device_swap", "host_swap", "recompute")
FEATURES = (
    "context_tokens", "batch", "page_runs", "kv_pressure", "kv_bytes", "reuse_distance"
)
MODEL_FEATURES = tuple(f"model_feature_{index}" for index in range(9))
POLICY_ALLOWED_INCLUDES = {
    "server-kv-action-policy.cpp": (
        '"server-kv-action-policy.h"', "<algorithm>", "<chrono>", "<cmath>",
    ),
    "server-kv-action-policy.h": ("<array>", "<cstddef>", "<cstdint>", "<limits>"),
}
POLICY_PROHIBITED_SYMBOLS = (
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cuCtxSynchronize",
    "ggml_backend_synchronize",
    "cuda_runtime",
    "ggml-cuda",
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
    if int(protocol.get("confirmatory", {}).get("minimum_pairs_per_regime", 0)) < 20:
        raise ValueError("KV action protocol requires at least 20 pairs per regime")
    if int(protocol.get("confirmatory", {}).get("bootstrap_samples", 0)) != 10000:
        raise ValueError("KV action protocol requires 10,000 bootstrap resamples")
    if protocol.get("confirmatory", {}).get("bootstrap_unit") != "trace_id":
        raise ValueError("KV action protocol bootstrap unit must be trace_id")
    if protocol.get("confirmatory", {}).get("order_randomization") != "seeded balanced Latin blocks":
        raise ValueError("KV action protocol requires seeded balanced Latin blocks")
    if not isinstance(protocol.get("confirmatory", {}).get("order_seed"), int):
        raise ValueError("KV action protocol requires an integer collection-order seed")
    if protocol.get("overhead_gates", {}).get("cuda_synchronizations") != 0:
        raise ValueError("KV action policy must prohibit CUDA synchronization")
    return protocol


def make_balanced_action_orders(
    actions: Iterable[str], trace_count: int, seed: int
) -> list[list[str]]:
    action_list = list(actions)
    if not action_list or trace_count % len(action_list) != 0:
        raise ValueError("balanced action orders require complete Latin blocks")
    generator = random.Random(seed)
    result: list[list[str]] = []
    for _ in range(trace_count // len(action_list)):
        base = action_list.copy()
        generator.shuffle(base)
        block = [base[offset:] + base[:offset] for offset in range(len(base))]
        generator.shuffle(block)
        result.extend(block)
    return result


def audit_kv_action_policy_no_cuda_sync(path: Path) -> dict[str, Any]:
    paths = (path, path.with_suffix(".h"))
    sources = {candidate.name: candidate.read_text(encoding="utf-8") for candidate in paths}
    includes = {
        name: re.findall(r"^\s*#\s*include\s+([^\s]+)", source, flags=re.MULTILINE)
        for name, source in sources.items()
    }
    unexpected = {
        name: [include for include in observed if include not in POLICY_ALLOWED_INCLUDES[name]]
        for name, observed in includes.items()
        if any(include not in POLICY_ALLOWED_INCLUDES[name] for include in observed)
    }
    matches = [
        {"file": name, "symbol": symbol}
        for name, source in sources.items()
        for symbol in POLICY_PROHIBITED_SYMBOLS
        if symbol in source
    ]
    return {
        "source_files": {candidate.name: _sha256(candidate) for candidate in paths},
        "allowed_includes": {
            name: list(allowed) for name, allowed in POLICY_ALLOWED_INCLUDES.items()
        },
        "observed_includes": includes,
        "unexpected_includes": unexpected,
        "prohibited_symbols": list(POLICY_PROHIBITED_SYMBOLS),
        "matches": matches,
        "passed": not unexpected and not matches,
    }


def validate_kv_action_collection_order(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> None:
    actions = list(protocol["production_actions"])
    trace_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        trace_rows[str(row["trace_id"])].append(row)
    ordered_traces = sorted(
        trace_rows, key=lambda trace: int(trace_rows[trace][0]["timestamp_order"])
    )
    expected = make_balanced_action_orders(
        actions, len(ordered_traces), int(protocol["confirmatory"]["order_seed"])
    )
    for trace, expected_order in zip(ordered_traces, expected):
        actual_by_action: dict[str, int] = {}
        for row in trace_rows[trace]:
            action = str(row["action"])
            order = row.get("collection_order")
            if action not in actions or not isinstance(order, int):
                raise ValueError("KV action collection order is missing or invalid")
            prior = actual_by_action.setdefault(action, order)
            if prior != order:
                raise ValueError("KV action collection order differs across paired snapshots")
        if set(actual_by_action) != set(actions):
            raise ValueError("KV action collection order lacks a production action")
        actual_order = [action for action, _ in sorted(actual_by_action.items(), key=lambda item: item[1])]
        if actual_order != expected_order or set(actual_by_action.values()) != set(range(len(actions))):
            raise ValueError("KV action collection order differs from preregistered balance")


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_kv_action_rows(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    required = {
        "snapshot_id", "trace_id", "session_id", "prefix_family", "split",
        "timestamp_order", "backend", "action", "baseline_action",
        "regime", "analytical_cost_ms", "observed_cost_ms", *FEATURES, *MODEL_FEATURES,
    }
    if not rows:
        raise ValueError("KV action evaluation rows are empty")
    split_by_trace: dict[str, str] = {}
    split_by_group: dict[tuple[str, str], str] = {}
    snapshots: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"KV action row missing fields: {sorted(missing)}")
        if row["split"] not in {"train", "evaluation"}:
            raise ValueError("KV action row split must be train or evaluation")
        if row["action"] not in ACTIONS or row["baseline_action"] not in ACTIONS:
            raise ValueError("KV action row contains an unknown action")
        for field in (*FEATURES, *MODEL_FEATURES, "analytical_cost_ms", "observed_cost_ms"):
            if not _finite(row[field]) or float(row[field]) < 0:
                raise ValueError(f"KV action row has invalid {field}")
        runtime_features = row.get("runtime_model_features")
        action_features = row.get("action_runtime_model_features")
        if not isinstance(runtime_features, list) or len(runtime_features) != 9 or not all(
            _finite(value) for value in runtime_features
        ):
            raise ValueError("KV action canonical runtime feature vector is invalid")
        if not isinstance(action_features, list) or len(action_features) != 9 or not all(
            _finite(value) for value in action_features
        ):
            raise ValueError("KV action per-action runtime feature vector is invalid")
        if [float(row[field]) for field in MODEL_FEATURES] != [
            float(value) for value in runtime_features
        ]:
            raise ValueError("KV action model features differ from runtime evidence")
        trace = str(row["trace_id"])
        prior = split_by_trace.setdefault(trace, row["split"])
        if prior != row["split"]:
            raise ValueError("trace_id leaks across train/evaluation")
        group = (str(row["session_id"]), str(row["prefix_family"]))
        prior_group = split_by_group.setdefault(group, row["split"])
        if prior_group != row["split"]:
            raise ValueError("session/prefix family leaks across train/evaluation")
        order = int(row["timestamp_order"])
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
        fixed = (
            "trace_id", "session_id", "prefix_family", "split", "timestamp_order",
            "regime", *FEATURES, *MODEL_FEATURES,
        )
        reference = group_rows[0]
        if any(any(row[field] != reference[field] for field in fixed) for row in group_rows[1:]):
            raise ValueError(f"snapshot {snapshot_id} changes paired features")
    minimum_pairs = int(protocol["confirmatory"]["minimum_pairs_per_regime"])
    evaluation_pairs: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["split"] == "evaluation":
            evaluation_pairs[str(row["regime"])].add(str(row["snapshot_id"]))
    if set(evaluation_pairs) != set(protocol["confirmatory"]["regimes"]):
        raise ValueError("KV action evaluation regime coverage differs from protocol")
    if any(
        len(pairs) < minimum_pairs for pairs in evaluation_pairs.values()
    ):
        raise ValueError("KV action evaluation pairs per regime are below protocol")


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


def _model_features(row: dict[str, Any]) -> list[float]:
    return [float(row[field]) for field in MODEL_FEATURES]


class _OnlineRidge:
    """Exact Python replay of server_kv_action_policy::ridge_model."""

    def __init__(self, ridge_lambda: float) -> None:
        self.inverse = [
            [1.0 / ridge_lambda if row == column else 0.0 for column in range(9)]
            for row in range(9)
        ]
        self.rhs = [0.0] * 9
        self.observations = 0
        self.residual_variance_ewma = 0.0

    def predict(self, features: list[float]) -> float:
        result = 0.0
        for row in range(9):
            theta = sum(
                self.inverse[row][column] * self.rhs[column] for column in range(9)
            )
            result += theta * features[row]
        return max(0.0, result)

    def radius(self, features: list[float], beta: float) -> float:
        quadratic = sum(
            features[row] * self.inverse[row][column] * features[column]
            for row in range(9) for column in range(9)
        )
        sigma = math.sqrt(max(1e-9, self.residual_variance_ewma))
        return beta * sigma * math.sqrt(max(0.0, quadratic))

    def observe(self, features: list[float], cost: float) -> None:
        previous = self.predict(features)
        inverse_x = [
            sum(self.inverse[row][column] * features[column] for column in range(9))
            for row in range(9)
        ]
        denominator = 1.0 + sum(
            features[row] * inverse_x[row] for row in range(9)
        )
        for row in range(9):
            for column in range(9):
                self.inverse[row][column] -= (
                    inverse_x[row] * inverse_x[column] / denominator
                )
            self.rhs[row] += features[row] * cost
        residual = cost - previous
        self.residual_variance_ewma = (
            residual * residual if self.observations == 0
            else 0.1 * residual * residual + 0.9 * self.residual_variance_ewma
        )
        self.observations += 1


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
    learned = {
        action: _OnlineRidge(float(protocol["learned"]["ridge_lambda"]))
        for action in ACTIONS
    }
    for row in sorted(
        train, key=lambda value: (int(value["timestamp_order"]), str(value["snapshot_id"]))
    ):
        learned[str(row["action"])].observe(
            _model_features(row), float(row["observed_cost_ms"])
        )
    minimum = int(protocol["learned"]["minimum_action_observations"])
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
        baseline_model = learned[baseline]
        learned_choice = baseline
        if baseline_model.observations >= minimum:
            features = _model_features(snapshot[0])
            baseline_prediction = baseline_model.predict(features)
            baseline_lower = max(
                0.0, baseline_prediction - baseline_model.radius(features, beta)
            )
            candidates: list[tuple[float, str]] = []
            for action in by_action:
                model = learned[action]
                if model.observations < minimum:
                    continue
                candidates.append(
                    (model.predict(features) + model.radius(features, beta), action)
                )
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
                "trace_id": snapshot[0]["trace_id"],
                "regime": snapshot[0]["regime"],
                "chosen": action,
                "oracle": oracle["action"],
                "cost_ms": cost,
                "regret_ms": cost - oracle_cost,
                "harmful": cost > baseline_cost * (1.0 + harm_ratio),
            })
    summary: dict[str, Any] = {}
    h0_by_snapshot = {
        str(row["snapshot_id"]): float(row["regret_ms"]) for row in results["H0"]
    }
    bootstrap_samples = int(protocol["confirmatory"]["bootstrap_samples"])
    bootstrap_seed = int(protocol["confirmatory"]["bootstrap_seed"])
    for model, decisions in results.items():
        regrets = [float(row["regret_ms"]) for row in decisions]
        deltas_by_trace: dict[str, list[float]] = defaultdict(list)
        for decision in decisions:
            deltas_by_trace[str(decision["trace_id"])].append(
                float(decision["regret_ms"]) - h0_by_snapshot[str(decision["snapshot_id"])]
            )
        trace_ids = sorted(deltas_by_trace)
        generator = random.Random(bootstrap_seed)
        bootstrapped = [
            statistics.fmean(
                delta
                for _ in trace_ids
                for delta in deltas_by_trace[trace_ids[generator.randrange(len(trace_ids))]]
            )
            for _ in range(bootstrap_samples)
        ]
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
            "paired_trace_cluster_mean_regret_delta_vs_h0_ci95_ms": [
                _percentile(bootstrapped, 0.025), _percentile(bootstrapped, 0.975)
            ],
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
    if manifest.get("artifact_version") != "h4-kv-action-v1.2.0":
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
    if not isinstance(outer_commit, str):
        raise ValueError("KV action source provenance is missing")
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{outer_commit}^{{commit}}"], cwd=project_root,
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        patch_path = manifest.get("patch_path")
        upstream_revision = manifest.get("upstream_revision")
        if not isinstance(patch_path, str) or not isinstance(upstream_revision, str):
            raise ValueError("KV action replay-patch provenance is missing")
        committed_patch = subprocess.check_output(
            ["git", "show", f"{outer_commit}:{patch_path}"], cwd=project_root,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("KV action source commit is unavailable") from error
    if manifest.get("patch_sha256") != hashlib.sha256(committed_patch).hexdigest():
        raise ValueError("KV action committed replay-patch hash differs")
    vendor_root = project_root / "vendor/llama.cpp"
    if vendor_root.is_dir():
        try:
            vendor_head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=vendor_root, text=True,
            ).strip()
            vendor_status = subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=vendor_root, text=True,
            )
            if vendor_head == upstream_revision:
                subprocess.run(
                    ["git", "apply", "--reverse", "--check", str(project_root / patch_path)],
                    cwd=vendor_root, check=True, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                patch_paths = {
                    line.split(" b/", 1)[1].decode()
                    for line in committed_patch.splitlines() if line.startswith(b"diff --git a/")
                }
                status_paths = {
                    line[3:].strip().replace("\\", "/") for line in vendor_status.splitlines()
                }
                if status_paths != patch_paths:
                    raise ValueError("KV action vendor worktree differs from the replay patch")
            else:
                if vendor_status:
                    raise ValueError("KV action committed vendor tree is dirty")
                vendor_patch = subprocess.check_output(
                    ["git", "diff", "--binary", f"{upstream_revision}..HEAD"], cwd=vendor_root,
                )
                if committed_patch != vendor_patch:
                    raise ValueError("KV action vendor tree differs from the replay patch")
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError("KV action vendor replay state is unavailable") from error
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
    validate_kv_action_collection_order(rows, protocol)
    for row in rows:
        if row.get("selected_delta") != 1 or row.get("observation_delta") != 1:
            raise ValueError("KV action runtime decision/observation linkage differs")
        if row.get("paged_decisions") != 0:
            raise ValueError("Paged action crossed the formal production evidence gate")
        if row.get("invalid_features") != 0:
            raise ValueError("KV action formal run contains invalid feature fallbacks")
        if not _finite(row.get("http_elapsed_ms")) or float(row["http_elapsed_ms"]) <= 0:
            raise ValueError("KV action HTTP diagnostic timing differs")
        expected_ratio = float(row["decision_cpu_ms"]) / float(row["observed_cost_ms"])
        if not math.isclose(
            float(row["decision_to_action_ratio"]), expected_ratio, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("KV action decision/action ratio differs from internal cost")
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
    policy_source = project_root / "vendor/llama.cpp/tools/server/server-kv-action-policy.cpp"
    sync_audit = json.loads((artifact / "sync-audit.json").read_text(encoding="utf-8"))
    expected_sync_audit = audit_kv_action_policy_no_cuda_sync(policy_source)
    if sync_audit != expected_sync_audit or not expected_sync_audit["passed"]:
        raise ValueError("KV action CUDA synchronization audit differs or did not pass")
    gates = protocol["overhead_gates"]
    ratios = sorted(float(row["decision_to_action_ratio"]) for row in rows)
    expected_overhead = {
        "p99_choose_microseconds": max(float(row["p99_ns"]) for row in overhead_rows) / 1000.0,
        "measured_max_choose_microseconds": max(float(row["max_ns"]) for row in overhead_rows) / 1000.0,
        "hot_loop_allocations": sum(int(row["allocations"]) for row in overhead_rows),
        "cuda_synchronizations": len(expected_sync_audit["matches"]),
        "scheduler_cpu_ratio_p99": ratios[int(0.99 * (len(ratios) - 1))],
    }
    expected_overhead["passed"] = (
        expected_overhead["p99_choose_microseconds"] <= gates["p99_choose_microseconds_max"]
        and expected_overhead["hot_loop_allocations"] == gates["hot_loop_allocations"]
        and expected_overhead["cuda_synchronizations"] == gates["cuda_synchronizations"]
        and expected_overhead["scheduler_cpu_ratio_p99"] <= gates["scheduler_cpu_ratio_p99_max"]
    )
    if report.get("overhead") != expected_overhead or not expected_overhead["passed"]:
        raise ValueError("KV action overhead gates differ or did not pass")
    return report
