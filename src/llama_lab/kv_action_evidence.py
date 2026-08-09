"""Trace-safe evaluation for the unified KV execution action policy."""

from __future__ import annotations

import copy
import json
import hashlib
import math
import os
import random
import re
import statistics
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ACTIONS = ("direct", "remap", "paged", "device_swap", "host_swap", "recompute")
FEATURES = (
    "context_tokens", "batch", "page_runs", "kv_pressure", "kv_bytes", "reuse_distance"
)
MODEL_FEATURES = tuple(f"model_feature_{index}" for index in range(9))
DELTA_INTERACTION_FEATURES = {
    "cached_tokens_x_prefill_cost": (1, 7),
    "kv_bytes_x_kv_pressure": (3, 6),
    "host_transfer_x_reuse_distance": (8, 5),
    "fragmentation_x_kv_bytes": (4, 3),
}
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
    if protocol.get("models") != ["H0", "A1", "T1", "L1", "D1"]:
        raise ValueError("KV action protocol must compare H0/A1/T1/L1/D1")
    paired_delta = protocol.get("paired_delta", {})
    interactions = paired_delta.get("interaction_features")
    if (
        not isinstance(interactions, list)
        or len(interactions) != len(set(interactions))
        or any(name not in DELTA_INTERACTION_FEATURES for name in interactions)
    ):
        raise ValueError("KV action protocol has invalid paired-delta interactions")
    if paired_delta.get("regime_conditioned") is not True:
        raise ValueError("KV action protocol must condition paired deltas by regime")
    if paired_delta.get("calibration_enabled") is not True:
        raise ValueError("KV action protocol must enable paired-delta calibration")
    calibration_traces = paired_delta.get("calibration_traces")
    training_traces = protocol.get("split", {}).get("minimum_train_traces")
    if (
        not isinstance(calibration_traces, int)
        or not isinstance(training_traces, int)
        or calibration_traces <= 0
        or calibration_traces >= training_traces
    ):
        raise ValueError("KV action protocol has invalid paired-delta calibration traces")
    quantile = paired_delta.get("one_sided_quantile")
    if not isinstance(quantile, (int, float)) or not 0.5 < float(quantile) < 1.0:
        raise ValueError("KV action protocol has invalid paired-delta calibration quantile")
    acceptance = paired_delta.get("acceptance")
    if not isinstance(acceptance, dict) or acceptance != {
        "minimum_switches_vs_h0": 1,
        "mean_regret_delta_ci95_upper_must_be_negative": True,
        "p95_regret_must_not_exceed_h0": True,
        "harmful_rate_must_not_exceed_h0": True,
    }:
        raise ValueError("KV action protocol has invalid paired-delta acceptance gates")
    if protocol.get("split", {}).get("unit") != "trace_id":
        raise ValueError("KV action protocol split unit must be trace_id")
    if not protocol.get("split", {}).get("row_randomization_prohibited"):
        raise ValueError("KV action protocol must prohibit row randomization")
    if int(protocol.get("confirmatory", {}).get("minimum_pairs_per_regime", 0)) < 40:
        raise ValueError("KV action protocol requires at least 40 pairs per regime")
    if int(protocol.get("confirmatory", {}).get("bootstrap_samples", 0)) != 10000:
        raise ValueError("KV action protocol requires 10,000 bootstrap resamples")
    if protocol.get("confirmatory", {}).get("bootstrap_unit") != "trace_id":
        raise ValueError("KV action protocol bootstrap unit must be trace_id")
    if protocol.get("confirmatory", {}).get("order_randomization") != "seeded balanced Latin blocks":
        raise ValueError("KV action protocol requires seeded balanced Latin blocks")
    if not isinstance(protocol.get("confirmatory", {}).get("order_seed"), int):
        raise ValueError("KV action protocol requires an integer collection-order seed")
    if protocol.get("overhead_gates", {}).get("direct_cuda_sync_symbols") != 0:
        raise ValueError("KV action policy must prohibit direct CUDA synchronization symbols")
    divergence = protocol.get("confirmatory", {}).get("max_action_feature_delta")
    if not isinstance(divergence, list) or len(divergence) != 9 or not all(
        _finite(value) and float(value) >= 0 for value in divergence
    ):
        raise ValueError("KV action protocol requires nine feature-divergence gates")
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


def _audit_kv_action_policy_sources(sources: dict[str, bytes]) -> dict[str, Any]:
    decoded = {name: source.decode("utf-8") for name, source in sources.items()}
    includes = {
        name: re.findall(r"^\s*#\s*include\s+([^\s]+)", source, flags=re.MULTILINE)
        for name, source in decoded.items()
    }
    unexpected = {
        name: [include for include in observed if include not in POLICY_ALLOWED_INCLUDES[name]]
        for name, observed in includes.items()
        if any(include not in POLICY_ALLOWED_INCLUDES[name] for include in observed)
    }
    matches = [
        {"file": name, "symbol": symbol}
        for name, source in decoded.items()
        for symbol in POLICY_PROHIBITED_SYMBOLS
        if symbol in source
    ]
    return {
        "source_files": {
            name: hashlib.sha256(source).hexdigest() for name, source in sources.items()
        },
        "allowed_includes": {
            name: list(allowed) for name, allowed in POLICY_ALLOWED_INCLUDES.items()
        },
        "observed_includes": includes,
        "unexpected_includes": unexpected,
        "prohibited_symbols": list(POLICY_PROHIBITED_SYMBOLS),
        "matches": matches,
        "passed": not unexpected and not matches,
    }


def audit_kv_action_policy_no_cuda_sync(path: Path) -> dict[str, Any]:
    paths = (path, path.with_suffix(".h"))
    return _audit_kv_action_policy_sources(
        {candidate.name: candidate.read_bytes() for candidate in paths}
    )


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
        "regime", "analytical_cost_ms", "observed_cost_ms", "observation_id",
        "observation_order", *FEATURES, *MODEL_FEATURES,
    }
    if not rows:
        raise ValueError("KV action evaluation rows are empty")
    split_by_trace: dict[str, str] = {}
    split_by_group: dict[tuple[str, str], str] = {}
    snapshots: dict[str, list[dict[str, Any]]] = defaultdict(list)
    observations: dict[str, dict[str, Any]] = {}
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
        observation_id = str(row["observation_id"])
        if observation_id in observations:
            raise ValueError("KV action physical observation is reused across rows")
        observations[observation_id] = row
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
        reference = group_rows[0]
        actions = [row["action"] for row in group_rows]
        if len(actions) != len(set(actions)) or len(actions) < 2:
            raise ValueError(f"snapshot {snapshot_id} lacks unique paired actions")
        baseline = {row["baseline_action"] for row in group_rows}
        if len(baseline) != 1 or next(iter(baseline)) not in actions:
            raise ValueError(f"snapshot {snapshot_id} has an invalid H0 baseline")
        baseline_action = next(iter(baseline))
        baseline_row = next(row for row in group_rows if row["action"] == baseline_action)
        if [float(value) for value in reference["runtime_model_features"]] != [
            float(value) for value in baseline_row["action_runtime_model_features"]
        ]:
            raise ValueError(
                f"snapshot {snapshot_id} canonical features differ from its runtime H0 anchor"
            )
        feature_gates = [
            float(value) for value in protocol["confirmatory"]["max_action_feature_delta"]
        ]
        for row in group_rows:
            for index, (canonical, actual, gate) in enumerate(zip(
                reference["runtime_model_features"], row["action_runtime_model_features"],
                feature_gates,
            )):
                if abs(float(canonical) - float(actual)) > gate + 1e-12:
                    raise ValueError(
                        f"snapshot {snapshot_id} action feature {index} exceeds divergence gate"
                    )
        fixed = (
            "trace_id", "session_id", "prefix_family", "split", "timestamp_order",
            "regime", *FEATURES, *MODEL_FEATURES,
        )
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
    required_regimes = set(protocol["confirmatory"]["regimes"])
    for trace, split_name in split_by_trace.items():
        trace_snapshots = {
            str(row["regime"]): str(row["snapshot_id"])
            for row in rows if str(row["trace_id"]) == trace
        }
        if set(trace_snapshots) != required_regimes or len(set(trace_snapshots.values())) != len(
            required_regimes
        ):
            raise ValueError(f"KV action {split_name} trace cluster is incomplete")


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


def _paired_delta_features(row: dict[str, Any], interaction_names: list[str]) -> list[float]:
    features = _model_features(row)
    for name in interaction_names:
        if name not in DELTA_INTERACTION_FEATURES:
            raise ValueError(f"unknown paired-delta interaction feature: {name}")
        left, right = DELTA_INTERACTION_FEATURES[name]
        features.append(features[left] * features[right])
    return features


def _paired_delta_key(regime: str, action: str, regime_conditioned: bool) -> tuple[str, str]:
    return (regime if regime_conditioned else "pooled", action)


class _OnlineRidge:
    """Exact Python replay of server_kv_action_policy::ridge_model."""

    def __init__(
        self, ridge_lambda: float, *, feature_count: int = 9, nonnegative: bool = True
    ) -> None:
        self.inverse = [
            [
                1.0 / ridge_lambda if row == column else 0.0
                for column in range(feature_count)
            ]
            for row in range(feature_count)
        ]
        self.rhs = [0.0] * feature_count
        self.observations = 0
        self.residual_variance_ewma = 0.0
        self.nonnegative = nonnegative

    def predict(self, features: list[float]) -> float:
        if len(features) != len(self.rhs):
            raise ValueError("ridge feature count differs from model")
        result = 0.0
        for row in range(len(self.rhs)):
            theta = sum(
                self.inverse[row][column] * self.rhs[column]
                for column in range(len(self.rhs))
            )
            result += theta * features[row]
        return max(0.0, result) if self.nonnegative else result

    def radius(self, features: list[float], beta: float) -> float:
        quadratic = sum(
            features[row] * self.inverse[row][column] * features[column]
            for row in range(len(self.rhs)) for column in range(len(self.rhs))
        )
        sigma = math.sqrt(max(1e-9, self.residual_variance_ewma))
        return beta * sigma * math.sqrt(max(0.0, quadratic))

    def observe(self, features: list[float], cost: float) -> None:
        previous = self.predict(features)
        inverse_x = [
            sum(
                self.inverse[row][column] * features[column]
                for column in range(len(self.rhs))
            )
            for row in range(len(self.rhs))
        ]
        denominator = 1.0 + sum(
            features[row] * inverse_x[row] for row in range(len(self.rhs))
        )
        for row in range(len(self.rhs)):
            for column in range(len(self.rhs)):
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


def _conformal_upper_quantile(values: list[float], quantile: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    rank = math.ceil((len(ordered) + 1) * quantile) - 1
    return ordered[min(len(ordered) - 1, max(0, rank))]


def evaluate_kv_action_models(
    rows: list[dict[str, Any]], protocol: dict[str, Any], *, _include_ablations: bool = True
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
    for row in sorted(train, key=lambda value: int(value["observation_order"])):
        learned[str(row["action"])].observe(
            _model_features(row), float(row["observed_cost_ms"])
        )
    delta_config = protocol["paired_delta"]
    regime_conditioned = bool(delta_config["regime_conditioned"])
    interaction_names = [str(name) for name in delta_config["interaction_features"]]
    delta_feature_names = [*MODEL_FEATURES, *interaction_names]
    delta_models: dict[tuple[str, str], _OnlineRidge] = {}
    train_snapshots: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train:
        train_snapshots[str(row["snapshot_id"])].append(row)
    ordered_train_traces = sorted(
        {str(row["trace_id"]): int(row["timestamp_order"]) for row in train}.items(),
        key=lambda item: item[1],
    )
    calibration_trace_count = int(delta_config["calibration_traces"])
    if calibration_trace_count <= 0 or calibration_trace_count >= len(ordered_train_traces):
        raise ValueError("paired-delta calibration trace count is invalid")
    calibration_traces = {
        trace for trace, _ in ordered_train_traces[-calibration_trace_count:]
    }
    fit_snapshots = [
        snapshot for snapshot in train_snapshots.values()
        if str(snapshot[0]["trace_id"]) not in calibration_traces
    ]
    calibration_snapshots = [
        snapshot for snapshot in train_snapshots.values()
        if str(snapshot[0]["trace_id"]) in calibration_traces
    ]
    for snapshot in sorted(
        fit_snapshots, key=lambda value: int(value[0]["observation_order"])
    ):
        by_action = {str(row["action"]): row for row in snapshot}
        baseline = str(snapshot[0]["baseline_action"])
        baseline_cost = float(by_action[baseline]["observed_cost_ms"])
        regime = str(snapshot[0]["regime"])
        features = _paired_delta_features(snapshot[0], interaction_names)
        for action, row in by_action.items():
            if action == baseline:
                continue
            key = _paired_delta_key(regime, action, regime_conditioned)
            model = delta_models.setdefault(
                key,
                _OnlineRidge(
                    float(delta_config["ridge_lambda"]),
                    feature_count=len(delta_feature_names),
                    nonnegative=False,
                ),
            )
            model.observe(features, float(row["observed_cost_ms"]) - baseline_cost)
    calibration_residuals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for snapshot in calibration_snapshots:
        by_action = {str(row["action"]): row for row in snapshot}
        baseline = str(snapshot[0]["baseline_action"])
        baseline_cost = float(by_action[baseline]["observed_cost_ms"])
        regime = str(snapshot[0]["regime"])
        features = _paired_delta_features(snapshot[0], interaction_names)
        for action, row in by_action.items():
            key = _paired_delta_key(regime, action, regime_conditioned)
            if action == baseline or key not in delta_models:
                continue
            observed_delta = float(row["observed_cost_ms"]) - baseline_cost
            calibration_residuals[key].append(
                observed_delta - delta_models[key].predict(features)
            )
    delta_calibration = (
        {
            key: max(0.0, _conformal_upper_quantile(
                values, float(delta_config["one_sided_quantile"])
            ))
            for key, values in calibration_residuals.items()
        }
        if bool(delta_config["calibration_enabled"])
        else {key: 0.0 for key in delta_models}
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
        delta_choice = baseline
        delta_candidates: list[tuple[float, str, float, float, float]] = []
        delta_minimum = int(delta_config["minimum_action_observations"])
        features = _paired_delta_features(snapshot[0], interaction_names)
        for action in by_action:
            if action == baseline:
                continue
            key = _paired_delta_key(
                str(snapshot[0]["regime"]), str(action), regime_conditioned
            )
            model = delta_models.get(key)
            if model is None or model.observations < delta_minimum:
                continue
            prediction = model.predict(features)
            radius = model.radius(features, float(delta_config["confidence_beta"]))
            calibration = delta_calibration.get(key, math.inf)
            upper_delta = prediction + radius + calibration
            delta_candidates.append(
                (upper_delta, str(action), prediction, radius, calibration)
            )
        delta_detail: dict[str, Any] = {
            "candidate": baseline,
            "predicted_delta_ms": 0.0,
            "ridge_radius_ms": 0.0,
            "calibration_offset_ms": 0.0,
            "upper_delta_ms": 0.0,
            "switch_margin_ms": float(delta_config["switch_margin_ms"]),
            "reason": "insufficient_observations",
        }
        if delta_candidates:
            best_upper_delta, candidate, prediction, radius, calibration = min(delta_candidates)
            delta_detail.update({
                "candidate": candidate,
                "predicted_delta_ms": prediction,
                "ridge_radius_ms": radius,
                "calibration_offset_ms": calibration,
                "upper_delta_ms": best_upper_delta,
                "reason": "nonnegative_upper_bound",
            })
            if best_upper_delta + float(delta_config["switch_margin_ms"]) < 0.0:
                delta_choice = candidate
                delta_detail["reason"] = "negative_upper_bound"
        choices["D1"] = delta_choice
        oracle_cost = float(oracle["observed_cost_ms"])
        baseline_cost = float(by_action[baseline]["observed_cost_ms"])
        for model, action in choices.items():
            cost = float(by_action[action]["observed_cost_ms"])
            result_row = {
                "snapshot_id": snapshot_id,
                "trace_id": snapshot[0]["trace_id"],
                "regime": snapshot[0]["regime"],
                "chosen": action,
                "oracle": oracle["action"],
                "cost_ms": cost,
                "regret_ms": cost - oracle_cost,
                "harmful": cost > baseline_cost * (1.0 + harm_ratio),
            }
            if model == "D1":
                result_row.update(delta_detail)
            results[model].append(result_row)
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
            "switches_vs_h0": sum(
                row["chosen"] != next(
                    baseline["chosen"] for baseline in results["H0"]
                    if baseline["snapshot_id"] == row["snapshot_id"]
                )
                for row in decisions
            ),
            "paired_trace_cluster_mean_regret_delta_vs_h0_ci95_ms": [
                _percentile(bootstrapped, 0.025), _percentile(bootstrapped, 0.975)
            ],
            "choices": {action: sum(row["chosen"] == action for row in decisions) for action in ACTIONS},
        }
    delta_acceptance_config = delta_config["acceptance"]
    delta_summary = summary["D1"]
    h0_summary = summary["H0"]
    delta_acceptance = {
        "minimum_switches_vs_h0": (
            int(delta_summary["switches_vs_h0"])
            >= int(delta_acceptance_config["minimum_switches_vs_h0"])
        ),
        "mean_regret_delta_ci95_upper_negative": (
            float(delta_summary["paired_trace_cluster_mean_regret_delta_vs_h0_ci95_ms"][1])
            < 0.0
        ),
        "p95_regret_not_above_h0": (
            float(delta_summary["p95_regret_ms"]) <= float(h0_summary["p95_regret_ms"])
        ),
        "harmful_rate_not_above_h0": (
            float(delta_summary["harmful_rate"]) <= float(h0_summary["harmful_rate"])
        ),
    }
    delta_acceptance["passed"] = all(delta_acceptance.values())
    max_feature_delta = [0.0] * 9
    for row in rows:
        for index, (canonical, actual) in enumerate(zip(
            row["runtime_model_features"], row["action_runtime_model_features"]
        )):
            max_feature_delta[index] = max(
                max_feature_delta[index], abs(float(canonical) - float(actual))
            )
    ablations: dict[str, Any] = {}
    if _include_ablations:
        no_interactions = copy.deepcopy(protocol)
        no_interactions["paired_delta"]["interaction_features"] = []
        ablations["D1-I0-no-interactions"] = evaluate_kv_action_models(
            rows, no_interactions, _include_ablations=False
        )["models"]["D1"]
        pooled = copy.deepcopy(protocol)
        pooled["paired_delta"]["regime_conditioned"] = False
        ablations["D1-R0-pooled-regimes"] = evaluate_kv_action_models(
            rows, pooled, _include_ablations=False
        )["models"]["D1"]
        uncalibrated = copy.deepcopy(protocol)
        uncalibrated["paired_delta"]["calibration_enabled"] = False
        ablations["D1-C0-no-calibration"] = evaluate_kv_action_models(
            rows, uncalibrated, _include_ablations=False
        )["models"]["D1"]
    return {
        "train_traces": len({row["trace_id"] for row in train}),
        "evaluation_traces": len({row["trace_id"] for row in evaluation}),
        "group_overlap": False,
        "max_action_feature_delta": max_feature_delta,
        "models": summary,
        "decisions": results,
        "paired_delta": {
            "feature_names": list(delta_feature_names),
            "fit_traces": len(ordered_train_traces) - calibration_trace_count,
            "calibration_traces": calibration_trace_count,
            "one_sided_quantile": float(delta_config["one_sided_quantile"]),
            "regime_conditioned": regime_conditioned,
            "calibration_enabled": bool(delta_config["calibration_enabled"]),
            "calibration_offsets_ms": {
                f"{regime}:{action}": value
                for (regime, action), value in sorted(delta_calibration.items())
            },
            "acceptance": delta_acceptance,
        },
        "ablations": ablations,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prometheus_metric(text: str, sample: str) -> float:
    prefix = sample + " "
    line = next((row for row in text.splitlines() if row.startswith(prefix)), None)
    if line is None:
        raise ValueError(f"KV action raw evidence lacks metric {sample}")
    value = float(line[len(prefix):])
    if not math.isfinite(value):
        raise ValueError("KV action raw evidence contains a non-finite metric")
    return value


def validate_kv_action_raw_evidence(
    rows: list[dict[str, Any]], evidence_rows: list[dict[str, Any]]
) -> None:
    by_observation = {str(row["observation_id"]): row for row in evidence_rows}
    if len(by_observation) != len(evidence_rows) or set(by_observation) != {
        str(row["observation_id"]) for row in rows
    }:
        raise ValueError("KV action raw observation identity coverage differs")
    if [int(row.get("observation_order", -1)) for row in evidence_rows] != list(
        range(1, len(evidence_rows) + 1)
    ):
        raise ValueError("KV action raw observation sequence differs")
    for row in rows:
        evidence = by_observation[str(row["observation_id"])]
        if evidence.get("action") != row["action"] or evidence.get("trace_id") != row["trace_id"]:
            raise ValueError("KV action raw observation identity differs")
        if int(evidence.get("observation_order", -1)) != int(row["observation_order"]):
            raise ValueError("KV action row order differs from raw observation sequence")
        before = evidence.get("before_metrics")
        after = evidence.get("after_metrics")
        response = evidence.get("response")
        if not isinstance(before, str) or not isinstance(after, str) or not isinstance(response, dict):
            raise ValueError("KV action raw observation payload differs")
        action = str(row["action"])
        selected = f'llamacpp:kv_action_decisions_total{{action="{action}"}}'
        observed = f'llamacpp:kv_action_observations_total{{action="{action}"}}'
        cost = f'llamacpp:kv_action_observation_seconds_total{{action="{action}"}}'
        decision = "llamacpp:kv_action_decision_seconds_total"
        checks = {
            "selected_delta": _prometheus_metric(after, selected) - _prometheus_metric(before, selected),
            "observation_delta": _prometheus_metric(after, observed) - _prometheus_metric(before, observed),
            "observed_cost_ms": 1000.0 * (
                _prometheus_metric(after, cost) - _prometheus_metric(before, cost)
            ),
            "decision_cpu_ms": 1000.0 * (
                _prometheus_metric(after, decision) - _prometheus_metric(before, decision)
            ),
            "paged_decisions": _prometheus_metric(
                after, 'llamacpp:kv_action_decisions_total{action="paged"}'
            ),
            "invalid_features": _prometheus_metric(
                after, "llamacpp:kv_action_invalid_features_total"
            ),
        }
        for field, expected in checks.items():
            if not math.isclose(float(row[field]), expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"KV action raw observation {field} differs")
        raw_features = [
            _prometheus_metric(
                after, f'llamacpp:kv_action_last_model_feature{{index="{index}"}}'
            )
            for index in range(9)
        ]
        if raw_features != [float(value) for value in row["action_runtime_model_features"]]:
            raise ValueError("KV action raw runtime feature vector differs")
        if int(response.get("timings", {}).get("prompt_n", -1)) != int(row["prompt_tokens"]):
            raise ValueError("KV action raw response prompt token count differs")


def validate_kv_action_artifact(artifact: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = load_kv_action_protocol(protocol_path)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((artifact / "report.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_version") != protocol.get("artifact_version"):
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
        "direct": {"port": 19840, "kv_action_policy": "fixed", "cache_idle_slots": False,
                   "endpoint_slots": True},
        "device_swap": {"port": 19841, "kv_action_policy": "fixed", "cache_idle_slots": True,
                        "endpoint_slots": True},
        "host_swap": {
            "port": 19842, "kv_action_policy": "fixed", "cache_idle_slots": True,
            "kv_swap_path": "memory", "kv_swap_budget_mib": 256, "endpoint_slots": True,
        },
        "recompute": {"port": 19843, "kv_action_policy": "fixed", "cache_idle_slots": False,
                      "endpoint_slots": True, "cache_prompt": False,
                      "observations_per_trace": 2},
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
    replayed_sync_audit: dict[str, Any] | None = None
    if vendor_root.is_dir():
        try:
            subprocess.run(
                ["git", "cat-file", "-e", f"{upstream_revision}^{{commit}}"],
                cwd=vendor_root, check=True, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Historical evidence must remain replayable after later vendor
            # commits. Validate the committed patch against its recorded base
            # in an isolated index instead of equating it with the live HEAD.
            with tempfile.TemporaryDirectory() as temporary:
                environment = os.environ.copy()
                environment["GIT_INDEX_FILE"] = str(Path(temporary) / "replay.index")
                subprocess.run(
                    ["git", "read-tree", upstream_revision], cwd=vendor_root,
                    env=environment, check=True, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                subprocess.run(
                    ["git", "apply", "--cached", "--check", "--whitespace=nowarn", "-"],
                    cwd=vendor_root, env=environment, input=committed_patch, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                subprocess.run(
                    ["git", "apply", "--cached", "--whitespace=nowarn", "-"],
                    cwd=vendor_root, env=environment, input=committed_patch, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                replayed_sources = {
                    Path(relative).name: subprocess.check_output(
                        ["git", "show", f":{relative}"], cwd=vendor_root, env=environment,
                    )
                    for relative in (
                        "tools/server/server-kv-action-policy.cpp",
                        "tools/server/server-kv-action-policy.h",
                    )
                }
                replayed_sync_audit = _audit_kv_action_policy_sources(replayed_sources)
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
    evidence_rows = [
        json.loads(line) for line in (artifact / "runtime-evidence.jsonl").read_text(
            encoding="utf-8"
        ).splitlines() if line
    ]
    validate_kv_action_raw_evidence(rows, evidence_rows)
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
    sync_audit = json.loads((artifact / "sync-audit.json").read_text(encoding="utf-8"))
    if replayed_sync_audit is None:
        raise ValueError("KV action vendor replay state is unavailable")
    expected_sync_audit = replayed_sync_audit
    if sync_audit != expected_sync_audit or not expected_sync_audit["passed"]:
        raise ValueError("KV action CUDA synchronization audit differs or did not pass")
    gates = protocol["overhead_gates"]
    ratios = sorted(float(row["decision_to_action_ratio"]) for row in rows)
    expected_overhead = {
        "p99_choose_microseconds": max(float(row["p99_ns"]) for row in overhead_rows) / 1000.0,
        "measured_max_choose_microseconds": max(float(row["max_ns"]) for row in overhead_rows) / 1000.0,
        "hot_loop_allocations": sum(int(row["allocations"]) for row in overhead_rows),
        "direct_cuda_sync_symbols": len(expected_sync_audit["matches"]),
        "scheduler_cpu_ratio_p99": ratios[int(0.99 * (len(ratios) - 1))],
    }
    expected_overhead["passed"] = (
        expected_overhead["p99_choose_microseconds"] <= gates["p99_choose_microseconds_max"]
        and expected_overhead["hot_loop_allocations"] == gates["hot_loop_allocations"]
        and expected_overhead["direct_cuda_sync_symbols"] == gates["direct_cuda_sync_symbols"]
        and expected_overhead["scheduler_cpu_ratio_p99"] <= gates["scheduler_cpu_ratio_p99_max"]
    )
    if report.get("overhead") != expected_overhead or not expected_overhead["passed"]:
        raise ValueError("KV action overhead gates differ or did not pass")
    return report
