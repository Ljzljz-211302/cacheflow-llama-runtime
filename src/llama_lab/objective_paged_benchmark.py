from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "minimum": min(values),
        "maximum": max(values),
    }


def load_definition(root: Path, protocol_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if "prompt" in protocol.get("request", {}):
        raise ValueError("objective protocol must not embed a benchmark prompt")
    corpus_path = root / protocol["workload_file"]
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    workloads = corpus.get("workloads", [])
    ids = [row.get("id") for row in workloads]
    if len(workloads) < int(protocol["acceptance"]["minimum_workload_coverage"]):
        raise ValueError("objective corpus has insufficient workload coverage")
    if len(ids) != len(set(ids)) or any(not row.get("prompt") for row in workloads):
        raise ValueError("objective corpus IDs/prompts must be non-empty and unique")
    if len({row.get("category") for row in workloads}) < 4:
        raise ValueError("objective corpus must cover at least four prompt categories")
    return protocol, corpus


def arm_plan(protocol: dict[str, Any]) -> list[tuple[int, int, str]]:
    generator = random.Random(int(protocol["random_seed"]))
    result = []
    for pair in range(1, int(protocol["paired_trials"]) + 1):
        actions = ["direct", "paged"]
        generator.shuffle(actions)
        result.extend((pair, order, action) for order, action in enumerate(actions, 1))
    return result


def workload_order(protocol: dict[str, Any], corpus: dict[str, Any], pair: int, action: str) -> list[str]:
    ids = [row["id"] for row in corpus["workloads"]]
    action_offset = 0 if action == "direct" else 1_000_003
    random.Random(int(protocol["random_seed"]) + pair * 31 + action_offset).shuffle(ids)
    return ids


def _cluster_bootstrap(effects: dict[int, list[float]], seed: int) -> list[float]:
    generator = random.Random(seed)
    pair_ids = sorted(effects)
    estimates = []
    for _ in range(10_000):
        sampled = generator.choices(pair_ids, k=len(pair_ids))
        estimates.append(statistics.median(
            value for pair in sampled for value in effects[pair]
        ))
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def analyze(
    protocol: dict[str, Any], corpus: dict[str, Any], rows: list[dict[str, Any]],
) -> dict[str, Any]:
    workloads = {row["id"]: row for row in corpus["workloads"]}
    pairs = int(protocol["paired_trials"])
    measured = int(protocol["request"]["measured_requests_per_workload_arm"])
    expected_keys = {
        (pair, action, workload_id)
        for pair in range(1, pairs + 1)
        for action in ("direct", "paged")
        for workload_id in workloads
    }
    keyed: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["pair"]), row["action"], row["workload_id"])
        if key in keyed or key not in expected_keys:
            raise ValueError("objective artifact contains duplicate or unexpected row")
        if row["category"] != workloads[row["workload_id"]]["category"]:
            raise ValueError("objective row category differs from frozen corpus")
        prompt_hash = hashlib.sha256(workloads[row["workload_id"]]["prompt"].encode()).hexdigest()
        if row["prompt_sha256"] != prompt_hash:
            raise ValueError("objective row prompt differs from frozen corpus")
        samples = row["client_elapsed_ms"]
        if len(samples) != measured or any(float(value) <= 0 for value in samples):
            raise ValueError("objective row has incomplete timing samples")
        contexts = {int(value) for value in row["actual_context_tokens"]}
        if len(contexts) != 1:
            raise ValueError("objective context changed within an arm")
        context = next(iter(contexts))
        capability = protocol["capability"]
        if not int(capability["minimum_actual_context_tokens"]) <= context <= int(capability["maximum_actual_context_tokens"]):
            raise ValueError("objective workload is outside the registered capability")
        if row["action"] == "paged" and (row["paged_calls"] < measured or row["paged_fallbacks"] != 0):
            raise ValueError("objective Paged row did not execute without fallback")
        keyed[key] = row
    if set(keyed) != expected_keys:
        raise ValueError("objective artifact does not cover the full pair/action/workload matrix")

    effects_by_pair: dict[int, list[float]] = {pair: [] for pair in range(1, pairs + 1)}
    per_workload: dict[str, dict[str, Any]] = {}
    for workload_id, definition in workloads.items():
        direct_values, paged_values, regressions = [], [], []
        context = None
        for pair in range(1, pairs + 1):
            direct = keyed[(pair, "direct", workload_id)]
            paged = keyed[(pair, "paged", workload_id)]
            if direct["contents"] != paged["contents"]:
                raise ValueError("objective Direct/Paged output mismatch")
            direct_context = int(direct["actual_context_tokens"][0])
            paged_context = int(paged["actual_context_tokens"][0])
            if direct_context != paged_context:
                raise ValueError("objective Direct/Paged context mismatch")
            context = direct_context
            direct_median = statistics.median(map(float, direct["client_elapsed_ms"]))
            paged_median = statistics.median(map(float, paged["client_elapsed_ms"]))
            regression = (paged_median / direct_median - 1.0) * 100.0
            direct_values.append(direct_median)
            paged_values.append(paged_median)
            regressions.append(regression)
            effects_by_pair[pair].append(regression)
        per_workload[workload_id] = {
            "category": definition["category"],
            "actual_context_tokens": context,
            "actual_page_count": math.ceil(context / int(protocol["capability"]["page_size_tokens"])),
            "paired_trials": pairs,
            "direct_arm_median_ms": summarize(direct_values),
            "paged_arm_median_ms": summarize(paged_values),
            "paired_regression_percent": summarize(regressions),
        }
    all_effects = [value for values in effects_by_pair.values() for value in values]
    interval = _cluster_bootstrap(effects_by_pair, int(protocol["random_seed"]))
    limit = float(protocol["acceptance"]["maximum_primary_regression_upper_95_percent"])
    primary = summarize(all_effects)
    required_pages = set(map(int, protocol["capability"].get("required_actual_page_counts", [])))
    actual_pages = {int(row["actual_page_count"]) for row in per_workload.values()}
    page_coverage_passed = not required_pages or actual_pages == required_pages
    has_extended_gates = "maximum_primary_p95_regression_percent" in protocol["acceptance"]
    tail_limit = float(protocol["acceptance"].get("maximum_primary_p95_regression_percent", float("inf")))
    worst_limit = float(protocol["acceptance"].get("maximum_any_workload_median_regression_percent", float("inf")))
    worst_workload_median = max(
        float(row["paired_regression_percent"]["median"]) for row in per_workload.values()
    )
    result = {
        "schema_version": 1,
        "protocol_version": protocol["protocol_version"],
        "paired_trials": pairs,
        "workload_count": len(workloads),
        "observations": len(rows),
        "primary_paired_regression_percent": primary,
        "primary_pair_cluster_bootstrap_95_percent": interval,
        "promotion_limit_percent": limit,
        "promotion_passed": (
            interval[1] <= limit and primary["p95"] <= tail_limit
            and worst_workload_median <= worst_limit and page_coverage_passed
        ),
        "per_workload": per_workload,
    }
    if has_extended_gates:
        result.update({
            "primary_p95_limit_percent": tail_limit,
            "worst_workload_median_regression_percent": worst_workload_median,
            "worst_workload_limit_percent": worst_limit,
            "page_coverage_passed": page_coverage_passed,
        })
    return result
