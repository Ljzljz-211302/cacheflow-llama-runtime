from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import subprocess
from pathlib import Path
from typing import Any


_NORMALIZED_TRIAL_FIELDS_V1 = frozenset({
    "pair", "order_in_pair", "action", "workload_id", "category",
    "prompt_sha256", "client_elapsed_ms", "actual_context_tokens", "contents",
    "paged_calls", "paged_fallbacks", "action_decisions",
    "action_reason_decisions", "action_observations", "raw",
})
_NORMALIZED_TRIAL_FIELDS_V2 = _NORMALIZED_TRIAL_FIELDS_V1 | {
    "server_decode_ms", "server_prompt_ms",
}


def validate_reconstructed_rows(
    protocol: dict[str, Any],
    persisted_rows: list[dict[str, Any]],
    reconstructed_rows: list[dict[str, Any]],
) -> None:
    """Validate raw-to-normalized evidence using the protocol's frozen row schema."""
    schema_version = int(protocol.get("schema_version", 0))
    if schema_version == 1:
        expected_fields = _NORMALIZED_TRIAL_FIELDS_V1
    elif schema_version == 2:
        expected_fields = _NORMALIZED_TRIAL_FIELDS_V2
        if protocol.get("service", {}).get("paged_kernel_variant") is not None:
            expected_fields = expected_fields | {"paged_kernel_variant"}
    else:
        raise ValueError(f"unsupported objective protocol schema_version: {schema_version}")
    if len(persisted_rows) != len(reconstructed_rows):
        raise AssertionError("objective normalized trial count differs from raw arm evidence")
    for index, (persisted, reconstructed) in enumerate(zip(persisted_rows, reconstructed_rows)):
        if set(persisted) != expected_fields:
            raise AssertionError(
                f"objective normalized trial schema differs at row {index} for schema v{schema_version}"
            )
        projected = {field: reconstructed[field] for field in expected_fields}
        if persisted != projected:
            raise AssertionError(
                f"objective normalized trials differ from raw arm evidence at row {index}"
            )


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


def load_definition(
    root: Path, protocol_path: Path, *, validate_live_sources: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    minimum_categories = int(protocol.get("minimum_prompt_categories", 4))
    if len({row.get("category") for row in workloads}) < minimum_categories:
        raise ValueError("objective corpus has insufficient prompt-category coverage")
    if validate_live_sources:
        for row in workloads:
            if "source_sha256" not in row:
                continue
            source = root / row["source"]
            if not source.is_file() or file_sha256(source) != row["source_sha256"]:
                raise ValueError(f"objective workload source binding differs: {row['id']}")
    return protocol, corpus


def validate_frozen_source_revision(
    root: Path, corpus: dict[str, Any], revision: str,
) -> None:
    """Prove source provenance at measurement revision while allowing docs to evolve."""
    expected_by_source = {
        row["source"]: row["source_sha256"]
        for row in corpus.get("workloads", []) if "source_sha256" in row
    }
    for source, expected_sha256 in expected_by_source.items():
        completed = subprocess.run(
            ["git", "show", f"{revision}:{source}"], cwd=root,
            check=True, capture_output=True,
        )
        if hashlib.sha256(completed.stdout).hexdigest() != expected_sha256:
            raise ValueError(f"objective historical source binding differs: {source}")


def arm_plan(protocol: dict[str, Any]) -> list[tuple[int, int, str]]:
    generator = random.Random(int(protocol["random_seed"]))
    result = []
    block_count = int(protocol.get("matched_process_blocks", protocol.get("paired_trials")))
    balanced = bool(protocol.get("balanced_arm_order", False))
    if balanced:
        first_actions = ["direct"] * (block_count // 2) + ["paged"] * (block_count // 2)
        if block_count % 2:
            first_actions.append(generator.choice(["direct", "paged"]))
        generator.shuffle(first_actions)
    else:
        first_actions = []
        for _ in range(block_count):
            actions = ["direct", "paged"]
            generator.shuffle(actions)
            first_actions.append(actions[0])
    for pair, first in enumerate(first_actions, 1):
        actions = [first, "paged" if first == "direct" else "direct"]
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
    matched_blocks = "matched_process_blocks" in protocol
    pairs = int(protocol.get("matched_process_blocks", protocol.get("paired_trials")))
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
        expected_kernel = protocol["service"].get("paged_kernel_variant")
        if expected_kernel is not None and row.get("paged_kernel_variant") != expected_kernel:
            raise ValueError("objective row differs from the registered Paged kernel")
        primary_field = protocol.get("analysis", {}).get("primary_timing_field", "client_elapsed_ms")
        samples = row[primary_field]
        if len(samples) != measured or any(float(value) <= 0 for value in samples):
            raise ValueError("objective row has incomplete timing samples")
        contexts = {int(value) for value in row["actual_context_tokens"]}
        if len(contexts) != 1:
            raise ValueError("objective context changed within an arm")
        context = next(iter(contexts))
        expected_context = workloads[row["workload_id"]].get("actual_prompt_tokens")
        if expected_context is not None and context != int(expected_context):
            raise ValueError("objective context differs from frozen tokenizer length")
        capability = protocol["capability"]
        if not int(capability["minimum_actual_context_tokens"]) <= context <= int(capability["maximum_actual_context_tokens"]):
            raise ValueError("objective workload is outside the registered capability")
        if any(float(row[field]) != measured for field in (
            "action_decisions", "action_reason_decisions", "action_observations",
        )):
            raise ValueError("objective action counters do not match measured requests")
        if row["action"] == "paged":
            if float(row["paged_calls"]) != measured or float(row["paged_fallbacks"]) != 0:
                raise ValueError("objective Paged row did not execute exactly once per measured request")
        elif float(row["paged_calls"]) != 0 or float(row["paged_fallbacks"]) != 0:
            raise ValueError("objective Direct row entered the Paged graph")
        keyed[key] = row
    if set(keyed) != expected_keys:
        raise ValueError("objective artifact does not cover the full pair/action/workload matrix")

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
            timing_field = protocol.get("analysis", {}).get("primary_timing_field", "client_elapsed_ms")
            direct_median = statistics.median(map(float, direct[timing_field]))
            paged_median = statistics.median(map(float, paged[timing_field]))
            regression = (paged_median / direct_median - 1.0) * 100.0
            direct_values.append(direct_median)
            paged_values.append(paged_median)
            regressions.append(regression)
        per_workload[workload_id] = {
            "category": definition["category"],
            "actual_context_tokens": context,
            "actual_page_count": math.ceil(context / int(protocol["capability"]["page_size_tokens"])),
            ("matched_process_blocks" if matched_blocks else "paired_trials"): pairs,
            "direct_arm_median_ms": summarize(direct_values),
            "paged_arm_median_ms": summarize(paged_values),
            ("matched_block_regression_percent" if matched_blocks else "paired_regression_percent"): summarize(regressions),
        }
        for secondary_field in protocol.get("analysis", {}).get("secondary_metrics", []):
            direct_secondary = [
                statistics.median(map(float, keyed[(pair, "direct", workload_id)][secondary_field]))
                for pair in range(1, pairs + 1)
            ]
            paged_secondary = [
                statistics.median(map(float, keyed[(pair, "paged", workload_id)][secondary_field]))
                for pair in range(1, pairs + 1)
            ]
            per_workload[workload_id][f"direct_{secondary_field}"] = summarize(direct_secondary)
            per_workload[workload_id][f"paged_{secondary_field}"] = summarize(paged_secondary)
    primary_minimum = int(protocol.get("analysis", {}).get("primary_minimum_context_tokens", 0))
    primary_workloads = {
        workload_id for workload_id, row in per_workload.items()
        if int(row["actual_context_tokens"]) >= primary_minimum
    }
    primary_effects_by_pair = {
        pair: [
            (statistics.median(map(float, keyed[(pair, "paged", workload_id)][
                protocol.get("analysis", {}).get("primary_timing_field", "client_elapsed_ms")])) /
             statistics.median(map(float, keyed[(pair, "direct", workload_id)][
                protocol.get("analysis", {}).get("primary_timing_field", "client_elapsed_ms")])) - 1.0) * 100.0
            for workload_id in sorted(primary_workloads)
        ]
        for pair in range(1, pairs + 1)
    }
    all_effects = [value for values in primary_effects_by_pair.values() for value in values]
    interval = _cluster_bootstrap(primary_effects_by_pair, int(protocol["random_seed"]))
    limit = float(protocol["acceptance"]["maximum_primary_regression_upper_95_percent"])
    primary = summarize(all_effects)
    required_pages = set(map(int, protocol["capability"].get("required_actual_page_counts", [])))
    actual_pages = {int(row["actual_page_count"]) for row in per_workload.values()}
    page_coverage_passed = not required_pages or actual_pages == required_pages
    has_extended_gates = "maximum_primary_p95_regression_percent" in protocol["acceptance"]
    tail_limit = float(protocol["acceptance"].get("maximum_primary_p95_regression_percent", float("inf")))
    worst_limit = float(protocol["acceptance"].get("maximum_any_workload_median_regression_percent", float("inf")))
    effect_key = "matched_block_regression_percent" if matched_blocks else "paired_regression_percent"
    worst_workload_median = max(
        float(per_workload[workload_id][effect_key]["median"])
        for workload_id in primary_workloads
    )
    by_context: dict[str, dict[str, float]] = {}
    for context in sorted({int(row["actual_context_tokens"]) for row in per_workload.values()}):
        effects = [
            float(row[effect_key]["median"])
            for row in per_workload.values() if int(row["actual_context_tokens"]) == context
        ]
        by_context[str(context)] = summarize(effects)
    result = {
        "schema_version": 1,
        "protocol_version": protocol["protocol_version"],
        ("matched_process_blocks" if matched_blocks else "paired_trials"): pairs,
        "workload_count": len(workloads),
        "observations": len(rows),
        ("primary_matched_block_regression_percent" if matched_blocks else "primary_paired_regression_percent"): primary,
        ("primary_block_cluster_bootstrap_95_percent" if matched_blocks else "primary_pair_cluster_bootstrap_95_percent"): interval,
        "promotion_limit_percent": limit,
        "promotion_passed": (
            interval[1] <= limit and primary["p95"] <= tail_limit
            and worst_workload_median <= worst_limit and page_coverage_passed
        ),
        "per_workload": per_workload,
    }
    if "primary_minimum_context_tokens" in protocol.get("analysis", {}):
        result.update({
            "primary_workload_count": len(primary_workloads),
            "primary_minimum_context_tokens": primary_minimum,
            "primary_timing_field": protocol["analysis"]["primary_timing_field"],
            "regression_by_context_tokens": by_context,
        })
    if has_extended_gates:
        result.update({
            "primary_p95_limit_percent": tail_limit,
            "worst_workload_median_regression_percent": worst_workload_median,
            "worst_workload_limit_percent": worst_limit,
            "page_coverage_passed": page_coverage_passed,
        })
    return result
