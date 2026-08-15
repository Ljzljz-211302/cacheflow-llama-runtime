from __future__ import annotations

import math
import random
import statistics
from typing import Any


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def cluster_interval(effects: dict[int, list[float]], protocol: dict[str, Any]) -> list[float]:
    generator = random.Random(int(protocol["random_seed"]))
    blocks = sorted(effects)
    estimates = []
    for _ in range(int(protocol["statistics"]["bootstrap_resamples"])):
        sampled = generator.choices(blocks, k=len(blocks))
        estimates.append(statistics.median(value for block in sampled for value in effects[block]))
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def analyze_public_external(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    blocks = range(1, int(protocol["matched_process_blocks"]) + 1)
    traces = tuple(protocol["trace_sources"])
    expected = {(block, trace, action) for block in blocks for trace in traces
                for action in ("direct", "paged")}
    keyed = {}
    for row in rows:
        key = (int(row["block"]), str(row["trace_source"]), str(row["action"]))
        if key not in expected or key in keyed:
            raise ValueError("public replay contains an unexpected or duplicate cell")
        if float(row["elapsed_ms"]) <= 0 or not row["requests"]:
            raise ValueError("public replay timing evidence is incomplete")
        route = row["route"]
        if key[2] == "direct":
            if any(float(route[field]) != 0 for field in (
                    "paged_contiguous_fastpath_calls", "paged_contiguous_fastpath_sequences",
                    "paged_calls", "cuda_dispatches", "paged_fallbacks")):
                raise ValueError("Direct arm entered a Paged route")
        else:
            expected_decode_sequences = sum(
                max(0, len(request["output_token_ids"]) - 1) for request in row["requests"]
            )
            calls = float(route["paged_contiguous_fastpath_calls"])
            sequences = float(route["paged_contiguous_fastpath_sequences"])
            if (expected_decode_sequences <= 0 or calls <= 0 or calls > expected_decode_sequences or
                    sequences != expected_decode_sequences or
              any(float(route[field]) != 0 for field in (
                      "paged_calls", "cuda_dispatches", "paged_fallbacks"))):
                raise ValueError("Paged route evidence is incomplete")
        keyed[key] = row
    if set(keyed) != expected:
        raise ValueError("public replay does not cover the complete paired matrix")

    effects = {block: [] for block in blocks}
    direct_latencies, paged_latencies = [], []
    token_matches = token_comparisons = 0
    quality_matches = quality_comparisons = 0
    maximum_quality_delta = 0.0
    per_trace = {}
    for block in blocks:
        for trace in traces:
            direct, paged = keyed[(block, trace, "direct")], keyed[(block, trace, "paged")]
            direct_by_id = {int(row["trace_row"]): row for row in direct["requests"]}
            paged_by_id = {int(row["trace_row"]): row for row in paged["requests"]}
            if direct_by_id.keys() != paged_by_id.keys() or len(direct_by_id) != len(direct["requests"]):
                raise ValueError("public trace request pairing is incomplete")
            gain = (float(direct["elapsed_ms"]) / float(paged["elapsed_ms"]) - 1.0) * 100.0
            effects[block].append(gain)
            direct_latencies.extend(float(row["latency_ms"]) for row in direct_by_id.values())
            paged_latencies.extend(float(row["latency_ms"]) for row in paged_by_id.values())
            for trace_row in direct_by_id:
                left, right = direct_by_id[trace_row], paged_by_id[trace_row]
                token_comparisons += 1
                token_matches += left["output_token_ids"] == right["output_token_ids"]
                if int(left["cache_tokens"]) != int(right["cache_tokens"]):
                    raise ValueError("Direct/Paged cache-token evidence differs")
            if block == 1 and trace == traces[0]:
                direct_quality = {(row["dataset"], row["record_id"]): row for row in direct["quality"]}
                paged_quality = {(row["dataset"], row["record_id"]): row for row in paged["quality"]}
                if direct_quality.keys() != paged_quality.keys():
                    raise ValueError("LongBench quality pairing is incomplete")
                for key in direct_quality:
                    left, right = direct_quality[key], paged_quality[key]
                    delta = abs(float(left["score"]) - float(right["score"]))
                    maximum_quality_delta = max(maximum_quality_delta, delta)
                    quality_comparisons += 1
                    quality_matches += left["output_token_ids"] == right["output_token_ids"]
            per_trace[f"block-{block}-{trace}"] = {"throughput_gain_percent": gain}

    effect_values = [value for block_values in effects.values() for value in block_values]
    interval = cluster_interval(effects, protocol)
    direct_p95, paged_p95 = percentile(direct_latencies, 0.95), percentile(paged_latencies, 0.95)
    p95_regression = (paged_p95 / direct_p95 - 1.0) * 100.0
    gates = protocol["acceptance"]
    correctness_passed = (not gates["require_exact_output_tokens"] or token_matches == token_comparisons)
    quality_passed = (quality_comparisons > 0 and quality_matches == quality_comparisons and
                      maximum_quality_delta <= float(gates["maximum_quality_score_delta"]))
    return {
        "schema_version": 1, "protocol_version": protocol["protocol_version"],
        "trace_sources": sorted(traces), "observations": len(rows),
        "throughput_gain_percent": {
            "median": statistics.median(effect_values), "minimum": min(effect_values),
            "maximum": max(effect_values),
        },
        "block_cluster_bootstrap_95_percent": interval,
        "direct_request_latency_p95_ms": direct_p95,
        "paged_request_latency_p95_ms": paged_p95,
        "p95_latency_regression_percent": p95_regression,
        "correctness": {"token_matches": token_matches, "token_comparisons": token_comparisons,
                        "passed": correctness_passed},
        "quality": {"comparisons": quality_comparisons, "token_matches": quality_matches,
                    "maximum_score_delta": maximum_quality_delta, "passed": quality_passed},
        "per_trace": per_trace,
        "promotion_passed": (correctness_passed and quality_passed and
            interval[0] >= float(gates["minimum_throughput_gain_lower_95_percent"]) and
            p95_regression <= float(gates["maximum_p95_latency_regression_percent"])),
    }
