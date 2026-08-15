from __future__ import annotations

import math
import random
import statistics
from typing import Any

from llama_lab.public_workloads import longbench_qa_f1


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


def p95_cluster_interval(latencies: dict[int, dict[str, list[float]]],
                         protocol: dict[str, Any]) -> list[float]:
    generator = random.Random(int(protocol["random_seed"]) + 1)
    blocks = sorted(latencies)
    estimates = []
    for _ in range(int(protocol["statistics"]["bootstrap_resamples"])):
        sampled = generator.choices(blocks, k=len(blocks))
        direct = [value for block in sampled for value in latencies[block]["direct"]]
        paged = [value for block in sampled for value in latencies[block]["paged"]]
        estimates.append((percentile(paged, 0.95) / percentile(direct, 0.95) - 1.0) * 100.0)
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def validate_recorded_workloads(protocol: dict[str, Any], workloads: dict[str, Any],
                                rows: list[dict[str, Any]]) -> None:
    if not protocol.get("acceptance", {}).get("require_raw_workload_binding", False):
        return
    replay_by_source = {
        source: {int(item["trace_row"]): item for item in replay}
        for source, replay in workloads["performance_replays"].items()
    }
    target_span_ms = float(protocol["replay"]["target_arrival_span_seconds"]) * 1000.0
    quality_by_key = {
        (item["dataset"], item["record_id"]): item for item in workloads["quality_cases"]
    }
    first_trace = protocol["trace_sources"][0]
    for row in rows:
        source = row["trace_source"]
        frozen = replay_by_source[source]
        if {int(item["trace_row"]) for item in row["requests"]} != set(frozen):
            raise ValueError("raw requests differ from the frozen trace rows")
        span = max(float(item["source_arrival_seconds"]) for item in frozen.values())
        for request in row["requests"]:
            expected = frozen[int(request["trace_row"])]
            expected_schedule = (float(expected["source_arrival_seconds"]) / span * target_span_ms
                                 if span > 0 else 0.0)
            if (request["prompt_id"] != expected["prompt_id"] or
                    request["prompt_sha256"] != expected["prompt_sha256"] or
                    int(request["actual_local_input_tokens"]) != int(expected["actual_local_input_tokens"]) or
                    float(request["source_arrival_seconds"]) != float(expected["source_arrival_seconds"]) or
                    not math.isclose(float(request["scheduled_arrival_ms"]), expected_schedule,
                                     rel_tol=0.0, abs_tol=1e-6)):
                raise ValueError("raw request differs from its frozen workload")
        should_have_quality = int(row["block"]) == 1 and source == first_trace
        if bool(row["quality"]) != should_have_quality:
            raise ValueError("quality evidence is attached to the wrong raw cell")
        if should_have_quality and {
                (item["dataset"], item["record_id"]) for item in row["quality"]
        } != set(quality_by_key):
            raise ValueError("quality evidence does not cover the frozen LongBench cases")
        for result in row["quality"]:
            expected = quality_by_key.get((result["dataset"], result["record_id"]))
            if expected is None or result["prompt_sha256"] != expected["prompt_sha256"] or \
                    result["answers"] != expected["answers"]:
                raise ValueError("quality result differs from its frozen LongBench case")
            score = max(longbench_qa_f1(result["prediction"], answer) for answer in expected["answers"])
            if not math.isclose(float(result["score"]), score, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("LongBench score was not reconstructed from prediction and answers")


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
    latencies_by_block = {block: {"direct": [], "paged": []} for block in blocks}
    arrival_slips = []
    token_matches = token_comparisons = 0
    probability_rows_compared = 0
    minimum_top_overlap = int(protocol["request"].get("top_probabilities", 0))
    maximum_common_logprob_error = 0.0
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
            latencies_by_block[block]["direct"].extend(
                float(row["latency_ms"]) for row in direct_by_id.values())
            latencies_by_block[block]["paged"].extend(
                float(row["latency_ms"]) for row in paged_by_id.values())
            arrival_slips.extend(
                max(0.0, float(request["actual_start_ms"]) - float(request["scheduled_arrival_ms"]))
                for request in (*direct_by_id.values(), *paged_by_id.values())
            )
            for trace_row in direct_by_id:
                left, right = direct_by_id[trace_row], paged_by_id[trace_row]
                token_comparisons += 1
                token_matches += left["output_token_ids"] == right["output_token_ids"]
                if int(left["cache_tokens"]) != int(right["cache_tokens"]):
                    raise ValueError("Direct/Paged cache-token evidence differs")
                for left_token, right_token, left_probs, right_probs in zip(
                        left["output_token_ids"], right["output_token_ids"],
                        left["top_logprobs"], right["top_logprobs"]):
                    left_by_id = {int(item["id"]): float(item["logprob"]) for item in left_probs}
                    right_by_id = {int(item["id"]): float(item["logprob"]) for item in right_probs}
                    if not left_by_id or not right_by_id:
                        raise ValueError("public trace probability evidence is incomplete")
                    common = left_by_id.keys() & right_by_id.keys()
                    minimum_top_overlap = min(minimum_top_overlap, len(common))
                    if common:
                        maximum_common_logprob_error = max(maximum_common_logprob_error, max(
                            abs(left_by_id[token] - right_by_id[token]) for token in common
                        ))
                    probability_rows_compared += 1
                    # Once greedy tokens differ, later rows condition on different
                    # histories and are not a like-for-like numerical comparison.
                    if left_token != right_token:
                        break
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
    p95_interval = p95_cluster_interval(latencies_by_block, protocol)
    maximum_arrival_slip = max(arrival_slips)
    gates = protocol["acceptance"]
    correctness_passed = (
        (not gates["require_exact_output_tokens"] or token_matches == token_comparisons)
        and minimum_top_overlap >= int(gates.get("minimum_top_probability_overlap", 0))
        and maximum_common_logprob_error <= float(gates.get("maximum_common_logprob_error", math.inf))
    )
    quality_passed = (quality_comparisons > 0 and quality_matches == quality_comparisons and
                      maximum_quality_delta <= float(gates["maximum_quality_score_delta"]))
    result = {
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
                        "probability_rows_compared": probability_rows_compared,
                        "minimum_top_probability_overlap": minimum_top_overlap,
                        "maximum_common_logprob_error": maximum_common_logprob_error,
                        "passed": correctness_passed},
        "quality": {"comparisons": quality_comparisons, "token_matches": quality_matches,
                    "maximum_score_delta": maximum_quality_delta, "passed": quality_passed},
        "per_trace": per_trace,
        "promotion_passed": (correctness_passed and quality_passed and
            interval[0] >= float(gates["minimum_throughput_gain_lower_95_percent"]) and
            p95_regression <= float(gates["maximum_p95_latency_regression_percent"])),
    }
    if "maximum_arrival_slip_ms" in gates:
        result["p95_latency_regression_block_bootstrap_95_percent"] = p95_interval
        result["maximum_arrival_slip_ms"] = maximum_arrival_slip
        result["promotion_passed"] = (
            result["promotion_passed"] and
            p95_interval[1] <= float(gates[
                "maximum_p95_latency_regression_upper_95_percent"]) and
            maximum_arrival_slip <= float(gates["maximum_arrival_slip_ms"])
        )
    return result
