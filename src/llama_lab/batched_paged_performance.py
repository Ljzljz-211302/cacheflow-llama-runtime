from __future__ import annotations

import math
import random
import statistics
from typing import Any


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "minimum": min(values),
        "maximum": max(values),
    }


def experiment_plan(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the frozen matched-process execution order and complete matrix."""
    blocks = int(protocol["matched_process_blocks"])
    generator = random.Random(int(protocol["random_seed"]))
    first = ["direct"] * (blocks // 2) + ["paged"] * (blocks // 2)
    if blocks % 2:
        first.append(generator.choice(("direct", "paged")))
    generator.shuffle(first)
    plan = []
    for block, first_action in enumerate(first, 1):
        actions = (first_action, "paged" if first_action == "direct" else "direct")
        cells = [
            (int(batch), int(context))
            for batch in protocol["matrix"]["batch_sizes"]
            for context in protocol["matrix"]["context_tokens"]
        ]
        generator.shuffle(cells)
        for order, action in enumerate(actions, 1):
            for batch, context in cells:
                plan.append({
                    "block": block, "order_in_block": order, "action": action,
                    "batch_size": batch, "context_tokens": context,
                })
    return plan


def _cluster_interval(effects: dict[int, list[float]], protocol: dict[str, Any]) -> list[float]:
    generator = random.Random(int(protocol["random_seed"]))
    blocks = sorted(effects)
    estimates = []
    for _ in range(int(protocol["statistics"]["bootstrap_resamples"])):
        sampled = generator.choices(blocks, k=len(blocks))
        estimates.append(statistics.median(value for block in sampled for value in effects[block]))
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def analyze(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    execution_mode = protocol.get("service", {}).get("paged_execution_mode", "custom_cuda")
    if execution_mode not in ("custom_cuda", "contiguous_fastpath"):
        raise ValueError("unknown Paged execution mode")
    expected = {
        (block, action, int(batch), int(context))
        for block in range(1, int(protocol["matched_process_blocks"]) + 1)
        for action in ("direct", "paged")
        for batch in protocol["matrix"]["batch_sizes"]
        for context in protocol["matrix"]["context_tokens"]
    }
    keyed: dict[tuple[int, str, int, int], dict[str, Any]] = {}
    waves = int(protocol["measurement"]["waves_per_cell"])
    for row in rows:
        key = (int(row["block"]), str(row["action"]), int(row["batch_size"]), int(row["context_tokens"]))
        if key not in expected or key in keyed:
            raise ValueError("batched performance artifact contains an unexpected or duplicate cell")
        batch = key[2]
        if len(row["wave_elapsed_ms"]) != waves:
            raise ValueError("batched performance cell has incomplete measurements")
        if len(row["output_token_ids"]) != waves * batch or len(row["cache_tokens"]) != waves * batch:
            raise ValueError("batched performance cell has incomplete response evidence")
        if len(row["top_logprobs"]) != waves * batch:
            raise ValueError("batched performance cell has incomplete probability evidence")
        if any(float(value) <= 0 for value in row["wave_elapsed_ms"]):
            raise ValueError("batched performance timings must be positive")
        if any(int(value) != key[3] for value in row["cache_tokens"]):
            raise ValueError("runtime tokenizer context differs from the frozen cell")
        if float(row["action_decisions"]) != waves * batch or float(row["paged_fallbacks"]) != 0:
            raise ValueError("action counters differ from the requested workload")
        if key[1] == "paged":
            if execution_mode == "contiguous_fastpath":
                if (float(row.get("paged_contiguous_fastpath_calls", 0)) != waves or
                        float(row.get("paged_contiguous_fastpath_sequences", 0)) != waves * batch or
                        any(float(row[field]) != 0 for field in (
                            "paged_calls", "paged_sequences", "cuda_dispatches", "cuda_sequences"
                        ))):
                    raise ValueError("Paged cell lacks full contiguous fast path evidence")
            else:
                dispatches = 24 * waves
                calls = float(row["paged_calls"])
                exact_batches = set(map(int, protocol["measurement"].get("require_exact_graph_batch_sizes", [])))
                if (float(row["paged_sequences"]) != waves * batch or
                        float(row["cuda_sequences"]) != dispatches * batch or
                        float(row["cuda_dispatches"]) != calls * 24 or
                        calls < waves or calls > waves * batch or
                        (batch in exact_batches and calls != waves)):
                    raise ValueError("Paged cell lacks full CUDA dispatch evidence")
        elif (any(float(row[field]) != 0 for field in (
                "paged_calls", "paged_sequences", "cuda_dispatches", "cuda_sequences")) or
                any(float(row.get(field, 0)) != 0 for field in (
                    "paged_contiguous_fastpath_calls", "paged_contiguous_fastpath_sequences"))):
            raise ValueError("Direct cell entered a Paged execution route")
        keyed[key] = row
    if set(keyed) != expected:
        raise ValueError("batched performance artifact does not cover the frozen matrix")

    effects_by_batch: dict[int, dict[int, list[float]]] = {
        int(batch): {block: [] for block in range(1, int(protocol["matched_process_blocks"]) + 1)}
        for batch in protocol["matrix"]["batch_sizes"]
    }
    latency_regressions: list[float] = []
    output_matches = 0
    output_comparisons = 0
    global_minimum_overlap = 64
    global_maximum_error = 0.0
    probability_comparisons = 0
    incomplete_probability_rows = 0
    per_cell = {}
    for block in range(1, int(protocol["matched_process_blocks"]) + 1):
        for batch in map(int, protocol["matrix"]["batch_sizes"]):
            for context in map(int, protocol["matrix"]["context_tokens"]):
                direct = keyed[(block, "direct", batch, context)]
                paged = keyed[(block, "paged", batch, context)]
                cell_matches = sum(
                    direct_token == paged_token for direct_token, paged_token in
                    zip(direct["output_token_ids"], paged["output_token_ids"])
                )
                output_matches += cell_matches
                output_comparisons += len(direct["output_token_ids"])
                minimum_overlap = 64
                maximum_error = 0.0
                for direct_probs, paged_probs in zip(direct["top_logprobs"], paged["top_logprobs"]):
                    direct_by_id = {int(item["id"]): float(item["logprob"]) for item in direct_probs}
                    paged_by_id = {int(item["id"]): float(item["logprob"]) for item in paged_probs}
                    common = direct_by_id.keys() & paged_by_id.keys()
                    if not direct_by_id or not paged_by_id:
                        incomplete_probability_rows += 1
                        continue
                    probability_comparisons += 1
                    minimum_overlap = min(minimum_overlap, len(common))
                    if common:
                        maximum_error = max(maximum_error, max(
                            abs(direct_by_id[token] - paged_by_id[token]) for token in common
                        ))
                global_minimum_overlap = min(global_minimum_overlap, minimum_overlap)
                global_maximum_error = max(global_maximum_error, maximum_error)
                direct_throughput = batch * waves * 1000.0 / sum(map(float, direct["wave_elapsed_ms"]))
                paged_throughput = batch * waves * 1000.0 / sum(map(float, paged["wave_elapsed_ms"]))
                gain = (paged_throughput / direct_throughput - 1.0) * 100.0
                latency = (
                    statistics.median(map(float, paged["wave_elapsed_ms"])) /
                    statistics.median(map(float, direct["wave_elapsed_ms"])) - 1.0
                ) * 100.0
                effects_by_batch[batch][block].append(gain)
                latency_regressions.append(latency)
                route_calls = (paged.get("paged_contiguous_fastpath_calls", 0)
                               if execution_mode == "contiguous_fastpath" else paged["paged_calls"])
                route_sequences = (paged.get("paged_contiguous_fastpath_sequences", 0)
                                   if execution_mode == "contiguous_fastpath" else paged["paged_sequences"])
                per_cell[f"block-{block}-batch-{batch}-context-{context}"] = {
                    "throughput_gain_percent": gain,
                    "median_wave_latency_regression_percent": latency,
                    "direct_peak_gpu_memory_mib": direct["peak_gpu_memory_mib"],
                    "paged_peak_gpu_memory_mib": paged["peak_gpu_memory_mib"],
                    "minimum_top64_overlap": minimum_overlap,
                    "maximum_common_logprob_error": maximum_error,
                    "output_token_matches": cell_matches,
                    "output_token_comparisons": len(direct["output_token_ids"]),
                    "paged_graph_calls": paged["paged_calls"],
                    "execution_route": ("upstream-contiguous-fastpath"
                                        if execution_mode == "contiguous_fastpath" else "custom-paged-cuda"),
                    "realized_sequences_per_graph": route_sequences / route_calls,
                }

    primary_batch = int(protocol["acceptance"]["primary_batch_size"])
    primary_effects = [value for values in effects_by_batch[primary_batch].values() for value in values]
    primary_interval = _cluster_interval(effects_by_batch[primary_batch], protocol)
    primary_direct_waves = [
        float(value)
        for block in range(1, int(protocol["matched_process_blocks"]) + 1)
        for context in map(int, protocol["matrix"]["context_tokens"])
        for value in keyed[(block, "direct", primary_batch, context)]["wave_elapsed_ms"]
    ]
    primary_paged_waves = [
        float(value)
        for block in range(1, int(protocol["matched_process_blocks"]) + 1)
        for context in map(int, protocol["matrix"]["context_tokens"])
        for value in keyed[(block, "paged", primary_batch, context)]["wave_elapsed_ms"]
    ]
    by_batch = {}
    for batch, block_effects in effects_by_batch.items():
        values = [value for effects in block_effects.values() for value in effects]
        by_batch[str(batch)] = {
            "throughput_gain_percent": _summary(values),
            "block_cluster_bootstrap_95_percent": _cluster_interval(block_effects, protocol),
        }
    gates = protocol["acceptance"]
    correctness_passed = (
        output_matches == output_comparisons
        and incomplete_probability_rows == 0
        and global_minimum_overlap >= int(gates["minimum_top64_overlap"])
        and global_maximum_error <= float(gates["maximum_common_logprob_error"])
    )
    worst_latency = max(latency_regressions)
    primary_direct_p95 = _percentile(primary_direct_waves, 0.95)
    primary_paged_p95 = _percentile(primary_paged_waves, 0.95)
    primary_p95_latency = (primary_paged_p95 / primary_direct_p95 - 1.0) * 100.0
    return {
        "schema_version": 1,
        "protocol_version": protocol["protocol_version"],
        "observations": len(rows),
        "primary_batch_size": primary_batch,
        "primary_throughput_gain_percent": _summary(primary_effects),
        "primary_block_cluster_bootstrap_95_percent": primary_interval,
        "primary_direct_wave_latency_p95_ms": primary_direct_p95,
        "primary_paged_wave_latency_p95_ms": primary_paged_p95,
        "primary_p95_wave_latency_regression_percent": primary_p95_latency,
        "worst_cell_median_wave_latency_regression_percent": worst_latency,
        "throughput_by_batch": by_batch,
        "correctness": {
            "output_token_matches": output_matches,
            "output_token_comparisons": output_comparisons,
            "minimum_top64_overlap": global_minimum_overlap,
            "maximum_common_logprob_error": global_maximum_error,
            "probability_rows_compared": probability_comparisons,
            "incomplete_probability_rows": incomplete_probability_rows,
            "passed": correctness_passed,
        },
        "per_cell": per_cell,
        "promotion_passed": (
            correctness_passed
            and primary_interval[0] > float(gates["minimum_throughput_gain_lower_95_percent"])
            and primary_p95_latency <= float(gates["maximum_p95_latency_regression_percent"])
            and worst_latency <= float(gates["maximum_any_cell_median_latency_regression_percent"])
        ),
    }
